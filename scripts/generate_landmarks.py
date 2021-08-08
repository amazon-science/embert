import base64
import copy
import json
import math
import os
from argparse import ArgumentParser
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any

import jsonlines
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from grolp.envs.thor_env import ThorEnv
from scripts.generate_maskrcnn import TRAJ_DATA_JSON_FILENAME


@dataclass
class Event:
    metadata: Dict[str, Any]
    instance_detections2D: Dict[str, np.ndarray]
    instance_masks: Dict[str, np.ndarray]


def _refine_event(event) -> Event:
    metadata = copy.deepcopy(event.metadata)
    metadata["objects"] = {o["objectId"]: o for o in metadata["objects"]}

    ref_event = Event(
        instance_detections2D=copy.deepcopy(event.instance_detections2D),
        instance_masks=copy.deepcopy(event.instance_masks),
        metadata=metadata
    )

    return ref_event


def _get_panorama_events(env, is_navigation=False):
    events = [_refine_event(env.last_event)]

    if is_navigation:
        initial_agent = env.last_event.metadata["agent"]

        la_cmd = {'action': 'LookUp', 'forceAction': True}
        env.step(la_cmd)

        for look in range(3):
            la_cmd = {'action': 'RotateRight', 'forceAction': True}
            event = env.step(la_cmd)
            events.append(_refine_event(event))

        teleport_action = {
            'action': 'TeleportFull',
            'rotation': initial_agent["rotation"],
            'x': initial_agent["position"]['x'],
            'z': initial_agent["position"]['z'],
            'y': initial_agent["position"]['y'],
            'horizon': initial_agent["cameraHorizon"],
            'forceAction': True
        }
        last_event = env.step(teleport_action)

        assert last_event.metadata["lastActionSuccess"], "This shouldn't happen!"

    return events


def _get_receptacle(object_id, event):
    object_meta = event.metadata["objects"].get(object_id, None)
    receptacle_id = None

    if object_meta is not None and 'parentReceptacles' in object_meta and object_meta[
        'parentReceptacles'] is not None and \
            len(object_meta['parentReceptacles']) > 0:
        receptacle_id = object_meta['parentReceptacles'][0]

    return receptacle_id


class ALFREDFeatureExtractorDataset(IterableDataset):
    def __init__(self, args):
        # trajectories are scattered among several files
        # we gather their file names here
        self.trajectories = []

        self.args = args
        visited = set()

        with open(args.splits) as in_file:
            splits = json.load(in_file)

        stop = False

        for k, d in splits.items():
            if args.split_id in k:
                for task in d:
                    # load json file
                    json_path = os.path.join(args.data_path, k, task['task'], 'traj_data.json')
                    dest_path = os.path.join(args.data_path, k, task['task'], 'ref_traj_data.json')

                    if json_path not in visited:
                        visited.add(json_path)
                        with open(json_path) as f:
                            ex = json.load(f)

                        # copy trajectory
                        r_idx = task['repeat_idx']  # repeat_idx is the index of the annotation for each trajectory
                        traj = ex.copy()

                        # root & split
                        traj['root'] = os.path.join(args.data_path, task['task'])
                        traj['split'] = k
                        traj['repeat_idx'] = r_idx
                        traj['path'] = json_path

                        self.trajectories.append(traj)

                        if args.debug and len(self.trajectories) == 3:
                            stop = True
                            break

                if stop:
                    print("Debugging mode on!")
                    break

        print(f"Discovered {len(self.trajectories)} files from the directory {args.data_path}")
        if args.debug:
            for traj in self.trajectories:
                print(traj["path"])

        self.start = 0
        self.end = len(self.trajectories)

    def __len__(self):
        return len(self.trajectories)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        # single worker
        if worker_info is None:
            # in this case we just process the entire dataset
            iter_start = self.start
            iter_end = self.end
        else:
            # we need to split the load among the workers making sure that there are no duplicates
            per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)

        # start THOR env
        env = ThorEnv(player_screen_width=self.args.image_width,
                      player_screen_height=self.args.image_height)

        try:
            for idx in range(iter_start, iter_end):
                yield self.get_instance(idx, env)
        except Exception as ex:
            print(f"Error raised for trajectory: {self.trajectories[idx]}")
            raise ex
        finally:
            env.stop()

    def get_instance(self, item, env):
        traj_data = self.trajectories[item]
        json_file = traj_data["path"]

        # make directories
        dest_file = json_file.replace(TRAJ_DATA_JSON_FILENAME, self.args.ref_traj_file)

        # scene setup
        scene_num = traj_data['scene']['scene_num']
        object_poses = traj_data['scene']['object_poses']
        object_toggles = traj_data['scene']['object_toggles']
        dirty_and_empty = traj_data['scene']['dirty_and_empty']

        # reset
        scene_name = 'FloorPlan%d' % scene_num
        env.reset(scene_name)
        env.restore_scene(object_poses, object_toggles, dirty_and_empty)

        env.step(dict(traj_data['scene']['init_action']))

        # setup task
        env.set_task(traj_data, self.args, reward_type='dense')
        low_actions = traj_data['plan']['low_actions']
        num_actions = len(low_actions)
        missing = []
        ref_traj_data = copy.deepcopy(traj_data)
        events = []

        # we first replay the entire trajectory
        for ll_idx, ll_action in enumerate(low_actions):
            # next cmd under the current hl_action
            cmd = ll_action['api_action']
            hl_action = traj_data["plan"]['high_pddl'][low_actions[ll_idx]["high_idx"]]
            is_navigation = hl_action["discrete_action"]["action"] == "GotoLocation"
            # we extract panorama info only for navigation instructions
            events.append(_get_panorama_events(env, is_navigation))

            cmd = {k: cmd[k] for k in
                   ['action', 'objectId', 'receptacleObjectId', 'placeStationary', 'forceAction'] if
                   k in cmd}
            env.step(cmd)

        # events.append(self._refine_event(event))

        subgoal_start = 0

        # at this point we can trace the elements of the trajectory
        for ll_idx, ll_action in enumerate(low_actions):
            hl_action = traj_data["plan"]['high_pddl'][low_actions[ll_idx]["high_idx"]]

            # we are at the end of a subgoal
            # we need to go back and refine all the previous steps
            if ll_idx == num_actions - 1 or low_actions[ll_idx]["high_idx"] != low_actions[ll_idx + 1]["high_idx"]:
                object_id = None
                receptacle_id = None

                # we go through each event object for the current timestep
                for main_event in events[ll_idx]:
                    # if we are dealing with a GotoLocation subgoal we want to make sure to extract the bounding box of the
                    # landmark object
                    if hl_action["discrete_action"]["action"] == "GotoLocation":
                        # First we check that the next instruction doesn't involve an open. If it's the case we execute it
                        # so that we can actually see the object
                        if low_actions[ll_idx + 1]["api_action"]["action"] == "OpenObject":
                            # the landmark is the object that we want to open and not the one that is inside (it's not visible!)
                            open_action = low_actions[ll_idx + 1]
                            object_id = open_action["api_action"]["objectId"]
                        else:
                            landmark_name = hl_action["discrete_action"]["args"][0]

                            for obj_id in main_event.instance_detections2D.keys():
                                if landmark_name in obj_id.lower():
                                    object_id = obj_id
                                    break

                        if object_id is not None:
                            receptacle_id = _get_receptacle(object_id, main_event)

                    elif hl_action["discrete_action"]["action"] == "HeatObject" or hl_action["discrete_action"][
                        "action"] == "CoolObject":
                        # HeatObject high-level instructions are somehow wonky. We need to hack them.
                        # So we assume that an HeatObject high-level instruction will have a PutObject low-level instruction
                        # In this way we can derive the objectId and receptacleId from it. So we go back until we find it
                        for i in range(ll_idx, 0, -1):
                            if low_actions[i]["api_action"]["action"] == "PutObject":
                                object_id = low_actions[i]["api_action"]["objectId"]
                                if "receptacleObjectId" in low_actions[i]["api_action"]:
                                    receptacle_id = low_actions[i]["api_action"]["receptacleObjectId"]
                                else:
                                    receptacle_id = _get_receptacle(object_id, main_event)
                                break

                    elif hl_action["discrete_action"]["action"] in {"PutObject"}:
                        # for these actions the API requires only the receptacle
                        # so we assume it is the object of this action -- counter-intuitive though...
                        if "receptacleObjectId" in hl_action["planner_action"]:
                            object_id = hl_action["planner_action"]["receptacleObjectId"]
                        else:
                            act_object_id = ll_action["api_action"]["objectId"]
                            object_id = _get_receptacle(act_object_id, main_event)

                        receptacle_id = _get_receptacle(object_id, main_event)

                    else:
                        object_id = hl_action["planner_action"]["objectId"]
                        receptacle_id = _get_receptacle(object_id, main_event)

                    if object_id is not None:
                        break

                if object_id is None and receptacle_id is None:
                    missing.append({"action": ll_action, "traj": json_file})

                ref_low_actions = ref_traj_data["plan"]["low_actions"]
                for subgoal_idx in range(subgoal_start, ll_idx + 1):
                    object_box = None
                    object_mask = None
                    receptacle_box = None
                    receptacle_mask = None

                    for event in events[subgoal_idx]:
                        object_box = event.instance_detections2D.get(object_id, object_box)
                        object_mask = event.instance_masks.get(object_id, object_mask)
                        receptacle_box = event.instance_detections2D.get(receptacle_id, receptacle_box)
                        receptacle_mask = event.instance_masks.get(receptacle_id, receptacle_mask)

                    ref_low_actions[subgoal_idx]["subgoal"] = dict()

                    if ref_low_actions[subgoal_idx]["api_action"]["action"] == "PutObject":
                        if receptacle_id is not None and receptacle_box is not None:
                            # this is for actions like open and close that work on the receptacle object
                            ref_low_actions[subgoal_idx]["subgoal"]["object"] = dict(
                                bbox=receptacle_box.tolist(),
                                mask=base64.b64encode(receptacle_mask).decode("utf-8"),
                                mask_shape=list(receptacle_mask.shape),
                                id=receptacle_id
                            )
                    else:
                        if ref_low_actions[subgoal_idx]["api_action"].get("objectId") is not None and \
                                ref_low_actions[subgoal_idx]["api_action"]["objectId"] == object_id:

                            if object_box is not None:
                                # in this case we have a manipulation action that requires the object as reference
                                # if we have a receptacle we can specify it here as well
                                ref_low_actions[subgoal_idx]["subgoal"]["object"] = dict(
                                    bbox=object_box.tolist(),
                                    mask=base64.b64encode(object_mask).decode("utf-8"),
                                    mask_shape=list(object_mask.shape),
                                    id=object_id
                                )

                            if receptacle_id is not None:
                                if ref_low_actions[subgoal_idx]["api_action"].get("receptacleObjectId") is not None:
                                    assert ref_low_actions[subgoal_idx]["api_action"][
                                               "receptacleObjectId"] == receptacle_id

                                if receptacle_box is not None:
                                    ref_low_actions[subgoal_idx]["subgoal"]["receptacle"] = dict(
                                        bbox=receptacle_box.tolist(),
                                        mask=base64.b64encode(receptacle_mask).decode("utf-8"),
                                        mask_shape=list(receptacle_mask.shape),
                                        id=receptacle_id
                                    )
                        elif receptacle_id is not None and ref_low_actions[subgoal_idx]["api_action"].get(
                                "objectId") is not None \
                                and ref_low_actions[subgoal_idx]["api_action"]["objectId"] == receptacle_id:

                            if receptacle_box is not None:
                                # this is for actions like open and close that work on the receptacle object
                                ref_low_actions[subgoal_idx]["subgoal"]["object"] = dict(
                                    bbox=receptacle_box.tolist(),
                                    mask=base64.b64encode(receptacle_mask).decode("utf-8"),
                                    mask_shape=list(receptacle_mask.shape),
                                    id=receptacle_id
                                )
                            # usually they do not have a receptacle so we ignore it here
                        else:
                            # in this last case we have a navigation subgoal for which we just set objectId and receptacleId
                            # as they are!
                            if object_box is not None:
                                ref_low_actions[subgoal_idx]["subgoal"]["object"] = dict(
                                    bbox=object_box.tolist(),
                                    mask=base64.b64encode(object_mask).decode("utf-8"),
                                    mask_shape=list(object_mask.shape),
                                    id=object_id
                                )

                            if receptacle_box is not None:
                                ref_low_actions[subgoal_idx]["subgoal"]["receptacle"] = dict(
                                    bbox=receptacle_box.tolist(),
                                    mask=base64.b64encode(receptacle_mask).decode("utf-8"),
                                    mask_shape=list(receptacle_mask.shape),
                                    id=receptacle_id
                                )
                subgoal_start = ll_idx + 1

        return {"file": dest_file, "traj": ref_traj_data, "missing": missing}


def identity(x):
    return x


def main(args):
    dataset = ALFREDFeatureExtractorDataset(args)
    missing_landmarks = []
    data_loader = DataLoader(dataset, batch_size=1, num_workers=args.num_workers, collate_fn=identity)
    start_time = datetime.now()
    for batch in tqdm(data_loader, desc=f"Extracting location features for split: {args.split_id}"):
        for datum in batch:
            missing_landmarks.extend(datum["missing"])
            with open(datum["file"], mode="w") as out_file:
                json.dump(datum["traj"], out_file)

    end_time = datetime.now() - start_time

    print(f"Extraction completed in {end_time}")

    print(f"Number of missing GotoLocation without landmark is: {len(missing_landmarks)}")
    with open(f"{args.split_id}_missing_landmarks.jsonl", mode="w") as out_file:
        writer = jsonlines.Writer(out_file)
        writer.write_all(missing_landmarks)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--reward_config', type=str, default='configs/rewards.json')
    parser.add_argument('--image_width', type=int, default=300)
    parser.add_argument('--image_height', type=int, default=300)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--split_id', type=str, default="train",
                        help="The identifier of the split for which we should extract the features for")
    parser.add_argument('--ref_traj_file', type=str, default="ref_traj_data.json")
    parser.add_argument('--data_path', type=str, default="storage/data/alfred/json_feat_2.1.0")
    parser.add_argument('--splits', type=str, default="storage/data/alfred/splits/oct21.json")

    args = parser.parse_args()

    main(args)
