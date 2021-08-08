import copy
import json
import math
import os
from argparse import ArgumentParser
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from grolp.envs.thor_env import ThorEnv


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


def extract_visible_objects(event: Event):
    visible_objects = defaultdict(set)
    in_hand_obj = event.metadata['inventoryObjects'][0]['objectId'] if event.metadata["inventoryObjects"] else None

    for obj_data in event.metadata["objects"]:
        o_id = obj_data["objectId"]
        o_type = obj_data["objectType"]
        is_visible = obj_data["visible"]

        if is_visible and o_id != in_hand_obj and o_id in event.instance_masks and o_id in event.instance_detections2D:
            visible_objects[o_type].add(o_id)

    return visible_objects


class ALFREDReplayDataset(IterableDataset):
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
                    # try with our version first
                    json_path = os.path.join(args.data_path, k, task['task'], 'ref_traj_data.json')
                    # if doesn't exist, backup to the original trajectory file
                    if not os.path.exists(json_path):
                        json_path = os.path.join(args.data_path, k, task['task'], 'traj_data.json')

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

        # make directories
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
        nav_actions = ['MoveAhead_25', 'RotateLeft_90', 'RotateRight_90', 'LookDown_15', 'LookUp_15']
        counter = 0
        total = 0

        # we first replay the entire trajectory
        for ll_idx, ll_action in enumerate(low_actions):
            hl_action = traj_data["plan"]['high_pddl'][low_actions[ll_idx]["high_idx"]]

            if hl_action["discrete_action"]["action"] != "GotoLocation" and ll_action["api_action"][
                "action"] not in nav_actions:
                visible_objects = extract_visible_objects(env.last_event)
                ref_object_class = ll_action["api_action"]["objectId"].split("|")[0]
                is_ambiguous = False
                mentions = 0
                for k, vals in visible_objects.items():
                    if ref_object_class in k:
                        # ButterKnife != Knife but still the user might refer to it as "knife"
                        mentions += 1
                        if len(vals) > 1:
                            is_ambiguous = True
                            break

                if is_ambiguous or mentions > 1:
                    counter += 1

                total += 1

            # next cmd under the current hl_action
            cmd = ll_action['api_action']

            cmd = {k: cmd[k] for k in
                   ['action', 'objectId', 'receptacleObjectId', 'placeStationary', 'forceAction'] if
                   k in cmd}
            env.step(cmd)

        return counter, total


def identity(x):
    return x


def main(args):
    dataset = ALFREDReplayDataset(args)
    data_loader = DataLoader(dataset, batch_size=1, num_workers=args.num_workers, collate_fn=identity)
    start_time = datetime.now()
    counter = 0
    total = 0
    for data in tqdm(data_loader, desc=f"Extracting statistics for split: {args.split_id}"):
        for (curr_counter, curr_total) in data:
            counter += curr_counter
            total += curr_total
    end_time = datetime.now() - start_time

    print(f"Extraction completed in {end_time}")

    print(f"Ambiguity rate: {(counter / total):.2f} ({counter}/{total})")
    with open(f"ambiguity_rate_{args.split_id}.json", mode="w") as out_file:
        json.dump({"counter": counter, "total": total, "ambiguity_rate": float(counter / total)}, out_file)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--reward_config', type=str, default='configs/rewards.json')
    parser.add_argument('--image_width', type=int, default=300)
    parser.add_argument('--image_height', type=int, default=300)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--split_id', type=str, default="train",
                        help="The identifier of the split for which we should extract the features for")
    parser.add_argument('--data_path', type=str, default="storage/data/alfred/json_feat_2.1.0")
    parser.add_argument('--splits', type=str, default="storage/data/alfred/splits/oct21.json")

    args = parser.parse_args()

    main(args)
