import argparse
import base64
import json
import math
import os
from datetime import datetime

from ai2thor.server import Event

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from PIL import Image
import tqdm as tqdm
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
from grolp.envs.thor_env import ThorEnv

TRAJ_DATA_JSON_FILENAME = "traj_data.json"


def mask_to_bbox(mask):
    # Bounding box.
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not np.any(rows) or not np.any(cols):
        return [0, 0, 0, 0]

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return [int(cmin), int(rmin), int(cmax), int(rmax)]


def extract_image_annotations(event: Event):
    annotations = []
    in_hand_obj = event.metadata['inventoryObjects'][0]['objectId'] if event.metadata["inventoryObjects"] else None

    for obj_data in event.metadata["objects"]:
        o_id = obj_data["objectId"]
        o_type = obj_data["objectType"]
        is_visible = obj_data["visible"]

        if is_visible and o_id != in_hand_obj and o_id in event.instance_masks and o_id in event.instance_detections2D:
            mask = event.instance_masks[o_id]
            bbox = event.instance_detections2D[o_id]

            obj = {
                "bbox": bbox.tolist(),
                "bbox_mode": "xyxy_abs",
                "mask": base64.b64encode(mask).decode("utf-8"),
                "category": o_type,
                # we store this so that we can read the mask back and reshape it
                "mask_shape": list(mask.shape)
            }

            annotations.append(obj)

    return annotations


class FailedReplay(Exception):
    def __init__(self, message):
        super(FailedReplay, self).__init__(message)


class AI2ThorRandomDataset(IterableDataset):
    def __init__(self, args):
        self.num_frames = args.num_frames
        self.num_samples = args.num_samples
        self.num_scenes = args.num_scences
        self.split_id = args.split_id
        self.start = 0
        self.end = self.num_samples * self.num_frames * self.num_scenes
        # TODO(jesse): you can store in self.samples some metadata associated with the random samples that you will
        # use later on to generate the images
        # In particular, make sure that the metadata dictionary for each instance has the following fields:
        # meta["split"] = reference split for the current example
        # meta["task_type"] = task type -- you could use it to group images together by scene?
        # meta['task_id'] = task id -- you could use it to group images by trajectory
        self.samples = []
        self.failed = []

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
        env = ThorEnv(player_screen_width=args.image_width,
                      player_screen_height=args.image_height)

        for idx in range(iter_start, iter_end):
            try:
                for root_dir, step_idx, image, annotations in self._get_instance(env, self.samples[idx]):
                    yield self.samples[idx], step_idx, image, annotations

            except FailedReplay:
                self.failed.append(self.samples[idx])

        env.stop()

    def _get_instance(self, env, sample_metadata):
        """
        TODO(jesse): you can specify here the logic to set up the environment and generate a trajectory

        This method assumes that you generate an iterator that `yields`:
            - step_idx: int
            - image: PIL.Image
            - annotations: List[Dict]: this should be the output of the function `extract_image_annotations` above
        """
        num_timesteps = 10

        for step in range(num_timesteps):
            # TODO: extract current frame
            image = None
            # TODO: extract annotations for current frame
            annotations = None

            yield step, image, annotations


class ALFREDImageDataset(IterableDataset):
    def __init__(self, args):
        # trajectories are scattered among several files
        # we gather their file names here
        self.trajectories = []
        visited = set()

        with open(args.splits) as in_file:
            splits = json.load(in_file)

        self.split_id = args.split_id
        self.num_images = 0

        stop = False
        for k, d in splits.items():
            if args.split_id in k:
                for task in d:
                    # load json file
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

                        self.num_images += len(traj['plan']['low_actions'])
                        self.trajectories.append(traj)

                        if args.debug and len(self.trajectories) == 2:
                            stop = True
                            break

                if stop:
                    print("Debugging mode on!")
                    break

        print(f"Discovered {len(self.trajectories)} files from the directory {args.data_path}")
        if args.debug:
            for traj in self.trajectories:
                print(traj["path"])

        self.image_size = args.image_width
        self.start = 0
        self.end = len(self.trajectories)
        self.failed = []

    def __len__(self):
        return self.num_images

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
        env = ThorEnv(player_screen_width=args.image_width,
                      player_screen_height=args.image_height)

        for idx in range(iter_start, iter_end):
            try:
                for root_dir, step_idx, image, annotations in self._get_instance(env, self.trajectories[idx]):
                    yield self.trajectories[idx], step_idx, image, annotations

            except FailedReplay:
                self.failed.append(self.trajectories[idx])

        env.stop()

    def _get_instance(self, env, traj_data):
        json_file = traj_data["path"]

        # make directories
        root_dir = json_file.replace(TRAJ_DATA_JSON_FILENAME, "")

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
        env.set_task(traj_data, args, reward_type='dense')

        for ll_idx, ll_action in enumerate(traj_data['plan']['low_actions']):
            curr_image = Image.fromarray(env.last_event.frame)

            annotations = extract_image_annotations(env.last_event)

            yield root_dir, ll_idx, curr_image, annotations

            # next cmd under the current hl_action
            cmd = ll_action['api_action']

            # remove unnecessary keys
            cmd = {k: cmd[k] for k in ['action', 'objectId', 'receptacleObjectId', 'placeStationary', 'forceAction'] if
                   k in cmd}
            event = env.step(cmd)

        if not event.metadata['lastActionSuccess']:
            raise FailedReplay("Replay Failed: %s" % (env.last_event.metadata['errorMessage']))


def run(args):
    assert args.image_width == args.image_height, f"Squared video frames only (w={args.image_width} != h={args.image_height})"

    if args.type == "alfred":
        dataset = ALFREDImageDataset(args)
    elif args.type == "random":
        dataset = AI2ThorRandomDataset(args)

    # We manually specify the batch size because the data loader doesn't have to wait other workers to generate instances
    # There is no real benefit in working in batch mode for the dataset generation procedure
    loader = DataLoader(dataset, shuffle=False, batch_size=1, num_workers=args.num_workers,
                        collate_fn=lambda x: x)

    start_time = datetime.now()

    all_data = []

    for batch_idx, batch in enumerate(
            tqdm.tqdm(loader, desc=f"Generating {args.split_id} dataset from ALFRED trajectories...")):
        for meta, step_id, image, annotations in batch:
            images_path = os.path.join(args.images_folder, meta["split"], meta["task_type"], meta['task_id'])
            if not os.path.exists(images_path):
                os.makedirs(images_path)

            if annotations:
                image_file = os.path.join(images_path, f"{step_id}.jpg")

                # we first store the image file in JPEG format
                image.save(image_file, "JPEG")

                # we add the 'step_id' to the metadata
                meta["step_id"] = step_id

                annotations_file = os.path.join(images_path, f"{step_id}.json")
                # then we store the metadata and annotations
                with open(annotations_file, mode="w") as out_file:
                    json.dump(dict(
                        metadata=meta,
                        file_name=image_file,
                        height=args.image_height,
                        width=args.image_width,
                        annotations=annotations
                    ), out_file)

                all_data.append(
                    annotations_file
                )

    if len(dataset.failed) > 0:
        print(f"Trajectory execution failed for {len(dataset.failed)} trajectories: ")
        for traj in dataset.failed:
            print(traj["path"])

    metadata_file = os.path.join(args.images_folder, args.split_id, "metadata.json")

    with open(metadata_file, mode="w") as out_file:
        json.dump(all_data, out_file)

    end_time = datetime.now()

    print(f"Dataset creation completed in: {end_time - start_time}")


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--type', type=str, default="alfred", choices=("alfred", "random"))
    parser.add_argument('--split_id', type=str,
                        help="The identifier of the split for which we should extract the data for", required=True)
    parser.add_argument('--images_folder', type=str, default="storage/data/alfred/images/")
    parser.add_argument('--data_path', type=str, default="storage/data/alfred/json_feat_2.1.0")
    parser.add_argument('--splits', type=str, default="storage/data/alfred/splits/oct21.json")
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--reward_config', type=str, default='configs/rewards.json')
    parser.add_argument('--image_width', type=int, default=300)
    parser.add_argument('--image_height', type=int, default=300)

    args = parser.parse_args()

    run(args)
