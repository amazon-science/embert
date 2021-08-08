import json
import logging
import pprint
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.multiprocessing as mp

from grolp import AlfredPredictor
from grolp.utils.geometry_utils import calculate_angles
from scripts.generate_maskrcnn_horizon0 import CustomImageLoader, MaskRCNNDetector, create_panorama
from scripts.process_dataset import process_with_reader

logger = logging.getLogger(__name__)


class Eval(object):
    # tokens
    STOP_TOKEN = "<<stop>>"
    SEQ_TOKEN = "<<seg>>"
    TERMINAL_TOKENS = [STOP_TOKEN, SEQ_TOKEN]

    def __init__(self, args, manager):
        # args and manager
        self.args = args
        self.manager = manager

        # # load splits
        with open(self.args.splits) as f:
            self.splits = json.load(f)
            pprint.pprint({k: len(v) for k, v in self.splits.items()})

        if self.args.cuda_device != -1 and not torch.cuda.is_available():
            logger.warning(
                f"You've specified CUDA device {self.args.cuda_device} but your setup doesn't actually support it. "
                f"We'll be using CPU for the evaluation...")
            # we default to CPU
            self.args.cuda_device = -1

        # load model
        print("Loading: ", self.args.model_path)

        self.predictor = AlfredPredictor.load_checkpoint(
            self.args
        )

        if hasattr(args, 'subgoals') and self.args.subgoals is not None and hasattr(self.args,
                                                                                    'preprocess') and self.args.preprocess:
            logger.info("Preprocessing enabled. Starting instance cache generation")
            start_time = datetime.now()
            process_with_reader(self.predictor._dataset_reader, self.args.preprocess_workers, self.args.eval_split)
            end_time = datetime.now() - start_time
            logger.info(f"Processing finished after: {end_time}")

        self.predictor.share_memory()

        self.region_detector = MaskRCNNDetector(
            box_score_thresh=self.args.box_score_thresh,
            box_nms_thresh=self.args.box_nms_thresh,
            max_boxes_per_image=self.args.max_boxes_per_image,
            checkpoint_path=self.args.maskrcnn_checkpoint
        )
        self.region_detector.eval()
        self.image_loader = CustomImageLoader(min_size=self.args.frame_size, max_size=self.args.frame_size)
        args.gpu = self.args.cuda_device != -1 and torch.cuda.is_available()
        # move models to device
        self.region_detector.share_memory()
        if self.args.cuda_device != -1:
            self.region_detector = self.region_detector.to(self.args.cuda_device)

        # success and failure lists
        self.create_stats()

        # set random seed for shuffling
        random.seed(int(time.time()))

    @classmethod
    def get_visual_features(cls, env, image_loader, region_detector, args,
                            cuda_device):
        # collect current robot view
        panorama_images, camera_infos = create_panorama(env, args.rotation_steps - 1)

        images, sizes = image_loader(panorama_images, pack=True)

        # FasterRCNN feature extraction for the current frame
        if cuda_device >= 0:
            images = images.to(cuda_device)

        detector_results = region_detector(images)

        object_features = []

        for i in range(len(detector_results)):
            num_boxes = args.panoramic_boxes[i]
            features = detector_results[i]["features"]
            coordinates = detector_results[i]["boxes"]
            class_probs = detector_results[i]["scores"]
            class_labels = detector_results[i]["labels"]
            masks = detector_results[i]["masks"]

            if coordinates.shape[0] > 0:
                coordinates = coordinates.cpu().numpy()
                center_coords = (coordinates[:, 0] + coordinates[:, 2]) // 2, (
                        coordinates[:, 1] + coordinates[:, 3]) // 2

                h_angle, v_angle = calculate_angles(
                    center_coords[0],
                    center_coords[1],
                    camera_infos[i]["h_view_angle"],
                    camera_infos[i]["v_view_angle"]
                )

                boxes_angles = np.stack([h_angle, v_angle], 1)
            else:
                boxes_angles = np.zeros((coordinates.shape[0], 2))
                coordinates = coordinates.cpu().numpy()

            box_features = features[:num_boxes]
            boxes_angles = boxes_angles[:num_boxes]
            boxes = coordinates[:num_boxes]
            masks = masks[:num_boxes]
            class_probs = class_probs[:num_boxes]
            class_labels = class_labels[:num_boxes]

            object_features.append(dict(
                box_features=box_features.cpu().numpy(),
                roi_angles=boxes_angles,
                boxes=boxes,
                masks=(masks > 0.5).cpu().numpy(),
                class_probs=class_probs.cpu().numpy(),
                class_labels=class_labels.cpu().numpy(),
                camera_info=camera_infos[i],
                num_objects=box_features.shape[0]
            ))

        return object_features

    def queue_tasks(self):
        '''
        create queue of trajectories to be evaluated
        '''
        task_queue = self.manager.Queue()
        files = self.splits[self.args.eval_split]

        # debugging: fast epoch
        if self.args.fast_epoch:
            files = files[:5]
            print("Running evaluation on 5 trajectories only: ")
            for f in files:
                print(f)

        for traj in files:
            task_queue.put(traj)
        return task_queue

    def spawn_threads(self):
        '''
        spawn multiple threads to run eval in parallel
        '''
        task_queue = self.queue_tasks()

        if self.args.debug:
            # run the code in main process when debugging
            lock = self.manager.Lock()
            self.run(self.args.eval_split,
                     self.predictor,
                     self.image_loader,
                     self.region_detector,
                     task_queue,
                     self.args,
                     lock,
                     self.successes,
                     self.failures,
                     self.results)
        else:
            # start multiple workers
            workers = []
            lock = self.manager.Lock()
            for n in range(self.args.num_workers):
                worker = mp.Process(target=self.run, args=(
                    self.args.eval_split,
                    self.predictor,
                    self.image_loader,
                    self.region_detector,
                    task_queue,
                    self.args,
                    lock,
                    self.successes,
                    self.failures,
                    self.results
                ))
                worker.start()
                workers.append(worker)

            for t in workers:
                t.join()

        # save
        self.save_results()

    @classmethod
    def setup_scene(cls, env, traj_data, r_idx, args, reward_type='dense'):
        '''
        intialize the scene and agent from the task info
        '''
        # scene setup
        scene_num = traj_data['scene']['scene_num']
        object_poses = traj_data['scene']['object_poses']
        dirty_and_empty = traj_data['scene']['dirty_and_empty']
        object_toggles = traj_data['scene']['object_toggles']

        scene_name = 'FloorPlan%d' % scene_num
        env.reset(scene_name)
        env.restore_scene(object_poses, object_toggles, dirty_and_empty)

        # initialize to start position
        env.step(dict(traj_data['scene']['init_action']))

        # print goal instr
        print("Task: %s" % (traj_data['turk_annotations']['anns'][r_idx]['task_desc']))

        # setup task for reward
        env.set_task(traj_data, args, reward_type=reward_type)

    @classmethod
    def run(cls, model, resnet, task_queue, args, lock, successes, failures):
        raise NotImplementedError()

    @classmethod
    def evaluate(cls, env, model, r_idx, resnet, traj_data, args, lock, successes, failures):
        raise NotImplementedError()

    def save_results(self):
        raise NotImplementedError()

    def create_stats(self):
        raise NotImplementedError()
