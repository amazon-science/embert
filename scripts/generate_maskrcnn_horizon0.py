import argparse
import collections
import json
import math
import os
from datetime import datetime

import torchvision
from allennlp.modules.vision.region_detector import RegionDetectorOutput
from torch import Tensor
from torchvision.models.detection.roi_heads import maskrcnn_inference
from torchvision.transforms import transforms

from grolp.utils.geometry_utils import calculate_angles
from vision.finetune import MaskRCNN

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from typing import cast, Iterable, Union, List, Tuple, NamedTuple, Optional, Dict

import numpy as np
import torch
from PIL import Image
import tqdm as tqdm
from allennlp.data import TorchImageLoader
from torch.utils.data import IterableDataset, DataLoader

from grolp.envs.thor_env import ThorEnv
import torchvision.ops.boxes as box_ops
import torch.nn.functional as F

TRAJ_DATA_JSON_FILENAME = "traj_data.json"

render_settings = dict()
render_settings['renderImage'] = True
render_settings['renderDepthImage'] = False
render_settings['renderObjectImage'] = False
render_settings['renderClassImage'] = False


class MaskDetectorOutput(NamedTuple):
    """
    The output type from the forward pass of a `RegionDetector`.
    """

    box_features: List[Tensor]
    """
    A list of tensors, each with shape `(num_boxes, feature_dim)`.
    """

    boxes: List[Tensor]

    masks: List[Tensor]

    class_probs: Optional[List[Tensor]] = None
    """
    An optional list of tensors. These tensors can have shape `(num_boxes,)` or
    `(num_boxes, *)` if probabilities for multiple classes are given.
    """

    class_labels: Optional[List[Tensor]] = None
    """
    An optional list of tensors that give the labels corresponding to the `class_probs`
    tensors. This should be non-`None` whenever `class_probs` is, and each tensor
    should have the same shape as the corresponding tensor from `class_probs`.
    """
    """
    A list of tensors containing the coordinates for each box. Each has shape `(num_boxes, 4)`.
    """


class MaskRCNNDetector(torch.nn.Module):
    """
    !!! Note
        This module does not have any trainable parameters by default.
        All pretrained weights are frozen.

    # Parameters

    box_score_thresh : `float`, optional (default = `0.05`)
        During inference, only proposal boxes / regions with a label classification score
        greater than `box_score_thresh` will be returned.

    box_nms_thresh : `float`, optional (default = `0.5`)
        During inference, non-maximum suppression (NMS) will applied to groups of boxes
        that share a common label.

        NMS iteratively removes lower scoring boxes which have an intersection-over-union (IoU)
        greater than `box_nms_thresh` with another higher scoring box.

    max_boxes_per_image : `int`, optional (default = `100`)
        During inference, at most `max_boxes_per_image` boxes will be returned. The
        number of boxes returned will vary by image and will often be lower
        than `max_boxes_per_image` depending on the values of `box_score_thresh`
        and `box_nms_thresh`.

    checkpoint: `str`, optional (default = `None`)
        If specified, we assume that we're loading a fine-tuned MaskRCNN model and not the one from Torchvision
    """

    def __init__(
            self,
            *,
            box_score_thresh: float = 0.05,
            box_nms_thresh: float = 0.5,
            max_boxes_per_image: int = 100,
            checkpoint_path: str = None,
            device="cpu"
    ):
        super().__init__()

        if checkpoint_path is None:
            self.detector = torchvision.models.detection.maskrcnn_resnet50_fpn(
                pretrained=True,
                box_score_thresh=box_score_thresh,
                box_nms_thresh=box_nms_thresh,
                box_detections_per_img=max_boxes_per_image,
            )
        else:

            if "moca" in checkpoint_path:
                maskrcnn = MaskRCNN(num_classes=119, hidden_size=256,
                                    inference_params=dict(box_score_thresh=box_score_thresh,
                                                          box_nms_thresh=box_nms_thresh,
                                                          box_detections_per_img=max_boxes_per_image))

                state_dict = torch.load(checkpoint_path, map_location="cpu")

                new_state_dict = {"detector." + k: v for k, v in state_dict.items()}

                maskrcnn.load_state_dict(new_state_dict)
                self.detector = maskrcnn.detector
            else:
                self.detector = MaskRCNN.load_from_checkpoint(
                    checkpoint_path,
                    inference_params=dict(box_score_thresh=box_score_thresh,
                                          box_nms_thresh=box_nms_thresh,
                                          box_detections_per_img=max_boxes_per_image)
                )
                # access to the actual MaskRCNN reference
                self.detector = self.detector.detector

        # Freeze all weights.
        for parameter in self.detector.parameters():
            parameter.requires_grad = False
        self.detector.eval()

    def forward(
            self,
            images: torch.FloatTensor
    ) -> RegionDetectorOutput:
        """
        Extract regions and region features from the given images.

        In most cases `image_features` should come directly from the `ResnetBackbone`
        `GridEmbedder`. The `images` themselves should be standardized and resized
        using the default settings for the `TorchImageLoader`.
        """
        if self.detector.training:
            raise RuntimeError(
                "MaskRcnnRegionDetector can not be used for training at the moment"
            )

        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.detector.transform(images)

        image_features = self.detector.backbone(images.tensors)
        if isinstance(image_features, torch.Tensor):
            image_features = collections.OrderedDict([('0', image_features)])

        # # `proposals` is a list of tensors, one tensor per image, each representing a
        # # fixed number of proposed regions/boxes.
        # # shape (proposals[i]): (proposals_per_image, 4)
        proposals: List[Tensor]
        proposals, _ = self.detector.rpn(images, image_features)
        #
        # outputs = self.detector.roi_heads(image_features, proposals, image_shapes)
        # # shape: (batch_size * proposals_per_image, *)
        box_features = self.detector.roi_heads.box_roi_pool(image_features, proposals, images.image_sizes)
        #
        # # shape: (batch_size * proposals_per_image, *)
        box_features = self.detector.roi_heads.box_head(box_features)
        #
        # # shape (class_logits): (batch_size * proposals_per_image, num_classes)
        # # shape (box_regression): (batch_size * proposals_per_image, regression_output_size)
        class_logits, box_regression = self.detector.roi_heads.box_predictor(box_features)

        # This step filters down the `proposals` to only detections that reach
        # a certain threshold.
        # Each of these is a list of tensors, one for each image in the batch.
        # shape (boxes[i]): (num_predicted_boxes, 4)
        # shape (features[i]): (num_predicted_boxes, feature_size)
        # shape (scores[i]): (num_predicted_classes,)
        # shape (labels[i]): (num_predicted_classes,)
        boxes, box_features, scores, labels = self._postprocess_detections(
            class_logits, box_features, box_regression, proposals, images.image_sizes
        )

        num_images = len(boxes)
        result = torch.jit.annotate(List[Dict[str, torch.Tensor]], [])
        for i in range(num_images):
            result.append(
                {
                    "features": box_features[i],
                    "boxes": boxes[i],
                    "labels": labels[i],
                    "scores": scores[i],
                }
            )

        # compute masks as well
        mask_proposals = boxes
        mask_features = self.detector.roi_heads.mask_roi_pool(image_features, mask_proposals, images.image_sizes)
        mask_features = self.detector.roi_heads.mask_head(mask_features)
        mask_logits = self.detector.roi_heads.mask_predictor(mask_features)

        labels = [r["labels"] for r in result]
        masks_probs = maskrcnn_inference(mask_logits, labels)
        for mask_prob, r in zip(masks_probs, result):
            r["masks"] = mask_prob

        detections = self.detector.transform.postprocess(result, images.image_sizes, original_image_sizes)

        return detections

    def _postprocess_detections(
            self,
            class_logits: Tensor,
            box_features: Tensor,
            box_regression: Tensor,
            proposals: List[Tensor],
            image_shapes: List[Tuple[int, int]],
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]:
        """
        Adapted from https://github.com/pytorch/vision/blob/
        4521f6d152875974e317fa247a633e9ad1ea05c8/torchvision/models/detection/roi_heads.py#L664.

        The only reason we have to re-implement this method is so we can pull out the box
        features that we want.
        """
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]

        # shape: (batch_size * boxes_per_image, num_classes, 4)
        pred_boxes = self.detector.roi_heads.box_coder.decode(box_regression, proposals)

        pred_scores = F.softmax(class_logits, -1)

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        features_list = box_features.split(boxes_per_image, dim=0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_features = []
        all_scores = []
        all_labels = []
        for boxes, features, scores, image_shape in zip(
                pred_boxes_list, features_list, pred_scores_list, image_shapes
        ):
            # shape: (boxes_per_image, num_classes, 4)
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # shape: (boxes_per_image, num_classes, feature_size)
            features = features.unsqueeze(1).expand(boxes.shape[0], boxes.shape[1], -1)

            # create labels for each prediction
            # shape: (num_classes,)
            labels = torch.arange(num_classes, device=device)
            # shape: (boxes_per_image, num_classes,)
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            # shape: (boxes_per_image, num_classes - 1, 4)
            boxes = boxes[:, 1:]
            # shape: (boxes_per_image, num_classes, feature_size)
            features = features[:, 1:]
            # shape: (boxes_per_image, num_classes - 1,)
            scores = scores[:, 1:]
            # shape: (boxes_per_image, num_classes - 1,)
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            # shape: (boxes_per_image * (num_classes - 1), 4)
            boxes = boxes.reshape(-1, 4)
            # shape: (boxes_per_image * (num_classes - 1), feature_size)
            features = features.reshape(boxes.shape[0], -1)
            # shape: (boxes_per_image * (num_classes - 1),)
            scores = scores.reshape(-1)
            # shape: (boxes_per_image * (num_classes - 1),)
            labels = labels.reshape(-1)

            # remove low scoring boxes
            inds = torch.where(scores > self.detector.roi_heads.score_thresh)[0]
            boxes, features, scores, labels = (
                boxes[inds],
                features[inds],
                scores[inds],
                labels[inds],
            )

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, features, scores, labels = (
                boxes[keep],
                features[keep],
                scores[keep],
                labels[keep],
            )

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.detector.roi_heads.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[: self.detector.roi_heads.detections_per_img]
            boxes, features, scores, labels = (
                boxes[keep],
                features[keep],
                scores[keep],
                labels[keep],
            )

            all_boxes.append(boxes)
            all_features.append(features)
            all_scores.append(scores)
            all_labels.append(labels)

        return all_boxes, all_features, all_scores, all_labels


class CustomImageLoader(TorchImageLoader):
    def __init__(self, *,
                 image_backend: str = None,
                 size_divisibility: int = 32,
                 **kwargs, ):
        super().__init__(image_backend=image_backend, size_divisibility=size_divisibility, **kwargs)
        self.transforms = transforms.Compose([
            transforms.ToTensor()
        ])

    def load(self, image):
        return self.transforms(image)

    def __call__(self, image_or_images: Union[Image.Image, Iterable[Image.Image]], pack=False):
        if not isinstance(image_or_images, (list, tuple)):
            image, size = self([image_or_images])
            return image[0], size[0]
            # return cast(torch.FloatTensor, image.squeeze(0)), cast(torch.IntTensor, size.squeeze(0))

        images: List[torch.FloatTensor] = []
        sizes: List[torch.IntTensor] = []
        for image in image_or_images:
            image = self.load(image).to(self.device)
            size = cast(
                torch.IntTensor,
                torch.tensor(
                    [image.shape[-2], image.shape[-1]], dtype=torch.int32, device=self.device
                ),
            )
            images.append(image)
            sizes.append(size)

        if pack:
            return torch.stack(images), torch.stack(sizes)

        return images, sizes


def create_panorama(env, rotation_steps):
    # This is the front view of the agent
    initial_agent = env.last_event.metadata["agent"]
    curr_image = Image.fromarray(env.last_event.frame)
    panorama_frames = [curr_image]
    camera_info = [dict(
        h_view_angle=env.last_event.metadata["agent"]["rotation"]["y"],
        # flip direction of heading angle - negative will be down and positive will be up
        v_view_angle=-env.last_event.metadata["agent"]["cameraHorizon"]
    )]

    # LookUp
    env.step({'action': 'LookUp', 'forceAction': True})
    completed_rotations = 0
    success = True
    angle = env.last_event.metadata["agent"]["rotation"]["y"]

    for look in range(rotation_steps):
        la_cmd = {'action': 'RotateRight', 'forceAction': True}
        event = env.step(la_cmd)

        success = success and event.metadata["lastActionSuccess"]
        angle = (angle + 90.0) % 360.0
        if success:
            completed_rotations += 1
            curr_image = Image.fromarray(event.frame)
            panorama_frames.append(curr_image)
            camera_info.append(dict(
                h_view_angle=env.last_event.metadata["agent"]["rotation"]["y"],
                v_view_angle=-env.last_event.metadata["agent"]["cameraHorizon"]
            ))
        else:
            # in this case
            panorama_frames.append(Image.fromarray(np.zeros_like(event.frame)))
            camera_info.append(dict(
                h_view_angle=angle,
                v_view_angle=-env.last_event.metadata["agent"]["cameraHorizon"]
            ))

    # at this step we just teleport to the original location
    teleport_action = {
        'action': 'TeleportFull',
        'rotation': initial_agent["rotation"],
        'x': initial_agent["position"]['x'],
        'z': initial_agent["position"]['z'],
        'y': initial_agent["position"]['y'],
        'horizon': initial_agent["cameraHorizon"],
        'tempRenderChange': True,
        'renderNormalsImage': False,
        'renderImage': render_settings['renderImage'],
        'renderClassImage': render_settings['renderClassImage'],
        'renderObjectImage': render_settings['renderObjectImage'],
        'renderDepthImage': render_settings['renderDepthImage'],
        'forceAction': True
    }
    env.step(teleport_action)

    assert env.last_event.metadata["lastActionSuccess"], "This shouldn't happen!"

    return panorama_frames, camera_info


class FailedReplay(Exception):
    def __init__(self, message):
        super(FailedReplay, self).__init__(message)


class ALFREDImageDataset(IterableDataset):
    def __init__(self, args):
        # trajectories are scattered among several files
        # we gather their file names here
        self.trajectories = []
        self.num_images = 0
        self.image_size = args.image_width
        self.rotation_degrees = 90
        # we have already the current frame
        self.rotation_steps = (360 // self.rotation_degrees) - 1

        visited = set()

        with open(args.splits) as in_file:
            splits = json.load(in_file)

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

                        self.trajectories.append(traj)
                        self.num_images += len(traj['plan']['low_actions']) * (self.rotation_steps + 1)

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
        # assumes that we have a squared video frame
        self.image_loader = CustomImageLoader(min_size=args.image_width, max_size=args.image_width)
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

        try:
            for idx in range(iter_start, iter_end):
                try:
                    for root_dir, step_index, images, camera_infos, is_traj_finished in \
                            self._get_instance(env, self.trajectories[idx]):
                        for pano_idx, (image, camera, done) in enumerate(zip(images, camera_infos, is_traj_finished)):
                            norm_image, size = self.image_loader(image)
                            yield root_dir, step_index, pano_idx, norm_image, size, camera, done
                except FailedReplay:
                    self.failed.append(self.trajectories[idx])
        except KeyboardInterrupt:
            pass
        finally:
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
        num_actions = len(traj_data['plan']['low_actions'])
        num_views = self.rotation_steps + 1

        for ll_idx, ll_action in enumerate(traj_data['plan']['low_actions']):
            # next cmd under the current hl_action
            cmd = ll_action['api_action']

            # remove unnecessary keys
            cmd = {k: cmd[k] for k in ['action', 'objectId', 'receptacleObjectId', 'placeStationary', 'forceAction'] if
                   k in cmd}

            # panorama_images = None
            panorama_images, camera_infos = create_panorama(env, self.rotation_steps)

            is_finished = [False] * num_views

            yield root_dir, ll_idx, panorama_images, camera_infos, is_finished
            event = env.step(cmd)

        if not event.metadata['lastActionSuccess']:
            raise FailedReplay("Replay Failed: %s" % (env.last_event.metadata['errorMessage']))

        curr_image = Image.fromarray(env.last_event.frame)
        panorama_images = [curr_image] + [Image.fromarray(np.zeros_like(curr_image)) for _ in
                                          range(num_views - 1)]
        camera_info = dict(
            h_view_angle=env.last_event.metadata["agent"]["rotation"]["y"],
            # flip direction of heading angle - negative will be down and positive will be up
            v_view_angle=-env.last_event.metadata["agent"]["cameraHorizon"]
        )
        camera_infos = [camera_info] + [dict(h_view_angle=0.0, v_view_angle=0.0) for _ in
                                        range(num_views - 1)]

        is_finished = [True] * (num_views)

        yield root_dir, num_actions, panorama_images, camera_infos, is_finished


def run(args):
    assert args.image_width == args.image_height, f"Squared video frames only (w={args.image_width} != h={args.image_height})"

    region_detector = MaskRCNNDetector(
        box_score_thresh=args.box_score_thresh,
        box_nms_thresh=args.box_nms_thresh,
        max_boxes_per_image=args.max_boxes_per_image,
        checkpoint_path=args.model_checkpoint
    )
    device = torch.device(args.cuda_device) if args.cuda_device != -1 else torch.device("cpu")

    region_detector.to(device)

    dataset = ALFREDImageDataset(args)

    loader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)

    start_time = datetime.now()

    for batch_idx, batch in enumerate(
            tqdm.tqdm(loader, desc=f"Generating MaskRCNN features for ALFRED {args.split_id}")):
        dirs, step_ids, pano_ids, images, sizes, camera_infos, is_finished = batch
        with torch.no_grad():
            # FasterRCNN feature extraction for the current frame
            images = images.to(device)
            detector_results = region_detector(images)

            paths_to_tensors = [
                (path, step_id, pano_ids[i], is_finished[i],
                 detector_results[i]["features"],
                 detector_results[i]["boxes"],
                 detector_results[i]["masks"],
                 detector_results[i]["scores"],
                 detector_results[i]["labels"]) for i, (path, step_id) in enumerate(zip(dirs, step_ids))
            ]

            for i, data in enumerate(paths_to_tensors):
                path, step_id, pano_id, done, box_features, boxes, masks, class_probs, class_labels = data
                features_path = os.path.join(path, args.features_folder)
                if not os.path.exists(features_path):
                    os.makedirs(features_path)
                output_file = os.path.join(features_path, f"{str(step_id.item())}-{str(pano_id.item())}.npz")

                num_boxes = args.panoramic_boxes[pano_id.item()]

                if boxes.shape[0] > 0:
                    boxes = boxes.cpu().numpy()
                    center_coords = (boxes[:, 0] + boxes[:, 2]) // 2, (
                            boxes[:, 1] + boxes[:, 3]) // 2

                    h_angle, v_angle = calculate_angles(
                        center_coords[0],
                        center_coords[1],
                        camera_infos["h_view_angle"][i].item(),
                        camera_infos["v_view_angle"][i].item()
                    )

                    boxes_angles = np.stack([h_angle, v_angle], 1)
                else:
                    boxes_angles = np.zeros((boxes.shape[0], 2))
                    boxes = boxes.cpu().numpy()

                box_features = box_features[:num_boxes]
                boxes_angles = boxes_angles[:num_boxes]
                boxes = boxes[:num_boxes]
                masks = masks[:num_boxes]
                class_probs = class_probs[:num_boxes]
                class_labels = class_labels[:num_boxes]

                np.savez_compressed(
                    output_file,
                    box_features=box_features.cpu().numpy(),
                    roi_angles=boxes_angles,
                    boxes=boxes,
                    masks=(masks > 0.5).cpu().numpy(),
                    class_probs=class_probs.cpu().numpy(),
                    class_labels=class_labels.cpu().numpy(),
                    num_objects=box_features.shape[0],
                    pano_id=pano_id
                )

                done = done.item()

                if done:
                    # this will store a file specifying that all the features have been generated for the current
                    # trajectory
                    with open(os.path.join(features_path, "done"), mode="w") as out_file:
                        out_file.write(str(done))

    if len(dataset.failed) > 0:
        print(f"Trajectory execution failed for {len(dataset.failed)} trajectories: ")
        for traj in dataset.failed:
            print(traj["path"])

    end_time = datetime.now()

    print(f"Total feature extraction time: {end_time - start_time}")


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--split_id', type=str, default="train",
                        help="The identifier of the split for which we should extract the features for")
    parser.add_argument('--features_folder', type=str, default="torch_maskrcnn")
    parser.add_argument('--model_checkpoint', type=str)
    parser.add_argument('--data_path', type=str, default="storage/data/alfred/json_feat_2.1.0")
    parser.add_argument('--splits', type=str, default="storage/data/alfred/splits/oct21.json")
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--reward_config', type=str, default='configs/rewards.json')
    parser.add_argument('--cuda_device', type=int, default=-1)
    parser.add_argument('--image_width', type=int, default=300)
    parser.add_argument('--image_height', type=int, default=300)

    ## FasterRCNN parameters
    parser.add_argument('--box_score_thresh', type=float, default=0.05)
    parser.add_argument('--box_nms_thresh', type=float, default=0.5)
    parser.add_argument('--max_boxes_per_image', type=float, default=36)
    parser.add_argument('--panoramic_boxes', nargs="+", default=(36, 18, 18, 18), type=int)

    args = parser.parse_args()

    run(args)
