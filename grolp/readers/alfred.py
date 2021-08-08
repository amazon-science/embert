import base64
import copy
import json
import logging
import os
import pickle
import random
from collections import defaultdict
from typing import Iterable, Dict, Optional, List

import lmdb
import numpy as np
import torch
from allennlp.data import DatasetReader, Instance, TokenIndexer, Tokenizer, Token
from allennlp.data.fields import LabelField, ArrayField, TextField, ListField, MetadataField, TensorField
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer

from grolp.gen import constants
from grolp.models.config import VISUAL_EMB_SIZE, ROI_ANGLES_DIM, NUM_OBJECTS_PER_VIEW, ROI_REL_AREA_DIM, \
    ROI_COORDINATES_DIM
from grolp.utils.image_util import decompress_mask, mask_iou

# Makes sure to disable HuggingFace tokenizers parallelization to avoid deadlocks!
# https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def check_nan(array, extra=None):
    extra = extra or "None"

    if isinstance(array, torch.Tensor):
        tmp = array.sum()
        is_nan = torch.isnan(tmp) or torch.isinf(tmp)
    else:
        tmp = np.sum(array)
        is_nan = np.isnan(tmp) or np.isinf(tmp)

    if is_nan:
        logging.warning(f'NaN or Inf found in input tensor. Info: {extra}')
    return array


def get_angle_representation(angles):
    horizontal_rot = angles[:, 0]
    vertical_rot = angles[:, 1]

    # we use to model sin and cosine of the horizontal rotation to differentiate between different angles
    return np.stack([
        np.sin(horizontal_rot),
        np.cos(horizontal_rot),
        np.sin(vertical_rot)
    ], -1)


classes = ['0'] + constants.OBJECTS + ['AppleSliced', 'ShowerCurtain', 'TomatoSliced', 'LettuceSliced', 'Lamp',
                                       'ShowerHead', 'EggCracked', 'BreadSliced', 'PotatoSliced', 'Faucet']


@DatasetReader.register("alfred_pretraining")
class AlfredPretrainingReader(DatasetReader):
    def __init__(self,
                 data_root_path: str,
                 vis_feats_path: str,
                 splits_path: str,
                 visual_feature_repr: List[str] = ("box_features", "roi_angles", "boxes"),
                 frame_size: int = 300,  # it assumes a square frame size (300x300), default in AI2Thor
                 rotation_angle: float = 90,
                 max_objects_per_frame=9,
                 num_step_per_trajectory=15,
                 mask_token_prob: float = 0.15,
                 mask_visual_prob: float = 0.15,
                 tokenizer: Optional[Tokenizer] = None,
                 token_indexers: Optional[Dict[str, TokenIndexer]] = None,
                 dry_run=False,
                 write_cache=False,
                 instance_cache_dir="pretraining_instance_cache"):
        super().__init__(manual_multiprocess_sharding=False,
                         manual_distributed_sharding=False)

        self.mask_visual_prob = mask_visual_prob
        self.mask_token_prob = mask_token_prob
        self.num_step_per_trajectory = num_step_per_trajectory
        self.data_root_path = data_root_path
        self.instance_cache_dir = instance_cache_dir
        self.splits_path = splits_path
        self.visual_feature_repr = visual_feature_repr
        self.write_cache = write_cache
        # tokenizers and indexers
        if tokenizer is None:
            tokenizer = PretrainedTransformerTokenizer("bert-base-uncased", add_special_tokens=False)
        self._tokenizer = tokenizer
        if token_indexers is None:
            token_indexers = {"tokens": PretrainedTransformerIndexer("bert-base-uncased")}

        self._token_indexers = token_indexers
        self.frame_size = frame_size
        self.vis_feats_path = vis_feats_path
        self.rotation_angle = rotation_angle
        self.max_objects_per_frame = max_objects_per_frame
        self.dry_run = dry_run

    def apply_token_indexers(self, instance: Instance) -> None:
        if "instructions" in instance:
            if isinstance(instance["instructions"], TextField):
                instance["instructions"].token_indexers = self._token_indexers
            elif isinstance(instance["instructions"], ListField):
                for inst in instance["instructions"]:
                    inst.token_indexers = self._token_indexers

    def read(self, file_path):
        """
        Returns an iterator of instances that can be read from the file path.
        """
        for instance in self._read(file_path):  # type: ignore
            yield instance

    def __del__(self):
        if hasattr(self, "lmdb_env") and self.lmdb_env is not None:
            self.lmdb_env.close()
            self.lmdb_env = None

    def _read(self, split_id: str) -> Iterable[Instance]:
        with open(self.splits_path) as in_file:
            split_data = json.load(in_file)[split_id]

        for task in split_data:
            json_path = os.path.join(self.data_root_path, split_id, task['task'], 'traj_data.json')
            with open(json_path) as f:
                ex = json.load(f)
            # root & split
            ex['root'] = os.path.join(self.data_root_path, task['task'])
            ex['split'] = split_id
            ex['repeat_idx'] = task["repeat_idx"]
            r_idx = task["repeat_idx"]

            #########
            # inputs
            #########
            # step-by-step instructions
            high_descs = ex['turk_annotations']['anns'][r_idx]['high_descs']
            actions_low = ex['plan']['low_actions']
            high_low_mapping = defaultdict(list)
            for idx, action in enumerate(actions_low):
                high_low_mapping[action["high_idx"]].append(idx)

            # for high_idx, actions in high_low_mapping.items():
            #     for idx in random.sample(actions, min(self.num_step_per_trajectory, len(actions))):
            #         if high_idx < len(high_descs):
            #             curr_example = copy.copy(ex)
            #             curr_example["action_low"] = actions_low[idx]
            #             curr_example["high_desc"] = high_descs[high_idx]
            #             curr_example["step_idx"] = idx
            #             yield curr_example
            for high_idx, actions in high_low_mapping.items():
                # we extract the frame at the end of a given sub-goal
                idx = actions[-1]
                if high_idx < len(high_descs):
                    curr_example = copy.copy(ex)
                    curr_example["action_low"] = actions_low[idx]
                    curr_example["high_desc"] = high_descs[high_idx]
                    curr_example["step_idx"] = idx
                    yield curr_example

    def random_word(self, tokens):
        output_label = []

        for i, token in enumerate(tokens):
            prob = random.random()
            # mask token with 15% probability

            if prob < self.mask_token_prob:
                prob /= self.mask_token_prob

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = Token(self._tokenizer.tokenizer.mask_token)

                # 10% randomly change token to random token
                elif prob < 0.9:
                    random_idx = np.random.randint(self._tokenizer.tokenizer.vocab_size)
                    tokens[i] = Token(self._tokenizer.tokenizer.convert_ids_to_tokens(random_idx))

                # -> rest 10% randomly keep current token
                # append current token to output (we will predict these later)
                output_label.append(self._tokenizer.tokenizer.convert_tokens_to_ids([token.text])[0])
            else:
                # no masking token (will be ignored by loss function later)
                output_label.append(-100)

        return tokens, output_label

    def random_region(self, image_feat, image_mask, image_classes):
        """
        """
        num_boxes = image_mask.shape[0]
        output_labels = []

        for i in range(num_boxes):
            if image_mask[i] == 1:
                prob = random.random()
                # mask token with 15% probability

                if prob < self.mask_visual_prob:
                    prob /= self.mask_visual_prob

                    # 90% randomly change token to mask token
                    if prob < 0.9:
                        image_feat[i] = 0.0

                    # -> rest 10% randomly keep current token
                    # append current token to output (we will predict these later)
                    output_labels.append(image_classes[i])
                else:
                    # no masking token (will be ignored by loss function later)
                    output_labels.append(-100)
            else:
                # no masking token (will be ignored by loss function later)
                output_labels.append(-100)

        return np.array(output_labels, dtype=np.long)

    def get_task_root(self, ex):
        '''
        returns the folder path of a trajectory
        '''
        return os.path.join(self.data_root_path, ex['split'], *(ex['root'].split('/')[-2:]))

    def extract_visual_features(self, features_dict):
        """
        features_dict will contain:
            box_features=box_features.cpu().numpy(),
            roi_angles=boxes_angles,
            boxes=boxes,
            masks=(masks > 0.5).cpu().numpy(),
            class_probs=class_probs.cpu().numpy(),
            class_labels=class_labels.cpu().numpy(),
            num_objects=box_features.shape[0],
            pano_id=pano_id

        self.visual_feature_repr is a tuple indicating the key that we want to concatenate
        """

        # for this option we simply concatenate all the features
        visual_features = []

        for k in self.visual_feature_repr:
            feat = features_dict[k]

            # empty feature vector
            if feat.shape[0] == 0:
                remaining_shape = feat.shape[1:] if k != "roi_angles" else [ROI_ANGLES_DIM]
                feat = np.zeros((self.max_objects_per_frame, *remaining_shape))
            else:
                if k == "roi_angles":
                    # we extend the roi angles features here
                    feat = get_angle_representation(feat)

                if len(features_dict[k].shape) == 1:
                    feat = np.expand_dims(feat, -1)

            visual_features.append(feat)

        visual_features = np.concatenate(visual_features, 1).astype(np.float32)

        padding_elements = self.max_objects_per_frame - visual_features.shape[0]

        num_objects = int(features_dict["num_objects"])

        if num_objects == 0:
            num_objects = self.max_objects_per_frame
            object_masks = np.zeros((self.max_objects_per_frame, self.frame_size, self.frame_size), dtype=np.bool)
        else:
            object_masks = features_dict["masks"].squeeze(1).astype(np.bool)

        # we randomly shuffle the object features so that we break the ordering based on class scores
        # in this way we avoid imposing a bias on the target
        idx = list(range(visual_features.shape[0]))
        random.shuffle(idx)

        visual_features = visual_features[idx]

        object_masks = object_masks[idx]

        if padding_elements > 0:
            visual_features = np.concatenate(
                [visual_features, np.zeros((padding_elements, VISUAL_EMB_SIZE), dtype=np.float32)])
        visual_mask = [1] * num_objects + [0] * padding_elements

        return visual_features, visual_mask, object_masks

    def text_to_instance(self, split_id, ex) -> Instance:
        r_idx = ex["repeat_idx"]
        step_idx = ex['step_idx']

        if not hasattr(self, "lmdb_env"):
            instance_cache_path = os.path.join(self.data_root_path, self.instance_cache_dir)

            self.lmdb_env = lmdb.Environment(
                instance_cache_path,
                readonly=not self.write_cache,
                subdir=True,
                map_size=1024 * 1024 * 1024 * 1024,
                lock=self.write_cache
            )

        root = self.get_task_root(ex)

        instance_cache_key = os.path.join(root, f"instance_{r_idx}_{step_idx}").encode()

        # we do a cache lookup only if we set `write_cache`
        if not self.write_cache:
            with self.lmdb_env.begin(write=False, buffers=True) as txt:
                buffer = txt.get(instance_cache_key)
                # we check if we did a cache-hit, if not we will proceed
                if buffer is None:
                    raise ValueError(
                        f"Missing element from the cache for dataset split {split_id}: {instance_cache_key.decode()} "
                        f"Enable the flag `write_cache` of the dataset reader to create the cache."
                        f"You should be using the flag `--preprocess` for it!")

                instance = pickle.loads(buffer)

                return instance

        metadata = dict(
            split_id=split_id,
            root=root,
            high_desc=ex["high_desc"],
            task_type=ex["task_type"],
            task_id=ex["task_id"],
            repeat_idx=ex["repeat_idx"],
            step_idx=step_idx
        )

        # we simply count the number of instances that we could potentially generate
        if self.dry_run:
            return Instance(dict(metadata=MetadataField(metadata)))

        hd = ex["high_desc"]
        text = hd if hd.endswith(".") else hd + "."
        tokens = self._tokenizer.tokenize(text)

        tokens, tokens_label = self.random_word(tokens)

        tokens = self._tokenizer.add_special_tokens(tokens)
        # concatenate lm labels and account for CLS and SEP: [CLS] tokens [SEP]
        lang_labels = [-100] + tokens_label + [-100]

        language_instruction_field = TextField(tokens)

        rotation_steps = int(360 // self.rotation_angle)
        features_root = os.path.join(root, self.vis_feats_path)

        if not os.path.exists(features_root):
            raise ValueError(f"The visual features path {features_root} does not exist!")

        visual_features = []
        # always include an extra slot for the ResNet feature associated with the front view
        visual_attention_mask = []
        visual_class_labels = []

        for j in range(rotation_steps):
            feature_path = os.path.join(features_root, f"{step_idx}-{j}.npz")

            if os.path.exists(feature_path):
                with np.load(feature_path) as f_features:
                    class_labels = f_features["class_labels"]
                    features, attn_mask, _ = self.extract_visual_features(f_features)

                check_nan(features,
                          f"MaskRCNN features -- ID: {metadata['task_id']} -- "
                          f"Type: {metadata['task_type']} -- Rotation step: {j}")

                visual_features.append(features)
                visual_attention_mask.extend(attn_mask)
                visual_class_labels.append(class_labels)
            else:
                # handles cases where we don't have information for the current trajectory step (--rare--)
                visual_features.append(np.zeros((self.max_objects_per_frame, VISUAL_EMB_SIZE), dtype=np.float32))
                visual_attention_mask.extend([0] * self.max_objects_per_frame)
                visual_class_labels.extend([-100] * self.max_objects_per_frame)

        visual_features = np.concatenate(visual_features, 0)
        visual_attention_mask = np.array(visual_attention_mask, dtype=np.uint8)
        visual_class_labels = np.array(visual_attention_mask, dtype=np.uint8)
        visual_labels = self.random_region(visual_features, visual_attention_mask, visual_class_labels)

        visual_features = ArrayField(visual_features)
        visual_attention_mask = ArrayField(visual_attention_mask)
        visual_labels = TensorField(visual_labels, padding_value=-100)
        lang_labels = TensorField(np.array(lang_labels, dtype=np.long), padding_value=-100)

        metadata_field = MetadataField(metadata)

        instance = Instance(dict(
            metadata=metadata_field,
            instructions=language_instruction_field,
            visual_features=visual_features,
            visual_attention_mask=visual_attention_mask,
            lang_labels=lang_labels,
            visual_labels=visual_labels
        ))

        with self.lmdb_env.begin(write=True, buffers=True) as txt:
            txt.put(instance_cache_key, pickle.dumps(instance))

        return instance


@DatasetReader.register("alfred_supervised")
class AlfredSupervisedReader(DatasetReader):
    def __init__(self,
                 data_root_path: str,
                 vis_feats_path: str,
                 splits_path: str,
                 visual_feature_repr: List[str] = ("box_features", "roi_angles", "boxes"),
                 frame_size: int = 300,  # it assumes a square frame size (300x300), default in AI2Thor
                 use_progress_monitor: bool = False,
                 use_subgoal_completion: bool = False,
                 max_sub_goals: int = 25,
                 rotation_angle: float = 90,
                 max_objects_per_frame=9,
                 tokenizer: Optional[Tokenizer] = None,
                 token_indexers: Optional[Dict[str, TokenIndexer]] = None,
                 dry_run=False,
                 write_cache=False,
                 # this is ignored only for training
                 max_traj_length: int = 200,
                 instance_cache_dir="instance_cache"):
        super().__init__(manual_multiprocess_sharding=False,
                         manual_distributed_sharding=False)

        self.data_root_path = data_root_path
        self.instance_cache_dir = instance_cache_dir
        self.splits_path = splits_path
        self.visual_feature_repr = visual_feature_repr
        self.write_cache = write_cache
        # tokenizers and indexers
        if tokenizer is None:
            tokenizer = PretrainedTransformerTokenizer("bert-base-uncased", add_special_tokens=False)
        self._tokenizer = tokenizer
        if token_indexers is None:
            token_indexers = {"tokens": PretrainedTransformerIndexer("bert-base-uncased")}

        self._token_indexers = token_indexers
        self.frame_size = frame_size
        self.use_progress_monitor = use_progress_monitor
        self.max_subgoals = max_sub_goals
        self.use_subgoal_completion = use_subgoal_completion
        self.vis_feats_path = vis_feats_path
        self.rotation_angle = rotation_angle
        self.max_objects_per_frame = max_objects_per_frame
        self.dry_run = dry_run
        self.max_traj_length = max_traj_length

    def has_interaction(self, action):
        '''
        check if low-level action is interactive
        '''
        non_interact_actions = ['MoveAhead', 'Rotate', 'Look', '<<stop>>', '<<pad>>', '<<seg>>']
        if any(a in action for a in non_interact_actions):
            return False
        else:
            return True

    def fix_missing_high_pddl_end_action(self, ex):
        '''
        appends a terminal action to a sequence of high-level actions
        '''
        if ex['plan']['high_pddl'][-1]['planner_action']['action'] != 'End':
            ex['plan']['high_pddl'].append({
                'discrete_action': {'action': 'NoOp', 'args': []},
                'planner_action': {'value': 1, 'action': 'End'},
                'high_idx': len(ex['plan']['high_pddl'])
            })

    def merge_last_two_low_actions(self, ex, action_low):
        '''
        combines the last two action sequences into one sequence
        '''
        extra_seg = copy.deepcopy(action_low[-2])
        for sub in extra_seg:
            sub['high_idx'] = action_low[-3][0]['high_idx']
            action_low[-3].append(sub)
        del action_low[-2]
        action_low[-1][0]['high_idx'] = len(action_low) - 1

        return action_low

    def process_actions(self, ex, high_desc_tokens):
        # deal with missing end high-level action
        self.fix_missing_high_pddl_end_action(ex)

        # end action for low_actions
        end_action = {
            'api_action': {'action': 'NoOp'},
            'discrete_action': {'action': '<<stop>>', 'args': {}},
            'high_idx': ex['plan']['high_pddl'][-1]['high_idx']
        }

        # init action_low and action_high
        num_hl_actions = len(ex['plan']['high_pddl'])
        action_low = [list() for _ in range(num_hl_actions)]  # temporally aligned with HL actions
        action_high = []
        low_to_high_idx = []

        for a in (ex['plan']['low_actions'] + [end_action]):
            # high-level action index (subgoals)
            high_idx = a['high_idx']
            low_to_high_idx.append(high_idx)

            # low-level action (API commands)
            action_low[high_idx].append({
                'high_idx': a['high_idx'],
                'action': a['discrete_action']['action'],
                'action_high_args': a['discrete_action']['args'],
            })

            # low-level bounding box (not used in the model)
            if 'bbox' in a['discrete_action']['args']:
                xmin, ymin, xmax, ymax = [float(x) if x != 'NULL' else -1 for x in a['discrete_action']['args']['bbox']]
                action_low[high_idx][-1]['centroid'] = [
                    (xmin + (xmax - xmin) / 2) / self.frame_size,
                    (ymin + (ymax - ymin) / 2) / self.frame_size,
                ]
            else:
                action_low[high_idx][-1]['centroid'] = [-1, -1]

            # low-level interaction mask (Note: this mask needs to be decompressed)
            if 'mask' in a['discrete_action']['args']:
                mask = self.decompress_mask(a['discrete_action']['args']['mask'])
            else:
                mask = None
            action_low[high_idx][-1]['mask'] = mask

            # interaction validity
            valid_interact = 1 if self.has_interaction(a['discrete_action']['action']) else 0
            action_low[high_idx][-1]['valid_interact'] = valid_interact

        # high-level actions
        for a in ex['plan']['high_pddl']:
            action_high.append({
                'high_idx': a['high_idx'],
                'action': a['discrete_action']['action'],
                'action_high_args': a['discrete_action']['args'],
            })

        # check alignment between step-by-step language and action sequence segments
        action_low_seg_len = len(action_low)
        lang_instr_seg_len = len(high_desc_tokens)
        seg_len_diff = action_low_seg_len - lang_instr_seg_len
        if seg_len_diff != 0:
            assert (seg_len_diff == 1)  # sometimes the alignment is off by one  ¯\_(ツ)_/¯
            action_low = self.merge_last_two_low_actions(ex, action_low)
            low_to_high_idx[-1] = action_low[-1][0]["high_idx"]
            low_to_high_idx[-2] = action_low[-2][0]["high_idx"]
            action_high[-1]["high_idx"] = action_high[-2]["high_idx"]
            action_high[-2]["high_idx"] = action_high[-3]["high_idx"]

        # now we flatten action_low
        action_low = [a for action_seg in action_low for a in action_seg]
        return action_low, action_high, low_to_high_idx

    def get_task_root(self, ex):
        '''
        returns the folder path of a trajectory
        '''
        return os.path.join(self.data_root_path, ex['split'], *(ex['root'].split('/')[-2:]))

    def extract_visual_features(self, features_dict):
        """
        features_dict will contain:
            box_features=box_features.cpu().numpy(),
            roi_angles=boxes_angles,
            boxes=boxes,
            masks=(masks > 0.5).cpu().numpy(),
            class_probs=class_probs.cpu().numpy(),
            class_labels=class_labels.cpu().numpy(),
            num_objects=box_features.shape[0],
            pano_id=pano_id

        self.visual_feature_repr is a tuple indicating the key that we want to concatenate
        """

        # for this option we simply concatenate all the features
        visual_features = []

        for k in self.visual_feature_repr:
            feat = features_dict[k]

            # empty feature vector
            if feat.shape[0] == 0:
                remaining_shape = feat.shape[1:] if k != "roi_angles" else [ROI_ANGLES_DIM]
                feat = np.zeros((self.max_objects_per_frame, *remaining_shape))
            else:
                if k == "roi_angles":
                    # we extend the roi angles features here
                    feat = get_angle_representation(feat)

                if len(features_dict[k].shape) == 1:
                    feat = np.expand_dims(feat, -1)

            visual_features.append(feat)

        visual_features = np.concatenate(visual_features, 1).astype(np.float32)

        padding_elements = self.max_objects_per_frame - visual_features.shape[0]

        num_objects = int(features_dict["num_objects"])

        if num_objects == 0:
            num_objects = self.max_objects_per_frame
            object_masks = np.zeros((self.max_objects_per_frame, self.frame_size, self.frame_size), dtype=np.bool)
        else:
            object_masks = features_dict["masks"].astype(np.bool)

        # we randomly shuffle the object features so that we break the ordering based on class scores
        # in this way we avoid imposing a bias on the target
        idx = list(range(visual_features.shape[0]))
        random.shuffle(idx)

        visual_features = visual_features[idx]
        object_masks = object_masks[idx]

        if padding_elements > 0:
            visual_features = np.concatenate(
                [visual_features, np.zeros((padding_elements, VISUAL_EMB_SIZE), dtype=np.float32)])
        visual_mask = [1] * num_objects + [0] * padding_elements

        return visual_features, visual_mask, object_masks

    def decompress_mask(self, compressed_mask):
        '''
        decompress mask from json files
        '''
        mask = np.array(decompress_mask(compressed_mask), dtype=np.float32)
        mask = np.expand_dims(mask, axis=0)
        return mask

    def __del__(self):
        if hasattr(self, "lmdb_env") and self.lmdb_env is not None:
            self.lmdb_env.close()
            self.lmdb_env = None

    def apply_token_indexers(self, instance: Instance) -> None:
        if "instructions" in instance:
            if isinstance(instance["instructions"], TextField):
                instance["instructions"].token_indexers = self._token_indexers
            elif isinstance(instance["instructions"], ListField):
                for inst in instance["instructions"]:
                    inst.token_indexers = self._token_indexers

    def read(self, file_path):
        """
        Returns an iterator of instances that can be read from the file path.
        """
        for instance in self._read(file_path):  # type: ignore
            yield instance

    def _read(self, split_id: str) -> Iterable[Instance]:
        with open(self.splits_path) as in_file:
            split_data = json.load(in_file)[split_id]

        for task in split_data:
            json_path = os.path.join(self.data_root_path, split_id, task['task'], 'traj_data.json')
            with open(json_path) as f:
                ex = json.load(f)
            # root & split
            ex['root'] = os.path.join(self.data_root_path, task['task'])
            ex['split'] = split_id
            ex['repeat_idx'] = task["repeat_idx"]

            # we ignore all the trajectories with length higher than self.max_traj_length (default: 200)
            if split_id == "train" and len(ex["plan"]["low_actions"]) >= self.max_traj_length:
                continue

            yield ex  # self.text_to_instance(split_id, ex)

    def text_to_instance(self, split_id, ex) -> Instance:
        r_idx = ex["repeat_idx"]

        if not hasattr(self, "lmdb_env"):
            instance_cache_path = os.path.join(self.data_root_path, self.instance_cache_dir)

            self.lmdb_env = lmdb.Environment(
                instance_cache_path,
                readonly=not self.write_cache,
                subdir=True,
                map_size=1024 * 1024 * 1024 * 1024,
                lock=self.write_cache
            )

        # TODO: check how to compute these values for extra supervision signals
        # if test_mode:
        #     # subgoal completion supervision
        #     if self.use_subgoal_completion:
        #         features['subgoals_completed'] = np.array(traj['num']['low_to_high_idx']) / self.max_subgoals
        #
        #     # progress monitor supervision
        #     if self.use_progress_monitor:
        #         num_actions = len([a for sg in traj['num']['action_low'] for a in sg])
        #         subgoal_progress = [(i + 1) / float(num_actions) for i in range(num_actions)]
        #         features['subgoal_progress'] = subgoal_progress

        #########
        # inputs
        #########

        task_desc = ex['turk_annotations']['anns'][r_idx]['task_desc']

        # step-by-step instructions
        high_descs = ex['turk_annotations']['anns'][r_idx]['high_descs'] + ['stop.']

        root = self.get_task_root(ex)

        instance_cache_key = os.path.join(root, f"instance_{r_idx}").encode()

        # we do a cache lookup only if we set `write_cache`
        if not self.write_cache:
            with self.lmdb_env.begin(write=False, buffers=True) as txt:
                buffer = txt.get(instance_cache_key)
                # we check if we did a cache-hit, if not we will proceed
                if buffer is None:
                    raise ValueError(f"Missing element from the cache for dataset split {split_id}: {root}. "
                                     f"Enable the flag `write_cache` of the dataset reader to create the cache."
                                     f"You should be using the script `process_dataset.py` for it!")

                instance = pickle.loads(buffer)

                return instance

        metadata = dict(
            split_id=split_id,
            root=root,
            task_desc=task_desc,
            high_descs=high_descs,
            task_type=ex["task_type"],
            task_id=ex["task_id"],
            repeat_idx=ex["repeat_idx"]
        )

        # we simply count the number of instances that we could potentially generate
        if self.dry_run:
            return Instance(dict(metadata=MetadataField(metadata)))

        task_desc_tokens = self._tokenizer.tokenize(task_desc)

        high_desc_tokens = []

        for hd in high_descs:
            hd = hd.strip()
            text = hd if hd.endswith(".") else hd + "."
            high_desc_tokens.extend(self._tokenizer.tokenize(text))

        lang_tokens = self._tokenizer.add_special_tokens(task_desc_tokens, high_desc_tokens)

        language_instruction_field = TextField(lang_tokens)

        actions_low, actions_high, low_to_high = self.process_actions(ex, high_descs)

        num_low_actions = len(ex['plan']['low_actions']) + 1  # +1 for additional stop action

        assert num_low_actions == len(actions_low), "Number of actions is not correct!"
        # This should be a list of
        object_features = []

        rotation_steps = 360 // self.rotation_angle
        visual_attention_mask = []
        interact_object_masks = []
        obj_interact_targets = []

        features_root = os.path.join(root, self.vis_feats_path)

        if not os.path.exists(features_root):
            raise ValueError(f"The visual features path {features_root} does not exist!")

        for i in range(num_low_actions):
            curr_step_features = []
            # always include an extra slot for the ResNet feature associated with the front view
            curr_visual_attention_mask = []
            curr_objects_masks = []

            for j in range(rotation_steps):
                feature_path = os.path.join(features_root, f"{i}-{j}.npz")

                if os.path.exists(feature_path):
                    with np.load(feature_path) as f_features:
                        features, attn_mask, masks = self.extract_visual_features(f_features)

                    check_nan(features,
                              f"MaskRCNN features -- ID: {metadata['task_id']} -- "
                              f"Type: {metadata['task_type']} -- Rotation step: {j}")

                    curr_step_features.append(features)
                    curr_objects_masks.append(masks)
                    curr_visual_attention_mask.extend(attn_mask)
                else:
                    # handles cases where we don't have information for the current trajectory step (--rare--)
                    curr_step_features.append(np.zeros((self.max_objects_per_frame, VISUAL_EMB_SIZE), dtype=np.float32))
                    curr_objects_masks.append(
                        np.zeros((self.max_objects_per_frame, self.frame_size, self.frame_size), dtype=np.uint8))
                    curr_visual_attention_mask.extend([0] * self.max_objects_per_frame)

            curr_step_features = ArrayField(np.concatenate(curr_step_features, 0))
            curr_visual_attention_mask = ArrayField(np.array(curr_visual_attention_mask, dtype=np.uint8))
            object_features.append(curr_step_features)
            visual_attention_mask.append(curr_visual_attention_mask)
            interact_object_masks.append(curr_objects_masks)

            # derive gold targets for mask interaction
            if actions_low[i]["mask"] is not None:
                # we extract the gold mask for the current trajectory
                gold_mask = actions_low[i]["mask"].astype(np.bool)

                # we compute the IoU between the gold mask and the target
                front_view_masks = curr_objects_masks[0]
                iou_scores = [mask_iou(gold_mask, mask) for mask in front_view_masks]
                label = np.argmax(iou_scores)
            else:
                # we don't need a target for this step
                label = -100

            obj_interact_targets.append(label)

        scene_objects_features = ListField(object_features)
        visual_attention_mask = ListField(visual_attention_mask)

        episode_mask = ArrayField(np.ones((num_low_actions,), dtype=np.uint8))

        actions_low_field = ListField([LabelField(a['action'], "low_action_labels") for a in actions_low])

        # metadata["interactive_object_masks"] = interact_object_masks

        obj_interact_targets = torch.tensor(obj_interact_targets, dtype=torch.int64)
        obj_interact_mask = (obj_interact_targets != -100)
        obj_interact_targets = TensorField(obj_interact_targets, padding_value=-100)
        obj_interact_mask = TensorField(obj_interact_mask, dtype=torch.bool)

        metadata_field = MetadataField(metadata)

        instance = Instance(dict(
            metadata=metadata_field,
            instructions=language_instruction_field,
            actions=actions_low_field,
            visual_features=scene_objects_features,
            visual_attention_mask=visual_attention_mask,
            actions_mask=episode_mask,
            obj_interact_targets=obj_interact_targets,
            obj_interact_mask=obj_interact_mask
        ))

        with self.lmdb_env.begin(write=True, buffers=True) as txt:
            txt.put(instance_cache_key, pickle.dumps(instance))

        return instance


@DatasetReader.register("alfred_split_supervised_finegrained")
class AlfredSplitFinegrainedSupervisedReader(AlfredSupervisedReader):
    def __init__(self, data_root_path: str, vis_feats_path: str, splits_path: str,
                 visual_feature_repr: List[str] = ("box_features", "roi_angles", "boxes"), frame_size: int = 300,
                 use_progress_monitor: bool = False, use_subgoal_completion: bool = False, max_sub_goals: int = 25,
                 rotation_angle: float = 90, max_objects_per_frame=9, tokenizer: Optional[Tokenizer] = None,
                 token_indexers: Optional[Dict[str, TokenIndexer]] = None, dry_run=False, write_cache=False,
                 max_traj_length: int = 200, instance_cache_dir="split_finegrained_instance_cache"):
        super().__init__(data_root_path, vis_feats_path, splits_path, visual_feature_repr, frame_size,
                         use_progress_monitor, use_subgoal_completion, max_sub_goals, rotation_angle,
                         max_objects_per_frame, tokenizer, token_indexers, dry_run, write_cache, max_traj_length,
                         instance_cache_dir)

    def _read(self, split_id: str) -> Iterable[Instance]:
        with open(self.splits_path) as in_file:
            split_data = json.load(in_file)[split_id]

        for task in split_data:
            json_path = os.path.join(self.data_root_path, split_id, task['task'], 'ref_traj_data.json')
            with open(json_path) as f:
                ex = json.load(f)
            # root & split
            ex['root'] = os.path.join(self.data_root_path, task['task'])
            ex['split'] = split_id
            ex['repeat_idx'] = task["repeat_idx"]

            # we ignore all the trajectories with length higher than self.max_traj_length (default: 200)
            if split_id == "train" and len(ex["plan"]["low_actions"]) >= self.max_traj_length:
                continue

            if not self.write_cache:
                # Remove some useless fields to save memory
                for k in ("turk_annotations", "plan", "pddl_params", "scene", "images"):
                    if k == "plan":
                        # before we remove the field we create a fake one to be backcompatible
                        low_actions = ex[k]["low_actions"]
                        del ex[k]
                        ex["plan"] = {
                            "low_actions": [a['api_action'] for a in low_actions]
                        }
                    else:
                        del ex[k]

            yield ex  # self.text_to_instance(split_id, ex)

    def process_actions(self, ex, high_desc_tokens):
        # deal with missing end high-level action
        self.fix_missing_high_pddl_end_action(ex)

        # end action for low_actions
        end_action = {
            'api_action': {'action': 'NoOp'},
            'discrete_action': {'action': '<<stop>>', 'args': {}},
            'high_idx': ex['plan']['high_pddl'][-1]['high_idx']
        }

        # init action_low and action_high
        num_hl_actions = len(ex['plan']['high_pddl'])
        action_low = [list() for _ in range(num_hl_actions)]  # temporally aligned with HL actions
        action_high = []
        low_to_high_idx = []

        for a in (ex['plan']['low_actions'] + [end_action]):
            # high-level action index (subgoals)
            high_idx = a['high_idx']
            low_to_high_idx.append(high_idx)

            # low-level action (API commands)
            curr_action = {
                'high_idx': a['high_idx'],
                'action': a['discrete_action']['action'],
                'action_high_args': a['discrete_action']['args'],
                'subgoal': copy.deepcopy(a.get('subgoal', {}))
            }

            # low-level bounding box (not used in the model)
            if 'bbox' in a['discrete_action']['args']:
                xmin, ymin, xmax, ymax = [float(x) if x != 'NULL' else -1 for x in a['discrete_action']['args']['bbox']]
                curr_action['centroid'] = [
                    (xmin + (xmax - xmin) / 2) / self.frame_size,
                    (ymin + (ymax - ymin) / 2) / self.frame_size,
                ]
            else:
                curr_action['centroid'] = [-1, -1]

            # low-level interaction mask (Note: this mask needs to be decompressed)
            if 'mask' in a['discrete_action']['args']:
                mask = self.decompress_mask(a['discrete_action']['args']['mask'])
            else:
                mask = None

            object_mask = curr_action['subgoal'].get('object', {}).get('mask', None)

            if object_mask is not None:
                object_mask = np.frombuffer(base64.b64decode(object_mask.encode("utf-8")), dtype=np.uint8).reshape(
                    *curr_action['subgoal']['object']["mask_shape"])
                curr_action['subgoal']['object']['mask'] = object_mask

            receptacle_mask = curr_action['subgoal'].get('receptacle', {}).get('mask', None)
            if receptacle_mask is not None:
                receptacle_mask = np.frombuffer(base64.b64decode(receptacle_mask.encode("utf-8")),
                                                dtype=np.uint8).reshape(
                    *curr_action['subgoal']['receptacle']["mask_shape"])
                curr_action['subgoal']['receptacle']['mask'] = receptacle_mask

            curr_action['mask'] = mask

            # interaction validity
            valid_interact = 1 if self.has_interaction(a['discrete_action']['action']) else 0
            curr_action['valid_interact'] = valid_interact

            action_low[high_idx].append(curr_action)

        # high-level actions
        for a in ex['plan']['high_pddl']:
            action_high.append({
                'high_idx': a['high_idx'],
                'action': a['discrete_action']['action'],
                'action_high_args': a['discrete_action']['args'],
            })

        # check alignment between step-by-step language and action sequence segments
        action_low_seg_len = len(action_low)
        lang_instr_seg_len = len(high_desc_tokens)
        seg_len_diff = action_low_seg_len - lang_instr_seg_len
        if seg_len_diff != 0:
            assert (seg_len_diff == 1)  # sometimes the alignment is off by one  ¯\_(ツ)_/¯
            action_low = self.merge_last_two_low_actions(ex, action_low)
            low_to_high_idx[-1] = action_low[-1][0]["high_idx"]
            low_to_high_idx[-2] = action_low[-2][0]["high_idx"]
            action_high[-1]["high_idx"] = action_high[-2]["high_idx"]
            action_high[-2]["high_idx"] = action_high[-3]["high_idx"]

        # now we flatten action_low
        action_low = [a for action_seg in action_low for a in action_seg]
        return action_low, action_high, low_to_high_idx

    def text_to_instance(self, split_id, ex) -> Instance:
        r_idx = ex["repeat_idx"]

        if not hasattr(self, "lmdb_env"):
            instance_cache_path = os.path.join(self.data_root_path, self.instance_cache_dir)

            self.lmdb_env = lmdb.Environment(
                instance_cache_path,
                readonly=not self.write_cache,
                subdir=True,
                map_size=1024 * 1024 * 1024 * 1024,
                lock=self.write_cache
            )

        root = self.get_task_root(ex)
        instance_cache_key = os.path.join(root, f"instance_{r_idx}").encode()

        # we do a cache lookup only if we set `write_cache`
        if not self.write_cache:
            with self.lmdb_env.begin(write=False, buffers=True) as txt:
                buffer = txt.get(instance_cache_key)
                # we check if we did a cache-hit, if not we will proceed
                if buffer is None:
                    raise ValueError(f"Missing element from the cache for dataset split {split_id}: {root}. "
                                     f"Enable the flag `write_cache` of the dataset reader to create the cache."
                                     f"You should be using the script `process_dataset.py` for it!")

                instance = pickle.loads(buffer)

                return instance

        # TODO: check how to compute these values for extra supervision signals
        # if test_mode:
        #     # subgoal completion supervision
        #     if self.use_subgoal_completion:
        #         features['subgoals_completed'] = np.array(traj['num']['low_to_high_idx']) / self.max_subgoals
        #
        #     # progress monitor supervision
        #     if self.use_progress_monitor:
        #         num_actions = len([a for sg in traj['num']['action_low'] for a in sg])
        #         subgoal_progress = [(i + 1) / float(num_actions) for i in range(num_actions)]
        #         features['subgoal_progress'] = subgoal_progress

        #########
        # inputs
        #########

        task_desc = ex['turk_annotations']['anns'][r_idx]['task_desc']

        # step-by-step instructions
        high_descs = ex['turk_annotations']['anns'][r_idx]['high_descs'] + ['stop.']

        metadata = dict(
            split_id=split_id,
            root=root,
            task_desc=task_desc,
            high_descs=high_descs,
            task_type=ex["task_type"],
            task_id=ex["task_id"],
            repeat_idx=ex["repeat_idx"]
        )

        # we simply count the number of instances that we could potentially generate
        if self.dry_run:
            return Instance(dict(metadata=MetadataField(metadata)))

        actions_low, actions_high, low_to_high = self.process_actions(ex, high_descs)

        task_desc_tokens = self._tokenizer.tokenize(task_desc)
        high_descs_tokens = []
        start_instr_labels = []
        language_instructions = []

        for i, hd in enumerate(high_descs):
            high_desc_tokens = self._tokenizer.tokenize(hd)
            turn_tokens = self._tokenizer.add_special_tokens(task_desc_tokens, high_desc_tokens)

            high_descs_tokens.append(TextField(turn_tokens))

        for i in range(len(low_to_high)):
            # we now extract the language instruction associated with the current action
            language_instructions.append(high_descs_tokens[low_to_high[i]])

            # if we are at the beginning of a new language instruction we mark it as 1, 0 otherwise
            if i == len(low_to_high) - 1:
                start_instr_labels.append(1.0)
            else:
                if low_to_high[i + 1] != low_to_high[i]:
                    start_instr_labels.append(1.0)
                else:
                    start_instr_labels.append(0.0)

        start_instr_labels = TensorField(torch.tensor(start_instr_labels, dtype=torch.int32), padding_value=0)
        language_instructions = ListField(language_instructions)

        num_low_actions = len(ex['plan']['low_actions']) + 1  # +1 for additional stop action

        assert num_low_actions == len(actions_low), "Number of actions is not correct!"
        # This should be a list of
        object_features = []

        rotation_steps = 360 // self.rotation_angle
        visual_attention_mask = []
        interact_object_masks = []
        obj_interact_targets = []
        nav_object_labels = []
        nav_receptacle_labels = []

        features_root = os.path.join(root, self.vis_feats_path)

        if not os.path.exists(features_root):
            raise ValueError(f"The visual features path {features_root} does not exist!")

        for i in range(num_low_actions):
            curr_step_features = []
            # always include an extra slot for the ResNet feature associated with the front view
            curr_visual_attention_mask = []
            curr_objects_masks = []

            for j in range(rotation_steps):
                feature_path = os.path.join(features_root, f"{i}-{j}.npz")

                if os.path.exists(feature_path):
                    with np.load(feature_path) as f_features:
                        features, attn_mask, masks = self.extract_visual_features(f_features)

                    check_nan(features,
                              f"MaskRCNN features -- ID: {metadata['task_id']} -- "
                              f"Type: {metadata['task_type']} -- Rotation step: {j}")

                    curr_step_features.append(features)
                    curr_objects_masks.append(masks)

                    curr_visual_attention_mask.extend(attn_mask)
                else:
                    # handles cases where we don't have information for the current trajectory step (--rare--)
                    curr_step_features.append(np.zeros((self.max_objects_per_frame, VISUAL_EMB_SIZE), dtype=np.float32))
                    curr_objects_masks.append(
                        np.zeros((self.max_objects_per_frame, self.frame_size, self.frame_size), dtype=np.uint8))
                    curr_visual_attention_mask.extend([0] * self.max_objects_per_frame)

            curr_step_features = ArrayField(np.concatenate(curr_step_features, 0))
            curr_visual_attention_mask = ArrayField(np.array(curr_visual_attention_mask, dtype=np.uint8))
            object_features.append(curr_step_features)
            visual_attention_mask.append(curr_visual_attention_mask)
            interact_object_masks.append(curr_objects_masks)

            # -- derive gold targets for mask interaction
            if actions_low[i]["mask"] is not None:
                # we extract the gold mask for the current trajectory
                gold_mask = actions_low[i]["mask"].astype(np.bool)

                # we compute the IoU between the gold mask and the target
                front_view_masks = curr_objects_masks[0]
                iou_scores = [mask_iou(gold_mask, mask) for mask in front_view_masks]
                label = np.argmax(iou_scores)
            else:
                # we don't need a target for this step
                label = -100

            obj_interact_targets.append(label)
            # --------------------------------------------

            # -- derive fine-grained targets for each step
            subgoal_data = actions_low[i]["subgoal"]

            object_mask = subgoal_data.get("object", {}).get("mask")
            receptacle_mask = subgoal_data.get("receptacle", {}).get("mask")

            if object_mask is not None:

                # we compute the IoU between the gold mask and the target
                iou_scores = [mask_iou(object_mask, mask) for mask in [m for ms in curr_objects_masks for m in ms]]
                label = np.argmax(iou_scores)
                nav_object_labels.append(label)
            else:
                nav_object_labels.append(-100)

            if receptacle_mask is not None:
                # we compute the IoU between the gold mask and the target
                iou_scores = [mask_iou(receptacle_mask, mask) for mask in [m for ms in curr_objects_masks for m in ms]]
                label = np.argmax(iou_scores)
                nav_receptacle_labels.append(label)
            else:
                nav_receptacle_labels.append(-100)
            # --------------------------------------------

        scene_objects_features = ListField(object_features)
        visual_attention_mask = ListField(visual_attention_mask)

        episode_mask = ArrayField(np.ones((num_low_actions,), dtype=np.uint8))

        actions_low_field = ListField([LabelField(a['action'], "low_action_labels") for a in actions_low])

        # metadata["interactive_object_masks"] = interact_object_masks

        obj_interact_targets = torch.tensor(obj_interact_targets, dtype=torch.int64)
        obj_interact_mask = (obj_interact_targets != -100)
        obj_interact_targets = TensorField(obj_interact_targets, padding_value=-100)
        obj_interact_mask = TensorField(obj_interact_mask, dtype=torch.bool)

        nav_object_labels = TensorField(np.array(nav_object_labels), padding_value=-100)
        nav_receptacle_labels = TensorField(np.array(nav_receptacle_labels), padding_value=-100)

        metadata_field = MetadataField(metadata)

        instance = Instance(dict(
            metadata=metadata_field,
            instructions=language_instructions,
            start_instr_labels=start_instr_labels,
            actions=actions_low_field,
            visual_features=scene_objects_features,
            visual_attention_mask=visual_attention_mask,
            actions_mask=episode_mask,
            obj_interact_targets=obj_interact_targets,
            obj_interact_mask=obj_interact_mask,
            nav_object_labels=nav_object_labels,
            nav_receptacle_labels=nav_receptacle_labels
        ))

        with self.lmdb_env.begin(write=True, buffers=True) as txt:
            txt.put(instance_cache_key, pickle.dumps(instance))

        return instance


@DatasetReader.register("alfred_split_supervised")
class AlfredSplittedSupervisedReader(AlfredSupervisedReader):
    def __init__(self, data_root_path: str, vis_feats_path: str, splits_path: str,
                 visual_feature_repr: List[str] = ("box_features", "roi_angles", "boxes"), frame_size: int = 300,
                 use_progress_monitor: bool = False, use_subgoal_completion: bool = False, max_sub_goals: int = 25,
                 rotation_angle: float = 90, max_objects_per_frame=9, tokenizer: Optional[Tokenizer] = None,
                 token_indexers: Optional[Dict[str, TokenIndexer]] = None, dry_run=False, write_cache=False,
                 max_traj_length: int = 200, instance_cache_dir="split_instance_cache"):
        super().__init__(data_root_path, vis_feats_path, splits_path, visual_feature_repr, frame_size,
                         use_progress_monitor, use_subgoal_completion, max_sub_goals, rotation_angle,
                         max_objects_per_frame, tokenizer, token_indexers, dry_run, write_cache, max_traj_length,
                         instance_cache_dir)

    def text_to_instance(self, split_id, ex) -> Instance:
        r_idx = ex["repeat_idx"]

        if not hasattr(self, "lmdb_env"):
            instance_cache_path = os.path.join(self.data_root_path, self.instance_cache_dir)

            self.lmdb_env = lmdb.Environment(
                instance_cache_path,
                readonly=not self.write_cache,
                subdir=True,
                map_size=1024 * 1024 * 1024 * 1024,
                lock=self.write_cache
            )

        # TODO: check how to compute these values for extra supervision signals
        # if test_mode:
        #     # subgoal completion supervision
        #     if self.use_subgoal_completion:
        #         features['subgoals_completed'] = np.array(traj['num']['low_to_high_idx']) / self.max_subgoals
        #
        #     # progress monitor supervision
        #     if self.use_progress_monitor:
        #         num_actions = len([a for sg in traj['num']['action_low'] for a in sg])
        #         subgoal_progress = [(i + 1) / float(num_actions) for i in range(num_actions)]
        #         features['subgoal_progress'] = subgoal_progress

        #########
        # inputs
        #########

        task_desc = ex['turk_annotations']['anns'][r_idx]['task_desc']

        # step-by-step instructions
        high_descs = ex['turk_annotations']['anns'][r_idx]['high_descs'] + ['stop.']

        root = self.get_task_root(ex)

        metadata = dict(
            split_id=split_id,
            root=root,
            task_desc=task_desc,
            high_descs=high_descs,
            task_type=ex["task_type"],
            task_id=ex["task_id"],
            repeat_idx=ex["repeat_idx"]
        )

        # we simply count the number of instances that we could potentially generate
        if self.dry_run:
            return Instance(dict(metadata=MetadataField(metadata)))

        instance_cache_key = os.path.join(root, f"instance_{r_idx}").encode()

        # we do a cache lookup only if we set `write_cache`
        if not self.write_cache:
            with self.lmdb_env.begin(write=False, buffers=True) as txt:
                buffer = txt.get(instance_cache_key)
                # we check if we did a cache-hit, if not we will proceed
                if buffer is None:
                    raise ValueError(f"Missing element from the cache for dataset split {split_id}: {root}. "
                                     f"Enable the flag `write_cache` of the dataset reader to create the cache."
                                     f"You should be using the script `process_dataset.py` for it!")

                instance = pickle.loads(buffer)

                return instance

        actions_low, actions_high, low_to_high = self.process_actions(ex, high_descs)

        task_desc_tokens = self._tokenizer.tokenize(task_desc)
        high_descs_tokens = []
        start_instr_labels = []
        language_instructions = []

        for i, hd in enumerate(high_descs):
            high_desc_tokens = self._tokenizer.tokenize(hd)
            turn_tokens = self._tokenizer.add_special_tokens(task_desc_tokens, high_desc_tokens)

            high_descs_tokens.append(TextField(turn_tokens))

        for i in range(len(low_to_high)):
            # we now extract the language instruction associated with the current action
            language_instructions.append(high_descs_tokens[low_to_high[i]])

            # if we are at the beginning of a new language instruction we mark it as 1, 0 otherwise
            if i == len(low_to_high) - 1:
                start_instr_labels.append(1.0)
            else:
                if low_to_high[i + 1] != low_to_high[i]:
                    start_instr_labels.append(1.0)
                else:
                    start_instr_labels.append(0.0)

        start_instr_labels = TensorField(torch.tensor(start_instr_labels, dtype=torch.int32), padding_value=0)
        language_instructions = ListField(language_instructions)

        num_low_actions = len(ex['plan']['low_actions']) + 1  # +1 for additional stop action

        assert num_low_actions == len(actions_low), "Number of actions is not correct!"
        # This should be a list of
        object_features = []

        rotation_steps = 360 // self.rotation_angle
        visual_attention_mask = []
        interact_object_masks = []
        obj_interact_targets = []

        features_root = os.path.join(root, self.vis_feats_path)

        if not os.path.exists(features_root):
            raise ValueError(f"The visual features path {features_root} does not exist!")

        for i in range(num_low_actions):
            curr_step_features = []
            # always include an extra slot for the ResNet feature associated with the front view
            curr_visual_attention_mask = []
            curr_objects_masks = []

            for j in range(rotation_steps):
                feature_path = os.path.join(features_root, f"{i}-{j}.npz")

                if os.path.exists(feature_path):
                    with np.load(feature_path) as f_features:
                        features, attn_mask, masks = self.extract_visual_features(f_features)

                    check_nan(features,
                              f"MaskRCNN features -- ID: {metadata['task_id']} -- "
                              f"Type: {metadata['task_type']} -- Rotation step: {j}")

                    curr_step_features.append(features)
                    curr_objects_masks.append(masks)

                    curr_visual_attention_mask.extend(attn_mask)
                else:
                    # handles cases where we don't have information for the current trajectory step (--rare--)
                    curr_step_features.append(np.zeros((self.max_objects_per_frame, VISUAL_EMB_SIZE), dtype=np.float32))
                    curr_objects_masks.append(
                        np.zeros((self.max_objects_per_frame, self.frame_size, self.frame_size), dtype=np.uint8))
                    curr_visual_attention_mask.extend([0] * self.max_objects_per_frame)

            curr_step_features = ArrayField(np.concatenate(curr_step_features, 0))
            curr_visual_attention_mask = ArrayField(np.array(curr_visual_attention_mask, dtype=np.uint8))
            object_features.append(curr_step_features)
            visual_attention_mask.append(curr_visual_attention_mask)
            interact_object_masks.append(curr_objects_masks)

            # derive gold targets for mask interaction
            if actions_low[i]["mask"] is not None:
                # we extract the gold mask for the current trajectory
                gold_mask = actions_low[i]["mask"].astype(np.bool)

                # we compute the IoU between the gold mask and the target
                front_view_masks = curr_objects_masks[0]
                iou_scores = [mask_iou(gold_mask, mask) for mask in front_view_masks]
                label = np.argmax(iou_scores)
            else:
                # we don't need a target for this step
                label = -100

            obj_interact_targets.append(label)

        scene_objects_features = ListField(object_features)
        visual_attention_mask = ListField(visual_attention_mask)

        episode_mask = ArrayField(np.ones((num_low_actions,), dtype=np.uint8))

        actions_low_field = ListField([LabelField(a['action'], "low_action_labels") for a in actions_low])

        # metadata["interactive_object_masks"] = interact_object_masks

        obj_interact_targets = torch.tensor(obj_interact_targets, dtype=torch.int64)
        obj_interact_mask = (obj_interact_targets != -100)
        obj_interact_targets = TensorField(obj_interact_targets, padding_value=-100)
        obj_interact_mask = TensorField(obj_interact_mask, dtype=torch.bool)

        metadata_field = MetadataField(metadata)

        instance = Instance(dict(
            metadata=metadata_field,
            instructions=language_instructions,
            start_instr_labels=start_instr_labels,
            actions=actions_low_field,
            visual_features=scene_objects_features,
            visual_attention_mask=visual_attention_mask,
            actions_mask=episode_mask,
            obj_interact_targets=obj_interact_targets,
            obj_interact_mask=obj_interact_mask
        ))

        with self.lmdb_env.begin(write=True, buffers=True) as txt:
            txt.put(instance_cache_key, pickle.dumps(instance))

        return instance


@DatasetReader.register("alfred_split_next_supervised")
class AlfredSplittedWithNextSupervisedReader(AlfredSupervisedReader):
    def __init__(self, data_root_path: str, vis_feats_path: str, splits_path: str,
                 visual_feature_repr: List[str] = ("box_features", "roi_angles", "boxes"), frame_size: int = 300,
                 use_progress_monitor: bool = False, use_subgoal_completion: bool = False, max_sub_goals: int = 25,
                 rotation_angle: float = 90, max_objects_per_frame=9, tokenizer: Optional[Tokenizer] = None,
                 token_indexers: Optional[Dict[str, TokenIndexer]] = None, dry_run=False, write_cache=False,
                 max_traj_length: int = 200, instance_cache_dir="split_next_instance_cache"):
        super().__init__(data_root_path, vis_feats_path, splits_path, visual_feature_repr, frame_size,
                         use_progress_monitor, use_subgoal_completion, max_sub_goals, rotation_angle,
                         max_objects_per_frame, tokenizer, token_indexers, dry_run, write_cache, max_traj_length,
                         instance_cache_dir)

    def text_to_instance(self, split_id, ex) -> Instance:
        r_idx = ex["repeat_idx"]

        if not hasattr(self, "lmdb_env"):
            instance_cache_path = os.path.join(self.data_root_path, self.instance_cache_dir)

            self.lmdb_env = lmdb.Environment(
                instance_cache_path,
                readonly=not self.write_cache,
                subdir=True,
                map_size=1024 * 1024 * 1024 * 1024,
                lock=self.write_cache
            )

        # TODO: check how to compute these values for extra supervision signals
        # if test_mode:
        #     # subgoal completion supervision
        #     if self.use_subgoal_completion:
        #         features['subgoals_completed'] = np.array(traj['num']['low_to_high_idx']) / self.max_subgoals
        #
        #     # progress monitor supervision
        #     if self.use_progress_monitor:
        #         num_actions = len([a for sg in traj['num']['action_low'] for a in sg])
        #         subgoal_progress = [(i + 1) / float(num_actions) for i in range(num_actions)]
        #         features['subgoal_progress'] = subgoal_progress

        #########
        # inputs
        #########

        task_desc = ex['turk_annotations']['anns'][r_idx]['task_desc']

        # step-by-step instructions
        high_descs = ex['turk_annotations']['anns'][r_idx]['high_descs'] + ['stop.']

        root = self.get_task_root(ex)

        metadata = dict(
            split_id=split_id,
            root=root,
            task_desc=task_desc,
            high_descs=high_descs,
            task_type=ex["task_type"],
            task_id=ex["task_id"],
            repeat_idx=ex["repeat_idx"]
        )

        # we simply count the number of instances that we could potentially generate
        if self.dry_run:
            return Instance(dict(metadata=MetadataField(metadata)))

        instance_cache_key = os.path.join(root, f"instance_{r_idx}").encode()

        # we do a cache lookup only if we set `write_cache`
        if not self.write_cache:
            with self.lmdb_env.begin(write=False, buffers=True) as txt:
                buffer = txt.get(instance_cache_key)
                # we check if we did a cache-hit, if not we will proceed
                if buffer is None:
                    raise ValueError(f"Missing element from the cache for dataset split {split_id}: {root}. "
                                     f"Enable the flag `write_cache` of the dataset reader to create the cache."
                                     f"You should be using the script `process_dataset.py` for it!")

                instance = pickle.loads(buffer)

                return instance

        actions_low, actions_high, low_to_high = self.process_actions(ex, high_descs)

        task_desc_tokens = self._tokenizer.tokenize(task_desc)
        high_descs_tokens = []
        start_instr_labels = []
        language_instructions = []

        for i, hd in enumerate(high_descs):
            hd1 = hd.strip()
            text = hd1 if hd1.endswith(".") else hd1 + "."
            if i + 1 < len(high_descs):
                hd2 = high_descs[i + 1].strip()
                hd2 = hd2 if hd2.endswith(".") else hd2 + "."
                text += hd2
            high_desc_tokens = self._tokenizer.tokenize(text)
            turn_tokens = self._tokenizer.add_special_tokens(task_desc_tokens, high_desc_tokens)

            high_descs_tokens.append(TextField(turn_tokens))

        for i in range(len(low_to_high)):
            # we now extract the language instruction associated with the current action
            language_instructions.append(high_descs_tokens[low_to_high[i]])

            # if we are at the beginning of a new language instruction we mark it as 1, 0 otherwise
            if i == len(low_to_high) - 1:
                start_instr_labels.append(1.0)
            else:
                if low_to_high[i + 1] != low_to_high[i]:
                    start_instr_labels.append(1.0)
                else:
                    start_instr_labels.append(0.0)

        start_instr_labels = TensorField(torch.tensor(start_instr_labels, dtype=torch.int32), padding_value=0)
        language_instructions = ListField(language_instructions)

        num_low_actions = len(ex['plan']['low_actions']) + 1  # +1 for additional stop action

        assert num_low_actions == len(actions_low), "Number of actions is not correct!"
        # This should be a list of
        object_features = []

        rotation_steps = 360 // self.rotation_angle
        visual_attention_mask = []
        interact_object_masks = []
        obj_interact_targets = []

        features_root = os.path.join(root, self.vis_feats_path)

        if not os.path.exists(features_root):
            raise ValueError(f"The visual features path {features_root} does not exist!")

        for i in range(num_low_actions):
            curr_step_features = []
            # always include an extra slot for the ResNet feature associated with the front view
            curr_visual_attention_mask = []
            curr_objects_masks = []

            for j in range(rotation_steps):
                feature_path = os.path.join(features_root, f"{i}-{j}.npz")

                if os.path.exists(feature_path):
                    with np.load(feature_path) as f_features:
                        features, attn_mask, masks = self.extract_visual_features(f_features)

                    check_nan(features,
                              f"MaskRCNN features -- ID: {metadata['task_id']} -- "
                              f"Type: {metadata['task_type']} -- Rotation step: {j}")

                    curr_step_features.append(features)
                    curr_objects_masks.append(masks)

                    curr_visual_attention_mask.extend(attn_mask)
                else:
                    # handles cases where we don't have information for the current trajectory step (--rare--)
                    curr_step_features.append(np.zeros((self.max_objects_per_frame, VISUAL_EMB_SIZE), dtype=np.float32))
                    curr_objects_masks.append(
                        np.zeros((self.max_objects_per_frame, self.frame_size, self.frame_size), dtype=np.uint8))
                    curr_visual_attention_mask.extend([0] * self.max_objects_per_frame)

            curr_step_features = ArrayField(np.concatenate(curr_step_features, 0))
            curr_visual_attention_mask = ArrayField(np.array(curr_visual_attention_mask, dtype=np.uint8))
            object_features.append(curr_step_features)
            visual_attention_mask.append(curr_visual_attention_mask)
            interact_object_masks.append(curr_objects_masks)

            # derive gold targets for mask interaction
            if actions_low[i]["mask"] is not None:
                # we extract the gold mask for the current trajectory
                gold_mask = actions_low[i]["mask"].astype(np.bool)

                # we compute the IoU between the gold mask and the target
                front_view_masks = curr_objects_masks[0]
                iou_scores = [mask_iou(gold_mask, mask) for mask in front_view_masks]
                label = np.argmax(iou_scores)
            else:
                # we don't need a target for this step
                label = -100

            obj_interact_targets.append(label)

        scene_objects_features = ListField(object_features)
        visual_attention_mask = ListField(visual_attention_mask)

        episode_mask = ArrayField(np.ones((num_low_actions,), dtype=np.uint8))

        actions_low_field = ListField([LabelField(a['action'], "low_action_labels") for a in actions_low])

        # metadata["interactive_object_masks"] = interact_object_masks

        obj_interact_targets = torch.tensor(obj_interact_targets, dtype=torch.int64)
        obj_interact_mask = (obj_interact_targets != -100)
        obj_interact_targets = TensorField(obj_interact_targets, padding_value=-100)
        obj_interact_mask = TensorField(obj_interact_mask, dtype=torch.bool)

        metadata_field = MetadataField(metadata)

        instance = Instance(dict(
            metadata=metadata_field,
            instructions=language_instructions,
            start_instr_labels=start_instr_labels,
            actions=actions_low_field,
            visual_features=scene_objects_features,
            visual_attention_mask=visual_attention_mask,
            actions_mask=episode_mask,
            obj_interact_targets=obj_interact_targets,
            obj_interact_mask=obj_interact_mask
        ))

        with self.lmdb_env.begin(write=True, buffers=True) as txt:
            txt.put(instance_cache_key, pickle.dumps(instance))

        return instance


@DatasetReader.register("embert_supervised_finegrained")
class EmbertSupervisedReader(AlfredSplitFinegrainedSupervisedReader):
    def __init__(self, data_root_path: str, vis_feats_path: str, splits_path: str,
                 visual_feature_repr: List[str] = ("box_features", "roi_angles", "boxes"), frame_size: int = 300,
                 use_progress_monitor: bool = False, use_subgoal_completion: bool = False, max_sub_goals: int = 25,
                 mask_token_prob: float = -0.0, mask_object_prob: float = -0.0,
                 rotation_angle: float = 90, num_objects_per_view=NUM_OBJECTS_PER_VIEW,
                 tokenizer: Optional[Tokenizer] = None,
                 token_indexers: Optional[Dict[str, TokenIndexer]] = None, dry_run=False, write_cache=False,
                 max_traj_length: int = 200, instance_cache_dir="embert_instance_cache",
                 use_obj_nav=True):
        super().__init__(data_root_path, vis_feats_path, splits_path, visual_feature_repr, frame_size,
                         use_progress_monitor, use_subgoal_completion, max_sub_goals, rotation_angle,
                         num_objects_per_view[0], tokenizer, token_indexers, dry_run, write_cache, max_traj_length,
                         instance_cache_dir)
        self.mask_token_prob = mask_token_prob
        self.mask_object_prob = mask_object_prob
        self.num_objects_per_view = num_objects_per_view
        self.use_obj_nav = use_obj_nav

    def random_word(self, tokens):
        output_label = []
        output_tokens = []

        for i, token in enumerate(tokens):
            prob = random.random()
            # mask token with 15% probability

            if prob < self.mask_token_prob and token.text not in {"[CLS]", "[SEP]"}:
                prob /= self.mask_token_prob

                # 80% randomly change token to mask token
                if prob < 0.8:
                    output_tokens.append(Token(self._tokenizer.tokenizer.mask_token))

                # 10% randomly change token to random token
                elif prob < 0.9:
                    random_idx = np.random.randint(self._tokenizer.tokenizer.vocab_size)
                    output_tokens.append(Token(self._tokenizer.tokenizer.convert_ids_to_tokens(random_idx)))
                else:
                    output_tokens.append(tokens[i])

                # -> rest 10% randomly keep current token
                # append current token to output (we will predict these later)
                output_label.append(self._tokenizer.tokenizer.convert_tokens_to_ids([token.text])[0])
            else:
                # no masking token (will be ignored by loss function later)
                output_label.append(-100)
                output_tokens.append(tokens[i])

        return output_tokens, output_label

    def random_region(self, image_feat, image_mask, image_classes):
        """
        """
        num_boxes = image_mask.shape[0]
        output_labels = []

        for i in range(num_boxes):
            if image_mask[i] == 1:
                prob = random.random()
                # mask token with 15% probability

                if prob < self.mask_object_prob:
                    prob /= self.mask_object_prob

                    # append current token to output (we will predict these later)
                    output_labels.append(image_classes[i])
                else:
                    # no masking token (will be ignored by loss function later)
                    output_labels.append(-100)
            else:
                # no masking token (will be ignored by loss function later)
                output_labels.append(-100)

        return output_labels

    def extract_visual_features(self, features_dict, step):
        """
        features_dict will contain:
            box_features=box_features.cpu().numpy(),
            roi_angles=boxes_angles,
            boxes=boxes,
            masks=(masks > 0.5).cpu().numpy(),
            class_probs=class_probs.cpu().numpy(),
            class_labels=class_labels.cpu().numpy(),
            num_objects=box_features.shape[0],
            pano_id=pano_id

        self.visual_feature_repr is a tuple indicating the key that we want to concatenate
        """

        max_objects_in_frame = self.num_objects_per_view[step]

        # for this option we simply concatenate all the features
        visual_features = []

        for k in self.visual_feature_repr:
            feat = features_dict[k]

            # empty feature vector
            if feat.shape[0] == 0:
                if k == "roi_angles":
                    remaining_shape = [ROI_ANGLES_DIM]
                elif k == "boxes":
                    remaining_shape = [ROI_REL_AREA_DIM + ROI_COORDINATES_DIM]
                else:
                    remaining_shape = feat.shape[1:]
                feat = np.zeros((max_objects_in_frame, *remaining_shape))
            else:
                if k == "roi_angles":
                    # we extend the roi angles features here
                    feat = get_angle_representation(feat)
                elif k == "boxes":
                    vis_pe = torch.from_numpy(feat)
                    w_est = torch.max(vis_pe[:, [0, 2]]) * 1. + 1e-5
                    h_est = torch.max(vis_pe[:, [1, 3]]) * 1. + 1e-5
                    vis_pe[:, [0, 2]] /= w_est
                    vis_pe[:, [1, 3]] /= h_est
                    assert h_est > 0, 'should greater than 0! {}'.format(h_est)
                    assert w_est > 0, 'should greater than 0! {}'.format(w_est)
                    rel_area = (vis_pe[:, 3] - vis_pe[:, 1]) * (vis_pe[:, 2] - vis_pe[:, 0])
                    rel_area.clamp_(0)
                    vis_pe = torch.cat((vis_pe, rel_area.view(-1, 1)), -1)
                    vis_pe = torch.nn.functional.normalize(vis_pe - 0.5, dim=-1)
                    feat = vis_pe.numpy()

                if len(features_dict[k].shape) == 1:
                    feat = np.expand_dims(feat, -1)

            visual_features.append(feat)

        visual_features = np.concatenate(visual_features, 1).astype(np.float32)

        padding_elements = max_objects_in_frame - visual_features.shape[0]

        num_objects = int(features_dict["num_objects"])

        if num_objects == 0:
            num_objects = max_objects_in_frame
            object_masks = np.zeros((max_objects_in_frame, self.frame_size, self.frame_size), dtype=np.bool)
            object_classes = np.ones((max_objects_in_frame,), dtype=np.long) * -100
        else:
            object_masks = features_dict["masks"].squeeze(1).astype(np.bool)
            object_classes = features_dict["class_labels"]

        # we randomly shuffle the object features so that we break the ordering based on class scores
        # in this way we avoid imposing a bias on the target
        idx = list(range(visual_features.shape[0]))
        random.shuffle(idx)

        visual_features = visual_features[idx]
        object_classes = object_classes[idx]
        object_masks = object_masks[idx]

        if padding_elements > 0:
            visual_features = np.concatenate(
                [visual_features, np.zeros((padding_elements, VISUAL_EMB_SIZE), dtype=np.float32)])
            object_classes = np.concatenate([object_classes, np.ones((padding_elements,), dtype=np.long) * -100])
            object_masks = np.concatenate(
                [object_masks, np.zeros((padding_elements, *object_masks.shape[1:]), dtype=np.bool)])
        visual_mask = [1] * num_objects + [0] * padding_elements

        return visual_features, visual_mask, object_masks, object_classes

    def _mask_instance(self, instance: Instance):
        start_instr_labels = instance.fields["start_instr_labels"].tensor
        token_labels = []
        visual_labels = []

        for idx, flag in enumerate(start_instr_labels):
            flag = flag.item()
            instruction = instance['instructions'].field_list[idx]
            visual_features = instance['visual_features'].field_list[idx].tensor
            visual_attention_mask = instance['visual_attention_mask'].field_list[idx].tensor
            curr_visual_labels = instance['visual_labels'].tensor[idx]

            if flag == 1:
                if self.mask_token_prob > 0:
                    tokens, masked_token_labels = self.random_word(instruction.tokens)
                    instruction.tokens = tokens
                    token_labels.append(
                        TensorField(torch.tensor(masked_token_labels, dtype=torch.long), padding_value=-100))

                if self.mask_object_prob > 0:
                    masked_visual_labels = self.random_region(visual_features, visual_attention_mask,
                                                              curr_visual_labels)
                    visual_labels.append(
                        TensorField(torch.tensor(masked_visual_labels, dtype=torch.long), padding_value=-100))
            else:
                if self.mask_token_prob > 0:
                    token_labels.append(
                        TensorField(
                            torch.ones((instruction.sequence_length(),), dtype=torch.long) * -100,
                            padding_value=-100
                        ),
                    )
                if self.mask_object_prob > 0:
                    visual_labels.append(
                        TensorField(torch.ones((curr_visual_labels.shape[0],), dtype=torch.long) * -100,
                                    padding_value=-100)
                    )

        if self.mask_token_prob > 0:
            instance.fields['token_labels'] = ListField(token_labels)

        if self.mask_object_prob > 0:
            instance.fields['visual_labels'] = ListField(visual_labels)

    def text_to_instance(self, split_id, ex) -> Instance:
        r_idx = ex["repeat_idx"]

        if not hasattr(self, "lmdb_env"):
            instance_cache_path = os.path.join(self.data_root_path, self.instance_cache_dir)

            self.lmdb_env = lmdb.Environment(
                instance_cache_path,
                readonly=not self.write_cache,
                subdir=True,
                map_size=1024 * 1024 * 1024 * 1024,
                lock=self.write_cache
            )

        root = self.get_task_root(ex)
        instance_cache_key = os.path.join(root, f"instance_{r_idx}").encode()

        # we do a cache lookup only if we set `write_cache`
        if not self.write_cache:
            with self.lmdb_env.begin(write=False, buffers=True) as txt:
                buffer = txt.get(instance_cache_key)
                # we check if we did a cache-hit, if not we will proceed
                if buffer is None:
                    raise ValueError(f"Missing element from the cache for dataset split {split_id}: {root}. "
                                     f"Enable the flag `write_cache` of the dataset reader to create the cache."
                                     f"You should be using the script `process_dataset.py` for it!")

                instance = pickle.loads(buffer)

                self._mask_instance(instance)

                return instance

        # TODO: check how to compute these values for extra supervision signals
        # if test_mode:
        #     # subgoal completion supervision
        #     if self.use_subgoal_completion:
        #         features['subgoals_completed'] = np.array(traj['num']['low_to_high_idx']) / self.max_subgoals
        #
        #     # progress monitor supervision
        #     if self.use_progress_monitor:
        #         num_actions = len([a for sg in traj['num']['action_low'] for a in sg])
        #         subgoal_progress = [(i + 1) / float(num_actions) for i in range(num_actions)]
        #         features['subgoal_progress'] = subgoal_progress

        #########
        # inputs
        #########

        task_desc = ex['turk_annotations']['anns'][r_idx]['task_desc']

        # step-by-step instructions
        high_descs = ex['turk_annotations']['anns'][r_idx]['high_descs'] + ['stop.']

        metadata = dict(
            split_id=split_id,
            root=root,
            task_desc=task_desc,
            high_descs=high_descs,
            task_type=ex["task_type"],
            task_id=ex["task_id"],
            repeat_idx=ex["repeat_idx"]
        )

        # we simply count the number of instances that we could potentially generate
        if self.dry_run:
            return Instance(dict(metadata=MetadataField(metadata)))

        actions_low, actions_high, low_to_high = self.process_actions(ex, high_descs)

        task_desc_tokens = self._tokenizer.tokenize(task_desc)
        high_descs_tokens = []
        start_instr_labels = []
        language_instructions = []

        for i, hd in enumerate(high_descs):
            high_desc_tokens = self._tokenizer.tokenize(hd)
            turn_tokens = self._tokenizer.add_special_tokens(task_desc_tokens, high_desc_tokens)
            high_descs_tokens.append(turn_tokens)

        # token_labels = []
        for i in range(len(low_to_high)):
            turn_tokens = high_descs_tokens[low_to_high[i]]
            # if we are at the beginning of a new language instruction we mark it as 1, 0 otherwise
            if i == len(low_to_high) - 1:
                start_instr_labels.append(1.0)
            else:
                if low_to_high[i + 1] != low_to_high[i]:
                    start_instr_labels.append(1.0)
                else:
                    start_instr_labels.append(0.0)

            language_instructions.append(TextField(turn_tokens))

        num_low_actions = len(ex['plan']['low_actions']) + 1  # +1 for additional stop action

        assert num_low_actions == len(actions_low), "Number of actions is not correct!"
        # This should be a list of
        object_features = []

        rotation_steps = 360 // self.rotation_angle
        visual_attention_mask = []
        interact_object_masks = []
        obj_interact_targets = []
        nav_receptacle_labels = []
        visual_class_labels = []

        features_root = os.path.join(root, self.vis_feats_path)

        if not os.path.exists(features_root):
            raise ValueError(f"The visual features path {features_root} does not exist!")

        for i in range(num_low_actions):
            curr_step_features = []
            # always include an extra slot for the ResNet feature associated with the front view
            curr_visual_attention_mask = []
            curr_objects_masks = []
            curr_visual_class_labels = []

            for j in range(rotation_steps):
                feature_path = os.path.join(features_root, f"{i}-{j}.npz")
                if os.path.exists(feature_path):
                    with np.load(feature_path) as f_features:
                        features, attn_mask, masks, class_labels = self.extract_visual_features(f_features, j)

                    check_nan(features,
                              f"MaskRCNN features -- ID: {metadata['task_id']} -- "
                              f"Type: {metadata['task_type']} -- Rotation step: {j}")

                    curr_step_features.append(features)
                    curr_objects_masks.append(masks)
                    curr_visual_class_labels.extend(class_labels)
                    curr_visual_attention_mask.extend(attn_mask)
                else:
                    num_objects_in_frame = self.num_objects_per_view[j]
                    # handles cases where we don't have information for the current trajectory step (--rare--)
                    curr_step_features.append(np.zeros((num_objects_in_frame, VISUAL_EMB_SIZE), dtype=np.float32))
                    curr_objects_masks.append(
                        np.zeros((num_objects_in_frame, self.frame_size, self.frame_size), dtype=np.bool))
                    curr_visual_attention_mask.extend([0] * num_objects_in_frame)
                    curr_visual_class_labels.extend([-100] * num_objects_in_frame)

            curr_step_features = np.concatenate(curr_step_features, 0)
            curr_visual_attention_mask = np.array(curr_visual_attention_mask, dtype=np.uint8)
            # curr_visual_class_labels = self.random_region(curr_step_features, curr_visual_attention_mask,
            #                                              curr_visual_class_labels)
            curr_step_features = ArrayField(curr_step_features)
            curr_visual_attention_mask = ArrayField(curr_visual_attention_mask)
            object_features.append(curr_step_features)
            visual_attention_mask.append(curr_visual_attention_mask)
            interact_object_masks.append(curr_objects_masks)
            if start_instr_labels[i] == 1.0:
                visual_class_labels.append(curr_visual_class_labels)
            else:
                visual_class_labels.append([-100] * curr_visual_attention_mask.array.shape[0])

            # -- derive gold targets for mask interaction
            # -- derive fine-grained targets for each step
            subgoal_data = actions_low[i]["subgoal"]
            object_mask = subgoal_data.get("object", {}).get("mask")

            if actions_low[i]["mask"] is not None:
                # we extract the gold mask for the current manipulation step
                gold_mask = actions_low[i]["mask"].astype(np.bool)
                front_view_masks = curr_objects_masks[0]
                iou_scores = [mask_iou(gold_mask, mask) for mask in front_view_masks]
            elif object_mask is not None and self.use_obj_nav:
                # in this case we have a navigation step, we extract the object of interest
                iou_scores = [mask_iou(object_mask, mask) for mask in [m for ms in curr_objects_masks for m in ms]]
            else:
                # if object of interest is not available, we just mask that step
                iou_scores = None

            if iou_scores is not None:
                label = np.argmax(iou_scores)
            else:
                # we don't need a target for this step
                label = -100

            obj_interact_targets.append(label)

            receptacle_mask = subgoal_data.get("receptacle", {}).get("mask")

            if receptacle_mask is not None:
                # we compute the IoU between the gold mask and the target
                iou_scores = [mask_iou(receptacle_mask, mask) for mask in [m for ms in curr_objects_masks for m in ms]]
                label = np.argmax(iou_scores)
                nav_receptacle_labels.append(label)
            else:
                nav_receptacle_labels.append(-100)
            # --------------------------------------------

        scene_objects_features = ListField(object_features)
        visual_attention_mask = ListField(visual_attention_mask)
        visual_class_labels = TensorField(torch.tensor(visual_class_labels, dtype=torch.long), padding_value=-100)
        episode_mask = ArrayField(np.ones((num_low_actions,), dtype=np.uint8))

        actions_low_field = ListField([LabelField(a['action'], "low_action_labels") for a in actions_low])

        obj_interact_targets = torch.tensor(obj_interact_targets, dtype=torch.int64)
        obj_interact_mask = (obj_interact_targets != -100)
        obj_interact_targets = TensorField(obj_interact_targets, padding_value=-100)
        obj_interact_mask = TensorField(obj_interact_mask, dtype=torch.bool)
        nav_receptacle_labels = TensorField(np.array(nav_receptacle_labels), padding_value=-100)

        start_instr_labels = TensorField(torch.tensor(start_instr_labels, dtype=torch.int32), padding_value=0)
        language_instructions = ListField(language_instructions)
        metadata_field = MetadataField(metadata)

        instance = Instance(dict(
            metadata=metadata_field,
            instructions=language_instructions,
            start_instr_labels=start_instr_labels,
            actions=actions_low_field,
            visual_features=scene_objects_features,
            visual_attention_mask=visual_attention_mask,
            actions_mask=episode_mask,
            obj_interact_targets=obj_interact_targets,
            obj_interact_mask=obj_interact_mask,
            nav_receptacle_labels=nav_receptacle_labels,
            visual_labels=visual_class_labels
        ))

        with self.lmdb_env.begin(write=True, buffers=True) as txt:
            txt.put(instance_cache_key, pickle.dumps(instance))

        self._mask_instance(instance)

        return instance


@DatasetReader.register("alfred_inference")
class AlfredInferenceReader(DatasetReader):
    def _read(self, file_path) -> Iterable[Instance]:
        pass

    def __init__(self,
                 data_root_path: str,
                 visual_feature_repr: List[str] = ("box_features", "roi_angles", "boxes"),
                 frame_size: int = 300,  # it assumes a square frame size (300x300), default in AI2Thor
                 rotation_angle: float = 90,
                 num_objects_per_view=NUM_OBJECTS_PER_VIEW,
                 tokenizer: Optional[Tokenizer] = None,
                 token_indexers: Optional[Dict[str, TokenIndexer]] = None,
                 manual_distributed_sharding=True,
                 manual_multiprocess_sharding=True):
        super().__init__(manual_multiprocess_sharding=manual_multiprocess_sharding,
                         manual_distributed_sharding=manual_distributed_sharding)

        self.data_root_path = data_root_path
        self.visual_feature_repr = visual_feature_repr
        # tokenizers and indexers
        if tokenizer is None:
            tokenizer = PretrainedTransformerTokenizer("bert-base-uncased", add_special_tokens=False)
        self._tokenizer = tokenizer
        if token_indexers is None:
            token_indexers = {"tokens": PretrainedTransformerIndexer("bert-base-uncased")}

        self._token_indexers = token_indexers
        self.frame_size = frame_size
        self.rotation_angle = rotation_angle
        self.num_objects_per_view = num_objects_per_view

    def load_task_json(self, split_id, task):
        json_path = os.path.join(self.data_root_path, split_id, task['task'], 'ref_traj_data.json')
        if not os.path.exists(json_path):
            json_path = os.path.join(self.data_root_path, split_id, task['task'], 'traj_data.json')

        with open(json_path) as f:
            ex = json.load(f)
        # root & split
        ex['root'] = os.path.join(self.data_root_path, task['task'])
        ex['split'] = split_id
        ex['repeat_idx'] = task["repeat_idx"]

        return ex

    def get_task_root(self, ex):
        '''
        returns the folder path of a trajectory
        '''
        return os.path.join(self.data_root_path, ex['split'], *(ex['root'].split('/')[-2:]))

    def extract_visual_features(self, features_dict, step):
        """
        features_dict will contain:
            box_features=box_features.cpu().numpy(),
            roi_angles=boxes_angles,
            boxes=boxes,
            masks=(masks > 0.5).cpu().numpy(),
            class_probs=class_probs.cpu().numpy(),
            class_labels=class_labels.cpu().numpy(),
            num_objects=box_features.shape[0],
            pano_id=pano_id

        self.visual_feature_repr is a tuple indicating the key that we want to concatenate
        """

        max_objects_in_frame = self.num_objects_per_view[step]

        # for this option we simply concatenate all the features
        visual_features = []

        for k in self.visual_feature_repr:
            feat = features_dict[k]

            # empty feature vector
            if feat.shape[0] == 0:
                if k == "roi_angles":
                    remaining_shape = [ROI_ANGLES_DIM]
                elif k == "boxes":
                    remaining_shape = [ROI_REL_AREA_DIM + ROI_COORDINATES_DIM]
                else:
                    remaining_shape = feat.shape[1:]
                feat = np.zeros((max_objects_in_frame, *remaining_shape))
            else:
                if k == "roi_angles":
                    # we extend the roi angles features here
                    feat = get_angle_representation(feat)
                elif k == "boxes":
                    vis_pe = torch.from_numpy(feat)
                    w_est = torch.max(vis_pe[:, [0, 2]]) * 1. + 1e-5
                    h_est = torch.max(vis_pe[:, [1, 3]]) * 1. + 1e-5
                    vis_pe[:, [0, 2]] /= w_est
                    vis_pe[:, [1, 3]] /= h_est
                    assert h_est > 0, 'should greater than 0! {}'.format(h_est)
                    assert w_est > 0, 'should greater than 0! {}'.format(w_est)
                    rel_area = (vis_pe[:, 3] - vis_pe[:, 1]) * (vis_pe[:, 2] - vis_pe[:, 0])
                    rel_area.clamp_(0)
                    vis_pe = torch.cat((vis_pe, rel_area.view(-1, 1)), -1)
                    vis_pe = torch.nn.functional.normalize(vis_pe - 0.5, dim=-1)
                    feat = vis_pe.numpy()

                if len(features_dict[k].shape) == 1:
                    feat = np.expand_dims(feat, -1)

            visual_features.append(feat)

        visual_features = np.concatenate(visual_features, 1).astype(np.float32)

        padding_elements = max_objects_in_frame - visual_features.shape[0]

        num_objects = int(features_dict["num_objects"])

        if num_objects == 0:
            num_objects = max_objects_in_frame
            object_masks = np.zeros((max_objects_in_frame, self.frame_size, self.frame_size), dtype=np.bool)
            object_classes = np.ones((max_objects_in_frame,), dtype=np.long) * -100
        else:
            object_masks = features_dict["masks"].squeeze(1).astype(np.bool)
            object_classes = features_dict["class_labels"]

        if padding_elements > 0:
            visual_features = np.concatenate(
                [visual_features, np.zeros((padding_elements, VISUAL_EMB_SIZE), dtype=np.float32)])
        visual_mask = [1] * num_objects + [0] * padding_elements

        return visual_features, visual_mask, object_masks, object_classes

    def text_to_instance(self, ex, object_detections, instruction_idx=None) -> Instance:
        r_idx = ex["repeat_idx"]
        split_id = ex["split"]
        task_desc = ex['turk_annotations']['anns'][r_idx]['task_desc']

        # step-by-step instructions
        high_descs = ex['turk_annotations']['anns'][r_idx]['high_descs'] + ['stop.']

        root = self.get_task_root(ex)

        metadata = dict(
            split_id=split_id,
            root=root,
            task_desc=task_desc,
            high_descs=high_descs,
            task_type=ex.get("task_type", ""),
            task_id=ex["task_id"],
            repeat_idx=ex["repeat_idx"]
        )

        if instruction_idx is None:
            task_desc_tokens = self._tokenizer.tokenize(task_desc)
            high_desc_tokens = [token for hd in high_descs for token in self._tokenizer.tokenize(hd)]

            lang_tokens = self._tokenizer.add_special_tokens(task_desc_tokens, high_desc_tokens)
            language_instruction_field = TextField(lang_tokens)
        else:
            task_desc_tokens = self._tokenizer.tokenize(task_desc)

            # depending on the instruction index we encode
            high_desc_tokens = self._tokenizer.tokenize(high_descs[instruction_idx])
            turn_tokens = self._tokenizer.add_special_tokens(task_desc_tokens, high_desc_tokens)
            language_instruction_field = TextField(turn_tokens)

        object_features = []
        visual_attention_mask = []
        interact_object_masks = []

        for step, curr_detections in enumerate(object_detections):
            max_objects_in_frame = self.num_objects_per_view[step]
            curr_obj_features, curr_mask, masks, _ = self.extract_visual_features(curr_detections, step)

            if curr_obj_features.shape[0] > 0:
                object_features.append(curr_obj_features)
                visual_attention_mask.extend(curr_mask)
                interact_object_masks.append(masks)
            else:
                object_features.append(np.zeros((max_objects_in_frame, VISUAL_EMB_SIZE), dtype=np.float32))
                visual_attention_mask.extend([0] * self.max_objects_per_frame)
                interact_object_masks.append(
                    np.zeros((max_objects_in_frame, self.frame_size, self.frame_size), dtype=np.uint8))

        object_features = ArrayField(np.concatenate(object_features, 0))
        visual_attention_mask = ArrayField(np.array(visual_attention_mask, dtype=np.float32))

        #########
        # outputs
        #########

        metadata["interactive_object_masks"] = interact_object_masks

        metadata_field = MetadataField(metadata)

        return Instance(dict(
            metadata=metadata_field,
            instructions=language_instruction_field,
            visual_features=object_features,
            visual_attention_mask=visual_attention_mask
        ))

    def apply_token_indexers(self, instance: Instance) -> None:
        if "instructions" in instance:
            instance["instructions"].token_indexers = self._token_indexers


@DatasetReader.register("embert_subgoal")
class AlfredInferenceSubgoalReader(EmbertSupervisedReader):
    def __init__(self, data_root_path: str, vis_feats_path: str, splits_path: str,
                 visual_feature_repr: List[str] = ("box_features", "roi_angles", "boxes"), frame_size: int = 300,
                 use_progress_monitor: bool = False, use_subgoal_completion: bool = False, max_sub_goals: int = 25,
                 mask_token_prob: float = -0.0, mask_object_prob: float = -0.0, rotation_angle: float = 90,
                 num_objects_per_view=NUM_OBJECTS_PER_VIEW, tokenizer: Optional[Tokenizer] = None,
                 token_indexers: Optional[Dict[str, TokenIndexer]] = None, dry_run=False, write_cache=False,
                 max_traj_length: int = 200, instance_cache_dir="subgoal_instance_cache"):
        super().__init__(data_root_path, vis_feats_path, splits_path, visual_feature_repr, frame_size,
                         use_progress_monitor, use_subgoal_completion, max_sub_goals, mask_token_prob, mask_object_prob,
                         rotation_angle, num_objects_per_view, tokenizer, token_indexers, dry_run, write_cache,
                         max_traj_length, instance_cache_dir)

    def _read(self, split_id: str) -> Iterable[Instance]:
        with open(self.splits_path) as in_file:
            split_data = json.load(in_file)[split_id]

        for task in split_data:
            json_path = os.path.join(self.data_root_path, split_id, task['task'], 'ref_traj_data.json')
            if not os.path.exists(json_path):
                json_path = os.path.join(self.data_root_path, split_id, task['task'], 'traj_data.json')
            with open(json_path) as f:
                ex = json.load(f)
            # root & split
            ex['root'] = os.path.join(self.data_root_path, task['task'])
            ex['split'] = split_id
            ex['repeat_idx'] = task["repeat_idx"]

            yield ex  # self.text_to_instance(split_id, ex)

    def load_task_json(self, split_id, task):
        json_path = os.path.join(self.data_root_path, split_id, task['task'], 'ref_traj_data.json')
        if not os.path.exists(json_path):
            json_path = os.path.join(self.data_root_path, split_id, task['task'], 'traj_data.json')

        with open(json_path) as f:
            ex = json.load(f)
        # root & split
        ex['root'] = os.path.join(self.data_root_path, task['task'])
        ex['split'] = split_id
        ex['repeat_idx'] = task["repeat_idx"]

        return ex

    def get_task_root(self, ex):
        '''
        returns the folder path of a trajectory
        '''
        return os.path.join(self.data_root_path, ex['split'], *(ex['root'].split('/')[-2:]))

    def _inference_text_to_instance(self, ex, object_detections, instruction_idx=None) -> Instance:
        r_idx = ex["repeat_idx"]
        split_id = ex["split"]
        task_desc = ex['turk_annotations']['anns'][r_idx]['task_desc']

        # step-by-step instructions
        high_descs = ex['turk_annotations']['anns'][r_idx]['high_descs'] + ['stop.']

        root = self.get_task_root(ex)

        metadata = dict(
            split_id=split_id,
            root=root,
            task_desc=task_desc,
            high_descs=high_descs,
            task_type=ex["task_type"],
            task_id=ex["task_id"],
            repeat_idx=ex["repeat_idx"]
        )

        if instruction_idx is None:
            task_desc_tokens = self._tokenizer.tokenize(task_desc)
            high_desc_tokens = [token for hd in high_descs for token in self._tokenizer.tokenize(hd)]

            lang_tokens = self._tokenizer.add_special_tokens(task_desc_tokens, high_desc_tokens)
            language_instruction_field = TextField(lang_tokens)
        else:
            task_desc_tokens = self._tokenizer.tokenize(task_desc)

            # depending on the instruction index we encode
            high_desc_tokens = self._tokenizer.tokenize(high_descs[instruction_idx])
            turn_tokens = self._tokenizer.add_special_tokens(task_desc_tokens, high_desc_tokens)
            language_instruction_field = TextField(turn_tokens)

        object_features = []
        visual_attention_mask = []
        interact_object_masks = []

        for step, curr_detections in enumerate(object_detections):
            max_objects_in_frame = self.num_objects_per_view[step]
            curr_obj_features, curr_mask, masks, _ = self.extract_visual_features(curr_detections, step)

            if curr_obj_features.shape[0] > 0:
                object_features.append(curr_obj_features)
                visual_attention_mask.extend(curr_mask)
                interact_object_masks.append(masks)
            else:
                object_features.append(np.zeros((max_objects_in_frame, VISUAL_EMB_SIZE), dtype=np.float32))
                visual_attention_mask.extend([0] * self.max_objects_per_frame)
                interact_object_masks.append(
                    np.zeros((max_objects_in_frame, self.frame_size, self.frame_size), dtype=np.uint8))

        object_features = ArrayField(np.concatenate(object_features, 0))
        visual_attention_mask = ArrayField(np.array(visual_attention_mask, dtype=np.float32))

        #########
        # outputs
        #########

        metadata["interactive_object_masks"] = interact_object_masks

        metadata_field = MetadataField(metadata)

        return Instance(dict(
            metadata=metadata_field,
            instructions=language_instruction_field,
            visual_features=object_features,
            visual_attention_mask=visual_attention_mask
        ))

    def text_to_instance(self, ex, object_features=None, instruction_idx=None) -> Instance:
        if object_features is not None:
            return self._inference_text_to_instance(ex, object_features, instruction_idx)

        r_idx = ex["repeat_idx"]
        split_id = ex["split"]

        if not hasattr(self, "lmdb_env"):
            instance_cache_path = os.path.join(self.data_root_path, self.instance_cache_dir)

            self.lmdb_env = lmdb.Environment(
                instance_cache_path,
                readonly=not self.write_cache,
                subdir=True,
                map_size=1024 * 1024 * 1024 * 1024,
                lock=self.write_cache
            )

        root = self.get_task_root(ex)
        instance_cache_key = os.path.join(root, f"instance_{r_idx}").encode()

        # we do a cache lookup only if we set `write_cache`
        if not self.write_cache:
            with self.lmdb_env.begin(write=False, buffers=True) as txt:
                buffer = txt.get(instance_cache_key)
                # we check if we did a cache-hit, if not we will proceed
                if buffer is None:
                    raise ValueError(f"Missing element from the cache for dataset split {split_id}: {root}. "
                                     f"Enable the flag `write_cache` of the dataset reader to create the cache."
                                     f"You should be using the script `process_dataset.py` for it!")

                instance = pickle.loads(buffer)
                return instance

        #########
        # inputs
        #########

        task_desc = ex['turk_annotations']['anns'][r_idx]['task_desc']

        # step-by-step instructions
        high_descs = ex['turk_annotations']['anns'][r_idx]['high_descs'] + ['stop.']

        metadata = dict(
            split_id=split_id,
            root=root,
            task_desc=task_desc,
            high_descs=high_descs,
            task_type=ex["task_type"],
            task_id=ex["task_id"],
            repeat_idx=ex["repeat_idx"]
        )

        # we simply count the number of instances that we could potentially generate
        if self.dry_run:
            return Instance(dict(metadata=MetadataField(metadata)))

        actions_low, actions_high, low_to_high = self.process_actions(ex, high_descs)

        task_desc_tokens = self._tokenizer.tokenize(task_desc)
        high_descs_tokens = []
        language_instructions = []

        for i, hd in enumerate(high_descs):
            high_desc_tokens = self._tokenizer.tokenize(hd)
            turn_tokens = self._tokenizer.add_special_tokens(task_desc_tokens, high_desc_tokens)
            high_descs_tokens.append(turn_tokens)

        for i in range(len(low_to_high)):
            turn_tokens = high_descs_tokens[low_to_high[i]]
            language_instructions.append(TextField(turn_tokens))

        num_low_actions = len(ex['plan']['low_actions']) + 1  # +1 for additional stop action

        assert num_low_actions == len(actions_low), "Number of actions is not correct!"
        # This should be a list of
        object_features = []

        rotation_steps = 360 // self.rotation_angle
        visual_attention_mask = []
        interact_object_masks = []

        features_root = os.path.join(root, self.vis_feats_path)

        if not os.path.exists(features_root):
            raise ValueError(f"The visual features path {features_root} does not exist!")

        for i in range(num_low_actions):
            curr_step_features = []
            # always include an extra slot for the ResNet feature associated with the front view
            curr_visual_attention_mask = []
            curr_objects_masks = []

            for j in range(rotation_steps):
                feature_path = os.path.join(features_root, f"{i}-{j}.npz")
                if os.path.exists(feature_path):
                    with np.load(feature_path) as f_features:
                        features, attn_mask, masks, class_labels = self.extract_visual_features(f_features, j)

                    check_nan(features,
                              f"MaskRCNN features -- ID: {metadata['task_id']} -- "
                              f"Type: {metadata['task_type']} -- Rotation step: {j}")

                    curr_step_features.append(features)
                    curr_objects_masks.append(masks)
                    curr_visual_attention_mask.extend(attn_mask)
                else:
                    num_objects_in_frame = self.num_objects_per_view[j]
                    # handles cases where we don't have information for the current trajectory step (--rare--)
                    curr_step_features.append(np.zeros((num_objects_in_frame, VISUAL_EMB_SIZE), dtype=np.float32))
                    curr_objects_masks.append(
                        np.zeros((num_objects_in_frame, self.frame_size, self.frame_size), dtype=np.uint8))
                    curr_visual_attention_mask.extend([0] * num_objects_in_frame)

            curr_step_features = np.concatenate(curr_step_features, 0)
            curr_visual_attention_mask = np.array(curr_visual_attention_mask, dtype=np.uint8)
            curr_step_features = ArrayField(curr_step_features)
            curr_visual_attention_mask = ArrayField(curr_visual_attention_mask)
            object_features.append(curr_step_features)
            visual_attention_mask.append(curr_visual_attention_mask)
            interact_object_masks.append(curr_objects_masks)

        scene_objects_features = ListField(object_features)
        visual_attention_mask = ListField(visual_attention_mask)

        actions_low_field = ListField([LabelField(a['action'], "low_action_labels") for a in actions_low])

        language_instructions = ListField(language_instructions)
        metadata_field = MetadataField(metadata)

        instance = Instance(dict(
            metadata=metadata_field,
            instructions=language_instructions,
            actions=actions_low_field,
            visual_features=scene_objects_features,
            visual_attention_mask=visual_attention_mask
        ))

        with self.lmdb_env.begin(write=True, buffers=True) as txt:
            txt.put(instance_cache_key, pickle.dumps(instance))

        return instance
