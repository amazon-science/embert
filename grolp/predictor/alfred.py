import numpy as np
import torch
from allennlp.common import JsonDict
from allennlp.data import Instance, DatasetReader
from allennlp.predictors import Predictor

from grolp import EmbodiedBertForAlfred
from grolp.readers.alfred import AlfredInferenceReader, AlfredInferenceSubgoalReader


@Predictor.register("alfred")
class AlfredPredictor(Predictor):
    def __init__(self, model: EmbodiedBertForAlfred, dataset_reader: DatasetReader, frozen: bool = True) -> None:
        super().__init__(model, dataset_reader, frozen)

    @property
    def is_split_model(self):
        # checks whether the model has a component that allows the prediction of when the instruction starts
        return self._model.start_instr_predictor is not None

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        pass

    def predict_instance(self, instance: Instance, **kwargs) -> JsonDict:
        self._dataset_reader.apply_token_indexers(instance)
        outputs = self._model.step([instance], **kwargs)
        return outputs

    def featurize(self, traj_data, object_features=None, instruction_idx=None):
        if object_features is not None:
            instance = self._dataset_reader.text_to_instance(traj_data, object_features, instruction_idx)
        else:
            instance = self._dataset_reader.text_to_instance(traj_data)

        return instance

    def load_task_json(self, split_id, task):
        return self._dataset_reader.load_task_json(split_id, task)

    def share_memory(self):
        self._model.share_memory()

    def init_masks(self, num_objects_in_front):
        num_actions = self._model.vocab.get_vocab_size("low_action_labels")
        rotate_right_action = self._model.vocab.get_token_index("RotateRight_90", "low_action_labels")
        rotate_left_action = self._model.vocab.get_token_index("RotateLeft_90", "low_action_labels")
        move_action = self._model.vocab.get_token_index("MoveAhead_25", "low_action_labels")

        action_mask = np.zeros((num_actions,))
        action_mask[rotate_left_action] = 1.0
        action_mask[rotate_right_action] = 1.0
        action_mask[move_action] = 1.0

        object_mask = np.ones((num_objects_in_front,))

        return action_mask, object_mask

    @classmethod
    def load_checkpoint(
            cls,
            args
    ):
        checkpoint_path = args.model_path
        data_root_path = args.data
        cuda_device = args.cuda_device
        splits_path = args.splits
        is_subgoal_eval = hasattr(args, 'subgoals') and args.subgoals is not None

        model, train_dataset_reader = EmbodiedBertForAlfred.load_from_checkpoint(
            checkpoint_path,
            map_location=torch.device(cuda_device) if cuda_device != -1 else torch.device("cpu")
        )

        if is_subgoal_eval:
            dataset_reader = AlfredInferenceSubgoalReader(
                data_root_path=data_root_path,
                splits_path=splits_path,
                vis_feats_path=train_dataset_reader.vis_feats_path,
                visual_feature_repr=train_dataset_reader.visual_feature_repr,
                num_objects_per_view=train_dataset_reader.num_objects_per_view
            )
        else:
            dataset_reader = AlfredInferenceReader(
                data_root_path,
                num_objects_per_view=train_dataset_reader.num_objects_per_view
            )

        return AlfredPredictor(model, dataset_reader)
