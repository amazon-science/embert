import copy
import inspect
import os
import shutil
from typing import Optional, Tuple
from typing import Type, T, Callable, Union, Dict, List, Any

import allennlp
import numpy
import pandas as pd
import torch
import torch.nn.functional as F
from allennlp.common import Params
from allennlp.common.file_utils import cached_path
from allennlp.common.params import remove_keys_from_params
from allennlp.data import Vocabulary, Batch, Instance
from allennlp.models import Model
from allennlp.models.archival import extracted_archive, _load_dataset_readers
from allennlp.modules import FeedForward, Attention
from allennlp.nn import util
from allennlp.nn.util import masked_mean, masked_softmax
from pytorch_lightning import LightningModule
from pytorch_lightning.core.saving import CHECKPOINT_PAST_HPARAMS_KEYS, _convert_loaded_hparams, \
    load_hparams_from_tags_csv, load_hparams_from_yaml
from pytorch_lightning.utilities.cloud_io import load as pl_load
from pytorch_lightning.utilities.parsing import parse_class_init_keys
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torchmetrics import ConfusionMatrix, F1
from typing.io import IO

from grolp.models.config import AlfredConfig, NUM_OBJECTS_PER_VIEW
from grolp.models.embodied_bert import EmbodiedBert
from grolp.models.transformer import StateEncoderDecoder
from grolp.utils.archival import logger
from grolp.utils.metrics import Accuracy, TrajectoryPrecision, AverageMetric


def is_manipulation_action(action):
    '''
    check if low-level action is interactive
    '''
    non_interact_actions = ['MoveAhead', 'Rotate', 'Look', '<<stop>>', '<<pad>>', '<<seg>>']
    if any(a in action for a in non_interact_actions):
        return False
    else:
        return True


def get_final_encoder_states(
        encoder_outputs: torch.Tensor, mask: torch.BoolTensor, bidirectional: bool = False
) -> torch.Tensor:
    """
    Given the output from a `Seq2SeqEncoder`, with shape `(batch_size, sequence_length,
    encoding_dim)`, this method returns the final hidden state for each element of the batch,
    giving a tensor of shape `(batch_size, encoding_dim)`.  This is not as simple as
    `encoder_outputs[:, -1]`, because the sequences could have different lengths.  We use the
    mask (which has shape `(batch_size, sequence_length)`) to find the final state for each batch
    instance.

    Additionally, if `bidirectional` is `True`, we will split the final dimension of the
    `encoder_outputs` into two and assume that the first half is for the forward direction of the
    encoder and the second half is for the backward direction.  We will concatenate the last state
    for each encoder dimension, giving `encoder_outputs[:, -1, :encoding_dim/2]` concatenated with
    `encoder_outputs[:, 0, encoding_dim/2:]`.
    """
    # These are the indices of the last words in the sequences (i.e. length sans padding - 1).  We
    # are assuming sequences are right padded.
    # Shape: (batch_size,)
    last_word_indices = mask.sum(1) - 1
    # This prevents an error when we have an empty sequence. We assume that the final state for an empty sequence
    # is the hidden state at timestep 0. It will be masked anyway so we don't really care...
    last_word_indices[last_word_indices == -1] = 0
    batch_size, _, encoder_output_dim = encoder_outputs.size()
    expanded_indices = last_word_indices.view(-1, 1, 1).expand(batch_size, 1, encoder_output_dim)
    # Shape: (batch_size, 1, encoder_output_dim)
    final_encoder_output = encoder_outputs.gather(1, expanded_indices)
    final_encoder_output = final_encoder_output.squeeze(1)  # (batch_size, encoder_output_dim)
    if bidirectional:
        final_forward_output = final_encoder_output[:, : (encoder_output_dim // 2)]
        final_backward_output = encoder_outputs[:, 0, (encoder_output_dim // 2):]
        final_encoder_output = torch.cat([final_forward_output, final_backward_output], dim=-1)
    return final_encoder_output


def sequence_cross_entropy(logits, targets, mask, padding_idx=-100):
    """
    Computes the sequence-based cross-entropy loss

    :param logits: computed logits - Tensor with shape: (batch_size, seq_length, num_classes)
    :param targets: reference targets - Tensor with shape: (batch_size, seq_length)
    :param mask: mask to ignore masked values - Tensor with shape: (batch_size, seq_length)
    :param padding_idx: padding value (default: -100 which is the one used by PyTorch)
    :return:
    """
    batch_size, seq_size, num_classes = logits.shape

    losses = F.cross_entropy(
        logits.view(batch_size * seq_size, num_classes),
        targets.view(batch_size * seq_size),
        ignore_index=padding_idx,
        reduction="none"
    )

    losses = losses.view(batch_size, seq_size)
    losses = masked_mean(losses, mask.bool(), -1).mean()

    return losses


def sequence_binary_cross_entropy(logits, targets, mask):
    """
    Similar to the loss above which uses binary cross-entropy instead.

    :param logits: computed logits - Tensor with shape: (batch_size, seq_length, num_classes)
    :param targets: reference targets - Tensor with shape: (batch_size, seq_length)
    :param mask: mask to ignore masked values - Tensor with shape: (batch_size, seq_length)
    :return:
    """
    batch_size, seq_size, _ = logits.shape

    losses = F.binary_cross_entropy_with_logits(
        logits.view(batch_size * seq_size),
        targets.view(batch_size * seq_size).float(),
        reduction="none"
    )

    losses = losses.view(batch_size, seq_size)
    losses = masked_mean(losses, mask.bool(), -1).mean()

    return losses


@Model.register("embert_alfred")
class EmbodiedBertForAlfred(Model, LightningModule):
    def __init__(self,
                 vocab: Vocabulary,
                 pretrained_model_path: str,
                 actor: FeedForward,
                 start_instr_predictor: FeedForward,
                 object_scorer: Union[FeedForward, Attention],
                 state_encoder: StateEncoderDecoder,
                 action_loss_weight: float = 1.0,
                 object_interact_weight: float = 1.0,
                 use_lm_loss: bool = False,
                 use_vm_loss: bool = False,
                 use_itm_loss: bool = False,
                 use_pm_loss: bool = False,
                 use_start_instr_loss: bool = True,
                 state_repr_method: str = "dot_product",
                 action_embedding_size: int = 768,
                 compute_confusion_matrix: bool = False,
                 ignore_keys=("instructions", "metadata"),
                 no_language: bool = False,
                 no_vision: bool = False,
                 use_nav_receptacle_loss: bool = False,
                 **kwargs
                 ):
        """
        Creates a new EmBERT instance
        :param vocab: Vocabulary object automatically created by AllenNLP
        :param pretrained_model_path: path to a pretrained checkpoint (you can use OSCAR or BERT)
        :param actor: FeedForward network used to predict the action given the decoder hidden state
        :param start_instr_predictor: FeedForward network used to predict whether to advance to the next instruction or not
        :param object_scorer: FeedForward used to predict a logit for each object using the time-dependent representation
        :param state_encoder: TransformerXL-based state encoder
        :param action_loss_weight: weight associated to the action prediction loss (default: 1.0)
        :param object_interact_weight: weight associated to the object prediction loss (default: 1.0)
        :param use_lm_loss: whether or not to use the masked language labels (default: false)
        :param use_vm_loss: whether or not to use the visual region prediction (default: false)
        :param use_itm_loss: whether or not to use image-text matching (default: false)
        :param use_pm_loss: whether or not to use progress monitor (default: false, not implemented)
        :param use_start_instr_loss: whether or not to use the next step instruction loss (default: true)
        :param state_repr_method: how to generate state representation from the low-level encoder (default: 'dot_product')
        :param action_embedding_size: size of the action embeddings
        :param compute_confusion_matrix: whether or not to compute the action confusion matrix
        :param ignore_keys: when segmenting the trajectory over the time dimension, ignore these fields
        :param no_language: zero out language features
        :param no_vision: zero out visual features
        :param use_nav_receptacle_loss: use receptacle loss prediction
        :param kwargs:
        """
        super().__init__(vocab)
        self.optimizer = None
        self.lr_scheduler = None
        num_objects_per_view = kwargs.get("num_objects_per_view", NUM_OBJECTS_PER_VIEW)
        self.config: AlfredConfig = AlfredConfig(
            action_loss_weight=action_loss_weight,
            object_interact_weight=object_interact_weight,
            use_lm_loss=use_lm_loss,
            use_vm_loss=use_vm_loss,
            use_itm_loss=use_itm_loss,
            use_pm_loss=use_pm_loss,
            use_nav_receptacle_loss=use_nav_receptacle_loss,
            use_start_instr_loss=use_start_instr_loss,
            state_repr_method=state_repr_method,
            action_embedding_size=action_embedding_size,
            num_objects_per_view=num_objects_per_view
        )
        # These attributes are created only at training time by the Trainer
        self.condition_hidden = torch.nn.Linear(self.config.hidden_size, self.config.hidden_size)

        if use_nav_receptacle_loss:
            self.nav_receptacle_scorer = copy.deepcopy(start_instr_predictor)
        else:
            self.nav_receptacle_scorer = None

        self.no_language = no_language
        self.no_vision = no_vision
        self.ignore_keys = ignore_keys
        self.action_padding_index = self.config.num_actions + 1
        self.action_start_index = self.config.num_actions

        self.save_hyperparameters(self.config.to_dict())
        self.num_actions = self.config.num_labels
        self.embert = EmbodiedBert.from_pretrained(pretrained_model_name_or_path=pretrained_model_path,
                                                   config=self.config)
        self.compute_confusion_matrix = compute_confusion_matrix

        self.state_encoder = state_encoder
        # in this way we can determine whether we need to compute the action embeddings
        self.action_conditioning = isinstance(object_scorer, Attention)

        self.obj_scorer = object_scorer
        self.actor = actor

        self.start_instr_predictor = start_instr_predictor

        self.train_actions_accuracy = Accuracy()
        self.train_actions_f1 = F1(num_classes=self.config.num_actions)
        self.train_traj_precision = TrajectoryPrecision()
        self.train_objects_accuracy = Accuracy()
        self.train_overall = AverageMetric(self.train_objects_accuracy, self.train_actions_accuracy)

        self.valid_actions_accuracy = Accuracy()
        self.valid_actions_f1 = F1(num_classes=self.config.num_actions)
        self.valid_traj_precision = TrajectoryPrecision()
        self.valid_objects_accuracy = Accuracy()
        self.valid_overall = AverageMetric(self.valid_objects_accuracy, self.valid_actions_accuracy)

        self.test_actions_accuracy = Accuracy()
        self.test_actions_f1 = F1(num_classes=self.config.num_actions)
        self.test_traj_precision = TrajectoryPrecision()
        self.test_objects_accuracy = Accuracy()
        self.test_overall = AverageMetric(self.test_objects_accuracy, self.test_actions_accuracy)

        if self.compute_confusion_matrix:
            self.train_actions_confmat = ConfusionMatrix(self.config.num_actions)
            self.valid_actions_confmat = ConfusionMatrix(self.config.num_actions)
        self.init_weights()

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module,
                      (torch.nn.Linear, torch.nn.Embedding, allennlp.modules.token_embedders.embedding.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def init_weights(self):
        """
        Initializes and prunes weights if needed.
        """
        # Initialize weights
        self.apply(self._init_weights)

    def configure_optimizers(self):
        """
        Configures the optimizers and learning rate scheduler required by the PyTorch-Lightning infrastructure
        :return:
        """
        optim = {"optimizer": self.optimizer}
        if self.lr_scheduler.is_available:
            optim["lr_scheduler"] = self.lr_scheduler

        return optim

    def tbptt_split_batch(self, batch: Dict[str, Any], split_size: int) -> list:
        """
        Divides the current batch in several segments according to the specified `split_size` parameter along the
        dimension at index 1. This method will automatically avoid splitting the keys specified in `self.ignore_keys`.

        :param batch: batch of data generated by the AllenNLP data loader
        :param split_size: maximum segment size
        :return:
        """
        splittable_keys = [k for k in batch.keys() if k not in self.ignore_keys]
        key = splittable_keys[0]
        total_axis_size = batch[key].shape[1]

        for i in range(0, total_axis_size, split_size):
            split = {}

            valid_indices = batch["actions_mask"][:, i:i + split_size].sum(-1) != 0

            if not valid_indices.all():
                # ignore chunks that contain empty sequences
                return

            for k in batch.keys():
                if k in self.ignore_keys:
                    split[k] = batch[k]
                else:
                    split[k] = batch[k][:, i:i + split_size].contiguous()

            yield split

    def _compute_accuracy(self, batch, results, split_key):
        """
        Computes accuracies and precision scores
        """
        # actions accuracy
        action_predictions = results["action_logits"]
        action_num_classes = results["action_logits"].shape[-1]
        action_predictions = torch.softmax(action_predictions.view(-1, action_num_classes), -1).detach()
        action_targets = batch["actions"].clone().reshape(-1)
        action_mask = batch["actions_mask"].reshape(-1, 1)

        valid_indices = (action_mask.bool()).squeeze(-1)

        if split_key == "train":
            # self.train_actions_accuracy(action_predictions, action_targets, action_mask)
            self.train_traj_precision(results["action_logits"], batch["actions"], batch["actions_mask"])
            self.train_actions_f1(action_predictions[valid_indices], action_targets[valid_indices])
            if self.compute_confusion_matrix:
                try:
                    self.train_actions_confmat(action_predictions[valid_indices], action_targets[valid_indices])
                except:
                    pass
        elif split_key == "val":
            # self.valid_actions_accuracy(action_predictions, action_targets, action_mask)
            self.valid_traj_precision(results["action_logits"], batch["actions"], batch["actions_mask"])
            self.valid_actions_f1(action_predictions[valid_indices], action_targets[valid_indices])
            if self.compute_confusion_matrix:
                try:
                    self.valid_actions_confmat(action_predictions[valid_indices], action_targets[valid_indices])
                except:
                    pass
        elif split_key == "test":
            # self.test_actions_accuracy(action_predictions, action_targets, action_mask)
            self.test_actions_f1(action_predictions[valid_indices], action_targets[valid_indices])
            self.test_traj_precision(results["action_logits"], batch["actions"], batch["actions_mask"])
        else:
            raise ValueError(f"Wrong split key for accuracy: {split_key}")

        # objects predictions accuracy
        obj_predictions = results["object_logits"]
        obj_num_classes = results["object_logits"].shape[-1]
        obj_predictions = torch.softmax(obj_predictions, -1).view(-1, obj_num_classes)
        obj_targets = batch["obj_interact_targets"].clone().reshape(-1)
        obj_mask = batch["obj_interact_mask"].reshape(-1, 1)

        if split_key == "train":
            # self.train_objects_accuracy(predictions, targets, mask)
            self.train_overall(a=[action_predictions, action_targets, action_mask],
                               b=[obj_predictions, obj_targets, obj_mask])
        elif split_key == "val":
            # self.valid_objects_accuracy(predictions, targets, mask)
            self.valid_overall(a=[action_predictions, action_targets, action_mask],
                               b=[obj_predictions, obj_targets, obj_mask])
        elif split_key == "test":
            # self.test_objects_accuracy(predictions, targets, mask)
            self.test_overall(a=[action_predictions, action_targets, action_mask],
                              b=[obj_predictions, obj_targets, obj_mask])

        else:
            raise ValueError(f"Wrong split key for accuracy: {split_key}")

    def training_step(self, batch, batch_idx, hiddens=None):
        """
        Computes a training step given the current segment for the current batch.

        :param batch: features of the current segment
        :param batch_idx: idx of the current batch
        :param hiddens: hidden states generated in the previous segment (None if we are at the first step)
        :return: a dictionary with "loss" and "hiddens" fields
        """
        results = self.forward(
            **batch,
            hiddens=hiddens
        )

        self._compute_accuracy(batch, results, "train")

        self.log("train_traj_precision", self.train_traj_precision, on_epoch=True, tbptt_reduce_fx=torch.sum)
        self.log("train_actions_f1", self.train_actions_f1, on_epoch=True, tbptt_reduce_fx=torch.sum)
        self.log("train_overall", self.train_overall, on_epoch=True, prog_bar=True, on_step=True,
                 tbptt_reduce_fx=torch.sum)
        # self.log("train_actions_accuracy", self.train_actions_accuracy, on_epoch=True, prog_bar=True, tbptt_reduce_fx=torch.sum)
        # self.log("train_objects_accuracy", self.train_objects_accuracy, on_epoch=True, prog_bar=True, tbptt_reduce_fx=torch.sum)

        self.log_dict(
            {loss_key: loss_val for loss_key, loss_val in results.items() if "loss" in loss_key},
            on_epoch=True, tbptt_reduce_fx=torch.sum
        )

        return {"loss": results["loss"], "hiddens": results["hidden_states"]}

    def _inference_step(self, batch, split_key):
        """
        Runs a forward pass at inference and returns the metrics. At inference time we do not apply segment splitting
        because not required. We give the model the entire trajectory as input.
        """
        results = self.forward(
            **batch
        )
        self._compute_accuracy(batch, results, split_key)

        if split_key == "val":
            self.log("val_actions_f1", self.valid_actions_f1, on_epoch=True)
            self.log("val_traj_precision", self.valid_traj_precision, on_epoch=True)
            self.log("val_overall", self.valid_overall, on_epoch=True, prog_bar=True, metric_attribute='valid_overall')
            # self.log("val_actions_accuracy", self.valid_actions_accuracy, on_epoch=True, prog_bar=True)
            # self.log("val_objects_accuracy", self.valid_objects_accuracy, on_epoch=True, prog_bar=True)

        else:
            self.log("test_actions_f1", self.test_actions_f1, on_epoch=True)
            self.log("test_traj_precision", self.test_traj_precision, on_epoch=True)
            self.log("test_overall", self.test_overall, on_epoch=True, prog_bar=True, metric_attribute='test_overall')
            # self.log("test_actions_accuracy", self.test_actions_accuracy, on_epoch=True, prog_bar=True)
            # self.log("test_objects_accuracy", self.test_objects_accuracy, on_epoch=True, prog_bar=True)

        metrics = {f"{split_key}_loss": results["loss"]}
        self.log_dict(metrics, on_epoch=True, sync_dist=True)
        results["metrics"] = metrics

        return results

    def _print_actions_confmat(self, split, matrix):
        labels = [self.vocab.get_token_from_index(j, 'low_action_labels') for j in range(self.config.num_actions)]
        confmat = pd.DataFrame(matrix.cpu().numpy(), index=labels, columns=labels)

        tb_logger = self.logger.experiment

        tb_logger.add_text(f"Confusion matrix for {split}", confmat.to_markdown(), self.global_step)

    def on_train_epoch_end(self) -> None:
        if self.compute_confusion_matrix:
            # print confusion matrix for train and validation
            self._print_actions_confmat("train", self.train_actions_confmat.confmat)
            self.train_actions_confmat.reset()

    def on_validation_epoch_end(self) -> None:
        if self.compute_confusion_matrix:
            self._print_actions_confmat("valid", self.valid_actions_confmat.confmat)
            self.valid_actions_confmat.reset()

    def validation_step(self, batch, batch_idx):
        return self._inference_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._inference_step(batch, "test")

    @classmethod
    def from_params(
            cls: Type[T],
            params: Params,
            constructor_to_call: Callable[..., T] = None,
            constructor_to_inspect: Union[Callable[..., T], Callable[[T], None]] = None,
            **extras,
    ) -> T:
        vocab = extras.get("vocab")
        num_actions = vocab.get_vocab_size("low_action_labels")
        hidden_dims = params["actor"]["hidden_dims"]
        hidden_dims[-1] = num_actions
        params["actor"]["hidden_dims"] = hidden_dims

        if "start_instr_predictor" not in params:
            logger.warning("Missing 'start_instr_predictor' in config file. Looks like an older version of the model."
                           "Cloning 'object_scorer' module in place...")
            params['start_instr_predictor'] = params['object_scorer'].duplicate()

        return super(EmbodiedBertForAlfred, cls).from_params(params, constructor_to_call, constructor_to_inspect,
                                                             **extras)

    def forward(self,
                metadata: List[Dict[str, Any]],
                token_ids: torch.Tensor,
                token_type_ids: torch.Tensor,
                text_mask: torch.Tensor,
                visual_features: torch.Tensor,
                visual_attention_mask: torch.Tensor = None,
                actions_mask: torch.BoolTensor = None,
                actions: torch.LongTensor = None,
                obj_interact_targets: torch.LongTensor = None,
                obj_interact_mask: torch.BoolTensor = None,
                hiddens: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                start_instr_labels: Optional[torch.Tensor] = None,
                nav_receptacle_labels: Optional[torch.Tensor] = None,
                token_labels: Optional[torch.Tensor] = None,
                visual_labels: Optional[torch.Tensor] = None
                ):
        """
        Input parameters depend on the Instance object returned by the dataset reader class. See grolp/readers/alfred.py
        """
        batch_size, max_episode_len, num_objects = visual_features.shape[:3]

        # Processing language features
        if token_ids.ndim == 2:
            ## Case 1: the language features are specified all at once
            language_features = self.embert(
                mode="language",
                input_ids=token_ids,
                attention_mask=text_mask,
                token_type_ids=token_type_ids
            )

            num_tokens = language_features.shape[1]
            r_language_features = language_features.unsqueeze(1).expand(-1, max_episode_len, -1, -1)
            text_mask = text_mask.unsqueeze(1).expand(-1, max_episode_len, -1)

            r_language_features = r_language_features.reshape(batch_size * max_episode_len, num_tokens, -1)
        else:
            ## Case 2: We have only the goal and the current instruction
            num_tokens = token_ids.shape[-1]
            # These are already divided based on the subgoal information
            r_language_features = self.embert(
                mode="language",
                input_ids=token_ids.view(-1, num_tokens),
                attention_mask=text_mask.view(-1, num_tokens),
                token_type_ids=token_type_ids.view(-1, num_tokens)
            )

        # concat the attention mask
        attention_mask = torch.cat((text_mask, visual_attention_mask), dim=-1)

        # first we introduce the start of sequence tokens
        action_list = []

        input_actions = actions[:, :-1].clone()

        if hiddens is not None:
            action_mems = hiddens[0]
            state_mems = hiddens[1]
            prev_actions = hiddens[2]
            action_list.append(prev_actions)
        else:
            action_mems = None
            state_mems = None

            sos_actions = actions.new_full(
                fill_value=self.action_start_index,
                size=(actions.shape[0], 1)
            )
            action_list.append(sos_actions)

        # finally we add the current input actions shifted by 1
        action_list.append(input_actions)

        input_actions = torch.cat(action_list, -1)

        # we create the input actions for the action decoder
        input_actions[input_actions == -1] = self.action_padding_index

        visual_features = visual_features.reshape(batch_size * max_episode_len, num_objects, -1)

        # encode all the timesteps of all the trajectories independently
        context_embeddings, embeddings, attentions = self.embert(
            mode="visual",
            input_ids=torch.zeros_like(r_language_features) if self.no_language else r_language_features,
            attention_mask=attention_mask.reshape(batch_size * max_episode_len, -1),
            vis_feats=torch.zeros_like(visual_features) if self.no_vision else visual_features
        )

        context_embeddings = context_embeddings.view(batch_size, max_episode_len, -1)
        embeddings = embeddings.view(batch_size, max_episode_len, num_objects + num_tokens, -1)

        # given the low-level encoder embeddings, generate the representations of the state encoder using TransformerXL
        state_outputs = self.state_encoder(
            decoder_input_ids=input_actions,
            encoder_hidden_states=context_embeddings,
            action_mems=action_mems,
            state_mems=state_mems
        )

        hidden_states = state_outputs.last_hidden_state

        # given the hidden states of the decoder, apply conditional scaling
        cond_hidden_states = self.condition_hidden(hidden_states).unsqueeze(2)
        stateful_embeddings = cond_hidden_states * embeddings

        stateful_object_embeddings = stateful_embeddings[:, :, num_tokens:]

        action_logits = self.actor(hidden_states)

        if self.action_conditioning:
            action_cond = actions.clone()
            action_cond[action_cond == -1] = self.action_padding_index
            action_cond_emb = self.state_encoder.word_emb(action_cond)

            # Compute which object we should manipulate from the ones in the front view
            object_logits = self.obj_scorer(
                action_cond_emb.view(-1, self.config.hidden_size),
                stateful_object_embeddings.view(-1, num_objects, self.config.hidden_size)
            ).view(batch_size, max_episode_len, num_objects)
        else:
            # Compute which object we should manipulate from the ones in the front view
            object_logits = self.obj_scorer(
                stateful_object_embeddings
            ).squeeze(-1)

        # we make sure to replace the masked elements before computing the loss
        action_loss = sequence_cross_entropy(action_logits, actions, actions_mask, padding_idx=-1)

        obj_interact_loss = sequence_cross_entropy(object_logits, obj_interact_targets, obj_interact_mask,
                                                   padding_idx=-100)
        # Compute the joint loss
        loss = action_loss * self.config.action_loss_weight + self.config.obj_interact_weight * obj_interact_loss

        output_dict = {
            "action_loss": action_loss,
            "obj_interact_loss": obj_interact_loss,
            "action_logits": action_logits,
            "object_logits": object_logits,
        }
        if self.config.use_start_instr_loss:
            start_instr_logits = self.start_instr_predictor(hidden_states)

            start_instr_loss = sequence_binary_cross_entropy(start_instr_logits, start_instr_labels, actions_mask)

            loss = loss + start_instr_loss
            output_dict["start_instr_loss"] = start_instr_loss

        if self.config.use_nav_receptacle_loss and nav_receptacle_labels is not None:
            nav_receptacle_logits = self.nav_receptacle_scorer(
                stateful_object_embeddings
            ).squeeze(-1)

            nav_receptacle_loss = sequence_cross_entropy(
                nav_receptacle_logits,
                nav_receptacle_labels,
                actions_mask
            )

            loss = loss + nav_receptacle_loss
            output_dict["nav_receptacle_loss"] = nav_receptacle_loss

        mlm_steps = (start_instr_labels == 1.0).view(-1)
        # we extract the token embeddings only for those steps that actually require the MLM
        token_embeddings = embeddings[:, :, :num_tokens, :]
        object_embeddings = embeddings[:, :, num_tokens:, :]
        lang_hidden_states = token_embeddings.reshape(-1, num_tokens, self.config.hidden_size)[mlm_steps]
        vis_hidden_states = object_embeddings.reshape(-1, self.config.num_visual_features, self.config.hidden_size)[
            mlm_steps]
        has_mlm_steps = lang_hidden_states.shape[0] != 0

        if self.config.use_lm_loss and token_labels is not None:
            if has_mlm_steps:
                lang_scores, _ = self.embert.cls(lang_hidden_states, cond_hidden_states)
                token_labels = token_labels.view(-1, num_tokens)[mlm_steps]
                loss_fct = CrossEntropyLoss()
                masked_lm_loss = loss_fct(lang_scores.view(-1, self.config.vocab_size), token_labels.view(-1))
                loss = loss + masked_lm_loss
            else:
                masked_lm_loss = loss.new_zeros(())
            output_dict["lang_loss"] = masked_lm_loss

        if self.config.use_vm_loss and visual_labels is not None:
            if has_mlm_steps:
                visual_scores = self.embert.visual_cls(vis_hidden_states)
                visual_labels = visual_labels.view(-1, self.config.num_visual_features)[mlm_steps]
                loss_fct = CrossEntropyLoss()

                masked_visual_loss = loss_fct(visual_scores.view(-1, self.config.num_object_labels),
                                              visual_labels.view(-1))
                loss = loss + masked_visual_loss
            else:
                masked_visual_loss = loss.new_zeros(())
            output_dict["visual_loss"] = masked_visual_loss

        # compute image-text cross-modal loss
        if self.config.use_itm_loss:
            if has_mlm_steps:
                itm_loss = self.embert.clip_loss(lang_hidden_states, vis_hidden_states)
                loss = loss + itm_loss
                output_dict["itm_loss"] = itm_loss
            else:
                output_dict["itm_loss"] = loss.new_zeros(())

        output_dict["loss"] = loss
        # get the last state of the transformer layer
        prev_actions = actions[:, -1].clone().unsqueeze(-1)
        prev_actions[prev_actions == -1] = self.action_padding_index
        new_hiddens = (state_outputs.action_mems, state_outputs.state_mems, prev_actions)

        output_dict["hidden_states"] = new_hiddens

        return output_dict

    def forward_gold_trajectory(self,
                                metadata: List[Dict[str, Any]],
                                token_ids: torch.Tensor,
                                token_type_ids: torch.Tensor,
                                text_mask: torch.Tensor,
                                visual_features: torch.Tensor,
                                visual_attention_mask: torch.Tensor = None,
                                actions: torch.LongTensor = None,
                                hiddens: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                                ):
        """
        This function is used to complete a forward pass of the model and extract only the hidden states. We use this
        function in the subgoal evaluation script.
        """
        batch_size, max_episode_len, num_objects = visual_features.shape[:3]

        num_tokens = token_ids.shape[-1]
        # otherwise we have already the language instructions divided over the trajectory length
        r_language_features = self.embert(
            mode="language",
            input_ids=token_ids.view(-1, num_tokens),
            attention_mask=text_mask.view(-1, num_tokens),
            token_type_ids=token_type_ids.view(-1, num_tokens)
        )

        attention_mask = torch.cat((text_mask, visual_attention_mask), dim=-1)

        # first we introduce the start of sequence tokens
        action_list = []

        input_actions = actions[:, :-1].clone()

        if hiddens is not None:
            action_mems = hiddens[0]
            state_mems = hiddens[1]
            prev_actions = hiddens[2]
            action_list.append(prev_actions)
        else:
            action_mems = None
            state_mems = None

            sos_actions = actions.new_full(
                fill_value=self.action_start_index,
                size=(actions.shape[0], 1)
            )
            action_list.append(sos_actions)

        # finally we add the current input actions shifted by 1
        action_list.append(input_actions)

        input_actions = torch.cat(action_list, -1)

        # we create the input actions for the action decoder
        input_actions[input_actions == -1] = self.action_padding_index

        visual_features = visual_features.reshape(batch_size * max_episode_len, num_objects, -1)

        # encode all the timesteps of all the trajectories independently
        context_embeddings, embeddings, attentions = self.embert(
            mode="visual",
            input_ids=torch.zeros_like(r_language_features) if self.no_language else r_language_features,
            attention_mask=attention_mask.reshape(batch_size * max_episode_len, -1),
            vis_feats=torch.zeros_like(visual_features) if self.no_vision else visual_features
        )

        context_embeddings = context_embeddings.view(batch_size, max_episode_len, -1)
        state_outputs = self.state_encoder(
            decoder_input_ids=input_actions,
            encoder_hidden_states=context_embeddings,
            action_mems=action_mems,
            state_mems=state_mems
        )

        # get the last state of the transformer layer
        prev_actions = actions[:, -1].clone().unsqueeze(-1)
        prev_actions[prev_actions == -1] = self.action_padding_index

        output_dict = dict(
            action_mems=state_outputs.action_mems,
            state_mems=state_outputs.state_mems,
            prev_actions=prev_actions,
            trajectory_len=max_episode_len,
            language_features=r_language_features.view(batch_size, max_episode_len, -1),
            language_mask=text_mask
        )

        return output_dict

    def _step(self, **kwargs):
        """
        This method is used by the evaluation script to run a single interaction step with the AI2Thor environment
        conditioned on previous states and actions.
        """
        visual_features = kwargs.get("visual_features")
        action_mems = kwargs.get("action_mems")
        state_mems = kwargs.get("state_mems")
        prev_actions = kwargs.get("prev_actions")
        prev_objects = kwargs.get("prev_objects")
        visual_attention_mask = kwargs.get("visual_attention_mask")
        num_objects_in_front = kwargs.get("num_objects_in_front", self.config.num_objects_in_front)

        language_features = kwargs["language_features"]
        language_masks = kwargs["language_masks"]

        # this requires only language encoding
        if language_features is None:
            instructions = kwargs["instructions"]

            # extract textual features from the instructions
            input_ids = instructions["tokens"]["token_ids"]
            language_masks = instructions["tokens"]["mask"]
            type_ids = instructions["tokens"]["type_ids"]

            ''' Language BERT '''
            """
            mode, input_ids, token_type_ids=None, attention_mask=None,
                    position_ids=None, vis_feats=None, vis_angle_feats=None, vis_masks=None, vis_type_ids=None
            """
            language_features = self.embert(
                mode="language",
                input_ids=input_ids,
                attention_mask=language_masks,
                token_type_ids=type_ids
            )

        if state_mems is not None and action_mems is not None and prev_actions is not None:
            input_actions = prev_actions
        else:
            action_mems = None
            state_mems = None

            input_actions = language_masks.new_full(
                fill_value=self.action_start_index,
                size=(language_masks.shape[0], 1),
                dtype=torch.long
            )

            if self.state_encoder.use_object_embeddings:
                prev_objects = visual_features.new_zeros(
                    size=(visual_features.shape[0], 1, self.config.hidden_size)
                )

        num_tokens = language_features.shape[1]

        attention_mask = torch.cat((language_masks, visual_attention_mask), dim=-1)

        # encode all the timesteps of all the trajectories independently
        context_embeddings, embeddings, attentions = self.embert(
            mode="visual",
            input_ids=language_features,
            attention_mask=attention_mask,
            vis_feats=visual_features
        )

        context_embeddings = context_embeddings.view(1, 1, -1)

        state_outputs = self.state_encoder(
            decoder_input_ids=input_actions,
            encoder_hidden_states=context_embeddings,
            action_mems=action_mems,
            state_mems=state_mems,
            object_embeddings=prev_objects
        )

        hidden_states = state_outputs.last_hidden_state

        curr_state = hidden_states[:, -1]
        action_logits = self.actor(curr_state)
        action_probs = torch.softmax(action_logits, -1)
        pred_actions = torch.argmax(action_probs, -1)

        batch_size = embeddings.shape[0]

        if num_objects_in_front > 0:
            object_embeddings = embeddings[:, num_tokens:num_tokens + num_objects_in_front].reshape(
                batch_size, num_objects_in_front, -1
            )
            cond_hidden_states = self.condition_hidden(hidden_states).unsqueeze(2)
            stateful_object_embeddings = cond_hidden_states * object_embeddings

            if self.action_conditioning:
                action_cond_emb = self.state_encoder.word_emb(pred_actions)

                # Compute which object we should manipulate from the ones in the front view
                object_logits = self.obj_scorer(
                    action_cond_emb,
                    stateful_object_embeddings.view(-1, num_objects_in_front, self.config.hidden_size)
                ).view(-1, num_objects_in_front)
            else:
                # Compute which object we should manipulate from the ones in the front view
                object_logits = self.obj_scorer(
                    stateful_object_embeddings
                ).squeeze(-1)

            obj_mask = visual_attention_mask[:, :num_objects_in_front]
            obj_probs = masked_softmax(object_logits, obj_mask, -1)
            pred_objects = torch.argmax(obj_probs, -1)

            if self.state_encoder.use_object_embeddings:
                prev_objects = object_embeddings[:, pred_objects, :]
            else:
                prev_objects = None

        else:
            pred_objects = [None]
            obj_probs = [None]

        prev_actions = pred_actions.unsqueeze(-1)

        output_dict = {
            "action_probs": action_probs,
            "object_probs": obj_probs,
            "pred_actions": pred_actions,
            "pred_objects": pred_objects,
            "action_mems": state_outputs.action_mems,
            "state_mems": state_outputs.state_mems,
            "prev_actions": prev_actions,
            "language_features": language_features,
            "language_masks": language_masks,
            "interactive_object_masks": [m["interactive_object_masks"] for m in kwargs.get("metadata")]
        }

        if self.state_encoder.use_object_embeddings and prev_objects is not None:
            output_dict["prev_objects"] = prev_objects

        if self.start_instr_predictor is not None:
            next_instruction_logits = self.start_instr_predictor(hidden_states)
            goto_next_instruction = torch.sigmoid(next_instruction_logits)

            output_dict["goto_next_instruction"] = goto_next_instruction
        else:
            output_dict["goto_next_instruction"] = [None]

        return output_dict

    def make_output_human_readable(
            self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """
        Used to generate the model output: 1. action prediction, 2. next step instruction prediction and 3. mask prediction
        :param output_dict:
        :return:
        """
        action_ids = output_dict["pred_actions"]
        object_ids = output_dict["pred_objects"]
        goto_next_instruction = output_dict["goto_next_instruction"]
        interact_object_masks = output_dict["interactive_object_masks"]

        actions = []
        masks = []
        next_instruction = []

        for b_idx, (action_id, object_id, next_instr) in enumerate(zip(action_ids, object_ids, goto_next_instruction)):
            action = self.vocab.get_token_from_index(action_id.item(), "low_action_labels")

            if is_manipulation_action(action) and object_id is not None:
                # these are all the object masks in the current views
                curr_masks = numpy.concatenate(interact_object_masks[b_idx])
                pred_mask = curr_masks[object_id]
                masks.append(pred_mask)
            else:
                masks.append(None)

            if next_instr is not None:
                next_instruction.append((next_instr > 0.75).item())
            else:
                next_instruction.append(None)

            actions.append(action)

        output_dict["actions"] = actions
        output_dict["masks"] = masks
        output_dict["goto_next_instruction"] = next_instruction

        return output_dict

    def step(self, instances: List[Instance], action_mems: Tensor = None, state_mems: Tensor = None,
             language_features=None,
             language_masks=None, prev_actions: Tensor = None, prev_objects: Tensor = None,
             num_objects_in_front=None, is_gold_trajectory=False):
        """
        Takes a list of `Instances`, converts that text into arrays using this model's `Vocabulary`,
        passes those arrays through `self.forward()` and `self.make_output_human_readable()` (which
        by default does nothing) and returns the result.  Before returning the result, we convert
        any `torch.Tensors` into numpy arrays and separate the batched output into a list of
        individual dicts per instance. Note that typically this will be faster on a GPU (and
        conditionally, on a CPU) than repeated calls to `forward_on_instance`.

        # Parameters

        instances : `List[Instance]`, required
            The instances to run the model on.

        # Returns

        A list of the models output for each instance.
        """
        batch_size = len(instances)

        with torch.no_grad():
            cuda_device = self._get_prediction_device()
            dataset = Batch(instances)
            dataset.index_instances(self.vocab)
            tensor_dict = dataset.as_tensor_dict()

            if is_gold_trajectory:
                tensor_dict["token_ids"] = tensor_dict["instructions"]["tokens"]["token_ids"]
                tensor_dict["token_type_ids"] = tensor_dict["instructions"]["tokens"]["type_ids"]
                tensor_dict["text_mask"] = tensor_dict["instructions"]["tokens"]["mask"]
                del tensor_dict["instructions"]
                model_input = util.move_to_device(tensor_dict, cuda_device)

                return self.forward_gold_trajectory(**model_input)

            tensor_dict["action_mems"] = action_mems
            tensor_dict["state_mems"] = state_mems
            tensor_dict["language_features"] = language_features
            tensor_dict["language_masks"] = language_masks
            tensor_dict["prev_actions"] = prev_actions
            tensor_dict["prev_objects"] = prev_objects
            model_input = util.move_to_device(tensor_dict, cuda_device)
            if num_objects_in_front is not None:
                tensor_dict["num_objects_in_front"] = num_objects_in_front

            outputs = self.make_output_human_readable(self._step(**model_input))

            instance_separated_output: List[Dict[str, numpy.ndarray]] = [
                {} for _ in dataset.instances
            ]
            for name, output in list(outputs.items()):
                if name in {"action_mems", "state_mems"}:
                    instance_output[name] = output
                elif isinstance(output, torch.Tensor):
                    # NOTE(markn): This is a hack because 0-dim pytorch tensors are not iterable.
                    # This occurs with batch size 1, because we still want to include the loss in that case.
                    if output.dim() == 0:
                        output = output.unsqueeze(0)

                    if output.size(0) != batch_size:
                        self._maybe_warn_for_unseparable_batches(name)
                        continue
                    output = output.detach().cpu()

                    # we keep these as Torch tensors
                    if name not in {"language_features", "language_masks", "prev_actions", "prev_objects"}:
                        output = output.numpy()

                elif len(output) != batch_size:
                    self._maybe_warn_for_unseparable_batches(name)
                    continue

                for instance_output, batch_element in zip(instance_separated_output, output):
                    if name not in {"action_mems", "state_mems"}:
                        instance_output[name] = batch_element
                    else:
                        instance_output[name] = output
            return instance_separated_output

    @classmethod
    def load_from_checkpoint(
            cls,
            checkpoint_path: Union[str, IO],
            map_location: Optional[Union[Dict[str, str], str, torch.device, int, Callable]] = None,
            hparams_file: Optional[str] = None,
            strict: bool = True,
            **kwargs,
    ):
        tempdir = None

        try:
            if checkpoint_path.endswith("tar.gz"):
                resolved_archive_file = cached_path(checkpoint_path)

                if resolved_archive_file == checkpoint_path:
                    logger.info(f"loading archive file {checkpoint_path}")
                else:
                    logger.info(f"loading archive file {checkpoint_path} from cache at {resolved_archive_file}")

                if os.path.isdir(resolved_archive_file):
                    serialization_dir = resolved_archive_file
                else:
                    with extracted_archive(resolved_archive_file, cleanup=False) as tempdir:
                        serialization_dir = tempdir

                checkpoint_path = os.path.join(serialization_dir, "weights.th")

            elif os.path.isdir(checkpoint_path):
                serialization_dir = checkpoint_path
                checkpoint_path = os.path.join(serialization_dir, "best.th")
            else:
                serialization_dir = os.path.dirname(checkpoint_path)

            config_file = os.path.join(serialization_dir, "config.json")

            config = Params.from_file(config_file)

            if map_location is not None:
                checkpoint = pl_load(checkpoint_path, map_location=map_location)
            else:
                checkpoint = pl_load(checkpoint_path, map_location=lambda storage, loc: storage)

            if hparams_file is not None:
                extension = hparams_file.split('.')[-1]
                if extension.lower() == 'csv':
                    hparams = load_hparams_from_tags_csv(hparams_file)
                elif extension.lower() in ('yml', 'yaml'):
                    hparams = load_hparams_from_yaml(hparams_file)
                else:
                    raise ValueError('.csv, .yml or .yaml is required for `hparams_file`')

                hparams['on_gpu'] = False

                # overwrite hparams by the given file
                checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY] = hparams

            # for past checkpoint need to add the new key
            if cls.CHECKPOINT_HYPER_PARAMS_KEY not in checkpoint:
                checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY] = {}
            # override the hparams with values that were passed in
            checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY].update(kwargs)

            kwargs["config"] = config
            kwargs["serialization_dir"] = serialization_dir
            # assumes that map_location is a torch.device
            kwargs["cuda_device"] = map_location

            dataset_reader, _ = _load_dataset_readers(
                config.duplicate(), serialization_dir
            )
            model = cls._load_model_state(checkpoint, strict=strict, **kwargs)
        finally:
            if tempdir is not None:
                logger.info(f"removing temporary unarchived model dir at {tempdir}")
                shutil.rmtree(tempdir, ignore_errors=True)

        return model, dataset_reader

    @classmethod
    def _load_model_state(cls, checkpoint: Dict[str, Any], strict: bool = True, **cls_kwargs_new):
        cls_spec = inspect.getfullargspec(cls.__init__)
        cls_init_args_name = inspect.signature(cls.__init__).parameters.keys()

        self_var, args_var, kwargs_var = parse_class_init_keys(cls)
        drop_names = [n for n in (self_var, args_var, kwargs_var) if n]
        cls_init_args_name = list(filter(lambda n: n not in drop_names, cls_init_args_name))

        cls_kwargs_loaded = {}
        # pass in the values we saved automatically
        if cls.CHECKPOINT_HYPER_PARAMS_KEY in checkpoint:

            # 1. (backward compatibility) Try to restore model hparams from checkpoint using old/past keys
            for _old_hparam_key in CHECKPOINT_PAST_HPARAMS_KEYS:
                cls_kwargs_loaded.update(checkpoint.get(_old_hparam_key, {}))

            # 2. Try to restore model hparams from checkpoint using the new key
            _new_hparam_key = cls.CHECKPOINT_HYPER_PARAMS_KEY
            cls_kwargs_loaded.update(checkpoint.get(_new_hparam_key))

            # 3. Ensure that `cls_kwargs_old` has the right type, back compatibility between dict and Namespace
            cls_kwargs_loaded = _convert_loaded_hparams(
                cls_kwargs_loaded, checkpoint.get(cls.CHECKPOINT_HYPER_PARAMS_TYPE)
            )

            # 4. Update cls_kwargs_new with cls_kwargs_old, such that new has higher priority
            args_name = checkpoint.get(cls.CHECKPOINT_HYPER_PARAMS_NAME)
            if args_name and args_name in cls_init_args_name:
                cls_kwargs_loaded = {args_name: cls_kwargs_loaded}

        _cls_kwargs = {}
        _cls_kwargs.update(cls_kwargs_loaded)
        _cls_kwargs.update(cls_kwargs_new)

        if not cls_spec.varkw:
            # filter kwargs according to class init unless it allows any argument via kwargs
            _cls_kwargs = {k: v for k, v in _cls_kwargs.items() if k in cls_init_args_name}

        config = cls_kwargs_new.get("config")
        serialization_dir = cls_kwargs_new.get("serialization_dir")
        cuda_device = cls_kwargs_new.get("cuda_device")

        # Load vocabulary from file
        vocab_dir = os.path.join(serialization_dir, "vocabulary")
        # If the config specifies a vocabulary subclass, we need to use it.
        vocab_params = config.get("vocabulary", Params({}))
        vocab_choice = vocab_params.pop_choice("type", Vocabulary.list_available(), True)
        vocab_class, _ = Vocabulary.resolve_class_name(vocab_choice)
        vocab = vocab_class.from_files(
            vocab_dir, vocab_params.get("padding_token"), vocab_params.get("oov_token")
        )

        model_params = config.get("model")

        # The experiment config tells us how to _train_ a model, including where to get pre-trained
        # embeddings/weights from. We're now _loading_ the model, so those weights will already be
        # stored in our model. We don't need any pretrained weight file or initializers anymore,
        # and we don't want the code to look for it, so we remove it from the parameters here.
        remove_keys_from_params(model_params)
        model = Model.from_params(
            vocab=vocab, params=model_params, serialization_dir=serialization_dir
        )

        model = model.to(cuda_device)

        # Load state dict. We pass `strict=False` so PyTorch doesn't raise a RuntimeError
        # if the state dict is missing keys because we handle this case below.
        model_state = checkpoint['state_dict']
        missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)

        if unexpected_keys or missing_keys:
            raise RuntimeError(
                f"Error loading state dict for {model.__class__.__name__}\n\t"
                f"Missing keys: {missing_keys}\n\t"
                f"Unexpected keys: {unexpected_keys}"
            )

        model = model.eval()

        return model
