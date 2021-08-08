# Copyright (c) 2020 Microsoft Corporation. Licensed under the MIT license.
# Modified in Recurrent VLN-BERT, 2020, Yicong.Hong@anu.edu.au

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import math
import os
import re
from typing import Optional, Union

import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn.functional import normalize
from transformers import PretrainedConfig, TF2_WEIGHTS_NAME, WEIGHTS_NAME
from transformers.file_utils import is_remote_url, hf_bucket_url, cached_path, is_torch_tpu_available, TF_WEIGHTS_NAME
from transformers.models.bert.modeling_bert import BertSelfAttention, BertAttention, BertSelfOutput, BertLayer, \
    BertIntermediate, BertOutput, BertEncoder, BertPreTrainedModel, BertEmbeddings, BertPooler, BertPreTrainingHeads, \
    BertPredictionHeadTransform

from grolp.models.config import EmbodiedBertConfig

logger = logging.getLogger(__name__)


class CaptionBertSelfAttention(BertSelfAttention):
    """
    Modified from BertSelfAttention to add support for output_hidden_states.
    """

    def __init__(self, config: EmbodiedBertConfig):
        super(CaptionBertSelfAttention, self).__init__(config)
        self.config = config

    def forward(self, mode, hidden_states, attention_mask, head_mask=None,
                history_state=None):
        if history_state is not None:
            x_states = torch.cat([history_state, hidden_states], dim=1)
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(x_states)
            mixed_value_layer = self.value(x_states)
        else:
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        if mode == 'visual':
            mixed_query_layer = mixed_query_layer[:, [0] + list(range(-self.config.num_visual_features, 0)), :]

        ''' language feature only provide Keys and Values '''
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_scores)

        return outputs


class CaptionBertAttention(BertAttention):
    """
    Modified from BertAttention to add support for output_hidden_states.
    """

    def __init__(self, config: EmbodiedBertConfig):
        super(CaptionBertAttention, self).__init__(config)
        self.self = CaptionBertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.config = config

    def forward(self, mode, input_tensor, attention_mask, head_mask=None,
                history_state=None):
        ''' transformer processing '''
        self_outputs = self.self(mode, input_tensor, attention_mask, head_mask, history_state)

        ''' feed-forward network with residual '''
        if mode == 'visual':
            attention_output = self.output(self_outputs[0],
                                           input_tensor[:, [0] + list(range(-self.config.num_visual_features, 0)), :])
        if mode == 'language' or mode == 'vis_lang':
            attention_output = self.output(self_outputs[0], input_tensor)

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them

        return outputs


class CaptionBertLayer(BertLayer):
    """
    Modified from BertLayer to add support for output_hidden_states.
    """

    def __init__(self, config: EmbodiedBertConfig):
        super(CaptionBertLayer, self).__init__(config)
        self.attention = CaptionBertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, mode, hidden_states, attention_mask, head_mask=None,
                history_state=None):
        attention_outputs = self.attention(mode, hidden_states, attention_mask,
                                           head_mask, history_state)

        ''' feed-forward network with residual connections '''
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]

        return outputs


class CaptionBertEncoder(BertEncoder):
    """
    Modified from BertEncoder to add support for output_hidden_states.
    """

    def __init__(self, config: EmbodiedBertConfig):
        super(CaptionBertEncoder, self).__init__(config)
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        # 12 Bert layers
        self.layer = nn.ModuleList([CaptionBertLayer(config) for _ in range(config.num_hidden_layers)])
        self.config = config

    def forward(self, mode, hidden_states, attention_mask, head_mask=None,
                encoder_history_states=None):

        if mode == 'visual':
            for i, layer_module in enumerate(self.layer):
                history_state = None if encoder_history_states is None else encoder_history_states[i]

                layer_outputs = layer_module(mode,
                                             hidden_states, attention_mask, head_mask[i],
                                             history_state)

                concat_layer_outputs = torch.cat((layer_outputs[0][:, 0:1, :],
                                                  hidden_states[:, 1:-self.config.num_visual_features, :],
                                                  layer_outputs[0][:, 1:self.config.num_visual_features + 1, :]), 1)
                hidden_states = concat_layer_outputs

                if i == self.config.num_hidden_layers - 1:
                    state_attention_score = layer_outputs[1][:, :, 0, :]
                    lang_attention_score = layer_outputs[1][:, :, -self.config.num_visual_features:,
                                           1:-self.config.num_visual_features]
                    vis_attention_score = layer_outputs[1][:, :, :, :]

            outputs = (hidden_states, state_attention_score, lang_attention_score, vis_attention_score)

        elif mode == 'language' or mode == 'vis_lang':
            for i, layer_module in enumerate(self.layer):
                history_state = None if encoder_history_states is None else encoder_history_states[i]  # default None

                layer_outputs = layer_module(mode,
                                             hidden_states, attention_mask, head_mask[i],
                                             history_state)
                hidden_states = layer_outputs[0]

                if i == self.config.num_hidden_layers - 1:
                    slang_attention_score = layer_outputs[1]

            outputs = (hidden_states, slang_attention_score)

        return outputs


class BertImgModel(BertPreTrainedModel):
    """ Expand from BertModel to handle image region features as input
    """

    def __init__(self, config: EmbodiedBertConfig):
        super(BertImgModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = CaptionBertEncoder(config)
        self.pooler = BertPooler(config)
        # this allows us to reuse the original OSCAR weights
        self.img_projection = nn.Linear(self.config.visual_emb_size, self.config.oscar_img_feature_dim, bias=True)
        self.img_embedding = nn.Linear(self.config.oscar_img_feature_dim, self.config.hidden_size, bias=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.init_weights()

    def forward(self, mode, input_ids, token_type_ids=None, attention_mask=None,
                position_ids=None, vis_feats=None, vis_type_ids=None):

        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError

        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        head_mask = [None] * self.config.num_hidden_layers

        if mode == 'visual':
            language_features = input_ids

            # we first project back to the OSCAR visual features dimensionality
            vis_feats = self.img_projection(vis_feats)

            # then we create visual tokens
            vis_feats = self.img_embedding(
                vis_feats
            )
            concat_embedding_output = torch.cat((language_features, vis_feats), 1)
        elif mode == 'language':
            embedding_output = self.embeddings(input_ids, position_ids=position_ids,
                                               token_type_ids=token_type_ids)
            concat_embedding_output = embedding_output
        elif mode == "vis_lang":
            # we first project back to the OSCAR visual features dimensionality
            vis_feats = self.img_projection(vis_feats)

            # then we create visual tokens
            vis_feats = self.img_embedding(
                vis_feats
            )

            lang_feats = self.embeddings(input_ids, position_ids=position_ids,
                                         token_type_ids=token_type_ids)

            concat_embedding_output = torch.cat((lang_feats, vis_feats), 1)

        ''' pass to the Transformer layers '''
        encoder_outputs = self.encoder(mode, concat_embedding_output,
                                       extended_attention_mask, head_mask=head_mask)

        sequence_output = encoder_outputs[0]

        # add hidden_states and attentions if they are here
        outputs = (sequence_output,) + encoder_outputs[1:]

        return outputs


class EmbodiedBertVisPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        self.decoder = nn.Linear(config.hidden_size, config.num_object_labels)

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class EmbodiedBert(BertPreTrainedModel):
    """
    Modified from BertForMultipleChoice to support oscar training.
    """

    def __init__(self, config: EmbodiedBertConfig):
        super(EmbodiedBert, self).__init__(config)
        self.config = config
        self.bert = BertImgModel(config)

        if self.config.state_repr_method == "pooled":
            self.state_hidden_dim = config.hidden_size
        else:
            self.state_hidden_dim = config.hidden_size

        # Masked Language and Vision modelling
        if config.use_lm_loss:
            self.cls = BertPreTrainingHeads(self.config)
        else:
            self.cls = None

        if self.config.use_itm_loss:
            self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        else:
            self.logit_scale = None

        if config.use_vm_loss:
            self.visual_cls = EmbodiedBertVisPredictionHead(self.config)
        else:
            self.visual_cls = None

        self.init_weights()

    def clip_loss(self, text_embeddings, visual_embeddings):
        # image_encoder - ResNet or Vision Transformer
        # text_encoder - CBOW or Text Transformer
        # I[n, h, w, c] - minibatch of aligned images
        # T[n, l] - minibatch of aligned texts
        # W_i[d_i, d_e] - learned proj of image to embed
        # W_t[d_t, d_e] - learned proj of text to embed
        # t - learned temperature parameter
        # extract feature representations of each modality
        image_features = visual_embeddings.sum(1)  # [n, d_i]
        text_features = text_embeddings[:, 0, :]  # [n, d_t]
        # joint multimodal embedding [n, d_e]
        # I_e = normalize(np.dot(I_f, W_i), axis=1)
        # T_e = l2_normalize(np.dot(T_f, W_t), axis=1)
        # scaled pairwise cosine similarities [n, n]
        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        # symmetric loss function
        labels = torch.arange(logits_per_image.shape[0], device=text_features.device)
        loss_i = torch.nn.functional.cross_entropy(logits_per_image, labels)
        loss_t = torch.nn.functional.cross_entropy(logits_per_text, labels)
        loss = (loss_i + loss_t) / 2

        return loss

    def _generate_state_representation(self, sequence_output, pooled_output, language_att_mask):
        if self.config.state_repr_method == "pooled":
            state_output = pooled_output
        elif self.config.state_repr_method == "sum":
            state_output = sequence_output.sum(1)
        elif self.config.state_repr_method == "dot_product":
            valid_indices = language_att_mask.sum(-1).long() - 1
            visual_features = sequence_output[range(valid_indices.shape[0]), valid_indices]
            cls_features = sequence_output[:, 0]

            state_output = cls_features * visual_features
        else:
            state_output = sequence_output[:, 0]

        return state_output

    def forward(self, mode, input_ids, token_type_ids=None, attention_mask=None,
                position_ids=None, vis_feats=None):

        outputs = self.bert(mode, input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, vis_feats=vis_feats)

        sequence_output = outputs[0]

        if mode == 'language':
            return sequence_output

        elif mode == 'visual' or mode == "vis_lang":
            language_att_mask = attention_mask[:, :input_ids.shape[1]]
            state_output = self._generate_state_representation(sequence_output, outputs[1], language_att_mask)

            return state_output, sequence_output, outputs[-1]
        else:
            raise ValueError("Wrong mode for forward!")


class EmbodiedBertForPreTraining(BertPreTrainedModel):
    base_model_prefix = "embert"

    def __init__(self, config: EmbodiedBertConfig):
        super().__init__(config)
        self.config = config
        self.embert = EmbodiedBert(config)
        self.cls = BertPreTrainingHeads(config)
        self.visual_cls = EmbodiedBertVisPredictionHead(config)
        if config.use_itm_loss:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        else:
            self.logit_scale = None
        self.init_weights()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        config = kwargs.pop("config", None)
        state_dict = kwargs.pop("state_dict", None)
        cache_dir = kwargs.pop("cache_dir", None)
        from_tf = kwargs.pop("from_tf", False)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        output_loading_info = kwargs.pop("output_loading_info", False)
        local_files_only = kwargs.pop("local_files_only", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        mirror = kwargs.pop("mirror", None)

        # Load config if we don't provide a configuration
        if not isinstance(config, PretrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            config, model_kwargs = cls.config_class.from_pretrained(
                config_path,
                *model_args,
                cache_dir=cache_dir,
                return_unused_kwargs=True,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                revision=revision,
                **kwargs,
            )
        else:
            model_kwargs = kwargs

        # Load model
        if pretrained_model_name_or_path is not None:
            pretrained_model_name_or_path = str(pretrained_model_name_or_path)
            if os.path.isdir(pretrained_model_name_or_path):
                if from_tf and os.path.isfile(os.path.join(pretrained_model_name_or_path, TF_WEIGHTS_NAME + ".index")):
                    # Load from a TF 1.0 checkpoint in priority if from_tf
                    archive_file = os.path.join(pretrained_model_name_or_path, TF_WEIGHTS_NAME + ".index")
                elif from_tf and os.path.isfile(os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)):
                    # Load from a TF 2.0 checkpoint in priority if from_tf
                    archive_file = os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)
                elif os.path.isfile(os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)):
                    # Load from a PyTorch checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
                else:
                    raise EnvironmentError(
                        "Error no file named {} found in directory {} or `from_tf` set to False".format(
                            [WEIGHTS_NAME, TF2_WEIGHTS_NAME, TF_WEIGHTS_NAME + ".index"],
                            pretrained_model_name_or_path,
                        )
                    )
            elif os.path.isfile(pretrained_model_name_or_path) or is_remote_url(pretrained_model_name_or_path):
                archive_file = pretrained_model_name_or_path
            elif os.path.isfile(pretrained_model_name_or_path + ".index"):
                assert (
                    from_tf
                ), "We found a TensorFlow checkpoint at {}, please set from_tf to True to load from this checkpoint".format(
                    pretrained_model_name_or_path + ".index"
                )
                archive_file = pretrained_model_name_or_path + ".index"
            else:
                archive_file = hf_bucket_url(
                    pretrained_model_name_or_path,
                    filename=(TF2_WEIGHTS_NAME if from_tf else WEIGHTS_NAME),
                    revision=revision,
                    mirror=mirror,
                )

            try:
                # Load from URL or cache if already cached
                resolved_archive_file = cached_path(
                    archive_file,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                    use_auth_token=use_auth_token,
                )
            except EnvironmentError as err:
                logger.error(err)
                msg = (
                    f"Can't load weights for '{pretrained_model_name_or_path}'. Make sure that:\n\n"
                    f"- '{pretrained_model_name_or_path}' is a correct model identifier listed on 'https://huggingface.co/models'\n\n"
                    f"- or '{pretrained_model_name_or_path}' is the correct path to a directory containing a file named one of {WEIGHTS_NAME}, {TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME}.\n\n"
                )
                raise EnvironmentError(msg)

            if resolved_archive_file == archive_file:
                logger.info("loading weights file {}".format(archive_file))
            else:
                logger.info("loading weights file {} from cache at {}".format(archive_file, resolved_archive_file))
        else:
            resolved_archive_file = None

        config.name_or_path = pretrained_model_name_or_path

        # Instantiate model.
        model = cls(config, *model_args, **model_kwargs)

        if state_dict is None and not from_tf:
            try:
                state_dict = torch.load(resolved_archive_file, map_location="cpu")
            except Exception:
                raise OSError(
                    f"Unable to load weights from pytorch checkpoint file for '{pretrained_model_name_or_path}' "
                    f"at '{resolved_archive_file}'"
                    "If you tried to load a PyTorch model from a TF 2.0 checkpoint, please set from_tf=True. "
                )

        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        if from_tf:
            if resolved_archive_file.endswith(".index"):
                # Load from a TensorFlow 1.X checkpoint - provided by original authors
                model = cls.load_tf_weights(model, config, resolved_archive_file[:-6])  # Remove the '.index'
            else:
                # Load from our TensorFlow 2.0 checkpoints
                try:
                    from .modeling_tf_pytorch_utils import load_tf2_checkpoint_in_pytorch_model

                    model = load_tf2_checkpoint_in_pytorch_model(model, resolved_archive_file, allow_missing_keys=True)
                except ImportError:
                    logger.error(
                        "Loading a TensorFlow model in PyTorch, requires both PyTorch and TensorFlow to be installed. Please see "
                        "https://pytorch.org/ and https://www.tensorflow.org/install/ for installation instructions."
                    )
                    raise
        else:
            # Convert old format to new format if needed from a PyTorch state_dict
            old_keys = []
            new_keys = []
            for key in state_dict.keys():
                new_key = None
                if "gamma" in key:
                    new_key = key.replace("gamma", "weight")
                if "beta" in key:
                    new_key = key.replace("beta", "bias")
                if new_key:
                    old_keys.append(key)
                    new_keys.append(new_key)
            for old_key, new_key in zip(old_keys, new_keys):
                state_dict[new_key] = state_dict.pop(old_key)

            # copy state_dict so _load_from_state_dict can modify it
            metadata = getattr(state_dict, "_metadata", None)
            state_dict = state_dict.copy()
            if metadata is not None:
                state_dict._metadata = metadata

            # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
            # so we need to apply the function recursively.
            def load(module: nn.Module, prefix=""):
                local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
                module._load_from_state_dict(
                    state_dict,
                    prefix,
                    local_metadata,
                    True,
                    missing_keys,
                    unexpected_keys,
                    error_msgs,
                )
                for name, child in module._modules.items():
                    if child is not None:
                        load(child, prefix + name + ".")

            # Make sure we are able to load base models as well as derived models (with heads)
            start_prefix = ""
            model_to_load = model
            has_prefix_module = any(s.startswith(cls.base_model_prefix) for s in state_dict.keys())
            if not hasattr(model, cls.base_model_prefix) and has_prefix_module:
                start_prefix = cls.base_model_prefix + "."
            if hasattr(model, cls.base_model_prefix) and not has_prefix_module:
                model_to_load = getattr(model, cls.base_model_prefix)

            load(model_to_load, prefix=start_prefix)
            state_dict = {k: v for k, v in state_dict.items() if k.startswith("cls")}
            missing_keys = [k for k in missing_keys if not k.startswith("cls")]
            unexpected_keys = [k for k in unexpected_keys if not k.startswith("cls")]
            load(model.cls, prefix="cls.")

            if model.__class__.__name__ != model_to_load.__class__.__name__:
                base_model_state_dict = model_to_load.state_dict().keys()
                head_model_state_dict_without_base_prefix = [
                    key.split(cls.base_model_prefix + ".")[-1] for key in model.state_dict().keys()
                ]
                missing_keys.extend(head_model_state_dict_without_base_prefix - base_model_state_dict)

            # Some models may have keys that are not in the state by design, removing them before needlessly warning
            # the user.
            if cls._keys_to_ignore_on_load_missing is not None:
                for pat in cls._keys_to_ignore_on_load_missing:
                    missing_keys = [k for k in missing_keys if re.search(pat, k) is None]

            if cls._keys_to_ignore_on_load_unexpected is not None:
                for pat in cls._keys_to_ignore_on_load_unexpected:
                    unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

            if len(unexpected_keys) > 0:
                logger.warning(
                    f"Some weights of the model checkpoint at {pretrained_model_name_or_path} were not used when "
                    f"initializing {model.__class__.__name__}: {unexpected_keys}\n"
                    f"- This IS expected if you are initializing {model.__class__.__name__} from the checkpoint of a model trained on another task "
                    f"or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).\n"
                    f"- This IS NOT expected if you are initializing {model.__class__.__name__} from the checkpoint of a model that you expect "
                    f"to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model)."
                )
            else:
                logger.info(f"All model checkpoint weights were used when initializing {model.__class__.__name__}.\n")
            if len(missing_keys) > 0:
                logger.warning(
                    f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at {pretrained_model_name_or_path} "
                    f"and are newly initialized: {missing_keys}\n"
                    f"You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference."
                )
            else:
                logger.info(
                    f"All the weights of {model.__class__.__name__} were initialized from the model checkpoint at {pretrained_model_name_or_path}.\n"
                    f"If your task is similar to the task the model of the checkpoint was trained on, "
                    f"you can already use {model.__class__.__name__} for predictions without further training."
                )
            if len(error_msgs) > 0:
                raise RuntimeError(
                    "Error(s) in loading state_dict for {}:\n\t{}".format(
                        model.__class__.__name__, "\n\t".join(error_msgs)
                    )
                )
        # make sure token embedding weights are still tied if needed
        model.tie_weights()

        # Set model in evaluation mode to deactivate DropOut modules by default
        model.eval()

        if output_loading_info:
            loading_info = {
                "missing_keys": missing_keys,
                "unexpected_keys": unexpected_keys,
                "error_msgs": error_msgs,
            }
            return model, loading_info

        if hasattr(config, "xla_device") and config.xla_device and is_torch_tpu_available():
            import torch_xla.core.xla_model as xm

            model = xm.send_cpu_data_to_device(model, xm.xla_device())
            model.to(xm.xla_device())

        return model

    def clip_loss(self, text_embeddings, visual_embeddings):
        # image_encoder - ResNet or Vision Transformer
        # text_encoder - CBOW or Text Transformer
        # I[n, h, w, c] - minibatch of aligned images
        # T[n, l] - minibatch of aligned texts
        # W_i[d_i, d_e] - learned proj of image to embed
        # W_t[d_t, d_e] - learned proj of text to embed
        # t - learned temperature parameter
        # extract feature representations of each modality
        image_features = visual_embeddings.sum(1)  # [n, d_i]
        text_features = text_embeddings[:, 0, :]  # [n, d_t]
        # joint multimodal embedding [n, d_e]
        # I_e = normalize(np.dot(I_f, W_i), axis=1)
        # T_e = l2_normalize(np.dot(T_f, W_t), axis=1)
        # scaled pairwise cosine similarities [n, n]
        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        # symmetric loss function
        labels = torch.arange(logits_per_image.shape[0])
        loss_i = torch.nn.functional.cross_entropy(logits_per_image, labels)
        loss_t = torch.nn.functional.cross_entropy(logits_per_text, labels)
        loss = (loss_i + loss_t) / 2

        return loss

    def forward(
            self,
            token_ids,
            token_type_ids,
            text_mask,
            visual_features,
            visual_attention_mask,
            lang_labels=None,
            visual_labels=None
    ):
        attention_mask = torch.cat([text_mask, visual_attention_mask], 1)

        outputs = self.embert(
            "vis_lang", token_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
            vis_feats=visual_features
        )

        state_output, sequence_output = outputs[:2]
        lang_hidden_states = sequence_output[:, :-self.config.num_visual_features, :]
        vis_hidden_states = sequence_output[:, -self.config.num_visual_features:, :]

        lang_scores, _ = self.cls(lang_hidden_states, state_output)
        visual_scores = self.visual_cls(vis_hidden_states)

        total_loss = None
        output_dict = {}

        if lang_labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lang_scores.view(-1, self.config.vocab_size), lang_labels.view(-1))
            total_loss = masked_lm_loss
            output_dict["lang_loss"] = masked_lm_loss
            # next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            # total_loss = masked_lm_loss + next_sentence_loss

        if visual_labels is not None:
            loss_fct = CrossEntropyLoss()

            masked_visual_loss = loss_fct(visual_scores.view(-1, self.config.num_object_labels), visual_labels.view(-1))
            total_loss = total_loss + masked_visual_loss
            output_dict["visual_loss"] = masked_visual_loss

        # compute image-text cross-modal loss
        if self.config.use_itm_loss:
            itm_loss = self.clip_loss(lang_hidden_states, vis_hidden_states)
            total_loss = total_loss + itm_loss
            output_dict["itm_loss"] = itm_loss
        output_dict["loss"] = total_loss

        return output_dict
