# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
 PyTorch Transformer XL model. Adapted from https://github.com/kimiyoung/transformer-xl. In particular
 https://github.com/kimiyoung/transformer-xl/blob/master/pytorch/mem_transformer.py
"""
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Type, Callable, Union, T

import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.common import Registrable, Params
from torch import dtype, device
from transformers import PreTrainedModel
from transformers import TransfoXLConfig
from transformers.file_utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, \
    add_code_sample_docstrings
from transformers.utils import logging

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "TransfoXLConfig"
_TOKENIZER_FOR_DOC = "TransfoXLTokenizer"

TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "transfo-xl-wt103",
    # See all Transformer XL models at https://huggingface.co/models?filter=transfo-xl
]


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super().__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:, None, :].expand(-1, bsz, -1)
        else:
            return pos_emb[:, None, :]


class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout, pre_lnorm=False, layer_norm_epsilon=1e-5):
        super().__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_epsilon)

        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        if self.pre_lnorm:
            # layer normalization + positionwise feed-forward
            core_out = self.CoreNet(self.layer_norm(inp))

            # residual connection
            output = core_out + inp
        else:
            # positionwise feed-forward
            core_out = self.CoreNet(inp)

            # residual connection + layer normalization
            output = self.layer_norm(inp + core_out)

        return output


class RelPartialLearnableMultiHeadAttn(nn.Module):
    def __init__(
            self,
            n_head,
            d_model,
            d_head,
            dropout,
            dropatt=0,
            pre_lnorm=False,
            r_r_bias=None,
            r_w_bias=None,
            layer_norm_epsilon=1e-5,
    ):
        super().__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.q_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.kv_net = nn.Linear(d_model, 2 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_epsilon)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

        if r_r_bias is None or r_w_bias is None:  # Biases are not shared
            self.r_r_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
            self.r_w_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
        else:
            self.r_r_bias = r_r_bias
            self.r_w_bias = r_w_bias

        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)

    def _rel_shift(self, x):
        zero_pad_shape = (x.size(0), 1) + x.size()[2:]
        zero_pad = x.new_zeros(zero_pad_shape)
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded_shape = (x.size(1) + 1, x.size(0)) + x.size()[2:]
        x_padded = x_padded.view(*x_padded_shape)

        x = x_padded[1:].view_as(x)

        # ones = x.new_ones((x.size(0), x.size(1)))
        # x = x * torch.tril(ones, x.size(1) - x.size(0))[:,:,None,None]

        return x

    @property
    def device(self) -> device:
        """
        :obj:`torch.device`: The device on which the module is (assuming that all the module parameters are on the same
        device).
        """
        try:
            return next(self.parameters()).device
        except StopIteration:
            # For nn.DataParallel compatibility in PyTorch 1.5

            def find_tensor_attributes(module: nn.Module) -> List[Tuple[str, torch.Tensor]]:
                tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
                return tuples

            gen = self._named_members(get_members_fn=find_tensor_attributes)
            first_tuple = next(gen)
            return first_tuple[1].device

    @property
    def dtype(self) -> dtype:
        """
        :obj:`torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            # For nn.DataParallel compatibility in PyTorch 1.5

            def find_tensor_attributes(module: nn.Module) -> List[Tuple[str, torch.Tensor]]:
                tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
                return tuples

            gen = self._named_members(get_members_fn=find_tensor_attributes)
            first_tuple = next(gen)
            return first_tuple[1].dtype

    def forward(self, hidden_states, rel_pos_emb, encoder_hidden_states, attn_mask=None, state_mems=None,
                action_mems=None, head_mask=None, output_attentions=False):
        qlen, rlen, bsz = hidden_states.size(0), rel_pos_emb.size(0), hidden_states.size(1)

        if action_mems is not None and state_mems is not None:
            cat_actions = torch.cat([action_mems, hidden_states], 0)
            cat_states = torch.cat([state_mems, encoder_hidden_states], 0)

            if self.pre_lnorm:
                mem_heads = self.kv_net(self.layer_norm(cat_states))
                w_head_q = self.q_net(self.layer_norm(cat_actions))
            else:
                mem_heads = self.kv_net(cat_states)
                w_head_q = self.q_net(cat_actions)
            r_head_k = self.r_net(rel_pos_emb)

            w_head_k, w_head_v = torch.chunk(mem_heads, 2, dim=-1)

            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                mem_heads = self.kv_net(self.layer_norm(encoder_hidden_states))
                w_head_q = self.q_net(self.layer_norm(hidden_states))
            else:
                mem_heads = self.kv_net(encoder_hidden_states)
                w_head_q = self.q_net(hidden_states)
            r_head_k = self.r_net(rel_pos_emb)

            w_head_k, w_head_v = torch.chunk(mem_heads, 2, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head

        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)  # qlen x n_head x d_head

        # compute attention score
        rw_head_q = w_head_q + self.r_w_bias  # qlen x bsz x n_head x d_head
        AC = torch.einsum("ibnd,jbnd->ijbn", (rw_head_q, w_head_k))  # qlen x klen x bsz x n_head

        rr_head_q = w_head_q + self.r_r_bias
        BD = torch.einsum("ibnd,jnd->ijbn", (rr_head_q, r_head_k))  # qlen x klen x bsz x n_head
        BD = self._rel_shift(BD)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        # compute attention probability
        if attn_mask is not None and torch.sum(attn_mask).item():
            attn_mask = attn_mask == 1  # Switch to bool
            if attn_mask.dim() == 2:
                if self.dtype == torch.float16:
                    attn_score = (
                        attn_score.float().masked_fill(attn_mask[None, :, :, None], -65000).type_as(attn_score)
                    )
                else:
                    attn_score = attn_score.float().masked_fill(attn_mask[None, :, :, None], -1e30).type_as(attn_score)
            elif attn_mask.dim() == 3:
                if self.dtype == torch.float16:
                    attn_score = attn_score.float().masked_fill(attn_mask[:, :, :, None], -65000).type_as(attn_score)
                else:
                    attn_score = attn_score.float().masked_fill(attn_mask[:, :, :, None], -1e30).type_as(attn_score)

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        # Mask heads if we want to
        if head_mask is not None:
            attn_prob = attn_prob * head_mask

        # compute attention vector
        attn_vec = torch.einsum("ijbn,jbnd->ibnd", (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        # linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            outputs = [hidden_states + attn_out]
        else:
            # residual connection + layer normalization
            outputs = [self.layer_norm(hidden_states + attn_out)]

        if output_attentions:
            outputs.append(attn_prob)

        return outputs


class RelPartialLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, layer_norm_epsilon=1e-5, **kwargs):
        super().__init__()

        self.dec_attn = RelPartialLearnableMultiHeadAttn(
            n_head, d_model, d_head, dropout, layer_norm_epsilon=layer_norm_epsilon, **kwargs
        )
        self.pos_ff = PositionwiseFF(
            d_model, d_inner, dropout, pre_lnorm=kwargs.get("pre_lnorm"), layer_norm_epsilon=layer_norm_epsilon
        )

    def forward(self, dec_inp, r, encoder_hidden_states, dec_attn_mask=None, action_mems=None, state_mems=None,
                head_mask=None, output_attentions=False):
        attn_outputs = self.dec_attn(
            dec_inp,
            r,
            encoder_hidden_states,
            attn_mask=dec_attn_mask,
            action_mems=action_mems,
            state_mems=state_mems,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        ff_output = self.pos_ff(attn_outputs[0])

        outputs = [ff_output] + attn_outputs[1:]

        return outputs


class AdaptiveEmbedding(nn.Module):
    def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1, sample_softmax=False,
                 padding_token_idx=14):
        super().__init__()

        self.n_token = n_token
        self.d_embed = d_embed

        self.cutoffs = cutoffs + [n_token]
        self.div_val = div_val
        self.d_proj = d_proj

        self.emb_scale = d_proj ** 0.5
        self.emb_layers = nn.ModuleList()

        assert not sample_softmax, "Deprecated 'sample_softmax'==True"

        if div_val == 1:
            self.emb_layers.append(nn.Embedding(n_token, d_embed, padding_idx=padding_token_idx))
            if d_proj != d_embed:
                self.emb_layers.append(nn.Linear(d_embed, d_proj, bias=False))
        else:
            raise ValueError("Deprecated 'div_val' != 1")

    def forward(self, inp):
        if self.div_val == 1:
            embed = inp
            for layer in self.emb_layers:
                embed = layer(embed)
        else:
            raise ValueError("Deprecated 'div_val' != 1")

        embed.mul_(self.emb_scale)

        return embed


class TransfoXLPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = TransfoXLConfig
    base_model_prefix = "transformer"

    def _init_weight(self, weight):
        if self.config.init == "uniform":
            nn.init.uniform_(weight, -self.config.init_range, self.config.init_range)
        elif self.config.init == "normal":
            nn.init.normal_(weight, 0.0, self.config.init_std)

    def _init_bias(self, bias):
        nn.init.constant_(bias, 0.0)

    def _init_weights(self, m):
        """Initialize the weights."""
        classname = m.__class__.__name__
        if classname.find("Linear") != -1:
            if hasattr(m, "weight") and m.weight is not None:
                self._init_weight(m.weight)
            if hasattr(m, "bias") and m.bias is not None:
                self._init_bias(m.bias)
        elif classname.find("AdaptiveEmbedding") != -1:
            if hasattr(m, "emb_projs"):
                for i in range(len(m.emb_projs)):
                    if m.emb_projs[i] is not None:
                        nn.init.normal_(m.emb_projs[i], 0.0, self.config.proj_init_std)
        elif classname.find("Embedding") != -1:
            if hasattr(m, "weight"):
                self._init_weight(m.weight)
        elif classname.find("ProjectedAdaptiveLogSoftmax") != -1:
            if hasattr(m, "cluster_weight") and m.cluster_weight is not None:
                self._init_weight(m.cluster_weight)
            if hasattr(m, "cluster_bias") and m.cluster_bias is not None:
                self._init_bias(m.cluster_bias)
            if hasattr(m, "out_projs"):
                for i in range(len(m.out_projs)):
                    if m.out_projs[i] is not None:
                        nn.init.normal_(m.out_projs[i], 0.0, self.config.proj_init_std)
        elif classname.find("LayerNorm") != -1:
            if hasattr(m, "weight"):
                nn.init.normal_(m.weight, 1.0, self.config.init_std)
            if hasattr(m, "bias") and m.bias is not None:
                self._init_bias(m.bias)
        else:
            if hasattr(m, "r_emb"):
                self._init_weight(m.r_emb)
            if hasattr(m, "r_w_bias"):
                self._init_weight(m.r_w_bias)
            if hasattr(m, "r_r_bias"):
                self._init_weight(m.r_r_bias)
            if hasattr(m, "r_bias"):
                self._init_bias(m.r_bias)

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, layer: Optional[int] = -1):
        """
        Resize input token embeddings matrix of the model if new_num_tokens != config.vocab_size. Take care of tying
        weights embeddings afterwards if the model class has a `tie_weights()` method.

        Arguments:

            new_num_tokens: (`optional`) int:
                New number of tokens in the embedding matrix. Increasing the size will add newly initialized vectors at
                the end. Reducing the size will remove vectors from the end. If not provided or None: does nothing and
                just returns a pointer to the input tokens ``torch.nn.Embeddings`` Module of the model.
            layer: (`optional`) int:
                Layer of the `AdaptiveEmbedding` where the resizing should be done. Per default the last layer will be
                resized. Be aware that when resizing other than the last layer, you have to ensure that the new
                token(s) in the tokenizer are at the corresponding position.

        Return: ``torch.nn.Embeddings`` Pointer to the input tokens Embeddings Module of the model
        """
        base_model = getattr(self, self.base_model_prefix, self)  # get the base model if needed

        if new_num_tokens is None:
            return self.get_input_embeddings()

        new_num_tokens_layer, layer = self._get_new_num_tokens_layer(new_num_tokens, layer)
        assert new_num_tokens_layer > 0, "The size of the new embedding layer cannot be 0 or less"
        model_embeds = base_model._resize_token_embeddings(new_num_tokens_layer, layer)

        # Update base model and current model config
        self.config.vocab_size = new_num_tokens
        base_model.vocab_size = new_num_tokens
        base_model.n_token = new_num_tokens

        new_embedding_shapes = self._get_embedding_shapes()
        self._resize_cutoffs(new_num_tokens, new_num_tokens_layer, new_embedding_shapes, layer)

        # Tie weights again if needed
        self.tie_weights()

        return model_embeds

    def _get_new_num_tokens_layer(self, new_num_tokens, layer):
        embeddings = self.get_input_embeddings()
        if layer == -1:
            layer = len(embeddings.emb_layers) - 1
        assert 0 <= layer <= len(embeddings.emb_layers) - 1

        new_num_tokens_layer = (
                new_num_tokens
                - sum([emb.weight.shape[0] for emb in embeddings.emb_layers[:layer]])
                - sum([emb.weight.shape[0] for emb in embeddings.emb_layers[layer + 1:]])
        )
        return new_num_tokens_layer, layer

    def _get_embedding_shapes(self):
        embeddings = self.get_input_embeddings()
        return [emb.weight.shape[0] for emb in embeddings.emb_layers]

    def _resize_token_embeddings(self, new_num_tokens, layer=-1):
        embeddings = self.get_input_embeddings()
        if new_num_tokens is None:
            return embeddings
        new_embeddings_layer = self._get_resized_embeddings(embeddings.emb_layers[layer], new_num_tokens)
        embeddings.emb_layers[layer] = new_embeddings_layer

        self.set_input_embeddings(embeddings)

        return self.get_input_embeddings()

    def _resize_cutoffs(self, new_num_tokens, new_emb_size, new_embedding_shapes, layer):
        embeddings = self.get_input_embeddings()

        for i in range(layer, len(embeddings.cutoffs)):
            embeddings.cutoffs[i] = sum(new_embedding_shapes[: i + 1])

        embeddings.cutoff_ends = [0] + embeddings.cutoffs
        embeddings.n_token = new_num_tokens

        self.config.cutoffs = embeddings.cutoffs[:-1]

        return embeddings.cutoffs


@dataclass
class TransfoXLModelOutput(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        mems (:obj:`List[torch.FloatTensor]` of length :obj:`config.n_layers`):
            Contains pre-computed hidden-states (key and values in the attention blocks). Can be used (see :obj:`mems`
            input) to speed up sequential decoding. The token ids which have their past given to this model should not
            be passed as input ids as they have already been computed.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: torch.FloatTensor
    action_mems: List[torch.FloatTensor] = None
    state_mems: List[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class TransfoXLLMHeadModelOutput(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        losses (:obj:`torch.FloatTensor` of shape `(batch_size, sequence_length-1)`, `optional`, returned when ``labels`` is provided)
            Language modeling losses (not reduced).
        prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token after SoftMax).
        mems (:obj:`List[torch.FloatTensor]` of length :obj:`config.n_layers`):
            Contains pre-computed hidden-states (key and values in the attention blocks). Can be used (see :obj:`mems`
            input) to speed up sequential decoding. The token ids which have their past given to this model should not
            be passed as input ids as they have already been computed.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    losses: Optional[torch.FloatTensor] = None
    prediction_scores: torch.FloatTensor = None
    action_mems: List[torch.FloatTensor] = None
    state_mems: List[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

    @property
    def logits(self):
        # prediction scores are the output of the adaptive softmax, see
        # the file `modeling_transfo_xl_utilities`. Since the adaptive
        # softmax returns the log softmax value, `self.prediction_scores`
        # are strictly speaking not exactly `logits`, but behave the same
        # way logits do.
        return self.prediction_scores


class StateEncoderConfig(TransfoXLConfig):
    def __init__(self,
                 cutoffs=None,
                 mem_len=100,
                 div_val=1,
                 n_layer=2,
                 vocab_size=15,
                 d_model=768,
                 d_embed=100,
                 d_inner=512,
                 attn_type=0,
                 use_object_embeddings=False,
                 **kwargs):
        cutoffs = cutoffs or []

        super().__init__(
            cutoffs=cutoffs,
            mem_len=mem_len,
            div_val=div_val,
            n_layer=n_layer,
            vocab_size=vocab_size,
            d_model=d_model,
            d_embed=d_embed,
            d_inner=d_inner,
            attn_type=attn_type,
            **kwargs
        )

        self.use_object_embeddings = use_object_embeddings


TRANSFO_XL_START_DOCSTRING = r"""

    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Parameters:
        config (:class:`~transformers.TransfoXLConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""

TRANSFO_XL_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.TransfoXLTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        mems (:obj:`List[torch.FloatTensor]` of length :obj:`config.n_layers`):
            Contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model (see
            :obj:`mems` output below). Can be used to speed up sequential decoding. The token ids which have their mems
            given to this model should not be passed as :obj:`input_ids` as they have already been computed.
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare Bert Model transformer outputting raw hidden-states without any specific head on top.",
    TRANSFO_XL_START_DOCSTRING,
)
class StateEncoderDecoder(TransfoXLPreTrainedModel, Registrable):
    @classmethod
    def from_params(
            cls: Type[T],
            params: Params,
            constructor_to_call: Callable[..., T] = None,
            constructor_to_inspect: Union[Callable[..., T], Callable[[T], None]] = None,
            **extras,
    ) -> T:
        config = StateEncoderConfig(**params.as_dict())

        return cls(config)

    def __init__(self, config):
        super().__init__(config)

        self.n_token = config.vocab_size
        self.use_object_embeddings = config.use_object_embeddings
        self.d_embed = config.d_embed
        self.d_model = config.d_model
        self.n_head = config.n_head
        self.d_head = config.d_head

        self.word_emb = AdaptiveEmbedding(
            config.vocab_size, config.d_embed, config.d_model, config.cutoffs, div_val=config.div_val
        )

        self.drop = nn.Dropout(config.dropout)

        self.n_layer = config.n_layer
        self.mem_len = config.mem_len
        self.attn_type = config.attn_type

        if not config.untie_r:
            self.r_w_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
            self.r_r_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))

        self.layers = nn.ModuleList()
        if config.attn_type == 0:  # the default attention
            for i in range(config.n_layer):
                self.layers.append(
                    RelPartialLearnableDecoderLayer(
                        config.n_head,
                        config.d_model,
                        config.d_head,
                        config.d_inner,
                        config.dropout,
                        dropatt=config.dropatt,
                        pre_lnorm=config.pre_lnorm,
                        r_w_bias=None if config.untie_r else self.r_w_bias,
                        r_r_bias=None if config.untie_r else self.r_r_bias,
                        layer_norm_epsilon=config.layer_norm_epsilon,
                    )
                )
        else:  # learnable embeddings and absolute embeddings are not used in our pretrained checkpoints
            raise NotImplementedError  # Removed them to avoid maintaining dead code

        self.same_length = config.same_length
        self.clamp_len = config.clamp_len

        if self.attn_type == 0:  # default attention
            self.pos_emb = PositionalEmbedding(self.d_model)
        else:  # learnable embeddings and absolute embeddings
            raise NotImplementedError  # Removed these to avoid maintaining dead code - They are not used in our pretrained checkpoint

        if self.use_object_embeddings:
            self.obj_proj_layer = torch.nn.Sequential(
                torch.nn.Linear(config.d_model, config.d_model, bias=False)
            )

        self.init_weights()

    def get_input_embeddings(self):
        return self.word_emb

    def set_input_embeddings(self, new_embeddings):
        self.word_emb = new_embeddings

    def backward_compatible(self):
        self.sample_softmax = -1

    def reset_memory_length(self, mem_len):
        self.mem_len = mem_len

    def _prune_heads(self, heads):
        logger.info("Head pruning is not implemented for Transformer-XL model")
        pass

    def init_mems(self, bsz, is_encoder=False):
        if self.mem_len > 0:
            mems = []
            num_iter = 1 if is_encoder else self.n_layer

            for i in range(num_iter):
                empty = torch.zeros(self.mem_len, bsz, self.config.d_model, dtype=self.dtype, device=self.device)
                mems.append(empty)

            return mems
        else:
            return None

    def _update_mems(self, hids, mems, mlen, qlen):
        # does not deal with None
        if mems is None:
            return None

        # mems is not None
        assert len(hids) == len(mems), "len(hids) != len(mems)"

        # There are `mlen + qlen` steps that can be cached into mems
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen)
            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hids)):
                cat = torch.cat([mems[i], hids[i]], dim=0)
                new_mems.append(cat[beg_idx:end_idx].detach())

        return new_mems

    def create_object_embeddings(self, embeddings: torch.FloatTensor, indices: torch.LongTensor):
        val_indices = indices != -100
        mask_indices = ~val_indices
        indices = indices.clone()
        indices[indices == -100] = 0
        # This is 'gather' magic. The idea is that we want to index the 2nd dimension of the embeddings tensor
        # so that we can extract the gold object embeddings
        obj_emb = torch.gather(embeddings, 2,
                               indices.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, self.config.hidden_size)).squeeze(2)
        # because we are also extracting invalid positions for masked values, we zero them out
        obj_emb.masked_fill_(mask_indices.unsqueeze(-1), 0.0)

        return obj_emb

    @add_start_docstrings_to_model_forward(TRANSFO_XL_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="transfo-xl-wt103",
        output_type=TransfoXLModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
            self,
            encoder_hidden_states,
            decoder_input_ids=None,
            action_mems=None,
            state_mems=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            object_embeddings=None
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # the original code for Transformer-XL used shapes [len, bsz] but we want a unified interface in the library
        # so we transpose here from shape [bsz, len] to shape [len, bsz]
        if decoder_input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif decoder_input_ids is not None:
            decoder_input_ids = decoder_input_ids.transpose(0, 1).contiguous()
            encoder_hidden_states = encoder_hidden_states.transpose(0, 1).contiguous()
            qlen, bsz = decoder_input_ids.size()
        elif inputs_embeds is not None:
            inputs_embeds = inputs_embeds.transpose(0, 1).contiguous()
            encoder_hidden_states = encoder_hidden_states.transpose(0, 1).contiguous()
            qlen, bsz = inputs_embeds.shape[0], inputs_embeds.shape[1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if action_mems is None:
            action_mems = self.init_mems(bsz)

        if state_mems is None:
            state_mems = self.init_mems(bsz, is_encoder=True)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads] (a head_mask for each layer)
        # and head_mask is converted to shape [num_hidden_layers x qlen x klen x bsz x n_head]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
                head_mask = head_mask.expand(self.n_layer, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            head_mask = head_mask.to(
                dtype=self.dtype
            )  # switch to float if need + fp16 compatibility
        else:
            head_mask = [None] * self.n_layer

        if inputs_embeds is not None:
            word_emb = inputs_embeds
        else:
            word_emb = self.word_emb(decoder_input_ids)

        if object_embeddings is not None and self.use_object_embeddings:
            object_embeddings = self.obj_proj_layer(object_embeddings)
            word_emb = word_emb + object_embeddings.transpose(1, 0)
            # word_emb = torch.cat([word_emb, object_embeddings.transpose(1, 0)])

        mlen = action_mems[0].size(0) if action_mems is not None else 0
        klen = mlen + qlen
        if self.same_length:
            all_ones = word_emb.new_ones((qlen, klen), dtype=torch.uint8)
            mask_len = klen - self.mem_len
            if mask_len > 0:
                mask_shift_len = qlen - mask_len
            else:
                mask_shift_len = qlen
            dec_attn_mask = (torch.triu(all_ones, 1 + mlen) + torch.tril(all_ones, -mask_shift_len))[:, :, None]  # -1
        else:
            dec_attn_mask = torch.triu(word_emb.new_ones((qlen, klen), dtype=torch.uint8), diagonal=mlen)[
                            :, :, None]

        hids = []
        attentions = [] if output_attentions else None
        if self.attn_type == 0:  # default
            pos_seq = torch.arange(klen - 1, -1, -1.0, device=word_emb.device, dtype=word_emb.dtype)
            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)
            pos_emb = self.pos_emb(pos_seq)

            core_out = self.drop(word_emb)
            pos_emb = self.drop(pos_emb)

            for i, layer in enumerate(self.layers):
                hids.append(core_out)
                a_mems_i = None if action_mems is None else action_mems[i]
                # we only keep memories for a single layer because we just have the last layer of the encoder
                s_mems_i = None if state_mems is None else state_mems[0]

                layer_outputs = layer(
                    core_out,
                    pos_emb,
                    encoder_hidden_states,
                    dec_attn_mask=dec_attn_mask,
                    action_mems=a_mems_i,
                    state_mems=s_mems_i,
                    head_mask=head_mask[i],
                    output_attentions=output_attentions,
                )
                core_out = layer_outputs[0]
                if output_attentions:
                    attentions.append(layer_outputs[1])
        else:  # learnable embeddings and absolute embeddings
            raise NotImplementedError  # Removed these to avoid maintaining dead code - They are not used in our pretrained checkpoint

        core_out = self.drop(core_out)

        new_action_mems = self._update_mems(hids, action_mems, mlen, qlen)
        # we assume we have a single layer here so we wrap the encoder hidden states
        new_state_mems = self._update_mems([encoder_hidden_states], state_mems, mlen, qlen)

        if output_hidden_states:
            # Add last layer and transpose to library standard shape [bsz, len, hidden_dim]
            hids.append(core_out)
            hids = tuple(t.transpose(0, 1).contiguous() for t in hids)
        else:
            hids = None
        if output_attentions:
            # Transpose to library standard shape [bsz, n_heads, query_seq_len, key_seq_len]
            attentions = tuple(t.permute(2, 3, 0, 1).contiguous() for t in attentions)
        # We transpose back here to shape [bsz, len, hidden_dim]
        core_out = core_out.transpose(0, 1).contiguous()

        if not return_dict:
            return tuple(v for v in [core_out, new_action_mems, new_state_mems, hids, attentions] if v is not None)

        return TransfoXLModelOutput(
            last_hidden_state=core_out[:, -qlen:],
            action_mems=new_action_mems,
            state_mems=new_state_mems,
            hidden_states=hids,
            attentions=attentions,
        )


if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "true"
    model = StateEncoderDecoder.from_params(Params({
        "d_model": 768
    }))

    action_mems = None
    state_mems = None

    for i in range(5):
        print(f"Step {i}")
        out = model(encoder_hidden_states=torch.randn(5, 9, 768), decoder_input_ids=torch.randint(13, (5, 9)),
                    action_mems=action_mems, state_mems=state_mems)

        action_mems = out.action_mems
        state_mems = out.state_mems
