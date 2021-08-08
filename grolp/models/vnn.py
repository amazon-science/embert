import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class SelfAttn(nn.Module):
    '''
    self-attention with learnable parameters
    '''

    def __init__(self, dhid):
        super().__init__()
        self.scorer = nn.Linear(dhid, 1)

    def forward(self, inp):
        scores = F.softmax(self.scorer(inp), dim=1)
        cont = scores.transpose(1, 2).bmm(inp).squeeze(1)
        return cont


class DotAttn(nn.Module):
    '''
    dot-attention (or soft-attention)
    '''

    def forward(self, inp, h):
        score = self.softmax(inp, h)
        return score.expand_as(inp).mul(inp).sum(1), score

    def softmax(self, inp, h):
        raw_score = inp.bmm(h.unsqueeze(2))
        score = F.softmax(raw_score, dim=1)
        return score


class ResnetVisualEncoder(nn.Module):
    '''
    visual encoder
    '''

    def __init__(self, dframe=2500):
        # we use the default parameters in the original ALFRED implementation
        super(ResnetVisualEncoder, self).__init__()
        self.dframe = dframe
        self.flattened_size = 64 * 7 * 7

        self.conv1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)
        self.fc = nn.Linear(self.flattened_size, self.dframe)
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(64)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))

        x = self.conv2(x)
        x = F.relu(self.bn2(x))

        x = x.view(-1, self.flattened_size)
        x = self.fc(x)

        return x


class MaskDecoder(nn.Module):
    '''
    mask decoder
    '''

    def __init__(self, dhid, pframe=300, hshape=(64, 7, 7)):
        super(MaskDecoder, self).__init__()
        self.dhid = dhid
        self.hshape = hshape
        self.pframe = pframe

        self.d1 = nn.Linear(self.dhid, hshape[0] * hshape[1] * hshape[2])
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn1 = nn.BatchNorm2d(16)
        self.dconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.dconv2 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.dconv1 = nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.d1(x))
        x = x.view(-1, *self.hshape)

        x = self.upsample(x)
        x = self.dconv3(x)
        x = F.relu(self.bn2(x))

        x = self.upsample(x)
        x = self.dconv2(x)
        x = F.relu(self.bn1(x))

        x = self.dconv1(x)
        x = F.interpolate(x, size=(self.pframe, self.pframe), mode='bilinear', align_corners=True)

        return x


class ConvFrameMaskDecoder(nn.Module):
    '''
    action decoder
    '''

    def __init__(self, emb, dframe, dhid, pframe=300,
                 attn_dropout=0., hstate_dropout=0., actor_dropout=0., input_dropout=0.,
                 teacher_forcing=False):
        super().__init__()
        demb = emb.weight.size(1)

        self.emb = emb
        self.pframe = pframe
        self.dhid = dhid
        self.vis_encoder = ResnetVisualEncoder(dframe=dframe)
        self.cell = nn.LSTMCell(dhid + dframe + demb, dhid)
        self.attn = DotAttn()
        self.input_dropout = nn.Dropout(input_dropout)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.hstate_dropout = nn.Dropout(hstate_dropout)
        self.actor_dropout = nn.Dropout(actor_dropout)
        self.go = nn.Parameter(torch.Tensor(demb))
        self.actor = nn.Linear(dhid + dhid + dframe + demb, demb)
        self.mask_dec = nn.Sequential(
            nn.Linear(dhid + dhid + dframe + demb, dhid + dhid + dframe + demb // 2), nn.ReLU(),
            nn.Linear(dhid + dhid + dframe + demb // 2, 119)
        )  # background (0) + objects (1~118)
        self.teacher_forcing = teacher_forcing
        self.h_tm1_fc = nn.Linear(dhid, dhid)

        nn.init.uniform_(self.go, -0.1, 0.1)

    def step(self, enc, frame, e_t, state_tm1):
        # previous decoder hidden state
        h_tm1 = state_tm1[0]

        # encode vision and lang feat
        vis_feat_t = self.vis_encoder(frame)
        lang_feat_t = enc  # language is encoded once at the start

        # attend over language
        weighted_lang_t, lang_attn_t = self.attn(self.attn_dropout(lang_feat_t), self.h_tm1_fc(h_tm1))

        # concat visual feats, weight lang, and previous action embedding
        inp_t = torch.cat([vis_feat_t, weighted_lang_t, e_t], dim=1)
        inp_t = self.input_dropout(inp_t)

        # update hidden state
        state_t = self.cell(inp_t, state_tm1)
        state_t = [self.hstate_dropout(x) for x in state_t]
        h_t = state_t[0]

        # decode action and mask
        cont_t = torch.cat([h_t, inp_t], dim=1)
        action_emb_t = self.actor(self.actor_dropout(cont_t))
        action_t = action_emb_t.mm(self.emb.weight.t())
        mask_t = self.mask_dec(cont_t)

        return action_t, mask_t, state_t, lang_attn_t

    def forward(self, enc, frames, gold=None, max_decode=150, state_0=None):
        max_t = gold.size(1) if self.training else min(max_decode, frames.shape[1])
        batch = enc.size(0)
        e_t = self.go.repeat(batch, 1)
        state_t = state_0

        actions = []
        masks = []
        attn_scores = []
        for t in range(max_t):
            action_t, mask_t, state_t, attn_score_t = self.step(enc, frames[:, t], e_t, state_t)
            masks.append(mask_t)
            actions.append(action_t)
            attn_scores.append(attn_score_t)
            if self.teacher_forcing and self.training:
                w_t = gold[:, t]
            else:
                w_t = action_t.max(1)[1]
            e_t = self.emb(w_t)

        results = {
            'out_action_low': torch.stack(actions, dim=1),
            'out_action_low_mask': torch.stack(masks, dim=1),
            'out_attn_scores': torch.stack(attn_scores, dim=1),
            'state_t': state_t
        }
        return results


# Thanks to the released code by Federico Landi et al.,
#  dynamic filters are easily exploited.
# see the repo below for more details of dynamic convolution.
#  https://github.com/aimagelab/DynamicConv-agent
######################################################################################################################
class ScaledDotAttn(nn.Module):
    def __init__(self, dim_key_in=1024, dim_key_out=128, dim_query_in=1024, dim_query_out=128):
        super().__init__()
        self.fc_key = nn.Linear(dim_key_in, dim_key_out)
        self.fc_query = nn.Linear(dim_query_in, dim_query_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, value, h):  # key: lang_feat_t_instr, query: h_tm1_instr
        key = F.relu(self.fc_key(value))
        query = F.relu(self.fc_query(h)).unsqueeze(-1)

        scale_1 = np.sqrt(key.shape[-1])
        scaled_dot_product = torch.bmm(key, query) / scale_1
        softmax = self.softmax(scaled_dot_product)
        element_wise_product = value * softmax
        weighted_lang_t_instr = torch.sum(element_wise_product, dim=1)

        return weighted_lang_t_instr, softmax.squeeze(-1)


class DynamicConvLayer(nn.Module):
    def __init__(self, dhid=512):
        super().__init__()
        self.head1 = nn.Linear(dhid, 512)
        self.head2 = nn.Linear(dhid, 512)
        self.head3 = nn.Linear(dhid, 512)
        self.filter_activation = nn.Tanh()

    def forward(self, frame, weighted_lang_t_instr):
        """ dynamic convolutional filters """
        df1 = self.head1(weighted_lang_t_instr)
        df2 = self.head2(weighted_lang_t_instr)
        df3 = self.head3(weighted_lang_t_instr)
        dynamic_filters = torch.stack([df1, df2, df3]).transpose(0, 1)
        dynamic_filters = self.filter_activation(dynamic_filters)
        dynamic_filters = F.normalize(dynamic_filters, p=2, dim=-1)

        """ attention map """
        frame = frame.view(frame.size(0), frame.size(1), -1)
        scale_2 = np.sqrt(frame.shape[1])  # torch.sqrt(torch.tensor(frame.shape[1], dtype=torch.double))
        attention_map = torch.bmm(frame.transpose(1, 2), dynamic_filters.transpose(-1, -2)) / scale_2
        attention_map = attention_map.reshape(attention_map.size(0), -1)

        return attention_map


######################################################################################################################


class ConvFrameMaskDecoderProgressMonitor(nn.Module):
    '''
    action decoder with subgoal and progress monitoring
    '''

    def __init__(self, emb, dframe, dhid, pframe=300,
                 attn_dropout=0., hstate_dropout=0., actor_dropout=0., input_dropout=0.,
                 teacher_forcing=False):
        super().__init__()
        demb = emb.weight.size(1)

        self.emb = emb
        self.pframe = pframe
        self.dhid = dhid
        self.vis_encoder = ResnetVisualEncoder(dframe=dframe)
        self.cell_goal = nn.LSTMCell(dhid + dframe + demb, dhid)
        self.cell_instr = nn.LSTMCell(dhid + dframe + demb, dhid)
        self.attn = DotAttn()
        self.input_dropout = nn.Dropout(input_dropout)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.hstate_dropout = nn.Dropout(hstate_dropout)
        self.actor_dropout = nn.Dropout(actor_dropout)
        self.go = nn.Parameter(torch.Tensor(demb))
        self.actor = nn.Linear(dhid + dhid + dframe + demb, demb)
        self.mask_dec = nn.Sequential(
            nn.Linear(dhid, dhid // 2), nn.ReLU(),
            nn.Linear(dhid // 2, 119)
        )
        self.teacher_forcing = teacher_forcing
        self.h_tm1_fc_goal = nn.Linear(dhid, dhid)
        self.h_tm1_fc_instr = nn.Linear(dhid, dhid)

        self.subgoal = nn.Linear(dhid + dhid + dframe + demb, 1)
        self.progress = nn.Linear(dhid + dhid + dframe + demb, 1)

        nn.init.uniform_(self.go, -0.1, 0.1)

        self.scale_dot_attn = ScaledDotAttn(dhid, 128, dhid, 128)
        self.dynamic_conv = DynamicConvLayer(dhid)

    def step(self, enc_goal, enc_instr, frame, e_t, state_tm1_goal, state_tm1_instr):
        # previous decoder hidden state (goal, instr decoder)
        h_tm1_goal = state_tm1_goal[0]
        h_tm1_instr = state_tm1_instr[0]

        # encode vision and lang feat (goal, instr decoder)
        lang_feat_t_goal = enc_goal  # language is encoded once at the start
        lang_feat_t_instr = enc_instr  # language is encoded once at the start

        # scaled dot product attention
        weighted_lang_t_goal, lang_attn_t_goal = self.scale_dot_attn(lang_feat_t_goal, h_tm1_goal)
        weighted_lang_t_instr, lang_attn_t_instr = self.scale_dot_attn(lang_feat_t_instr, h_tm1_instr)

        # dynamic convolution
        vis_feat_t_goal = self.dynamic_conv(frame, weighted_lang_t_goal)
        vis_feat_t_instr = self.dynamic_conv(frame, weighted_lang_t_instr)

        # concat visual feats, weight lang, and previous action embedding (goal decoder)
        inp_t_goal = torch.cat([vis_feat_t_goal, weighted_lang_t_goal, e_t], dim=1)
        inp_t_goal = self.input_dropout(inp_t_goal)

        # concat visual feats, weight lang, and previous action embedding (instr decoder)
        inp_t_instr = torch.cat([vis_feat_t_instr, weighted_lang_t_instr, e_t], dim=1)
        inp_t_instr = self.input_dropout(inp_t_instr)

        # update hidden state (goal decoder)
        state_t_goal = self.cell_goal(inp_t_goal, state_tm1_goal)
        state_t_goal = [self.hstate_dropout(x) for x in state_t_goal]
        h_t_goal, _ = state_t_goal[0], state_t_goal[1]

        # decode mask (goal decoder)
        cont_t_goal = h_t_goal  # torch.cat([h_t_goal, inp_t_goal], dim=1)
        mask_t = self.mask_dec(cont_t_goal)

        # update hidden state (instr decoder)
        state_t_instr = self.cell_instr(inp_t_instr, state_tm1_instr)
        state_t_instr = [self.hstate_dropout(x) for x in state_t_instr]
        h_t_instr, _ = state_t_instr[0], state_t_instr[1]

        # decode action (instr decoder)
        cont_t_instr = torch.cat([h_t_instr, inp_t_instr], dim=1)
        action_emb_t = self.actor(self.actor_dropout(cont_t_instr))
        action_t = action_emb_t.mm(self.emb.weight.t())

        # predict subgoals completed and task progress
        subgoal_t = F.sigmoid(self.subgoal(cont_t_instr))
        progress_t = F.sigmoid(self.progress(cont_t_instr))

        return action_t, mask_t, state_t_goal, state_t_instr, lang_attn_t_goal, lang_attn_t_instr, subgoal_t, progress_t

    def forward(self, enc_goal, enc_instr, frames, gold=None, max_decode=150, state_0_goal=None, state_0_instr=None):
        max_t = gold.size(1) if self.training else min(max_decode, frames.shape[1])
        batch = enc_instr.size(0)
        e_t = self.go.repeat(batch, 1)
        state_t_goal = state_0_goal
        state_t_instr = state_0_instr

        actions = []
        masks = []
        attn_scores_goal = []
        attn_scores_instr = []
        subgoals = []
        progresses = []
        for t in range(max_t):
            action_t, mask_t, state_t_goal, state_t_instr, attn_score_t_goal, attn_score_t_instr, subgoal_t, progress_t = self.step(
                enc_goal, enc_instr, frames[:, t], e_t, state_t_goal, state_t_instr)
            masks.append(mask_t)
            actions.append(action_t)
            attn_scores_goal.append(attn_score_t_goal)
            attn_scores_instr.append(attn_score_t_instr)
            subgoals.append(subgoal_t)
            progresses.append(progress_t)

            # find next emb
            if self.teacher_forcing and self.training:
                w_t = gold[:, t]
            else:
                w_t = action_t.max(1)[1]
            e_t = self.emb(w_t)

        results = {
            'out_action_low': torch.stack(actions, dim=1),
            'out_action_low_mask': torch.stack(masks, dim=1),
            'out_attn_scores_goal': torch.stack(attn_scores_goal, dim=1),
            'out_attn_scores_instr': torch.stack(attn_scores_instr, dim=1),
            'out_subgoal': torch.stack(subgoals, dim=1),
            'out_progress': torch.stack(progresses, dim=1),
            'state_t_goal': state_t_goal,
            'state_t_instr': state_t_instr,
        }
        return results
