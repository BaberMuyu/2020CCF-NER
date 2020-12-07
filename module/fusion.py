import torch
import torch.nn as nn
from enum import Enum

from module.attn_rel import TransfSelfEncoderRel

FusionE = Enum('Fusion', ('linear', 'flat', 'relpos'))


class FusionConfig(object):
    def __init__(self):
        self.fusion = FusionE.linear  # 'linear', 'flat', 'relpos'
        self.num_flat_layers = 1
        self.num_pos = 1
        self.dim_pos = 160
        self._hidden_size = 160  # out_size
        self.num_heads = 8
        self.scaled = False
        self.attn_dropout = 0.1
        self.hidden_dropout = 0.1
        self.en_ffd = True
        self.layer_norm_eps = 1e-12
        self.intermediate_size = 640
        self.dropout = 0.1

        self.in_feat_size = None
        self.out_feat_size = self.hidden_size

    @property
    def hidden_size(self):
        return self._hidden_size

    @hidden_size.setter
    def hidden_size(self, value):
        self._hidden_size = value
        self.out_feat_size = value

    def __str__(self):
        string = ""
        string += "fusion: {}\n".format(self.fusion)
        string += "hidden_size: {}\n".format(self.hidden_size)
        string += "scaled: {}\n".format(self.scaled)
        string += "en_ffd: {}\n".format(self.en_ffd)
        return string


class FLAT(nn.Module):
    def __init__(self, config):
        super(FLAT, self).__init__()
        self.config = config
        self.params = {'other': []}

        if self.config.in_feat_size != self.config.out_feat_size:
            self.adapter = nn.Linear(self.config.in_feat_size, self.config.out_feat_size)
            self.params['other'].extend([p for p in self.adapter.parameters()])
        self.encoder_layers = []
        for _ in range(self.config.num_flat_layers):
            encoder_layer = TransfSelfEncoderRel(self.config)
            self.encoder_layers.append(encoder_layer)
            self.params['other'].extend([p for p in encoder_layer.parameters()])
        self.encoder_layers = nn.ModuleList(self.encoder_layers)

    def get_params(self):
        return self.params

    def forward(self, inputs):
        char_word_vec = inputs['char_word_vec']
        char_word_mask = inputs['char_word_mask']
        char_word_s = inputs['char_word_s']
        char_word_e = inputs['char_word_e']
        part_size = inputs['part_size']

        pos_emb_layer = inputs['pos_emb_layer']
        if self.config.in_feat_size != self.config.out_feat_size:
            hidden = self.adapter(char_word_vec)
        else:
            hidden = char_word_vec
        pe_ss = pos_emb_layer(char_word_s.unsqueeze(dim=2) - char_word_s.unsqueeze(dim=1))
        pe_se = pos_emb_layer(char_word_s.unsqueeze(dim=2) - char_word_e.unsqueeze(dim=1))
        pe_es = pos_emb_layer(char_word_e.unsqueeze(dim=2) - char_word_s.unsqueeze(dim=1))
        pe_ee = pos_emb_layer(char_word_e.unsqueeze(dim=2) - char_word_e.unsqueeze(dim=1))
        for layer in self.encoder_layers:
            hidden = layer(hidden, [pe_ss, pe_se, pe_es, pe_ee], char_word_mask)
        char_vec, _ = hidden.split([part_size[0] + part_size[1], part_size[2]], dim=1)
        return {'text_vec': char_vec}


class RelPosFusion(nn.Module):
    def __init__(self, config):
        super(RelPosFusion, self).__init__()
        self.config = config
        self.params = {'other': []}
        if self.config.in_feat_size != self.config.out_feat_size:
            self.adapter = nn.Linear(self.config.in_feat_size, self.config.out_feat_size)
            self.params['other'].extend([p for p in self.adapter.parameters()])
        self.trans_rel = TransfSelfEncoderRel(self.config)
        self.params['other'].extend([p for p in self.trans_rel.parameters()])

    def get_params(self):
        return self.params

    def forward(self, inputs):
        text_vec = inputs['text_vec']
        mask = inputs['mask']
        pos_emb_layer = inputs['pos_emb_layer']
        if self.config.in_feat_size != self.config.out_feat_size:
            text_vec = self.adapter(text_vec)
        pos = torch.arange(0, text_vec.size(1)).long()
        pos = pos_emb_layer(pos.unsqueeze(dim=1) - pos.unsqueeze(dim=0))
        pos = pos.unsqueeze(dim=0)
        text_vec = self.trans_rel(text_vec, [pos], mask)
        return {'text_vec': text_vec}


class LinearFusion(nn.Module):
    def __init__(self, config):
        super(LinearFusion, self).__init__()
        self.config = config
        self.linear = nn.Linear(self.config.in_feat_size, self.config.out_feat_size)
        self.params = {'other': [p for p in self.linear.parameters()]}
        self.dropout = nn.Dropout(config.dropout)
        self.relu = nn.ReLU()

    def get_params(self):
        return self.params

    def forward(self, inputs):
        text_vec = inputs['text_vec']
        text_vec = self.linear(text_vec)
        text_vec = self.relu(text_vec)
        text_vec = self.dropout(text_vec)
        return {'text_vec': text_vec}
