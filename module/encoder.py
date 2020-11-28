import torch
import torch.nn as nn
from transformers import BertModel
from enum import Enum

from .attn_rel import TransfEncoderRel

WordE = Enum('WordEncoding', ('none', 'w2v', 'cat', 'attn'))


class EncoderConfig(object):
    def __init__(self):
        self.worde = WordE.none
        self.dropout = 0.1

        # ptm
        self.ptm_model = 'hfl/chinese-roberta-wwm-ext'  # 'hfl/chinese-roberta-wwm-ext-large'
        self.ptm_feat_size = 768
        self.num_ptm_layers = 12

        # w2v
        self.w2v_feat_size = 300

        # attn
        self.num_wattn_layers = 2

        # commmon
        self.layer_norm_eps = 1e-12
        self.in_feat_size = None

    @property
    def out_feat_size(self):
        return self.ptm_feat_size

    def __str__(self):
        string = ""
        string += "worde: {}\n".format(self.worde)
        string += "ptm_model: {}\n".format(self.ptm_model)
        return string


class TextEncoder(nn.Module):
    def __init__(self, config):
        super(TextEncoder, self).__init__()
        self.config = config
        self.params = {'ptm': [], 'other': []}
        self.detach_ptm_flag = False

        self.bert = BertModel.from_pretrained(config.ptm_model, output_hidden_states=True)
        self.params['ptm'].extend([p for p in self.bert.parameters()])

        if config.worde != WordE.none:
            if config.worde == WordE.w2v:
                self.w2v_linear = nn.Linear(config.w2v_feat_size, config.ptm_feat_size)
                self.layer_norm = nn.LayerNorm(config.ptm_feat_size, eps=config.layer_norm_eps)
                self.params['other'].extend([p for p in self.w2v_linear.parameters()])
                self.params['other'].extend([p for p in self.layer_norm.parameters()])
            elif config.worde == WordE.attn:
                self.word_encode_layers = []
                for _ in range(config.num_wattn_layers):
                    encode_layer = TransfEncoderRel(config)
                    self.word_encode_layers.append(encode_layer)
                    self.params['other'].extend([p for p in encode_layer.parameters()])
                self.wrod_encode_layers = nn.ModuleList(self.word_encode_layers)
            elif config.worde == WordE.cat:
                self.cat_linear = nn.Linear(config.ptm_feat_size * 2, config.ptm_feat_size)
                self.layer_norm = nn.LayerNorm(config.ptm_feat_size, eps=config.layer_norm_eps)
                self.params['other'].extend([p for p in self.cat_linear.parameters()])
                self.params['other'].extend([p for p in self.layer_norm.parameters()])
            self.dropout = nn.Dropout(config.dropout)

    def get_params(self):
        return self.params

    def detach_ptm(self, flag):
        self.detach_ptm_flag = flag

    def get_bert_vec(self, text, text_mask, text_pos=None):
        if text_pos is None:
            _, _, text_vecs = self.bert(text, text_mask)
        else:
            _, _, text_vecs = self.bert(text, text_mask, position_ids=text_pos)
        text_vecs = list(text_vecs)
        if self.detach_ptm_flag:
            for i, vec in enumerate(text_vecs):
                text_vecs[i] = vec.detach()
        return text_vecs

    def forward(self, inputs):
        text, mask, = inputs['text'], inputs['mask']

        text_vecs = self.get_bert_vec(text, mask, inputs['pos'] if 'pos' in inputs else None)
        char_vec = text_vecs[self.config.num_ptm_layers]
        if self.config.worde != WordE.none:
            word_text, word_mask, word_indice = inputs['word_text'], inputs['word_mask'], inputs['word_indice']
            pos = torch.arange(0, text.size(1)).long().unsqueeze(dim=0).cuda()
            pos = pos * mask.long()
            char_s = char_e = pos
            word_s, word_e = inputs['word_pos_b'], inputs['word_pos_e']
            part_size = inputs['part_size']
            if self.config.worde == WordE.w2v:
                word_vec = self.w2v_linear(word_text)
                word_vec = self.dropout(word_vec)
                word_vec = self.layer_norm(word_vec)
            elif self.config.worde == WordE.cat:
                word_vec = torch.index_select(char_vec.view([-1, self.config.ptm_feat_size]), dim=0,
                                              index=word_indice)  # word_indice b * l * [s, e]
                word_vec = word_vec.reshape([text.size(0), -1, 2 * self.config.ptm_feat_size])
                word_vec = self.cat_linear(word_vec)  # B, L, F
                word_vec = self.dropout(word_vec)
                word_vec = self.layer_norm(word_vec)
            elif self.config.worde == WordE.attn:
                pe = inputs['pos_emb']
                pe_ss = pe(word_s.unsqueeze(dim=2) - char_s.unsqueeze(dim=1))
                pe_se = pe(word_s.unsqueeze(dim=2) - char_e.unsqueeze(dim=1))
                pe_es = pe(word_e.unsqueeze(dim=2) - char_s.unsqueeze(dim=1))
                pe_ee = pe(word_e.unsqueeze(dim=2) - char_e.unsqueeze(dim=1))

                # query = self.text_vecs[0](word_text)  # MASK标记
                query = torch.randn(list(word_text.size()) + [self.config.ptm_feat_size],
                                    requires_grad=False).cuda()  # MASK标记
                for layer_index in range(
                        self.config.num_ptm_layers - self.config.num_wattn_layers - self.config.num_ptm_layers):
                    key = value = text_vecs[layer_index]
                    query = self.word_encode_layers[layer_index](key, query, value, [pe_ss, pe_se, pe_es, pe_ee], mask)
                word_vec = query * word_mask.unsqueeze(dim=2)
            else:
                word_vec = None
            part0, part1_1 = torch.split(char_vec, part_size[:2], dim=1)
            part1_2, part2 = torch.split(word_vec, part_size[1:], dim=1)
            part1 = part1_1 + part1_2
            char_word_vec = torch.cat([part0, part1, part2], dim=1)

            part0, part1_1 = torch.split(mask, part_size[:2], dim=1)
            part1_2, part2 = torch.split(word_mask, part_size[1:], dim=1)
            part1 = part1_1 + part1_2
            char_word_mask = torch.cat([part0, part1, part2], dim=1)

            part0, part1_1 = torch.split(char_s, part_size[:2], dim=1)
            part1_2, part2 = torch.split(word_s, part_size[1:], dim=1)
            part1 = part1_1 + part1_2
            char_word_s = torch.cat([part0, part1, part2], dim=1)

            part0, part1_1 = torch.split(char_e, part_size[:2], dim=1)
            part1_2, part2 = torch.split(word_e, part_size[1:], dim=1)
            part1 = part1_1 + part1_2
            char_word_e = torch.cat([part0, part1, part2], dim=1)
            return {'char_word_vec': char_word_vec,
                    'char_word_mask': char_word_mask,
                    'char_word_s': char_word_s,
                    'char_word_e': char_word_e}
        else:
            return {'text_vec': char_vec}
