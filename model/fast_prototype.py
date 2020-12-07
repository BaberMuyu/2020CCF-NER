import torch.nn as nn

from module.encoder import WordE, TextEncoder
from module.attn_rel import RelPositionEmbedding
from module.fusion import FusionE, FLAT, RelPosFusion, LinearFusion
from module.output import LabelE, OutputPoint, OutputOBIE


class FastPrototype(nn.Module):
    def __init__(self, config):
        super(FastPrototype, self).__init__()
        self.config = config
        self.layer_list = []
        self.text_encoder = TextEncoder(self.config.encoder)
        self.layer_list.append(self.text_encoder)

        if self.config.fusion.fusion in (FusionE.flat, FusionE.relpos):
            self.pe = RelPositionEmbedding(self.config.data.max_len, self.config.fusion.dim_pos)

        if self.config.fusion.fusion == FusionE.flat:
            self.fusion = FLAT(self.config.fusion)
        elif self.config.fusion.fusion == FusionE.relpos:
            self.fusion = RelPosFusion(self.config.fusion)
        elif self.config.fusion.fusion == FusionE.linear:
            self.fusion = LinearFusion(self.config.fusion)
        self.layer_list.append(self.fusion)

        if self.config.output.label == LabelE.point:
            self.output = OutputPoint(self.config.output)
        elif self.config.output.label in (LabelE.Nobie, LabelE.Tobie, LabelE.Cobie):
            self.output = OutputOBIE(self.config.output)
        self.layer_list.append(self.output)

        self.params = {}

    def get_params(self):
        if not self.params:
            for layer in self.layer_list:
                for key, value in layer.get_params().items():
                    if key not in self.params:
                        self.params[key] = []
                    self.params[key].extend(value)
        return self.params

    def detach_ptm(self, flag):
        self.text_encoder.detach_ptm(flag)

    def forward(self, inputs, en_decode=True):
        # encoder
        encoder_inputs = {'text': inputs['text'],
                          'mask': inputs['mask'],
                          }
        if self.config.encoder.worde in (WordE.w2v, WordE.cat, WordE.attn):
            encoder_inputs['pos_emb'] = self.pe
            encoder_inputs['word_text'] = inputs['word_text']
            encoder_inputs['word_mask'] = inputs['word_mask']
            encoder_inputs['word_pos_b'] = inputs['word_pos_b']
            encoder_inputs['word_pos_e'] = inputs['word_pos_e']
            encoder_inputs['word_indice'] = inputs['word_indice']
            encoder_inputs['part_size'] = inputs['part_size']
        encoder_outputs = self.text_encoder(encoder_inputs)

        # fusion
        if self.config.fusion.fusion == FusionE.flat:
            fusion_inputs = {'char_word_vec': encoder_outputs['char_word_vec'],
                             'char_word_mask': encoder_outputs['char_word_mask'],
                             'char_word_s': encoder_outputs['char_word_s'],
                             'char_word_e': encoder_outputs['char_word_e'],
                             'part_size': inputs['part_size'],
                             'pos_emb_layer': self.pe}
        elif self.config.fusion.fusion == FusionE.relpos:
            fusion_inputs = {'text_vec': encoder_outputs['text_vec'],
                             'mask': inputs['mask'],
                             'pos_emb_layer': self.pe}
        elif self.config.fusion.fusion == FusionE.linear:
            fusion_inputs = {'text_vec': encoder_outputs['text_vec']}
        else:
            fusion_inputs = None
        fusion_outputs = self.fusion(fusion_inputs)

        # output
        output_inputs = {'text_vec': fusion_outputs['text_vec'],
                         'mask': inputs['mask']}
        result = self.output(output_inputs, en_pred=en_decode)

        return result

    def cal_loss(self, preds, targets, mask):
        loss_ = self.output.cal_loss(preds, targets, mask)
        return {'back': loss_, 'show': loss_}

    def find_entity(self, text, pred):
        return self.output.find_entity(text, pred)
