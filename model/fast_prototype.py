import torch
import torch.nn as nn

from module.encoder import WordE, TextEncoder
from module.attn_rel import RelPositionEmbedding
from module.fusion import FusionE, FLAT, RelPosFusion, LinearFusion
from module.output import LabelE, OutputPoint, OutputOBIE, OutputType


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
        return self.output.cal_loss(preds, targets, mask)

    def find_entity(self, text, pred):
        return self.output.find_entity(text, pred)


class CasPrototype(nn.Module):
    def __init__(self, config):
        super(CasPrototype, self).__init__()
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

        self.output = OutputOBIE(self.config.output)
        self.layer_list.append(self.output)

        self.output_type = OutputType(self.config.output_type)
        self.layer_list.append(self.output_type)

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

    def forward(self, inputs, en_decode=True, mode='train'):
        entity_inputs = inputs['entity']
        # encoder
        fusion_outputs = self.get_text_vec(entity_inputs)

        # entity pred
        results = {}
        output_inputs = {'text_vec': fusion_outputs['text_vec'],
                         'mask': entity_inputs['mask']}
        results['entity'] = self.output(output_inputs, en_pred=en_decode)  # result = {'emission', 'pred'}

        # type pred
        if mode in ('train', 'eval'):
            type_inputs = inputs['type']
            output_type_inputs = {'text_vec': fusion_outputs['text_vec'],
                                  'indice': type_inputs['indice']}
            results['type_gt'] = self.output_type(output_type_inputs, en_pred=en_decode)
        if mode in ('eval', 'test'):
            entities = self.output.find_entity(results['entity']['pred'])
            indice = self.get_indice(entities, fusion_outputs['text_vec'].size(1))
            output_type_inputs = {'text_vec': fusion_outputs['text_vec'],
                                  'indice': indice}
            results['type_p'] = self.output_type(output_type_inputs)
        return results

    def get_text_vec(self, inputs):
        # encoder
        encoder_inputs = {'text': inputs['text'],
                          'mask': inputs['mask'],
                          }
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
        return fusion_outputs

    def pred_type(self, inputs, en_pred):
        entity_inputs = inputs['entity']
        type_inputs = inputs['type']
        fusion_outputs = self.get_text_vec(entity_inputs)
        output_type_inputs = {'text_vec': fusion_outputs['text_vec'],
                              'indice': type_inputs['indice']}
        result = self.output_type(output_type_inputs, en_pred=en_pred)
        return result

    def cal_loss(self, results, targets, mask):
        entity_loss = self.output.cal_loss(results['entity'], targets['entity'], mask)
        category_loss = self.output_type.cal_loss(results['type_gt'], targets['type'])
        _loss = entity_loss + category_loss * 0.1
        return _loss

    def find_entity(self, results):
        entities = self.output.find_entity(results['entity']['pred'])
        type_pred = []
        if 'type_gt' in results.keys():
            type_pred = results['type_gt']['pred']
        if 'type_p' in results.keys():
            entities_type = results['type_p']['pred']
            index = 0
            for line in entities:
                for e in line:
                    e[0] = entities_type[index]
                    index += 1
        return entities, type_pred

    def get_indice(self, batch, text_len):
        indice = []
        base = 0
        for line in batch:
            for e in line:
                indice.append(e[1]+base)
                indice.append(e[2]+base)
            base += text_len
        indice = torch.tensor(indice).long().cuda()
        return indice



class CombineModel(nn.Module):
    def __init__(self, config, crf_params):
        super(CombineModel, self).__init__()
        self.config = config
        self.output = OutputOBIE(self.config.output)
        self.params = {'other': self.output.get_params()['other'], 'crf': self.output.get_params()['crf']}
        self.params['crf'][0][:] = torch.tensor(crf_params[0]).cuda()
        self.params['crf'][1][:] = torch.tensor(crf_params[1]).cuda()
        self.params['crf'][2][:] = torch.tensor(crf_params[2]).cuda()

    def forward(self, inputs):
        output_inputs = {'text_vec': inputs['emission'],
                         'mask': inputs['mask']}
        result = self.output(output_inputs)
        return result

    def cal_loss(self, preds, targets, mask):
        return self.output.cal_loss(preds, targets, mask)

    def find_entity(self, text, pred):
        return self.output.find_entity(text, pred)


