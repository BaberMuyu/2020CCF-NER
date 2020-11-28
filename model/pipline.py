import torch
import torch.nn as nn

from module.encoder import TextEncoder
from module.output import OutputType


class TypePrototype(nn.Module):
    def __init__(self, config):
        super(TypePrototype, self).__init__()
        self.config = config
        self.layer_list = []
        self.params = dict()

        self.encoder = TextEncoder(self.config.encoder)
        self.layer_list.append(self.encoder)
        self.output = OutputType(self.config.output_type)
        self.layer_list.append(self.output)

    def detach_ptm(self, flag):
        self.encoder.detach_ptm(flag)

    def get_params(self):
        if not self.params:
            for layer in self.layer_list:
                for key, value in layer.get_params().items():
                    if key not in self.params:
                        self.params[key] = []
                    self.params[key].extend(value)
        return self.params

    def forward(self, inputs, en_pred=True):
        # encoder
        encoder_inputs = {'text': inputs['text'],
                          'mask': inputs['mask'],
                          }
        encoder_outputs = self.encoder(encoder_inputs)
        output_type_inputs = {'text_vec': encoder_outputs['text_vec'],
                              'indice': inputs['indice']}
        preds = self.output(output_type_inputs, en_pred=en_pred)
        return preds

    def cal_loss(self, preds, targets):
        return self.output.cal_loss(preds, targets)

