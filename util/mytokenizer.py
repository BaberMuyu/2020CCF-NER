import warnings
from transformers import BertTokenizer
from typing import Optional, Union


def is_chinese(uchar):
    """判断一个unicode是否是汉字"""
    if u'\u4e00' <= uchar <= u'\u9fa5':
        return True
    else:
        return False


def have_chinese(string):
    for s in string:
        if is_chinese(s):
            return True
    return False


class MyBertTokenizer(BertTokenizer):
    def __init__(
            self,
            vocab_file,
            do_lower_case=True,
            do_basic_tokenize=True,
            never_split=['[unused1]'],
            unk_token="[UNK]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            mask_token="[MASK]",
            tokenize_chinese_chars=True,
            **kwargs
    ):
        super(MyBertTokenizer, self).__init__(
            vocab_file,
            do_lower_case=True,
            do_basic_tokenize=True,
            never_split=['[unused1]', '[unused2]', '[unused3]', '[unused4]', '[unused5]', '[unused6]'],
            unk_token="[UNK]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            mask_token="[MASK]",
            tokenize_chinese_chars=True,
            **kwargs)

    def my_decode(self, tokens):
        tokens = self.convert_ids_to_tokens(tokens)
        tokens = " ".join(tokens).replace("##", "").strip()
        tokens = tokens.split(' ')
        # for i, t in enumerate(tokens):
        #     if t == '[unused1]':
        #         tokens[i] = ' '
        #     elif t == '[unused2]':
        #         tokens[i] = '“'
        #     elif t == '[unused3]':
        #         tokens[i] = '”'
        return tokens

    def my_encode(
            self,
            text,
            text_pair=None,
            add_special_tokens: bool = True,
            padding: Union[bool, str] = False,
            truncation: Union[bool, str] = False,
            max_length: Optional[int] = None,
            stride: int = 0,
            return_tensors=None,
            **kwargs
    ):
        def _add_user_sign(string):
            new_string = ''
            index = 0
            chinese_index = -2
            while index < len(string):
                s = string[index]
                if is_chinese(s):
                    chinese_index = index
                if s in {' ', '\xa0', '\u3000'} and chinese_index == index - 1:  # 将前一个字为中文的空格转为[unused1]，并且忽略其他空格
                    new_string += ' [unused1] '
                elif s == '“':
                    new_string += ' [unused2] '
                elif s == '”':
                    new_string += ' [unused3] '
                else:
                    new_string += s
                index += 1
            return new_string

        text = _add_user_sign(text)
        return super(MyBertTokenizer, self).encode(text=text,
                                                   text_pair=text_pair,
                                                   add_special_tokens=add_special_tokens,
                                                   padding=padding,
                                                   truncation=truncation,
                                                   max_length=max_length,
                                                   stride=stride,
                                                   return_tensors=return_tensors,
                                                   **kwargs)

    def get_token_map(self, text):
        text_encoded = self.my_encode(text, add_special_tokens=False)
        text_decoded = self.my_decode(text_encoded)
        if len(text_encoded) != len(text_decoded):
            warnings.warn('token map')
        decode2raw_map = [0] * len(text_decoded)
        text_index = 0
        decode_index = 0

        while True:
            sub_str = text[text_index]
            sub_str_encode = self.my_encode(sub_str, add_special_tokens=False)
            if text_decoded[decode_index] in {'[unused1]'}:
                decode2raw_map[decode_index] = text_index
                decode_index += 1
                text_index += 1
            elif not sub_str_encode:
                text_index += 1
            elif text_decoded[decode_index] in {'[unused2]', '[unused3]'}:
                decode2raw_map[decode_index] = text_index
                decode_index += 1
                text_index += 1
            elif text_decoded[decode_index] == '[UNK]':  # "多个encode为一个UNK"
                unknow_start = text_index
                unknow_end = text_index
                while True:
                    if unknow_end == len(text) - 1:
                        decode2raw_map[decode_index] = unknow_start
                        text_index = unknow_end + 1
                        decode_index += 1
                        break
                    unknow_end += 1
                    sub_str = text[unknow_start: unknow_end + 1]
                    sub_str_encode = self.my_encode(sub_str, add_special_tokens=False)
                    if len(sub_str_encode) >= 2 and sub_str_encode[0] == 100:
                        decode2raw_map[decode_index] = unknow_start
                        text_index = unknow_end if text[unknow_end - 1] != ' ' else (unknow_end - 1)
                        decode_index += 1
                        break
            elif len(sub_str_encode) > 1:  # 一个encode为多个
                sub_str_decode = self.my_decode(sub_str_encode)
                for _char in sub_str_decode:
                    if _char == text_decoded[decode_index]:
                        decode2raw_map[decode_index] = text_index
                        decode_index += 1
                    else:
                        warnings.warn('token map 1')
                text_index += 1
            elif len(text_decoded[decode_index]) == 1:  # 一对一
                decode2raw_map[decode_index] = text_index
                text_index += 1
                decode_index += 1
            elif len(text_decoded[decode_index]) > 1:  # 除了UNK, user sign情况外，多个字符encode为1个(英文单词， 数字组合) 或多对多（日语）
                # flag = False
                # sub_len = 0
                # if re.match("^[A-Za-z0-9_-]*$", text_decoded[decode_index]):
                #     sub_len = len(text_decoded[decode_index])
                #     if text[text_index:text_index + sub_len].lower() == text_decoded[decode_index]:
                #         flag = True
                # if not flag:
                #     sub_len = 2
                sub_len = 2
                while True:
                    sub_str_encode = self.my_encode(text[text_index:text_index + sub_len], add_special_tokens=False)
                    sub_str_decode = self.my_decode(sub_str_encode)
                    if sub_str_decode[0] == text_decoded[decode_index] or "".join(sub_str_decode) == text_decoded[decode_index]:
                        break
                    elif len(sub_str_decode) > len(text_decoded[decode_index]):
                        warnings.warn('token map 1')
                        sub_len += 1
                        # break
                    else:
                        sub_len += 1
                decode2raw_map[decode_index] = text_index
                text_index += sub_len
                decode_index += 1
            else:
                warnings.warn('token map')
                text_index += 1
            if text_index >= len(text) or decode_index >= len(text_decoded):
                break

        if decode2raw_map[-1] == 0 and len(decode2raw_map) > 1:
            warnings.warn('token map')

        raw2decode_map = [0] * len(text)
        for i, j in enumerate(decode2raw_map):
            if raw2decode_map[j] == 0:
                raw2decode_map[j] = i
        last_j = 0
        for i, j in enumerate(raw2decode_map):
            if j != last_j and j != 0:
                last_j = j
            raw2decode_map[i] = last_j
        decode2raw_map = [-1] + decode2raw_map + [len(text)]
        raw2decode_map = [i + 1 for i in raw2decode_map]
        return decode2raw_map, raw2decode_map
