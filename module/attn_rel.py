import torch
import torch.nn as nn
import math


class RelPositionEmbedding(nn.Module):
    def __init__(self, max_len, dim):
        super(RelPositionEmbedding, self).__init__()
        self.max_len = max_len
        num_embedding = max_len * 2 - 1
        half_dim = int(dim // 2)
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(-max_len + 1, max_len, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embedding, -1)
        if dim % 2 == 1:
            print('embedding dim is odd')
            emb = torch.cat([emb, torch.zeros(num_embedding, 1)], dim=1)
        self.emb = nn.Parameter(emb, requires_grad=False)
        self.dim = dim

    def forward(self, pos):
        pos = pos + (self.max_len - 1)
        pos_shape = pos.size()
        pos_emb = self.emb[pos.view(-1)]
        pos_emb = pos_emb.reshape(list(pos_shape) + [self.dim])
        return pos_emb


class RelPositionFusion(nn.Module):
    def __init__(self, config):
        super(RelPositionFusion, self).__init__()
        self.config = config
        self.pos_fusion_forward = nn.Linear(self.config.dim_pos * self.config.num_pos, self.config.hidden_size)

    def forward(self, pos):
        pe_4 = torch.cat(pos, dim=-1)
        rel_pos_embedding = nn.functional.relu(self.pos_fusion_forward(pe_4))
        return rel_pos_embedding


class MultiHeadAttentionRel(nn.Module):
    def __init__(self, hidden_size, num_heads, scaled=True, attn_dropout=None):
        super(MultiHeadAttentionRel, self).__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.per_head_size = self.hidden_size // self.num_heads
        self.scaled = scaled
        assert (self.per_head_size * self.num_heads == self.hidden_size)

        self.w_k = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_q = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_v = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_r = nn.Linear(self.hidden_size, self.hidden_size)
        # self.w_final = nn.Linear(self.hidden_size, self.hidden_size)
        self.u = nn.Parameter(torch.randn(self.num_heads, self.per_head_size), requires_grad=True)
        self.v = nn.Parameter(torch.randn(self.num_heads, self.per_head_size), requires_grad=True)

        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, key, query, value, pos, key_mask):
        key = self.w_k(key)
        query = self.w_q(query)
        value = self.w_v(value)
        rel_pos_embedding = self.w_r(pos)

        batch = key.size(0)

        # batch * seq_len * n_head * d_head
        key = torch.reshape(key, [batch, -1, self.num_heads, self.per_head_size])
        query = torch.reshape(query, [batch, -1, self.num_heads, self.per_head_size])
        value = torch.reshape(value, [batch, -1, self.num_heads, self.per_head_size])
        rel_pos_embedding = torch.reshape(rel_pos_embedding,
                                          list(rel_pos_embedding.size()[:3]) + [self.num_heads, self.per_head_size])

        # batch * n_head * seq_len * d_head
        key = key.transpose(1, 2)
        query = query.transpose(1, 2)
        value = value.transpose(1, 2)

        # batch * n_head * d_head * key_len
        key = key.transpose(-1, -2)

        u_for_c = self.u.unsqueeze(0).unsqueeze(-2)
        query_and_u_for_c = query + u_for_c
        A_C = torch.matmul(query_and_u_for_c, key)

        rel_pos_embedding_for_b = rel_pos_embedding.permute(0, 3, 1, 4, 2)
        query_for_b = query.view([batch, self.num_heads, query.size(2), 1, self.per_head_size])
        query_for_b_and_v_for_d = query_for_b + self.v.view(1, self.num_heads, 1, 1, self.per_head_size)
        B_D = torch.matmul(query_for_b_and_v_for_d, rel_pos_embedding_for_b).squeeze(-2)

        attn_score_raw = A_C + B_D

        # A
        # A_ = torch.matmul(query, key)

        # B
        # rel_pos_embedding_for_b = rel_pos_embedding.permute(0, 3, 1, 4, 2)
        # after above, rel_pos_embedding: batch * num_head * query_len * per_head_size * key_len
        # query_for_b = query.view([batch, self.num_heads, -1, 1, self.per_head_size])
        # B_ = torch.matmul(query_for_b, rel_pos_embedding_for_b).squeeze(-2)

        # D
        # rel_pos_embedding_for_d = rel_pos_embedding.unsqueeze(-2)
        # after above, rel_pos_embedding: batch * query_seq_len * key_seq_len * num_heads * 1 *per_head_size
        # v_for_d = self.v.unsqueeze(-1)
        # v_for_d: num_heads * per_head_size * 1
        # D_ = torch.matmul(rel_pos_embedding_for_d, v_for_d).squeeze(-1).squeeze(-1).permute(0, 3, 1, 2)

        # C
        # key: batch * n_head * d_head * key_len
        # u_for_c = self.u.unsqueeze(0).unsqueeze(-2)
        # u_for_c: 1(batch broadcast) * num_heads * 1 *per_head_size
        # C_ = torch.matmul(u_for_c, key)

        # att_score: Batch * num_heads * query_len * key_len
        # A, B C and D is exactly the shape
        # attn_score_raw = A_ + B_ + C_ + D_

        if self.scaled:
            attn_score_raw = attn_score_raw / math.sqrt(self.per_head_size)

        # mask = seq_len_to_mask(seq_len).bool().unsqueeze(1).unsqueeze(1)
        mask = key_mask.byte().unsqueeze(1).unsqueeze(1)
        attn_score_raw_masked = attn_score_raw.masked_fill(1-mask, -1e15)

        attn_score = nn.functional.softmax(attn_score_raw_masked, dim=-1)

        attn_score = self.dropout(attn_score)

        value_weighted_sum = torch.matmul(attn_score, value)

        result = value_weighted_sum.transpose(1, 2).contiguous(). \
            reshape(batch, -1, self.hidden_size)

        return result


class BertSelfOutput(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob, layer_norm_eps):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


def gelu(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initialy created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class BertIntermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = gelu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, intermediate_size, hidden_size, layer_norm_eps, hidden_dropout_prob):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class TransfAttenRel(nn.Module):
    def __init__(self, config):
        super(TransfAttenRel, self).__init__()
        self.config = config
        self.attn = MultiHeadAttentionRel(self.config.hidden_size,
                                          self.config.num_heads,
                                          scaled=self.config.scaled,
                                          attn_dropout=self.config.attn_dropout)
        self.attn_out = BertSelfOutput(self.config.hidden_size, self.config.hidden_dropout,
                                       self.config.layer_norm_eps)

    def forward(self, key, query, value, pos, seq_mask):
        attn_vec = self.attn(key, query, value, pos, seq_mask)
        attn_vec = self.attn_out(attn_vec, query)
        return attn_vec


class TransfFFD(nn.Module):
    def __init__(self, config):
        super(TransfFFD, self).__init__()
        self.config = config
        self.intermediate = BertIntermediate(self.config.hidden_size, self.config.intermediate_size)
        self.output = BertOutput(self.config.intermediate_size, self.config.hidden_size,
                                 self.config.layer_norm_eps, self.config.hidden_dropout)

    def forward(self, input_):
        hidden = self.intermediate(input_)
        hidden = self.output(hidden, input_)
        return hidden


class TransfEncoderRel(nn.Module):
    def __init__(self, config):
        super(TransfEncoderRel, self).__init__()
        self.config = config
        self.pos_fusion = RelPositionFusion(self.config)
        self.attn = TransfAttenRel(self.config)
        self.ffd = TransfFFD(self.config)

    def forward(self, key, query, value, pos, seq_mask):
        pos = self.pos_fusion(pos)
        attn_vec = self.attn(key, query, value, pos, seq_mask)
        ffd_vec = self.ffd(attn_vec)
        return ffd_vec


class TransfSelfEncoderRel(nn.Module):
    def __init__(self, config):
        super(TransfSelfEncoderRel, self).__init__()
        self.config = config
        self.pos_fusion = RelPositionFusion(self.config)
        self.attn = TransfAttenRel(self.config)
        if config.en_ffd:
            self.ffd = TransfFFD(self.config)

    def forward(self, hidden, pos, seq_mask):
        pos = self.pos_fusion(pos)
        vec = self.attn(hidden, hidden, hidden, pos, seq_mask)
        if self.config.en_ffd:
            vec = self.ffd(vec)
        return vec
