# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.modules import (
    LayerNorm,
    MultiheadAttention,
    TransformerSentenceEncoderLayer,
)


def init_bert_params(module):
    """
    Initialize the weights specific to the BERT Model.
    This overrides the default initializations depending on the specified arguments.
        1. If normal_init_linear_weights is set then weights of linear
           layer will be initialized using the normal distribution and
           bais will be set to the specified value.
        2. If normal_init_embed_weights is set then weights of embedding
           layer will be initialized using the normal distribution.
        3. If normal_init_proj_weights is set then weights of
           in_project_weight for MultiHeadAttention initialized using
           the normal distribution (to be validated).
    """

    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        module.in_proj_weight.data.normal_(mean=0.0, std=0.02)

# this is from T5
def relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
    ret = 0
    n = -relative_position
    if bidirectional:
        num_buckets //= 2
        ret += (n < 0).to(torch.long) * num_buckets  # mtf.to_int32(mtf.less(n, 0)) * num_buckets
        n = torch.abs(n)
    else:
        n = torch.max(n, torch.zeros_like(n))
    # now n is in the range [0, inf)

    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = n < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    val_if_large = max_exact + (
        torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
    ).to(torch.long)
    val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

    ret += torch.where(is_small, n, val_if_large)
    return nn.Parameter(ret, requires_grad=False)

class TransformerSentenceEncoder(nn.Module):
    """
    Implementation for a Bi-directional Transformer based Sentence Encoder used
    in BERT/XLM style pre-trained models.

    This first computes the token embedding using the token embedding matrix,
    position embeddings (if specified) and segment embeddings
    (if specified). After applying the specified number of
    TransformerEncoderLayers, it outputs all the internal states of the
    encoder as well as the final representation associated with the first
    token (usually CLS token).

    Input:
        - tokens: B x T matrix representing sentences
        - segment_labels: B x T matrix representing segment label for tokens

    Output:
        - a tuple of the following:
            - a list of internal model states used to compute the
              predictions where each tensor has shape B x T x C
            - sentence representation associated with first input token
              in format B x C.
    """

    def __init__(
        self,
        padding_idx: int,
        vocab_size: int,
        num_encoder_layers: int = 6,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        max_seq_len: int = 256,
        encoder_normalize_before: bool = False,
        embedding_normalize: bool = False,
        apply_bert_init: bool = False,
        activation_fn: str = "relu",
        embed_scale: float = None,
        rel_pos: bool = False,
        rel_pos_bins: int = 32,
        max_rel_pos: int = 128,
        export: bool = False,
    ) -> None:

        super().__init__()
        self.padding_idx = padding_idx
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        self.apply_bert_init = apply_bert_init
        self.embed_tokens = nn.Embedding(
                self.vocab_size, self.embedding_dim, self.padding_idx
            )
        self.embed_scale = embed_scale

        self.attn_scale_factor = 2
        self.num_attention_heads = num_attention_heads
        self.pos = nn.Embedding(self.max_seq_len + 1, self.embedding_dim)
        self.pos_proj = nn.Linear(self.embedding_dim, self.embedding_dim * 2)
        self.pos_scaling = float(self.embedding_dim / num_attention_heads * self.attn_scale_factor) ** -0.5 
        self.pos_ln = LayerNorm(self.embedding_dim, export=export)
        self.layers = nn.ModuleList(
            [
                TransformerSentenceEncoderLayer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=ffn_embedding_dim,
                    num_attention_heads=num_attention_heads,
                    dropout=self.dropout,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    activation_fn=activation_fn,
                    attn_scale_factor=self.attn_scale_factor,
                    export=export,
                    encoder_normalize_before=encoder_normalize_before,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        if embedding_normalize:
            self.emb_layer_norm = LayerNorm(self.embedding_dim, export=export)
        else:
            self.emb_layer_norm = None

        if encoder_normalize_before:
            self.emb_out_layer_norm = LayerNorm(self.embedding_dim, export=export)
        else:
            self.emb_out_layer_norm = None

        # Apply initialization of model params after building the model
        if self.apply_bert_init:
            self.apply(init_bert_params)

        self.rel_pos = rel_pos
        if self.rel_pos:
            assert rel_pos_bins % 2 == 0
            self.rel_pos_bins = rel_pos_bins
            self.max_rel_pos = max_rel_pos
            self.relative_attention_bias = nn.Embedding(self.rel_pos_bins + 1, self.num_attention_heads)
            seq_len = self.max_seq_len
            context_position = torch.arange(seq_len, dtype=torch.long)[:, None]
            memory_position = torch.arange(seq_len, dtype=torch.long)[None, :]
            relative_position = memory_position - context_position
            self.rp_bucket = relative_position_bucket(
                relative_position,
                num_buckets=self.rel_pos_bins,
                max_distance=self.max_rel_pos
            )
            # others to [CLS]
            self.rp_bucket[:, 0] = self.rel_pos_bins
            # [CLS] to others, Note: self.rel_pos_bins // 2 is not used in relative_position_bucket
            self.rp_bucket[0, :] = self.rel_pos_bins // 2

    def get_rel_pos_bias(self, x):
        # Assume the input is ordered. If your input token is permuted, you may need to update this accordingly
        seq_len = x.size(1)
        rp_bucket = self.rp_bucket[:seq_len, :seq_len]
        values = self.relative_attention_bias(rp_bucket)
        values = values.permute([2, 0, 1])
        return values.contiguous()

    def forward(
        self,
        tokens: torch.Tensor,
        last_state_only: bool = False,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # compute padding mask. This is needed for multi-head attention
        padding_mask = tokens.eq(self.padding_idx)
        if not padding_mask.any():
            padding_mask = None
        
        rel_pos_bias = self.get_rel_pos_bias(tokens) if self.rel_pos else None

        x = self.embed_tokens(tokens)
        if self.embed_scale is not None:
            x *= self.embed_scale

        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # account for padding while computing the representation
        if padding_mask is not None:
            x *= 1 - padding_mask.unsqueeze(-1).type_as(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        
        seq_len = x.size(0)
        # 0 is for other-to-cls 1 is for cls-to-other
        # Assume the input is ordered. If your input token is permuted, you may need to update this accordingly
        weight = self.pos_ln(self.pos.weight[:seq_len + 1, :])
        pos_q, pos_k = self.pos_proj(weight).reshape(seq_len + 1, 2, self.num_attention_heads, -1).permute(1, 2, 0, 3)
        pos_q *= self.pos_scaling
        abs_pos_bias = torch.bmm(pos_q, pos_k.transpose(1, 2))
        # p_0 \dot p_0 is cls to others
        cls_2_other = abs_pos_bias[:, 0, 0]
        # p_1 \dot p_1 is others to cls
        other_2_cls = abs_pos_bias[:, 1, 1]
        # offset 
        abs_pos_bias = abs_pos_bias[:, 1:, 1:]
        abs_pos_bias[:, :, 0] = other_2_cls.view(-1, 1)
        abs_pos_bias[:, 0, :] = cls_2_other.view(-1, 1)
        if rel_pos_bias is not None:
            abs_pos_bias += rel_pos_bias

        abs_pos_bias = abs_pos_bias.unsqueeze(0).expand(x.size(1), -1, -1, -1).reshape(-1, seq_len, seq_len)

        inner_states = []
        if not last_state_only:
            inner_states.append(x)
        for layer in self.layers:
            x = layer(x, self_attn_padding_mask=padding_mask, self_attn_bias=abs_pos_bias)
            if not last_state_only:
                inner_states.append(x)

        if self.emb_out_layer_norm is not None:
            x = self.emb_out_layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        sentence_rep = x[:, 0, :]

        if last_state_only:
            inner_states = [x]

        return inner_states, sentence_rep
