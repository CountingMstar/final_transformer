"""
@author : Hyunwoong
@when : 2019-10-24
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn
from conf import *

from models.layers.layer_norm import LayerNorm
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.position_wise_feed_forward import PositionwiseFeedForward


class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.enc_dec_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(
            d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob
        )
        self.norm3 = LayerNorm(d_model=d_model)
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, dec, enc, t_mask, s_mask):
        a = torch.zeros(1).to(device)
        # 1. compute self attention
        _x = dec
        dec = self.norm1(dec)
        x = self.self_attention(q=dec, k=dec, v=dec, mask=t_mask)

        # 2. add and norm
        x = self.dropout1(x)
        x = x + _x * a

        if enc is not None:
            # 3. compute encoder - decoder attention
            _x = x
            x = self.norm2(x)
            x = self.enc_dec_attention(q=x, k=enc, v=enc, mask=s_mask)

            # 4. add and norm
            x = self.dropout2(x)
            x = x + _x * a

        # 5. positionwise feed forward network
        _x = x
        x = self.norm3(x)
        x = self.ffn(x)

        # 6. add and norm
        x = self.dropout3(x)
        x = x + _x * a
        return x