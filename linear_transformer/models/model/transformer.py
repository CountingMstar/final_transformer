"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn

# from models.model.decoder import Decoder
# from models.model.encoder import Encoder

from models.blocks.encoder_layer import EncoderLayer
from models.blocks.decoder_layer import DecoderLayer
from models.embedding.transformer_embedding import TransformerEmbedding


class EncoderDecoder(nn.Module):
    def __init__(
        self,
        enc_voc_size,
        dec_voc_size,
        max_len,
        d_model,
        ffn_hidden,
        n_head,
        n_layers,
        drop_prob,
        device,
        k,
    ):
        super().__init__()
        self.encoder_emb = TransformerEmbedding(
            d_model=d_model,
            max_len=max_len,
            vocab_size=enc_voc_size,
            drop_prob=drop_prob,
            device=device,
            k=k,
        )
        self.decoder_emb = TransformerEmbedding(
            d_model=d_model,
            drop_prob=drop_prob,
            max_len=max_len,
            vocab_size=dec_voc_size,
            device=device,
            k=k,
        )

        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model=d_model,
                    ffn_hidden=ffn_hidden,
                    n_head=n_head,
                    drop_prob=drop_prob,
                )
                for _ in range(n_layers)
            ]
        )

        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(
                    d_model=d_model,
                    ffn_hidden=ffn_hidden,
                    n_head=n_head,
                    drop_prob=drop_prob,
                )
                for _ in range(n_layers)
            ]
        )

        self.linear = nn.Linear(d_model, dec_voc_size)

    def forward(self, src, s_mask, trg, trg_mask, src_mask):
        src = self.encoder_emb(src)
        trg = self.decoder_emb(trg)

        for i in range(len(self.encoder_layers)):
            encoder_layer = self.encoder_layers[i]
            src = encoder_layer(src, s_mask)

            decoder_layer = self.decoder_layers[i]
            trg = decoder_layer(trg, src, trg_mask, src_mask)

        # pass to LM head
        output = self.linear(trg)
        return output


class Transformer(nn.Module):
    def __init__(
        self,
        src_pad_idx,
        trg_pad_idx,
        trg_sos_idx,
        enc_voc_size,
        dec_voc_size,
        d_model,
        n_head,
        max_len,
        ffn_hidden,
        n_layers,
        drop_prob,
        device,
        k,
    ):
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.device = device

        # self.encoder = Encoder(
        #     d_model=d_model,
        #     n_head=n_head,
        #     max_len=max_len,
        #     ffn_hidden=ffn_hidden,
        #     enc_voc_size=enc_voc_size,
        #     drop_prob=drop_prob,
        #     n_layers=n_layers,
        #     device=device,
        #     k=k,
        # )

        # self.decoder = Decoder(
        #     d_model=d_model,
        #     n_head=n_head,
        #     max_len=max_len,
        #     ffn_hidden=ffn_hidden,
        #     dec_voc_size=dec_voc_size,
        #     drop_prob=drop_prob,
        #     n_layers=n_layers,
        #     device=device,
        #     k=k,
        # )

        self.encoder_decoder = EncoderDecoder(
            d_model=d_model,
            n_head=n_head,
            max_len=max_len,
            ffn_hidden=ffn_hidden,
            enc_voc_size=enc_voc_size,
            dec_voc_size=dec_voc_size,
            drop_prob=drop_prob,
            n_layers=n_layers,
            device=device,
            k=k,
        )

    def forward(self, src, trg):
        src_mask = self.make_pad_mask(src, src, self.src_pad_idx, self.src_pad_idx)

        src_trg_mask = self.make_pad_mask(trg, src, self.trg_pad_idx, self.src_pad_idx)

        trg_mask = self.make_pad_mask(
            trg, trg, self.trg_pad_idx, self.trg_pad_idx
        ) * self.make_no_peak_mask(trg, trg)

        # enc_src = self.encoder(src, src_mask)
        # output = self.decoder(trg, enc_src, trg_mask, src_trg_mask)
        output = self.encoder_decoder(src, src_mask, trg, trg_mask, src_trg_mask)

        return output

    def make_pad_mask(self, q, k, q_pad_idx, k_pad_idx):
        len_q, len_k = q.size(1), k.size(1)

        # batch_size x 1 x 1 x len_k
        k = k.ne(k_pad_idx).unsqueeze(1).unsqueeze(2)
        # batch_size x 1 x len_q x len_k
        k = k.repeat(1, 1, len_q, 1)

        # batch_size x 1 x len_q x 1
        q = q.ne(q_pad_idx).unsqueeze(1).unsqueeze(3)
        # batch_size x 1 x len_q x len_k
        q = q.repeat(1, 1, 1, len_k)

        mask = k & q
        return mask

    def make_no_peak_mask(self, q, k):
        len_q, len_k = q.size(1), k.size(1)

        # len_q x len_k
        mask = (
            torch.tril(torch.ones(len_q, len_k)).type(torch.BoolTensor).to(self.device)
        )

        return mask
