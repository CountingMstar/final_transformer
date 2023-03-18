"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn

# print all of tensor
torch.set_printoptions(profile="full")
# reset
# torch.set_printoptions(profile="default")

from models.embedding.positional_encoding import PostionalEncoding
from models.embedding.token_embeddings import TokenEmbedding
from models.embedding.autoencoder import AutoEncoder, LinearLayer
from models.layers.layer_norm import LayerNorm

from conf import device


class TransformerEmbedding(nn.Module):
    """
    token embedding + positional encoding (sinusoid)
    positional encoding can give positional information to network
    """

    def __init__(self, vocab_size, d_model, max_len, drop_prob, device, k):
        """
        class for word embedding that included positional information

        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model
        """
        # "Concatenate" Token, Position 임베딩 크기 조절 파라미터
        print("The size of PE is " + str(k))

        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PostionalEncoding(d_model, max_len, device)

        # self.cat_tok_emb = TokenEmbedding(vocab_size, d_model - k)
        # self.cat_pos_emb = PostionalEncoding(k, max_len, device)
        self.cat_tok_emb = TokenEmbedding(vocab_size, d_model)
        self.cat_pos_emb = PostionalEncoding(k, max_len, device)

        self.linearlayer = LinearLayer(d_model, k)
        self.drop_out = nn.Dropout(p=drop_prob)

        self.d_model = d_model
        self.device = device

    def expander(self, x):
        # normal
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        tok_batch_size, tok_sentence_size, tok_embedding_size = tok_emb.shape
        pos_sentence_size, pos_embedding_size = pos_emb.shape
        pos_emb = pos_emb.expand(tok_batch_size, pos_sentence_size, pos_embedding_size)

        # concatenate
        cat_tok_emb = self.cat_tok_emb(x)
        cat_pos_emb = self.cat_pos_emb(x)
        tok_batch_size, tok_sentence_size, tok_embedding_size = cat_tok_emb.shape
        pos_sentence_size, pos_embedding_size = cat_pos_emb.shape
        cat_pos_emb = cat_pos_emb.expand(
            tok_batch_size, pos_sentence_size, pos_embedding_size
        )
        return tok_emb, pos_emb, cat_tok_emb, cat_pos_emb

    ##########################auto encoder를 집어넣자##############################
    def forward(self, x):
        tok_emb, pos_emb, cat_tok_emb, cat_pos_emb = self.expander(x)
        # print("########transformer_emb#######")
        # print(tok_emb.shape)
        # print(tok_emb)
        # print(pos_emb.shape)
        # print(pos_emb)
        # print(cat_tok_emb.shape)
        # print(cat_pos_emb)
        model = SummationEmbedding(
            tok_emb,
            pos_emb,
            cat_tok_emb,
            cat_pos_emb,
            self.linearlayer,
            self.d_model,
            self.device,
        )
        """
        positional encoding type 결정
        """
        final_emb = model.summation()
        # final_emb = model.concatenate()
        # final_emb = model.linear()
        # final_emb = model.autoencoder()

        # return self.drop_out(tok_emb + pos_emb)
        # return self.drop_out(final_emb)
        return final_emb


class SummationEmbedding(TransformerEmbedding):
    def __init__(
        self,
        token_emb,
        positional_emb,
        cat_token_emb,
        cat_positional_emb,
        linearlayer,
        d_model,
        device,
    ):
        super(TransformerEmbedding, self).__init__()

        self.token_emb = token_emb
        self.positional_emb = positional_emb
        self.cat_token_emb = cat_token_emb
        self.cat_positional_emb = cat_positional_emb

        self.d_model = d_model
        self.norm = LayerNorm(d_model=d_model).to(device)
        self.norm2 = LayerNorm(d_model=2 * d_model).to(device)
        """
        from models.embedding.autoencoder import LinearLayer의
        linear layer 선언
        """
        self.linearlayer = linearlayer

    def summation(self):
        self.token_emb = self.norm(self.token_emb)
        self.positional_emb = self.norm(self.positional_emb)

        embedding = self.token_emb + self.positional_emb
        embedding = self.norm(embedding)
        return embedding

    # def summation(self):
    #     embedding = self.token_emb + self.positional_emb
    #     return embedding

    def concatenate(self):
        embedding = torch.cat([self.cat_token_emb, self.cat_positional_emb], 2)
        # print('#####CAT#####')
        # print(embedding.shape)
        return embedding

    def linear(self):
        """
        token embedding과 positional embedding을 결합하는 linear layer
        """
        residual_token = self.token_emb
        residual_positional = self.positional_emb

        self.token_emb = self.norm(self.token_emb)
        self.positional_emb = self.norm(self.positional_emb)

        embedding = torch.cat([self.token_emb, self.positional_emb], 2)
        # embedding = torch.cat([self.cat_token_emb, self.cat_positional_emb], 2)
        batch_size, sentence_size, embedding_size = embedding.shape
        embedding = embedding.view(batch_size * sentence_size, -1)
        embedding = self.norm2(embedding)

        """
        Residual Connection
        """
        # residual = torch.split(embedding, self.d_model, dim=1)[0]
        embedding = self.linearlayer(embedding)

        residual_token = residual_token.reshape(batch_size * sentence_size, -1)
        residual_positional = residual_positional.reshape(
            batch_size * sentence_size, -1
        )

        embedding = embedding + residual_token + residual_positional

        new_sentences_size, new_embedding_size = embedding.shape
        embedding = embedding.view(batch_size, sentence_size, new_embedding_size)
        return embedding

    def autoencoder(self):
        embedding = torch.cat([self.token_emb, self.positional_emb], 2)
        batch_size, sentence_size, embedding_size = embedding.shape
        embedding = embedding.view(batch_size * sentence_size, -1)
        self.auto_encoder = AutoEncoder(embedding).to(device)
        encoded, decoded = self.auto_encoder(embedding)
        embedding = embedding.view(batch_size, sentence_size, embedding_size)

        # 텐서를 비교하는 함수
        # print(torch.eq(a, embedding))
        return embedding
