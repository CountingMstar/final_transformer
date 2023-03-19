"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn


# class PostionalEncoding(nn.Module):
#     """
#     compute sinusoid encoding.
#     """

#     def __init__(self, d_model, max_len, device):
#         """
#         constructor of sinusoid encoding class

#         :param d_model: dimension of model
#         :param max_len: max sequence length
#         :param device: hardware device setting
#         """
#         super(PostionalEncoding, self).__init__()

#         # same size with input matrix (for adding with input matrix)
#         self.encoding = torch.zeros(max_len, d_model, device=device)
#         self.encoding.requires_grad = False  # we don't need to compute gradient

#         pos = torch.arange(0, max_len, device=device)
#         pos = pos.float().unsqueeze(dim=1)
#         # 1D => 2D unsqueeze to represent word's position

#         _2i = torch.arange(0, d_model, step=2, device=device).float()
#         # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
#         # "step=2" means 'i' multiplied with two (same with 2 * i)

#         self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
#         self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
#         # compute positional encoding to consider positional information of words

#     def forward(self, x):
#         # self.encoding
#         # [max_len = 512, d_model = 512]

#         batch_size, seq_len = x.size()
#         # [batch_size = 128, seq_len = 30]
#         return self.encoding[:seq_len, :]


class PostionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """

    def __init__(self, d_model, max_len, device):
        """
        constructor of sinusoid encoding class

        :param d_model: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super(PostionalEncoding, self).__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.ones(max_len, d_model, device=device) * (-1)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        unit_len = round(d_model * (1 / (max_len - 1)))
        for i in range(max_len):
            self.encoding[i, : unit_len * (i + 1)] = 1

    def forward(self, x):
        # self.encoding
        # [max_len = 512, d_model = 512]

        batch_size, seq_len = x.size()
        # [batch_size = 128, seq_len = 30]
        return self.encoding[:seq_len, :]
