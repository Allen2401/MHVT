import torch
import torch.nn as nn
import io
from collections import defaultdict
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
# from PIL import Image
# from .dcn.deform_conv import DeformConv
from .Deformal2D import DeformConv2D
# import util.box_ops as box_ops
# from util.misc import NestedTensor, interpolate, nested_tensor_from_tensor_list
class Att_in(nn.Module):
    def __init__(self,frame_num,dim):
        super(Att_in, self).__init__()
        self.conv1_1 = nn.Conv2d(frame_num * dim,dim,kernel_size=1,bias = False)
        self.conv3_1 = nn.Conv2d(dim,dim,kernel_size=3,padding=1,stride=1)
        self.conv3_2 = nn.Conv2d(dim,dim,kernel_size=3,padding=1,stride=1)
        self.conv1_2 = nn.Conv2d(dim,frame_num,kernel_size=1,bias=False)

    def forward(self,input):
        '''
        we want to fuse the frame dimension
        :param input: the size is B * f * c * h * w
        :return: return the fused data the size is b * c * h * w
        '''
        b,f,c,h,w = input.size()
        f_att = input.reshape(b,f*c,h,w)
        f_att = self.conv1_1(f_att)
        f_att = self.conv3_1(f_att)
        f_att = self.conv3_2(f_att)
        f_att = self.conv1_2(f_att) ## the size is b,f,h,w
        f_att = torch.softmax(f_att,dim=1)
        ## the size is b * f * c * h * w
        output = torch.unsqueeze(f_att,dim=2) * input
        output = torch.sum(output,dim=1)
        return output

def _expand(tensor, length: int):
    return tensor.unsqueeze(1).repeat(1, int(length), 1, 1, 1).flatten(0, 1)


class MHAttentionMap(nn.Module):
    """
    this function is to get the relationship between the O(decoder output) and E(encoder output,after fusion)
    This is a 2D attention module, which only returns the attention softmax (no multiplication by value)"""

    def __init__(self, query_dim, hidden_dim, num_heads, dropout=0.0, bias=True):
        '''

        :param query_dim: the dim of input
        :param hidden_dim: the hidden dim can set
        :param num_heads:
        :param dropout:
        :param bias:
        '''
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.k_linear = nn.Linear(query_dim, hidden_dim, bias=bias)

        nn.init.zeros_(self.k_linear.bias)
        nn.init.zeros_(self.q_linear.bias)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.q_linear.weight)
        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

    def forward(self, q, k, mask: Optional[Tensor] = None):
        '''

        :param q: in lSTR,the size of q is b * num_query * hidden_dim
        :param k: the size of k is b * hidden_dim * h * w
        :param mask: is the valid mask
        :return:
        '''
        ## the size of q is b * instance_num * hidden_dim
        ## the size of k is b * c * h *  w
        q = self.q_linear(q) # b*num_query* hidden_dim
        k = F.conv2d(k, self.k_linear.weight.unsqueeze(-1).unsqueeze(-1), self.k_linear.bias)
        qh = q.view(q.shape[0], q.shape[1], self.num_heads, self.hidden_dim // self.num_heads)## b * num_query * n * c//n
        kh = k.view(k.shape[0], self.num_heads, self.hidden_dim // self.num_heads, k.shape[-2], k.shape[-1])## 对k进行同样的操作：b * n * c//n * h*w
        weights = torch.einsum("bqnc,bnchw->bqnhw", qh * self.normalize_fact, kh)
        if mask is not None:
            weights.masked_fill_(mask.unsqueeze(1).unsqueeze(1), float("-inf"))
        weights = F.softmax(weights.flatten(2), dim=-1).view_as(weights)
        weights = self.dropout(weights)
        return weights ## the size is b * q * n * h * w

class MaskHeadSmallConv(nn.Module):
    """
    Simple convolutional head, using group norm.
    Upsampling is done using a FPN approach
    ## fusion the backone infomation

    MaskHeadSmallConv(hidden_dim + nheads, [1024, 512, 256], hidden_dim)
    """

    def __init__(self, hidden_dim,num_head,fpn_dims, context_dim):
        super().__init__()
        '''
        hidden_dim is the channel num of e(encoder output), 96
        num_head is the number of head in attention between E(encoder output) and O(decoder output),8
        the frame num is the frame number
        the fpn_dims is the channel num of tensor of per layer of backbone output. LSTR:[16, 32, 64, 128]
        context_dim: mainly control the output channel num 
        ## 7 ,14 ,28,56,112,如果输入能够刚好保持这个大小的话就刚好是num_query 个输出图。
        ## 得是16的倍数，还得是3的倍数 64   32,16,8,4
        '''
        dim = hidden_dim + num_head
        inter_dims = [dim, context_dim // 2, context_dim // 4, context_dim // 8, context_dim // 16]
        self.lay1 = torch.nn.Conv2d(dim, dim, 3, padding=1)
        self.gn1 = torch.nn.GroupNorm(2, dim)  ## 就是说将dim分成8个组
        self.lay2 = torch.nn.Conv2d(dim, inter_dims[1], 3, padding=1)
        self.gn2 = torch.nn.GroupNorm(2, inter_dims[1])
        ##上面这些是个encoder用的，下面的是给融合用的
        self.lay3 = torch.nn.Conv2d(inter_dims[1], inter_dims[2], 3, padding=1)
        self.gn3 = torch.nn.GroupNorm(2, inter_dims[2])
        self.lay4 = torch.nn.Conv2d(inter_dims[2], inter_dims[3], 3, padding=1)
        self.gn4 = torch.nn.GroupNorm(2, inter_dims[3])
        self.gn5 = torch.nn.GroupNorm(2, inter_dims[4])
        self.conv_offset = torch.nn.Conv2d(inter_dims[3], 18, 1)#, bias=False)
        self.dcn = DeformConv2D(inter_dims[3],inter_dims[4], 3, padding=1)
        self.finalconv = torch.nn.Conv2d(inter_dims[4],1,kernel_size=1)

        self.dim = dim

        self.adapter1 = torch.nn.Conv2d(fpn_dims[0], inter_dims[1], 1)
        self.adapter2 = torch.nn.Conv2d(fpn_dims[1], inter_dims[2], 1)
        self.adapter3 = torch.nn.Conv2d(fpn_dims[2], inter_dims[3], 1)


        for name, m in self.named_modules():
            if name == "conv_offset":
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)
            else:
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_uniform_(m.weight, a=1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor, bbox_mask: Tensor, fpns: List[Tensor]):
        '''
        We want to fusion the information of the three part
        :param x: the tensor from encoder output,the size is b * hidden_dim * h * w
        :param bbox_mask:  the tensor from attention of O and E。the size is b * q * n * h * w  (the n is the num_head)
        :param fpns:  the tensor from backbone. the size is b *  c * hh * ww (hh and ww represent different layer has differnt size)
        [降16，降8，降4]这是一个信息融合，如果这些都需要Attention融合感觉十分麻烦。。
        :return:
        '''

        x = torch.cat([_expand(x, bbox_mask.shape[1]), bbox_mask.flatten(0, 1)], 1)
        x = self.lay1(x)
        x = self.gn1(x)
        x = F.relu(x)
        x = self.lay2(x)
        x = self.gn2(x)
        x = F.relu(x) ## c变成context_dim //2

        cur_fpn = self.adapter1(fpns[0])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest") ## 把x进行上采样
        x = self.lay3(x)
        x = self.gn3(x)
        x = F.relu(x)

        cur_fpn = self.adapter2(fpns[1])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay4(x)
        x = self.gn4(x)
        x = F.relu(x)

        cur_fpn = self.adapter3(fpns[2])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        # dcn for the last layer
        offset = self.conv_offset(x)
        x = self.dcn(x,offset)
        x = self.gn5(x)
        x = F.relu(x)  ## the size is bq * output_dim * h * w
        x = self.finalconv(x)
        return x

