# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)  # layer, 6, norm

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model  # 256
        self.nhead = nhead  # 8

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        ## src: b * hidden_dim * 12 * 20 ,mask:B * 12 * 20 ,query_embedding 7 * 32  ,其中7是最大的车道线的数量   pos_embed: : B *hidden_dim * 12 * 20
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1) ## B * hidden_dim * 240 =>240 * B * hidden_dim

        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)  ## 同上

        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)  # 7 * b* 32

        mask = mask.flatten(1)  # b* hw

        tgt = torch.zeros_like(query_embed)  ## tgt  7 * b* 32

        memory, weights = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        ## memory 240 * B * 32
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        # 2 * 7 * B * 32   ## c是不动的
        ## return 2 * B * 7 * 32 , B * 32 * 12 * 20 , weight： B * 240 * 240
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w), weights


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:

            output, weights = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output, weights


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,

                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        # src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
        #                       key_padding_mask=src_key_padding_mask)[0]
        src2, weights = self.self_attn(q, k, value=src, attn_mask=src_mask,
                                       key_padding_mask=src_key_padding_mask)

        src = src + self.dropout1(src2)

        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))

        src = src + self.dropout2(src2)

        src = self.norm2(src)

        return src, weights

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):

        q = k = self.with_pos_embed(tgt, query_pos)

        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]

        tgt = tgt + self.dropout1(tgt2)

        tgt = self.norm1(tgt)


        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]

        tgt = tgt + self.dropout2(tgt2)

        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))

        tgt = tgt + self.dropout3(tgt2)

        tgt = self.norm3(tgt)

        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Allen_transformer(nn.Module):
    def __init__(self,row_dim = 512,col_dim = 512,nhead = 8,num_row_layer=2,num_col_layer=2,dim_feedward=1024,dropout = 0.1,activate = "relu",normalize_before=False):
        super().__init__()
        row_layer = TransformerEncoderLayer(row_dim,nhead,dim_feedward,dropout,activate,normalize_before)
        row_norm = nn.LayerNorm(row_dim) if normalize_before else None
        self.row_encoder = TransformerEncoder(row_layer, num_row_layer, row_norm)  # layer, 6, norm
        col_layer = TransformerEncoderLayer(col_dim,nhead,dim_feedward,dropout,activate,normalize_before)
        col_norm = nn.LayerNorm(col_dim) if normalize_before else None
        self.col_encoder = TransformerEncoder(col_layer, num_col_layer, col_norm)  # layer, 6, norm
        self._reset_parameters()
        self.nhead = nhead
        ###############从这里往下是新添加的部分

        # decoder_layer = TransformerDecoderLayer(36, nhead, dim_feedward,
        #                                         dropout, activate, normalize_before)
        # decoder_norm = nn.LayerNorm(36)
        # self.decoder = TransformerDecoder(decoder_layer, 2, decoder_norm,
        #                                   return_intermediate=True)
        #
        # self._reset_parameters()
        #
        # self.d_model = 36  # 256
        # self.nhead = nhead  # 8

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # def forward(self, src, mask,pos_embed,query_embed):# 用decoder的时候才需要这个
    def forward(self, src, mask,pos_embed):


        '''

        :param src: the size is b * f * c * h * w 需要变为 单词个数 * b * hidden_dim的形式
        :param mask: the size is b * f*  h * w
        :param pos_embed: information of postion,size is b * f * c * h * w
        位置信息在这里生成好了
        :return:
        '''
        ## src: b * hidden_dim * 12 * 20 ,mask:B * 12 * 20 ,query_embedding 7 * 32  ,其中7是最大的车道线的数量   pos_embed: : B *hidden_dim * 12 * 20
        bs, f, c, h, w = src.shape
        row_src =  src.permute(0,2,4,1,3).reshape(bs,c*w,f*h)
        row_src = row_src.permute(2,0,1)
        ## the size is need to be b * f * h=>b * fh
        #need = torch.ones(mask.shape[-2:],dtype=torch.uint8)
        # row_mask = mask.all(dim=-1).flatten(1)
        row_mask = ((mask.int().sum(dim=-1))== mask.shape[-1]).flatten(1)
        row_pos = pos_embed.permute(0, 2, 4, 1, 3).reshape(bs, -1, f*h).permute(2,0,1)  ## the size is b * hidden_dim * w * f * h => b * new_hidden * f * h
        #################col #################################33
        col_src = src.permute(0,2,3,1,4).reshape(bs,c*h,f*w)
        col_src = col_src.permute(2,0,1)
        ## the size is need to be b * f * w =>b * fw
        col_mask =((mask.int().sum(dim=-2))== mask.shape[-2]).flatten(1)
        col_pos = pos_embed.permute(0, 2, 3, 1, 4).reshape(bs, -1, f*w).permute(2,0,1)  # the size is b * new_hidden * f * w
        # src = src.flatten(2).permute(2, 0, 1)  ## B * hidden_dim * 240 =>240 * B * hidden_dim
        row_memory,row_weights = self.row_encoder(row_src,src_key_padding_mask=row_mask, pos=row_pos) ## the size is fh * b * cw
        row_memory = row_memory.reshape(f,h,bs,c,w).permute(2,0,3,1,4)  ## bs,f,c,h,w
# w        ## 串联需要添加下面的代码
#         col_src = row_memory.permute(0, 2, 3, 1, 4).reshape(bs, c * h, f * w)
#         col_src = col_src.permute(2, 0, 1)

        col_memory,col_weights = self.col_encoder(col_src, src_key_padding_mask=col_mask, pos=col_pos) ## the size is fw * b * ch
        col_memory = col_memory.reshape(f,w,bs,c,h).permute(2,0,3,4,1)
        ## 串行这么返回
        #return col_memory,row_weights,col_weights
        ## 然而我们想要结合这部分的内容
        ## 并行的话这么返回
        return col_memory+row_memory,row_weights,col_weights
        # memory = (col_memory+row_memory).reshape(bs,c,f*h*w).permute(2,0,1)
        #
        #
        # query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)  # 7 * b* 32
        #
        # mask = mask.flatten(1)  # b* hw
        #
        # tgt = torch.zeros_like(query_embed)  ## tgt  7 * b* 32
        #
        # # memory, weights = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        # ## memory 240 * B * 32
        # # print(pos_embed.size())
        #
        # # print(pos_embed.size())
        # pos_embed = pos_embed.permute(0,2,1,3,4).flatten(2).permute(2, 0, 1)
        # # print(pos_embed.size())
        # hs = self.decoder(tgt, memory, memory_key_padding_mask=mask.byte(),
        #                   pos=pos_embed, query_pos=query_embed)
        # # 2 * 7 * B * 32   ## c是不动的
        # ## return 2 * B * 7 * 32 , B * 32 * 12 * 20 , weight： B * 240 * 240
        # return hs.transpose(1, 2)# memory.permute(1, 2, 0).view(bs, c, h, w)

class Allen_transformer2(nn.Module):
    '''
    串行的代码
    '''
    def __init__(self,row_dim = 512,col_dim = 512,nhead = 8,num_row_layer=2,num_col_layer=2,dim_feedward=1024,dropout = 0.1,activate = "relu",normalize_before=False):
        super().__init__()
        row_layer = TransformerEncoderLayer(row_dim,nhead,dim_feedward,dropout,activate,normalize_before)
        row_norm = nn.LayerNorm(row_dim) if normalize_before else None
        self.row_encoder = TransformerEncoder(row_layer, num_row_layer, row_norm)  # layer, 6, norm
        col_layer = TransformerEncoderLayer(col_dim,nhead,dim_feedward,dropout,activate,normalize_before)
        col_norm = nn.LayerNorm(col_dim) if normalize_before else None
        self.col_encoder = TransformerEncoder(col_layer, num_col_layer, col_norm)  # layer, 6, norm
        self._reset_parameters()
        self.nhead = nhead


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # def forward(self, src, mask,pos_embed,query_embed):# 用decoder的时候才需要这个
    def forward(self, src, mask,pos_embed):


        '''

        :param src: the size is b * f * c * h * w 需要变为 单词个数 * b * hidden_dim的形式
        :param mask: the size is b * f*  h * w
        :param pos_embed: information of postion,size is b * f * c * h * w
        位置信息在这里生成好了
        :return:
        '''
        ## src: b * hidden_dim * 12 * 20 ,mask:B * 12 * 20 ,query_embedding 7 * 32  ,其中7是最大的车道线的数量   pos_embed: : B *hidden_dim * 12 * 20
        bs, f, c, h, w = src.shape
        row_src =  src.permute(0,2,4,1,3).reshape(bs,c*w,f*h)
        row_src = row_src.permute(2,0,1)
        ## the size is need to be b * f * h=>b * fh
        #need = torch.ones(mask.shape[-2:],dtype=torch.uint8)
        # row_mask = mask.all(dim=-1).flatten(1)
        row_mask = ((mask.int().sum(dim=-1))== mask.shape[-1]).flatten(1)
        row_pos = pos_embed.permute(0, 2, 4, 1, 3).reshape(bs, -1, f*h).permute(2,0,1)  ## the size is b * hidden_dim * w * f * h => b * new_hidden * f * h
        #################col #################################33
        col_src = src.permute(0,2,3,1,4).reshape(bs,c*h,f*w)
        col_src = col_src.permute(2,0,1)
        ## the size is need to be b * f * w =>b * fw
        col_mask =((mask.int().sum(dim=-2))== mask.shape[-2]).flatten(1)
        col_pos = pos_embed.permute(0, 2, 3, 1, 4).reshape(bs, -1, f*w).permute(2,0,1)  # the size is b * new_hidden * f * w
        # src = src.flatten(2).permute(2, 0, 1)  ## B * hidden_dim * 240 =>240 * B * hidden_dim
        row_memory,row_weights = self.row_encoder(row_src,src_key_padding_mask=row_mask, pos=row_pos) ## the size is fh * b * cw
        row_memory = row_memory.reshape(f,h,bs,c,w).permute(2,0,3,1,4)  ## bs,f,c,h,w
#         ## 串联需要添加下面的代码
        col_src = row_memory.permute(0, 2, 3, 1, 4).reshape(bs, c * h, f * w)
        col_src = col_src.permute(2, 0, 1)

        col_memory,col_weights = self.col_encoder(col_src, src_key_padding_mask=col_mask, pos=col_pos) ## the size is fw * b * ch
        col_memory = col_memory.reshape(f,w,bs,c,h).permute(2,0,3,4,1)
        ## 串行这么返回
        return col_memory,row_weights,col_weights
        ## 然而我们想要结合这部分的内容
        ## 并行的话这么返回
        # return col_memory+row_memory,row_weights,col_weights
        # memory = (col_memory+row_memory).reshape(bs,c,f*h*w).permute(2,0,1)
        #
        #
        # query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)  # 7 * b* 32
        #
        # mask = mask.flatten(1)  # b* hw
        #
        # tgt = torch.zeros_like(query_embed)  ## tgt  7 * b* 32
        #
        # # memory, weights = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        # ## memory 240 * B * 32
        # # print(pos_embed.size())
        #
        # # print(pos_embed.size())
        # pos_embed = pos_embed.permute(0,2,1,3,4).flatten(2).permute(2, 0, 1)
        # # print(pos_embed.size())
        # hs = self.decoder(tgt, memory, memory_key_padding_mask=mask.byte(),
        #                   pos=pos_embed, query_pos=query_embed)
        # # 2 * 7 * B * 32   ## c是不动的
        # ## return 2 * B * 7 * 32 , B * 32 * 12 * 20 , weight： B * 240 * 240
        # return hs.transpose(1, 2)# memory.permute(1, 2, 0).view(bs, c, h, w)

class Allentransformer_oframe(nn.Module):

    def __init__(self,row_dim = 512,col_dim = 512,nhead = 8,num_row_layer=2,num_col_layer=2,dim_feedward=1024,dropout = 0.1,activate = "relu",normalize_before=False):
        super().__init__()
        row_layer = TransformerEncoderLayer(row_dim,nhead,dim_feedward,dropout,activate,normalize_before)
        row_norm = nn.LayerNorm(row_dim) if normalize_before else None
        self.row_encoder = TransformerEncoder(row_layer, num_row_layer, row_norm)  # layer, 6, norm
        col_layer = TransformerEncoderLayer(col_dim,nhead,dim_feedward,dropout,activate,normalize_before)
        col_norm = nn.LayerNorm(col_dim) if normalize_before else None
        self.col_encoder = TransformerEncoder(col_layer, num_col_layer, col_norm)  # layer, 6, norm
        self._reset_parameters()
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask,pos_embed):
        '''

        :param src: the size is b * f * c * h * w 需要变为 单词个数 * b * hidden_dim的形式
        :param mask: the size is b * f*  h * w
        :param pos_embed: information of postion,size is b * f * c * h * w
        位置信息在这里生成好了
        :return:
        '''
        ## src: b * hidden_dim * 12 * 20 ,mask:B * 12 * 20 ,query_embedding 7 * 32  ,其中7是最大的车道线的数量   pos_embed: : B *hidden_dim * 12 * 20
        bs, c, h, w = src.shape
        row_src =  src.permute(0,1,3,2).reshape(bs,-1,h).permute(2,0,1)
        ## the size is b * h
        # print(mask.size())
        row_mask = mask.all(dim=-1).flatten(1)
        # print(row_mask.size())
        ## b c h w => h* b * cw
        row_pos = pos_embed.permute(0,1,3,2).reshape(bs, -1, h).permute(2,0,1)
        #################col #################################33
        col_src = src.reshape(bs,-1,w).permute(2,0,1)
        ## the size is need to be b * f * w =>b * fw
        col_mask =((mask.int().sum(dim=-2))== mask.shape[-2]).flatten(1)
        col_pos = pos_embed.reshape(bs, -1, w).permute(2,0,1)  # the size is b * new_hidden * f * w
        row_memory,row_weights = self.row_encoder(row_src,src_key_padding_mask=row_mask, pos=row_pos) ## the size is h * b * cw
        row_memory = row_memory.reshape(h,bs,c,w).permute(1,2,0,3)

        col_memory,col_weights = self.col_encoder(col_src, src_key_padding_mask=col_mask, pos=col_pos) ## the size is fw * b * ch
        col_memory = col_memory.reshape(w,bs,c,h).permute(1,2,3,0)

        return col_memory+row_memory,row_weights,col_weights

def build_transformer(hidden_dim,
                      dropout,
                      nheads,
                      dim_feedforward,
                      enc_layers,
                      dec_layers,
                      pre_norm=False,
                      return_intermediate_dec=False):

    return Transformer(
        d_model=hidden_dim,
        dropout=dropout,
        nhead=nheads,
        dim_feedforward=dim_feedforward,
        num_encoder_layers=enc_layers,
        num_decoder_layers=dec_layers,
        normalize_before=pre_norm,
        return_intermediate_dec=return_intermediate_dec,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
