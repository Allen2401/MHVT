import torch
import torchvision
import torch.nn as nn
import numpy as np
from model.transformer import TransformerEncoderLayer
import torch.nn.functional as F
class Tokenizer(nn.Module):
    def __init__(self, L, C,dynamic=False):
        super().__init__()
         # Code for adjusting the channel sizes in case C is not equal to CT
        self.feature = nn.Conv2d(C, C, kernel_size=1)
        if not dynamic :
            # use static weights to compute token coefficients.
            self.WA = nn.Linear(C, L)
            self.conv_token_coef = nn.Conv2d(C, L, kernel_size=1)
        else:
            # use previous tokens to compute a query weight, which is then used to compute token coefficients.
            self.WTR = nn.Linear(C,C)
        self.dynamic = dynamic
        self.C = C
        self.L = L

    def forward(self, X, tokens=None):
        '''

        :param X:
        :param tokens: the size of token is b * l * C ,我们需要使用其去计算一个W
        :return:
        '''
        B, C, H, W = X.shape
        X = self.feature(X)
        X = X.flatten(2).permute(0, 2, 1)  ## the size is n * hw * c
        # compute token coefficients
        #feature: N, C, H, W, token: N, CT, L
        if not self.dynamic :
            A = self.WA(X) ## the size is n * hw * L
        else:
            ## tokens的大小是 B * L * C
            assert tokens is not None
            L = tokens.size(1)
            assert self.L==L
            W = self.WTR(tokens) # the size is B * L * C
            A = torch.matmul(X,W.permute(0,2,1)) ## the size is N * HW * L

        A = F.softmax(A,dim = 1) ## B * hw * L
        T = torch.matmul(A.permute(0,2,1),X) ## the size is n * L * C
        return T,A

class Projector(nn.Module):
    def __init__(self, C,head=16):
        super(Projector , self).__init__()
        self.proj_query_conv = nn.Conv2d(C, C, kernel_size=1)
        self.proj_value_conv = nn.Conv1d(C, C, kernel_size=1)
        self.proj_key_conv = nn.Conv1d(C, C, kernel_size=1)
        self.head = head

    def forward(self, feature, token):
        N, L, C = token.shape
        #token = token.view(N, C, L)
        token = token.permute(0, 2, 1)  # the size is to N * c *  l to conv
        h = self.head
        proj_v = self.proj_value_conv(token).view(N, h, -1, L)
        proj_k = self.proj_key_conv(token).view(N, h, -1, L)
        proj_q = self.proj_query_conv(feature)
        N, C, H, W = proj_q.shape
        proj_q = proj_q.view(N, h, C // h, H * W).permute(0, 1, 3, 2)
        proj_coef = F.softmax(torch.Tensor.matmul(proj_q, proj_k) / np.sqrt(C / h), dim=3) ## the size is N * h * HW * L
        proj = torch.Tensor.matmul(proj_v, proj_coef.permute(0, 1, 3, 2)) ## the size is N * h * C//h * HW
        _, _, H, W = feature.shape
        return feature + proj.view(N, -1, H, W)         ## 这里存在问题，因为feature和proj的C 大小不一样
class VTLayer(nn.Module):
    def __init__(self,L,C,dynamic,head,withProject = True):
        super().__init__()
        self.token = Tokenizer(L=L,C=C,dynamic=dynamic)
        self.transformer = TransformerEncoderLayer(d_model=C, nhead=head, dim_feedforward=128, activation='relu',
                                                   dropout=0.2)
        self.withProject = withProject
        if withProject:
            self.project = Projector(C=C,head=head)


    def forward(self,X,token = None):
        tokens = self.token(X,token)[0]
        tokens = self.transformer(tokens.permute(1,0,2))[0]  ## the size is L * N * C
        tokens = tokens.permute(1,0,2) ## the size is n * L * C
        if self.withProject:
            X = self.project(X,tokens)
        return X,tokens

class VT(nn.Module):
    def __init__(self,layer,L,C,input_channels,head,return_intermediate=True):
        super().__init__()
        self.layers =[]
        self.conv = nn.Conv2d(input_channels, C, kernel_size=1)
        if layer ==1:
            self.layers.append(VTLayer(L,C,dynamic=False,head = head,withProject= False))
        else:
            self.layers.append(VTLayer(L, C, dynamic=False, head=head))
            for i in range(1,layer-1):
                self.layers.append(VTLayer(L,C,dynamic=True,head = head))
            self.layers.append(VTLayer(L,C,dynamic=True,head = head,withProject=False))
        self.layers = nn.ModuleList(self.layers)
        self.return_intermediate = return_intermediate
    def forward(self,x,tokens=None):
        x= self.conv(x)
        # tokens = None
        return_tokens = []
        for layer in self.layers:
            x,tokens = layer(x,tokens)
            return_tokens.append(tokens)
        if self.return_intermediate:
            return x,torch.stack(return_tokens,dim=0)
        else:
            return x,return_tokens[-1].unsqueeze(0)  ## 扩维
    ## 对于我们的任务来说，只有token对我们是重要的。