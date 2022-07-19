## 这是一个全新的idea，当我们一条路走不通的时候，我们就要换一种思路，另辟蹊径
from model.resnet import *
from model.position_encoding import build_position_encoding
from model.transformer import build_transformer,Allen_transformer
from model.loss import SetCriterion
from model.Att import *
from model.VT import VT
from model.resnet import ResNetV1b
class MHVT(nn.Module):
    def __init__(self, feature_size,aux_loss,res_dims, layers,row_layer,col_layer,vt_layer, hidden_dim, num_queries, pos_type,frame_num=5, nheads=8, dropout=0.1, dim_feedforward=2048,
                 mlp_layers=3):
        super().__init__()
        self.backbone = ResNetV1b(dims=res_dims,layers=layers)
        # self.backbone = resnet("resnet18",pretrained=True)
        ## 这是LSTR的关键代码
        self.aux_loss = aux_loss
        ## two position embedding
        self.position_embedding = build_position_encoding(hidden_dim=hidden_dim, type="v3")
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(res_dims[-1], hidden_dim, kernel_size=1)  # the same as channel of self.layer4
        ## dim_feedforward的含义是两个linear之间的大小。
        self.transformer = Allen_transformer(row_dim = hidden_dim * feature_size[1],col_dim =hidden_dim * feature_size[0] ,nhead = nheads,num_row_layer=row_layer,num_col_layer=col_layer,dim_feedward=dim_feedforward,dropout =dropout)
        self.VT =VT(layer= vt_layer, L=num_queries,C=hidden_dim,input_channels=hidden_dim,head=nheads) ## 从这里得到的大小是N * L * C
        self.class_embed = nn.Linear(hidden_dim, 2 + 1)  ##为何这里的类别为3呢？
        self.specific_embed = MLP(hidden_dim, hidden_dim,  4, mlp_layers)## 四个特质参数
        self.shared_embed = MLP(hidden_dim, hidden_dim, 4, mlp_layers) ## 四个共享参数
        self.loss = SetCriterion(aux_loss=aux_loss,dec_layers=vt_layer)  ## 这个layer对应VT的layer
        ## generate mask
        self.encoder_attention = Att_in(frame_num,hidden_dim)
        self.fpn_attention = []
        for d in res_dims[:-1]:
            self.fpn_attention.append(Att_in(frame_num,d))
        self.fpn_attention = nn.Sequential(*list([m for m in self.fpn_attention]))
        self.attentionMap = MHAttentionMap(hidden_dim, hidden_dim, nheads, dropout=0.0)
        self.fusionMask = MaskHeadSmallConv(hidden_dim,nheads, [64,32,16],context_dim= 64)
    def forward(self,images,masks=None,target=None):

        ## 直接修改了transformer的方式，别的没有变化
        ## the size of images is b * 5 * 3 *  h * w
        ## the size of masks is b * 1 *  h * w
        b,f,c,h,w = images.size()
        images = images.reshape(-1,c,h,w) ##  the size is (bf) * 3 * h *w
        features = self.backbone(images)
        src_proj = self.input_proj(features[-1])
        # src_proj = self.input_proj(features)
        _,c,h,w = src_proj.shape
        src_proj = src_proj.reshape(b,f,c,h,w)
        ## generate masks
        masks = F.interpolate(masks[:, 0, :, :][None], size=(h,w)).to(torch.bool)[0] # the size is b * h * w
        masks = masks.unsqueeze(1).repeat(1,f,1,1)
        pos = self.position_embedding(src_proj, masks) ## the masks is b * f * h * w, the output is b * f * hidden_dim * h * w

        ## 注意position到底是哪里的
        ## the size is b * f * c * h * w
        features = self.transformer(src_proj, masks,pos)[0] ## input_proj将c变为hidden dim大小
        ## change the feature's to b * c * fh * w
        features = features.permute(0,2,1,3,4).reshape(b,c,-1,w)
        hs = self.VT(features)[1] ## the size b * L * hidden_dim
        output_class = self.class_embed(hs)  # 2,B,7,3
        output_specific = self.specific_embed(hs)  ## 2,B ,7,4  ## 四个独立的变量
        output_shared = self.shared_embed(hs)  ## 2 , b ,7,4  ## 四个共享的变量
        output_shared = torch.mean(output_shared, dim=-2, keepdim=True)  # 2, b , 1, 4  ## 因为是共享，所以需要求mean操作
        output_shared = output_shared.repeat(1, 1, output_specific.shape[2], 1)  ## 然后重复7次，是的每条车道线的共享参数相同
        output_specific = torch.cat([output_specific[:, :, :, :2], output_shared, output_specific[:, :, :, 2:]],
                                    dim=-1)  ### 排列顺序是 2 特指+ 4个共享 + 2 个特指 2 b 7,8  把lower 和 upper放在了前面
        out = {'pred_logits': output_class[-1], 'pred_curves': output_specific[-1]}  # B 7,3   b,7,8
        ## 不明白的地方，这个2的作用是什么 ？？ 难道是两个decoder layer concat的结果？？答案是yes
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(output_class, output_specific)
        if target is None:
            return out

        ## 这部分写loss
        losses,indices = self.loss(out, target)
        return out,losses,indices ##weight 1 240 240  是 最后一个encoder layer的像素之间的参数

    # @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_curves': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]