# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1,
                 curves_weight: float = 1, lower_weight: float = 1, upper_weight: float = 1):
        """Creates the matcher
        """

        super().__init__()
        self.cost_class = cost_class
        threshold = 15 / 720.
        self.threshold = nn.Threshold(threshold**2, 0.)

        self.curves_weight = curves_weight
        self.lower_weight = lower_weight
        self.upper_weight = upper_weight

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching
        """
        num_curves = sum(tgt.shape[0] for tgt in targets)
        if num_curves ==0:
            return []
        ## num_queries is the max number of lanes
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # the size is (bs*num_quries)*3   softmax
        tgt_ids  = torch.cat([tgt[:, 0] for tgt in targets]).long()  ## target 取第一列，就是类别


        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]  ## 取-softmax## 这里把一个值取了很多遍
        out_bbox = outputs["pred_curves"]  ## 这是8个参数部分 b*num_queries * 8
        tgt_uppers = torch.cat([tgt[:, 2] for tgt in targets])
        tgt_lowers = torch.cat([tgt[:, 1] for tgt in targets])

        # # Compute the L1 cost between lowers and uppers
        cost_lower = torch.cdist(out_bbox[:, :, 0].view((-1, 1)), tgt_lowers.unsqueeze(-1), p=1)  ## L1范式是求绝对值
        cost_upper = torch.cdist(out_bbox[:, :, 1].view((-1, 1)), tgt_uppers.unsqueeze(-1), p=1)

        # # Compute the poly cost
        tgt_points = torch.cat([tgt[:, 3:] for tgt in targets]) # 0~20 112
        tgt_xs = tgt_points[:, :tgt_points.shape[1] // 2]  ## 因为tgt没有进行invalid填充,只有有效的车道线~，所以大小不等于 bs*num_quries
        valid_xs = tgt_xs >= 0
        weights = (torch.sum(valid_xs, dtype=torch.float32) / torch.sum(valid_xs, dim=1, dtype=torch.float32))**0.5  ## 有效的点数/每条车道线的有效点数 点数越多，则权重越小
        weights = weights / torch.max(weights)  ## 对权重进行归一化

        tgt_ys = tgt_points[:, tgt_points.shape[1] // 2:]
        out_polys = out_bbox[:, :, 2:].view((-1, 6))  ## 从2开始就是函数的参数部分了  (bs*max_lane)*参数个数
        tgt_ys = tgt_ys.repeat(out_polys.shape[0], 1, 1) ## 三维  # (bs*max_lane)*valid_lanes*point_num
        tgt_ys = tgt_ys.transpose(0, 2) ## point_num * valid_lanes*(bs*max_lane)
        tgt_ys = tgt_ys.transpose(0, 1)## valid_lanes * point_num * (bs*max_lane)  ##
        # Calculate the predicted
        out_xs = out_polys[:, 0] / (tgt_ys - out_polys[:, 1]) ** 2 + out_polys[:, 2] / (tgt_ys - out_polys[:, 1]) + \
                 out_polys[:, 3] + out_polys[:, 4] * tgt_ys - out_polys[:, 5]
        tgt_xs = tgt_xs.repeat(out_polys.shape[0], 1, 1)
        tgt_xs = tgt_xs.transpose(0, 2)
        tgt_xs = tgt_xs.transpose(0, 1)
        # the size of tgt_x is point_number * (bs_max_lane)  ,使用valid_x 进行选择之后，变成valid_point * (bs_max_lane) ,相减求sum后大小为(bs*max_lane)
        ## stack之后的大小为 (bs_max_lane) * valid_lane
        cost_polys = torch.stack([torch.sum(torch.abs(tgt_x[valid_x] - out_x[valid_x]), dim=0) for tgt_x, out_x, valid_x in zip(tgt_xs, out_xs, valid_xs)], dim=-1)
        cost_polys = cost_polys * weights

        # # Final cost matrix
        C = self.cost_class * cost_class + self.curves_weight * cost_polys + \
            self.lower_weight * cost_lower + self.upper_weight * cost_upper

        C = C.view(bs, num_queries, -1).cpu()

        sizes = [tgt.shape[0] for tgt in targets]

        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]  ## i是rowindx,j是col_index,写的挺棒的


def build_matcher(set_cost_class,
                  curves_weight, lower_weight, upper_weight):
    return HungarianMatcher(cost_class=set_cost_class,
                            curves_weight=curves_weight, lower_weight=lower_weight, upper_weight=upper_weight)
## 今天的任务还是挺重的，看完这两部分。