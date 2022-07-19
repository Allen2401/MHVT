import torch
import torch.nn as nn
import torch.nn.functional as F
from model.matcher import build_matcher
import numpy as np
import cv2
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
## 让我们学习两个小时
class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self,aux_loss,dec_layers=2,eos_coef=1.0,with_mask = False):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        # cardinality is not used to propagate loss
        self.num_classes = 2
        self.weight_dict = {'loss_ce': 3, 'loss_curves': 5, 'loss_lowers': 2, 'loss_uppers': 2}
        if with_mask:
            self.weight_dict["loss_mask"] = 2
            self.weight_dict["loss_dice"] = 2   ## 这个参数我们还要进行具体的调整
        self.matcher = build_matcher(set_cost_class=self.weight_dict['loss_ce'],
                                    curves_weight=self.weight_dict['loss_curves'],
                                    lower_weight=self.weight_dict['loss_lowers'],
                                    upper_weight=self.weight_dict['loss_uppers'])
        if aux_loss:
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f'_{i}': v for k, v in self.weight_dict.items()})  ## i don't know what is this？
            self.weight_dict.update(aux_weight_dict)
        self.eos_coef = eos_coef
        self.losses = ['labels', 'curves', 'cardinality']
        if with_mask:
            self.losses = ['labels', 'curves', 'cardinality', 'masks']
        # threshold = 15 / 720.
        # self.threshold = nn.Threshold(threshold**2, 0.)
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef

        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_curves, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([tgt[:, 0][J].long() for tgt, (_, J) in zip (targets, indices)])  ## 是所有valid_lane的数量长度的1
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device) ## the size is batch_size * max_lane
        target_classes[idx] = target_classes_o  ## 把对方的类别填充上去
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)  ## b * 3 * max_lane
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_curves):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits'] # b*max_lane*3
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([tgt.shape[0] for tgt in targets], device=device) ## tgt中每张图片中车道线的数目
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1) ## b*max_lane 每张图片预测出现的车道线的数目
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())  # 对车道线数目进行loss
        losses = {'cardinality_error':card_err}
        return losses

    def loss_curves(self, outputs, targets, indices, num_curves):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_curves' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_lowers = outputs['pred_curves'][:, :, 0][idx]
        src_uppers = outputs['pred_curves'][:, :, 1][idx]
        src_polys  = outputs['pred_curves'][:, :, 2:][idx]
        target_lowers = torch.cat([tgt[:, 1][i] for tgt, (_, i) in zip(targets, indices)], dim=0)
        target_uppers = torch.cat([tgt[:, 2][i] for tgt, (_, i) in zip(targets, indices)], dim=0)
        target_points = torch.cat([tgt[:, 3:][i] for tgt, (_, i) in zip(targets, indices)], dim=0)

        target_xs = target_points[:, :target_points.shape[1] // 2]
        ys = target_points[:, target_points.shape[1] // 2:].transpose(1, 0)
        valid_xs = target_xs >= 0
        weights = (torch.sum(valid_xs, dtype=torch.float32) / torch.sum(valid_xs, dim=1, dtype=torch.float32)) ** 0.5
        weights = weights / torch.max(weights)

        # Calculate the predicted xs
        pred_xs = src_polys[:, 0] / (ys - src_polys[:, 1]) ** 2 + src_polys[:, 2] / (ys - src_polys[:, 1]) + \
                  src_polys[:, 3] + src_polys[:, 4] * ys - src_polys[:, 5]
        distance_weight = ((1-(ys-0.3)) ** 2) * 2
        pred_xs = pred_xs * weights * distance_weight
        pred_xs = pred_xs.transpose(1, 0)
        target_xs = target_xs.transpose(1, 0) * weights * distance_weight
        target_xs = target_xs.transpose(1, 0)

        loss_lowers = F.l1_loss(src_lowers, target_lowers, reduction='none')
        loss_uppers = F.l1_loss(src_uppers, target_uppers, reduction='none')
        loss_polys  = F.l1_loss(pred_xs[valid_xs], target_xs[valid_xs], reduction='none')

        losses = {}
        losses['loss_lowers']  = loss_lowers.sum() / num_curves
        losses['loss_uppers']  = loss_uppers.sum() / num_curves
        losses['loss_curves']   = loss_polys.sum() / num_curves

        return losses    ## 对于曲线进行loss
    def loss_masks(self,outputs,targets,indices,num_maps):
        """
        如果加上这部分，我们的训练会更加更加缓慢，因为还有一张mask的信息需要传递，之正期待这种方法
        最好真的能够提升速度，感觉有很大的概率会影响我的batch_size,感觉还是自己画一下比较靠谱呢
        :param output:
        :param targets:
        :return:
        """
        assert "pred_masks" in outputs
        ## 这个target部分可能需要我们自己生成一下，但放在这里能够减轻数据加载的压力，我们可以降4倍下采样，360 * 640的大小的话？可以降低吗
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        ## 这里应该修改，因为target是点的形式，但是我们需要的是mask的形式，得自己画一下
        ## gt的大小：b*6*（3+点的数目）因为我们已经有需要的tgt_idx是，所以只需要生成需要的就好了
        ## 360 和640的大小需要进行设置

        #targets_need = targets[tgt_idx]
        targets_need = [targets[batch_id][tgt_id] for (batch_id,tgt_id) in zip(*tgt_idx)]
        num = len(tgt_idx[0])
        #num,n = targets_need.size()
        target_masks = np.zeros((num,360,640),dtype = np.uint8)
        for i in range(num):
            lane = targets_need[i][3:].reshape(2,-1)
            lane = lane[:,lane[1]>0]
            lane[0,:] *=640
            lane[1,:] *=360
            lane = lane.permute(1,0).cpu().numpy()[None,:,:].astype(np.int64)
            cv2.polylines(target_masks[i], lane, isClosed=False, color=1, thickness=5)
        target_masks = torch.from_numpy(target_masks).to(src_masks)

        src_masks = src_masks[src_idx]
        src_masks = src_masks.flatten(1)
        target_masks = target_masks.flatten(1)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_maps),
            "loss_dice": dice_loss(src_masks, target_masks, num_maps),
        }
        return losses



    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])## [0,0,0,0,1,1,1,2,2,2] 重复的次数就是每张图片中车道线的数量
        src_idx = torch.cat([src for (src, _) in indices]) # 再将indices的pred进行展开

        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_curves, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'curves': self.loss_curves,
            'masks':self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_curves, **kwargs)
    ## 现在关键看这个部分，朋友
    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format  # the size is pred_logits b*7*2  curve b*7*8 八个参数
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        ## 我们同样需要对车道线tgt进行guolv，因为其中包含无效车道线
        targets = [targets[batch_i][targets[batch_i][:,0]>0] for batch_i in range(targets.size(0))]
        # Retrieve the matching between the outputs of the last layer and the targets
        ## row表示pred,col表示的是tgt，indices 是bs的tuple ，tuple内容为（row_index,col_index）
        indices = self.matcher(outputs_without_aux, targets)
        if len(indices)==0:
            return None,None


        # Compute the average number of target boxes accross all nodes, for normalization purposes
        ## 车道线总共的数目，可以等会在这里设置一下
        num_curves = sum(tgt.shape[0] for tgt in targets)
        num_curves = torch.as_tensor([num_curves], dtype=torch.float, device=next(iter(outputs.values())).device)
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_curves))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_curves, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        total_loss = sum(losses[k] * self.weight_dict[k] for k in losses.keys() if k in self.weight_dict)  ## 对loss进行加权和，这一部分是只要在就加进去
        losses_scaled = {k: (v * self.weight_dict[k]).cpu().item()
                                    for k, v in losses.items() if k in self.weight_dict}
        losses_scaled['total_loss'] = total_loss
        ## 我跟他们不一样，是因为对方的
        ##
        ##dice如何合并？返回？
        batch_size = len(targets)
        return_dices = torch.ones([batch_size,2,6],dtype=torch.int8,device = next(iter(outputs.values())).device)*-1
        for idx in range(batch_size):
            src_index,tgt_index = indices[idx]
            length = len(src_index)
            return_dices[idx,0,:length] = src_index[:]
            return_dices[idx,1,:length] = tgt_index[:]
        return losses_scaled,return_dices
        ## 现在这个进展看来十分不平稳


def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes