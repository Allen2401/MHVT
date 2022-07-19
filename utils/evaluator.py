import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from dataset.LaneDataset import LaneDataset
from utils.my_metric import My_LaneEval
import cv2
import json
EXPS_DIR = 'experiments'
from utils.lane import LaneEval
from tabulate import tabulate
import matplotlib.pyplot as plt
import os
import random
class PostProcess(nn.Module):
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_logits, out_curves = outputs['pred_logits'], outputs['pred_curves']
        out_logits = out_logits[0].unsqueeze(0) ## 为什么要这样？没变啊
        out_curves = out_curves[0].unsqueeze(0)
        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2
        prob = F.softmax(out_logits, -1)
        scores, labels = prob.max(-1)
        labels[labels != 1] = 0
        results = torch.cat([labels.unsqueeze(-1).float(), out_curves], dim=-1)

        return results

def evaluate(self,label,idx,outputs=None):
    batch_size = label.size(0)
    gt_mask= np.zeros((batch_size,self.img_h,self.img_w),np.uint8)
    pred_mask = np.zeros((batch_size,self.img_h, self.img_w), np.uint8)
    acc = []
    for i in range(batch_size):
        gt = label[i]
        gt = gt[gt[:,0]>0]
        for l in range(gt.shape[0]):
            lane = gt[l][3:].reshape((2,-1))
            lane = lane[:,lane[1]>0].numpy()
            lane[0,:] *=self.img_w
            lane[1,:] *=self.img_h
            print(lane.shape)
            lane = np.transpose(lane,(1,0))
            print(lane.shape,lane.dtype)
            print(lane)
            lane = np.array([lane], np.int64)
            cv2.polylines(gt_mask[i],lane,isClosed=False,color=1,thickness=5)

        out_logits, out_curves = outputs['pred_logits'][i], outputs['pred_curves'][i]
        prob = F.softmax(out_logits, -1)
        scores, mask = prob.max(-1)
        mask[mask != 1] = 0
        pred = torch.cat([mask.unsqueeze(-1).float(), out_curves], dim=-1)
        pred = pred[pred[:, 0] != 0]  # filter invalid lanes,这里进行了选择
        matches, accs, _ = self.dataset.get_metrics(pred, idx[i])
        acc.append(accs)
        ## pred是全部的吗？应该是为每个车道线都匹配了一个
        for i, lane in enumerate(pred):
            lane = lane[1:]  # remove conf
            lower, upper = lane[0], lane[1]
            lane = lane[2:]  # remove upper, lower positions

            # generate points from the polynomial
            ys = np.linspace(lower, upper, num=100)
            points = np.zeros((len(ys), 2), dtype=np.int32)
            points[:, 1] = (ys * self.img_h).astype(int)
            points[:, 0] = (np.polyval(lane, ys) * self.img_w).astype(int)
            points = points[(points[:, 0] > 0) & (points[:, 0] < self.img_w)]
            points = np.array([points],np.int64)
            cv2.polylines(pred_mask[i],points,isClosed=False,color =1 ,thickness=1)

    ## 接下来来计算他们的F-measure
    gt_mask = gt_mask.reshape((batch_size,-1))
    pred_mask = pred_mask.reshape((batch_size,-1))
    TP = np.sum(gt_mask * pred_mask)
    precision = TP/np.sum(pred_mask)
    recall = TP/np.sum(gt_mask)
    f1_measure = 2*precision * recall/(precision + recall)
    return sum(acc),precision,recall,f1_measure
    # return {'t_acc':accs,'precision':precison,'recall':recall,'f1-measure':f1_measure}

class Evaluator(object):
    def __init__(self, dataset, exp_dir, poly_degree=3):
        self.dataset = dataset
        # self.predictions = np.zeros((len(dataset.image_list), dataset.max_lanes, 4 + poly_degree))
        self.predictions = None
        self.runtimes = np.zeros(len(dataset))
        self.loss = np.zeros(len(dataset))
        self.exp_dir = exp_dir
        self.new_preds = False
        self.img_h, self.img_w = 720,1280
        self.ids = []

    def add_prediction(self, idx, outputs, runtime=1):
        out_logits, out_curves = outputs['pred_logits'], outputs['pred_curves']
        prob = F.softmax(out_logits, -1)
        scores, labels = prob.max(-1)
        labels[labels != 1] = 0
        pred = torch.cat([labels.unsqueeze(-1).float(), out_curves], dim=-1).detach().cpu().numpy()
        if self.predictions is None:
            self.predictions = np.zeros((len(self.dataset), pred.shape[1], pred.shape[2]))
        self.ids.extend(idx)
        self.predictions[idx, :pred.shape[1], :] = pred
        self.runtimes[idx] = runtime
        self.new_preds = True
        ### 这里存储的是参数

    def pred2lanes(self, pred, y_samples):
        '''
        this is from parameters to calculate the point
        :param pred:
        :param y_samples:
        :return:
        '''
        ys = np.array(y_samples) /720
        lanes = []
        for lane in pred:
            if lane[0] == 0:
                continue
            # lane_pred = np.polyval(lane[3:], ys) * self.img_w
            polys = lane[3:]
            lane_pred = (polys[0] / (ys - polys[1]) ** 2 + polys[2] / (ys - polys[1]) + polys[3] + polys[4] * ys -
                         polys[5]) * 1280
            lane_pred[(ys < lane[1]) | (ys > lane[2])] = -2
            lanes.append(list(lane_pred))
        return lanes

    def save_tusimple_predictions(self, predictions, runtimes, filename):
        lines = []
        for idx in range(len(predictions)):
            img_name = self.dataset.image_list[idx]
            h_samples = self.dataset.lanes_y[idx]
            lanes = self.pred2lanes(predictions[idx],h_samples[0])
            output = {'raw_file': img_name, 'lanes': lanes, 'run_time': runtimes}
            lines.append(json.dumps(output))
        with open(filename, 'w') as output_file:
            output_file.write('\n'.join(lines))

    def f1_evaluate(self):
        error = 0
        tusimple_acc,acc,fp,fn,precision,recall,F1 = [],[],[],[],[],[],[]
        return_gt,return_perd = None,None
        select = random.choice(self.ids)
        for id in self.ids:
            img = cv2.imread(os.path.join(self.dataset.root_path, self.dataset.image_list[id] + "/20.jpg"),
                       cv2.IMREAD_COLOR)
            # print(os.path.join(self.dataset.root_path, self.dataset.image_list[id] + "20.jpg"))
            gt_mask = np.zeros((self.img_h, self.img_w), np.uint8)
            lanes,y_samples = self.dataset.lanes_x[id],self.dataset.lanes_y[id]
            gt_lanes = list(zip(lanes,y_samples))
            # gt_lanes = self.dataset.dataset.annotations[id]['old_anno']['lanes']
            for xs,ys in gt_lanes:
                points = list(filter(lambda x: x[0] > 0, list(zip(xs, ys))))
                gt_points = np.array([points], np.int64)
                # cv2.polylines(gt_mask, gt_points, isClosed=False, color=[255,0,255], thickness=20)
                cv2.polylines(img, gt_points, isClosed=False, color=[255,0,255], thickness=20)
            pred_mask = np.zeros((self.img_h, self.img_w), np.uint8)
            t_acc,p,n= self.get_metrics(self.predictions[id], id)
            tusimple_acc.append(t_acc)
            fp.append(p)
            fn.append(n)
            for i, lane in enumerate(self.predictions[id]):
                if lane[0]==0:
                    continue
                lane = lane[1:]  # remove conf
                lower, upper = lane[0], lane[1]
                lane = lane[2:]  # remove upper, lower positions
                ys = np.linspace(lower, upper, num=100)
                points = np.zeros((len(ys), 2), dtype=np.int32)
                points[:, 1] = (ys * self.img_h).astype(int)
                points[:, 0] = ((lane[0] / (ys - lane[1]) ** 2 + lane[2] / (ys - lane[1]) + lane[3] + lane[4] * ys - lane[5]) * self.img_w).astype(int)
                points = points[(points[:, 0] > 0) & (points[:, 0] < self.img_w)]
                points = np.array([points], np.int64)
                # cv2.polylines(pred_mask, points, isClosed=False, color=1, thickness=20)
                cv2.polylines(img, points, isClosed=False, color=[255,255,0], thickness=20)
            # plt.figure()
            # plt.subplot(1,2,1)
            # plt.imshow(gt_mask)
            # plt.subplot(1,2,2)
            # plt.imshow(pred_mask)
            # plt.show()
            # plt.pause(0)
            if id == select:
                return_gt = gt_mask[:,:]
                return_perd = pred_mask[:,:]
            gt_mask = gt_mask.flatten()
            pred_mask = pred_mask.flatten()
            TP = np.sum(gt_mask * pred_mask)
            acc.append(np.sum(gt_mask == pred_mask)/(self.img_h * self.img_w))

            cv2.imwrite(f"G:/learningDat/tusimple/jiaocha/{id}.jpg", img)
            if np.sum(pred_mask)==0 or np.sum(gt_mask)==0 or TP==0:
                error+=1
                continue
            p = TP / np.sum(pred_mask)
            r = TP / np.sum(gt_mask)
            f1 = 2* p * r / (p + r)
            precision.append(p)
            recall.append(r)
            F1.append(f1)
        # imgs = torch.tensor(np.concatenate([np.expand_dims(return_gt,0) ,np.expand_dims(return_perd,0)],axis=0))
        if len(self.ids)==error:
            return  {'tusimple_acc':sum(tusimple_acc)/len(self.ids),'fp':sum(fp)/len(self.ids),'fn':sum(fn)/len(self.ids),'acc':sum(acc)/len(self.ids)}
        ddict = {
            'tusimple_acc':sum(tusimple_acc)/len(self.ids),
            'fp': sum(fp) / len(self.ids),
            'fn': sum(fn) / len(self.ids),
            'acc':sum(acc)/len(self.ids),
            'precision':sum(precision)/(len(self.ids)-error),
            'recall':sum(recall)/(len(self.ids)-error),
            'F1-measure':sum(F1)/(len(self.ids)-error),
            'error':error
        }
        ## 然后还想画一张图

        return ddict


    def tusimple_evaluate(self,runtimes=1, label=None):
        pred_filename = './save/LSTR_tusimple_1/tusimple_predictions_{}.json'.format(label)
        self.save_tusimple_predictions(self.predictions, runtimes, pred_filename)
        result = json.loads(LaneEval.bench_one_submit(pred_filename,self.dataset.files[0]))
        table = {}
        for metric in result:
            table[metric['name']] = [metric['value']]
        #
        # # table = tabulate(table, headers='keys')
        #
        # if not only_metrics:
        #     filename = 'tusimple_{}_eval_result_{}.json'.format(self.split, label)
        #     with open(os.path.join(exp_dir, filename), 'w') as out_file:
        #         json.dump(result, out_file)

        return tabulate(table, headers='keys'),table

    def get_metrics(self, lanes, idx):
        y_samples = self.dataset.lanes_y[idx][0]
        pred = self.pred2lanes(lanes,y_samples)
        acc,fp,fn = LaneEval.bench(pred, self.dataset.lanes_x[idx],y_samples, 0)
        return acc,fp,fn

    def eval(self, **kwargs):
        return self.dataset.dataset.eval(self.exp_dir, self.predictions, self.runtimes, **kwargs)

