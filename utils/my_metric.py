import numpy as np
import ujson as json
from sklearn.linear_model import LinearRegression
import cv2
import matplotlib.pyplot as plt

class My_LaneEval(object):
    lr = LinearRegression()
    pixel_thresh = 20
    pt_thresh = 0.85

    @staticmethod
    def f1_metric_bench(pred_lanes, gt_lanes, y_samples):
        ## 就是画图呗
        ## 注意有多个车道线
        gt_mask = np.zeros((720, 1280), np.uint8)
        pred_mask = np.zeros((720, 1280), np.uint8)
        for lane in gt_lanes:
            gt_points = list(filter(lambda x: x[0] > 0, list(zip(lane, y_samples))))
            gt_points = np.array([gt_points], np.int64)
            cv2.polylines(gt_mask, gt_points, isClosed=False, color=1, thickness=5)




        for lane in pred_lanes:
            pred_points = list(filter(lambda x: x[0] > 0, list(zip(lane, y_samples))))
            pred_points = np.array([pred_points], np.int64)
            cv2.polylines(pred_mask, pred_points, isClosed=False, color=1, thickness=5)
        gt_mask = gt_mask.reshape((1, -1))
        pred_mask = pred_mask.reshape((1, -1))
        TP = np.sum(gt_mask * pred_mask)
        precision = TP / np.sum(pred_mask)
        recall = TP / np.sum(gt_mask)
        return precision, recall


    @staticmethod
    def bench_one_submit(pred_file, gt_file):
        try:
            json_pred = [json.loads(line) for line in open(pred_file).readlines()]
        except BaseException as e:
            raise Exception('Fail to load json file of the prediction.')
        json_gt = [json.loads(line) for line in open(gt_file).readlines()]
        if len(json_gt) != len(json_pred):
            raise Exception('We do not get the predictions of all the test tasks')
        gts = {l['raw_file']: l for l in json_gt}
        precision, recall, F1_measure = 0., 0., 0.
        run_times = []
        error = 0
        for pred in json_pred:
            if 'raw_file' not in pred or 'lanes' not in pred or 'run_time' not in pred:
                raise Exception('raw_file or lanes or run_time not in some predictions.')
            raw_file = pred['raw_file']
            pred_lanes = pred['lanes']
            run_time = pred['run_time']
            run_times.append(run_time)
            if raw_file not in gts:
                raise Exception('Some raw_file from your predictions do not exist in the test tasks.')
            gt = gts[raw_file]
            gt_lanes = gt['lanes']
            y_samples = gt['h_samples']
            try:
                p, r = My_LaneEval.f1_metric_bench(pred_lanes, gt_lanes, y_samples)
                if p==0 or r ==0:
                    error = error+1
                    continue
                f1 = 2 * p * r / (p + r)

            except BaseException as e:
                raise Exception('Format of lanes error.')
            precision += p
            recall += r
            F1_measure += f1
        num = len(gts)-error
        return json.dumps([{
            'name': 'precison',
            'value': precision / num,
            'order': 'desc'
        }, {
            'name': 'recall',
            'value': recall / num,
            'order': 'asc'
        }, {
            'name': 'f1-measure',
            'value': F1_measure/num,
            'order': 'asc'
        },{
            'name': 'error',
            'value':error,
            'order':'asc'
        }])