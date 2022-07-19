import os
import json
import random

import numpy as np
from tabulate import tabulate
import pickle
from utils.lane import LaneEval
from utils.my_metric import My_LaneEval
from utils.metric import eval_json

SPLIT_FILES = {
    'train+val': ['label_data_0313.json','label_data_0601.json','label_data_0531.json'],
    'train': ['label_data_0313.json','label_data_0601.json'],
    'val': ['label_data_0531.json'],
    'test': ['test_label.json'],
}


class TuSimple(object):
    def __init__(self, split='train', max_lanes=None, root=None, metric='default'):
        self.split = split
        self.root = root
        self.metric = metric

        if split not in SPLIT_FILES.keys():
            raise Exception('Split `{}` does not exist.'.format(split))

        self.anno_files = [os.path.join(self.root, path) for path in SPLIT_FILES[split]]

        if root is None:
            raise Exception('Please specify the root directory')
        self.image_file = []
        self.img_w, self.img_h = 1280, 720
        self.max_points = 0
        name = [s.strip(".json") for s in SPLIT_FILES[split]]
        self.cache_file = os.path.join("E:\PycharmProjects\MutilLSTR\cache", "tusimple_{}.pkl".format(name))
        self.load_data()
        # self.load_annotations()

        # Force max_lanes, used when evaluating testing with models trained on other datasets
        if max_lanes is not None:
            self.max_lanes = max_lanes

    def load_data(self):
        ## 如果数据在cache中存在，直接获取即可，否则生成并保存
        print("loading from cache file: {}".format(self.cache_file))
        if not os.path.exists(self.cache_file):
            print("No cache file found...")
            self.load_annotations()
            self.transform_annotations()  ## 使用transformer进行转化
            ## pickle.dump 降python数据对象转化为字节流的过程。
            with open(self.cache_file, "wb") as f:
                pickle.dump([self.annotations,
                             self.image_file,
                             self.max_lanes,
                             self.max_points], f)
        else:
            with open(self.cache_file, "rb") as f:
                (self.annotations,
                 self.image_file,
                 self.max_lanes,
                 self.max_points) = pickle.load(f)

    def get_img_heigth(self, path):
        return 720

    def get_img_width(self, path):
        return 1280

    def get_metrics(self, lanes, idx):
        label = self.annotations[idx]
        org_anno = label['old_anno']
        pred = self.pred2lanes(org_anno['path'], lanes, org_anno['y_samples'])
        _, _, _, matches, accs, dist = LaneEval.bench(pred, org_anno['org_lanes'], org_anno['y_samples'], 0, True)

        return matches, accs, dist

    def pred2lanes(self, path, pred, y_samples):
        ys = np.array(y_samples) / self.img_h
        lanes = []
        for lane in pred:
            if lane[0] == 0:
                continue
            # lane_pred = np.polyval(lane[3:], ys) * self.img_w
            polys = lane[3:]
            lane_pred = (polys[0] / (ys - polys[1]) ** 2 + polys[ 2] / (ys - polys[1]) + polys[3] + polys[4] * ys - polys[5])*self.img_w
            lane_pred[(ys < lane[1]) | (ys > lane[2])] = -2
            lanes.append(list(lane_pred))
        return lanes

    def load_annotations(self):
        self.annotations = []
        max_lanes = 0
        for anno_file in self.anno_files:
            with open(anno_file, 'r') as anno_obj:
                lines = anno_obj.readlines()
            for line in lines:
                data = json.loads(line)
                y_samples = data['h_samples']
                gt_lanes = data['lanes']
                lanes = [[(x, y) for (x, y) in zip(lane, y_samples) if x >= 0] for lane in gt_lanes]
                lanes = [lane for lane in lanes if len(lane) > 0]
                max_lanes = max(max_lanes, len(lanes))
                img_path = os.path.join(self.root, data['raw_file'])
                self.image_file.append(img_path)
                self.max_points = max(self.max_points, max([len(l) for l in gt_lanes]))
                self.annotations.append({
                    'path': os.path.join(self.root, data['raw_file']),
                    'org_path': data['raw_file'],
                    'org_lanes': gt_lanes,
                    'lanes': lanes,
                    'aug': False,
                    'y_samples': y_samples
                })

        if self.split == 'train':
            random.shuffle(self.annotations)
        print('total annos', len(self.annotations))
        self.max_lanes = max_lanes

    def transform_annotation(self, anno, img_wh=None):
        if img_wh is None:
            img_h = self.get_img_heigth(anno['path'])
            img_w = self.get_img_width(anno['path'])
        else:
            img_w, img_h = img_wh

        old_lanes = anno['lanes']
        categories = anno['categories'] if 'categories' in anno else [1] * len(old_lanes)
        old_lanes = zip(old_lanes, categories)
        old_lanes = filter(lambda x: len(x[0]) > 0, old_lanes)
        lanes = np.ones((self.max_lanes, 1 + 2 + 2 * self.max_points), dtype=np.float32) * -1e5
        lanes[:, 0] = 0
        old_lanes = sorted(old_lanes, key=lambda x: x[0][0][0])
        for lane_pos, (lane, category) in enumerate(old_lanes):
            lower, upper = lane[0][1], lane[-1][1]
            xs = np.array([p[0] for p in lane]) / img_w
            ys = np.array([p[1] for p in lane]) / img_h
            lanes[lane_pos, 0] = category
            lanes[lane_pos, 1] = lower / img_h
            lanes[lane_pos, 2] = upper / img_h
            lanes[lane_pos, 3:3 + len(xs)] = xs
            lanes[lane_pos, (3 + self.max_points):(3 + self.max_points + len(ys))] = ys

        new_anno = {
            'path': anno['path'],
            'label': lanes,
            'old_anno': anno,
            'categories': [cat for _, cat in old_lanes]
        }

        return new_anno

    def transform_annotations(self):
        print('Transforming annotations...')
        self.annotations = np.array(list(map(self.transform_annotation, self.annotations)))
        print('Done.')
    #
    # def transform_annotations(self, transform):
    #     self.annotations = list(map(transform, self.annotations))

    def pred2tusimpleformat(self, idx, pred, runtime):
        runtime *= 1000.  # s to ms
        img_name = self.annotations[idx]['old_anno']['org_path']
        h_samples = self.annotations[idx]['old_anno']['y_samples']
        lanes = self.pred2lanes(img_name, pred, h_samples)
        output = {'raw_file': img_name, 'lanes': lanes, 'run_time': runtime}
        return json.dumps(output)

    def save_tusimple_predictions(self, predictions, runtimes, filename):
        lines = []
        for idx in range(len(predictions)):
            line = self.pred2tusimpleformat(idx, predictions[idx], runtimes[idx])
            lines.append(line)
        with open(filename, 'w') as output_file:
            output_file.write('\n'.join(lines))

    def eval(self, exp_dir, predictions, runtimes, label=None, only_metrics=True):
        pred_filename = './save/tusimple_predictions_{}.json'.format(label)
        self.save_tusimple_predictions(predictions, runtimes, pred_filename)
        if self.metric == 'default':
            result = json.loads(LaneEval.bench_one_submit(pred_filename,  self.anno_files[0]))
        elif self.metric == 'ours':
            result = json.loads(My_LaneEval.bench_one_submit(pred_filename,  self.anno_files[0]))

            # result = json.loads(eval_json(pred_filename, self.anno_files[0], json_type='tusimple'))
        table = {}
        for metric in result:
            table[metric['name']] = [metric['value']]

        # table = tabulate(table, headers='keys')

        if not only_metrics:
            filename = 'tusimple_{}_eval_result_{}.json'.format(self.split, label)
            with open(os.path.join(exp_dir, filename), 'w') as out_file:
                json.dump(result, out_file)

        return tabulate(table, headers='keys'),table

    def __getitem__(self, idx):
        return self.annotations[idx]

    def __len__(self):
        return len(self.annotations)
