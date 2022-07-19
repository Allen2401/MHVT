from torch.utils.data.dataset import Dataset
# from imgaug.augmentables.lines import LineString, LineStringsOnImage
from torch.utils.data import DataLoader
import torch
from dataset.transformers import *
# from .elas import ELAS
# from .llamas import LLAMAS
# from .tusimple import TuSimple
# from .nolabel_dataset import NoLabelDataset
from dataset.tusimple import TuSimple
### 经过测试，我们的这个方法可以更快地加载数据
GT_COLOR = (255, 0, 0)
PRED_HIT_COLOR = (0, 255, 0)
PRED_MISS_COLOR = (0, 0, 255)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])
mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32)
std = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32)
transfromer =  {'train':Compose(Resize((360, 640)),Flip(),Translation(),Rotate(),Gaussian(),Change_intensity(),ColorNoise(),Shadow(),ToTensor(),Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)),
                'train+val':Compose(Resize((360, 640)),Flip(),Translation(),Rotate(),Gaussian(),Change_intensity(),Shadow(),ToTensor(),Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)),
                'val':Compose(Resize((360, 640)), ToTensor(), Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)),
                'test':Compose(Resize((360, 640)), ToTensor(), Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))}
SPLIT_FILES = {
    'train+val': ['label_data_0313.json','label_data_0601.json','label_data_0531.json'],
    'train': ['label_data_0313.json','label_data_0601.json'],
    'val': ['label_data_0531.json'],
    'test': ['test_label.json'],
}
## 想弄一个json文件，包含所有的车道线的信息。这样就不用遍历很多文件了
import os
import json
import numpy as np
import yaml
class LaneDataset(Dataset):
    def __init__(self,split,sampled_frame = 5, max_skip = 2,increment = 1,samples_per_video=20):
        self.root = os.path.join('E:/data/V100/dataset/VIL100')
        dbfile = os.path.join(self.root,'data','db_info.yaml')
        self.imgdir = os.path.join(self.root,'JPEGImages')
        self.jsondir = os.path.join(self.root,'json')
        with open(dbfile,'r') as f:
            db = yaml.load(f,Loader = yaml.Loader)['sequences']
            targetset = split ## 只允许train 和test两种模式
            self.info = db
            # self.videos = [info['name'] for info in db if info['set'] == "train"]
            self.videos = [info['name'] for info in db if info['set'] == targetset]
        self.samples_per_video = samples_per_video
        self.sampled_frames = sampled_frame
        self.length = samples_per_video * len(self.videos)
        self.max_skip = max_skip
        self.increment = increment
        self.split = split
        self.transformer = transfromer[split]
        self.max_lanes = 6
        self.max_points = 91 ## 最多的点的数目是81
        self.anno = None
        with open(f"./cache/V100_{split}.json") as f:
            self.anno = json.load(f)
        # with open(f"./cache/V100_train.json") as f:
        #     self.anno = json.load(f)

        self.img_w,self.img_h = 640,360

    def __len__(self):
        return self.length
    def increase_max_skip(self):
        self.max_skip = min(self.max_skip + self.increment, 100)

    def set_max_skip(self, max_skip):
        self.max_skip = max_skip

    def transform_annotation(self, lanes, info,img_wh=None):
        lanes_x,lanes_y = lanes
        lanes = [[(x,y) for (x,y) in zip(lane_x,lane_y) if x>=0 ] for (lane_x,lane_y) in zip(lanes_x,lanes_y)]
        lanes = [lane for lane in lanes if len(lane) > 0]
        categories = [1] * len(lanes)
        lanes = zip(lanes, categories)
        lanes = filter(lambda x: len(x[0]) > 0, lanes)
        labels = np.ones((self.max_lanes, 1 + 2 + 2 * self.max_points), dtype=np.float32) * -1e5
        labels[:, 0] = 0
        lanes = sorted(lanes, key=lambda x: x[0][0][0])  ## 根据第一个节点进行排序了
        ## 因为有的车道线根本就没有点，因此需要加这个，否则会np.min会报错
        if len(lanes)==0:
            return labels
        for lane_pos, (lane, category) in enumerate(lanes):
            lower, upper = lane[0][1], lane[-1][1]
            xs = np.array([p[0] for p in lane]) / self.img_w
            ys = np.array([p[1] for p in lane]) / self.img_h
            labels[lane_pos, 0] = category
            labels[lane_pos, 1] = lower / self.img_h
            labels[lane_pos, 2] = upper / self.img_h
            labels[lane_pos, 3:3 + len(xs)] = xs
            labels[lane_pos, (3 + self.max_points):(3 + self.max_points + len(ys))] = ys
        ## 使最小的为lower
        tgt_ids  = labels[:, 0]
        last = labels[tgt_ids ==0]
        labels = labels[tgt_ids > 0]

        # make lower the same ## 使得lower相同
        labels[:, 1][labels[:, 1] < 0] = 1
        labels[:, 1][...] = np.min(labels[:, 1])
        labels = np.vstack([labels, last])
        return labels

    def __getitem__(self, idx):

        vid = self.videos[(idx // self.samples_per_video)]
        ## to get the imgfloder and annofolder of the selected video.
        imgfolder = os.path.join(self.imgdir, vid)
        annofolder = os.path.join(self.jsondir, vid)
        ## get the name of all frames in the selected video.(not containing the ".jpg")
        frames = [name[:5] for name in os.listdir(annofolder)]
        ## because the frames is unorder, so we need to sort it to the right time sequence.
        frames.sort()
        nframes = len(frames)

        if self.split=='train':
            last_sample = -1  ## the index of last_frame (to ensure the correct time sequence)
            sample_frame = []

            nsamples = min(self.sampled_frames, nframes)
            last_frame = None
            for i in range(nsamples):
                if i == 0:
                    last_sample = random.sample(range(0, nframes - nsamples + 1), 1)[0]  ## 保证第一帧的位置在前面，后面的帧的数目才能够。
                # else:
                #     last_sample = random.sample(
                #         range(last_sample + 1, min(last_sample + self.max_skip + 1, nframes - nsamples + i + 1)),
                #         1)[0]
                sample_frame.append(frames[last_sample])
        else:
            # skip = 1
            # times = (idx % self.samples_per_video) +1
            # if times <self.sampled_frames*skip:## 是说当前的帧不够取
            #     temp =  frames[0:times] ## 可以取的帧的数目为times+1
            #     sample_frame = temp + [frames[times-1]] * (self.sampled_frames-len(temp))
            # else:
            #     sample_frame = frames[(times-1-(self.sampled_frames-1)*skip):times:skip]  ## 测试的时候取了全部的帧数据
            times = idx % self.samples_per_video
            sample_frame = [frames[times]]* self.sampled_frames
        # print(os.path.join(imgfolder,sample_frame[0] + ".jpg"))
        frames = [np.array(cv2.imread(os.path.join(imgfolder,name + ".jpg"))) for name in sample_frame]
        # print(len(frames),frames[0].shape)
        target = self.anno[vid][sample_frame[-1]]
        size = target['size']
        lanes_id,lanes_x,lanes_y = target['lanes_id'],target['lanes_x'],target['lanes_y']
        # lanes_id,lanes_x,lanes_y = self.extract_anno(os.path.join(annofolder,sample_frame[-1]+ 'jpg.json'))
        ## 可以做一个对比实验看看，mask 有没有用

        mask = np.ones((frames[0].shape[0], frames[0].shape[1]), dtype=np.uint8)
        sample = {'imgs':frames,'lanes_x':lanes_x,'lanes_y':lanes_y,'mask':mask}
        sample = self.transformer(sample)
        ## 然后得到的是什么么？？
        lanes = [sample['lanes_x'], sample["lanes_y"]]
        info = {'video_name': vid, 'frame_name': sample_frame[-1], 'size': size}
        labels = self.transform_annotation(lanes,info)
        ## 但是每条车道线的车道线数量是不一样的，不一样能摞起来的
        return torch.stack(sample['imgs']),sample['mask'],np.array(labels),info

if __name__ == '__main__':
    from utils.utils import seed_everything
    seed_everything(42)
    import matplotlib.pyplot as plt
    dataset = LaneDataset("train")
    dataloder = DataLoader(dataset,batch_size=2,shuffle=False)
    import time
    start = time.time()
    print(len(dataloder),dataloder.dataset.__len__())
    for index,data in enumerate(dataloder):
        if index !=0 and index % 100==0:
            break
    # end = time.time()-start
    # print(end)
        print(data[0].shape)
        plt.figure()
        imgs = data[0][0]
        for i in range(1,6):
            plt.subplot(2,3,i)
            img = np.transpose(imgs[i-1], (1, 2, 0))
            plt.imshow((img.numpy()* 225).astype(np.uint8))
        plt.subplot(2,3,6)
        gt = data[2][0]
        gt = gt[gt[:, 0] > 0]
        print(len(gt))
        gt_mask = np.zeros((360,640),np.uint8)
        img = np.transpose(imgs[-1].numpy(), (1, 2, 0))
        gt_image = (img* 225).astype(np.uint8).copy()
        for l in range(gt.shape[0]):
            lane = gt[l][3:].reshape((2, -1))
            lane = lane[:, lane[1] > 0].numpy()
            lane[0, :] *= 640
            lane[1, :] *= 360
            lane = np.transpose(lane, (1, 0))
            lane = np.array([lane], np.int64)
            cv2.polylines(gt_image, lane, isClosed=False, color=[255,255,0], thickness=5)
        plt.imshow(gt_image)
        # mask = data[1][0]
        # plt.imshow(mask.numpy()[0].astype(np.uint8))
        # print(mask.size())
        plt.show()
        plt.pause(0)

        #
        # print(data[0].shape, data[1].shape,data[2].shape)
        # imgs = data[0],gt = data[2],
        # img = np.transpose(img[0], (1, 2, 0))
        # print(img)
        # mask = data[1][0][0]
        # plt.figure()
        # plt.subplot(3, 1, 1)
        # plt.imshow((img.numpy()[:,:,::-1]*225).astype(np.uint8))
        # plt.subplot(3, 1, 2)
        # plt.imshow(mask)
        # gt = data[2][0]
        # gt = gt[gt[:, 0] > 0]
        # gt_mask = np.zeros((360,640),np.uint8)
        # for l in range(gt.shape[0]):
        #     lane = gt[l][3:].reshape((2, -1))
        #     lane = lane[:, lane[1] > 0].numpy()
        #     lane[0, :] *= 640
        #     lane[1, :] *= 360
        #     print(lane.shape)
        #     lane = np.transpose(lane, (1, 0))
        #     print(lane.shape, lane.dtype)
        #     print(lane)
        #     lane = np.array([lane], np.int64)
        #     cv2.polylines(gt_mask, lane, isClosed=False, color=1, thickness=5)
        # plt.subplot(3,1,3)
        # plt.imshow(gt_mask)
        # plt.show()

    ## 如何能把剩下的部分进行补全呢？？
