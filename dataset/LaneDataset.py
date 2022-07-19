import cv2
import numpy as np
import cv2
import numpy as np
import random
import imgaug.augmenters as iaa
from imgaug.augmenters import Resize
from torchvision.transforms import ToTensor
from torch.utils.data.dataset import Dataset
from imgaug.augmentables.lines import LineString, LineStringsOnImage
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
from dataset.tusimple import TuSimple
import os
GT_COLOR = (255, 0, 0)
PRED_HIT_COLOR = (0, 255, 0)
PRED_MISS_COLOR = (0, 0, 255)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])
eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
eig_vec = np.array([
    [-0.58752847, -0.69563484, 0.41340352],
    [-0.5832747, 0.00994535, -0.81221408],
    [-0.56089297, 0.71832671, 0.41158938]
], dtype=np.float32)
mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32)
std = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32)

class LaneDataset(Dataset):
    def __init__(self,
                 dataset='tusimple',
                 augmentations=None,
                 normalize=True,
                 split='train',
                 img_size=(360, 640),
                 aug_chance=1.,
                 **kwargs):
        super(LaneDataset, self).__init__()
        self.split = split
        if dataset == 'tusimple':
            self.dataset = TuSimple(split=split,root = "E:/data/Tusimple/LaneDetection",**kwargs)
        else:
            raise NotImplementedError()
        ## self.transform_annotations()开始就行了这个函数
        self.img_h, self.img_w = img_size
        if augmentations is not None:
            # add augmentations
            augmentations = [getattr(iaa, aug['name'])(**aug['parameters'])
                             for aug in augmentations]  # add augmentation
        self._data_rng = np.random.RandomState(os.getpid())
        self.normalize = normalize
        self.transformations = iaa.Sequential([Resize({'height': self.img_h, 'width': self.img_w})])
        self.aug_chance = aug_chance
        self.augmentations = augmentations
        self.to_tensor = ToTensor()
        # self.transform = iaa.Sequential([iaa.Sometimes(then_list=augmentations, p=aug_chance), transformations])
        self.max_lanes = self.dataset.max_lanes

    def lane_to_linestrings(self, lanes):
        lines = []
        for lane in lanes:
            lines.append(LineString(lane))

        return lines

    def linestrings_to_lanes(self, lines):
        lanes = []
        for line in lines:
            lanes.append(line.coords)

        return lanes

    def __getitem__(self, idx, transform=True):
        item = self.dataset[idx]
        transform = iaa.Sequential([iaa.Sometimes(then_list=self.augmentations, p=self.aug_chance), self.transformations])
        transform = transform.to_deterministic()
        img = cv2.imread(item['path'],cv2.IMREAD_COLOR)
        mask = np.ones((1,img.shape[0], img.shape[1], 1), dtype=np.bool)
        label = item['label']
        if transform:
            line_strings = self.lane_to_linestrings(item['old_anno']['lanes'])
            line_strings = LineStringsOnImage(line_strings, shape=img.shape)
            ##就是说同时对这些点也进行变换
            img, line_strings,mask = transform(image=img, line_strings=line_strings,segmentation_maps = mask)
            line_strings.clip_out_of_image_()
            new_anno = {'path': item['path'], 'lanes': self.linestrings_to_lanes(line_strings)}
            new_anno['categories'] = item['categories']
            label = self.dataset.transform_annotation(new_anno, img_wh=(self.img_w, self.img_h))['label']
        ## 这里的作用是让lower一样，尽可能地小
        tgt_ids   = label[:, 0]
        last = label[tgt_ids ==0]
        label = label[tgt_ids > 0]

        # make lower the same ## 使得lower相同
        label[:, 1][label[:, 1] < 0] = 1
        label[:, 1][...] = np.min(label[:, 1])
        label = np.vstack([label, last])
        name = "/".join(item['path'].split("/")[:-1])
        imgs= []
        for i in range(4,-1,-1):
            origin = cv2.imread(os.path.join(name+ "/" + str(20-i*2)+".jpg"),cv2.IMREAD_COLOR)
            origin = (transform(image = origin)/255.).astype(np.float32)
            imgs.append(origin)
        # img = (img / 255.).astype(np.float32)
        ## 在这里进行一下可视化部分

        color_jittering_(self._data_rng, imgs)
        lighting_(self._data_rng, imgs, 0.1, eig_val, eig_vec)
        if self.normalize:
            for i in range(len(imgs)):
                imgs[i] = self.to_tensor((imgs[i]-mean)/std)  ## 这里搞了两遍，当然不行了
                # print(imgs[i].dtype)
            # img = (img - mean) / std
        # plt.figure()
        # plt.subplot(1,3,1)
        # plt.imshow(img)
        # plt.subplot(1,3,2)
        # plt.imshow(imgs[-1])
        # plt.subplot(1,3,3)
        # plt.imshow(imgs[0])
        # plt.show()
        # plt.pause(0)
        # imgs = self.to_tensor(img.astype(np.float32))
        mask = torch.from_numpy(np.logical_not(mask).astype(np.float32))

        return (torch.stack(imgs), mask, label, idx)

    def __len__(self):
        return len(self.dataset)


    def draw_annotation(self, idx, pred=None, img=None, cls_pred=None):
        if img is None:
            img, label, _ = self.__getitem__(idx, transform=True)
            # Tensor to opencv image
            img = img.permute(1, 2, 0).numpy()
            # Unnormalize
            if self.normalize:
                img = img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
            img = (img * 255).astype(np.uint8)
        else:
            _, label, _ = self.__getitem__(idx)

        img_h, img_w, _ = img.shape

        # Draw label
        for i, lane in enumerate(label):
            if lane[0] == 0:  # Skip invalid lanes
                continue
            lane = lane[3:]  # remove conf, upper and lower positions
            xs = lane[:len(lane) // 2]
            ys = lane[len(lane) // 2:]
            ys = ys[xs >= 0]
            xs = xs[xs >= 0]

            # draw GT points
            for p in zip(xs, ys):
                p = (int(p[0] * img_w), int(p[1] * img_h))
                img = cv2.circle(img, p, 5, color=GT_COLOR, thickness=-1)

            # draw GT lane ID
            cv2.putText(img,
                        str(i), (int(xs[0] * img_w), int(ys[0] * img_h)),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX,
                        fontScale=1,
                        color=(0, 255, 0))

        if pred is None:
            return img

        # Draw predictions
        pred = pred[pred[:, 0] != 0]  # filter invalid lanes
        matches, accs, _ = self.dataset.get_metrics(pred, idx)
        overlay = img.copy()
        for i, lane in enumerate(pred):
            if matches[i]:
                color = PRED_HIT_COLOR
            else:
                color = PRED_MISS_COLOR
            lane = lane[1:]  # remove conf
            lower, upper = lane[0], lane[1]
            lane = lane[2:]  # remove upper, lower positions

            # generate points from the polynomial
            ys = np.linspace(lower, upper, num=100)
            points = np.zeros((len(ys), 2), dtype=np.int32)
            points[:, 1] = (ys * img_h).astype(int)
            points[:, 0] = (np.polyval(lane, ys) * img_w).astype(int)
            points = points[(points[:, 0] > 0) & (points[:, 0] < img_w)]

            # draw lane with a polyline on the overlay
            for current_point, next_point in zip(points[:-1], points[1:]):
                overlay = cv2.line(overlay, tuple(current_point), tuple(next_point), color=color, thickness=2)

            # draw class icon
            if cls_pred is not None and len(points) > 0:
                class_icon = self.dataset.get_class_icon(cls_pred[i])
                class_icon = cv2.resize(class_icon, (32, 32))
                mid = tuple(points[len(points) // 2] - 60)
                x, y = mid

                img[y:y + class_icon.shape[0], x:x + class_icon.shape[1]] = class_icon

            # draw lane ID
            if len(points) > 0:
                cv2.putText(img, str(i), tuple(points[0]), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=color)

            # draw lane accuracy
            if len(points) > 0:
                cv2.putText(img,
                            '{:.2f}'.format(accs[i] * 100),
                            tuple(points[len(points) // 2] - 30),
                            fontFace=cv2.FONT_HERSHEY_COMPLEX,
                            fontScale=.75,
                            color=color)
        # Add lanes overlay
        w = 0.6
        img = ((1. - w) * img + w * overlay).astype(np.uint8)

        return img
def lighting_(data_rng, images, alphastd, eigval, eigvec):
    alpha = data_rng.normal(scale=alphastd, size=(3, ))
    for i in range(len(images)):
        images[i] += np.dot(eigvec, eigval * alpha)

def blend_(alpha, image1, image2):
    image1 *= alpha
    image2 *= (1 - alpha)
    image1 += image2

def saturation_(data_rng, images, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    for i in range(len(images)):
        blend_(alpha, images[i], gs[i][:, :, None])

def brightness_(data_rng, images, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    for i in range(len(images)):
        images[i] *= alpha

def contrast_(data_rng, images, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    for i in range(len(images)):
        blend_(alpha, images[i], gs_mean[i])


def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def color_jittering_(data_rng, images):
    functions = [brightness_, contrast_, saturation_]
    random.shuffle(functions)
    gs = []
    gs_mean = []
    for image in images:
        temp = grayscale(image)
        gs.append(temp.copy())
        gs_mean.append(temp.mean())
    for f in functions:
        f(data_rng, images, gs, gs_mean, 0.4)

def main():
    dataset = LaneDataset(dataset='tusimple',augmentations=None,normalize=False,split='train+val',img_size=(360, 640),aug_chance=1.)
    loader = DataLoader(dataset=dataset,batch_size=16,shuffle =True)
    for idx,batch_data in enumerate(loader):
        ## 输出的信息是image和label,这些都是有待处理的。
        print(len(batch_data))
        print(batch_data[0].size(),batch_data[1].size(),batch_data[2].size())
        if idx==1:
            break
    # import torch
    # # from lib.config import Config
    # np.random.seed(0)
    # torch.manual_seed(0)
    # # cfg = Config('config.yaml')
    # # train_dataset = cfg.get_dataset('train')
    #
    # for idx in range(len(train_dataset)):
    #     img = train_dataset.draw_annotation(idx)
    #     cv2.imshow('sample', img)
    #     cv2.waitKey(0)


if __name__ == "__main__":
    from config import sys_config

    dataset = sys_config.get_dataset('train')
    import time
    start = time.time()
    loader = DataLoader(dataset,shuffle=False,batch_size=2)
    for index,data in enumerate(loader):
        pass
    #     if index != 0 and index % 100 == 0:
    #         break
    # end = time.time() - start
    # print(end)
        print(data[0].shape,data[1].shape)
        img = np.transpose(data[0][0][0],(1,2,0))
        mask = data[1][0][0][:,:,0]
        gt = data[2][0]
        plt.figure()
        plt.subplot(3,1,1)
        plt.imshow(img.numpy())
        plt.subplot(3,1,2)
        plt.imshow(mask)

        gt = gt[gt[:, 0] > 0]
        # gt_mask = np.zeros((288,800),np.uint8)
        gt_image = img.numpy().copy()
        for l in range(gt.shape[0]):
            lane = gt[l][3:].reshape((2, -1))
            lane = lane[:, lane[1] > 0].numpy()
            lane[0, :] *= 640
            lane[1, :] *= 360
            lane = np.transpose(lane, (1, 0))
            lane = np.array([lane], np.int64)
            cv2.polylines(gt_image, lane, isClosed=False, color=1, thickness=5)
        plt.subplot(3, 1, 3)
        plt.imshow(gt_image)
        plt.show()
