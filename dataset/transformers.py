import cv2
import math
import numpy as np
import torch
from torchvision.transforms import Normalize as Normalize_th
import random
from copy import deepcopy
from utils.utils import seed_everything
class CustomTransform:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__

    def __eq__(self, name):
        return str(self) == name

    def __iter__(self):
        def iter_fn():
            for t in [self]:
                yield t
        return iter_fn()

    def __contains__(self, name):
        for t in self.__iter__():
            if isinstance(t, Compose):
                if name in t:
                    return True
            elif name == t:
                return True
        return False


class Compose(CustomTransform):
    """
    All transform in Compose should be able to accept two non None variable, img and boxes
    """
    def __init__(self, *transforms):
        # seed_everything(42)
        self.transforms = [*transforms]

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample

    def __iter__(self):
        return iter(self.transforms)

    def modules(self):
        yield self
        for t in self.transforms:
            if isinstance(t, Compose):
                for _t in t.modules():
                    yield _t
            else:
                yield t


def Rotate_Points(origin, point, angle):
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


class Resize(CustomTransform):
    def __init__(self, size):
        if isinstance(size, int):
            size = (size, size)
        self.size = size  #(W, H)

    def __call__(self, sample):
        imgs = sample.get('imgs')
        lanes_x = sample.get("lanes_x")
        lanes_y = sample.get("lanes_y")
        mask = sample.get("mask")
        size_y,size_x = self.size
        h,w = imgs[0].shape[:2]
        ratio_w = size_x * 1.0 / w
        ratio_h = size_y * 1.0 / h
        temp_images = []
        for img in imgs:
            temp_images.append(cv2.resize(img,(size_x,size_y)))
        mask = cv2.resize(mask,(size_x,size_y))
        temp_x = []
        temp_y = []
        for x,y in zip(lanes_x,lanes_y):
            new_x = np.array(x)*ratio_w
            new_y = np.array(y)*ratio_h
            # print("//////////////////////////////////////////", (new_x >= size_x).sum(), (new_y >=size_y).sum())
            index = (new_x >= size_x)*(new_y >=size_y)
            new_x[index] =-2
            new_y[index]= -2
            temp_x.append(new_x)
            temp_y.append(new_y)
        _sample = sample.copy()  ## copy浅拷贝，只是重新创建了一个dict，但是新新变量的每个key都跟原来的变量指向一样的value对象。
        _sample['imgs'] = temp_images
        _sample['lanes_x'] = temp_x
        _sample['lanes_y'] = temp_y
        _sample['mask'] = mask
        return _sample

    def reset_size(self, size):
        if isinstance(size, int):
            size = (size, size)
        self.size = size


class Normalize(CustomTransform):
    def __init__(self, mean, std):
        self.transform = Normalize_th(mean, std)

    def __call__(self, sample):
        imgs = sample.get('imgs')
        temp_images = []
        for img in imgs:
            temp_images.append(self.transform(img))

        _sample = sample.copy()
        _sample['img'] = temp_images
        return _sample


class ToTensor(CustomTransform):
    def __init__(self, dtype=torch.float):
        self.dtype =dtype

    def __call__(self, sample):
        temp_images = []
        imgs = sample.get('imgs')
        for img in imgs:
            img = img[...,::-1].copy()
            img =img.transpose(2,0,1)
            img = torch.from_numpy(img).float() / 255.
            temp_images.append(img)
        # img = sample.get('img')[...,::-1].copy()  ## 在这里进行brg和rgb的转换
        # img = img.transpose(2, 0, 1)
        # img = torch.from_numpy(img).float()/ 255.
        mask = sample.get('mask')
        mask = torch.from_numpy(np.logical_not(np.expand_dims(mask,0)).astype(np.float32))
        _sample = sample.copy()
        _sample['imgs'] = temp_images
        _sample['mask'] = mask
        return _sample
#################################################################################################################
## Add Gaussian noise
#################################################################################################################
class Gaussian(CustomTransform):
    def __init__(self, noise_ratio=0.5):
        self.noise_ratio = noise_ratio

    def __call__(self, sample):
        origin_imgs = sample.get('imgs')
        _sample = sample.copy()
        img = np.zeros((360, 640, 3), np.uint8)
        m = (0, 0, 0)
        s = (20, 20, 20)
        temp_images = []
        if random.random() < self.noise_ratio:
            cv2.randn(img,m,s)
            for origin_img in origin_imgs:
                test_image = deepcopy(origin_img)
                test_image = test_image + img
                temp_images.append(test_image)
            # test_image = np.rollaxis(test_image, axis=2, start=0)
            _sample['imgs'] = temp_images
        return _sample

#################################################################################################################
## Change intensity
## 相当于只改了 hsv 空间的v
#################################################################################################################
class Change_intensity(CustomTransform):
    def __init__(self, intensity_ratio=0.5):
        self.intensity_ratio = intensity_ratio

    def __call__(self, sample):
        imgs = sample.get('imgs')
        _sample = sample.copy()
        if random.random() < self.intensity_ratio:
            temp_images =[]
            value = int(random.uniform(-60.0, 60.0))
            for img in imgs:
                test_image = deepcopy(img)
                hsv = cv2.cvtColor(test_image, cv2.COLOR_BGR2HSV)
                h, s, v = cv2.split(hsv)
                if value > 0:
                    lim = 255 - value
                    v[v > lim] = 255
                    v[v <= lim] += value
                else:
                    lim = -1 * value
                    v[v < lim] = 0
                    v[v >= lim] -= lim
                final_hsv = cv2.merge((h, s, v))
                test_image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
                temp_images.append(test_image)
            # test_image = np.rollaxis(test_image, axis=2, start=0)
            _sample['imgs'] = temp_images
        return _sample
#################################################################################################################
## Generate random shadow in random region
#################################################################################################################
class Shadow(CustomTransform):
    def __init__(self, min_alpha=0.5, max_alpha=0.75,shadow_ratio=0.6):
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.shadow_ratio = shadow_ratio

    def __call__(self, sample):
        imgs = sample.get('imgs')
        _sample = sample.copy()
        if random.random()< self.shadow_ratio:
            top_x, bottom_x = np.random.randint(0, 640, 2)
            coin = np.random.randint(2)
            rand = np.random.randint(2)
            index = random.choice(range(len(sample['imgs'])))
            temp_images = sample['imgs'][:index]
            for img in imgs[index:]:
                test_image = deepcopy(img)
                rows, cols, _ = test_image.shape
                shadow_img = test_image.copy()
                if coin == 0:
                    vertices = np.array([[(50, 65), (45, 0), (145, 0), (150, 65)]], dtype=np.int32)
                    if rand == 0:
                        vertices = np.array([[top_x, 0], [0, 0], [0, rows], [bottom_x, rows]], dtype=np.int32)
                    elif rand == 1:
                        vertices = np.array([[top_x, 0], [cols, 0], [cols, rows], [bottom_x, rows]], dtype=np.int32)
                    mask = test_image.copy()
                    channel_count = test_image.shape[2]  # i.e. 3 or 4 depending on your image
                    ignore_mask_color = (0,) * channel_count
                    cv2.fillPoly(mask, [vertices], ignore_mask_color)
                    rand_alpha = np.random.uniform(self.min_alpha, self.max_alpha)
                    cv2.addWeighted(mask, rand_alpha, test_image, 1 - rand_alpha, 0., shadow_img)
                temp_images.append(shadow_img)
            _sample['imgs'] = temp_images
        return _sample

#################################################################################################################
## Flip
#################################################################################################################
class Flip(CustomTransform):
    def __init__(self, flip_ratio=0.5):
        self.flip_ratio = flip_ratio

    def __call__(self, sample):
        imgs = sample.get('imgs')
        lanes_x = sample.get('lanes_x')
        _sample = sample.copy()
        size_y, size_x = imgs[0].shape[:2]
        temp_images = []
        if random.random()< self.flip_ratio:
            for img in imgs:
                temp_image = deepcopy(img)
                temp_image = cv2.flip(temp_image, 1)
                temp_images.append(temp_image)
            #看来这个麻烦是由于水平翻转导致的
            x = lanes_x

            for i in range(len(x)):
                x[i][x[i] >= 0] = size_x - x[i][x[i] >= 0]
                x[i][x[i] < 0] = -2
                x[i][x[i] >= size_x] = -2

            _sample['imgs'] = temp_images
            _sample['lanes_x'] = x

        return _sample
#################################################################################################################
## Translation
## 这个函数的作用仅仅是一个简单的平移变换
## 横坐标在[-50,50]
## 纵坐标在[-30,30]
#################################################################################################################
class Translation(CustomTransform):
    def __init__(self, trans_ratio=0.8):
        self.trans_ratio = trans_ratio

    def __call__(self, sample):
        imgs = sample.get('imgs')
        lanes_x = sample.get('lanes_x')
        lanes_y = sample.get('lanes_y')
        mask = sample.get('mask')
        _sample = sample.copy()
        size_y, size_x = imgs[0].shape[:2]
        if random.random()< self.trans_ratio:
            tx = np.random.randint(-50, 50)
            ty = np.random.randint(-30, 30)
            temp_images = []
            for img in imgs:
                temp_image = deepcopy(img)
                temp_image = cv2.warpAffine(temp_image, np.float32([[1, 0, tx], [0, 1, ty]]),
                                        (size_x, size_y))
                temp_images.append(temp_image)
            temp_mask = cv2.warpAffine(mask, np.float32([[1, 0, tx], [0, 1, ty]]),
                                        (size_x, size_y))
            # temp_image = np.rollaxis(temp_image, axis=2, start=0)

            x = lanes_x
            ##坐标进行简单评平移
            for j in range(len(x)):
                x[j][x[j] >= 0] = x[j][x[j] >= 0] + tx
                x[j][x[j] < 0] = -2
                x[j][x[j] >= size_x] = -2
            #
            y = lanes_y
            for j in range(len(y)):
                y[j][y[j] >= 0] = y[j][y[j] >= 0] + ty
                x[j][y[j] < 0] = -2
                x[j][y[j] >= size_y] = -2

            _sample['imgs'] = temp_images
            _sample['lanes_x'] = x
            _sample['lanes_y'] = y
            _sample['mask'] = temp_mask
        return _sample
# class supply(CustomTransform):
#     def __init__(self):
#         pass
#     ## 这个函数对我们是不是无效，因为我们生成的直接是参数
#     def __call__(self,sample):
#         img = sample.get('img')
#         height = img.shape[0]
#         lanes_x = sample.get('lanes_x')
#         lanes_y = sample.get('lanes_y')
#         for (xs,ys) in zip(lanes_x,lanes_y):
#             valid_xs = xs[xs>0]
#             valid_ys = ys[xs>0]
#             if abs(valid_ys[-1]

#################################################################################################################
## Rotate
#################################################################################################################
class Rotate(CustomTransform):
    def __init__(self,rotate_ratio=0.8):
        self.rotate_ratio = rotate_ratio

    def __call__(self, sample):
        imgs = sample.get('imgs')
        lanes_x = sample.get('lanes_x')
        lanes_y = sample.get('lanes_y')
        mask = sample.get('mask')
        _sample = sample.copy()
        size_y,size_x = imgs[0].shape[:2]
        if random.random()< self.rotate_ratio:
            angle = np.random.randint(-10, 10)
            M = cv2.getRotationMatrix2D((size_x / 2, size_y / 2), angle, 1)
            temp_images = []
            for img in imgs:
                temp_image = deepcopy(img)
                temp_image = cv2.warpAffine(temp_image, M, (size_x, size_y))
                temp_images.append(temp_image)
            temp_mask = cv2.warpAffine(mask, M, (size_x, size_y))
            # temp_image = np.rollaxis(temp_image, axis=2, start=0)

            x = lanes_x
            y = lanes_y
            for j in range(len(x)):
                index_mask = deepcopy(x[j] > 0)
                x[j][index_mask], y[j][index_mask] = Rotate_Points((size_x / 2, size_y / 2),
                                                                   (x[j][index_mask], y[j][index_mask]),
                                                                   (-angle * 2 * np.pi) / 360)
                x[j][x[j] < 0] = -2
                x[j][x[j] >= size_x] = -2
                x[j][y[j] < 0] = -2
                x[j][y[j] >= size_y] = -2
            _sample['imgs'] = temp_images
            _sample['lanes_x'] = x
            _sample['lanes_y'] = y
            _sample['mask'] = temp_mask
        return _sample

class ColorNoise(CustomTransform):
    def __init__(self,rotate_ratio=1):
        self.rotate_ratio = rotate_ratio

    def __call__(self, sample):
        _sample = sample.copy()
        if random.random()< self.rotate_ratio:
            index = random.choice(range(len(sample['imgs'])))
            _sample['imgs'][index:] = color_jittering_(_sample['imgs'][index:])
        return _sample

# def lighting_(data_rng, images, alphastd, eigval, eigvec):
#     alpha = data_rng.normal(scale=alphastd, size=(3, ))
#     for i in range(len(images)):
#         images[i] += np.dot(eigvec, eigval * alpha)

def blend_(alpha, image1, image2):
    image1 *= alpha
    image2 *= (1 - alpha)
    image1 += image2

def saturation_(images, gs, gs_mean, var):
    alpha = 1. + random.uniform(-var, var)
    for i in range(len(images)):
        blend_(alpha, images[i], gs[i][:, :, None])

def brightness_(images, gs, gs_mean, var):
    alpha = 1. + random.uniform(-var, var)
    for i in range(len(images)):
        images[i] *= alpha

def contrast_(images, gs, gs_mean, var):
    alpha = 1. + random.uniform(-var,var)
    for i in range(len(images)):
        blend_(alpha, images[i], gs_mean[i])


def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def color_jittering_(images):
    functions = [brightness_, contrast_, saturation_]
    random.shuffle(functions)
    gs = []
    gs_mean = []
    for index,image in enumerate(images):
        images[index] = (image / 255.).astype(np.float32)
        temp = grayscale(images[index])
        gs.append(temp.copy())
        gs_mean.append(temp.mean())
    for f in functions:
        f(images, gs, gs_mean, 0.4)

    for index,image in enumerate(images):
        images[index] =(image * 255).astype(np.uint8)
    return images
