import os

import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset


class PolypDataset(Dataset):
    class_weights = torch.FloatTensor([0.7, 0.3]).cuda()

    def __init__(self,
                 root,
                 img_path,
                 mask_path,
                 num_classes=2,
                 multi_scale=True,
                 flip=True,
                 base_size=512,
                 crop_size=(256, 512),
                 center_crop_test=False,
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):
        self.base_size = base_size
        self.crop_size = crop_size
        self.mean = mean
        self.std = std
        self.scale_factor = scale_factor
        self.root = root
        self.img_root_path = img_path
        self.mask_root_path = mask_path
        self.num_classes = num_classes
        # self.class_weights = torch.FloatTensor([0.7, 0.3]).cuda()

        self.multi_scale = multi_scale
        self.flip = flip
        self.center_crop_test = center_crop_test

        self.img_list = sorted(
            [f for f in os.listdir(os.path.join(root, img_path)) if f.endswith('.png')])
        self.msk_list = sorted(
            [f for f in os.listdir(os.path.join(root, mask_path)) if f.endswith('.png')])
        inter = [f for f in self.img_list if f in self.msk_list]

        self.img_list = [os.path.join(img_path, f) for f in inter]
        self.msk_list = [os.path.join(mask_path, f) for f in inter]

    def __getitem__(self, index):
        img_path = self.img_list[index]
        msk_path = self.msk_list[index]

        image = cv2.imread(os.path.join(self.root, img_path),
                           cv2.IMREAD_COLOR)
        size = image.shape

        label = cv2.imread(os.path.join(self.root, msk_path),
                           cv2.IMREAD_GRAYSCALE)

        image, label = self.gen_sample(image, label,
                                       self.multi_scale, self.flip,
                                       self.center_crop_test)

        return image.copy(), label.copy(), np.array(size)

    def gen_sample(self, image, label,
                   multi_scale=True, is_flip=True, center_crop_test=False):
        if multi_scale:
            rand_scale = 0.5 + random.randint(0, self.scale_factor) / 10.0
            image, label = self.multi_scale_aug(image, label,
                                                rand_scale=rand_scale)

        if center_crop_test:
            image, label = self.image_resize(image,
                                             self.base_size,

                                             label=label)
            image, label = self.center_crop(image, label)

        image = self.input_transform(image)
        label = self.label_transform(label)

        image = image.transpose((2, 0, 1))

        if is_flip:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        return image, label

    def multi_scale_aug(self, image, label=None,
                        rand_scale=1, rand_crop=True):
        long_size = np.int(self.base_size * rand_scale + 0.5)
        if label is not None:
            image, label = self.image_resize(image, long_size, label=label)
            if rand_crop:
                image, label = self.rand_crop(image, label)
                # lh, lw = label.shape[:2]
                # label = cv2.resize(label, (lw // 4, lh // 4),
                #                    interpolation=cv2.INTER_NEAREST)
            return image, label
        else:
            image = self.image_resize(image, long_size)
            return image

    def image_resize(self, image, long_size, label=None):
        h, w = image.shape[:2]
        if h > w:
            new_h = long_size
            new_w = np.int(w * long_size / h + 0.5)
        else:
            new_w = long_size
            new_h = np.int(h * long_size / w + 0.5)
        image = cv2.resize(image, (new_w, new_h),
                           interpolation=cv2.INTER_LINEAR)
        if label is not None:
            label = cv2.resize(label, (new_w, new_h),
                               interpolation=cv2.INTER_NEAREST)
        else:
            return image

        return image, label

    def __len__(self):
        return len(self.img_list)

    def input_transform(self, image):
        image = image.astype(np.float32)[:, :, ::-1]
        image = image / 255.0
        image -= self.mean
        image /= self.std
        return image

    def label_transform(self, label):
        label = np.array(label)
        return np.where(label < 127, 0, 1)

    def pad_image(self, image, h, w, size, padvalue=0):
        pad_image = image.copy()
        pad_h = max(size[0] - h, 0)
        pad_w = max(size[1] - w, 0)
        if pad_h > 0 or pad_w > 0:
            pad_image = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                           pad_w, cv2.BORDER_CONSTANT,
                                           value=padvalue)

        return pad_image

    def rand_crop(self, image, label):
        h, w = image.shape[:-1]
        image = self.pad_image(image, h, w, self.crop_size,
                               (0.0, 0.0, 0.0))
        label = self.pad_image(label, h, w, self.crop_size)

        new_h, new_w = label.shape
        x = random.randint(0, new_w - self.crop_size[1])
        y = random.randint(0, new_h - self.crop_size[0])
        image = image[y:y + self.crop_size[0], x:x + self.crop_size[1]]
        label = label[y:y + self.crop_size[0], x:x + self.crop_size[1]]

        return image, label

    def center_crop(self, image, label):
        h, w = image.shape[:2]
        x = int(round((w - self.crop_size[1]) / 2.))
        y = int(round((h - self.crop_size[0]) / 2.))
        image = image[y:y + self.crop_size[0], x:x + self.crop_size[1]]
        label = label[y:y + self.crop_size[0], x:x + self.crop_size[1]]

        return image, label


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dataset = PolypDataset(root=r"D:\LTGiang\MSV-20194430\project 2\mdeq\data\TrainDataset", img_path="image",
                           mask_path="mask")
    img, label, size = dataset[0]