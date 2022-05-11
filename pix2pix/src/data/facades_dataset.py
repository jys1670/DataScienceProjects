import os
import random

import torch
import torchvision.transforms as tt
import torchvision.transforms.functional as tf
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class ImgDataset(Dataset):
    def __init__(self, files, mode):
        super(ImgDataset, self).__init__()
        self.files = sorted(files)
        self.mode = mode
        self.ImageToTensor = tt.ToTensor()
        self.NormalizeImage = tt.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.TensorToImage = tt.Compose(
            [
                tt.Normalize(
                    mean=[-1, -1, -1],
                    std=[2, 2, 2],
                ),
                tt.ToPILImage(),
            ]
        )

    def __getitem__(self, index):
        combined = Image.open(self.files[index])
        combined = self.ImageToTensor(combined)
        img, mask = torch.tensor_split(combined, 2, dim=2)
        if self.mode == "train":
            mask, img = self.random_transform(mask, img)

        mask = self.NormalizeImage(mask)
        img = self.NormalizeImage(img)
        return mask, img

    def __len__(self):
        return len(self.files)

    def random_transform(self, mask, img):
        # Random horizontal flipping
        if random.random() > 0.5:
            mask = tf.hflip(mask)
            img = tf.hflip(img)

        # Random jitter
        if random.random() > 0.5:
            c, h, w = mask.size()
            crop_h = random.randint(int(h * 0.895), h) # 256/286 = 0.895...
            crop_w = crop_h
            crop_y_cord = random.randint(0, h - crop_h)
            crop_x_cord = random.randint(0, w - crop_w)
            mask = tf.resized_crop(
                img=mask,
                top=crop_y_cord,
                left=crop_x_cord,
                height=crop_h,
                width=crop_w,
                size=[h, w],
            )
            img = tf.resized_crop(
                img=img,
                top=crop_y_cord,
                left=crop_x_cord,
                height=crop_h,
                width=crop_w,
                size=[h, w],
            )

        return mask, img


def prepare(batch_size):
    if not os.path.exists("./dataset/facades"):
        print("Dataset folder does not exist")
    train_files = []
    test_files = []
    for root, dirs, files in os.walk("./dataset/facades"):
        for img in files:
            if root.endswith(os.path.join("facades", "train")):
                train_files.append(os.path.join(root, img))
            if root.endswith(os.path.join("facades", "test")):
                test_files.append(os.path.join(root, img))
    print("Found", len(train_files), "train and", len(test_files), "test files")

    train_dataset = ImgDataset(train_files, "train")
    test_dataset = ImgDataset(test_files, "test")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, test_loader
