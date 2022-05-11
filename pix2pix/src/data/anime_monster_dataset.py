import os
import random

import torchvision.transforms as tt
import torchvision.transforms.functional as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


class ImgDataset(Dataset):
    def __init__(self, or_files, sk_files, mode):
        assert len(or_files) == len(sk_files)
        super(ImgDataset, self).__init__()
        self.or_files = sorted(or_files)
        self.sk_files = sorted(sk_files)
        self.mode = mode
        self.ImageToTensor = tt.Compose(
            [tt.ToTensor(), tt.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
        )
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
        img = Image.open(self.or_files[index])
        mask = Image.open(self.sk_files[index])
        img = self.ImageToTensor(img)
        mask = self.ImageToTensor(mask)
        if self.mode == "train":
            mask, img = self.random_transform(mask, img)

        return mask, img

    def __len__(self):
        return len(self.or_files)

    def random_transform(self, mask, img):
        # Random horizontal flipping
        if random.random() > 0.5:
            mask = tf.hflip(mask)
            img = tf.hflip(img)

        return mask, img


def prepare(batch_size):
    if not os.path.exists("./dataset/anime_monster"):
        print("Dataset folder does not exist")
    original_files = []
    sketch_files = []
    for root, dirs, files in os.walk("./dataset/anime_monster"):
        for img in files:
            if root.endswith(os.path.join("anime_monster", "original")):
                original_files.append(os.path.join(root, img))
            if root.endswith(os.path.join("anime_monster", "sketch")):
                sketch_files.append(os.path.join(root, img))
    print(
        "Found", len(original_files), "original and", len(sketch_files), "sketch files"
    )
    original_files = sorted(original_files)
    sketch_files = sorted(sketch_files)
    or_files_train, or_files_test, sk_files_train, sk_files_test = train_test_split(
        original_files, sketch_files, test_size=0.05
    )
    train_dataset = ImgDataset(or_files_train, sk_files_train, "train")
    test_dataset = ImgDataset(or_files_test, sk_files_test, "test")
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
