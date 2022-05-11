import os
import shutil
import tarfile
import urllib.request

import torch
import torchvision.transforms as tt
from PIL import Image


def download():
    if not os.path.exists("./dataset/facades"):
        print("Preparing facades dataset... ", end="")
        file_url = "http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/facades.tar.gz"
        filestream = urllib.request.urlopen(file_url)
        dataset = tarfile.open(fileobj=filestream, mode="r|gz")
        dataset.extractall(path="./dataset")

        os.makedirs("./dataset/facades/mask_only")
        for root, dirs, files in os.walk("./dataset/facades/val"):
            for file in files:
                combined = Image.open(os.path.join(root, file))
                combined = tt.ToTensor()(combined)
                img, mask = torch.tensor_split(combined, 2, dim=2)
                mask = tt.ToPILImage()(mask)
                mask.save(os.path.join("./dataset/facades/mask_only", file))
        shutil.rmtree("./dataset/facades/val")
        print("Done")
    else:
        print("Dataset folder already exists")


if __name__ == "__main__":
    download()
