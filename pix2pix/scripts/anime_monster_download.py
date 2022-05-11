import os
import tarfile

import gdown


def download():
    if not os.path.exists("./dataset/anime_monster"):
        print("Preparing anime_monster dataset... ", end="")
        file_url = "https://drive.google.com/uc?id=1-0Pm3NZeOxDii0ZSn4At0WyYq6f6-AdE"
        if not os.path.exists("anime_monster_dataset.tar.gz"):
            gdown.download(file_url, "anime_monster_dataset.tar.gz", quiet=False)
        dataset = tarfile.open("anime_monster_dataset.tar.gz", mode="r|gz")
        dataset.extractall(path=".")
        # os.remove("anime_monster.tar.gz")
        print("Done")
    else:
        print("Dataset folder already exists")


if __name__ == "__main__":
    download()
