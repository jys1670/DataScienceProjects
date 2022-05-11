import argparse
import importlib.util
import os
import sys

import torch
import torchvision.transforms as tt
from PIL import Image

from src.model import generator
from src.utils.model_state import load as load_model


def main():
    parser = argparse.ArgumentParser(description="Script to generate images with pretrained model")
    parser.add_argument(
        "config",
        type=str,
        help="Path to the model configuration file",
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to the input folder",
    )
    parser.add_argument(
        "output",
        type=str,
        help="Path to the output folder",
    )
    args = parser.parse_args()
    if not (
        os.path.exists(args.config)
        and os.path.exists(args.input)
        and os.path.exists(args.output)
    ):
        print("Invalid path specified")
        sys.exit()
    else:
        cfg_name = str(os.path.basename(args.config)).split(".")[0]
        spec = importlib.util.spec_from_file_location("cfg", args.config)
        cfg = importlib.util.module_from_spec(spec)
        sys.modules["cfg"] = cfg
        spec.loader.exec_module(cfg)

    image_to_tensor = tt.Compose(
        [
            tt.ToTensor(),
            tt.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
        ]
    )
    tensor_to_image = tt.Compose(
        [
            tt.Normalize(
                mean=[-1, -1, -1],
                std=[2, 2, 2],
            ),
            tt.ToPILImage(),
        ]
    )
    gen = generator.GetGenerator(in_channels=cfg.IMG_CHANNELS).to(cfg.DEVICE)
    load_model(cfg_name, cfg.DATASET, cfg.DEVICE, gen)

    count = 0
    with torch.no_grad():
        gen.eval()
        for root, dirs, files in os.walk(args.input):
            for file in files:
                count += 1
                img = Image.open(os.path.join(root, file))
                img = image_to_tensor(img).to(cfg.DEVICE)
                img = torch.unsqueeze(img, dim=0)
                img = gen(img)
                img = torch.squeeze(img, dim=0)
                img = tensor_to_image(img)
                img.save(os.path.join(args.output, file))
                print("Saved", file)
    print("Finished. Processed", count, "files in total.")


if __name__ == "__main__":
    main()
