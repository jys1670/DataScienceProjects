import argparse
import os
import sys
import importlib.util
from src.model import discriminator, generator
from src.utils import model_state
import subprocess
import torch
import matplotlib.pyplot as plt


def train():
    train_loader, test_loader = dataset.prepare(cfg.BATCH_SIZE)
    dis = discriminator.GetDiscriminator(in_channels=cfg.IMG_CHANNELS).to(cfg.DEVICE)
    gen = generator.GetGenerator(in_channels=cfg.IMG_CHANNELS).to(cfg.DEVICE)
    dis_op = torch.optim.Adam(
        dis.parameters(),
        lr=cfg.LEARNING_RATE,
        betas=(0.5, 0.999),
    )
    gen_op = torch.optim.Adam(
        gen.parameters(),
        lr=cfg.LEARNING_RATE,
        betas=(0.5, 0.999),
    )
    bce_loss = torch.nn.BCEWithLogitsLoss()
    l1_loss = torch.nn.L1Loss()

    if cfg.LOAD_SAVED_ON_TRAIN:
        print("Attempting to load model state...")
        model_state.load(cfg_name, cfg.DATASET, cfg.DEVICE, gen, dis, gen_op, dis_op)
        gen.train()
        dis.train()

    print("Evaluating train loader...")
    for epoch in range(cfg.NUM_EPOCHS):
        for ind, (mask, img) in enumerate(train_loader):
            mask = mask.to(cfg.DEVICE)
            img = img.to(cfg.DEVICE)
            fake_img = gen(mask)

            # Discriminator
            dis.zero_grad()
            dis_real = dis(mask, img)
            dis_loss_real = bce_loss(dis_real, torch.ones_like(dis_real))
            dis_fake = dis(mask, fake_img.detach())
            dis_loss_fake = bce_loss(dis_fake, torch.zeros_like(dis_fake))
            dis_loss_total = 0.5 * (dis_loss_real + dis_loss_fake)
            dis_loss_total.backward()
            dis_op.step()

            # Generator
            gen.zero_grad()
            dis_fake = dis(mask, fake_img)
            gen_loss_total = cfg.LAMBDA * l1_loss(fake_img, img) + bce_loss(
                dis_fake, torch.ones_like(dis_fake)
            )
            gen_loss_total.backward()
            gen_op.step()

            # Print train stats
            print(
                "[{:>3d}/{:>3d}][{:>3d}/{:>3d}]   Loss_D: {:>3.4f}   Loss_G: {:>3.4f}".format(
                    epoch,
                    cfg.NUM_EPOCHS,
                    ind,
                    len(train_loader),
                    dis_loss_total.item(),
                    gen_loss_total.item(),
                )
            )

    print("Done")
    if cfg.SAVE_MODEL:
        model_state.save(cfg_name, cfg.DATASET, gen, dis, gen_op, dis_op)

    print("Evaluating test loader...")
    os.makedirs("./output/" + cfg.DATASET + "/results/test", exist_ok=True)
    to_image = test_loader.dataset.TensorToImage
    images_on_plot = 4
    with torch.no_grad():
        gen.eval()
        for batch_ind, (mask, img) in enumerate(test_loader):
            mask = mask.to(cfg.DEVICE)
            img = img.to(cfg.DEVICE)
            gen_imgs = gen(mask)
            ind = batch_ind * cfg.BATCH_SIZE
            for i in range(len(mask)):
                ind += i
                if ind % images_on_plot == 0:
                    fig = plt.figure(figsize=(15, 15))
                    spec = fig.add_gridspec(
                        ncols=images_on_plot, nrows=6, height_ratios=[1, 9, 1, 9, 1, 9]
                    )
                    fig.subplots_adjust(hspace=0.03, wspace=0.03)

                    mask_ax = fig.add_subplot(spec[0, :])
                    mask_ax.text(0.5, 0.5, "Mask:", ha="center", va="center", size=20)
                    mask_ax.axis("off")

                    gen_ax = fig.add_subplot(spec[2, :])
                    gen_ax.text(
                        0.5, 0.5, "Generator output:", ha="center", va="center", size=20
                    )
                    gen_ax.axis("off")

                    orig_ax = fig.add_subplot(spec[4, :])
                    orig_ax.text(
                        0.5, 0.5, "Original:", ha="center", va="center", size=20
                    )
                    orig_ax.axis("off")

                f1_ax = fig.add_subplot(spec[1, ind % images_on_plot])
                f1_ax.axis("off")
                f1_ax.imshow(to_image(mask[i]))

                f2_ax = fig.add_subplot(spec[3, ind % images_on_plot])
                f2_ax.axis("off")
                f2_ax.imshow(to_image(gen_imgs[i]))

                f3_ax = fig.add_subplot(spec[5, ind % images_on_plot])
                f3_ax.axis("off")
                f3_ax.imshow(to_image(img[i]))

                if (
                    ind % images_on_plot == images_on_plot - 1
                    or ind == len(test_loader) * cfg.BATCH_SIZE - 1
                ):
                    path = os.path.join(
                        "./output/" + cfg.DATASET + "/results/test/", str(ind) + ".png"
                    )
                    fig.savefig(path)
                    print("Saved", path)
                    plt.close("all")
    print("Training finished")


def main():
    parser = argparse.ArgumentParser(description="Script to train pix2pix")
    parser.add_argument(
        "-c",
        type=str,
        help="Path to the configuration file to use for the training",
        default=os.path.join(os.getcwd(), "configs/default_config.py"),
        dest="config",
    )
    args = parser.parse_args()
    if not os.path.exists(args.config):
        print("Configuration file was not found")
        sys.exit()
    else:
        print("Found configuration file, loading it as module... ", end="")
        global cfg, cfg_name
        cfg_name = str(os.path.basename(args.config)).split(".")[0]
        spec = importlib.util.spec_from_file_location("cfg", args.config)
        cfg = importlib.util.module_from_spec(spec)
        sys.modules["cfg"] = cfg
        spec.loader.exec_module(cfg)
        print("Done")

    print('Dataset was set to "', cfg.DATASET, '"; running download script...')
    command = "python scripts/" + cfg.DATASET + "_download.py"
    p = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    print(p.communicate()[0].decode("ascii"), end="")

    print("Preparing DataLoader module... ", end="")
    global dataset
    spec = importlib.util.spec_from_file_location(
        "dataset", "./src/data/" + cfg.DATASET + "_dataset.py"
    )
    dataset = importlib.util.module_from_spec(spec)
    sys.modules["dataset"] = dataset
    spec.loader.exec_module(dataset)
    print("Done")

    train()


if __name__ == "__main__":
    main()
