import os

import torch


def save(cfg_name, dataset_name, gen, dis, gen_op, dis_op):
    path = "./output/" + dataset_name
    os.makedirs(path, exist_ok=True)
    filename = cfg_name + "_" + dataset_name + ".tar"
    path = os.path.join(path, filename)
    torch.save(
        {
            "dis_state_dict": dis.state_dict(),
            "gen_state_dict": gen.state_dict(),
            "dis_op_state_dict": dis_op.state_dict(),
            "gen_op_state_dict": gen_op.state_dict(),
        },
        path,
    )
    print("Model was saved as", path)


def load(cfg_name, dataset_name, device, gen, dis=None, gen_op=None, dis_op=None):
    path = os.path.join(
        "./output/" + dataset_name, cfg_name + "_" + dataset_name + ".tar"
    )
    if not os.path.exists(path):
        print("Model state was not found")
    else:
        state_dict = torch.load(path, map_location=device)
        gen.load_state_dict(state_dict["gen_state_dict"])
        if dis is not None:
            dis.load_state_dict(state_dict["dis_state_dict"])
        if dis_op is not None:
            dis_op.load_state_dict(state_dict["dis_op_state_dict"])
        if gen_op is not None:
            gen_op.load_state_dict(state_dict["gen_op_state_dict"])
        print("Model was loaded from", path)
