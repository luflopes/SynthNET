import os
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader
import argparse
from src.dataset import Synthbuster
from src.synthnet import SynthNET
from torch import nn
import torch.optim as optim

def main(
    train_dataset: Path,
    eval_dataset: Path,
    ckp:Path,
    freeze_conv:bool,
    b_size:int,
    img_dim:int,
    epochs:int,
    out:str
    ):

    img_transform = transforms.Compose([
        #transforms.Resize((img_dim, img_dim)),
        transforms.ToTensor()
    ])

    train_dataset = Synthbuster(metadata_file=train_dataset, transform=img_transform)
    eval_dataset = Synthbuster(metadata_file=eval_dataset, transform=img_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=b_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=b_size, shuffle=False)

    root_dir = os.path.join(os.getcwd(), "experiments")
    exp_dir = os.path.join(root_dir, out if out is not None else "exp_0")

    model = SynthNET(ckp=ckp, input_size=(3, img_dim, img_dim), freeze_conv=freeze_conv)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    model.train_model(
        root_dir=exp_dir,
        dataloader={
            "train": train_dataloader,
            "eval": eval_dataloader
        },
        num_epochs=epochs,
        optimizer=optimizer,
        criterion=criterion
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_dataset", type=Path, help="Train Dataset file path")
    parser.add_argument("eval_dataset", type=Path, help="Evaluation Dataset file path")
    parser.add_argument("--ckp", type=Path, default=None, help="Load model checkpoint")
    parser.add_argument("--freeze_conv", type=bool, default=True, help="Load model checkpoint")
    parser.add_argument("--b_size", type=int, default=32, help="Batch size")
    parser.add_argument("--img_dim", type=int, default=224, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Batch size")
    parser.add_argument("--out", type=Path, default="exp_0", help="Output directory")
    
    return parser.parse_args()


if __name__ == "__main__":
    main(**vars(parse_args()))

    # Train new model
    # python3 train.py ./data/train.csv ./data/eval.csv --b_size 64 --img_dim 224 --epochs 2 --out exp_1

    # Start training from previous checkpoint
    # python3 train.py ./data/train.csv ./data/eval.csv --ckp ./experiments/exp_1/weights/synthnet-best.pth --b_size 64 --img_dim 224 --epochs 2 --out exp_2