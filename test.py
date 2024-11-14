import os
import torch
import csv
import pandas as pd
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader
import argparse
from src.dataset import Synthbuster
from src.synthnet import SynthNET


def main(
    dataset_path: Path,
    ckp: Path,
    b_size: int,
    img_dim: int,
    out: str
    ):

    img_transform = transforms.Compose([
        transforms.Resize((img_dim, img_dim)),
        transforms.ToTensor()
    ])

    test_dataset = Synthbuster(metadata_file=dataset_path, transform=img_transform, stratify=True)
    test_dataloader = DataLoader(test_dataset, batch_size=b_size, shuffle=False)

    root_dir = os.path.join(os.getcwd(), "experiments")
    exp_dir = os.path.join(root_dir, out if out is not None else "exp_0")
    out_file = os.path.join(exp_dir, "metrics", "metrics.csv")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = SynthNET(ckp=ckp, input_size=(3, img_dim, img_dim), device=device)
    model.eval()
    model.make_dirs(exp_dir)

    results = []
 
    for ids, images, labels, datasets, subsets in test_dataloader:
        images = images.to(device)
        labels = labels.to(device)
        preds = model.predict(images)
        preds_bin = preds.cpu().numpy()
        labels_np = labels.cpu().numpy()

        for i in range(len(ids)):
            results.append({
                "id": ids[i],
                "label": labels_np[i],
                "dataset": datasets[i],
                "subset": subsets[i],
                "prediction": preds_bin[i]
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(out_file, index=False)

    print(f"Predictions saved to {out_file}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=Path, help="Dataset file path")
    parser.add_argument("--ckp", type=Path, help="Checkpoint model weights")
    parser.add_argument("--b_size", type=int, default=32, help="Batch size")
    parser.add_argument("--img_dim", type=int, default=224, help="Image dimension")
    parser.add_argument("--out", type=str, default="exp_0", help="Output directory")
    
    return parser.parse_args()


if __name__ == "__main__":
    main(**vars(parse_args()))

    # Test model:
    # python3 test.py ./data/test.csv --ckp ./experiments/exp_1/weights/synthnet-best.pth --b_size 64 --img_dim 224 --out exp_1
