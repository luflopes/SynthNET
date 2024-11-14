import os
import torch
import pandas as pd
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader
import argparse
from src.dataset import Synthbuster
from src.synthnet import SynthNET


def main(
    dataset_path: Path,
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
    ckp_path = os.path.join(exp_dir, "weights", "synthnet-best.pth")
    out_file = os.path.join(exp_dir, "metrics", "metrics.csv")

    # Carregar o checkpoint e fazer uma predição
    model = SynthNET(input_size=(3, img_dim, img_dim), pretrained=True)
    model.load_state_dict(torch.load(ckp_path, weights_only=True))

    corrects = 0
    results = []  # Lista para armazenar os resultados

    for ids, images, labels, datasets, subsets in test_dataloader:
        preds = model.predict(images)
        preds_bin = preds.cpu().numpy()  # Converte para numpy para facilitar o armazenamento
        labels_np = labels.cpu().numpy()

        # Armazenar cada item em um dicionário e adicionar à lista de resultados
        for i in range(len(ids)):
            results.append({
                "id": ids[i],
                "label": labels_np[i],
                "dataset": datasets[i],
                "subset": subsets[i],
                "prediction": preds_bin[i]
            })


    # Converte a lista de resultados para um DataFrame do pandas e salva em CSV
    results_df = pd.DataFrame(results)
    csv_path = os.path.join(exp_dir, "predictions.csv")
    results_df.to_csv(csv_path, index=False)

    # Exibe a acurácia
    accuracy = corrects.double() / len(test_dataset)
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Predictions saved to {csv_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=Path, help="Dataset file path")
    parser.add_argument("--b_size", type=int, default=32, help="Batch size")
    parser.add_argument("--img_dim", type=int, default=224, help="Image dimension")
    parser.add_argument("--out", type=str, default="exp_0", help="Output directory")
    return parser.parse_args()


if __name__ == "__main__":
    main(**vars(parse_args()))
    # Executar o script com o comando:
    # python3 test.py ./data/images.csv --b_size 8 --img_dim 224 --out exp_1
