from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
import rawpy


class Synthbuster(Dataset):
    def __init__(self, metadata_file, transform=None, stratify=False):
        self.medadata = pd.read_csv(metadata_file)
        self.transform = transform
        self.stratify = stratify

    def __len__(self):
        return self.medadata.shape[0]

    def __getitem__(self, idx):
        id = self.medadata['id'][idx]
        dataset = self.medadata['dataset'][idx]
        subset = self.medadata['subset'][idx]
        img_path = self.medadata['path'][idx]
        label = self.medadata['label'][idx]

        if self.medadata['extension'][idx] == 'NEF':
            with rawpy.imread(img_path) as raw:
                rgb = raw.postprocess()
                image = Image.fromarray(rgb).convert('RGB')
        else:
            image = Image.open(img_path).convert('RGB')
            
        if self.transform:
            image = self.transform(image)

        if not self.stratify:
            return image, label
        
        return id, image, label, dataset, subset