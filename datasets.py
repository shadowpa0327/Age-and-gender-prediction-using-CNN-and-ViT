from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd
from pathlib import Path
from torchvision import datasets, transforms

class AFADDatasetAge(Dataset):
    """Custom Dataset for loading AFAD face images"""

    def __init__(self, csv_path, img_dir, transform=None):

        df = pd.read_csv(csv_path, index_col=0)
        self.img_dir = img_dir
        self.csv_path = csv_path
        self.img_paths = df['path']
        self.age = df['age'].values
        self.gender = df['gender'].values
        self.transform = transform
    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir,
                                      self.img_paths[index]))

        if self.transform is not None:
            img = self.transform(img)

        age = self.age[index]
        gender = self.gender[index]

        return img, age, gender

    def __len__(self):
        return self.age.shape[0]


def build_transform():
    custom_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.ToTensor()]
    )
    return custom_transform
    
def build_dataset(args, is_train=False):
    print(f"Building datasets from {args.data_path}")
    if is_train:
        csv_path = Path(args.data_path)/'train.csv'
    else:
        csv_path = Path(args.data_path)/'val.csv'

    transform = build_transform()
    dataset = AFADDatasetAge(csv_path=csv_path, img_dir=args.data_path, transform=transform)
    return dataset