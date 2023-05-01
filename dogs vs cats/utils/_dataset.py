import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import ToTensor, Resize, Normalize
from glob import glob
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, List

class DogVsCats(Dataset):
    def __init__(
        self,
        X: List,
        y: List
    ) -> None:
        self.X = X
        self.y = y
    
    def __len__(self) -> int:
        return len(self.y)
    
    def __getitem__(
        self,
        index: int
    ) -> Tuple:
        return self.X[index], self.y[index]
    
def create_dataloader(
    X: List,
    y: List,
    batch_size: int,
    num_workers: int,
    shuffle: bool
) -> DataLoader:
    """
    Args:
        X (pd.DataFrame): the data's feature.
        y (pd.DataFrame): the data's label.
        batch_size (int): the size of the batch.
        num_workers (int): the number of workers to use.
        shuffle (bool): shuffle the data or not.

    Returns:
        DataLoader: creates a dataloader instance.
    """
    dataset = DogVsCats(
        X=X,
        y=y
    )
    
    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle
    )
    
    return dataloader

def read_data(
    input_dir: str,
    inference_mode: bool = False
) -> pd.DataFrame:
    """
    Reads the input data.

    Args:
        input_dir (str): the input directory folder.
        inference_mode (bool): boolean indicating whether is inference mode or not.

    Returns:
        pd.DataFrame: the data in the dataframe format.
    """
    data = glob(f"{input_dir}/*.jpg")
    df = pd.DataFrame()
    
    converter = ToTensor()
    transformer = torch.nn.Sequential(
        Resize((120, 120), antialias=True),
        # Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    )
    
    for d in data:
        name = d.split("/")[-1].replace(".jpg", "")
        
        if inference_mode:
            label = -1
        else:
            label = 1 if "dog" in d.split("/")[-1] else 0
        
        image = Image.open(d)
        image = converter(image)
        image = transformer(image)
        row = pd.DataFrame({
            "name": [name],
            "data": [image],
            "label": [label]
        })
        df = pd.concat([df, row], axis=0)
    
    df = df.reset_index(drop=True)
    return df