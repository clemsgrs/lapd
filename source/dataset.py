import h5py
import torch
import numpy as np
import pandas as pd
from typing import Callable, Dict
from pathlib import Path
from collections import defaultdict

class SingleContourTensorDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df,
        features_dir: Path,
        tile_size: int,
        fmt: str = 'jpg',
        emb_size: int = 192,
        label_name: str = "label",
        label_mapping: Dict[int, int] = {},
        transform: Callable = None,
    ):
        """_summary_

        Args:
            df (pd.DataFrame): dataframe containing following columns 'slide_id', 'x', 'y'
            features_dir (Path): _description_
            tile_size (int): _description_
            fmt (str, optional): _description_. Defaults to 'jpg'.
            emb_size (int, optional): _description_. Defaults to 192.
            transform (Callable, optional): _description_. Defaults to None.
        """
        self.features_dir = features_dir
        self.tile_size = tile_size
        self.fmt = fmt
        self.emb_size = emb_size
        self.label_name = label_name
        self.label_mapping = label_mapping
        self.transform = transform

        self.tile_df = self.prepare_data(df)
        self.slide_df = self.tile_df.drop_duplicates(['slide_id'], ignore_index=True)

    def prepare_data(self, df, ntile_min: int = -1):
        if self.label_mapping:
            df["label"] = df[self.label_name].apply(lambda x: self.label_mapping[x])
        elif self.label_name != "label":
            df["label"] = df[self.label_name]
        tmp = df.groupby(['slide_id', 'contour']).contour.size().to_frame().rename(columns={'contour': 'ntile'}).reset_index()
        self.filtered_contours = defaultdict(dict)
        for slide_id, contour, ntile in tmp[tmp.ntile >= ntile_min].values:
            self.filtered_contours[slide_id][contour] = ntile
        return df

    def get_tensor_shape(coords: np.ndarray, tile_size: int, min_size: int = 32):
        max_x, max_y = coords[:,0].max(), coords[:,1].max()
        min_x, min_y = coords[:,0].min(), coords[:,1].min()
        delta_x = max_x - min_x
        delta_y = max_y - min_y
        M = max(delta_x,delta_y)
        m = int(np.ceil(M/tile_size)) + 1
        if m < min_size:
            m = min_size
        return m
    
    def get_contour_with_max_ntile(self, slide_id: str):
        return max(self.filtered_contours[slide_id], key=lambda key: self.filtered_contours[slide_id][key])

    def __getitem__(self, idx: int):
        row = self.slide_df.loc[idx]
        slide_id = row.slide_id
        df = self.tile_df[self.tile_df.slide_id == slide_id]
        max_cont = self.get_contour_with_max_ntile(slide_id)
        df_cont = df[df.contour == max_cont]
        coords = df_cont[['x', 'y']].values
        m = self.get_tensor_shape(coords, self.tile_size)
        t = torch.zeros((m,m,self.emb_size))

        for (x,y) in coords:
            emb_path = Path(self.features_dir, f'{slide_id}_{x}_{y}.pt')
            emb = torch.load(emb_path)
            i, j = y // self.tile_size, x // self.tile_size
            t[i,j] = emb

        label = row.label
        return idx, t, label

    def __len__(self):
        return len(self.slide_df)


class SparseTensorDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df,
        tensor_dir: Path,
        transform: Callable = None,
    ):
        self.tensor_dir = tensor_dir
        self.transform = transform

        self.df = df

    def __getitem__(self, idx: int):
        row = self.df.loc[idx]
        slide_id = row.slide_id
        tensor_fp = Path(self.tensor_dir, f'{slide_id}.pt')
        t = torch.load(tensor_fp)
        if self.transform:
            t = self.transform(t)
        label = row.label
        return idx, t, label

    def __len__(self):
        return len(self.df)