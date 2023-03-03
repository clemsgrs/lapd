import torch
import random
import numpy as np
import pandas as pd
from typing import Callable, Dict
from pathlib import Path
from collections import defaultdict


def ppcess_survival_data(
    df,
    label_name: str = "label",
    nbins: int = 4,
    eps: float = 1e-6,
):
    patient_df = df.drop_duplicates(['case_id'])
    patient_df = patient_df.drop('slide_id', axis=1)
    uncensored_df = patient_df[patient_df['censorship'] < 1]

    _, q_bins = pd.qcut(uncensored_df[label_name], q=nbins, retbins=True, labels=False)
    q_bins[-1] = df[label_name].max() + eps
    q_bins[0] = df[label_name].min() - eps

    disc_labels, bins = pd.cut(patient_df[label_name], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True)
    patient_df.insert(2, "disc_label", disc_labels.values.astype(int))

    label_dict = {}
    label_count = 0
    for label in range(len(bins)-1):
        for censorship in [0, 1]:
            label_dict.update({(label, censorship): label_count})
            label_count += 1

    patient_df.reset_index(drop=True, inplace=True)
    for i in patient_df.index:
        disc_label = patient_df.loc[i, "disc_label"]
        censorship = patient_df.loc[i, "censorship"]
        key = (disc_label, int(censorship))
        patient_df.at[i, "label"] = label_dict[key]

    slide_df = pd.merge(df, patient_df[["case_id", "disc_label", "label"]], how="left", on="case_id")

    return patient_df, slide_df


class MaxSlideTensorSurvivalDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        patient_df: pd.DataFrame,
        slide_df: pd.DataFrame,
        features_dir: Path,
        tile_size: int,
        fmt: str = 'jpg',
        emb_size: int = 192,
        label_name: str = "label",
    ):

        self.features_dir = features_dir
        self.tile_size = tile_size
        self.fmt = fmt
        self.emb_size = emb_size
        self.label_name = label_name

        self.tile_df = self.prepare_data(slide_df)
        self.slide_df = self.tile_df.drop_duplicates(['slide_id'], ignore_index=True)
        self.patient_df = patient_df

    def prepare_data(self, df, ntile_min: int = -1):
        if self.label_name != "label":
            df["label"] = df.loc[:, self.label_name]
        tmp = df.groupby(['slide_id', 'contour']).contour.size().to_frame().rename(columns={'contour': 'ntile'}).reset_index()
        self.filtered_contours = defaultdict(dict)
        for slide_id, contour, ntile in tmp[tmp.ntile >= ntile_min].values:
            self.filtered_contours[slide_id][contour] = ntile
        return df

    def get_tensor_shape(self, coords: np.ndarray, tile_size: int, min_size: int = 224):
        max_x, max_y = coords[:,0].max(), coords[:,1].max()
        min_x, min_y = coords[:,0].min(), coords[:,1].min()
        delta_x = max_x - min_x
        delta_y = max_y - min_y
        M = max(delta_x,delta_y)
        m = int(np.floor(M/tile_size)) + 1
        if m < min_size:
            m = min_size
        return m

    def get_slide_id_with_max_ntile(self, case_id: str):
        slide_ids = self.slide_df[self.slide_df.case_id == case_id].slide_id.values.tolist()
        ntile_dict = {}
        for slide_id in slide_ids:
            ntile = np.sum([self.filtered_contours[slide_id][c] for c in self.filtered_contours[slide_id].keys()])
            ntile_dict[slide_id] = ntile
        return max(ntile_dict, key=lambda key: ntile_dict[key])

    def __getitem__(self, idx: int):
        row = self.patient_df.loc[idx]
        case_id = row.case_id
        slide_id = self.get_slide_id_with_max_ntile(case_id)
        df = self.tile_df[self.tile_df.slide_id == slide_id]

        coords = df[['x', 'y']].values
        m = self.get_tensor_shape(coords, self.tile_size)
        t = torch.zeros((m,m,self.emb_size))

        for (x,y) in coords:
            emb_path = Path(self.features_dir, f'{slide_id}_{x}_{y}.pt')
            emb = torch.load(emb_path)
            i, j = y // self.tile_size, x // self.tile_size
            t[i,j] = emb

        # put channels first
        t = t.permute(2, 0, 1)

        label = row.disc_label
        event_time = row[self.label_name]
        c = row.censorship
        return idx, t, label, event_time, c

    def __len__(self):
        return len(self.patient_df)


class RandomSlideTensorSurvivalDataset(MaxSlideTensorSurvivalDataset):
    def __init__(
        self,
        patient_df: pd.DataFrame,
        slide_df: pd.DataFrame,
        features_dir: Path,
        tile_size: int,
        fmt: str = 'jpg',
        emb_size: int = 192,
        label_name: str = "label",
    ):

        super().__init__(patient_df, slide_df, features_dir, tile_size, fmt, emb_size, label_name)

    def __getitem__(self, idx: int):
        row = self.patient_df.loc[idx]
        case_id = row.case_id
        slide_ids = self.slide_df[self.slide_df.case_id == case_id].slide_id.values.tolist()
        random.shuffle(slide_ids)
        slide_id = slide_ids[0]

        df = self.tile_df[self.tile_df.slide_id == slide_id]

        coords = df[['x', 'y']].values
        m = self.get_tensor_shape(coords, self.tile_size)
        t = torch.zeros((m,m,self.emb_size))

        for (x,y) in coords:
            emb_path = Path(self.features_dir, f'{slide_id}_{x}_{y}.pt')
            emb = torch.load(emb_path)
            i, j = y // self.tile_size, x // self.tile_size
            t[i,j] = emb

        # put channels first
        t = t.permute(2, 0, 1)

        label = row.disc_label
        event_time = row[self.label_name]
        c = row.censorship
        return idx, t, label, event_time, c

    def __len__(self):
        return len(self.patient_df)


class MaxSlideMaxContourTensorSurvivalDataset(MaxSlideTensorSurvivalDataset):
    def __init__(
        self,
        patient_df: pd.DataFrame,
        slide_df: pd.DataFrame,
        features_dir: Path,
        tile_size: int,
        fmt: str = 'jpg',
        emb_size: int = 192,
        label_name: str = "label",
    ):

        super().__init__(patient_df, slide_df, features_dir, tile_size, fmt, emb_size, label_name)

    def get_contour_with_max_ntile(self, slide_id: str):
        return max(self.filtered_contours[slide_id], key=lambda key: self.filtered_contours[slide_id][key])

    def __getitem__(self, idx: int):
        row = self.patient_df.loc[idx]
        case_id = row.case_id
        slide_id = self.get_slide_id_with_max_ntile(case_id)
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

        # put channels first
        t = t.permute(2, 0, 1)

        label = row.disc_label
        event_time = row[self.label_name]
        c = row.censorship
        return idx, t, label, event_time, c


class TensorSurvivalDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        patient_df: pd.DataFrame,
        slide_df: pd.DataFrame,
        features_dir: Path,
        tile_size: int,
        fmt: str = 'jpg',
        emb_size: int = 192,
        label_name: str = "label",
    ):

        self.features_dir = features_dir
        self.tile_size = tile_size
        self.fmt = fmt
        self.emb_size = emb_size
        self.label_name = label_name

        self.tile_df = self.prepare_data(slide_df)
        self.slide_df = self.tile_df.drop_duplicates(['slide_id'], ignore_index=True)
        self.patient_df = patient_df

    def prepare_data(self, df, ntile_min: int = -1):
        if self.label_name != "label":
            df.loc[:, "label"] = df[self.label_name]
        tmp = df.groupby(['slide_id', 'contour']).contour.size().to_frame().rename(columns={'contour': 'ntile'}).reset_index()
        self.filtered_contours = defaultdict(dict)
        for slide_id, contour, ntile in tmp[tmp.ntile >= ntile_min].values:
            self.filtered_contours[slide_id][contour] = ntile
        return df

    def get_tensor_shape(self, coords: np.ndarray, tile_size: int, min_size: int = 224):
        max_x, max_y = coords[:,0].max(), coords[:,1].max()
        min_x, min_y = coords[:,0].min(), coords[:,1].min()
        delta_x = max_x - min_x
        delta_y = max_y - min_y
        M = max(delta_x,delta_y)
        m = int(np.floor(M/tile_size)) + 1
        if m < min_size:
            m = min_size
        return m

    def __getitem__(self, idx: int):
        row = self.patient_df.loc[idx]
        case_id = row.case_id
        slide_ids = self.slide_df[self.slide_df.case_id == case_id].slide_id.values.tolist()

        tensors = []
        for slide_id in slide_ids:
            df = self.tile_df[self.tile_df.slide_id == slide_id]
            coords = df[['x', 'y']].values
            m = self.get_tensor_shape(coords, self.tile_size)
            t = torch.zeros((m,m,self.emb_size))
            for (x,y) in coords:
                emb_path = Path(self.features_dir, f'{slide_id}_{x}_{y}.pt')
                emb = torch.load(emb_path)
                i, j = y // self.tile_size, x // self.tile_size
                t[i,j] = emb
            # put channels first
            t = t.permute(2, 0, 1)
            tensors.append(t)

        label = row.disc_label
        event_time = row[self.label_name]
        c = row.censorship
        return idx, tensors, label, event_time, c

    def __len__(self):
        return len(self.patient_df)


class MaxContourMaxSlideTensorDataset(torch.utils.data.Dataset):
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
            emb_size (int, optional): _description_. Defaults to 192.
            transform (Callable, optional): _description_. Defaults to None.
        """
        self.features_dir = features_dir
        self.tile_size = tile_size
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
        m = int(np.floor(M/tile_size)) + 1
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