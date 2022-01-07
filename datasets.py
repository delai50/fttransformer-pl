import os

from typing import Optional, List, Tuple, Dict, Any

import pandas as pd

import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, BatchSampler, Dataset

import pytorch_lightning as pl

from dataprep import create_folds, prep_data


class TabDataSet(Dataset):
    def __init__(
        self, 
        df: pd.DataFrame, 
        target: str, 
        cont_cols: List[str], 
        cat_cols: Optional[List[str]]
    ):
        self.df = df
        self.target = target
        self.cont_cols = cont_cols
        self.cat_cols = cat_cols
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        y = self.df[self.target].values[idx] if self.target in self.df.columns else 0
        X = self.df.drop([self.target], axis=1) if self.target in self.df.columns else self.df
        X_cont = X[self.cont_cols].values[idx] if self.cont_cols != None else 0
        X_cat = X[self.cat_cols].values[idx] if self.cat_cols != None else 0
        
        return {
            "y": torch.tensor(y, dtype=torch.long),
            "X_cont": torch.tensor(X_cont, dtype=torch.float),
            "X_cat": torch.tensor(X_cat, dtype=torch.float)
        }



class MyDataLoader:

    @staticmethod
    def collate_fn(tensors:Dict[str, torch.Tensor]):
        return tensors

    @classmethod
    def create_dataloader(cls, dataset: Dataset, batch_size: int, shuffle: bool=False,
        drop_last: bool=False, num_workers: int=0, prefetch_factor: int=2,
        seed: Optional[int]=None, replacement: bool=False,
        num_samples: Optional[int]=None, pin_memory=True) -> DataLoader:
        
        if seed != None:
            generator = torch.Generator()
            generator.manual_seed(seed) # Seeds for reproducibility
        else:
            generator = None
        
        # Map-type Dataset
        if shuffle:
            if replacement:
                if type(num_samples) == None: num_samples = len(dataset)
            else:
                num_samples = None
            sampler = RandomSampler(dataset, replacement=replacement,
                num_samples=num_samples, generator=generator)  # type: ignore
        else:
            sampler = SequentialSampler(dataset)
        batch_sampler = BatchSampler(sampler, batch_size, drop_last)

        return DataLoader(dataset, sampler=batch_sampler, batch_size=None,
            num_workers=num_workers, collate_fn=cls.collate_fn,
            prefetch_factor=prefetch_factor, pin_memory=pin_memory)



class TabDataModule(pl.LightningDataModule):
    def __init__(self,
                 path: str,
                 seed: int,
                 stage: str,
                 target: str,
                 n_folds: int,
                 batch_size: int,
                 num_workers: int,
                 debug_pct: float,
                ):
        super().__init__()
        self.path = path
        self.seed = seed
        self.stage = stage
        self.target = target
        self.n_folds = n_folds
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.debug_pct = debug_pct

    def prepare_data(self):
        # called only on 1 GPU
        df_train = pd.read_csv(os.path.join(self.path, "data", "train.csv"))
        df_test = pd.read_csv(os.path.join(self.path, "data", "test.csv"))
        
        if self.debug_pct < 1.0:
            df_train = df_train.sample(frac=self.debug_pct, random_state=self.seed).reset_index(drop=True)
        
        df_train, self.df_test, self.cont_cols, self.cat_cols, self.n_num_features, self.cat_cardinalities = prep_data(df_train, df_test, self.target)

        self.df_train = create_folds(df_train, self.target, n_splits=self.n_folds, seed=self.seed)

    def setup(self, stage: Optional[str]=None, fold: int=0):
        
        self.df_train_ = self.df_train.query(f"fold != {fold}")
        self.df_val = self.df_train.query(f"fold == {fold}")
                
        self.ds_train = TabDataSet(self.df_train_.drop(["fold"], axis=1), 
                                   self.target, self.cont_cols, self.cat_cols)
        self.ds_val = TabDataSet(self.df_val.drop(["fold"], axis=1), 
                                 self.target, self.cont_cols, self.cat_cols)
        self.ds_test = TabDataSet(self.df_test, self.target, self.cont_cols, self.cat_cols)
    
    def train_dataloader(self):
        return MyDataLoader.create_dataloader(
            self.ds_train,
            batch_size = self.batch_size,
            shuffle = True,
            drop_last = True,
            seed = self.seed,
            num_workers = self.num_workers,
            pin_memory = True,
        )

    def val_dataloader(self):
        return MyDataLoader.create_dataloader(
            self.ds_val,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            pin_memory = True,
        )
        
    def predict_dataloader(self):
        if self.stage == "train" or self.stage == "hpopt":
            return MyDataLoader.create_dataloader(
                self.ds_val,
                batch_size = self.batch_size,
                num_workers = self.num_workers,
                pin_memory = True,
            )
        elif self.stage == "infer":
            return MyDataLoader.create_dataloader(
                self.ds_test,
                batch_size = self.batch_size * 2,
                num_workers = self.num_workers,
                pin_memory = True,
            )


