import os
import glob
import joblib
import sys
import yaml

import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score

import torch

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging, LearningRateMonitor

import wandb

from datasets import TabDataModule
from model import LitFTTransformer

import optuna

from coolname import generate_slug

import warnings
warnings.filterwarnings("ignore")


# CONFIG FILE
config = {
    "path": "/home/adelaiglesia/Projects/Kaggle/Tabular_Playground_Series_Dec_2021", # base path
    "sim_name": generate_slug(2), # simulation name
    "seed": 42, # random seed: int
    "stage": "train", # stage: "train", "infer", "hpopt"
    "target": "Cover_Type", # target name: str
    "n_folds": 5, # number of folds to split the data for CV: int
    "folds_used": [0], # folds used for CV: List[int]
    "gpus": [1], # list of gpus to use: List[int]
    "precision": 16, # precision for training: int
    "swa": False, # use stochastic weight averaging: bool
    "lr": 1e-4, # learning rate: float
    "auto_lr_find": True, # use automatic learning rate: bool
    "weight_decay": 1e-5, # weight decay: float
    "max_epochs": 4, # max number of epochs: int
    "warmup_epochs": 1, # number of epochs to warmup in linear cosine annealing optimizer: int
    "batch_size": 128, # batch size: int
    "num_workers": 1, # number of workers: int
    "debug_pct": 0.005, # percentage of data to use for debugging: float
    "log": False, # log to wandb: bool
    "train_batches": 1., # percentage of batches to train: float
    "val_batches": 1., # percentage of batches to validate: float
    "resume": None, # # resume training from checkpoint: Optional["str"]. Resume does not work with auto_lr_find
    "d_out": 7, # number of output classes: int
    "d_token": 8, # int = 8
    "n_blocks": 2, # int = 2,
    "attention_dropout": 0.2, # float = 0.2,
    "ffn_d_hidden": 6, # int = 6,
    "ffn_dropout": 0.2, # float = 0.2,
    "residual_dropout": 0.0, # float = 0.0,
    "timeout": 120, # seconds to wait for optimization: int
    "mode": "max", # optimization mode: "max", "min"
}


# TRAINING
def train(config):
    
    pl.seed_everything(config["seed"], workers=True)
    
    dm = TabDataModule(
        path = config["path"],
        seed = config["seed"],
        stage = config["stage"],
        target = config["target"],
        n_folds = config["n_folds"],
        batch_size = config["batch_size"],
        num_workers = config["num_workers"],
        debug_pct = config["debug_pct"],
    )
    dm.prepare_data()
    
    oofs = dm.df_train[[config["target"], "fold"]].copy()
    oofs["preds"] = 0
    
    for fold in config["folds_used"]:
        
        dm._has_setup_fit = False
        dm.setup(stage="fit", fold=fold)
        
        # Callbacks and logger
        wandb_logger = WandbLogger(project="Test", name=f"{config['sim_name']}-fold{fold}", config=config)
        
        callbacks = []
        if config["stage"] == "train":
            checkpoint = ModelCheckpoint(
                dirpath = os.path.join(config["path"], config["sim_name"], "checkpoints"),
                filename = f"fold={fold}_" + "{epoch}_{metric:.4f}_{val_metric:.4f}",
                monitor = "val_metric",
                save_top_k = 1,
                mode = config["mode"],
            )
            callbacks.append(checkpoint)
        if config["log"]:
            callbacks.append(LearningRateMonitor())
        if config["swa"]:
            callbacks.append(StochasticWeightAveraging(swa_epoch_start=0.5))

        pl.seed_everything(config["seed"], workers=True)
        
        config.update(
            {
                "n_num_features": dm.n_num_features, # int,
                "cat_cardinalities": dm.cat_cardinalities, # Optional[List[int]],
            }
        ) 
        model = LitFTTransformer(config)
    
        trainer = pl.Trainer(
            gpus = config["gpus"],
            precision = config["precision"],
            max_epochs = config["max_epochs"],
            logger = wandb_logger if config["log"] else None,
            deterministic = True,
            callbacks = callbacks,
            limit_train_batches = config["train_batches"],
            limit_val_batches = config["val_batches"],
            resume_from_checkpoint = config["resume"],
            auto_lr_find = config["auto_lr_find"], 
        )
        
        if config["auto_lr_find"]:
            lr_finder = trainer.tuner.lr_find(model, datamodule=dm)
            model.hparams.lr = lr_finder.suggestion()
        
        trainer.fit(model, datamodule=dm)

        # OOFS
        dirpath = os.path.join(config["path"], config["sim_name"], "checkpoints", f"fold={fold}*")
        checkpoints = glob.glob(dirpath)
        
        # Code makes predictions using all the checkpoints in the checkpoints folder
        for chpkt in checkpoints:
            
            model = LitFTTransformer.load_from_checkpoint(chpkt)
            y_hat = trainer.predict(model, datamodule=dm)
            y_hat = torch.cat(y_hat, axis=0).detach().to("cpu").numpy()

            oofs.loc[oofs["fold"]==fold, "preds"] += y_hat / len(checkpoints)
        
        wandb.finish()
    
    # Save oofs
    oofs.loc[~oofs["fold"].isin(config["folds_used"]), "preds"] = np.nan
    oofs.to_csv(os.path.join(config["path"], config["sim_name"], "oofs.csv"), index=False)
    
    # Get score
    y = oofs.loc[oofs["fold"].isin(config["folds_used"]), config["target"]].values
    y_hat = oofs.loc[oofs["fold"].isin(config["folds_used"]), "preds"].values
    y_hat = np.round(y_hat)
    score = accuracy_score(y, y_hat)
    print(f"Overall score: {score:.4f}")
    
    return score


# INFERENCE
def infer(config):
    
    dm = TabDataModule(
        path = config["path"],
        seed = config["seed"],
        stage = config["stage"],
        target = config["target"],
        n_folds = config["n_folds"],
        batch_size = config["batch_size"],
        num_workers = config["num_workers"],
        debug_pct = config["debug_pct"],
    )
    dm.prepare_data()
    
    for fold in config["folds_used"]:
        
        dm._has_setup_predict = False
        dm.setup(stage="predict", fold=fold)
        
        trainer = pl.Trainer(gpus=config["gpus"])
        
        dirpath = os.path.join(config["path"], config["sim_name"], "checkpoints", f"fold={fold}*")
        checkpoints = glob.glob(dirpath)
        
        y_hat = 0
        # Code makes predictions using all the checkpoints in the checkpoints folder
        for chpkt in checkpoints:
            
            model = LitFTTransformer.load_from_checkpoint(chpkt)
    
            y_hat_ = trainer.predict(model, datamodule=dm)
            y_hat_ = torch.cat(y_hat_, axis=0).detach().to("cpu").numpy() + 1
            y_hat += y_hat_ / len(checkpoints)
        
    return y_hat


def hpopt(config):

    def objective(trial):
        
        hparams = {
            "lr": config["lr"], # trial.suggest_loguniform("lr", 1e-5, 1e-3),
            "max_epochs": config["max_epochs"], # trial.suggest_int("max_epochs", 10, 20),
            "weight_decay": trial.suggest_loguniform("weight_decay", 1e-6, 1e-3),
            "d_token": trial.suggest_int("d_token", 64, 512, 8),
            "n_blocks": trial.suggest_int("n_blocks", 1, 4),
            "attention_dropout": trial.suggest_uniform("residual_dropout", 0.0, 0.5),
            "ffn_d_hidden": trial.suggest_int("ffn_d_hidden", 43, 1365),
            "ffn_dropout": trial.suggest_uniform("residual_dropout", 0.0, 0.5),
            "residual_dropout": trial.suggest_uniform("residual_dropout", 0.0, 0.2)
        }
        config.update(hparams)
    
        score = train(config)
        return score
    
    if os.path.isfile(os.path.join(config["path"], config["sim_name"], "hpopt.pkl")):
        study = joblib.load(os.path.join(config["path"], config["sim_name"], "hpopt.pkl"))
    else:
        study = optuna.create_study(direction="maximize" if config["mode"]=="max" else "minimize")
    study.optimize(objective, timeout=config["timeout"])
    joblib.dump(study, os.path.join(config["path"], config["sim_name"], "hpopt.pkl"))


if __name__ == "__main__":
    
    config_file = sys.argv[1]
    if config_file:
        with open(config_file, "r") as stream:
            loaded_config = yaml.safe_load(stream)
        config.update(loaded_config)
    
    if config["stage"] == "train":
        train(config)
        
    elif config["stage"] == "infer":
        sub = pd.read_csv(os.path.join(config["path"], "data", "sample_submission.csv"))
        y_hat = infer(config)
        sub[config["target"]] = np.round(y_hat)
        sub.to_csv(os.path.join(config["path"], config["sim_name"], "submission.csv"), index=False)
        
    elif config["stage"] == "hpopt":
        hpopt(config)
        
