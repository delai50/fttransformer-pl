from typing import Optional, List

from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn

import pytorch_lightning as pl

from rtdl import FTTransformer

from torch.optim import AdamW
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

class LitFTTransformer(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        
        self.fttransformer = FTTransformer.make_baseline(
            n_num_features = self.hparams.n_num_features,
            cat_cardinalities = self.hparams.cat_cardinalities,
            d_token = self.hparams.d_token,
            n_blocks = self.hparams.n_blocks,
            attention_dropout = self.hparams.attention_dropout,
            ffn_d_hidden = self.hparams.ffn_d_hidden,
            ffn_dropout = self.hparams.ffn_dropout,
            residual_dropout = self.hparams.residual_dropout,
            d_out = self.hparams.d_out
        )
        
        self.loss_fn = nn.CrossEntropyLoss()
        self.metric = accuracy_score
    
    def forward(self, X_cont, X_cat):
        out = self.fttransformer(X_cont, X_cat)
        return out
    
    # def predict(self, dl):
    #     self.eval()
    #     y_pred = []
    #     with torch.no_grad():
    #         for batch in tqdm(dl):
    #             X_cont = batch["X_cont"].to("cuda") if self.n_num_features != 0 else None
    #             X_cat = batch["X_cat"].to("cuda") if self.cat_cardinalities != None else None 
    #             logits = self(X_cont, X_cat)
    #             y_pred.append(
    #                 torch.argmax(torch.softmax(logits, axis=1), axis=1)
    #             )
    #     return torch.cat(y_pred, axis=0)
    
    def shared_step(self, batch):
        y = batch["y"]
        X_cont = batch["X_cont"] if self.hparams.n_num_features != 0 else None
        X_cat = batch["X_cat"] if self.hparams.cat_cardinalities != None else None
        logits = self(X_cont, X_cat)
        loss = self.loss_fn(logits, y)
        y_hat = torch.argmax(torch.softmax(logits, axis=1), axis=1)
        metric = self.metric(y.to("cpu"), y_hat.to("cpu"))
        return loss, metric
    
    def training_step(self, batch, batch_idx):
        loss, metric = self.shared_step(batch)
        self.log("loss", loss, prog_bar=True)
        self.log("metric", metric, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        val_loss, val_metric = self.shared_step(batch)
        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_metric", val_metric, prog_bar=True)
        
    def predict_step(self, batch, batch_idx):
        X_cont = batch["X_cont"] if self.hparams.n_num_features != 0 else None
        X_cat = batch["X_cat"] if self.hparams.cat_cardinalities != None else None
        logits = self(X_cont, X_cat)
        y_hat = torch.argmax(torch.softmax(logits, axis=1), axis=1)
        return y_hat
    
    def configure_optimizers(self):
        opt = AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        sch = LinearWarmupCosineAnnealingLR(opt, warmup_epochs=self.hparams.warmup_epochs, max_epochs=self.hparams.max_epochs, 
                                            warmup_start_lr=0.0, eta_min=0.0, last_epoch=-1)
        return [opt], [sch]