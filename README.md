### Description

This repository contains a full Pytorch Lightning pipeline with the main goal of reusing it as a template for Kaggle competitions. It is particularized to the FTTransformer model from *Gorishniy, Rubachev, Khrulkov, & Babenko (2021, Nov) Revisiting Deep Learning Models for Tabular Data*, but any Pytorch model can be used.

The code was tested using the data from https://www.kaggle.com/c/tabular-playground-series-dec-2021.

***
### Folder structure
```
project name    
│
└───data
│   │   train.csv
│   │   test.csv
│   │   sample_submission.csv
│   
└───src
│   │   main.py
│   │   datasets.py
│   │   dataprep.py
│   │   model.py
│   
└───simulations
    └───simulation_name
        |   config.yml
        │   oofs.csv
        │   hpopt.pkl
        |   submission.csv
        └───checkpoints
            │   fold=0_epoch=[]_metric=[]_val_metric=[].ckpt
            │   fold=1_epoch=[]_metric=[]_val_metric=[].ckpt
            |   ...
```
***

### Usage

This Pytorch Lightning pipeline can used for any Machine Learning problem with the following modifications:
- *create_folds* function inside the *dataprep.py* module
- *prep_data* function within the *dataprep.py* module
- Datasets within the *datasets.py* module
- *prepare_data* method within the *datasets.py* module
- Loss function within the *model.py* module
- Metric within the *model.py* and the *main.py* module
- The Pytorch model
- The way that predictions are made

