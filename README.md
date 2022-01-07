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
- Modify the crossvalidation scheme inside the *dataprep.py* module
- Modify the data prep_data function within the *dataprep.py* module
- Modify the Datasets within the *datasets.py* module
- Modify the prepare_data method within the *datasets.py* module
- Change the loss function within the *model.py* module
- Change the metric within the *model.py* and the *main.py* module
- Use another Pytorch model
- Modify the way that predictions are made

