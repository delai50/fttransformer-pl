import pandas as pd

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold


def create_folds(df: pd.DataFrame, target: str, n_splits: int=5, seed: int=2021):
    df["fold"] = -1
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, val_idx) in enumerate(cv.split(df, df[target])):
        df.loc[val_idx, "fold"] = fold
    return df


def prep_data(df_train: pd.DataFrame, df_test: pd.DataFrame, target: str):

    df_train.drop(["Id"], axis=1, inplace=True)
    df_test.drop(["Id"], axis=1, inplace=True)
    
    df_train["Cover_Type"] -= 1
    
    cont_cols = [
       'Elevation', 'Aspect', 'Slope',
       'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
       'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon',
       'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points',
       'Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3',
       'Wilderness_Area4', 'Soil_Type1', 'Soil_Type2', 'Soil_Type3',
       'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8',
       'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12',
       'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16',
       'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20',
       'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24',
       'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28',
       'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32',
       'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36',
       'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40'
    ]
    cat_cols = None # Optional[List[str]]
    n_num_features = len(cont_cols)
    cat_cardinalities = None # Optional[List[str]]
    
    df_all = pd.concat([df_train, df_test], axis=0)
    sc = RobustScaler()
    df_all[cont_cols] = sc.fit_transform(df_all[cont_cols])
    df_train = df_all.iloc[:df_train.shape[0],:]
    df_test = df_all.iloc[df_train.shape[0]:,:]
    df_test.drop([target], axis=1, inplace=True)
    
    return df_train, df_test, cont_cols, cat_cols, n_num_features, cat_cardinalities