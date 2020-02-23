import pandas as pd 
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

if __name__ == '__main__':
    df = pd.read_csv("../input/train.csv")
    print(df.head())
    df.loc[:,"kfolf"] = -1
    df = df.sample(frac = 1).reset_index(drop = True)
    X = df.image_id.values
    y = df[["grapheme_root", "vowel_diacritic", "consonant_diacritic"]].values

    mskf = MultilabelStratifiedKFold(n_splits = 5)
    for n_fold,(trn,val) in enumerate(mskf.split(X,y)):
        print("Train: ", trn, "Val: ",val)
        df.loc[val,"kfold"] = n_fold
    print(df.kfold.value_counts())
    df.to_csv("../input/train_fols.csv",index=False)
