# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import pandas as pd
import os

df = pd.read_csv(os.path.join(
    os.environ['DATASETS_PATH'],
    "civilcomments",
    "all_data_with_identities.csv"),
    index_col=0)

group_attrs = ["male", "female", "LGBTQ", "christian",
               "muslim", "other_religions", "black", "white"]
cols_to_keep = ['comment_text', 'split', 'toxicity']
df = df[cols_to_keep+group_attrs]
df = df.rename(columns={'toxicity': 'y'})
df['y'] = (df['y'] >= 0.5).astype(int)
df[group_attrs] = (df[group_attrs] >= 0.5).astype(int)
df["no active attributes"] = 0
df.loc[(df[group_attrs].sum(axis=1)) == 0, "no active attributes"] = 1

few_groups, all_groups = [], []
train_df = df.groupby('split').get_group('train')
split_df = train_df.rename(columns={'no active attributes': 'place'})
few_groups.append(split_df[['y', 'split', 'comment_text', 'place']])

for split, split_df in df.groupby('split'):
    for i, attr in enumerate(group_attrs):
        test_df = split_df.loc[split_df[attr] ==
                               1, ["y", "split", "comment_text"]].copy()
        test_df['place'] = i
        all_groups.append(test_df)
        if split != 'train':
            few_groups.append(test_df)

few_groups = pd.concat(few_groups).reset_index(drop=True)
all_groups = pd.concat(all_groups).reset_index(drop=True)

for name, df in {'coarse': few_groups, 'fine': all_groups}.items():
    df["place_filename"] = 'unavailable'
    df.index.name = 'img_filename'
    df = df.reset_index()
    df.insert(0, 'img_id', 'unavailable')
    df["split"] = df["split"].replace(
        {'train': 0, 'val': 1, 'test': 2})
    text = df.pop('comment_text')

    df[["img_id", "img_filename", "y", "split", "place",
        "place_filename"]].to_csv(f'metadata_civilcomments_{name}.csv', index=False)
    text.to_csv(os.path.join(
        os.environ['DATASETS_PATH'], 'civilcomments', f"civilcomments_{name}.csv"),
        index=False)
