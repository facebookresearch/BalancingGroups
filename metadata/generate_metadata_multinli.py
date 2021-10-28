# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import pandas as pd
import os

df = pd.read_csv(os.path.join(
    os.environ['DATASETS_PATH'], "multinli", "data", "metadata_random.csv"), index_col=0)

df.insert(0, 'img_id', "unavailable")
df = df.rename(columns={'gold_label': 'y', 'sentence2_has_negation': 'place'})
df["place_filename"] = "unavailable"
df = df.reset_index(drop=True)
df.index.name = "img_filename"
df = df.reset_index()[["img_id", "img_filename", "y",
                       "split", "place", "place_filename"]]
df.to_csv('metadata_multinli.csv', index=False)
