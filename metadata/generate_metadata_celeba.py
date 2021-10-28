# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os

print("img_id,img_filename,y,split,place,place_filename")

with open(os.path.join(os.environ["DATASETS_PATH"],
                       "celeba/list_eval_partition.txt"), "r") as f:
    splits = f.readlines()

with open(os.path.join(os.environ["DATASETS_PATH"],
          "celeba/list_attr_celeba.txt"), "r") as f:
    attrs = f.readlines()[2:]

f = open("metadata_celeba.csv", "w")

for i, (split, attr) in enumerate(zip(splits, attrs)):
    fi, si = split.strip().split()
    ai = attr.strip().split()[1:]
    yi = 1 if ai[9] == '1' else 0
    gi = 1 if ai[20] == '1' else 0
    f.write("{},{},{},{},{},{}\n".format(i + 1, fi, yi, si, gi, "unavailable"))

f.close()
