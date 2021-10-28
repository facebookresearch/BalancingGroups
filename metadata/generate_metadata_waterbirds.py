# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
from shutil import copyfile

copyfile(
  os.path.join(os.environ["DATASETS_PATH"], "waterbirds/metadata.csv"),
  "metadata_waterbirds.csv")
