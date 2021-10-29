# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import os
import re
import tarfile
from zipfile import ZipFile
import logging

logging.basicConfig(level=logging.INFO)

import gdown
import pandas as pd
from six import remove_move


def download_and_extract(url, dst, remove=True):
    gdown.download(url, dst, quiet=False)

    if dst.endswith(".tar.gz"):
        tar = tarfile.open(dst, "r:gz")
        tar.extractall(os.path.dirname(dst))
        tar.close()

    if dst.endswith(".tar"):
        tar = tarfile.open(dst, "r:")
        tar.extractall(os.path.dirname(dst))
        tar.close()

    if dst.endswith(".zip"):
        zf = ZipFile(dst, "r")
        zf.extractall(os.path.dirname(dst))
        zf.close()

    if remove:
        os.remove(dst)


def download_datasets(data_path):
    os.makedirs(data_path, exist_ok=True)

    logging.info("Downloading CelebA")
    celeba_dir = os.path.join(data_path, "celeba")
    os.makedirs(celeba_dir, exist_ok=True)
    download_and_extract(
        "https://drive.google.com/uc?id=1mb1R6dXfWbvk3DnlWOBO8pDeoBKOcLE6",
        os.path.join(celeba_dir, "img_align_celeba.zip"),
    )
    download_and_extract(
        "https://drive.google.com/uc?id=1acn0-nE4W7Wa17sIkKB0GtfW4Z41CMFB",
        os.path.join(celeba_dir, "list_eval_partition.txt"),
        remove=False
    )
    download_and_extract(
        "https://drive.google.com/uc?id=11um21kRUuaUNoMl59TCe2fb01FNjqNms",
        os.path.join(celeba_dir, "list_attr_celeba.txt"),
        remove=False
    )

    logging.info("Downloading Waterbirds")
    water_birds_dir = os.path.join(data_path, "waterbirds")
    os.makedirs(water_birds_dir, exist_ok=True)
    water_birds_dir_tar = os.path.join(water_birds_dir, "waterbirds.tar.gz")
    download_and_extract(
        "https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz",
        water_birds_dir_tar,
    )

    logging.info("Downloading MultiNLI")
    multinli_dir = os.path.join(data_path, "multinli")
    glue_dir = os.path.join(multinli_dir, "glue_data/MNLI/")
    os.makedirs(glue_dir, exist_ok=True)
    multinli_tar = os.path.join(glue_dir, "multinli_bert_features.tar.gz")
    download_and_extract(
        "https://nlp.stanford.edu/data/dro/multinli_bert_features.tar.gz",
        multinli_tar,
    )
    os.makedirs(os.path.join(multinli_dir, "data"), exist_ok=True)
    download_and_extract(
        "https://raw.githubusercontent.com/kohpangwei/group_DRO/master/dataset_metadata/multinli/metadata_random.csv",
        os.path.join(multinli_dir, "data", "metadata_random.csv"),
        remove=False
    )

    logging.info("Downloading CivilComments")
    civilcomments_dir = os.path.join(data_path, "civilcomments")
    os.makedirs(civilcomments_dir, exist_ok=True)
    download_and_extract(
        "https://worksheets.codalab.org/rest/bundles/0x8cd3de0634154aeaad2ee6eb96723c6e/contents/blob/",
        os.path.join(civilcomments_dir, "civilcomments.tar.gz"),
    )


def generate_metadata_celeba(data_path):
    logging.info("Generating metadata for CelebA")
    with open(os.path.join(data_path, "celeba/list_eval_partition.txt"), "r") as f:
        splits = f.readlines()

    with open(os.path.join(data_path, "celeba/list_attr_celeba.txt"), "r") as f:
        attrs = f.readlines()[2:]

    f = open(os.path.join(data_path, "metadata_celeba.csv"), "w")
    f.write("id,filename,split,y,a\n")

    for i, (split, attr) in enumerate(zip(splits, attrs)):
        fi, si = split.strip().split()
        ai = attr.strip().split()[1:]
        yi = 1 if ai[9] == "1" else 0
        gi = 1 if ai[20] == "1" else 0
        f.write("{},{},{},{},{}\n".format(i + 1, fi, si, yi, gi))

    f.close()


def generate_metadata_waterbirds(data_path):
    logging.info("Generating metadata for waterbirds")
    df = pd.read_csv(os.path.join(data_path, "waterbirds/waterbird_complete95_forest2water2/metadata.csv"))
    df = df.rename(columns={"img_id": "id", "img_filename": "filename", "place": "a"})
    df[["id", "filename", "split", "y", "a"]].to_csv(
        os.path.join(data_path, "metadata_waterbirds.csv"), index=False
    )


def generate_metadata_civilcomments(data_path):
    logging.info("Generating metadata for civilcomments")
    df = pd.read_csv(
        os.path.join(data_path, "civilcomments", "all_data_with_identities.csv"),
        index_col=0,
    )

    group_attrs = [
        "male",
        "female",
        "LGBTQ",
        "christian",
        "muslim",
        "other_religions",
        "black",
        "white",
    ]
    cols_to_keep = ["comment_text", "split", "toxicity"]
    df = df[cols_to_keep + group_attrs]
    df = df.rename(columns={"toxicity": "y"})
    df["y"] = (df["y"] >= 0.5).astype(int)
    df[group_attrs] = (df[group_attrs] >= 0.5).astype(int)
    df["no active attributes"] = 0
    df.loc[(df[group_attrs].sum(axis=1)) == 0, "no active attributes"] = 1

    few_groups, all_groups = [], []
    train_df = df.groupby("split").get_group("train")
    split_df = train_df.rename(columns={"no active attributes": "a"})
    few_groups.append(split_df[["y", "split", "comment_text", "a"]])

    for split, split_df in df.groupby("split"):
        for i, attr in enumerate(group_attrs):
            test_df = split_df.loc[
                split_df[attr] == 1, ["y", "split", "comment_text"]
            ].copy()
            test_df["a"] = i
            all_groups.append(test_df)
            if split != "train":
                few_groups.append(test_df)

    few_groups = pd.concat(few_groups).reset_index(drop=True)
    all_groups = pd.concat(all_groups).reset_index(drop=True)

    for name, df in {"coarse": few_groups, "fine": all_groups}.items():
        df.index.name = "filename"
        df = df.reset_index()
        df["id"] = df["filename"]
        df["split"] = df["split"].replace({"train": 0, "val": 1, "test": 2})
        text = df.pop("comment_text")

        df[["id", "filename", "split", "y", "a"]].to_csv(
            os.path.join(data_path, f"metadata_civilcomments_{name}.csv"), index=False
        )
        text.to_csv(
            os.path.join(data_path, "civilcomments", f"civilcomments_{name}.csv"),
            index=False,
        )


def generate_metadata_multinli(data_path):
    logging.info("Generating metadata for multinli")
    df = pd.read_csv(
        os.path.join(data_path, "multinli", "data", "metadata_random.csv"), index_col=0
    )

    df = df.rename(columns={"gold_label": "y", "sentence2_has_negation": "a"})
    df = df.reset_index(drop=True)
    df.index.name = "id"
    df = df.reset_index()
    df["filename"] = df["id"]
    df = df.reset_index()[["id", "filename", "split", "y", "a"]]
    df.to_csv(os.path.join(data_path, "metadata_multinli.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize repo with datasets")
    parser.add_argument(
        "--data_path",
        default="data",
        type=str,
        help="Root directory to store datasets",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    if args.download:
        download_datasets(args.data_path)
    generate_metadata_celeba(args.data_path)
    generate_metadata_waterbirds(args.data_path)
    generate_metadata_civilcomments(args.data_path)
    generate_metadata_multinli(args.data_path)
