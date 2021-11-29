# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import torch
import pandas as pd
import numpy as np

from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from sklearn.datasets import make_blobs
import pandas as pd


class GroupDataset:
    def __init__(
        self, split, root, metadata, transform, subsample_what=None, duplicates=None
    ):
        self.transform_ = transform
        df = pd.read_csv(metadata)
        df = df[df["split"] == ({"tr": 0, "va": 1, "te": 2}[split])]

        self.i = list(range(len(df)))
        self.x = df["filename"].astype(str).map(lambda x: os.path.join(root, x)).tolist()
        self.y = df["y"].tolist()
        self.g = df["a"].tolist()

        self.count_groups()

        if subsample_what is not None:
            self.subsample_(subsample_what)

        if duplicates is not None:
            self.duplicate_(duplicates)

    def count_groups(self):
        self.wg, self.wy = [], []

        self.nb_groups = len(set(self.g))
        self.nb_labels = len(set(self.y))
        self.group_sizes = [0] * self.nb_groups * self.nb_labels
        self.class_sizes = [0] * self.nb_labels

        for i in self.i:
            self.group_sizes[self.nb_groups * self.y[i] + self.g[i]] += 1
            self.class_sizes[self.y[i]] += 1

        for i in self.i:
            self.wg.append(
                len(self) / self.group_sizes[self.nb_groups * self.y[i] + self.g[i]]
            )
            self.wy.append(len(self) / self.class_sizes[self.y[i]])

    def subsample_(self, subsample_what):
        perm = torch.randperm(len(self)).tolist()

        if subsample_what == "groups":
            min_size = min(list(self.group_sizes))
        else:
            min_size = min(list(self.class_sizes))

        counts_g = [0] * self.nb_groups * self.nb_labels
        counts_y = [0] * self.nb_labels
        new_i = []
        for p in perm:
            y, g = self.y[self.i[p]], self.g[self.i[p]]

            if (
                subsample_what == "groups"
                and counts_g[self.nb_groups * int(y) + int(g)] < min_size
            ) or (subsample_what == "classes" and counts_y[int(y)] < min_size):
                counts_g[self.nb_groups * int(y) + int(g)] += 1
                counts_y[int(y)] += 1
                new_i.append(self.i[p])

        self.i = new_i
        self.count_groups()

    def duplicate_(self, duplicates):
        new_i = []
        for i, duplicate in zip(self.i, duplicates):
            new_i += [i] * duplicate
        self.i = new_i
        self.count_groups()

    def __getitem__(self, i):
        j = self.i[i]
        x = self.transform(self.x[j])
        y = torch.tensor(self.y[j], dtype=torch.long)
        g = torch.tensor(self.g[j], dtype=torch.long)
        return torch.tensor(i, dtype=torch.long), x, y, g

    def __len__(self):
        return len(self.i)


class Waterbirds(GroupDataset):
    def __init__(self, data_path, split, subsample_what=None, duplicates=None):
        root = os.path.join(data_path, "waterbirds/waterbird_complete95_forest2water2/")
        metadata = os.path.join(data_path,"metadata_waterbirds.csv")

        transform = transforms.Compose(
            [
                transforms.Resize(
                    (
                        int(224 * (256 / 224)),
                        int(224 * (256 / 224)),
                    )
                ),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        super().__init__(split, root, metadata, transform, subsample_what, duplicates)
        self.data_type = "images"

    def transform(self, x):
        return self.transform_(Image.open(x).convert("RGB"))


class CelebA(GroupDataset):
    def __init__(self, data_path, split, subsample_what=None, duplicates=None):
        root = os.path.join(data_path, "celeba/img_align_celeba/")
        metadata = os.path.join(data_path,"metadata_celeba.csv")

        transform = transforms.Compose(
            [
                transforms.CenterCrop(178),
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        super().__init__(split, root, metadata, transform, subsample_what, duplicates)
        self.data_type = "images"

    def transform(self, x):
        return self.transform_(Image.open(x).convert("RGB"))


class MultiNLI(GroupDataset):
    def __init__(self, data_path, split, subsample_what=None, duplicates=None):
        root = os.path.join(data_path, "multinli", "glue_data", "MNLI")
        metadata = os.path.join(data_path, "metadata_multinli.csv")

        self.features_array = []
        for feature_file in [
            "cached_train_bert-base-uncased_128_mnli",
            "cached_dev_bert-base-uncased_128_mnli",
            "cached_dev_bert-base-uncased_128_mnli-mm",
        ]:
            features = torch.load(os.path.join(root, feature_file))
            self.features_array += features

        self.all_input_ids = torch.tensor(
            [f.input_ids for f in self.features_array], dtype=torch.long
        )
        self.all_input_masks = torch.tensor(
            [f.input_mask for f in self.features_array], dtype=torch.long
        )
        self.all_segment_ids = torch.tensor(
            [f.segment_ids for f in self.features_array], dtype=torch.long
        )
        self.all_label_ids = torch.tensor(
            [f.label_id for f in self.features_array], dtype=torch.long
        )

        self.x_array = torch.stack(
            (self.all_input_ids, self.all_input_masks, self.all_segment_ids), dim=2
        )

        self.data_type = "text"

        super().__init__(
            split, "", metadata, self.transform, subsample_what, duplicates
        )

    def transform(self, i):
        return self.x_array[int(i)]


class CivilComments(GroupDataset):
    def __init__(
        self,
        data_path,
        split,
        subsample_what=None,
        duplicates=None,
        granularity="coarse",
    ):
        metadata = os.path.join(data_path,"metadata_civilcomments_{}.csv".format(granularity))

        text = pd.read_csv(
            os.path.join(
                data_path, "civilcomments/civilcomments_{}.csv".format(granularity)
            )
        )

        self.text_array = list(text["comment_text"])
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.data_type = "text"

        super().__init__(
            split, "", metadata, self.transform, subsample_what, duplicates
        )

    def transform(self, idx):
        text = self.text_array[int(idx)]

        tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=220,
            return_tensors="pt",
        )

        return torch.squeeze(
            torch.stack(
                (
                    tokens["input_ids"],
                    tokens["attention_mask"],
                    tokens["token_type_ids"],
                ),
                dim=2,
            ),
            dim=0,
        )


class CivilCommentsFine(CivilComments):
    def __init__(self, data_path, split, subsample_what=None, duplicates=None):
        super().__init__(data_path, split, subsample_what, duplicates, "fine")


class Toy(GroupDataset):
    def __init__(self, data_path, split, subsample_what=None, duplicates=None):
        self.data_type = "toy"
        n_samples = 1000
        dim_noise = 1200

        self.i, self.x, self.y, self.g = self.make_dataset(
            n_samples=n_samples,
            dim_noise=dim_noise,
            core_cor=1.0,
            spu_cor=0.8,
            train=(split == "tr"),
        )
        self.count_groups()

        if subsample_what is not None:
            self.subsample_(subsample_what)

        if duplicates is not None:
            self.duplicate_(duplicates)

    def transform(self, x):
        return torch.tensor(x)

    def make_dataset(
        self,
        n_samples=1000,
        dim_noise=1200,
        blob_std=0.15,
        core_cor=1.0,
        spu_cor=0.8,
        train=True,
    ):
        X = make_blobs(n_samples=n_samples, centers=1, cluster_std=[blob_std])[0]
        X -= X.mean(0, keepdims=True) + np.array([[1.0, 1.0]])
        y = np.array([-1] * (n_samples // 2) + [1] * (n_samples // 2))
        g = np.ones((n_samples))

        # making of the core feature
        core_features = X[:, 0] * y
        # random without replacement
        random_indices_for_core = np.random.permutation(np.arange(n_samples))[
            : int((1 - core_cor) * n_samples)
        ]
        core_features[random_indices_for_core] *= -1
        g[random_indices_for_core] *= -1

        # making of the spurious feature
        spu_features = X[:, 1] * y
        random_indices_for_spu = np.random.permutation(np.arange(n_samples))[
            : int((1 - spu_cor) * n_samples)
        ]
        spu_features[random_indices_for_spu] *= -1
        g[random_indices_for_spu] *= -1

        X = np.vstack([spu_features, core_features]).T

        # noise = np.random.randn(n_samples, dim_noise) / np.sqrt(dim_noise)
        noise = np.random.randn(n_samples, dim_noise)
        if not train:
            # The average of noise is zero for both training and the test sets.
            # However, for the test set, we compute the "Expected loss" instead
            # of the "Empirical loss". For that reason, we can simply set the
            # noise to be zero for the test set.
            noise *= 0.0

        X = np.concatenate([X, noise], 1)
        i = np.arange(len(y))
        # y denotes the label
        # g denotes the group (minority or majority)
        # i denotes the index
        y = ((y + 1) / 2).astype(int)  # 0 or 1
        g = ((g + 1) / 2).astype(int)  # 0 or 1

        return i, X, y, g


def get_loaders(data_path, dataset_name, batch_size, method="erm", duplicates=None):
    Dataset = {
        "waterbirds": Waterbirds,
        "celeba": CelebA,
        "multinli": MultiNLI,
        "civilcomments": CivilCommentsFine
        if method in ("subg", "rwg")
        else CivilComments,
        "toy": Toy,
    }[dataset_name]

    def dl(dataset, bs, shuffle, weights):
        if weights is not None:
            sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))
        else:
            sampler = None
        return DataLoader(
            dataset,
            batch_size=bs,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
        )

    if method == "subg":
        subsample_what = "groups"
    elif method == "suby":
        subsample_what = "classes"
    else:
        subsample_what = None

    dataset_tr = Dataset(data_path, "tr", subsample_what, duplicates)

    if method == "rwg" or method == "dro":
        weights_tr = dataset_tr.wg
    elif method == "rwy":
        weights_tr = dataset_tr.wy
    else:
        weights_tr = None

    return {
        "tr": dl(dataset_tr, batch_size, weights_tr is None, weights_tr),
        "va": dl(Dataset(data_path, "va", None), 128, False, None),
        "te": dl(Dataset(data_path, "te", None), 128, False, None),
    }
