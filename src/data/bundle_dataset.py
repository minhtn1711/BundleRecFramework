import os
import random
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader

from src.data.base_dataset import BaseBundleDataset


def read_interaction_file(path):
    """
    Support both formats:

    Pair format:
        left_id right_id

    Grouped format:
        left_id right_id_1 right_id_2 right_id_3

    Return:
        interactions: dict[left_id] = sorted unique list of right ids
        max_left_id
        max_right_id
    """
    interactions = defaultdict(list)
    max_left_id = -1
    max_right_id = -1

    if not os.path.exists(path):
        print(f"[Warning] File not found: {path}")
        return {}, max_left_id, max_right_id

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line == "":
                continue

            parts = line.replace(",", " ").replace("\t", " ").split()

            if len(parts) <= 1:
                continue

            left_id = int(parts[0])
            right_ids = [int(x) for x in parts[1:]]

            interactions[left_id].extend(right_ids)

            max_left_id = max(max_left_id, left_id)

            if len(right_ids) > 0:
                max_right_id = max(max_right_id, max(right_ids))

    final_interactions = {}

    for left_id, right_ids in interactions.items():
        final_interactions[left_id] = sorted(set(right_ids))

    return final_interactions, max_left_id, max_right_id


def find_existing_file(data_dir, candidates):
    for filename in candidates:
        path = os.path.join(data_dir, filename)

        if os.path.exists(path):
            return path

    return os.path.join(data_dir, candidates[0])


class PairwiseBundleTrainDataset(Dataset):
    """
    General pairwise training dataset for bundle recommendation.

    It returns a common batch format:
        {
            "user": user_id,
            "pos_bundle": positive_bundle_id,
            "neg_bundle": negative_bundle_id or tensor of negative ids
        }

    Model-specific adapters can convert this common batch format internally.
    """

    def __init__(self, user_bundle_train, n_bundles, neg_num=1):
        self.user_bundle_train = user_bundle_train
        self.n_bundles = n_bundles
        self.neg_num = neg_num

        self.samples = []
        self.user_pos_set = {}

        for user, bundles in user_bundle_train.items():
            bundle_set = set(bundles)
            self.user_pos_set[user] = bundle_set

            for bundle in bundles:
                self.samples.append((user, bundle))

        if len(self.samples) == 0:
            raise ValueError("No training samples found in user_bundle_train.")

    def __len__(self):
        return len(self.samples)

    def sample_one_negative(self, user):
        positive_set = self.user_pos_set.get(user, set())

        while True:
            neg_bundle = random.randint(0, self.n_bundles - 1)

            if neg_bundle not in positive_set:
                return neg_bundle

    def sample_negative_bundles(self, user):
        negs = []
        used = set()

        while len(negs) < self.neg_num:
            neg_bundle = self.sample_one_negative(user)

            if neg_bundle not in used:
                used.add(neg_bundle)
                negs.append(neg_bundle)

        return negs

    def __getitem__(self, idx):
        user, pos_bundle = self.samples[idx]

        neg_bundles = self.sample_negative_bundles(user)

        if self.neg_num == 1:
            neg_bundle = neg_bundles[0]
            neg_tensor = torch.tensor(neg_bundle, dtype=torch.long)
        else:
            neg_tensor = torch.tensor(neg_bundles, dtype=torch.long)

        return {
            "user": torch.tensor(user, dtype=torch.long),
            "pos_bundle": torch.tensor(pos_bundle, dtype=torch.long),
            "neg_bundle": neg_tensor
        }


class GeneralBundleDataset(BaseBundleDataset):
    """
    General dataset object for bundle recommendation.

    It does not contain model-specific logic.
    It only prepares:
        user-bundle train/val/test
        user-item
        bundle-item
        train dataloader
        eval dictionaries
    """

    def __init__(self, config):
        super().__init__(config)

        self.data_dir = config["data_dir"]
        self.batch_size = config["batch_size"]
        self.num_workers = config.get("num_workers", 2)
        self.neg_num = config.get("neg_num", 1)
        self.drop_last = config.get("drop_last", True)

        self._load_files()
        self._build_train_dataset()

    def _load_files(self):
        train_path = find_existing_file(
            self.data_dir,
            ["user_bundle_train.txt", "train.txt"]
        )

        val_path = find_existing_file(
            self.data_dir,
            ["user_bundle_val.txt", "user_bundle_tune.txt", "val.txt", "tune.txt"]
        )

        test_path = find_existing_file(
            self.data_dir,
            ["user_bundle_test.txt", "test.txt"]
        )

        bundle_item_path = find_existing_file(
            self.data_dir,
            ["bundle_item.txt", "bundle_items.txt"]
        )

        user_item_path = find_existing_file(
            self.data_dir,
            ["user_item.txt", "user_items.txt"]
        )

        self.user_bundle_train, max_user_train, max_bundle_train = read_interaction_file(train_path)
        self.user_bundle_val, max_user_val, max_bundle_val = read_interaction_file(val_path)
        self.user_bundle_test, max_user_test, max_bundle_test = read_interaction_file(test_path)

        self.bundle_item, max_bundle_bi, max_item_bi = read_interaction_file(bundle_item_path)
        self.user_item, max_user_ui, max_item_ui = read_interaction_file(user_item_path)

        self.n_users = max(
            max_user_train,
            max_user_val,
            max_user_test,
            max_user_ui
        ) + 1

        self.n_bundles = max(
            max_bundle_train,
            max_bundle_val,
            max_bundle_test,
            max_bundle_bi
        ) + 1

        self.n_items = max(
            max_item_bi,
            max_item_ui
        ) + 1

        if self.n_users <= 0 or self.n_bundles <= 0:
            raise ValueError(
                "Dataset is empty or invalid. "
                "Please check user_bundle_train/test files."
            )

        print(f"[Dataset] name = {self.config['dataset']}")
        print(f"[Dataset] data_dir = {self.data_dir}")
        print(f"[Dataset] n_users = {self.n_users}")
        print(f"[Dataset] n_bundles = {self.n_bundles}")
        print(f"[Dataset] n_items = {self.n_items}")
        print(f"[Dataset] train file = {train_path}")
        print(f"[Dataset] val file = {val_path}")
        print(f"[Dataset] test file = {test_path}")
        print(f"[Dataset] bundle-item file = {bundle_item_path}")
        print(f"[Dataset] user-item file = {user_item_path}")

    def _build_train_dataset(self):
        self.train_dataset = PairwiseBundleTrainDataset(
            user_bundle_train=self.user_bundle_train,
            n_bundles=self.n_bundles,
            neg_num=self.neg_num
        )

        print(f"[Dataset] train pairs = {len(self.train_dataset)}")
        print(f"[Dataset] neg_num = {self.neg_num}")

    def get_train_loader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=self.drop_last
        )

    def get_train_user_pos_items(self):
        return self.user_bundle_train

    def get_eval_dict(self, split="test"):
        if split == "val":
            return self.user_bundle_val

        if split == "test":
            return self.user_bundle_test

        raise ValueError(f"Unknown split: {split}")

    def get_all_user_bundle_dict(self, split="train"):
        if split == "train":
            return self.user_bundle_train

        if split == "val":
            return self.user_bundle_val

        if split == "test":
            return self.user_bundle_test

        raise ValueError(f"Unknown split: {split}")
