import os
import importlib.util

import numpy as np
import scipy.sparse as sp
import torch

from src.models.base_model import BaseBundleModel


def load_official_crosscbr_class(official_root):
    model_path = os.path.join(official_root, "models", "CrossCBR.py")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Cannot find official CrossCBR model file: {model_path}"
        )

    spec = importlib.util.spec_from_file_location(
        "official_crosscbr_module",
        model_path
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.CrossCBR


def dict_to_sparse_matrix(interaction_dict, n_left, n_right):
    rows = []
    cols = []

    for left_id, right_ids in interaction_dict.items():
        for right_id in right_ids:
            if 0 <= left_id < n_left and 0 <= right_id < n_right:
                rows.append(left_id)
                cols.append(right_id)

    if len(rows) == 0:
        return sp.coo_matrix(
            (n_left, n_right),
            dtype=np.float32
        ).tocsr()

    values = np.ones(len(rows), dtype=np.float32)

    matrix = sp.coo_matrix(
        (values, (np.array(rows), np.array(cols))),
        shape=(n_left, n_right)
    ).tocsr()

    return matrix


class CrossCBR(BaseBundleModel):
    """
    Official CrossCBR adapter.

    This class adopts the official CrossCBR model logic from:
        third_party/CrossCBR/models/CrossCBR.py

    It only adapts the outer interface to BundleRecFramework:
        calculate_loss(batch)
        predict(users)

    Official logic kept:
    - user/item/bundle embeddings
    - item-level graph
    - bundle-level graph
    - bundle aggregation graph
    - BPR loss
    - cross-view contrastive loss
    - official scoring function
    """

    def __init__(self, data, config):
        super().__init__(data, config)

        self.official_root = config.get(
            "official_root",
            "third_party/CrossCBR"
        )

        if not os.path.exists(self.official_root):
            raise FileNotFoundError(
                f"Official CrossCBR repo not found at {self.official_root}. "
                f"Run: bash scripts/official/setup_official_repos.sh"
            )

        self.device_name = config.get("device", "cuda")
        self.device = torch.device(
            "cuda" if self.device_name == "cuda" and torch.cuda.is_available()
            else "cpu"
        )

        self.official_conf = self.build_official_conf(data, config)
        self.raw_graph = self.build_raw_graph(data)

        OfficialCrossCBR = load_official_crosscbr_class(self.official_root)

        self.official_model = OfficialCrossCBR(
            self.official_conf,
            self.raw_graph
        )

        self.c_lambda = self.official_conf["c_lambda"]
        self.aug_type = self.official_conf["aug_type"]
        self.ed_interval = self.official_conf["ed_interval"]

        self.global_step = 0
        self.ed_interval_bs = None

        self._eval_cache = None

        print("[CrossCBR Adapter] official_root =", self.official_root)
        print("[CrossCBR Adapter] Adopted official CrossCBR model.")
        print("[CrossCBR Adapter] n_users =", self.n_users)
        print("[CrossCBR Adapter] n_bundles =", self.n_bundles)
        print("[CrossCBR Adapter] n_items =", self.n_items)
        print("[CrossCBR Adapter] c_lambda =", self.c_lambda)
        print("[CrossCBR Adapter] aug_type =", self.aug_type)

    def build_official_conf(self, data, config):
        """
        Map BundleRecFramework config to official CrossCBR config keys.
        """
        conf = {}

        conf["device"] = self.device
        conf["embedding_size"] = config.get(
            "embedding_size",
            config.get("embedding_dim", 64)
        )

        conf["l2_reg"] = config.get(
            "l2_reg",
            config.get("weight_decay", 0.0001)
        )

        conf["num_users"] = data.n_users
        conf["num_bundles"] = data.n_bundles
        conf["num_items"] = data.n_items

        conf["num_layers"] = config.get("num_layers", 1)

        conf["c_temp"] = config.get("c_temp", 0.25)
        conf["c_lambda"] = config.get("c_lambda", 0.04)

        conf["aug_type"] = config.get("aug_type", "ED")
        conf["ed_interval"] = config.get("ed_interval", 1)

        conf["item_level_ratio"] = config.get("item_level_ratio", 0.2)
        conf["bundle_level_ratio"] = config.get("bundle_level_ratio", 0.2)
        conf["bundle_agg_ratio"] = config.get("bundle_agg_ratio", 0.2)

        return conf

    def build_raw_graph(self, data):
        """
        Build official raw_graph = [u_b_graph_train, u_i_graph, b_i_graph].
        This follows official utility.py logic.
        """
        u_b_graph_train = dict_to_sparse_matrix(
            data.user_bundle_train,
            data.n_users,
            data.n_bundles
        )

        u_i_graph = dict_to_sparse_matrix(
            data.user_item,
            data.n_users,
            data.n_items
        )

        b_i_graph = dict_to_sparse_matrix(
            data.bundle_item,
            data.n_bundles,
            data.n_items
        )

        return [u_b_graph_train, u_i_graph, b_i_graph]

    def set_train_context(self, num_batches):
        """
        Official CrossCBR uses:
            ed_interval_bs = int(batch_cnt * conf["ed_interval"])

        The trainer calls this once after building train_loader.
        """
        self.ed_interval_bs = max(1, int(num_batches * self.ed_interval))
        print("[CrossCBR Adapter] ed_interval_bs =", self.ed_interval_bs)

    def train(self, mode=True):
        self._eval_cache = None
        return super().train(mode)

    def clear_cache(self):
        self._eval_cache = None

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.official_model.to(*args, **kwargs)
        return self

    def calculate_loss(self, batch):
        """
        Framework batch:
            {
                "user": [bs],
                "pos_bundle": [bs],
                "neg_bundle": [bs]
            }

        Official CrossCBR batch:
            users:   [bs, 1]
            bundles: [bs, 1 + neg_num]
        """
        users = batch["user"].view(-1, 1)
        pos_bundles = batch["pos_bundle"].view(-1, 1)
        neg_bundles = batch["neg_bundle"].view(-1, 1)

        bundles = torch.cat(
            [pos_bundles, neg_bundles],
            dim=1
        )

        official_batch = [users, bundles]

        self.global_step += 1

        ED_drop = False

        if self.aug_type == "ED":
            if self.ed_interval_bs is not None:
                if self.global_step % self.ed_interval_bs == 0:
                    ED_drop = True

        bpr_loss, c_loss = self.official_model(
            official_batch,
            ED_drop=ED_drop
        )

        loss = bpr_loss + self.c_lambda * c_loss

        return loss

    def predict(self, users):
        """
        Return score matrix:
            [len(users), n_bundles]

        This follows official test logic:
            rs = model.propagate(test=True)
            scores = model.evaluate(rs, users)
        """
        if self._eval_cache is None:
            self._eval_cache = self.official_model.propagate(test=True)

        users = users.view(-1)

        scores = self.official_model.evaluate(
            self._eval_cache,
            users
        )

        return scores
