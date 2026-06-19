"""
CrossCBR model adopted into BundleRecFramework.

This implementation ports the core logic of the official CrossCBR model into
this framework, so the model can run without importing from third_party at
runtime.

Original model: CrossCBR
Official repository: https://github.com/mysbupt/CrossCBR

The outer interface is adapted to BundleRecFramework:
    - calculate_loss(batch)
    - predict(users)

The core model logic is kept:
    - user / bundle / item embeddings
    - item-level graph
    - bundle-level graph
    - bundle aggregation graph
    - graph propagation
    - BPR loss
    - cross-view contrastive loss
    - official scoring function
"""

import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base_model import BaseBundleModel


def cal_bpr_loss(pred):
    """
    Official CrossCBR BPR loss.

    pred: [batch_size, 1 + neg_num]
    pred[:, 0] is positive bundle score.
    pred[:, 1:] are negative bundle scores.
    """
    if pred.shape[1] > 2:
        negs = pred[:, 1:]
        pos = pred[:, 0].unsqueeze(1).expand_as(negs)
    else:
        negs = pred[:, 1].unsqueeze(1)
        pos = pred[:, 0].unsqueeze(1)

    loss = -torch.log(torch.sigmoid(pos - negs))
    loss = torch.mean(loss)

    return loss


def laplace_transform(graph):
    """
    Official CrossCBR graph normalization:
        D_row^-1/2 * A * D_col^-1/2
    """
    rowsum_sqrt = sp.diags(
        1 / (np.sqrt(graph.sum(axis=1).A.ravel()) + 1e-8)
    )

    colsum_sqrt = sp.diags(
        1 / (np.sqrt(graph.sum(axis=0).A.ravel()) + 1e-8)
    )

    graph = rowsum_sqrt @ graph @ colsum_sqrt

    return graph


def to_tensor(graph):
    """
    Convert scipy sparse matrix to torch sparse tensor.
    """
    graph = graph.tocoo()

    values = graph.data
    indices = np.vstack((graph.row, graph.col))

    graph = torch.sparse_coo_tensor(
        torch.LongTensor(indices),
        torch.FloatTensor(values),
        torch.Size(graph.shape)
    )

    return graph.coalesce()


def np_edge_dropout(values, dropout_ratio):
    """
    Official CrossCBR edge dropout.
    """
    mask = np.random.choice(
        [0, 1],
        size=(len(values),),
        p=[dropout_ratio, 1 - dropout_ratio]
    )

    values = mask * values

    return values


def dict_to_sparse_matrix(interaction_dict, n_left, n_right):
    rows = []
    cols = []

    for left_id, right_ids in interaction_dict.items():
        if left_id < 0 or left_id >= n_left:
            continue

        for right_id in right_ids:
            if right_id < 0 or right_id >= n_right:
                continue

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
    Self-contained CrossCBR implementation inside BundleRecFramework.

    This class no longer depends on:
        third_party/CrossCBR/models/CrossCBR.py

    It keeps the CrossCBR model logic and adapts only input/output to the
    framework.
    """

    def __init__(self, data, config):
        super().__init__(data, config)

        self.conf = self.build_model_conf(data, config)

        self.device = self.conf["device"]

        self.embedding_size = self.conf["embedding_size"]
        self.embed_L2_norm = self.conf["l2_reg"]

        self.num_users = self.conf["num_users"]
        self.num_bundles = self.conf["num_bundles"]
        self.num_items = self.conf["num_items"]

        self.num_layers = self.conf["num_layers"]
        self.c_temp = self.conf["c_temp"]

        self.c_lambda = self.conf["c_lambda"]
        self.aug_type = self.conf["aug_type"]
        self.ed_interval = self.conf["ed_interval"]

        self.global_step = 0
        self.ed_interval_bs = None

        self.init_emb()

        self.ub_graph, self.ui_graph, self.bi_graph = self.build_raw_graph(data)

        # Original graphs for testing.
        self.get_item_level_graph_ori()
        self.get_bundle_level_graph_ori()
        self.get_bundle_agg_graph_ori()

        # Augmented graphs for training.
        self.get_item_level_graph()
        self.get_bundle_level_graph()
        self.get_bundle_agg_graph()

        self.init_md_dropouts()

        self._eval_cache = None

        print("[CrossCBR] Self-contained CrossCBR adopted into framework.")
        print("[CrossCBR] n_users =", self.num_users)
        print("[CrossCBR] n_bundles =", self.num_bundles)
        print("[CrossCBR] n_items =", self.num_items)
        print("[CrossCBR] embedding_size =", self.embedding_size)
        print("[CrossCBR] num_layers =", self.num_layers)
        print("[CrossCBR] aug_type =", self.aug_type)
        print("[CrossCBR] c_lambda =", self.c_lambda)
        print("[CrossCBR] c_temp =", self.c_temp)

    def build_model_conf(self, data, config):
        device_name = config.get("device", "cuda")

        if device_name == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        conf = {
            "device": device,

            "embedding_size": config.get(
                "embedding_size",
                config.get("embedding_dim", 64)
            ),

            "l2_reg": config.get(
                "l2_reg",
                config.get("weight_decay", 0.0001)
            ),

            "num_users": data.n_users,
            "num_bundles": data.n_bundles,
            "num_items": data.n_items,

            "num_layers": config.get("num_layers", 1),

            "c_temp": config.get("c_temp", 0.25),
            "c_lambda": config.get("c_lambda", 0.04),

            "aug_type": config.get("aug_type", "ED"),
            "ed_interval": config.get("ed_interval", 1),

            "item_level_ratio": config.get("item_level_ratio", 0.2),
            "bundle_level_ratio": config.get("bundle_level_ratio", 0.2),
            "bundle_agg_ratio": config.get("bundle_agg_ratio", 0.2)
        }

        return conf

    def build_raw_graph(self, data):
        """
        Build:
            ub_graph: user-bundle graph
            ui_graph: user-item graph
            bi_graph: bundle-item graph
        """
        ub_graph = dict_to_sparse_matrix(
            data.user_bundle_train,
            self.num_users,
            self.num_bundles
        )

        ui_graph = dict_to_sparse_matrix(
            data.user_item,
            self.num_users,
            self.num_items
        )

        bi_graph = dict_to_sparse_matrix(
            data.bundle_item,
            self.num_bundles,
            self.num_items
        )

        return ub_graph, ui_graph, bi_graph

    def init_emb(self):
        self.users_feature = nn.Parameter(
            torch.FloatTensor(self.num_users, self.embedding_size)
        )
        nn.init.xavier_normal_(self.users_feature)

        self.bundles_feature = nn.Parameter(
            torch.FloatTensor(self.num_bundles, self.embedding_size)
        )
        nn.init.xavier_normal_(self.bundles_feature)

        self.items_feature = nn.Parameter(
            torch.FloatTensor(self.num_items, self.embedding_size)
        )
        nn.init.xavier_normal_(self.items_feature)

    def init_md_dropouts(self):
        self.item_level_dropout = nn.Dropout(
            self.conf["item_level_ratio"],
            inplace=True
        )

        self.bundle_level_dropout = nn.Dropout(
            self.conf["bundle_level_ratio"],
            inplace=True
        )

        self.bundle_agg_dropout = nn.Dropout(
            self.conf["bundle_agg_ratio"],
            inplace=True
        )

    def get_item_level_graph(self):
        ui_graph = self.ui_graph
        device = self.device
        modification_ratio = self.conf["item_level_ratio"]

        item_level_graph = sp.bmat(
            [
                [
                    sp.csr_matrix((ui_graph.shape[0], ui_graph.shape[0])),
                    ui_graph
                ],
                [
                    ui_graph.T,
                    sp.csr_matrix((ui_graph.shape[1], ui_graph.shape[1]))
                ]
            ]
        )

        if modification_ratio != 0:
            if self.conf["aug_type"] == "ED":
                graph = item_level_graph.tocoo()
                values = np_edge_dropout(graph.data, modification_ratio)
                item_level_graph = sp.coo_matrix(
                    (values, (graph.row, graph.col)),
                    shape=graph.shape
                ).tocsr()

        self.item_level_graph = to_tensor(
            laplace_transform(item_level_graph)
        ).to(device)

    def get_item_level_graph_ori(self):
        ui_graph = self.ui_graph
        device = self.device

        item_level_graph = sp.bmat(
            [
                [
                    sp.csr_matrix((ui_graph.shape[0], ui_graph.shape[0])),
                    ui_graph
                ],
                [
                    ui_graph.T,
                    sp.csr_matrix((ui_graph.shape[1], ui_graph.shape[1]))
                ]
            ]
        )

        self.item_level_graph_ori = to_tensor(
            laplace_transform(item_level_graph)
        ).to(device)

    def get_bundle_level_graph(self):
        ub_graph = self.ub_graph
        device = self.device
        modification_ratio = self.conf["bundle_level_ratio"]

        bundle_level_graph = sp.bmat(
            [
                [
                    sp.csr_matrix((ub_graph.shape[0], ub_graph.shape[0])),
                    ub_graph
                ],
                [
                    ub_graph.T,
                    sp.csr_matrix((ub_graph.shape[1], ub_graph.shape[1]))
                ]
            ]
        )

        if modification_ratio != 0:
            if self.conf["aug_type"] == "ED":
                graph = bundle_level_graph.tocoo()
                values = np_edge_dropout(graph.data, modification_ratio)
                bundle_level_graph = sp.coo_matrix(
                    (values, (graph.row, graph.col)),
                    shape=graph.shape
                ).tocsr()

        self.bundle_level_graph = to_tensor(
            laplace_transform(bundle_level_graph)
        ).to(device)

    def get_bundle_level_graph_ori(self):
        ub_graph = self.ub_graph
        device = self.device

        bundle_level_graph = sp.bmat(
            [
                [
                    sp.csr_matrix((ub_graph.shape[0], ub_graph.shape[0])),
                    ub_graph
                ],
                [
                    ub_graph.T,
                    sp.csr_matrix((ub_graph.shape[1], ub_graph.shape[1]))
                ]
            ]
        )

        self.bundle_level_graph_ori = to_tensor(
            laplace_transform(bundle_level_graph)
        ).to(device)

    def get_bundle_agg_graph(self):
        bi_graph = self.bi_graph
        device = self.device

        if self.conf["aug_type"] == "ED":
            modification_ratio = self.conf["bundle_agg_ratio"]
            graph = self.bi_graph.tocoo()
            values = np_edge_dropout(graph.data, modification_ratio)
            bi_graph = sp.coo_matrix(
                (values, (graph.row, graph.col)),
                shape=graph.shape
            ).tocsr()

        bundle_size = bi_graph.sum(axis=1) + 1e-8
        bi_graph = sp.diags(1 / bundle_size.A.ravel()) @ bi_graph

        self.bundle_agg_graph = to_tensor(bi_graph).to(device)

    def get_bundle_agg_graph_ori(self):
        bi_graph = self.bi_graph
        device = self.device

        bundle_size = bi_graph.sum(axis=1) + 1e-8
        bi_graph = sp.diags(1 / bundle_size.A.ravel()) @ bi_graph

        self.bundle_agg_graph_ori = to_tensor(bi_graph).to(device)

    def one_propagate(self, graph, A_feature, B_feature, mess_dropout, test):
        features = torch.cat((A_feature, B_feature), dim=0)

        all_features = [features]

        for i in range(self.num_layers):
            features = torch.spmm(graph, features)

            if self.conf["aug_type"] == "MD" and not test:
                features = mess_dropout(features)

            features = features / (i + 2)

            all_features.append(
                F.normalize(features, p=2, dim=1)
            )

        all_features = torch.stack(all_features, dim=1)
        all_features = torch.sum(all_features, dim=1).squeeze(1)

        A_feature, B_feature = torch.split(
            all_features,
            (A_feature.shape[0], B_feature.shape[0]),
            dim=0
        )

        return A_feature, B_feature

    def get_IL_bundle_rep(self, IL_items_feature, test):
        if test:
            IL_bundles_feature = torch.matmul(
                self.bundle_agg_graph_ori,
                IL_items_feature
            )
        else:
            IL_bundles_feature = torch.matmul(
                self.bundle_agg_graph,
                IL_items_feature
            )

        if (
            self.conf["bundle_agg_ratio"] != 0
            and self.conf["aug_type"] == "MD"
            and not test
        ):
            IL_bundles_feature = self.bundle_agg_dropout(
                IL_bundles_feature
            )

        return IL_bundles_feature

    def propagate(self, test=False):
        # Item-level propagation.
        if test:
            IL_users_feature, IL_items_feature = self.one_propagate(
                self.item_level_graph_ori,
                self.users_feature,
                self.items_feature,
                self.item_level_dropout,
                test
            )
        else:
            IL_users_feature, IL_items_feature = self.one_propagate(
                self.item_level_graph,
                self.users_feature,
                self.items_feature,
                self.item_level_dropout,
                test
            )

        IL_bundles_feature = self.get_IL_bundle_rep(
            IL_items_feature,
            test
        )

        # Bundle-level propagation.
        if test:
            BL_users_feature, BL_bundles_feature = self.one_propagate(
                self.bundle_level_graph_ori,
                self.users_feature,
                self.bundles_feature,
                self.bundle_level_dropout,
                test
            )
        else:
            BL_users_feature, BL_bundles_feature = self.one_propagate(
                self.bundle_level_graph,
                self.users_feature,
                self.bundles_feature,
                self.bundle_level_dropout,
                test
            )

        users_feature = [IL_users_feature, BL_users_feature]
        bundles_feature = [IL_bundles_feature, BL_bundles_feature]

        return users_feature, bundles_feature

    def cal_c_loss(self, pos, aug):
        pos = pos[:, 0, :]
        aug = aug[:, 0, :]

        pos = F.normalize(pos, p=2, dim=1)
        aug = F.normalize(aug, p=2, dim=1)

        pos_score = torch.sum(pos * aug, dim=1)
        ttl_score = torch.matmul(pos, aug.permute(1, 0))

        pos_score = torch.exp(pos_score / self.c_temp)
        ttl_score = torch.sum(
            torch.exp(ttl_score / self.c_temp),
            axis=1
        )

        c_loss = -torch.mean(torch.log(pos_score / ttl_score))

        return c_loss

    def cal_loss(self, users_feature, bundles_feature):
        IL_users_feature, BL_users_feature = users_feature
        IL_bundles_feature, BL_bundles_feature = bundles_feature

        pred = (
            torch.sum(IL_users_feature * IL_bundles_feature, dim=2)
            + torch.sum(BL_users_feature * BL_bundles_feature, dim=2)
        )

        bpr_loss = cal_bpr_loss(pred)

        u_cross_view_cl = self.cal_c_loss(
            IL_users_feature,
            BL_users_feature
        )

        b_cross_view_cl = self.cal_c_loss(
            IL_bundles_feature,
            BL_bundles_feature
        )

        c_losses = [u_cross_view_cl, b_cross_view_cl]
        c_loss = sum(c_losses) / len(c_losses)

        return bpr_loss, c_loss

    def forward(self, batch, ED_drop=False):
        if ED_drop:
            self.get_item_level_graph()
            self.get_bundle_level_graph()
            self.get_bundle_agg_graph()

        users, bundles = batch

        users_feature, bundles_feature = self.propagate()

        users_embedding = [
            feature[users].expand(-1, bundles.shape[1], -1)
            for feature in users_feature
        ]

        bundles_embedding = [
            feature[bundles]
            for feature in bundles_feature
        ]

        bpr_loss, c_loss = self.cal_loss(
            users_embedding,
            bundles_embedding
        )

        return bpr_loss, c_loss

    def evaluate(self, propagate_result, users):
        users_feature, bundles_feature = propagate_result

        users_feature_atom, users_feature_non_atom = [
            feature[users]
            for feature in users_feature
        ]

        bundles_feature_atom, bundles_feature_non_atom = bundles_feature

        scores = (
            torch.mm(users_feature_atom, bundles_feature_atom.t())
            + torch.mm(users_feature_non_atom, bundles_feature_non_atom.t())
        )

        return scores

    def set_train_context(self, num_batches):
        self.ed_interval_bs = max(
            1,
            int(num_batches * self.ed_interval)
        )

        print("[CrossCBR] ed_interval_bs =", self.ed_interval_bs)

    def train(self, mode=True):
        self._eval_cache = None
        return super().train(mode)

    def clear_cache(self):
        self._eval_cache = None

    def calculate_loss(self, batch):
        users = batch["user"].view(-1, 1)
        pos_bundles = batch["pos_bundle"].view(-1, 1)

        neg_bundles = batch["neg_bundle"]

        if neg_bundles.dim() == 1:
            neg_bundles = neg_bundles.view(-1, 1)

        bundles = torch.cat(
            [pos_bundles, neg_bundles],
            dim=1
        )

        official_batch = [users, bundles]

        self.global_step += 1

        ED_drop = False

        if self.aug_type == "ED" and self.ed_interval_bs is not None:
            if self.global_step % self.ed_interval_bs == 0:
                ED_drop = True

        bpr_loss, c_loss = self.forward(
            official_batch,
            ED_drop=ED_drop
        )

        loss = bpr_loss + self.c_lambda * c_loss

        return loss

    def predict(self, users):
        if self._eval_cache is None:
            self._eval_cache = self.propagate(test=True)

        users = users.view(-1)

        scores = self.evaluate(
            self._eval_cache,
            users
        )

        return scores
