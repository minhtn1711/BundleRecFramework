"""
BGCN model adopted into BundleRecFramework.

This implementation ports the core logic of the official BGCN model into
this framework, so the model can run without importing from third_party at
runtime.

Original model: BGCN
Official repository: https://github.com/cjx0525/BGCN

The outer interface is adapted to BundleRecFramework:
    - calculate_loss(batch)
    - predict(users)

The core model logic is kept:
    - user / bundle / item embeddings
    - atom graph: user-item graph
    - non-atom graph: user-bundle graph + bundle-bundle graph
    - pooling graph: bundle-item aggregation
    - BGCN propagation
    - BPR loss with model regularization
    - official scoring function
"""

import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base_model import BaseBundleModel


def laplace_transform(graph):
    """
    Official BGCN graph normalization:
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


class BGCN(BaseBundleModel):
    """
    Self-contained BGCN implementation inside BundleRecFramework.

    This class no longer depends on:
        third_party/BGCN/model/BGCN.py

    It keeps the BGCN model logic and adapts only input/output to the framework.
    """

    def __init__(self, data, config):
        super().__init__(data, config)

        self.device_name = config.get("device", "cuda")

        if self.device_name == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.embedding_size = config.get(
            "embedding_size",
            config.get("embedding_dim", 64)
        )

        self.embed_L2_norm = config.get(
            "embed_L2_norm",
            config.get("l2_reg", config.get("weight_decay", 1e-4))
        )

        self.mess_dropout_ratio = config.get("message_dropout", 0.1)
        self.node_dropout_ratio = config.get("node_dropout", 0.1)
        self.num_layers = config.get("num_layers", 2)

        self.act = nn.LeakyReLU()

        self.num_users = data.n_users
        self.num_bundles = data.n_bundles
        self.num_items = data.n_items

        self.epison = 1e-8

        self.init_embedding()
        self.build_graphs(data)
        self.init_dropouts()
        self.init_layers()

        self._eval_cache = None

        print("[BGCN] Self-contained BGCN adopted into framework.")
        print("[BGCN] n_users =", self.num_users)
        print("[BGCN] n_bundles =", self.num_bundles)
        print("[BGCN] n_items =", self.num_items)
        print("[BGCN] embedding_size =", self.embedding_size)
        print("[BGCN] num_layers =", self.num_layers)
        print("[BGCN] embed_L2_norm =", self.embed_L2_norm)
        print("[BGCN] message_dropout =", self.mess_dropout_ratio)
        print("[BGCN] node_dropout =", self.node_dropout_ratio)

    def init_embedding(self):
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

    def build_graphs(self, data):
        """
        Build official BGCN graphs from framework dataset object.

        raw_graph:
            ub_graph: user-bundle interaction graph
            ui_graph: user-item interaction graph
            bi_graph: bundle-item affiliation graph
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

        # Bundle-bundle graph from bundle-item graph.
        bi_norm = (
            sp.diags(
                1 / (
                    np.sqrt(
                        (bi_graph.multiply(bi_graph)).sum(axis=1).A.ravel()
                    ) + 1e-8
                )
            )
            @ bi_graph
        )

        bb_graph = bi_norm @ bi_norm.T

        # Pooling graph: row-normalized bundle-item graph.
        bundle_size = bi_graph.sum(axis=1) + 1e-8
        pooling_graph = sp.diags(1 / bundle_size.A.ravel()) @ bi_graph

        # Atom graph: user-item graph with self-loop.
        if ui_graph.shape != (self.num_users, self.num_items):
            raise ValueError("ui_graph shape is wrong.")

        atom_graph = sp.bmat(
            [
                [
                    sp.identity(ui_graph.shape[0]),
                    ui_graph
                ],
                [
                    ui_graph.T,
                    sp.identity(ui_graph.shape[1])
                ]
            ]
        )

        # Non-atom graph: user-bundle graph + bundle-bundle graph.
        if ub_graph.shape != (self.num_users, self.num_bundles):
            raise ValueError("ub_graph shape is wrong.")

        if bb_graph.shape != (self.num_bundles, self.num_bundles):
            raise ValueError("bb_graph shape is wrong.")

        non_atom_graph = sp.bmat(
            [
                [
                    sp.identity(ub_graph.shape[0]),
                    ub_graph
                ],
                [
                    ub_graph.T,
                    bb_graph
                ]
            ]
        )

        self.atom_graph = to_tensor(
            laplace_transform(atom_graph)
        ).to(self.device)

        self.non_atom_graph = to_tensor(
            laplace_transform(non_atom_graph)
        ).to(self.device)

        self.pooling_graph = to_tensor(
            pooling_graph
        ).to(self.device)

        print("[BGCN] finish generating atom graph")
        print("[BGCN] finish generating non-atom graph")
        print("[BGCN] finish generating pooling graph")
        print("[BGCN] atom graph edges =", self.atom_graph._nnz())
        print("[BGCN] non-atom graph edges =", self.non_atom_graph._nnz())
        print("[BGCN] pooling graph edges =", self.pooling_graph._nnz())

    def init_dropouts(self):
        self.mess_dropout = nn.Dropout(
            self.mess_dropout_ratio,
            inplace=True
        )

        self.node_dropout = nn.Dropout(
            self.node_dropout_ratio,
            inplace=True
        )

    def init_layers(self):
        self.dnns_atom = nn.ModuleList(
            [
                nn.Linear(
                    self.embedding_size * (layer + 1),
                    self.embedding_size
                )
                for layer in range(self.num_layers)
            ]
        )

        self.dnns_non_atom = nn.ModuleList(
            [
                nn.Linear(
                    self.embedding_size * (layer + 1),
                    self.embedding_size
                )
                for layer in range(self.num_layers)
            ]
        )

    def train(self, mode=True):
        self._eval_cache = None
        return super().train(mode)

    def clear_cache(self):
        self._eval_cache = None

    def one_propagate(self, graph, A_feature, B_feature, dnns):
        """
        Official BGCN one-view propagation.
        """
        # Node dropout on sparse graph values.
        indices = graph._indices()
        values = graph._values()

        values = self.node_dropout(values)

        graph = torch.sparse_coo_tensor(
            indices,
            values,
            size=graph.shape,
            device=values.device
        ).coalesce()

        # Propagate.
        features = torch.cat((A_feature, B_feature), dim=0)
        all_features = [features]

        for layer in range(self.num_layers):
            transformed = torch.matmul(graph, features)
            transformed = dnns[layer](transformed)
            transformed = self.act(transformed)

            features = torch.cat(
                [transformed, features],
                dim=1
            )

            features = self.mess_dropout(features)

            all_features.append(
                F.normalize(features, dim=1)
            )

        all_features = torch.cat(all_features, dim=1)

        A_feature, B_feature = torch.split(
            all_features,
            (A_feature.shape[0], B_feature.shape[0]),
            dim=0
        )

        return A_feature, B_feature

    def propagate(self):
        # Item-level propagation.
        atom_users_feature, atom_items_feature = self.one_propagate(
            self.atom_graph,
            self.users_feature,
            self.items_feature,
            self.dnns_atom
        )

        atom_bundles_feature = F.normalize(
            torch.matmul(
                self.pooling_graph,
                atom_items_feature
            ),
            dim=1
        )

        # Bundle-level propagation.
        non_atom_users_feature, non_atom_bundles_feature = self.one_propagate(
            self.non_atom_graph,
            self.users_feature,
            self.bundles_feature,
            self.dnns_non_atom
        )

        users_feature = [
            atom_users_feature,
            non_atom_users_feature
        ]

        bundles_feature = [
            atom_bundles_feature,
            non_atom_bundles_feature
        ]

        return users_feature, bundles_feature

    def score_target_bundles(self, users_feature, bundles_feature):
        users_feature_atom, users_feature_non_atom = users_feature
        bundles_feature_atom, bundles_feature_non_atom = bundles_feature

        pred = (
            torch.sum(
                users_feature_atom * bundles_feature_atom,
                dim=2
            )
            + torch.sum(
                users_feature_non_atom * bundles_feature_non_atom,
                dim=2
            )
        )

        return pred

    def forward(self, users, bundles):
        users_feature, bundles_feature = self.propagate()

        users_embedding = [
            feature[users].expand(-1, bundles.shape[1], -1)
            for feature in users_feature
        ]

        bundles_embedding = [
            feature[bundles]
            for feature in bundles_feature
        ]

        pred = self.score_target_bundles(
            users_embedding,
            bundles_embedding
        )

        reg_loss = self.regularize(
            users_embedding,
            bundles_embedding
        )

        return pred, reg_loss

    def regularize(self, users_feature, bundles_feature):
        users_feature_atom, users_feature_non_atom = users_feature
        bundles_feature_atom, bundles_feature_non_atom = bundles_feature

        loss = self.embed_L2_norm * (
            (users_feature_atom ** 2).sum()
            + (bundles_feature_atom ** 2).sum()
            + (users_feature_non_atom ** 2).sum()
            + (bundles_feature_non_atom ** 2).sum()
        )

        return loss

    def bpr_loss(self, pred):
        """
        Official BGCN BPR loss.

        pred[:, 0] is positive bundle score.
        pred[:, 1] is negative bundle score.
        """
        loss = -torch.log(
            torch.sigmoid(pred[:, 0] - pred[:, 1]) + 1e-8
        )

        loss = torch.mean(loss)

        return loss

    def calculate_loss(self, batch):
        users = batch["user"].view(-1, 1)
        pos_bundles = batch["pos_bundle"].view(-1, 1)

        neg_bundles = batch["neg_bundle"]

        if neg_bundles.dim() == 1:
            neg_bundles = neg_bundles.view(-1, 1)

        # BGCN official uses one negative bundle for BPR.
        if neg_bundles.shape[1] > 1:
            neg_bundles = neg_bundles[:, :1]

        bundles = torch.cat(
            [pos_bundles, neg_bundles],
            dim=1
        )

        pred, reg_loss = self.forward(
            users,
            bundles
        )

        bpr = self.bpr_loss(pred)

        loss = bpr + reg_loss / users.shape[0]

        return loss

    def evaluate(self, propagate_result, users):
        users_feature, bundles_feature = propagate_result

        users_feature_atom, users_feature_non_atom = [
            feature[users]
            for feature in users_feature
        ]

        bundles_feature_atom, bundles_feature_non_atom = bundles_feature

        scores = (
            torch.mm(
                users_feature_atom,
                bundles_feature_atom.t()
            )
            + torch.mm(
                users_feature_non_atom,
                bundles_feature_non_atom.t()
            )
        )

        return scores

    def predict(self, users):
        if self._eval_cache is None:
            self._eval_cache = self.propagate()

        users = users.view(-1)

        scores = self.evaluate(
            self._eval_cache,
            users
        )

        return scores
