import torch.nn as nn


class BaseBundleModel(nn.Module):
    def __init__(self, data, config):
        super().__init__()

        self.data = data
        self.config = config

        self.n_users = data.n_users
        self.n_bundles = data.n_bundles
        self.n_items = data.n_items

    def calculate_loss(self, batch):
        raise NotImplementedError

    def predict(self, users):
        raise NotImplementedError

    def full_sort_predict(self, users):
        return self.predict(users)
