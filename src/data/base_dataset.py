class BaseBundleDataset:
    def __init__(self, config):
        self.config = config

    def get_train_loader(self):
        raise NotImplementedError

    def get_train_user_pos_items(self):
        raise NotImplementedError

    def get_eval_dict(self, split="test"):
        raise NotImplementedError
