import os
import sys

from src.models.base_model import BaseBundleModel


class BGCN(BaseBundleModel):
    """
    Adapter for official BGCN implementation.

    Goal:
        Keep the original BGCN model logic as much as possible.
        Only adapt the outer interface to BundleRecFramework.

    Required interface:
        calculate_loss(batch)
        predict(users)

    Current status:
        This file is an adapter placeholder.
        The next step is to port/import official BGCN implementation from:
            third_party/BGCN
    """

    def __init__(self, data, config):
        super().__init__(data, config)

        self.official_root = config.get(
            "official_root",
            "third_party/BGCN"
        )

        if not os.path.exists(self.official_root):
            raise FileNotFoundError(
                f"Official BGCN repo not found at {self.official_root}. "
                f"Run: bash scripts/official/setup_official_repos.sh"
            )

        # TODO:
        # 1. Inspect official BGCN files.
        # 2. Import official model class/helper functions.
        # 3. Build official graph/data objects from self.data.
        # 4. Initialize official BGCN model.
        #
        # Example structure after port:
        # self.official_model = OfficialBGCN(...)
        #
        # Important:
        # Do not rewrite BGCN-style logic manually here.
        # Keep official propagation/loss/scoring logic.

        raise NotImplementedError(
            "BGCN official adapter is not implemented yet. "
            "First inspect third_party/BGCN and port official model logic here."
        )

    def calculate_loss(self, batch):
        # TODO:
        # Convert framework batch to official BGCN batch format.
        # Then call official model loss.
        raise NotImplementedError

    def predict(self, users):
        # TODO:
        # Call official BGCN scoring/prediction function.
        # Must return scores with shape [len(users), n_bundles].
        raise NotImplementedError
