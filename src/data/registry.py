from src.data.bundle_dataset import GeneralBundleDataset


DATASET_REGISTRY = {
    "youshu": GeneralBundleDataset,
    "ifashion": GeneralBundleDataset,
    "netease": GeneralBundleDataset
}


def build_dataset(config):
    dataset_name = config["dataset"].lower()

    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available datasets: {list(DATASET_REGISTRY.keys())}"
        )

    dataset_class = DATASET_REGISTRY[dataset_name]
    return dataset_class(config)
