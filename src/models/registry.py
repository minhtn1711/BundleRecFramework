from src.models.bgcn import BGCN
from src.models.crosscbr import CrossCBR


MODEL_REGISTRY = {
    "bgcn": BGCN,
    "crosscbr": CrossCBR
}


def build_model(config, data):
    model_name = config["model"].lower()

    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {list(MODEL_REGISTRY.keys())}"
        )

    model_class = MODEL_REGISTRY[model_name]
    return model_class(data, config)
