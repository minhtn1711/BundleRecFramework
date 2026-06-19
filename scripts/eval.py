import argparse
import os
import sys
import torch

sys.path.append(os.path.abspath("."))

from src.config import load_config
from src.utils import set_seed, get_device
from src.data.registry import build_dataset
from src.models.registry import build_model
from src.metrics.ranking import evaluate_topk


def parse_topk(topk_str):
    if topk_str is None:
        return None

    return [int(x) for x in topk_str.split(",")]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")

    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--topk", type=str, default=None)

    return parser.parse_args()


def build_config(args):
    if args.config is not None:
        config = load_config(args.config)
    else:
        if args.model is None or args.dataset is None:
            raise ValueError(
                "Please provide --config or both --model and --dataset."
            )

        config_path = f"configs/{args.model}_{args.dataset}.json"

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        config = load_config(config_path)

    if args.model is not None:
        config["model"] = args.model

    if args.dataset is not None:
        config["dataset"] = args.dataset
        config["data_dir"] = f"data/processed/{args.dataset}"

    if args.device is not None:
        config["device"] = args.device

    parsed_topk = parse_topk(args.topk)
    if parsed_topk is not None:
        config["topk"] = parsed_topk

    return config


def print_config(config):
    print("[Config]")
    for key, value in config.items():
        print(f"  {key}: {value}")


def main():
    args = parse_args()
    config = build_config(args)

    set_seed(config["seed"])
    device = get_device(config["device"])

    print_config(config)
    print(f"[Device] {device}")

    data = build_dataset(config)
    model = build_model(config, data)

    checkpoint = torch.load(args.checkpoint, map_location=device)

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain model_state_dict. "
            "Please check checkpoint format."
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"[Model] {model.__class__.__name__}")
    print(f"[Checkpoint] loaded from {args.checkpoint}")

    result = evaluate_topk(
        model=model,
        data=data,
        device=device,
        topk_list=config["topk"],
        split=args.split
    )

    print(f"[Eval Result] split = {args.split}")
    print(result)


if __name__ == "__main__":
    main()
