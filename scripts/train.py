import argparse
import os
import sys

sys.path.append(os.path.abspath("."))

from src.config import load_config
from src.utils import set_seed, get_device
from src.data.registry import build_dataset
from src.models.registry import build_model
from src.trainers.trainer import RecTrainer


def parse_topk(topk_str):
    if topk_str is None:
        return None

    return [int(x) for x in topk_str.split(",")]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default=None)

    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--embedding_dim", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--topk", type=str, default=None)
    parser.add_argument("--main_metric", type=str, default=None)


    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="BundleRecFramework")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online")
    parser.add_argument("--log_step_interval", type=int, default=20)

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

    if args.epochs is not None:
        config["epochs"] = args.epochs

    if args.batch_size is not None:
        config["batch_size"] = args.batch_size

    if args.lr is not None:
        config["lr"] = args.lr

    if args.embedding_dim is not None:
        config["embedding_dim"] = args.embedding_dim

    if args.device is not None:
        config["device"] = args.device

    parsed_topk = parse_topk(args.topk)
    if parsed_topk is not None:
        config["topk"] = parsed_topk

    if args.main_metric is not None:
        config["main_metric"] = args.main_metric


    config["use_wandb"] = args.use_wandb
    config["wandb_project"] = args.wandb_project
    config["wandb_entity"] = args.wandb_entity
    config["wandb_run_name"] = args.wandb_run_name
    config["wandb_mode"] = args.wandb_mode
    config["log_step_interval"] = args.log_step_interval

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

    print(f"[Model] {model.__class__.__name__}")

    trainer = RecTrainer(
        model=model,
        data=data,
        config=config,
        device=device
    )

    trainer.train()


if __name__ == "__main__":
    main()
