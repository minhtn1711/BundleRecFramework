import os
import json
import torch
from tqdm import tqdm

from src.metrics.ranking import evaluate_topk


class RecTrainer:
    def __init__(self, model, data, config, device):
        self.model = model
        self.data = data
        self.config = config
        self.device = device

        self.model.to(self.device)
        self.train_loader = data.get_train_loader()

        if hasattr(self.model, "set_train_context"):
            self.model.set_train_context(num_batches=len(self.train_loader))

        self.optimizer = self.build_optimizer()

        self.save_dir = config.get("save_dir", "checkpoints")
        os.makedirs(self.save_dir, exist_ok=True)

        self.main_metric = config.get(
            "main_metric",
            f"Recall@{max(config['topk'])}"
        )
        self.eval_interval = config.get("eval_interval", 1)

        self.best_score = -1.0
        self.global_step = 0

        self.use_wandb = bool(config.get("use_wandb", False))
        self.wandb_run = None

        if self.use_wandb:
            self.init_wandb()

    def init_wandb(self):
        try:
            import wandb
        except ImportError:
            raise ImportError("wandb is not installed. Run: pip install wandb")

        project = self.config.get("wandb_project", "BundleRecFramework")
        entity = self.config.get("wandb_entity", None)
        mode = self.config.get("wandb_mode", "online")

        run_name = self.config.get("wandb_run_name", None)
        if run_name is None:
            run_name = f"{self.config.get('model', 'model')}-{self.config.get('dataset', 'dataset')}"

        self.wandb_run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            config=self.config,
            mode=mode
        )

        wandb.define_metric("epoch")
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("val/*", step_metric="epoch")
        wandb.define_metric("test/*", step_metric="epoch")

        print(f"[W&B] enabled: project={project}, run={run_name}, mode={mode}")

    def wandb_log(self, log_dict, step=None):
        if not self.use_wandb or self.wandb_run is None:
            return

        import wandb
        wandb.log(log_dict, step=step)

    def build_optimizer(self):
        optimizer_name = self.config.get("optimizer", "adam").lower()
        lr = self.config["lr"]

        optimizer_weight_decay = self.config.get(
            "optimizer_weight_decay",
            self.config.get("l2_reg", 0.0)
        )

        if optimizer_name == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=optimizer_weight_decay
            )

        if optimizer_name == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                weight_decay=optimizer_weight_decay
            )

        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def move_batch_to_device(self, batch):
        if isinstance(batch, dict):
            return {
                key: value.to(self.device) if hasattr(value, "to") else value
                for key, value in batch.items()
            }

        if isinstance(batch, (list, tuple)):
            return [
                value.to(self.device) if hasattr(value, "to") else value
                for value in batch
            ]

        if hasattr(batch, "to"):
            return batch.to(self.device)

        return batch

    def train_one_epoch(self, epoch):
        self.model.train()

        if hasattr(self.model, "on_epoch_start"):
            self.model.on_epoch_start(epoch)

        total_loss = 0.0
        log_step_interval = self.config.get("log_step_interval", 20)

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}",
            ncols=100
        )

        for batch_idx, batch in enumerate(progress_bar):
            self.global_step += 1

            batch = self.move_batch_to_device(batch)

            if hasattr(self.model, "on_batch_start"):
                self.model.on_batch_start(epoch, batch_idx)

            loss = self.model.calculate_loss(batch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if hasattr(self.model, "on_batch_end"):
                self.model.on_batch_end(epoch, batch_idx)

            loss_value = loss.item()
            total_loss += loss_value

            progress_bar.set_postfix(loss=f"{loss_value:.4f}")

            if self.use_wandb and self.global_step % log_step_interval == 0:
                self.wandb_log(
                    {
                        "train/batch_loss": loss_value,
                        "train/lr": self.get_lr(),
                        "epoch": epoch,
                        "batch_idx": batch_idx
                    },
                    step=self.global_step
                )

        if hasattr(self.model, "on_epoch_end"):
            self.model.on_epoch_end(epoch)

        avg_loss = total_loss / max(len(self.train_loader), 1)

        self.wandb_log(
            {
                "train/epoch_loss": avg_loss,
                "train/lr": self.get_lr(),
                "epoch": epoch
            },
            step=self.global_step
        )

        return avg_loss

    def evaluate(self, split="test"):
        self.model.eval()

        if hasattr(self.model, "clear_cache"):
            self.model.clear_cache()

        result = evaluate_topk(
            model=self.model,
            data=self.data,
            device=self.device,
            topk_list=self.config["topk"],
            split=split
        )

        return result

    def log_eval_result(self, result, split, epoch):
        log_dict = {"epoch": epoch}

        for metric_name, metric_value in result.items():
            log_dict[f"{split}/{metric_name}"] = metric_value

        if split == "val":
            log_dict["val/main_metric"] = result.get(self.main_metric, 0.0)

        if split == "test":
            log_dict["test/main_metric"] = result.get(self.main_metric, 0.0)

        self.wandb_log(log_dict, step=self.global_step)

    def _checkpoint_path(self, tag):
        filename = (
            f"{self.config['model']}_"
            f"{self.config['dataset']}_"
            f"{tag}.pt"
        )
        return os.path.join(self.save_dir, filename)

    def save_checkpoint(self, epoch, result, tag):
        path = self._checkpoint_path(tag)

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
                "result": result
            },
            path
        )

        print(f"[Checkpoint] saved {tag}: {path}")

        if self.use_wandb and tag == "best":
            self.wandb_log(
                {
                    "best/epoch": epoch,
                    "best/score": result.get(self.main_metric, 0.0)
                },
                step=self.global_step
            )

    def save_config_snapshot(self):
        path = os.path.join(
            self.save_dir,
            f"{self.config['model']}_{self.config['dataset']}_config.json"
        )

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)

        print(f"[Config] saved snapshot: {path}")

    def finish_wandb(self, test_result=None):
        if not self.use_wandb or self.wandb_run is None:
            return

        if test_result is not None:
            for metric_name, metric_value in test_result.items():
                self.wandb_run.summary[f"final_test/{metric_name}"] = metric_value

            self.wandb_run.summary["best_score"] = self.best_score
            self.wandb_run.summary["main_metric"] = self.main_metric

        self.wandb_run.finish()

    def train(self):
        epochs = self.config["epochs"]

        print(f"[Trainer] main_metric = {self.main_metric}")
        print(f"[Trainer] eval_interval = {self.eval_interval}")

        self.save_config_snapshot()

        last_val_result = {}
        test_result = None

        try:
            for epoch in range(1, epochs + 1):
                train_loss = self.train_one_epoch(epoch)

                print(f"[Epoch {epoch}] train_loss = {train_loss:.6f}")

                if epoch % self.eval_interval == 0:
                    val_result = self.evaluate(split="val")
                    last_val_result = val_result

                    print(f"[Epoch {epoch}] val_result = {val_result}")

                    self.log_eval_result(
                        result=val_result,
                        split="val",
                        epoch=epoch
                    )

                    current_score = val_result.get(self.main_metric, 0.0)

                    if current_score > self.best_score:
                        self.best_score = current_score
                        self.save_checkpoint(epoch, val_result, tag="best")

                self.save_checkpoint(epoch, last_val_result, tag="last")

            test_result = self.evaluate(split="test")

            print("[Final Test Result]")
            print(test_result)

            self.log_eval_result(
                result=test_result,
                split="test",
                epoch=epochs
            )

            return test_result

        finally:
            self.finish_wandb(test_result)
