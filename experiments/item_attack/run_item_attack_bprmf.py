import os
import random
import argparse
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim


class BPRMF(nn.Module):
    def __init__(self, num_users, num_items, emb_dim=64):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)

        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

    def forward(self, users, pos_items, neg_items):
        u = self.user_emb(users)
        pos = self.item_emb(pos_items)
        neg = self.item_emb(neg_items)

        pos_score = torch.sum(u * pos, dim=1)
        neg_score = torch.sum(u * neg, dim=1)

        loss = -torch.mean(torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8))
        return loss

    def predict_all(self, users):
        u = self.user_emb(users)
        scores = torch.matmul(u, self.item_emb.weight.t())
        return scores


def load_movielens(path, min_rating=4):
    user_items = defaultdict(set)

    with open(path, "r") as f:
        for line in f:
            user, item, rating, _ = line.strip().split("\t")
            user = int(user) - 1
            item = int(item) - 1
            rating = int(rating)

            if rating >= min_rating:
                user_items[user].add(item)

    return user_items


def remap_data(user_items):
    users = sorted(user_items.keys())
    all_items = sorted({i for items in user_items.values() for i in items})

    user_map = {u: idx for idx, u in enumerate(users)}
    item_map = {i: idx for idx, i in enumerate(all_items)}

    remapped = defaultdict(set)

    for u, items in user_items.items():
        new_u = user_map[u]
        for i in items:
            remapped[new_u].add(item_map[i])

    return remapped, len(user_map), len(item_map)


def train_test_split(user_items):
    train = defaultdict(set)
    test = defaultdict(set)

    for u, items in user_items.items():
        items = list(items)
        if len(items) < 2:
            train[u] = set(items)
            continue

        test_item = random.choice(items)
        test[u].add(test_item)

        for i in items:
            if i != test_item:
                train[u].add(i)

    return train, test


def sample_batch(train_user_items, num_items, batch_size):
    users = list(train_user_items.keys())

    batch_users = []
    batch_pos = []
    batch_neg = []

    while len(batch_users) < batch_size:
        u = random.choice(users)
        if len(train_user_items[u]) == 0:
            continue

        pos = random.choice(list(train_user_items[u]))
        neg = random.randint(0, num_items - 1)

        while neg in train_user_items[u]:
            neg = random.randint(0, num_items - 1)

        batch_users.append(u)
        batch_pos.append(pos)
        batch_neg.append(neg)

    return (
        torch.LongTensor(batch_users),
        torch.LongTensor(batch_pos),
        torch.LongTensor(batch_neg),
    )


def train_model(train_user_items, num_users, num_items, device, epochs=50, batch_size=1024, lr=1e-3):
    model = BPRMF(num_users, num_items).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    num_batches = max(1, sum(len(v) for v in train_user_items.values()) // batch_size)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for _ in range(num_batches):
            users, pos, neg = sample_batch(train_user_items, num_items, batch_size)
            users = users.to(device)
            pos = pos.to(device)
            neg = neg.to(device)

            loss = model(users, pos, neg)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Loss: {total_loss / num_batches:.4f}")

    return model


def recall_ndcg_at_k(model, train_user_items, test_user_items, num_real_users, num_items, device, k=20):
    model.eval()

    recalls = []
    ndcgs = []

    with torch.no_grad():
        for u in range(num_real_users):
            if u not in test_user_items or len(test_user_items[u]) == 0:
                continue

            user_tensor = torch.LongTensor([u]).to(device)
            scores = model.predict_all(user_tensor).squeeze(0).cpu().numpy()

            for train_item in train_user_items[u]:
                scores[train_item] = -1e9

            topk = np.argsort(-scores)[:k]
            gt_items = test_user_items[u]

            hit = 0
            ndcg = 0.0

            for rank, item in enumerate(topk):
                if item in gt_items:
                    hit = 1
                    ndcg = 1.0 / np.log2(rank + 2)
                    break

            recalls.append(hit)
            ndcgs.append(ndcg)

    return float(np.mean(recalls)), float(np.mean(ndcgs))


def target_attack_metrics(model, train_user_items, num_real_users, target_item, num_items, device, k=20):
    model.eval()

    hit_count = 0
    ranks = []

    with torch.no_grad():
        for u in range(num_real_users):
            user_tensor = torch.LongTensor([u]).to(device)
            scores = model.predict_all(user_tensor).squeeze(0).cpu().numpy()

            for train_item in train_user_items[u]:
                scores[train_item] = -1e9

            sorted_items = np.argsort(-scores)
            rank = int(np.where(sorted_items == target_item)[0][0]) + 1
            ranks.append(rank)

            if rank <= k:
                hit_count += 1

    target_hit = hit_count / num_real_users
    avg_rank = float(np.mean(ranks))

    return target_hit, avg_rank


def random_attack(train_user_items, num_real_users, num_items, target_item, attack_ratio=0.05, filler_size=20):
    attacked = defaultdict(set)

    for u, items in train_user_items.items():
        attacked[u] = set(items)

    num_fake_users = max(1, int(num_real_users * attack_ratio))

    all_items = list(range(num_items))

    for fake_idx in range(num_fake_users):
        fake_user = num_real_users + fake_idx

        filler_items = random.sample(all_items, filler_size)
        fake_profile = set(filler_items)
        fake_profile.add(target_item)

        attacked[fake_user] = fake_profile

    total_users_after_attack = num_real_users + num_fake_users

    print(f"Injected fake users: {num_fake_users}")
    print(f"Each fake user has target item + {filler_size} filler items")

    return attacked, total_users_after_attack



def save_interactions(user_items, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        for u in sorted(user_items.keys()):
            for i in sorted(user_items[u]):
                f.write(f"{u}\t{i}\n")


def save_fake_profiles(attacked_user_items, num_real_users, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        for u in sorted(attacked_user_items.keys()):
            if u >= num_real_users:
                items = sorted(attacked_user_items[u])
                items_str = " ".join(map(str, items))
                f.write(f"{u}\t{items_str}\n")


def save_result_summary(save_path, args, target_item,
                        clean_recall, clean_ndcg, clean_target_hit, clean_avg_rank,
                        attacked_recall, attacked_ndcg, attacked_target_hit, attacked_avg_rank):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        f.write("Item Promotion Attack Result\n")
        f.write("=" * 50 + "\n")
        f.write(f"Target item: {target_item}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"TopK: {args.k}\n")
        f.write(f"Attack ratio: {args.attack_ratio}\n")
        f.write(f"Filler size: {args.filler_size}\n")
        if hasattr(args, "attack_method"):
            f.write(f"Attack method: {args.attack_method}\n")
        else:
            f.write("Attack method: random\n")

        f.write("\nClean Result\n")
        f.write(f"Recall@{args.k}: {clean_recall:.4f}\n")
        f.write(f"NDCG@{args.k}: {clean_ndcg:.4f}\n")
        f.write(f"TargetHit@{args.k}: {clean_target_hit:.4f}\n")
        f.write(f"AvgTargetRank: {clean_avg_rank:.2f}\n")

        f.write("\nAttacked Result\n")
        f.write(f"Recall@{args.k}: {attacked_recall:.4f}\n")
        f.write(f"NDCG@{args.k}: {attacked_ndcg:.4f}\n")
        f.write(f"TargetHit@{args.k}: {attacked_target_hit:.4f}\n")
        f.write(f"AvgTargetRank: {attacked_avg_rank:.2f}\n")

        f.write("\nChange\n")
        f.write(f"Recall change: {attacked_recall - clean_recall:.4f}\n")
        f.write(f"NDCG change: {attacked_ndcg - clean_ndcg:.4f}\n")
        f.write(f"TargetHit change: {attacked_target_hit - clean_target_hit:.4f}\n")
        f.write(f"AvgTargetRank change: {attacked_avg_rank - clean_avg_rank:.2f}\n")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default="ml-100k/u.data")
    parser.add_argument("--target_item", type=int, default=-1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--attack_ratio", type=float, default=0.05)
    parser.add_argument("--filler_size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output_dir", type=str, default="outputs")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)
    print("Loading MovieLens 100K...")

    raw_user_items = load_movielens(args.data_path)
    user_items, num_users, num_items = remap_data(raw_user_items)
    train_data, test_data = train_test_split(user_items)

    save_interactions(train_data, os.path.join(args.output_dir, "clean_train.txt"))
    save_interactions(test_data, os.path.join(args.output_dir, "test.txt"))

    num_real_users = num_users

    print(f"Num users: {num_users}")
    print(f"Num items: {num_items}")
    print(f"Num train interactions: {sum(len(v) for v in train_data.values())}")

    if args.target_item == -1:
        item_pop = defaultdict(int)
        for items in train_data.values():
            for i in items:
                item_pop[i] += 1

        sorted_items = sorted(item_pop.items(), key=lambda x: x[1])
        target_item = sorted_items[len(sorted_items) // 5][0]
    else:
        target_item = args.target_item

    print("=" * 60)
    print(f"Target item: {target_item}")
    print("=" * 60)

    print("\n[1] Train clean BPR-MF")
    clean_model = train_model(
        train_data,
        num_users,
        num_items,
        device,
        epochs=args.epochs,
    )

    clean_recall, clean_ndcg = recall_ndcg_at_k(
        clean_model,
        train_data,
        test_data,
        num_real_users,
        num_items,
        device,
        k=args.k,
    )

    clean_target_hit, clean_avg_rank = target_attack_metrics(
        clean_model,
        train_data,
        num_real_users,
        target_item,
        num_items,
        device,
        k=args.k,
    )

    print("\nClean Result")
    print(f"Recall@{args.k}: {clean_recall:.4f}")
    print(f"NDCG@{args.k}: {clean_ndcg:.4f}")
    print(f"TargetHit@{args.k}: {clean_target_hit:.4f}")
    print(f"AvgTargetRank: {clean_avg_rank:.2f}")

    print("\n[2] Inject fake users")
    attacked_train_data, attacked_num_users = random_attack(
        train_data,
        num_real_users,
        num_items,
        target_item,
        attack_ratio=args.attack_ratio,
        filler_size=args.filler_size,
    )

    save_interactions(attacked_train_data, os.path.join(args.output_dir, "attacked_train.txt"))
    save_fake_profiles(attacked_train_data, num_real_users, os.path.join(args.output_dir, "fake_profiles.txt"))

    print(f"Saved clean train to: {os.path.join(args.output_dir, 'clean_train.txt')}")
    print(f"Saved attacked train to: {os.path.join(args.output_dir, 'attacked_train.txt')}")
    print(f"Saved fake profiles to: {os.path.join(args.output_dir, 'fake_profiles.txt')}")

    print("\n[3] Train attacked BPR-MF")
    attacked_model = train_model(
        attacked_train_data,
        attacked_num_users,
        num_items,
        device,
        epochs=args.epochs,
    )

    attacked_recall, attacked_ndcg = recall_ndcg_at_k(
        attacked_model,
        attacked_train_data,
        test_data,
        num_real_users,
        num_items,
        device,
        k=args.k,
    )

    attacked_target_hit, attacked_avg_rank = target_attack_metrics(
        attacked_model,
        attacked_train_data,
        num_real_users,
        target_item,
        num_items,
        device,
        k=args.k,
    )

    print("\nAttacked Result")
    print(f"Recall@{args.k}: {attacked_recall:.4f}")
    print(f"NDCG@{args.k}: {attacked_ndcg:.4f}")
    print(f"TargetHit@{args.k}: {attacked_target_hit:.4f}")
    print(f"AvgTargetRank: {attacked_avg_rank:.2f}")

    print("\n" + "=" * 60)
    print("Final Comparison")
    print("=" * 60)

    print(f"{'Metric':<20} {'Clean':<15} {'Attacked':<15} {'Change':<15}")
    print("-" * 60)
    print(f"{'Recall@K':<20} {clean_recall:<15.4f} {attacked_recall:<15.4f} {attacked_recall - clean_recall:<15.4f}")
    print(f"{'NDCG@K':<20} {clean_ndcg:<15.4f} {attacked_ndcg:<15.4f} {attacked_ndcg - clean_ndcg:<15.4f}")
    print(f"{'TargetHit@K':<20} {clean_target_hit:<15.4f} {attacked_target_hit:<15.4f} {attacked_target_hit - clean_target_hit:<15.4f}")
    print(f"{'AvgTargetRank':<20} {clean_avg_rank:<15.2f} {attacked_avg_rank:<15.2f} {attacked_avg_rank - clean_avg_rank:<15.2f}")

    save_result_summary(
        os.path.join(args.output_dir, "result_summary.txt"),
        args,
        target_item,
        clean_recall,
        clean_ndcg,
        clean_target_hit,
        clean_avg_rank,
        attacked_recall,
        attacked_ndcg,
        attacked_target_hit,
        attacked_avg_rank,
    )
    print(f"Saved result summary to: {os.path.join(args.output_dir, 'result_summary.txt')}")


if __name__ == "__main__":
    main()
