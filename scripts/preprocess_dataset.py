import argparse
import os
from collections import defaultdict


def read_any_interaction_file(path):
    interactions = defaultdict(list)

    if not os.path.exists(path):
        print(f"[Warning] Missing file: {path}")
        return {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line == "":
                continue

            parts = line.replace(",", " ").replace("\t", " ").split()

            if len(parts) <= 1:
                continue

            left_id = int(parts[0])
            right_ids = [int(x) for x in parts[1:]]

            interactions[left_id].extend(right_ids)

    final_interactions = {}

    for left_id, right_ids in interactions.items():
        final_interactions[left_id] = sorted(set(right_ids))

    return final_interactions


def find_file(src_dir, candidates):
    for name in candidates:
        path = os.path.join(src_dir, name)

        if os.path.exists(path):
            return path

    return None


def save_grouped_file(interactions, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for left_id in sorted(interactions.keys()):
            right_ids = interactions[left_id]

            if len(right_ids) == 0:
                continue

            line = str(left_id) + " " + " ".join(map(str, right_ids))
            f.write(line + "\n")

    total_pairs = sum(len(v) for v in interactions.values())
    print(f"[Saved] {out_path} | rows={len(interactions)} | pairs={total_pairs}")


def convert(src_dir, dst_dir):
    file_map = {
        "user_bundle_train.txt": [
            "user_bundle_train.txt",
            "train.txt"
        ],
        "user_bundle_val.txt": [
            "user_bundle_val.txt",
            "user_bundle_tune.txt",
            "val.txt",
            "tune.txt"
        ],
        "user_bundle_test.txt": [
            "user_bundle_test.txt",
            "test.txt"
        ],
        "bundle_item.txt": [
            "bundle_item.txt",
            "bundle_items.txt"
        ],
        "user_item.txt": [
            "user_item.txt",
            "user_items.txt"
        ]
    }

    os.makedirs(dst_dir, exist_ok=True)

    for out_name, candidates in file_map.items():
        src_path = find_file(src_dir, candidates)

        if src_path is None:
            print(f"[Warning] Cannot find source for {out_name}")
            continue

        interactions = read_any_interaction_file(src_path)
        out_path = os.path.join(dst_dir, out_name)

        save_grouped_file(interactions, out_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--src_dir", type=str, required=True)
    parser.add_argument("--dst_dir", type=str, required=True)

    args = parser.parse_args()

    convert(args.src_dir, args.dst_dir)


if __name__ == "__main__":
    main()
