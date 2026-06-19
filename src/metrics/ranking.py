import math
import torch


def recall_at_k(ranked_list, ground_truth, k):
    ranked_list = ranked_list[:k]
    hit = len(set(ranked_list) & set(ground_truth))

    if len(ground_truth) == 0:
        return 0.0

    return hit / len(ground_truth)


def ndcg_at_k(ranked_list, ground_truth, k):
    ranked_list = ranked_list[:k]
    ground_truth = set(ground_truth)

    dcg = 0.0

    for idx, item in enumerate(ranked_list):
        if item in ground_truth:
            dcg += 1.0 / math.log2(idx + 2)

    ideal_hit = min(len(ground_truth), k)

    idcg = 0.0
    for idx in range(ideal_hit):
        idcg += 1.0 / math.log2(idx + 2)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def evaluate_topk(model, data, device, topk_list, split="test"):
    model.eval()

    train_dict = data.get_train_user_pos_items()
    eval_dict = data.get_eval_dict(split)

    result = {}

    for k in topk_list:
        result[f"Recall@{k}"] = 0.0
        result[f"NDCG@{k}"] = 0.0

    user_count = 0

    with torch.no_grad():
        for user, gt_bundles in eval_dict.items():
            if len(gt_bundles) == 0:
                continue

            user_tensor = torch.tensor([user], dtype=torch.long).to(device)
            scores = model.full_sort_predict(user_tensor).squeeze(0)

            train_bundles = train_dict.get(user, [])

            if len(train_bundles) > 0:
                train_tensor = torch.tensor(train_bundles, dtype=torch.long).to(device)
                scores[train_tensor] = -1e9

            n_candidates = scores.shape[0]
            max_k = min(max(topk_list), n_candidates)

            _, ranked_items = torch.topk(scores, k=max_k)
            ranked_items = ranked_items.cpu().numpy().tolist()

            for k in topk_list:
                safe_k = min(k, n_candidates)

                result[f"Recall@{k}"] += recall_at_k(
                    ranked_items,
                    gt_bundles,
                    safe_k
                )

                result[f"NDCG@{k}"] += ndcg_at_k(
                    ranked_items,
                    gt_bundles,
                    safe_k
                )

            user_count += 1

    if user_count == 0:
        return result

    for key in result:
        result[key] /= user_count

    return result
