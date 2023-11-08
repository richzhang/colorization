import json
import torch
from tqdm import tqdm
from collections import Counter
from PIL import Image
from skimage import color
from torch import Tensor
from train import mock_trainloader

def get_buckets():
    buckets = Counter()
    dataloader = mock_trainloader(batch_size=32)
    total = 0
    for batch in tqdm(dataloader, total=500):
        if total > 500:
            break
        total += 1
        batch = batch[1].div(10, rounding_mode="floor")
        batch = batch.permute(1, 0, 2, 3)
        batch = batch.reshape(2, -1).t()
        batch = batch.tolist()
        batch = [tuple(x) for x in batch]
        buckets.update(batch)

    top_313 = buckets.most_common(313)
    id2bucket = {}
    for i, (k, _) in enumerate(top_313):
        id2bucket[i] = k
    bucket2id = {str(v): k for k, v in id2bucket.items()}
    
    buckets_data = {
        "id2bucket": id2bucket,
        "bucket2id": bucket2id
    }
    with open("buckets_data.json", "w") as f:
        json.dump(buckets_data, f)

def load_buckets():
    with open("buckets_data.json", "r") as f:
        buckets_data = json.load(f)
    buckets_data["bucket2id"] = {eval(k): v for k, v in buckets_data["bucket2id"].items()}
    return buckets_data

def find_closest_bucket(bucket2id, ab_pair) -> int:
    best_dist = None
    best_bucket = None
    for k in bucket2id.keys():
        dist = torch.dist(ab_pair, torch.tensor(k))
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_bucket = k
    return bucket2id[best_bucket]
       

def label_tensor(tens_ab: Tensor) -> Tensor:
    """
    Convert a tensor of ab values to a tensor of bucket ids.
    """
    buckets_data = load_buckets()
    bucket2id = buckets_data["bucket2id"]
    tens_ab = tens_ab.div(10, rounding_mode="floor")
    tens_ab = tens_ab.permute(1, 0, 2, 3)

    tens_labels = torch.zeros(tens_ab[0].shape, dtype=torch.long)
    for i in range(tens_labels.shape[0]):
        for j in range(tens_labels.shape[1]):
            for k in range(tens_labels.shape[2]):
                tens_labels[i, j, k] = find_closest_bucket(bucket2id, tens_ab[:, i, j, k])
    return tens_labels

def labels_to_ab(labels: Tensor) -> Tensor:
    """
    Convert a tensor of bucket ids to a tensor of ab values.
    """
    buckets_data = load_buckets()
    id2bucket = buckets_data["id2bucket"]
    tens_ab = torch.zeros((labels.shape[0], 2, labels.shape[1], labels.shape[2]))
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            for k in range(labels.shape[2]):
                key = id2bucket[str(labels[i, j, k].item())]
                tens_ab[i, :, j, k] = 10*torch.tensor(key) + 5

    return tens_ab
