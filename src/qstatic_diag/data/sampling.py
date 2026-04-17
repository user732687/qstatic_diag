from __future__ import annotations

import logging
from typing import List, Literal, Optional, Tuple

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from torch import Tensor
from torch.utils.data import Dataset, Subset, TensorDataset

from qstatic_diag.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Unlabeled diversity sampling
# ---------------------------------------------------------------------------


def _flatten_for_clustering(tensors: Tensor) -> np.ndarray:
    """Flatten N-D tensors to 2-D (N, D) float32 array for sklearn."""
    arr = tensors.detach().cpu().float().numpy()
    return arr.reshape(len(arr), -1)


def random_sample(
    proxy_data: Tensor,
    m: int,
    seed: int = 42,
) -> Tuple[Tensor, np.ndarray]:
    """Uniform random sampling without replacement."""
    n = len(proxy_data)
    if m >= n:
        log.warning("m=%d >= N=%d; returning full proxy set.", m, n)
        return proxy_data, np.arange(n)

    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(n, size=m, replace=False))
    return proxy_data[indices], indices


def diversity_sample(
    proxy_data: Tensor,
    m: int,
    seed: int = 42,
) -> Tuple[Tensor, np.ndarray]:
    """
    Select m diverse representatives from an unlabeled dataset via k-means.
    """
    if m >= len(proxy_data):
        log.warning(
            "m=%d >= N=%d; returning full dataset as diagnostic set.", m, len(proxy_data)
        )
        return proxy_data, np.arange(len(proxy_data))

    X = _flatten_for_clustering(proxy_data)

    log.info("Running MiniBatchKMeans (k=%d, N=%d)…", m, len(X))
    kmeans = MiniBatchKMeans(
        n_clusters=m,
        random_state=seed,
        n_init=3,
        max_iter=300,
        batch_size=min(4096, len(X)),
    )
    labels = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_  # (m, D)

    # For each cluster, pick the sample closest to its centroid.
    indices: List[int] = []
    for c in range(m):
        mask = labels == c
        if not mask.any():
            # Empty cluster — fall back to closest point overall
            dists = np.linalg.norm(X - centroids[c], axis=1)
            indices.append(int(np.argmin(dists)))
        else:
            cluster_idx = np.where(mask)[0]
            dists = np.linalg.norm(X[cluster_idx] - centroids[c], axis=1)
            indices.append(int(cluster_idx[np.argmin(dists)]))

    idx_arr = np.array(indices)
    log.info("Selected %d diagnostic samples.", len(idx_arr))
    return proxy_data[idx_arr], idx_arr


def kmeanspp_sample(
    proxy_data: Tensor,
    m: int,
    seed: int = 42,
) -> Tuple[Tensor, np.ndarray]:
    """
    Select samples with k-means++ style D^2 seeding, no Lloyd iterations.
    """
    n = len(proxy_data)
    if m >= n:
        log.warning("m=%d >= N=%d; returning full proxy set.", m, n)
        return proxy_data, np.arange(n)

    X = _flatten_for_clustering(proxy_data)
    rng = np.random.default_rng(seed)

    first = int(rng.integers(0, n))
    selected = [first]
    min_sq_dists = np.sum((X - X[first]) ** 2, axis=1)

    for _ in range(1, m):
        total = float(min_sq_dists.sum())
        if total <= 0.0:
            remaining = np.setdiff1d(np.arange(n), np.array(selected), assume_unique=False)
            next_idx = int(rng.choice(remaining))
        else:
            probs = min_sq_dists / total
            next_idx = int(rng.choice(n, p=probs))
            while next_idx in selected:
                next_idx = int(rng.choice(n, p=probs))

        selected.append(next_idx)
        new_sq_dists = np.sum((X - X[next_idx]) ** 2, axis=1)
        min_sq_dists = np.minimum(min_sq_dists, new_sq_dists)
        min_sq_dists[selected] = 0.0

    idx_arr = np.array(sorted(selected))
    return proxy_data[idx_arr], idx_arr


def farthest_point_sample(
    proxy_data: Tensor,
    m: int,
    seed: int = 42,
) -> Tuple[Tensor, np.ndarray]:
    """
    Greedy farthest-point traversal (k-center approximation).
    """
    n = len(proxy_data)
    if m >= n:
        log.warning("m=%d >= N=%d; returning full proxy set.", m, n)
        return proxy_data, np.arange(n)

    X = _flatten_for_clustering(proxy_data)
    rng = np.random.default_rng(seed)

    first = int(rng.integers(0, n))
    selected = [first]
    min_sq_dists = np.sum((X - X[first]) ** 2, axis=1)
    min_sq_dists[first] = 0.0

    for _ in range(1, m):
        next_idx = int(np.argmax(min_sq_dists))
        selected.append(next_idx)
        new_sq_dists = np.sum((X - X[next_idx]) ** 2, axis=1)
        min_sq_dists = np.minimum(min_sq_dists, new_sq_dists)
        min_sq_dists[selected] = 0.0

    idx_arr = np.array(sorted(selected))
    return proxy_data[idx_arr], idx_arr


def sample_diagnostic_set(
    proxy_data: Tensor,
    m: int,
    seed: int = 42,
    method: Literal["kmeans", "kmeans++", "farthest", "random"] = "kmeans",
) -> Tuple[Tensor, np.ndarray]:
    """
    Unified dispatcher for unlabeled diagnostic sampling algorithms.
    """
    if method == "kmeans":
        return diversity_sample(proxy_data, m=m, seed=seed)
    if method == "kmeans++":
        return kmeanspp_sample(proxy_data, m=m, seed=seed)
    if method == "farthest":
        return farthest_point_sample(proxy_data, m=m, seed=seed)
    if method == "random":
        return random_sample(proxy_data, m=m, seed=seed)
    raise ValueError(f"Unknown sampling method: {method}")


# ---------------------------------------------------------------------------
# Stratified (labeled) sampling
# ---------------------------------------------------------------------------


def stratified_diversity_sample(
    proxy_data: Tensor,
    labels: Tensor,
    m: int,
    seed: int = 42,
    method: Literal["kmeans", "kmeans++", "farthest", "random"] = "kmeans",
) -> Tuple[Tensor, Tensor, np.ndarray]:
    """
    Class-stratified version of diversity_sample.
    """
    classes = labels.unique().tolist()
    K = len(classes)
    class_counts = {int(c): int((labels == c).sum()) for c in classes}
    total = len(labels)

    all_data: List[Tensor] = []
    all_labels: List[Tensor] = []
    all_indices: List[int] = []

    for c in classes:
        c_int = int(c)
        pi_k = class_counts[c_int] / total
        m_k = max(1, int(np.floor(m * pi_k)))

        mask = (labels == c).nonzero(as_tuple=True)[0]
        class_data = proxy_data[mask]

        sel_data, rel_idx = sample_diagnostic_set(
            class_data,
            m_k,
            seed=seed + c_int,
            method=method,
        )
        abs_idx = mask[rel_idx].cpu().numpy()

        all_data.append(sel_data)
        all_labels.append(torch.full((len(sel_data),), c_int, dtype=labels.dtype))
        all_indices.extend(abs_idx.tolist())

    selected_data = torch.cat(all_data, dim=0)
    selected_labels = torch.cat(all_labels, dim=0)
    indices = np.array(all_indices)

    log.info(
        "Stratified sampling: %d classes → %d diagnostic samples.",
        K,
        len(indices),
    )
    return selected_data, selected_labels, indices


# ---------------------------------------------------------------------------
# Dataset-aware wrappers
# ---------------------------------------------------------------------------


def build_diagnostic_loader(
    data: Tensor,
    labels: Optional[Tensor] = None,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
) -> torch.utils.data.DataLoader:
    """Wrap diagnostic tensors in a DataLoader for batched forward passes."""
    data_d = data.to(device) if device else data
    if labels is not None:
        labels_d = labels.to(device) if device else labels
        ds: Dataset = TensorDataset(data_d, labels_d)
    else:
        ds = TensorDataset(data_d)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)
