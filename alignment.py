import numpy as np


def _l2_normalize(A, eps=1e-8):
    return A / (np.linalg.norm(A, axis=1, keepdims=True) + eps)


def _pairs_to_arrays(pairs):
    X = np.array([p[0] for p in pairs], dtype=np.float32)
    Y = np.array([p[1] for p in pairs], dtype=np.float32)

    if X.ndim != 2:
        raise ValueError(f"Expected X to be 2D, got shape {X.shape}")

    if Y.ndim != 2:
        raise ValueError(f"Expected Y to be 2D, got shape {Y.shape}")

    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"Mismatch: X has {X.shape[0]} samples, Y has {Y.shape[0]} samples")

    return X, Y


def split_alignment_train_test(pairs, train_ratio=0.8):
    pairs = list(pairs)
    n = len(pairs)

    if n < 2:
        raise ValueError("Need at least 2 pairs.")

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    rng = np.random.default_rng(42)
    indices = np.arange(n)
    rng.shuffle(indices)

    train_size = int(n * train_ratio)

    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_pairs = [pairs[i] for i in train_indices]
    test_pairs = [pairs[i] for i in test_indices]

    return train_pairs, test_pairs


def fit_manifold_alignment(train_pairs):
    X, Y = _pairs_to_arrays(train_pairs)

    alpha = 1.0
    d_x = X.shape[1]

    A = X.T @ X + alpha * np.eye(d_x, dtype=np.float32)
    B = X.T @ Y

    W = np.linalg.solve(A, B)

    alignment_model = {
        "type": "ridge_alignment",
        "W": W.astype(np.float32),
        "alpha": alpha,
    }

    return alignment_model


def transform_3d_features(features, alignment_model):
    X = np.asarray(features, dtype=np.float32)

    single_sample = False

    if X.ndim == 1:
        X = X[None, :]
        single_sample = True

    W = alignment_model["W"]

    X_aligned = X @ W
    X_aligned = _l2_normalize(X_aligned)

    if single_sample:
        return X_aligned[0]

    return X_aligned


def evaluate_alignment_quality(test_pairs, alignment_model):
    X_test, Y_test = _pairs_to_arrays(test_pairs)

    X_aligned = transform_3d_features(X_test, alignment_model)

    Y_test = _l2_normalize(Y_test)

    similarity = X_aligned @ Y_test.T

    ranks = []

    for i in range(similarity.shape[0]):
        sorted_indices = np.argsort(-similarity[i])
        rank = np.where(sorted_indices == i)[0][0] + 1
        ranks.append(rank)

    ranks = np.asarray(ranks)

    metrics = {
        "top_1": float(np.mean(ranks <= 1)),
        "top_5": float(np.mean(ranks <= 5)),
        "top_10": float(np.mean(ranks <= 10)),
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
        "mean_correct_cosine": float(np.mean(np.diag(similarity))),
        "ranks": ranks,
        "similarity_matrix": similarity,
    }

    return metrics