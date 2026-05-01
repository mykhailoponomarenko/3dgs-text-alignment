import numpy as np


def load_3dgs_features(path):
    features = np.load(path)

    if features.ndim != 2:
        raise ValueError("3DGS features must be 2D")

    return features


def load_text_embeddings(path):
    embeddings = np.load(path)

    if embeddings.ndim != 2:
        raise ValueError("Text embeddings must be 2D")

    return embeddings


def build_cross_modal_pairs(scene_features, text_embeddings):
    if scene_features.shape[0] != text_embeddings.shape[0]:
        raise ValueError("Mismatch in number of samples")

    return [
        (scene_features[i], text_embeddings[i])
        for i in range(scene_features.shape[0])
    ]