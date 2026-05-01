import numpy as np


def preprocess_3d_features(features):
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True) + 1e-8
    return (features - mean) / std


def preprocess_text_embeddings(embeddings):
    mean = embeddings.mean(axis=0, keepdims=True)
    std = embeddings.std(axis=0, keepdims=True) + 1e-8
    return (embeddings - mean) / std


def normalize_feature_spaces(scene_features, text_embeddings):
    scene_norm = scene_features / (np.linalg.norm(scene_features, axis=1, keepdims=True) + 1e-8)
    text_norm = text_embeddings / (np.linalg.norm(text_embeddings, axis=1, keepdims=True) + 1e-8)

    return scene_norm, text_norm