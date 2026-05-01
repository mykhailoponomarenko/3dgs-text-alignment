import numpy as np

def retrieve_relevant_3d_features(query_embedding, aligned_scene_features, top_k=5):
    query = query_embedding / np.linalg.norm(query_embedding)
    features = aligned_scene_features / np.linalg.norm(
        aligned_scene_features, axis=1, keepdims=True
    )

    sims = features @ query

    top_indices = np.argsort(-sims)[:top_k]
    top_scores = sims[top_indices]

    return top_indices, top_scores


def build_3d_tokens_for_slm(indices, labels):
    return [labels[i] for i in indices]