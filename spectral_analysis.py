import numpy as np


def compute_singular_value_spectrum(features):
    _, S, _ = np.linalg.svd(features, full_matrices=False)
    return S


def compute_effective_rank(singular_values):
    s = singular_values
    p = s / (s.sum() + 1e-8)

    entropy = -np.sum(p * np.log(p + 1e-8))
    effective_rank = np.exp(entropy)

    return effective_rank


def compare_spectral_statistics(scene_spectrum, text_spectrum):
    scene_rank = compute_effective_rank(scene_spectrum)
    text_rank = compute_effective_rank(text_spectrum)

    print("Effective rank (vision):", scene_rank)
    print("Effective rank (text):", text_rank)

    print("\nTop singular values (vision):", scene_spectrum[:10])
    print("Top singular values (text):", text_spectrum[:10])

def project_to_top_k(features, k):
    U, S, Vt = np.linalg.svd(features, full_matrices=False)
    V_k = Vt[:k].T
    projected = features @ V_k
    return projected, S, V_k

def choose_k_from_spectrum(S, threshold=0.9):
    energy = np.cumsum(S**2) / np.sum(S**2)
    k = np.searchsorted(energy, threshold) + 1
    return k