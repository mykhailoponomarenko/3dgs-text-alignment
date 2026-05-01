import numpy as np
from sentence_transformers import SentenceTransformer
from build_dataset import build_dataset
from data_loading import (
    load_3dgs_features,
    load_text_embeddings,
    build_cross_modal_pairs
)
from data_preprocessing import (
    preprocess_3d_features,
    preprocess_text_embeddings,
    normalize_feature_spaces
)
from spectral_analysis import (
    compute_singular_value_spectrum,
    compare_spectral_statistics,
    choose_k_from_spectrum,
    project_to_top_k
)
from alignment import (
    split_alignment_train_test,
    fit_manifold_alignment,
    evaluate_alignment_quality,
    transform_3d_features
)
from testing import retrieve_relevant_3d_features, build_3d_tokens_for_slm
from qa_model import generate_answer




def main():
    print("Building dataset...")
    build_dataset()

    print("\nLoading data...")
    X = load_3dgs_features("scene_features.npy")
    Y = load_text_embeddings("text_embeddings.npy")

    try:
        labels = np.load("labels.npy", allow_pickle=True)
        print("Loaded labels:", len(labels))
        print("Sample labels:", labels[:5])
    except:
        print("No labels.npy found (skipping)")

    assert X.shape[0] == Y.shape[0], "Mismatch between vision and text samples!"

    print("X shape:", X.shape)
    print("Y shape:", Y.shape)

    print("\nPreprocessing...")

    X = preprocess_3d_features(X)
    Y = preprocess_text_embeddings(Y)

    X, Y = normalize_feature_spaces(X, Y)

    print("\nSpectral analysis...")

    Sx = compute_singular_value_spectrum(X)
    Sy = compute_singular_value_spectrum(Y)

    compare_spectral_statistics(Sx, Sy)

    print("\nSelecting dimensionality k...")

    kx = choose_k_from_spectrum(Sx, threshold=0.9)
    ky = choose_k_from_spectrum(Sy, threshold=0.9)

    k = min(kx, ky)

    print(f"k (vision): {kx}, k (text): {ky}, using k = {k}")
    print("\nProjecting to top-k subspace...")

    Y_k, Sy, Vy = project_to_top_k(Y, k)
    X_k, Sx, Vx = project_to_top_k(X, k)

    print("X_k shape:", X_k.shape)
    print("Y_k shape:", Y_k.shape)

    pairs_k = build_cross_modal_pairs(X_k, Y_k)
    print("Pairs (projected):", len(pairs_k))

    print("\nAlignment analysis...")

    np.random.seed(42)

    train_pairs, test_pairs = split_alignment_train_test(
        pairs_k, train_ratio=0.8
    )

    alignment_model = fit_manifold_alignment(train_pairs)

    alignment_metrics = evaluate_alignment_quality(
        test_pairs, alignment_model
    )

    print("\nAlignment quality:")
    for key, value in alignment_metrics.items():
        if key not in ["ranks", "similarity_matrix"]:
            print(f"{key}: {value:.4f}")

    print("Ranks:", alignment_metrics["ranks"])

    print("\nBuilding aligned feature space...")

    X_aligned = transform_3d_features(X_k, alignment_model)

    X_aligned = X_aligned / np.linalg.norm(X_aligned, axis=1, keepdims=True)

    print("Aligned features shape:", X_aligned.shape)

    labels = np.load("labels.npy", allow_pickle=True)

    text_model = SentenceTransformer("all-MiniLM-L6-v2")

    print("\nRunning QA demo...")

    questions = [
        "What is on the table?",
        "What is near the sheep?",
        "What objects are next to the mug?",
        "Is there any food in the scene?"
    ]

    for q in questions:
        print("\n==============================")
        print("Q:", q)

        q_emb = text_model.encode(q)
        q_emb = preprocess_text_embeddings(q_emb)
        q_emb = q_emb / np.linalg.norm(q_emb)

        q_emb_k = q_emb @ Vy
        q_emb_k = q_emb_k / np.linalg.norm(q_emb_k)

        indices, scores = retrieve_relevant_3d_features(
            q_emb_k, X_aligned, top_k=5
        )

        context = build_3d_tokens_for_slm(indices, labels)

        answer = generate_answer(q, context)

        print("\nTop retrieved:")
        for i, idx in enumerate(indices):
            print(f"{i+1}. {labels[idx]} (score={scores[i]:.3f})")

        print("\nGenerated answer:")
        print(answer)
if __name__ == "__main__":
    main()