def retrieve_relevant_3d_features(query_embedding, aligned_scene_features, top_k):
    """
    Retrieve the most relevant 3D scene features for a given question.
    """
    pass


def build_3d_tokens_for_slm(retrieved_features):
    """
    Convert retrieved 3D features into an SLM-consumable representation.
    """
    pass


def format_qa_prompt(question, scene_context):
    """
    Build the prompt that combines the question with 3D scene context.
    """
    pass


def run_small_slm_qa(prompt, slm):
    """
    Run the small language model on the prompt.
    """
    pass


def evaluate_downstream_qa(questions, references, aligned_scene_features, slm):
    """
    Evaluate the end-to-end 3D QA pipeline.
    """
    pass