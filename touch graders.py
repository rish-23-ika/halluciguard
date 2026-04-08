def compute_f1_score(predicted, ground_truth):
    """
    Computes F1 score between predicted indices and ground truth indices
    """

    predicted_set = set(predicted)
    truth_set = set(ground_truth)

    if len(predicted_set) == 0 and len(truth_set) == 0:
        return 1.0

    if len(predicted_set) == 0 or len(truth_set) == 0:
        return 0.0

    tp = len(predicted_set & truth_set)
    precision = tp / len(predicted_set)
    recall = tp / len(truth_set)

    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)