# Evaluation metrics used to measure classifier performance
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


def evaluate_model(y_true, y_pred):
    """
    Evaluates model performance using standard classification metrics.

    These metrics will be used in Report 3 results:
    1. Accuracy
    2. Precision
    3. Recall
    4. F1-score
    5. Full classification report

    Parameters:
    y_true -> actual class labels
    y_pred -> predicted class labels

    Returns:
    Dictionary containing all evaluation results
    """

    # Accuracy measures the overall percentage of correct predictions
    accuracy = accuracy_score(y_true, y_pred)

    # Weighted precision handles both binary and multi-class datasets
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)

    # Weighted recall handles class imbalance properly
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)

    # Weighted F1-score balances precision and recall
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # Detailed class-wise report for final report tables
    report = classification_report(y_true, y_pred, zero_division=0)

    # Return all metrics in one dictionary
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "classification_report": report
    }


def print_evaluation_results(dataset_name, results):
    """
    Prints model evaluation results in a clean readable format.

    This output can later be copied into Report 3 tables.
    """

    print("\n" + "=" * 60)
    print(f"Evaluation Results for {dataset_name}")
    print("=" * 60)

    print(f"Accuracy : {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall   : {results['recall']:.4f}")
    print(f"F1-score : {results['f1_score']:.4f}")

    print("\nDetailed Classification Report:")
    print(results["classification_report"])