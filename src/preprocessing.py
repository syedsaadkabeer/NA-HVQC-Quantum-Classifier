import numpy as np

# Import standard datasets from sklearn
from sklearn.datasets import load_iris, load_breast_cancer

# For splitting dataset into training and testing sets
from sklearn.model_selection import train_test_split

# For scaling features into a specific range
from sklearn.preprocessing import MinMaxScaler

# For dimensionality reduction 
from sklearn.decomposition import PCA


def load_and_preprocess_dataset(dataset_name="iris", n_qubits=4, test_size=0.2, random_state=42):
    """
    This function performs the full classical preprocessing pipeline
    described in your Report 2.

    Steps:
    1. Load dataset (Iris or Breast Cancer)
    2. Apply PCA to reduce features to match number of qubits
    3. Normalize features to [0, π] (required for quantum rotation gates)
    4. Split into training and testing sets

    Returns:
    X_train, X_test -> processed feature data
    y_train, y_test -> labels
    variance_ratio  -> used later for adaptive encoding decision
    """

    # -------------------------
    # Step 1: Load dataset
    # -------------------------
    if dataset_name == "iris":
        data = load_iris()         # Load Iris dataset (3-class classification)
        X = data.data              # Feature matrix
        y = data.target            # Labels

    elif dataset_name == "breast_cancer":
        data = load_breast_cancer()  # Binary classification dataset
        X = data.data
        y = data.target

    else:
        raise ValueError("dataset_name must be 'iris' or 'breast_cancer'")

    # -------------------------
    # Step 2: PCA (Dimensionality Reduction)
    # -------------------------
    # Reduce number of features to match available qubits
    # This directly follows your Report 2 design (qubit constraint)
    pca = PCA(n_components=n_qubits)
    X_pca = pca.fit_transform(X)

    # -------------------------
    # Step 3: Normalize to [0, π]
    # -------------------------
    # Required because quantum rotation gates (Ry, Rz) use angles
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_scaled = scaler.fit_transform(X_pca)

    # -------------------------
    # Step 4: Train-Test Split
    # -------------------------
    # Stratify ensures class balance in train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # -------------------------
    # Step 5: Compute variance ratio
    # -------------------------
    # This will be used later for adaptive encoding decision
    # (angle encoding vs Pauli-Z feature map)
    variance_ratio = np.sum(pca.explained_variance_ratio_)

    # Return processed data
    return X_train, X_test, y_train, y_test, variance_ratio