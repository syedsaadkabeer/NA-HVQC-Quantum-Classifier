import sys
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Add project root path so Python can import files from src correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import preprocessing function for loading and preparing datasets
from src.preprocessing import load_and_preprocess_dataset

# Import quantum circuit creation, simulation, and prediction functions
from src.quantum_model import (
    create_full_vqc_circuit,
    run_quantum_circuit,
    predict_from_counts,
    train_vqc_cobyla,
    predict_from_expectations,
    calculate_z_expectations
)

# Import evaluation functions for accuracy, precision, recall, and F1-score
from src.evaluation import evaluate_model, print_evaluation_results

def run_classical_baseline():
    X_train, X_test, y_train, y_test, _ = load_and_preprocess_dataset("iris", 4)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    results = evaluate_model(y_test, preds)
    print_evaluation_results("Classical Baseline", results)


def generate_random_parameters(n_layers, n_qubits, random_state=42):
    """
    Generates random trainable parameters for the variational quantum circuit.

    Shape:
    n_layers x n_qubits x 2

    Each qubit has:
    1. Ry parameter
    2. Rz parameter

    Note:
    This is only the initial circuit test stage.
    Later, we will replace this with COBYLA optimization.
    """

    # Set random seed so results are reproducible
    np.random.seed(random_state)

    # Generate random angles between 0 and pi
    parameters = np.random.uniform(
        low=0,
        high=np.pi,
        size=(n_layers, n_qubits, 2)
    )

    return parameters


def test_circuit_creation(dataset_name, n_qubits, n_layers):
    """
    Tests whether the preprocessing and quantum circuit creation
    are working correctly for a selected dataset.

    This step does not train the model yet.
    It only verifies:
    1. Dataset loading
    2. PCA preprocessing
    3. Adaptive encoding selection
    4. Full VQC circuit creation
    """

    print("\n" + "=" * 70)
    print(f"Testing NA-HVQC Circuit for {dataset_name}")
    print("=" * 70)

    # Load and preprocess dataset
    X_train, X_test, y_train, y_test, variance_ratio = load_and_preprocess_dataset(
        dataset_name=dataset_name,
        n_qubits=n_qubits
    )

    # Display dataset information
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples : {len(X_test)}")
    print(f"Number of qubits: {n_qubits}")
    print(f"PCA variance ratio: {variance_ratio:.4f}")

    # Generate initial random parameters for circuit testing
    parameters = generate_random_parameters(
        n_layers=n_layers,
        n_qubits=n_qubits
    )

    # Create one sample circuit using the first training example
    sample_features = X_train[0]

    circuit, encoding_type = create_full_vqc_circuit(
        features=sample_features,
        parameters=parameters,
        variance_ratio=variance_ratio,
        n_layers=n_layers
    )

    # Display selected encoding type
    print(f"Selected encoding: {encoding_type}")

    # Display circuit depth and gate count
    print(f"Circuit depth: {circuit.depth()}")
    print(f"Gate counts  : {circuit.count_ops()}")

    # Print the circuit diagram in text form
    print("\nQuantum Circuit:")
    print(circuit.draw(output="text"))

    return circuit


def run_baseline_vqc(dataset_name, n_qubits, n_layers, n_classes):
    """
    Runs the baseline NA-HVQC model without training.

    This function:
    1. Loads and preprocesses the dataset
    2. Builds a quantum circuit for each test sample
    3. Runs each circuit on Qiskit Aer Simulator
    4. Converts measurement results into class predictions
    5. Evaluates the predictions using standard metrics

    Note:
    This is still not the optimized final model.
    COBYLA training will be added in the next step.
    """

    print("\n" + "=" * 70)
    print(f"Running Baseline NA-HVQC for {dataset_name}")
    print("=" * 70)

    # Load processed train/test data
    X_train, X_test, y_train, y_test, variance_ratio = load_and_preprocess_dataset(
        dataset_name=dataset_name,
        n_qubits=n_qubits
    )

    # Generate initial random trainable parameters
    parameters = generate_random_parameters(
        n_layers=n_layers,
        n_qubits=n_qubits
    )

    # Store model predictions for all test samples
    predictions = []

    # Run quantum circuit for each test sample
    for index, sample_features in enumerate(X_test):
        # Build full adaptive VQC circuit
        circuit, encoding_type = create_full_vqc_circuit(
            features=sample_features,
            parameters=parameters,
            variance_ratio=variance_ratio,
            n_layers=n_layers
        )

        # Execute circuit on Qiskit Aer Simulator
        counts = run_quantum_circuit(circuit, shots=2048)

        # Convert quantum measurement results into predicted class
        predicted_class, probabilities = predict_from_expectations(
        counts=counts,
        n_qubits=n_qubits,
        n_classes=n_classes
)

        # Save prediction
        predictions.append(predicted_class)

        # Print progress so we know execution is working
        print(f"Processed sample {index + 1}/{len(X_test)}")

    # Evaluate predictions
    results = evaluate_model(y_test, predictions)

    # Print final results
    print_evaluation_results(dataset_name, results)

    return results

def run_trained_vqc(dataset_name, n_qubits, n_layers, n_classes):
    """
    Runs the complete trained NA-HVQC experiment.

    This function:
    1. Loads and preprocesses the dataset
    2. Trains the quantum circuit parameters using COBYLA
    3. Runs the trained circuit on the test set
    4. Evaluates final classification results
    """

    print("\n" + "=" * 70)
    print(f"Running Trained NA-HVQC for {dataset_name}")
    print("=" * 70)

    # Load preprocessed dataset
    X_train, X_test, y_train, y_test, variance_ratio = load_and_preprocess_dataset(
        dataset_name=dataset_name,
        n_qubits=n_qubits
    )

    # Train VQC parameters using COBYLA optimizer
    trained_parameters, optimization_result = train_vqc_cobyla(
        X_train=X_train,
        y_train=y_train,
        variance_ratio=variance_ratio,
        n_layers=n_layers,
        n_qubits=n_qubits,
        n_classes=n_classes,
        maxiter=100,
        shots=2048
    )

    # Store predictions from trained model
    predictions = []

    # Test trained model on all test samples
    for index, sample_features in enumerate(X_test):

        # Build circuit using trained parameters
        circuit, encoding_type = create_full_vqc_circuit(
            features=sample_features,
            parameters=trained_parameters,
            variance_ratio=variance_ratio,
            n_layers=n_layers
        )

        # Run circuit on simulator
        counts = run_quantum_circuit(circuit, shots=2048)

        # Convert measurement results into class label
        predicted_class = predict_from_counts(
            counts=counts,
            n_classes=n_classes
        )

        # Store prediction
        predictions.append(predicted_class)

        # Print progress
        print(f"Tested sample {index + 1}/{len(X_test)}")

    # Evaluate final trained model performance
    results = evaluate_model(y_test, predictions)

    # Print final results
    print_evaluation_results(dataset_name + " - Trained COBYLA", results)

    return results, trained_parameters, optimization_result


def run_hybrid_quantum_softmax(dataset_name, n_qubits, n_layers):
    """
    Runs a stronger hybrid quantum-classical classifier.

    This method follows the Report 2 idea more correctly:
    1. Use quantum circuit to transform classical data
    2. Extract Pauli-Z expectation values from measurements
    3. Pass expectation values into a classical softmax classifier

    LogisticRegression with multi_class='multinomial' acts as the softmax layer.
    """

    print("\n" + "=" * 70)
    print(f"Running Hybrid Quantum Softmax Classifier for {dataset_name}")
    print("=" * 70)

    # Load dataset using the same preprocessing pipeline
    X_train, X_test, y_train, y_test, variance_ratio = load_and_preprocess_dataset(
        dataset_name=dataset_name,
        n_qubits=n_qubits
    )

    # Generate fixed quantum circuit parameters
    # These create the quantum feature transformation
    parameters = generate_random_parameters(
        n_layers=n_layers,
        n_qubits=n_qubits
    )

    # Store quantum-transformed training features
    quantum_train_features = []

    print("\nExtracting quantum features from training data...")

    for index, sample_features in enumerate(X_train):

        # Build quantum circuit for one training sample
        circuit, encoding_type = create_full_vqc_circuit(
            features=sample_features,
            parameters=parameters,
            variance_ratio=variance_ratio,
            n_layers=n_layers
        )

        # Run circuit on simulator
        counts = run_quantum_circuit(circuit, shots=2048)

        # Convert measurements into Pauli-Z expectation values
        expectations = calculate_z_expectations(
            counts=counts,
            n_qubits=n_qubits
        )

        # Store expectation values as quantum features
        quantum_train_features.append(expectations)

        print(f"Training quantum feature {index + 1}/{len(X_train)} extracted")

    # Store quantum-transformed testing features
    quantum_test_features = []

    print("\nExtracting quantum features from testing data...")

    for index, sample_features in enumerate(X_test):

        # Build quantum circuit for one testing sample
        circuit, encoding_type = create_full_vqc_circuit(
            features=sample_features,
            parameters=parameters,
            variance_ratio=variance_ratio,
            n_layers=n_layers
        )

        # Run circuit on simulator
        counts = run_quantum_circuit(circuit, shots=2048)

        # Convert measurements into Pauli-Z expectation values
        expectations = calculate_z_expectations(
            counts=counts,
            n_qubits=n_qubits
        )

        # Store expectation values as quantum features
        quantum_test_features.append(expectations)

        print(f"Testing quantum feature {index + 1}/{len(X_test)} extracted")

    quantum_train_features = np.array(quantum_train_features)
    quantum_test_features = np.array(quantum_test_features)

    # Normalize features (VERY IMPORTANT)
    scaler = StandardScaler()
    quantum_train_features = scaler.fit_transform(quantum_train_features)
    quantum_test_features = scaler.transform(quantum_test_features)

    # Train classical softmax classifier on quantum features
    softmax_classifier = LogisticRegression(
        max_iter=1000,
    )

    softmax_classifier.fit(quantum_train_features, y_train)

    # Predict test labels
    predictions = softmax_classifier.predict(quantum_test_features)

    # Evaluate results
    results = evaluate_model(y_test, predictions)

    # Print final results
    print_evaluation_results(dataset_name + " - Hybrid Quantum Softmax", results)

    print(f"\nEncoding used: {encoding_type}")

    return results

if __name__ == "__main__":
    """
    Main execution point.

    First, we test the circuit creation.
    Then, we run a baseline prediction experiment.
    """

    run_classical_baseline()

    # Step 1: Verify circuit creation for Iris
    test_circuit_creation(
        dataset_name="iris",
        n_qubits=4,
        n_layers=3
    )

    # Step 2: Run baseline prediction experiment for Iris
    run_baseline_vqc(
        dataset_name="iris",
        n_qubits=4,
        n_layers=3,
        n_classes=3
    )

    # Step 3: Run trained COBYLA experiment for Iris
    run_trained_vqc(
        dataset_name="iris",
        n_qubits=4,
        n_layers=3,
        n_classes=3
    )

    # Step 4: Run stronger hybrid quantum-softmax classifier
    run_hybrid_quantum_softmax(
        dataset_name="iris",
        n_qubits=4,
        n_layers=3
    )