import numpy as np

# Core Qiskit class used to build quantum circuits
from qiskit import QuantumCircuit


def create_angle_encoding_circuit(features):
    """
    Creates an angle encoding circuit.

    In angle encoding, each classical feature value is encoded
    as a rotation angle on one qubit.

    Example:
    feature x_i -> Ry(x_i)

    This is suitable for datasets with lower feature correlation
    because it keeps the circuit shallow and NISQ-friendly.
    """

    # Number of qubits equals the number of selected PCA features
    n_qubits = len(features)

    # Create a quantum circuit with n qubits
    qc = QuantumCircuit(n_qubits)

    # Encode each feature into a separate qubit using Ry rotation
    for i, value in enumerate(features):
        qc.ry(value, i)

    return qc


def create_pauli_z_feature_map(features):
    """
    Creates a second-order Pauli-Z feature map.

    This encoding is used when features are more correlated.
    It includes:
    1. Single-qubit Z rotations
    2. Two-qubit ZZ interaction terms

    The purpose is to capture relationships between feature pairs.
    """

    # Number of qubits equals the number of selected PCA features
    n_qubits = len(features)

    # Create a quantum circuit
    qc = QuantumCircuit(n_qubits)

    # Apply Hadamard gates to create superposition
    # This allows the feature map to spread data across quantum state space
    for i in range(n_qubits):
        qc.h(i)

    # Encode individual feature values using Rz rotations
    for i, value in enumerate(features):
        qc.rz(value, i)

    # Add second-order ZZ interactions between neighboring qubits
    # This approximates the Pauli-Z feature map described in Report 2
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
        qc.rz(features[i] * features[i + 1], i + 1)
        qc.cx(i, i + 1)

    return qc


def select_adaptive_encoding(features, variance_ratio, threshold=0.85):
    """
    FORCE angle encoding for stability
    """
    return create_angle_encoding_circuit(features), "Angle Encoding"


def create_variational_layer(qc, parameters, layer_index):
    """
    Adds one trainable variational layer to the circuit.

    Each layer contains:
    1. Ry rotation gates
    2. Rz rotation gates
    3. Linear CNOT entanglement

    This follows the shallow variational circuit design from Report 2.
    """

    n_qubits = qc.num_qubits

    # Apply trainable Ry and Rz rotations to each qubit
    for qubit in range(n_qubits):
        theta_ry = parameters[layer_index][qubit][0]
        theta_rz = parameters[layer_index][qubit][1]

        qc.ry(theta_ry, qubit)
        qc.rz(theta_rz, qubit)

    # Apply linear entanglement using CNOT gates
    # Linear topology keeps the circuit simple and hardware-friendly
    for qubit in range(n_qubits - 1):
        qc.cx(qubit, qubit + 1)

    return qc


def create_full_vqc_circuit(features, parameters, variance_ratio, n_layers=3):
    """
    Builds the complete Variational Quantum Classifier circuit.

    Full pipeline:
    1. Adaptive feature encoding
    2. Shallow variational layers
    3. Measurement on all qubits

    This represents the NA-HVQC implementation from Report 2.
    """

    # Step 1: Select and create adaptive feature encoding circuit
    qc, encoding_type = select_adaptive_encoding(features, variance_ratio)

    # Step 2: Add trainable variational layers
    for layer in range(n_layers):
        qc = create_variational_layer(qc, parameters, layer)

    # Step 3: Measure all qubits
    # Measurements are needed to convert quantum states into classical output
    qc.measure_all()

    return qc, encoding_type

# AerSimulator is used to run quantum circuits on a local simulator
from qiskit_aer import AerSimulator


def run_quantum_circuit(circuit, shots=1024):
    """
    Runs a quantum circuit using Qiskit Aer Simulator.

    Parameters:
    circuit -> complete quantum circuit with measurements
    shots   -> number of repeated circuit executions

    Returns:
    counts -> measurement results, for example {'0000': 520, '1111': 504}
    """

    # Create local quantum simulator
    simulator = AerSimulator()

    # Run the circuit on the simulator
    job = simulator.run(circuit, shots=shots)

    # Get the completed result object
    result = job.result()

    # Extract measurement counts from the result
    counts = result.get_counts(circuit)

    return counts


def predict_from_counts(counts, n_classes):
    """
    Converts quantum measurement counts into a class prediction.

    For this simple baseline:
    - Measurement bitstrings are converted into integers.
    - Integer value is mapped into class labels using modulo operation.

    Example for Iris:
    '0000' -> 0
    '0001' -> 1
    '0010' -> 2

    This is the first working prediction method.
    Later, COBYLA optimization will improve these predictions.
    """

    # Select the most frequently measured bitstring
    most_common_bitstring = max(counts, key=counts.get)

    # Convert bitstring into integer
    measured_value = int(most_common_bitstring, 2)

    # Map integer into valid class range
    predicted_class = measured_value % n_classes

    return predicted_class


from scipy.optimize import minimize


def calculate_class_probabilities(counts, n_classes):
    """
    Converts quantum measurement counts into class probabilities.

    Why this is needed:
    - The simulator gives measurement bitstrings like '0001', '1010', etc.
    - Machine learning needs probabilities for each class.
    - We map each bitstring into a class using modulo operation.
    """

    # Create empty probability list for all classes
    class_counts = np.zeros(n_classes)

    # Count total number of measurement shots
    total_shots = sum(counts.values())

    # Convert each measured bitstring into a class label
    for bitstring, count in counts.items():
        measured_value = int(bitstring, 2)
        class_label = measured_value % n_classes
        class_counts[class_label] += count

    # Convert counts into probabilities
    probabilities = class_counts / total_shots

    return probabilities


def cross_entropy_loss(y_true, probabilities):
    """
    Calculates cross-entropy loss for one training sample.

    The optimizer needs one single numeric loss value.
    Therefore, this function always returns a float.
    """

    # Prevent log(0)
    epsilon = 1e-10

    # Convert true label into integer
    y_true = int(y_true)

    # Convert probability list/array into clean 1D NumPy array
    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)

    # If label index is invalid, return high penalty
    if y_true >= len(probabilities):
        return 10.0

    # Calculate negative log probability of the correct class
    loss = -np.log(probabilities[y_true] + epsilon)

    return float(loss)


def flatten_parameters(parameters):
    """
    Converts 3D circuit parameters into 1D array.

    SciPy COBYLA requires parameters in a flat 1D format.
    """

    return parameters.flatten()


def reshape_parameters(flat_parameters, n_layers, n_qubits):
    """
    Converts flat 1D parameters back into 3D circuit parameter shape.

    Original shape:
    n_layers x n_qubits x 2
    """

    return flat_parameters.reshape((n_layers, n_qubits, 2))


def train_vqc_cobyla(
    X_train,
    y_train,
    variance_ratio,
    n_layers,
    n_qubits,
    n_classes,
    maxiter=30,
    shots=1024
):
    """
    Trains the Variational Quantum Classifier using COBYLA.

    This is the main training function for your Report 3 implementation.

    Important:
    - We keep maxiter small initially because quantum simulation is slow.
    - Later we can increase it for final result generation.
    """

    # Generate initial trainable circuit parameters
    initial_parameters = np.random.uniform(
        low=0,
        high=np.pi,
        size=(n_layers, n_qubits, 2)
    )

    # Convert parameters to flat form for SciPy optimizer
    initial_flat_parameters = flatten_parameters(initial_parameters)

    def objective_function(flat_parameters):
        """
        Objective function minimized by COBYLA.

        For each training sample:
        1. Rebuild the quantum circuit
        2. Run it on AerSimulator
        3. Convert counts into probabilities
        4. Compute cross-entropy loss
        """

        # Restore parameters to circuit format
        parameters = reshape_parameters(flat_parameters, n_layers, n_qubits)

        # Store losses for all training samples
        losses = []

        # Use only a small subset for faster training during development
        for features, label in zip(X_train[:50], y_train[:50]):
            circuit, _ = create_full_vqc_circuit(
                features=features,
                parameters=parameters,
                variance_ratio=variance_ratio,
                n_layers=n_layers
            )

            counts = run_quantum_circuit(circuit, shots=shots)

            # predict_from_expectations returns two values:
            # 1. predicted_class
            # 2. probabilities
            # We only need probabilities for calculating loss.
            _, probabilities = predict_from_expectations(
            counts=counts,
            n_qubits=n_qubits,
            n_classes=n_classes
        )

            loss = cross_entropy_loss(label, probabilities)
            # Convert loss into a simple float number before saving
            losses.append(loss)

        # Calculate average loss safely as a float
        average_loss = float(np.mean(losses))

        print(f"Current loss: {average_loss:.4f}")

        return average_loss

    print("\nStarting COBYLA training...")

    # Run COBYLA optimizer
    optimization_result = minimize(
        objective_function,
        initial_flat_parameters,
        method="COBYLA",
        options={"maxiter": maxiter, "disp": True}
    )

    print("\nCOBYLA training completed.")

    # Convert optimized parameters back to circuit format
    trained_parameters = reshape_parameters(
        optimization_result.x,
        n_layers,
        n_qubits
    )

    return trained_parameters, optimization_result



def calculate_z_expectations(counts, n_qubits):
    """
    Converts measurement counts into Pauli-Z expectation values.

    Pauli-Z expectation:
    - If qubit is measured as 0, contribution is +1
    - If qubit is measured as 1, contribution is -1

    The final value lies between -1 and +1 for each qubit.
    """

    # Store expectation value for each qubit
    expectations = np.zeros(n_qubits)

    # Total number of measurement shots
    total_shots = sum(counts.values())

    # Process every measured bitstring
    for bitstring, count in counts.items():

        # Reverse bitstring because Qiskit displays classical bits in reverse order
        reversed_bits = bitstring[::-1]

        # Calculate contribution for each qubit
        for qubit in range(n_qubits):
            if reversed_bits[qubit] == "0":
                expectations[qubit] += count
            else:
                expectations[qubit] -= count

    # Normalize expectation values by total shots
    expectations = expectations / total_shots

    return expectations


def softmax(values):
    """
    Converts raw model outputs into class probabilities.

    Softmax is used for multi-class classification such as Iris.
    """

    # Shift values for numerical stability
    shifted_values = values - np.max(values)

    # Exponentiate values
    exp_values = np.exp(shifted_values)

    # Normalize into probabilities
    probabilities = exp_values / np.sum(exp_values)

    return probabilities


def predict_from_expectations(counts, n_qubits, n_classes):
    """
    Predict class using expectation values + softmax
    """

    # Get expectation values from measurements
    expectations = calculate_z_expectations(counts, n_qubits)

    # Ensure correct length for class scores
    if len(expectations) < n_classes:
        padded = np.zeros(n_classes)
        padded[:len(expectations)] = expectations
        class_scores = padded
    else:
        class_scores = expectations[:n_classes]

    # Convert to probabilities using softmax
    probabilities = softmax(class_scores)

    # Pick class with highest probability
    predicted_class = int(np.argmax(probabilities))

    return predicted_class, probabilities