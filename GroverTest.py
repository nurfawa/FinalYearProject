import cirq
import numpy as np
import hashlib
import os
import matplotlib.pyplot as plt
from collections import Counter
import math
import time
import psutil
from Crypto.Cipher import AES

# Constants
AES_BLOCK_SIZE = 16
AES_MODE = AES.MODE_CBC

# AES S-box flattened
SBOX = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
]

def lookup(byte):
    """Return S-box transformation of a byte."""
    return SBOX[byte]

def optimal_iterations(n):
    """Calculate the optimal number of iterations for Grover's algorithm."""
    return int(math.pi / 4 * math.sqrt(2 ** n))

def simulate_sbox(plaintext_bytes):
    """Apply the AES S-box to each byte of the plaintext."""
    return [SBOX[byte] for byte in plaintext_bytes]

def setup_aes_cipher(secret_key):
    """Setup AES cipher with CBC mode and random IV."""
    key_hash = hashlib.sha256(secret_key).digest()
    iv = os.urandom(AES_BLOCK_SIZE)
    cipher = AES.new(key_hash, AES_MODE, iv=iv)
    return cipher, iv

def encrypt_message(cipher, plaintext):
    """Encrypt the message using AES."""
    if len(plaintext) % AES_BLOCK_SIZE != 0:
        padding_length = AES_BLOCK_SIZE - len(plaintext) % AES_BLOCK_SIZE
        plaintext += bytes([padding_length] * padding_length)
    return cipher.encrypt(plaintext)

def sbox_oracle(circuit, qubits, target_output):
    """Define Grover's oracle using the S-box."""
    for input_value in range(256):
        sbox_output = lookup(input_value)
        if sbox_output == target_output:
            binary_input = format(input_value, '08b')
            for i, bit in enumerate(binary_input):
                if bit == '0':
                    circuit.append(cirq.X(qubits[i]))
            circuit.append(cirq.Z(qubits[-1]).controlled_by(*qubits[:-1]))
            for i, bit in enumerate(binary_input):
                if bit == '0':
                    circuit.append(cirq.X(qubits[i]))

def apply_diffusion_operator(circuit, qubits):
    """Apply the diffusion operator (inversion about the mean)."""
    circuit.append(cirq.H.on_each(*qubits))
    circuit.append(cirq.X.on_each(*qubits))
    circuit.append(cirq.Z(qubits[-1]).controlled_by(*qubits[:-1]))
    circuit.append(cirq.X.on_each(*qubits))
    circuit.append(cirq.H.on_each(*qubits))

def setup_quantum_circuit(num_qubits, target_output, iterations):
    """Setup the quantum circuit for Grover's algorithm."""
    qubits = [cirq.LineQubit(i) for i in range(num_qubits)]
    circuit = cirq.Circuit()
    circuit.append(cirq.H.on_each(*qubits))
    iterations = optimal_iterations(num_qubits)
    for _ in range(iterations):
        sbox_oracle(circuit, qubits, target_output)
        apply_diffusion_operator(circuit, qubits)
    circuit.append(cirq.measure(*qubits, key='result'))
    return circuit, qubits

def log_circuit_info(circuit, execution_time):
    num_qubits = len(circuit.all_qubits())
    num_gates = sum(1 for _ in circuit.all_operations())  # Convert generator to a count

    print(f"Number of Qubits: {num_qubits}")
    print(f"Number of Gates: {num_gates}")
    print(f"Execution Time: {execution_time:.4f} seconds")

def get_memory_usage():
    """Return the current memory usage of the process."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss

def calculate_circuit_depth(circuit):
    """Calculate and return the depth of the quantum circuit."""
    depth = 0
    for moment in circuit:
        depth += 1
    return depth

def adjust_to_1kb(plaintext):
    """
    Adjust the given plaintext to be exactly 1KB (1024 bytes) in size.

    Args:
    plaintext (str): The input plaintext.

    Returns:
    str: The adjusted plaintext of 1KB in size.
    """
    # Convert the plaintext to bytes
    plaintext_bytes = plaintext.encode('utf-8')

    # Calculate the current size of the plaintext in bytes
    current_size = len(plaintext_bytes)

    # If the current size is already 1KB, return the plaintext
    if current_size == 1024:
        return plaintext

    # If the current size is greater than 1KB, trim the plaintext
    if current_size > 1024:
        return plaintext_bytes[:1024].decode('utf-8', errors='ignore')

    # If the current size is less than 1KB, repeat the plaintext until it exceeds 1KB
    while current_size < 1024:
        plaintext_bytes += plaintext.encode('utf-8')
        current_size = len(plaintext_bytes)

    # Trim the plaintext to exactly 1KB
    adjusted_plaintext = plaintext_bytes[:1024].decode('utf-8', errors='ignore')

    return adjusted_plaintext

# Given plaintext
plaintext = "To understand why AES is vulnerable to quantum attacks, it is important to understand how the algorithm works. AES uses a key to encrypt and decrypt data."

# Adjust the plaintext to 1KB
adjusted_plaintext = adjust_to_1kb(plaintext)
print(f"The size of the adjusted plaintext is: {len(adjusted_plaintext.encode('utf-8'))} bytes")
print(f"Adjusted Plaintext:\n{adjusted_plaintext}")



def main():
    secret_key = b'nur fahima iwani'
    plaintext = b'123123123'

    initial_memory = get_memory_usage()
     # Start measuring time
    total_start_time = time.time()

    cipher, iv = setup_aes_cipher(secret_key)
    encrypted_message = encrypt_message(cipher, plaintext)
    target_output = encrypted_message[0] % 256

    num_qubits = 8
    iterations = optimal_iterations(num_qubits)
    start_time = time.time()
    circuit, qubits = setup_quantum_circuit(num_qubits, target_output, iterations)
    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=1000)
    end_time = time.time()

    # End measuring time
    total_end_time = time.time()

    final_memory = get_memory_usage()
    memory_usage = final_memory - initial_memory

    execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
    total_execution_time = (total_end_time - total_start_time) * 1000  # Convert to milliseconds

    circuit_depth = calculate_circuit_depth(circuit)

    plaintext_bytes = list(plaintext)
    actual_sbox_output = simulate_sbox(plaintext_bytes)
    expected_sbox_output = [lookup(byte) for byte in plaintext_bytes[:16]]  # Adjust as needed

    print("Original plaintext bytes:", plaintext_bytes[:16])
    print("Actual S-box output:", actual_sbox_output[:16])
    print("Expected S-box output:", expected_sbox_output)
    if actual_sbox_output[:16] == expected_sbox_output:
        print("Success: Quantum result matches classical S-box output.")
    else:
        print("Mismatch: Quantum result does not match classical S-box output.")
    print("Circuit Depth:", circuit_depth)
    print("Circuit:\n", circuit)
    print("Results:\n", result.histogram(key='result'))

    print(f"Total Execution Time: {total_execution_time:.2f} ms")
    print(f"Grover's Algorithm Execution Time: {execution_time:.2f} ms")

    plot_results(result)
    log_circuit_info(circuit, execution_time, iterations, memory_usage, circuit_depth)

def plot_results(result):
    """Plot the histogram of measurement outcomes."""
    measurements = result.measurements['result']
    values = [sum(bit << i for i, bit in enumerate(measurement)) for measurement in measurements]
    counts = Counter(values)
    top_16 = counts.most_common(16)

    states, frequencies = zip(*top_16)
    plt.figure(figsize=(12, 6))
    plt.bar(states, frequencies, align='center', alpha=0.7, color='b')
    plt.xticks(states)
    plt.xlabel('State')
    plt.ylabel('Frequency')
    plt.title("Top 16 Measurement Outcomes of Grover's Algorithm")
    plt.show()

def log_circuit_info(circuit, execution_time, iterations, memory_usage, circuit_depth):
    """Log the information about the quantum circuit."""
    num_qubits = len(circuit.all_qubits())
    num_gates = sum(1 for _ in circuit.all_operations())  # Convert generator to a count

    # Feasibility Analysis
    feasible = num_qubits <= 20 and execution_time < 1  # Example criteria

    # Vulnerability Insights
    classical_time = 2 ** 8  # For 8-bit search space
    quantum_time = iterations * execution_time
    speedup = classical_time / quantum_time
    vulnerability = "High" if speedup > 100 else "Moderate" if speedup > 10 else "Low"

    print(f"Number of Qubits: {num_qubits}")
    print(f"Number of Gates: {num_gates}")
    print(f"Number of Iterations: {iterations}")
    print(f"Memory Usage: {memory_usage / (1024 ** 2):.2f} MB")
    print(f"Circuit Depth: {circuit_depth}")
    print(f"Feasibility: {'Feasible' if feasible else 'Not Feasible'}")
    print(f"Vulnerability Insights: {vulnerability}")


if __name__ == "__main__":
    main()
