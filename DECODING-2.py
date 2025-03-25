import numpy as np
# -*- coding: utf-8 -*-
def compute_syndrome(H, r):
    return np.dot(H, r.T) % 2

def decode(H, received):
    n = len(received)
    r = np.array([int(b) for b in received])
    syndrome = compute_syndrome(H, r)

    if np.count_nonzero(syndrome) == 0:
        print(f"Received: {received}-> No error detected. Message: {r[:5].tolist()}")
        return

    # Try to find a single-bit error
    for i in range(n):
        error_vector = np.zeros(n, dtype=int)
        error_vector[i] = 1
        test_r = (r + error_vector) % 2
        test_syndrome = compute_syndrome(H, test_r)
        if np.count_nonzero(test_syndrome) == 0:
            print(f"Received: {received} -> Single-bit error at position {i}. Corrected to: {test_r.tolist()} -> Message: {test_r[:5].tolist()}")
            return

    # No correction possible
    print(f"Received: {received} -> Error detected but not correctable. Request retransmission.")

# Define the parity-check matrix H
H = np.array([
    [1, 1, 1, 0, 1, 0, 0, 0, 0, 0],
    [1, 0, 1, 1, 1, 0, 0, 0, 0, 0],
    [1, 1, 0, 1, 0, 0, 1, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 1, 0, 0],
    [0, 1, 0, 1, 0, 0, 0, 0, 0, 1]
])

# List of received messages
received_messages = [
    "1001011100",
    "0110100100",
    "1100000011",
    "1100101010",
    "1110111000"
]

# Decode each message
print("Decoding received messages...\n")
for msg in received_messages:
    decode(H, msg)
