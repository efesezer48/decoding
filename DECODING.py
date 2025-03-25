import numpy as np

### TASK 1 ###
def linear_code(A):
    A = np.array(A)
    n_minus_m, m = A.shape
    n = n_minus_m + m
    I_n_m = np.identity(n_minus_m, dtype=int)
    I_m = np.identity(m, dtype=int)
    H = np.concatenate((A, I_n_m), axis=1)
    G = np.concatenate((I_m, A.T), axis=1)
    return H, G

# linear encode 
def linear_encode(G, message):
    G = np.array(G)
    x = np.array(message)
    y = np.dot(x, G) % 2
    return y.tolist()

# linear decode 
def linear_decode(H, message):
    H = np.array(H)
    r = np.array(message)
    n = len(r)
    syndrome = np.dot(H, r.T) % 2
    if np.count_nonzero(syndrome) == 0:
        return r[:H.shape[1] - H.shape[0]].tolist()
    for i in range(n):
        e = np.zeros(n, dtype=int)
        e[i] = 1
        new_r = (r + e) % 2
        new_syndrome = np.dot(H, new_r.T) % 2
        if np.count_nonzero(new_syndrome) == 0:
            return new_r[:H.shape[1] - H.shape[0]].tolist()
    return "Error uncorrectable"

### TESTING ###

# Task 1 Example
A = np.array([[0,1,1],[1,1,0],[1,0,1]])
H, G = linear_code(A)
print("H =\n", H)
print("G =\n", G)
print()

# Task 2 Example
A = np.array([[1,1],[1,0],[1,1]])
H, G = linear_code(A)
print("Encoded messages:")
for w in [[0,0],[0,1],[1,0],[1,1]]:
    print(f"{w} -> {linear_encode(G,w)}")
print()

# Task 3 Example
A = np.array([[1,1,1,0,0],
              [1,0,1,1,1],
              [1,1,0,1,0],
              [0,0,1,1,1],
              [0,1,0,0,1]])
H, G = linear_code(A)
words = [[1,0,0,1,0,1,1,1,0,0],
         [0,1,1,0,1,0,0,1,0,0],
         [1,1,0,0,0,0,0,0,1,1],
         [1,1,0,0,1,0,1,0,1,0],
         [1,1,1,0,1,1,1,0,0,0]]
print("Decoded messages:")
for w in words:
    print(f"{w} -> {linear_decode(H, w)}")

### TASK 2 ### 
def linear_code(A):
    A = np.array(A)
    n_minus_m, m = A.shape
    I_n_m = np.identity(n_minus_m, dtype=int)
    I_m = np.identity(m, dtype=int)
    H = np.concatenate((A, I_n_m), axis=1)
    G = np.concatenate((I_m, A.T), axis=1)
    return H, G

def linear_encode(G, message):
    G = np.array(G)
    x = np.array(message)
    y = np.dot(x, G) % 2
    return y.tolist()

# Define A₃ from H₃ = [A | I]
A3 = [
    [1, 1],
    [1, 0],
    [1, 0]
]

# Get H₃ and G₃
H3, G3 = linear_code(A3)

print("Generator Matrix G₃:")
print(np.array(G3))
print()

# All messages in Z2^2
messages = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

print("Encoded Messages using G₃:")
for msg in messages:
    codeword = linear_encode(G3, msg)
    print(f"{msg} → {codeword}")


