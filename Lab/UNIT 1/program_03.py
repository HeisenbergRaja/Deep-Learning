# Program 3: Encryption matrix
# Question: Calculate determinant, inverse, and rank of encryption matrix, determine suitability.

import numpy as np

# ----------------------------------------------------------------------
# 1. Define Candidate Encryption Key Matrices
# ----------------------------------------------------------------------
# Candidate 1: Key Matrix A (Invertible, full-rank 3x3 matrix)
key_A = np.array([
    [3, 2, 7],
    [2, 5, 1],
    [4, 1, 6]
], dtype=float)

# Candidate 2: Key Matrix B (Singular 3x3 matrix where Row 3 = Row 1 + Row 2)
key_B = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [5, 7, 9]
], dtype=float)

# ----------------------------------------------------------------------
# 2. Evaluation Function for Encryption Suitability
# ----------------------------------------------------------------------
def analyze_encryption_matrix(K, name):
    print("=" * 65)
    print(f"EVALUATING ENCRYPTION KEY MATRIX: {name}")
    print("=" * 65)
    print("Matrix:")
    print(K)
    
    n_rows, n_cols = K.shape
    is_square = (n_rows == n_cols)
    print(f"\n1. Dimensions        : {n_rows}x{n_cols} (Square Matrix: {is_square})")
    
    if not is_square:
        print("   Status            : UNSUITABLE (Encryption key must be square).")
        return
        
    # Calculate Determinant
    det = np.linalg.det(K)
    print(f"2. Determinant (det) : {det:.4f}")
    
    # Calculate Rank
    rank = np.linalg.matrix_rank(K)
    full_rank = (rank == n_rows)
    print(f"3. Matrix Rank       : {rank} (Full Rank: {full_rank})")
    
    # Check Invertibility & Suitability
    if np.abs(det) > 1e-5 and full_rank:
        inv_K = np.linalg.inv(K)
        print("4. Matrix Inverse    : Inverse EXISTS")
        print(np.round(inv_K, 4))
        
        # Verify K @ inv_K == Identity
        is_identity = np.allclose(K @ inv_K, np.eye(n_rows))
        print(f"5. Identity Check (K @ K^-1 == I) : {is_identity}")
        
        # Encryption & Decryption Demonstration on Sample Vector
        plaintext_vector = np.array([8.0, 15.0, 21.0])
        ciphertext_vector = K @ plaintext_vector
        decrypted_vector = inv_K @ ciphertext_vector
        
        print("\n--- Encryption / Decryption Demonstration ---")
        print(f"   Plaintext Vector (P)        : {plaintext_vector}")
        print(f"   Ciphertext Vector (C = K*P) : {np.round(ciphertext_vector, 2)}")
        print(f"   Decrypted Vector (P = K^-1*C): {np.round(decrypted_vector, 2)}")
        print(f"   Perfect Recovery Check      : {np.allclose(plaintext_vector, decrypted_vector)}")
        
        print("\nSUITABILITY VERDICT: SUITABLE")
        print("Reason: Matrix is square, has non-zero determinant, full rank, and a valid")
        print("unique inverse, guaranteeing lossless decryption.")
    else:
        print("4. Matrix Inverse    : DOES NOT EXIST (Matrix is singular / det = 0)")
        print("\nSUITABILITY VERDICT: UNSUITABLE")
        print("Reason: Determinant is 0 and rank is deficient. Linear dependency causes irreversible")
        print("information loss during encryption; decryption is mathematically impossible.")

# Run evaluation on both candidate matrices
analyze_encryption_matrix(key_A, "Key A (Invertible Key)")
analyze_encryption_matrix(key_B, "Key B (Singular Key)")

# Output:
# =================================================================
# EVALUATING ENCRYPTION KEY MATRIX: Key A (Invertible Key)
# =================================================================
# Matrix:
# [[3. 2. 7.]
#  [2. 5. 1.]
#  [4. 1. 6.]]
#
# 1. Dimensions        : 3x3 (Square Matrix: True)
# 2. Determinant (det) : -55.0000
# 3. Matrix Rank       : 3 (Full Rank: True)
# 4. Matrix Inverse    : Inverse EXISTS
# [[-0.5273  0.0909  0.6   ]
#  [ 0.1455  0.1818 -0.2   ]
#  [ 0.3273 -0.0909 -0.2   ]]
# 5. Identity Check (K @ K^-1 == I) : True
#
# --- Encryption / Decryption Demonstration ---
#    Plaintext Vector (P)        : [ 8. 15. 21.]
#    Ciphertext Vector (C = K*P) : [201. 112. 173.]
#    Decrypted Vector (P = K^-1*C): [ 8. 15. 21.]
#    Perfect Recovery Check      : True
#
# SUITABILITY VERDICT: SUITABLE
# Reason: Matrix is square, has non-zero determinant, full rank, and a valid
# unique inverse, guaranteeing lossless decryption.
# =================================================================
# EVALUATING ENCRYPTION KEY MATRIX: Key B (Singular Key)
# =================================================================
# Matrix:
# [[1. 2. 3.]
#  [4. 5. 6.]
#  [5. 7. 9.]]
#
# 1. Dimensions        : 3x3 (Square Matrix: True)
# 2. Determinant (det) : 0.0000
# 3. Matrix Rank       : 2 (Full Rank: False)
# 4. Matrix Inverse    : DOES NOT EXIST (Matrix is singular / det = 0)
#
# SUITABILITY VERDICT: UNSUITABLE
# Reason: Determinant is 0 and rank is deficient. Linear dependency causes irreversible
# information loss during encryption; decryption is mathematically impossible.

# ======================================================================
