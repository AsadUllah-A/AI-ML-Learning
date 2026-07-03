import numpy as np

# ==========================================
# Linear Algebra Mini Project
# Student Marks Analyzer
# ==========================================

# 1. Marks Matrix: 5 students and 3 subjects
marks = np.array([
    [50, 80, 47],
    [34, 78, 90],
    [56, 79, 80],
    [34, 79, 90],
    [98, 76, 78]
])

print("\n========== Student Marks Matrix ==========")
print(marks)

# 2. Matrix shape
print(f"\nMatrix Shape: {marks.shape}")


# ==========================================
# Vector Operations: Student 1 and Student 2
# ==========================================

student1 = marks[0]
student2 = marks[1]

# Vector addition
v_add = student1 + student2

# Vector subtraction
v_sub = student1 - student2

# Dot product
dot_p = student1 @ student2

# Distance between Student 1 and Student 2
distance = np.linalg.norm(student1 - student2)

print("\n========== Vector Operations ==========")
print(f"S1 & S2 Vector Addition: {v_add}")
print(f"S1 & S2 Vector Subtraction: {v_sub}")
print(f"S1 & S2 Dot Product: {dot_p}")
print(f"S1 & S2 Distance: {round(distance, 2)}")


# ==========================================
# Norm of Each Student
# ==========================================

print("\n========== Norm of Each Student ==========")

for i, student in enumerate(marks):
    norm = np.linalg.norm(student)
    print(f"Norm of Student {i + 1}: {round(norm, 2)}")


# ==========================================
# Grace Marks
# ==========================================

grace_marks = 5

# Scalar addition
new_marks = marks + grace_marks

print("\n========== Grace Marks ==========")
print(f"Grace Marks Added: {grace_marks}")
print(new_marks)


# ==========================================
# Transpose Matrix
# ==========================================

marks_t = marks.T

print("\n========== Marks Transpose ==========")
print(marks_t)


# ==========================================
# Weighted Scores
# ==========================================

# Subject weights: Math, Science, English
weights = np.array([
    [0.40],
    [0.35],
    [0.25]
])

# Matrix multiplication
weighted_scores = marks @ weights

print("\n========== Weighted Scores ==========")
print(weighted_scores)


# ==========================================
# Determinant, Inverse, Identity Check
# ==========================================

A = np.array([
    [45, 54],
    [64, 13]
])

det_A = np.linalg.det(A)
inv_A = np.linalg.inv(A)
iden_check = A @ inv_A

print("\n========== Matrix A Analysis ==========")
print(f"Matrix A:\n{A}")
print(f"\nDeterminant of A: {round(det_A, 2)}")
print(f"\nInverse of A:\n{inv_A}")
print(f"\nIdentity Check A @ A^-1:\n{np.round(iden_check, 2)}")


# ==========================================
# Linear Equations using np.linalg.solve()
# ==========================================

# Equations:
# 2x + y = 3
# 5x + 3y = -1

B = np.array([
    [2, 1],
    [5, 3]
])

C = np.array([3, -1])

solution = np.linalg.solve(B, C)

print("\n========== Linear Equation Solution ==========")
print(f"x = {round(solution[0], 2)}")
print(f"y = {round(solution[1], 2)}")


# ==========================================
# Eigenvalues and Eigenvectors
# ==========================================

eigenvalues, eigenvectors = np.linalg.eig(B)

print("\n========== Eigenvalues and Eigenvectors ==========")
print(f"Eigenvalues:\n{eigenvalues}")
print(f"\nEigenvectors:\n{eigenvectors}")


# ==========================================
# Verify Av = λv
# ==========================================

lamda = eigenvalues[0]
v = eigenvectors[:, 0].reshape(2, 1)

left = B @ v
right = lamda * v

print("\n========== Verify Av = λv ==========")

if np.allclose(left, right):
    print("Av = λv Verified")
else:
    print("Av = λv Not Verified")


# ==========================================
# SVD: Singular Value Decomposition
# ==========================================

U, S, Vt = np.linalg.svd(B)

print("\n========== SVD ==========")
print(f"U:\n{U}")
print(f"\nSingular Values S:\n{S}")
print(f"\nVt:\n{Vt}")


# ==========================================
# Reconstruct Matrix using U Σ Vt
# ==========================================

sigma = np.zeros((B.shape[0], B.shape[1]))
np.fill_diagonal(sigma, S)

reconstruct = U @ sigma @ Vt

print("\n========== SVD Reconstruction ==========")
print(f"Reconstructed Matrix:\n{np.round(reconstruct, 2)}")

if np.allclose(B, reconstruct):
    print("Reconstruction Successful!")
else:
    print("Reconstruction Not Successful!")


# ==========================================
# PCA Basic Idea
# ==========================================

print("\n========== PCA Basic Idea ==========")
print("PCA is used for dimensionality reduction.")
print("It reduces features while keeping important information.")
print("Example: 3 subjects/features can be reduced to 2 important components.")