import numpy as np

n = int(input("Enter the order of the square matrix: "))

print("Enter the matrix elements row by row:")

matrix = []
for i in range(n):
    row = list(map(float, input().split()))
    matrix.append(row)

A = np.array(matrix)

print("\nMatrix A:")
print(A)


det = np.linalg.det(A)
print("\nDeterminant of Matrix:")
print(det)

if det != 0:
    inverse = np.linalg.inv(A)
    print("\nInverse of Matrix:")
    print(inverse)
else:

    print("\nInverse does not exist (Determinant = 0).")

# Eigen Values and Eigen Vectors
eigen_values, eigen_vectors = np.linalg.eig(A)

print("\nEigen Values:")
print(eigen_values)

print("\nEigen Vectors:")
print(eigen_vectors)