import numpy as np

row = int(input("Enter the number of rows: "))
col = int(input("Enter the number of columns: "))

if row != col:
    print("Trace is possible only for a square matrix.")
else:
    print("Enter", row * col, "elements:")
elements = list(map(int, input().split()))
if len(elements) != row * col:
    print("Invalid number of elements!")
else:

    matrix = np.array(elements).reshape(row, col)
print("\nMatrix:")
print(matrix)
# Find trace
trace = np.trace(matrix)
print("\nTrace of the Matrix =", trace)