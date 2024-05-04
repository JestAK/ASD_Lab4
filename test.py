def matrixMultiply(A, B):
    rowsA = len(A)
    colsA = len(A[0])
    rowsB = len(B)
    colsB = len(B[0])

    result = [[0 for col in range(colsB)] for row in range(rowsA)]

    for i in range(rowsA):
        for j in range(colsB):
            for k in range(colsA):
                result[i][j] += A[i][k] * B[k][j]

    return result

def matrixPower(matrix, power):
    if (power == 1):
        return matrix

    return matrixMultiply(matrix, matrixPower(matrix, power - 1))

matrix1 = [[1, 2, 3],
           [4, 5, 6],
           [4, 5, 6]]
matrix2 = [[7, 8],
           [9, 10],
           [11, 12]]


result_matrix = matrixPower(matrix1, 3)
print("Result of matrix power:")
for row in result_matrix:
    print(row)



result_matrix = matrixMultiply(matrix1, matrix2)
print("Result of matrix multiplication:")
for row in result_matrix:
    print(row)