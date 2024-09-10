import argparse

def read_matrix(file, matrix):
    while True:
        line = file.readline()
        if len(line) == 0 or line == '\n': 
            break
        nums = line.split()
        nums = [int(num) for num in nums]
        matrix.append(nums)

def check_matrix(file, matrix):
    prev_row_lenght = len(matrix[0])
    for row in matrix:
        if len(row) != prev_row_lenght:
            file.write("Matrices are invalid")
            return False
        else:
            prev_row_lenght = len(row)
    return True

def multiply_matrices(matrix1, matrix2):
    result = [[0 for i in range(len(matrix2[0]))] for j in range(len(matrix1))]
    for i in range(len(matrix1)):
        for j in range(len(matrix2[0])):
            for k in range(len(matrix2)):
                result[i][j] += matrix1[i][k] * matrix2[k][j]
    return result 

parser = argparse.ArgumentParser()
parser.add_argument('input_file', type=str)
parser.add_argument('output_file', type=str)
args = parser.parse_args()

fin = open(args.input_file, "r")
fout = open(args.output_file, "w")

matrix_1 = []
matrix_2 = []

read_matrix(fin, matrix_1)
read_matrix(fin, matrix_2)

if (len(matrix_1[0]) != len(matrix_2)):
    fout.write(f"Matrices can't be multipied ({len(matrix_1)}x{len(matrix_1[0])} and {len(matrix_2)}x{len(matrix_2[0])})")
    exit(1)

if check_matrix(fout, matrix_1) and check_matrix(fout, matrix_2):
    res = multiply_matrices(matrix_1, matrix_2)

    for i in range(len(res)):
        for j in range(len(res[0])):
            fout.write(f"{res[i][j]} ")
        fout.write("\n")

fin.close()
fout.close()