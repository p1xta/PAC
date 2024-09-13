import argparse

def read_matrix(file, matrix):
    while True:
        line = file.readline()
        if len(line) == 0 or line == '\n': 
            break
        nums = line.split()
        nums = [int(num) for num in nums]
        matrix.append(nums)

def matrix_conv(matrix1, matrix2):  # matrix 2 should be a square matrix
    kernel_center = len(matrix2) // 2  
    result = [[0 for i in range(len(matrix1[0]))] for j in range(len(matrix1))]
    for i in range(kernel_center, len(matrix1)-kernel_center):
        for j in range(kernel_center, len(matrix1[0])-kernel_center):
            conv_val = 0
            for ki in range(-kernel_center, kernel_center+1):
                for kj in range(-kernel_center, kernel_center+1):
                    conv_val += matrix1[i+ki][j+kj] * matrix2[ki+kernel_center][kj+kernel_center]
            result[i][j] = conv_val
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

res = matrix_conv(matrix_1, matrix_2)
kernel_center = len(matrix_2) // 2
for i in range(kernel_center, len(res)-kernel_center):
    for j in range(kernel_center, len(res[0])-kernel_center):
        fout.write(f"{res[i][j]} ")
    fout.write("\n")

fin.close()
fout.close()