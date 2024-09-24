
class Pupa:
    def __init__(self) -> None:
        self.money = 0

    def take_salary(self, amount : int):
        self.money += amount
    
    def do_work(self, filename1, filename2):
        file1 = open(filename1, "r")
        file2 = open(filename2, "r")

        matrix_1 = []
        matrix_2 = []

        read_matrix(file1, matrix_1)
        read_matrix(file2, matrix_2)

        for i in range(len(matrix_1)):
            for j in range(len(matrix_1[0])):
                matrix_1[i][j] += matrix_2[i][j]
                print(f"{matrix_1[i][j]} ", end='')
            print()

        file1.close()
        file2.close()

class Lupa:
    def __init__(self) -> None:
        self.money = 0
    
    def take_salary(self, amount : int):
        self.money += amount
    
    def do_work(self, filename1, filename2):
        file1 = open(filename1, "r")
        file2 = open(filename2, "r")

        matrix_1 = []
        matrix_2 = []

        read_matrix(file1, matrix_1)
        read_matrix(file2, matrix_2)

        for i in range(len(matrix_1)):
            for j in range(len(matrix_1[0])):
                matrix_1[i][j] -= matrix_2[i][j]
                print(f"{matrix_1[i][j]} ", end='')
            print()

        file1.close()
        file2.close()

class Accountant:
    def give_salary(self, worker, amount):
        worker.take_salary(amount)


def read_matrix(file, matrix):
    while True:
        line = file.readline()
        if len(line) == 0 or line == '\n': 
            break
        nums = line.split()
        nums = [int(num) for num in nums]
        matrix.append(nums)

pupa = Pupa()
lupa = Lupa()
accountant = Accountant()

pupa.do_work("file1.txt", "file2.txt")
print()
lupa.do_work("file1.txt", "file2.txt")

accountant.give_salary(pupa, 1000)
accountant.give_salary(lupa, 700)

print(f"Pupa's $$$: {pupa.money}\n")
print(f"Lupa's $$$: {lupa.money}")

accountant.give_salary(lupa, 301)
print(f"Lupa's $$$: {lupa.money}")
