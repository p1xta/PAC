import argparse
import random

def BubbleSort(N: int) -> list[int]:
    lst = [random.random() for i in range(N)]

    for i in range (N):
        for j in range(N):
            if lst[i] < lst[j]:
                lst[i], lst[j] = lst[j], lst[i]
    return lst

parser = argparse.ArgumentParser()
parser.add_argument('n', type=int)
args = parser.parse_args()
print(BubbleSort(int(args.n)))

