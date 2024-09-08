import argparse 

def BinomialCoef(n : int, k: int) -> int:
    res = 1 
    if k > n-k:
        k = n-k
    for i in range(k):
        res *= (n - i)
        res //= (i + 1)
    return res

def PascalTriangle(height):
    for n in range(height):
        print(" " * (height-n), end=" ")
        for k in range(n+1):
            print(f"{BinomialCoef(n, k)} ", end=" ")
        print()

parser = argparse.ArgumentParser()
parser.add_argument('num', type=int)
args = parser.parse_args()
PascalTriangle(args.num)