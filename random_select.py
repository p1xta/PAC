import numpy as np
import argparse

def choose(a, b, p):
    if np.random.rand() <= p:
        return b
    return a

def first_way(real, synthetic, P):
    vchoose = np.vectorize(choose)
    return vchoose(real, synthetic, P)
    

def second_way(real, synthetic, P):
    mask = np.random.choice(2, len(real), p=[P, 1-P])
    output = np.where(mask, real, synthetic).tolist()
    return output

parser = argparse.ArgumentParser()
parser.add_argument('input_file', type=str)
parser.add_argument('output_file', type=str)
parser.add_argument('prob', type=float)
args = parser.parse_args()

f1 = open(args.input_file, "r")
f2 = open(args.output_file, "r")
probability = args.prob

real_data = f1.readline().split()
synthetic_data = f2.readline().split()

if (len(real_data) != len(synthetic_data)):
    print("Arrays are different length")
    exit()

real_data = [int(r) for r in real_data]
synthetic_data = [int(s) for s in synthetic_data]

print("First way:")
print(first_way(real_data, synthetic_data, probability))
print("Second way:")
print(second_way(real_data, synthetic_data, probability))

f1.close()
f2.close()