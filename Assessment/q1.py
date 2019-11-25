#!/usr/bin/env python3

def probMLE(x, val = 0):
    return sum(1 for i in x if i==val) / len(x)

def probBayes(x, val = 0):
    return (sum(1 for i in x if i==val) + 1) / (len(x) + 1)

def main():
    with open("discrete.csv") as file:
        headers = file.readline().strip().split(',')
        data_lines = file.read().strip().split('\n')
        data = {i:[] for i in headers}
        for line in data_lines:
            print(line)
            for n, i in enumerate(line.strip().split(',')):
                data[headers[n]].append(int(i))
        print(probMLE(data['Y'], 1))
        print(probBayes(data['Y'], 1))
        print(probMLECond(data['X1'], data['Y']))


if __name__ == "__main__":
    main()
