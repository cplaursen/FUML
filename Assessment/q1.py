#!/usr/bin/env python3

def probMLE(x, val=0):
    return sum(1 for i in x if i == val) / len(x)

def probBayes(x, val=0):
    return (sum(1 for i in x if i == val) + 1) / (len(x) + 2)

def probBayesCond(x, y, yval=0, xval=0):
    r = sum(1 for i in range(len(x)) if x[i] == xval and y[i] == yval)
    n = sum(1 for i in y if i == yval)
    alpha = 1 + r
    beta = 1 + n - r
    return alpha/(alpha + beta)

def probJoint(x, y, yval=0, xval=0):
    return sum(1 for i in range(len(x)) if x[i] == xval and y[i] == yval) / len(x)

def probMLECond(x, y, yval=0, xval=0):
    pXY = probJoint(x, y, yval, xval)
    return pXY / probMLE(y, yval)


def main():
    with open("discrete.csv") as file:
        headers = file.readline().strip().split(',')
        data_lines = file.read().strip().split('\n')
        data = {i:[] for i in headers}
        for line in data_lines:
            for n, i in enumerate(line.strip().split(',')):
                data[headers[n]].append(int(i))
    formatAnswer(data)

def formatAnswer(data):
    print(f'''MLE:
P(Y=0) = {round(probMLE(data['Y'], 0), 4)}
P(X1=0|Y=0) = {round(probMLECond(data['X1'], data['Y'], 0), 4)}
P(X1=0|Y=1) = {round(probMLECond(data['X1'], data['Y'], 1), 4)}
P(X2=0|Y=0) = {round(probMLECond(data['X2'], data['Y'], 0), 4)}
P(X2=0|Y=1) = {round(probMLECond(data['X2'], data['Y'], 1), 4)}
P(X3=0|Y=0) = {round(probMLECond(data['X3'], data['Y'], 0), 4)}
P(X3=0|Y=1) = {round(probMLECond(data['X3'], data['Y'], 1), 4)}

Bayesian:
P(Y=0) = {round(probBayes(data['Y'], 0), 4)}
P(X1=0|Y=0) = {round(probBayesCond(data['X1'], data['Y'], 0), 4)}
P(X1=0|Y=1) = {round(probBayesCond(data['X1'], data['Y'], 1), 4)}
P(X2=0|Y=0) = {round(probBayesCond(data['X2'], data['Y'], 0), 4)}
P(X2=0|Y=1) = {round(probBayesCond(data['X2'], data['Y'], 1), 4)}
P(X3=0|Y=0) = {round(probBayesCond(data['X3'], data['Y'], 0), 4)}
P(X3=0|Y=1) = {round(probBayesCond(data['X3'], data['Y'], 1), 4)}''')

if __name__ == "__main__":
    main()
