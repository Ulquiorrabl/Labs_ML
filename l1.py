import numpy as np
import random


def create_examples(n_examples, m, q):
    n = []
    for i in range(nExamples):

        # [artificial->1,x,y]
        x = [1, random.uniform(-1, 1), random.uniform(-1, 1)]
        # Set the label to +1 or -1
        if x[2] > (m * x[1] + q):
            label = 1
        else:
            label = -1
        # Insert the values into the list of examples

        n.append((x, label))
    return n


# Define the h(x) value of a point for a given w
# return +1 or -1
def evaluateg(ex, w):
    r = 0
    for i in range(len(w)):
        r = r + ex[i] * w[i]

    if r > 0:
        return 1
    elif r <= 0:
        return - 1


w = [0, 0, 0]


def perceptron_training(N, w):
    misclassifiedPoints = True
    nIterations = 0
    while misclassifiedPoints:
        misclassifiedPointsList = []
        misclassifiedPoints = False
        nIterations += 1
        for ex in N:
            res = evaluateg(ex[0], w)
            if res != ex[1]:
                misclassifiedPointsList.append(ex)
                misclassifiedPoints = True
        if misclassifiedPointsList:
            randomPoint = misclassifiedPointsList[random.randint(0, len(misclassifiedPointsList) - 1)]
            for i in range(len(w)):
                w[i] = w[i] + randomPoint[1] * randomPoint[0][i]

    return nIterations, w


def Err(testPoints, w):
    er_points = 0
    for ex in testPoints:
        res = evaluateg(ex[0], w)
        if res != ex[1]:
            er_points += 1
    return er_points / 10000


x1 = random.uniform(-1, 1)
y1 = random.uniform(-1, 1)
x2 = random.uniform(-1, 1)
y2 = random.uniform(-1, 1)
m = (y2 - y1) / (x2 - x1)
q = y1 - m * x1
nExamples = 10
AvrErr = 0
AvrIter = 0
for i in range(1000):
    w = [0, 0, 0]
    Points = create_examples(nExamples, m, q)
    iter, w = perceptron_training(Points, w)
    TestPoints = create_examples(10000, m, q)
    AvrErr += Err(TestPoints, w)
    AvrIter += iter
print(AvrErr / 1000, AvrIter / 1000)
