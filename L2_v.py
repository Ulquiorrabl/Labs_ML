import random
import numpy as np


def f_func(x1, x2):
    res = x1 ** 2 + x2 ** 2 - 0.6
    if res < 0:
        return -1
    else:
        return 1


def create_examples_with_noise(n, percent):
    E = []
    for i in range(1, n):
        x = [1, random.uniform(-1, 1), random.uniform(-1, 1)]
        res = f_func(x[0], x[1])
        if random.uniform(0, 1) < percent / 100:
            res *= -1
        E.append([x, res])
    return E


def model_train(array, w):
    for i in range(0, len(array)):
        if np.dot(array[i][0], w) != array[i][1]:
            w += np.dot(array[i][0], array[i][1])
    return w


def predict(x, w):
    dot = np.dot(x, w)
    if dot < 0:
        return -1
    if dot > 0:
        return 1


def accuracy(example_array, w):
    count = 0
    for i in range(0, len(example_array)):
        if example_array[i][1] == predict(example_array[i][0], w):
            count += 1
    return count/len(example_array) * 100


N = 1000
acc = 0
for i in range(0, N):
    w = [0, 0, 0]
    e = create_examples_with_noise(1000, 10)
    w = model_train(e, w)
    acc += accuracy(e, w) / N
print("{0} total accuracy".format(acc))


