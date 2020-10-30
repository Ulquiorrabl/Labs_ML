import sys
import os
import csv
import random
import numpy as np

# if len(sys.argv) < 3:
# 	sys.stdout.write ("3\nNumber of tests\t1000\nNumber of training points for Linear Regression\t100\nNumber of training points for Perceptron\t10\n")
# 	sys.exit(0)
#
# in_parameters = [sys.argv[1], sys.argv[2], sys.argv[3]]
# in_rownumber = int(sys.argv[-2])
# out_filename = sys.argv[-1]
#
# tmp_filename = out_filename + "_"
# tmp_file = open(tmp_filename, "wb")
# tmp_csv = csv.writer (tmp_file, delimiter=',', quotechar='"')




# Create the points in the range [-1,1] X [-1,1],
# return the list formed by point,f(x)
def createExamples(nExamples, m, q):
    N = []
    for i in range(nExamples):
        # [artificial->1,x,y]
        x = [1, random.uniform(-1, 1), random.uniform(-1, 1)]
        # Set the label to +1 or -1
        if x[2] > (m * x[1] + q):
            label = 1
        else:
            label = -1
        # Insert the values into the list of examples
        N.append([x, label])
    return N


def createExampl2(nExamples):
    N = []
    for i in range(nExamples):
        # [artificial->1,x,y]
        x = [1, random.uniform(-1, 1), random.uniform(-1, 1)]
        # Set the label to +1 or -1
        if (x[1] * x[1] + x[2] * x[2] - 0.6) > 0:
            label = 1
        else:
            label = -1
        # Insert the values into the list of examples
        N.append([x, label])
    return N

# Define the h(x) value of a point for a given w
# return +1 or -1
def evaluateg(ex, w):
    r = 0
    for i in range(len(w)):
        r += ex[i] * w[i]

    if r > 0:
        return 1
    elif r < 0:
        return -1


def g1(ex):
    g = -1 - 0.05 * ex[1] + 0.08 * ex[2] + 0.13 * ex[1] * ex[2] + 1.5 * ex[1] * ex[1] + 1.5 * ex[2] * ex[2]
    if g > 0:
        return 1
    else:
        return -1


def g2(ex):
    g = -1 - 0.05 * ex[1] + 0.08 * ex[2] + 0.13 * ex[1] * ex[2] + 1.5 * ex[1] * ex[1] + 15 * ex[2] * ex[2]
    if g > 0:
        return 1
    else:
        return -1


def g3(ex):
    g = -1 - 0.05 * ex[1] + 0.08 * ex[2] + 0.13 * ex[1] * ex[2] + 15 * ex[1] * ex[1] + 1.5 * ex[2] * ex[2]
    if g > 0:
        return 1
    else:
        return -1


def g4(ex):
    g = -1 - 1.5 * ex[1] + 0.08 * ex[2] + 0.13 * ex[1] * ex[2] + 0.05 * ex[1] * ex[1] + 0.05 * ex[2] * ex[2]
    if g > 0:
        return 1
    else:
        return -1


def g5(ex):
    g = -1 - 0.05 * ex[1] + 0.08 * ex[2] + 1.5 * ex[1] * ex[2] + 0.15 * ex[1] * ex[1] + 15 * ex[2] * ex[2]
    if g > 0:
        return 1
    else:
        return -1




# Learning algorith of Linear Regression
# return weights w
def linearRegressionTraining(N):
    # Separate the set of examples into X[][] and y[]
    X, y = zip(*N)
    # Calculate the pseudo-inverse Matrix of X, by using numpy
    X_d = np.linalg.pinv(X)
    # Calculate the vector of weights w as multiplication of X_d and y
    w = np.dot(X_d, y)
    return w


# Calculate the probability of getting a wrong classification
# return the probability of misclassification
def calculateE(points, w):
    X, y = zip(*points)
    nWrong = 0
    # For every point, check if f(x) and h(x) values are egual
    for i in range(len(points)):
        if y[i] != evaluateg(X[i], w):
            # Count the wrong classified cases
            nWrong += 1
    # misclassified points/ total number of points
    pWrong = float(nWrong) / len(points)
    return pWrong


def makeNoise(points, percentage):
    RandomList = []
    while len(RandomList) < len(points) / percentage:
        i = random.randint(0, len(points) - 1)
        if (i in RandomList):
            i = 0
        else:
            RandomList.append(i)
    for i in RandomList:
        points[i][1] = -points[i][1]

    return points


def TransformData(points):
    datas_transformed = []
    for point in points:
        data = []
        data.append(1)
        data.append(point[0][1])
        data.append(point[0][2])
        data.append(point[0][1] * point[0][2])
        data.append(point[0][1] ** 2)
        data.append(point[0][2] ** 2)
        datas_transformed.append((data, point[1]))
    return datas_transformed


# Train the perceptron with w initialized by Linear Regression
# return the number of iterations to linearly separate the points
def perceptronTraining(N, w):
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
            w = w + np.dot(randomPoint[0], randomPoint[1])
    return nIterations
# Parameters
tests = 10
linearExamples = 100
perceptronExamples = 10
m = 0
q = 0
# tmp_csv.writerow (["Ein::number", "Eout::number", "Perceptron Iterations::number"])
Ein = 0
Eout = 0
nIterationsToConverg = 0
for t in range(1, tests + 1):
    # Generate two random points
    x1 = random.uniform(-1, 1)
    y1 = random.uniform(-1, 1)
    x2 = random.uniform(-1, 1)
    y2 = random.uniform(-1, 1)
    # Calculate f
    m = (y2 - y1) / (x2 - x1)
    q = y1 - m * x1
    # Create examples for Linear regression
    NLinearRegression = createExamples(linearExamples, m, q)
    # Train w with Linear Regression
    w = linearRegressionTraining(NLinearRegression)
    # Get Error in samples
    Ein += calculateE(NLinearRegression, w)
    # Create 1000 "out of sample" points
    testPoints = createExamples(1000, m, q)
    # Get out of sample error
    Eout += calculateE(testPoints, w)
    # Create the examples for the Perceptron
    NPerceptron = createExamples(perceptronExamples, m, q)
    # Get the number of iterations for the perceptron to converge.
    # Use weights generated from linear regression as start point for the Perceptron
    nIterationsToConverg += perceptronTraining(NPerceptron, w)
# tmp_csv.writerow ([str(Ein), str(Eout), str(nIterationsToConverg)])

# tmp_file.close()
print("2.Linear regression E_in error:{0} \n3.Linear regression E_out error:{1} \n4.Average PLA iterations:{2}".format(Ein / tests, Eout / tests, nIterationsToConverg / tests)) #2. 3, 4
# os.rename (tmp_filename, out_filename)
GList = [0, 0, 0, 0, 0]
for i in range(tests + 1):
    points = createExampl2(1000)
    testPoints = makeNoise(points, 10)
    w = linearRegressionTraining(testPoints)
    Ein += calculateE(testPoints, w)
    transformPoints = TransformData(testPoints)
    wnew = linearRegressionTraining(transformPoints)
    x1 = random.uniform(-1, 1)
    x2 = random.uniform(-1, 1)
    randompoint = [1, x1, x2, x1 * x2, x1 ** 2, x2 ** 2]
    if evaluateg(randompoint, wnew) == g1(randompoint[1:]):
        GList[0] += 1
    if evaluateg(randompoint, wnew) == g2(randompoint[1:]):
        GList[1] += 1
    if evaluateg(randompoint, wnew) == g3(randompoint[1:]):
        GList[2] += 1
    if evaluateg(randompoint, wnew) == g4(randompoint[1:]):
        GList[3] += 1
    if evaluateg(randompoint, wnew) == g5(randompoint[1:]):
        GList[4] += 1
    testPoints = createExampl2(1000)
    transformPoints = TransformData(testPoints)
    Eout += calculateE(transformPoints, wnew)

print("5.Linear Regression Inclusive errors: {0} \n6.Number of classified examples for g: {2} \n7.Exclusive errors: {1}".format(Ein / tests, Eout / tests, GList))
