import numpy as np
import random


def createExamples(nExamples, m, q):
    N = []  # массив в котором координаты и метки
    for i in range(nExamples):

        # [artificial->1,x,y]
        x = [1, random.uniform(-1, 1),
             random.uniform(-1, 1)]  # введена искуственная переменная 1 для линейного разделения данных
        # Set the label to +1 or -1
        if x[2] > (m * x[1] + q):  # подставляем x в уравнение прямой если значение x[2]>y, то выше прямой
            label = 1
        else:
            label = -1
        N.append((x, label))  # значения вставляются в список

    print(N)
    return N


# список содержит координаты x, y и значение метки

def evaluateg(Points, w):  # вычисление меток (результат)
    r = 0
    for i in range(len(w)):
        r = r + Points[i] * w[i]  # по формуле

    if r > 0:
        return 1
    elif r <= 0:
        return - 1


w = [0, 0, 0]


def perceptronTraining(Points, w):
    misclassifiedPoints = True
    nIterations = 0
    while misclassifiedPoints:  # цикл пока остаются неправильно классифицированные точки
        misclassifiedPointsList = []  #
        misclassifiedPoints = False
        nIterations += 1
        for point in Points:
            res = evaluateg(point[0], w)  # вычисление метки точки
            if res != point[1]:  # если вычисленная метка не совпадает с настоящей
                misclassifiedPointsList.append(point)  # эта точка добавляется в список
                misclassifiedPoints = True  # есть неклассифицированная точка
        if misclassifiedPointsList:  # если список не пуст
            randomPoint = misclassifiedPointsList[random.randint(0, len(misclassifiedPointsList) - 1)]
            # выбор случайного элемента из списка
            for i in range(len(w)):
                w[i] = w[i] + randomPoint[1] * randomPoint[0][i]  # пересчёт весов по формуле

    return nIterations, w


def Err(testPoints, w):  # количество ошибок классификации
    er_points = 0
    for point in testPoints:
        res = evaluateg(point[0], w)  # point[0] - 1, x, y, нахождение метки с помощью весов полученных от персептрона
        if res != point[1]:  # сравнение с настоящей меткой
            er_points += 1
    return er_points / 10000


x1 = random.uniform(-1, 1)
y1 = random.uniform(-1, 1)
x2 = random.uniform(-1, 1)
y2 = random.uniform(-1, 1)
m = (y2 - y1) / (x2 - x1)  # k тангенс угла наклона
q = y1 - m * x1  # b для уравнения прямой
# уравнение прямой y=mx+q
nExamples = 100  # количество примеров
AvrErr = 0  # количество неправильно определённых точек / на количество всех точек
AvrIter = 0  # количество итераций
for i in range(1000):
    w = [0, 0, 0]
    Points = createExamples(nExamples, m, q)  # создание обучающей выборки
    iter, w = perceptronTraining(Points, w)  # обучение модели
    TestPoints = createExamples(10000, m, q)  # создание тестовой выборки
    AvrErr += Err(TestPoints, w)  # проверка результатов и подсчёт количества ошибок
    AvrIter += iter  # количество итераций

print(AvrErr / 1000, AvrIter / 1000)