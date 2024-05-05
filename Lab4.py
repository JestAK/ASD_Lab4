# n1 = 3
# n2 = 2
# n3 = 1
# n4 = 3

import turtle
from turtle import *
import random
import math
from copy import deepcopy

turtle.speed(0)

VERTEX_AMOUNT = 11
VERTEX_RADIUS = 15
FONT_SIZE = 12
FONT = ("Arial", FONT_SIZE, "normal")
SQUARE_SIZE = 300
BREAK_GAP = 10
EXTRA_GAP = 50
K = 1.0 - 1 * 0.01 - 3 * 0.01 - 0.3


def drawVertex(x, y, text):
    up()
    goto(x, y - VERTEX_RADIUS)
    down()
    circle(VERTEX_RADIUS)
    turtle.up()
    turtle.goto(x, y - VERTEX_RADIUS + FONT_SIZE / 2)
    turtle.write(text, align="center", font=FONT)
    down()


def getVertexCoords(vertexAmount, squareSize):
    vertexCoords = []

    squareStartX = -squareSize / 2
    squareStartY = squareSize / 2

    xPos = squareStartX
    yPos = squareStartY

    isXMove = 1
    isYMove = 0

    xDirection = 1
    yDirection = -1

    vertexModulus = vertexAmount % 4
    vertexStep = vertexAmount // 4

    for i in range(4):

        vertexPerSide = vertexStep

        if (vertexModulus > 0):
            vertexPerSide += 1
            vertexModulus -= 1

        if (vertexPerSide > 0): vertexGap = squareSize / vertexPerSide
        else: vertexGap = 0

        for j in range(vertexPerSide):
            vertexCoords.append({"x": round(xPos), "y": round(yPos)})
            xPos += isXMove * xDirection * vertexGap
            yPos += isYMove * yDirection * vertexGap

        xPos = round(xPos)
        yPos = round(yPos)

        if (isXMove):
            isXMove = 0
            isYMove = 1
            xDirection *= -1
        elif (isYMove):
            isYMove = 0
            isXMove = 1
            yDirection *= -1

    return vertexCoords


def generateDirMatrix(vertexAmount, k):
    random.seed(3213)

    dirMatrix = []

    for i in range(vertexAmount):
        row = []

        for j in range(vertexAmount):
            randomNumber = random.uniform(0, 2.0)
            row.append(math.floor(randomNumber * k))

        dirMatrix.append(row)

    return dirMatrix


def dirIntoUndirMatrix(dirMatrix):
    undirMatrix = deepcopy(dirMatrix)

    for i in range(len(undirMatrix)):
        for j in range(len(undirMatrix)):
            if (undirMatrix[i][j] == 1):
                undirMatrix[j][i] = 1

    return undirMatrix


def getFi(vector):
    cosFi = vector[0] / math.sqrt(vector[0] ** 2 + vector[1] ** 2)
    fi = 180 + math.degrees(math.acos(cosFi))
    if (vector[1] < 0): fi = 360 - fi
    return fi


def arrow(startX, startY, endX, endY, fi=None):
    vector = [endX - startX, endY - startY]
    if (fi == None):
        fi = getFi(vector)
    fi = math.pi * fi / 180
    lx = endX + 15 * math.cos(fi + 0.3)
    rx = endX + 15 * math.cos(fi - 0.3)
    ly = endY + 15 * math.sin(fi + 0.3)
    ry = endY + 15 * math.sin(fi - 0.3)

    drawLine(endX, endY, lx, ly)
    drawLine(endX, endY, rx, ry)


def drawLine(startX, startY, endX, endY, withArrow=False):
    up()
    goto(startX, startY)
    down()
    goto(endX, endY)

    if (withArrow == True):
        arrow(startX, startY, endX, endY)


def getOrtVector(startX, startY, endX, endY):
    vector = [endX - startX, endY - startY]
    vectorLenght = math.sqrt(vector[0] ** 2 + vector[1] ** 2)
    ortVector = [vector[0] / vectorLenght, vector[1] / vectorLenght]
    return ortVector


def normVector(ortVector):
    nVector = [-ortVector[1], ortVector[0]]
    return nVector


def createGraph(VERTEX_AMOUNT, VERTEX_RADIUS, SQUARE_SIZE, BREAK_GAP, EXTRA_GAP, dirMatrix, graphType):
    vertexCoords = getVertexCoords(VERTEX_AMOUNT, SQUARE_SIZE)

    matrix = dirMatrix
    withArrow = True

    if (graphType == "undir"):
        undirMatrix = dirIntoUndirMatrix(dirMatrix)

        for row in undirMatrix:
            print(row)

        matrix = undirMatrix
        withArrow = False
    else:
        for row in dirMatrix:
            print(row)

    for i in range(len(vertexCoords)):
        x = vertexCoords[i]["x"]
        y = vertexCoords[i]["y"]
        drawVertex(x, y, str(i + 1))

    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if (matrix[i][j] == 1):
                startX = vertexCoords[i]["x"]
                startY = vertexCoords[i]["y"]
                if (i == j):
                    fi = 360 - getFi([startX, startY])
                    if (((0 < fi) and (fi < 45)) or ((315 < fi) and (fi < 360))):
                        ortVector = [1, 0]
                        setheading(270)
                    elif (((45 < fi) and (fi < 135))):
                        ortVector = [0, 1]
                        setheading(0)
                    elif (((135 < fi) and (fi < 225))):
                        ortVector = [-1, 0]
                        setheading(90)
                    elif (((225 < fi) and (fi < 315))):
                        ortVector = [0, -1]
                        setheading(180)
                    elif (fi == 45):
                        ortVector = [1, 1]
                        setheading(315)
                    elif (fi == 135):
                        ortVector = [-1, 1]
                        setheading(45)
                    elif (fi == 225):
                        ortVector = [-1, -1]
                        setheading(135)
                    elif (fi == 315):
                        ortVector = [1, -1]
                        setheading(225)

                    up()
                    goto(startX + ortVector[0] * VERTEX_RADIUS, startY + ortVector[1] * VERTEX_RADIUS)
                    down()
                    turtle.circle(10)
                    if (graphType == "dir"):
                        arrow(turtle.pos()[0], turtle.pos()[1], turtle.pos()[0], turtle.pos()[1],
                              150 + turtle.heading())

                else:
                    extraGapVector = [0, 0]
                    endX = vertexCoords[j]["x"]
                    endY = vertexCoords[j]["y"]
                    midX = (startX + endX) / 2
                    midY = (startY + endY) / 2
                    ortVector = getOrtVector(startX, startY, endX, endY)

                    if (((startX == endX) or (startY == endY)) and abs(i - j) <= (VERTEX_AMOUNT // 4 + 1)):
                        k = abs(i - j) - 1
                        extraGapOrtVector = getOrtVector(0, 0, midX, midY)
                        extraGapVector = [extraGapOrtVector[0] * EXTRA_GAP * k, extraGapOrtVector[1] * EXTRA_GAP * k]

                    if (matrix[j][i] == 1 and graphType == "dir"):
                        nVector = normVector(ortVector)
                        nVector[0] = nVector[0] * BREAK_GAP
                        nVector[1] = nVector[1] * BREAK_GAP
                        drawLine(startX + ortVector[0] * 15, startY + ortVector[1] * 15,
                                 midX + nVector[0] + extraGapVector[0],
                                 midY + nVector[1] + extraGapVector[1], False)
                        drawLine(midX + nVector[0] + extraGapVector[0], midY + nVector[1] + extraGapVector[1],
                                 endX - ortVector[0] * 15,
                                 endY - ortVector[1] * 15, True)

                    else:
                        drawLine(startX + ortVector[0] * 15, startY + ortVector[1] * 15,
                                 midX + extraGapVector[0],
                                 midY + extraGapVector[1], False)
                        drawLine(midX + extraGapVector[0], midY + extraGapVector[1],
                                 endX - ortVector[0] * 15,
                                 endY - ortVector[1] * 15, withArrow)

        if (graphType == "undir" and i == j):
            break

    turtle.setheading(0)

dirMatrix = generateDirMatrix(VERTEX_AMOUNT, K)
graphType = "dir"
createGraph(VERTEX_AMOUNT, VERTEX_RADIUS, SQUARE_SIZE, BREAK_GAP, EXTRA_GAP, dirMatrix, graphType)


def countVertexDegree(matrix, graphType):
    vertexDegree = []
    matrixLen = len(matrix)

    if graphType == "undir":
        matrix = dirIntoUndirMatrix(matrix)

        for i in range(matrixLen):
            degree = 0
            for j in range(matrixLen):
                if matrix[i][j] == 1:
                    degree += 1
                    if i == j:
                        degree += 1

            vertexDegree.append(degree)
    else:
        for i in range(matrixLen):
            degree = 0
            for j in range(matrixLen):
                if matrix[i][j] == 1: degree += 1
                if matrix[j][i] == 1: degree += 1

            vertexDegree.append(degree)

    return vertexDegree


def countSemiDegree(matrix):
    vertexInputDegree = []
    vertexOutputDegree = []
    matrixLen = len(matrix)

    for i in range(matrixLen):
        inputDegree = 0
        outputDegree = 0
        for j in range(matrixLen):
            if matrix[i][j] == 1: outputDegree += 1
            if matrix[j][i] == 1: inputDegree += 1

        vertexInputDegree.append(inputDegree)
        vertexOutputDegree.append(outputDegree)

    return [vertexInputDegree, vertexOutputDegree]


def regularGraphDegree(vertexDegreeArray):
    for i in range(len(vertexDegreeArray) - 1):
        if vertexDegreeArray[i] != vertexDegreeArray[i + 1]: return 0

    return vertexDegreeArray[0]

dirVertexDegrees = countVertexDegree(dirMatrix, "dir")
undirVertexDegrees = countVertexDegree(dirMatrix, "undir")

print("СТЕПЕНІ ВЕРШИН НАПРЯМЛЕНОГО ГРАФА:")
for index, degree in enumerate(dirVertexDegrees):
    print('\tV{}: {}'.format(index + 1, degree))

print("СТЕПЕНІ ВЕРШИН НЕНАПРЯМЛЕНОГО ГРАФА:")
for index, degree in enumerate(undirVertexDegrees):
    print('\tV{}: {}'.format(index + 1, degree))

print("НАПІВСТЕПЕНІ ВИХОДУ І ЗАХОДУ НАПРЯМЛЕНОГО ГРАФА:")
for index in range(len(dirMatrix)):
    print('\tV{}:'.format(index + 1))
    print('\t\tВиходу: {}\n\t\tЗаходу: {}'.format(countSemiDegree(dirMatrix)[1][index],
                                                  countSemiDegree(dirMatrix)[0][index]))

print("ЧИ Є ГРАФ ОДНОРІДНИМ(РЕГУЛЯРНИМ), ЯКЩО ТАК, ЯКИЙ СТЕПІНЬ ОДНОРІДНОСТІ ГРАФА:")
graphDegree = regularGraphDegree(dirVertexDegrees)
if (graphDegree != 0):
    print('\tГраф однорідний(регулярний) {}-го степеня'.format(graphDegree))
else:
    print('\tГраф НЕ однорідний(НЕ регулярний)')


print("ПЕРЕЛІК ВИСЯЧИХ ТА ІЗОЛЬОВАНИХ ВЕРШИН:")
if not(0 or 1) in dirVertexDegrees: print("\tНемає висячих або ізольованих вершин")
else:
    for index, degree in enumerate(dirVertexDegrees):
        if degree == 0: print('\tV{} — ізольована вершина'.format(index + 1))
        elif degree == 1: print('\tV{} — висяча вершина'.format(index + 1))


class Button:
    def __init__(self, x, y, width, height, label, function, fontSize=12, color="white"):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.label = label
        self.function = function
        self.fontSize = fontSize
        self.color = color

    def drawButton(self):
        up()
        goto(self.x - self.width / 2, self.y - self.height / 2)
        down()
        turtle.fillcolor(self.color)
        turtle.begin_fill()
        for i in range(2):
            turtle.forward(self.width)
            turtle.left(90)
            turtle.forward(self.height)
            turtle.left(90)
        turtle.end_fill()

        up()
        goto(self.x, self.y - self.fontSize/2)
        write(self.label, align="center", font=("Arial", self.fontSize, "normal"))

    def isButtonClicked(self, clickX, clickY):
        return ((self.x - self.width / 2 <= clickX <= self.x + self.width / 2) and
                (self.y - self.height / 2 <= clickY <= self.y + self.height / 2))

    def buttonClickHandler(self, x, y):
        if self.isButtonClicked(x, y):
            self.function()

def matrixMultiply(A, B):
    rowsA = len(A)
    colsA = len(A[0])
    rowsB = len(B)
    colsB = len(B[0])

    result = [[0 for col in range(colsB)] for row in range(rowsA)]

    for i in range(rowsA):
        for j in range(colsB):
            for k in range(colsA):
                result[i][j] += A[i][k] * B[k][j]

    return result

def matrixPower(matrix, power):
    if (power == 1):
        return matrix

    return matrixMultiply(matrix, matrixPower(matrix, power - 1))

def matrixSum(A, B):
    rowsA = len(A)
    colsA = len(A[0])
    rowsB = len(B)
    colsB = len(B[0])

    result = [[0 for col in range(colsB)] for row in range(rowsA)]

    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i][j] = A[i][j] + B[i][j]

    return result

def recursionFindPath(matrix, power, parent, target):
    poweredMatrix = matrixPower(matrix, power)

    pathsArray = []

    for i in range(len(poweredMatrix)):
        if (poweredMatrix[i][target] > 0):
            if (matrix[parent][i] == 1):
                if (power == 1):
                    pathsArray.append([i, target])
                else:
                    endOfPath = recursionFindPath(matrix, power - 1, i, target)
                    for path in endOfPath:
                        new_path = [i] + path
                        pathsArray.append(new_path)

    return pathsArray

def findPathes(matrix, power):
    poweredMatrix = matrixPower(matrix, power)
    pathsArray = []

    for i in range(len(poweredMatrix)):
        for j in range(len(poweredMatrix)):
            if poweredMatrix[i][j] > 0:
                endOfPath = recursionFindPath(matrix, power - 1, i, j)
                for path in endOfPath:
                    new_path = [i] + path
                    pathsArray.append(new_path)

    return pathsArray

def binaryDisplay(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if (matrix[i][j] != 0): matrix[i][j] = 1

    return matrix

def matrixReachability(matrix):

    I = [[0] * len(matrix) for i in range(len(matrix))]

    for i in range(len(matrix)):
        I[i][i] = 1

    answerMatrix = I

    for i in range(len(matrix)):
        answerMatrix = matrixSum(answerMatrix, matrixPower(matrix, i + 1))

    return binaryDisplay(answerMatrix)

def transposeMatrix(matrix):
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]

def multiplyMatrixByElements(A, B):
    result = []
    for i in range(len(A)):
        resultRow = []
        for j in range(len(A[0])):
            resultRow.append(A[i][j] * B[i][j])
        result.append(resultRow)
    return result

def strongConnectivityMartix(matrix):
    return multiplyMatrixByElements(matrix, transposeMatrix(matrix))


def strongConnectivityComponents(matrix):
    indeces = []
    uniqueRows = []

    for index, row in enumerate(matrix):
        if (row not in uniqueRows):
            uniqueRows.append(row)

    for uniqueRow in uniqueRows:
        groupIndeces = []
        for index, row in enumerate(matrix):
            if (row == uniqueRow):
                groupIndeces.append(index)
        indeces.append(groupIndeces)

    return indeces

def condensationGraph(matrix, groups):
    newMatrix = [[0] * len(groups) for i in range(len(groups))]

    for row in range(len(matrix)):
        for column in range(len(matrix[0])):
            if matrix[row][column] == 1:
                fromVertex = row
                toVertex = column
                for index, group in enumerate(groups):
                    if fromVertex in group: fromComponent = index
                    if toVertex in group: toComponent = index
                if (fromComponent != toComponent):
                    newMatrix[fromComponent][toComponent] = 1

    print("МАТРИЦЯ ГРАФА КОНДЕНСАЦІЇ:")

    turtle.clear()
    createGraph(len(groups), VERTEX_RADIUS, SQUARE_SIZE, BREAK_GAP, EXTRA_GAP, newMatrix, "dir")

def newGraph():
    turtle.clear()
    K = 1.0 - 1 * 0.005 - 3 * 0.005 - 0.27

    dirMatrix = generateDirMatrix(VERTEX_AMOUNT, K)
    graphType = "dir"
    createGraph(VERTEX_AMOUNT, VERTEX_RADIUS, SQUARE_SIZE, BREAK_GAP, EXTRA_GAP, dirMatrix, graphType)

    print("НАПІВСТЕПЕНІ ВИХОДУ І ЗАХОДУ НАПРЯМЛЕНОГО ГРАФА:")
    for index in range(len(dirMatrix)):
        print('\tV{}:'.format(index + 1))
        print('\t\tВиходу: {}\n\t\tЗаходу: {}'.format(countSemiDegree(dirMatrix)[1][index],
                                                      countSemiDegree(dirMatrix)[0][index]))

    print("УСІ ШЛЯХИ ДОВЖИНИ 2:")
    pathByLength = findPathes(dirMatrix, 2)
    for index, path in enumerate(pathByLength):
        print('\t{}: V{} -> V{} -> V{}'.format(index + 1, path[0] + 1, path[1] + 1, path[2] + 1))

    print("УСІ ШЛЯХИ ДОВЖИНИ 3:")
    pathByLength = findPathes(dirMatrix, 3)
    for index, path in enumerate(pathByLength):
        print('\t{}: V{} -> V{} -> V{} -> V{}'.format(index + 1, path[0] + 1, path[1] + 1, path[2] + 1, path[3] + 1))

    print("МАТРИЦЯ ДОСЯЖНОСТІ:")
    matrixReach = matrixReachability(dirMatrix)
    for row in matrixReach:
        print(f"\t{row}")

    print("МАТРИЦЯ СИЛЬНОЇ ЗВ'ЯЗНОСТІ:")
    matrixStrongConnectivity = strongConnectivityMartix(matrixReach)
    for row in matrixStrongConnectivity:
        print(f"\t{row}")

    print("ПЕРЕЛІК КОМПОНЕНТІВ СИЛЬНОЇ ЗВ'ЯЗНОСТІ:")
    componentsStrongConnectivity = strongConnectivityComponents(matrixStrongConnectivity)
    for group in componentsStrongConnectivity:
        vertexList = [f"V{x+1}" for x in group]
        print("\t{" + ", ".join(vertexList) + "}")

    condensationGraphButton = Button(-250, 250, 100, 40, "Condensation Graph", lambda: condensationGraph(dirMatrix, componentsStrongConnectivity), 8)
    condensationGraphButton.drawButton()
    turtle.onscreenclick(condensationGraphButton.buttonClickHandler)


newGraphButton = Button(-250, 250, 100, 40, "New Graph", newGraph)
newGraphButton.drawButton()
turtle.onscreenclick(newGraphButton.buttonClickHandler)

turtle.done()

