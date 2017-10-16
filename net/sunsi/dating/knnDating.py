# coding=utf-8
from numpy import *
import operator

'''
kNN算法实现分类
'''


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistance = sqDiffMat.sum(axis=1)
    distances = sqDistance ** 0.5
    sortedDistIndex = distances.argsort()
    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedDistIndex[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


'''
从文本文件中解析数据
'''


def file2matrix(filename):
    fr = open(filename)
    arrayOfLines = fr.readlines()
    numberOfLines = len(arrayOfLines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOfLines:
        line = line.strip()
        listOfLine = line.split('\t')
        returnMat[index, :] = listOfLine[0:3]
        classLabelVector.append(int(listOfLine[-1]))
        index += 1
    return returnMat, classLabelVector


'''
归一化特征值，公式为：newValue = (oldValue-min)/(max-min)
'''


def autoNorm(dataSet):
    minValues = dataSet.min(0)
    maxValues = dataSet.max(0)
    ranges = maxValues - minValues
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minValues, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minValues


'''
测试分类效果
'''


def datingClassifyTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minValues = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTest = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTest):
        # 用数据的10%用来测试算法(normMat[i, :], 0 <= 1 <= 99)
        # 用数据的90%用来训练算法（normMat[100:1000, :], datingLabels[100:1000]）
        classifyResult = classify0(normMat[i, :], normMat[numTest:m, :], datingLabels[numTest:m], 3)
        print "测试得到的类型是: %d, 正确的类型是: %d" % (classifyResult, datingLabels[i])
        if classifyResult != datingLabels[i]:
            errorCount += 1.0
    print "总的错误率为: %f" % (errorCount / float(numTest))


'''
构建一个可用的系统
'''


def classifyPerson():
    resultList = ['不感兴趣', '有一点魅力', '有很大魅力']
    playGames = float(raw_input("玩游戏所花费的时间占比是多少？"))
    flierMiles = float(raw_input("每年获取的飞行常客里程数是多少？"))
    iceCream = float(raw_input("每年消费的冰激凌公升数是多少？"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minValues = autoNorm(datingDataMat)
    inArr = array([flierMiles, playGames, iceCream])
    classifyResult = classify0((inArr - minValues) / ranges, normMat, datingLabels, 3)
    print "你可能会喜欢这样的人; ", resultList[classifyResult - 1]
    Q = raw_input("输入任意键继续，输入q退出")
    if Q.lower() == 'q':
        exit(0)
    elif Q.lower() != 'q':
        classifyPerson()
