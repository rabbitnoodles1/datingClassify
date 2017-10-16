# coding=utf-8
import knnDating


'''
输出文本数据的前三列数据，以及最后一列的标签
'''
datingDataMat, datingLabels = knnDating.file2matrix('datingTestSet2.txt')
print datingDataMat
print datingLabels

'''
输出归一化数值后的数据
输出数据中最大值最小值之差
输出数据中最小值
'''
normMat, ranges, minValues = knnDating.autoNorm(datingDataMat)
print normMat
print ranges
print minValues

'''
测试算法的准确度
'''
knnDating.datingClassifyTest()


