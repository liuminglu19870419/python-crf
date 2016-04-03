# encoding: utf-8
'''
Created on 2016年3月30日

@author: lml
'''
import numpy
from scipy import misc

if __name__ == '__main__':
    a = numpy. array([1, 2])
    b = numpy.array([[3, 4], [6, 5]])
    k = a + b
    print k
    kT = k.transpose()
    print k
    print k[0]
    print k[1]
    x = numpy.sum(numpy.exp(k[0]))
    print x
    y = numpy.sum(numpy.exp(k[1]))
    print y
    print numpy.log(x)
    print numpy.log(y)
    print misc.logsumexp(k, axis=0)
    print misc.logsumexp(kT, axis=1)
    print numpy.newaxis
    print k[:, numpy.newaxis]
    print k[:, numpy.newaxis]
