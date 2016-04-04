# encoding: utf-8
'''
Created on 2016年3月30日

@author: lml
'''
import numpy
from scipy import misc
import scipy

if __name__ == '__main__':
    a = numpy.array([[0, 1, 2], [3, 4, 5]], float)
    print a
    print type(a)
#     print a[0, :]
#     print a[:, 0]
#     print a.transpose()
#     print a.shape 
#     print a.size
#     print a.ndim
    a = a.reshape(3, 2)
    print a
    o = numpy.eye(3, 3, 1)
    print o
    
    print numpy.dot(a, a.transpose())
    print numpy.random.random((3, 3))
    print numpy.sum(a)
    print numpy.sum(a, axis=0)
    print numpy.sum(a, axis=1)
    print a[0:3, 1]
    print a[:, numpy.newaxis, :]
    for e in a.flat:
        print e,
    print 
    b = a[:, 0]
    b += 1
    print a
    b = a[:, 0].copy()
    b += 1
    print a