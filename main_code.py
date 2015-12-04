# Janani Kalyanam
# December 3 2015

import os, sys
import numpy, scipy.sparse, scipy
from matrix_factorization import *

if __name__ == '__main__':

    no_topics = 5; 
    alpha = 0.001; 
    epsilon = 0.001; 
    maxiter = 1000; 
    lambda_ = 10;

    t = 4;
    M = numpy.load('../input/co_occurrence_by_day/K_2014_10_' + str(t) + '.npz');
    K = scipy.sparse.csc_matrix((M['data'],(M['row'],M['col'])),M['shape']);
    print('symNMF on 2014_10_4');
    Qt_1 = symNMF(K,no_topics,alpha,epsilon,maxiter);
    numpy.savez('../input/results_by_day/2014_10_4',QNMF = Qt_1.toarray());

#    for t in range(5,32):
#        M = numpy.load('../input/co_occurrence_by_day/K_2014_10_' + str(t) + '.npz');
#        K = scipy.sparse.csc_matrix((M['data'],(M['row'],M['col'])),M['shape']);
#        print('temporalNMF on 2014_10_' + str(t));
#        (Q,M) = temporalNMF(K,Qt_1,no_topics,alpha,lambda_,epsilon,maxiter);
#        print('symNMF on 2014_10_' + str(t));
#        Qt_1 = symNMF(K,no_topics,alpha,epsilon,maxiter);
#        numpy.savez('../input/results_by_day/2014_10_' + str(t),Q = Q, M = M, QNMF = Qt_1);
