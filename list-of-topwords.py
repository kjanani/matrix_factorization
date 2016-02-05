# Janani Kalyanam
# Oct 30, 2015

import os, sys
import numpy, scipy, scipy.sparse
from matrix_factorization import *


if __name__ == '__main__':
    no_topics = 5;

    fid = open('words.txt','w');
    fid_temp = open('temp.txt','w');
    for d in range(5,31):
        print(d)
        Q_lda =  numpy.loadtxt('/mnt/spare/users/janani/Research/Ebola/scripts/input/lda_input_by_day/output_2014_10_'+str(d)+'/Q'+str(no_topics)+'.txt');
        F = numpy.load('/mnt/spare/users/janani/Research/Ebola/scripts/input/results_by_day/2014_10_'+str(d)+'_'+str(no_topics)+'.npz');
        Q = F['Q']; M = F['M']; Qt_1 = F['QNMF'];

        words_lda = topWords(scipy.sparse.csc_matrix(Q_lda));
        words_symNMF = topWords(scipy.sparse.csc_matrix(Qt_1));
        words_tempNMF = topWords(scipy.sparse.csc_matrix(Q));
        print(words_tempNMF)
        fid.write('2014_10_'+str(d)+'\n\n');
        for jj in range(len(words_lda[0])):
            for ii in range(len(words_lda)):
                fid.write(words_lda[ii][jj] + '\t');
                
            fid.write('\t\t')
            for ii in range(len(words_lda)):
                fid.write(words_symNMF[ii][jj] + '\t');

            fid.write('\t\t')
            for ii in range(len(words_lda)):
                fid.write(words_tempNMF[ii][jj] + '\t');

            fid.write('\n');

    fid.close();
