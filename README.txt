**************************** DEMO *********************************
1.  Open terminal and invoke python
2.  load ipython and type the following:

>> import os, sys
>> import numpy, scipy, scipy.sparse
>> from matrix_factorization import *
>>
>>
>> no_topics = 5; alpha = 0.001; epsilon = 0.001; maxiter = 1000; lambda_ = 10;
>> fid = open('words.txt','w');
>>
>>
>> file = numpy.load('data/K_2014_10_5.npz');
>> K = scipy.sparse.csc_matrix((file['data'],(file['row'],file['col'])),file['shape']);
>>
>> Qt_1 = symNMF(K,no_topics,alpha,epsilon,maxiter);
>> topwords(Qt_1,fid);
>>
>>
>> file = numpy.load('data/K_2014_10_6.npz');
>> K = scipy.sparse.csc_matrix((file['data'],(file['row'],file['col'])),file['shape']);
>> (Q,M) = temporalNMF(K,Qt_1,no_topics,alpha,lambda_,epsilon,maxiter)
******************************************************************

symNMF:

- Factorizes a matrix as K = QTQ.
- K is typically a symmetric matrix (all positive matrix) representing
    some notion of similarity between entities (i,j).
- It optimizes for the loss (||K - QTQ||_F)^2 - \alpha * ||Q||_1
- It does so by considering Q, and QT as two different matrices.  Given Q, find QT - this is the least squares problem.
- Given QT, find Q --> using multiplicative updates.


temporalNMF:

- Objective Function:  (||K - QTQ||_F)^2 + (||K - QTMQt_1||_F)^2  + \alpha * (||Q||_1 + ||M||_1 ) + \lambda_ (||M - I||_2)^2
- Use only (||K - XTQ||_F )^2 to find X.  Then, with XT as known, find Q and M using multiplicative updates.
