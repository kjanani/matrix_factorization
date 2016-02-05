# Janani Kalyanam
# November 18 2015

import os, sys
import numpy, scipy, scipy.sparse



def tr(A,B):
    '''
    Computes the trace of A*B.
    '''
    S_A = numpy.reshape(A.toarray(),A.shape[0]*A.shape[1]); # vector
    S_B = numpy.reshape(B.toarray(),B.shape[0]*B.shape[1]); # vector
    return numpy.multiply(S_A,S_B).sum() # element-wise multiply and then sum


def trAA(A):
    return (A**2).sum()


def computeLoss_symNMF(K,Q,trKK,alpha):


    loss = trKK;

    QQT = Q.dot(Q.transpose());
    QK = Q.dot(K);
    QKQT = QK.dot(Q.transpose());

    loss = loss + QQT.dot(QQT).diagonal().sum();
    loss = loss - 2*QKQT.diagonal().sum();
    loss = loss + alpha * Q.sum(); 

    return loss;

def computeLoss_temporalNMF(K,Q,Qt_1,M,trKK,Qt_1Qt_1T,lambda_,alpha):
    
    loss = trKK;

    loss = loss + trAA(M)*lambda_;

    QQT = Q.dot(Q.transpose());
    loss = loss + trAA(QQT);

    QK = Q.dot(K);
    loss = loss - 2*tr(QK.transpose(),Q);

    Qt_1Qt_1TM = Qt_1Qt_1T.dot(M);
    loss = loss + Qt_1Qt_1TM.dot(QQT).diagonal().sum();

    MTQK = M.transpose().dot(QK);
    loss = loss - 2*tr(Qt_1,MTQK);

    loss = loss + alpha * (Q.sum() + M.sum());

    return loss;

    
    
def temporalNMF(K,Qt_1,no_topics,alpha,lambda_,epsilon,maxiter,my_seed):
    '''
    ||K - (Q^T)Q|| + ||K - Q^TMQt_1|| + \alpha * (||Q||_1 + ||M||_1)
    Solve ||K - (Q^T)X|| wrt X first (by Least Squares)
    Then solve ||K - (A^T)Q|| + ||K - Q^TMQt_1|| wrt A (mult updates) (Here Q = X from before)
    '''
    numpy.random.seed(my_seed);
    Q = scipy.sparse.rand(no_topics,K.shape[0],density=1);
    Q = numpy.absolute(Q);

    M = scipy.sparse.rand(no_topics,no_topics,density=1);
    M = numpy.absolute(M);
    
    iter_no = 0;
    prevLoss = 100000;
    currentLoss = 2*prevLoss;

    trKK = trAA(K);
    Qt_1Qt_1T = Qt_1.dot(Qt_1.transpose());
    Qt_1KT = Qt_1.dot(K.transpose());
    while(numpy.absolute(currentLoss - prevLoss) > epsilon and (iter_no < maxiter)):
        iter_no += 1;
        
        # Solve ||K - QTX|| for X using least squares.
        QQT = Q.dot(Q.transpose());
        QQT_pinv = numpy.linalg.pinv(QQT.toarray());
        QK = Q.dot(K);
        X = QQT_pinv.dot(QK.toarray());
        X = numpy.multiply(X,X>=0);
        X = scipy.sparse.csc_matrix(X);

        # now, update Q
        XKT = X.dot(K.transpose());
        MQt_1KT = M.dot(Qt_1KT);
        num = numpy.multiply(XKT.toarray(),MQt_1KT.toarray());
        num = numpy.multiply(Q.toarray(),num);
        
        XXT = X.dot(X.transpose());
        MQt_1Qt_1TMT = M.dot(Qt_1Qt_1T).dot(M.transpose());
        MQt_1Qt_1TMTQ = MQt_1Qt_1TMT.dot(Q);
        denom = MQt_1Qt_1TMTQ.toarray() + alpha;
        
        Q = numpy.divide(num*10**8,denom);
        Q = scipy.sparse.csc_matrix(Q);

        # now, update M
        QKQt_1T = Q.dot(Qt_1KT.transpose());
        num = numpy.multiply(M.toarray(), QKQt_1T.toarray() + lambda_*numpy.eye(M.shape[0],dtype=float));

        QQT = Q.dot(Q.transpose());
        QQTMQt_1Qt_1T = QQT.dot(M).dot(Qt_1Qt_1T);
        denom = QQTMQt_1Qt_1T.toarray() + lambda_*M.toarray() + alpha;

        M = numpy.divide(num*10**8,denom);
        #row_sum = numpy.sum(M,1);
        #M = M/row_sum[:,numpy.newaxis];
        M = scipy.sparse.csc_matrix(M);

        prevLoss = currentLoss;
        currentLoss = computeLoss_temporalNMF(K,Q,Qt_1,M,trKK,Qt_1Qt_1T,lambda_,alpha);
        #print(str(currentLoss));
        


    return (Q,M);


def symNMF(K,no_topics,alpha,epsilon,maxiter,my_seed):
    '''
    ||K - (Q^T)Q|| + \alpha * ||Q||_1
    Solve ||K - (Q^T)X|| wrt X first (by Least Squares)
    Then solve ||K - (A^T)Q|| wrt A (mult updates) (Here Q = X from before)
    
    randomly initialize a no_topics x K.shape[0] matrix
    '''
    numpy.random.seed(my_seed);
    Q = scipy.sparse.rand(no_topics,K.shape[0],density=1);
    Q = numpy.absolute(Q);


    
    trKK = trAA(K);

    iter_no = 0;
    prevLoss = 100000;
    currentLoss = 2*prevLoss;

    while(numpy.absolute(currentLoss - prevLoss) > epsilon and (iter_no < maxiter)):
        iter_no += 1;
        QQT = Q.dot(Q.transpose());
        QQT_pinv = numpy.linalg.pinv(QQT.toarray());
        QK = Q.dot(K);
        X = QQT_pinv.dot(QK.toarray());
        X = numpy.multiply(X,X>=0);
        X = scipy.sparse.csc_matrix(X);

        XKT = X.dot(K.transpose());
        num = numpy.multiply(XKT.toarray(),Q.toarray());

        XXT = X.dot(X.transpose());
        XXTQ = XXT.dot(Q);
        denom = XXTQ.toarray() + alpha;

        Q = numpy.divide(num*10**8,denom);
        Q = scipy.sparse.csc_matrix(Q);

        prevLoss = currentLoss;
        currentLoss = computeLoss_symNMF(K,Q,trKK,alpha);
        #print(str(currentLoss));

    return Q;

def topWords(Q,vocab_file):

    all_vocab = list(map(lambda x: x.strip().split('\t')[1],open(vocab_file,'r').readlines()[:3470]));
    Q = Q.toarray();
    
    list_of_words = [];
    for i in range(Q.shape[0]):
        indices = sorted(zip(Q[i,:],range(len(Q[i,:]))),key = lambda x: x[0], reverse=True)[:10];
        words = map(lambda x: all_vocab[x[1]], indices);
        list_of_words.append(words);

    return list_of_words;
