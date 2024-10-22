# Janani Kalyanam
# December 7, 2015
import os, sys
import numpy, scipy, scipy.sparse
from matrix_factorization import *

def similarity_cosine(Q_predicted,Q_true):

    QQT_predicted = Q_predicted.dot(Q_predicted.transpose());
    QQT_predicted_norms = QQT_predicted.diagonal()**(0.5);
    Q_predicted = numpy.divide(Q_predicted.transpose(),QQT_predicted_norms).transpose();

    QQT_true = Q_true.dot(Q_true.transpose());
    QQT_true_norms = QQT_true.diagonal()**(0.5);
    Q_true = numpy.divide(Q_true.transpose(),QQT_true_norms).transpose();

    QpredictedQtrueT = Q_predicted.dot(Q_true.transpose());
    sim_ = 0;
    groundtruth_mapping = [];
    no_topics = Q_predicted.shape[0];


    for ii in range(no_topics):
        sim_ += numpy.max(QpredictedQtrueT[ii,:]);
        groundtruth_mapping.append(numpy.argmax(QpredictedQtrueT[ii,:]));

    return (sim_/no_topics, groundtruth_mapping);

def NDCG(P,T):
    # P is a list of predicted indices
    # T is the true list of indices

    rel = [];
    NDCG = 0;
    for ii in range(len(P)):
        if(P[ii] in T):
            rel.append(len(P) - T.index(P[ii]));
        else:
            rel.append(0);

    DCG = rel[0];
    sorted_rel = list(reversed(range(1,len(P)+1)));
    norm = sorted_rel[0];
    for ii in range(2,len(P)):
        DCG = DCG + rel[ii]/float(numpy.log2(ii));
        norm = norm + sorted_rel[ii]/float(numpy.log2(ii));

    if(norm == 0):
        return 0;
    else:
        return DCG/float(norm);

def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return numpy.mean([apk(a,p,k) for a,p in zip(actual, predicted)])


