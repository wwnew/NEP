import tensorflow as tf
import numpy as np
from matplotlib import pylab
from matplotlib import pyplot
from tensorflow.python.framework import ops
from tflearn.initializations import truncated_normal 
from tflearn.activations import relu

def check_symmetric(a, tol=1e-8):
    return np.allclose(a, a.T, atol=tol)

def row_normalize(a_matrix, substract_self_loop):
    if substract_self_loop == True:
        np.fill_diagonal(a_matrix,0)
    a_matrix = a_matrix.astype(float)
    row_sums = a_matrix.sum(axis=1)+1e-12
    new_matrix = a_matrix / row_sums[:, np.newaxis]
    new_matrix[np.isnan(new_matrix) | np.isinf(new_matrix)] = 0.0
    return new_matrix

def sum_rows(matrix):
    row_sums = matrix.sum(axis=1)+1e-12
    newMatrix=matrix.astype(float)
    for i in range(len(matrix)):
        newMatrix[i,:]=matrix[i,:]/row_sums[i]
    return newMatrix

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1,dtype=tf.float32)
    return tf.Variable(initial, dtype=tf.float32)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape,dtype=tf.float32)
    return tf.Variable(initial, dtype=tf.float32)

def a_layer(x,units):
    W = weight_variable([x.get_shape().as_list()[1],units])
    b = bias_variable([units])
    return relu(tf.matmul(x, W) + b)


def bi_layer(x0,x1,sym,dim_pred):
    if sym == False:
        W0p = weight_variable([x0.get_shape().as_list()[1],dim_pred])
        W1p = weight_variable([x1.get_shape().as_list()[1],dim_pred])
        return tf.matmul(tf.matmul(x0, W0p), 
                            tf.matmul(x1, W1p),transpose_b=True)
    else:
        W0p = weight_variable([x0.get_shape().as_list()[1],dim_pred])
        return tf.matmul(tf.matmul(x0, W0p), 
                            tf.matmul(x1, W0p),transpose_b=True)

def precision_score(y_true, y_pred):
    return (sum((y_true==1)*(y_pred==1))/sum((y_pred==1)))


def biresults(result,thr):
    br=[]
    for item in result:
        if item>thr:
            br.append(1)
        else:
            br.append(0)
    return br

def plotbar(vector,graphname,diseasename):
    pyplot.figure(num=None, figsize=(6, 5))
    pyplot.ylim([0.0, 1.0])
    name_list = ['','TOP-20','', 'TOP-40','', 'TOP-60','','TOP-80','', 'TOP-100']
    pyplot.title(diseasename)
    pyplot.xlabel('TOP-K')
    pyplot.ylabel(graphname)
    pyplot.bar(range(len(name_list)),vector,tick_label=name_list)
    pyplot.savefig('../figure/' + diseasename + graphname + '.png')

    pyplot.close()

def plotroc(fpr,tpr,score,name):
    pyplot.figure(num=None, figsize=(6, 5))
    pyplot.xlim([0.0, 1.0])
    pyplot.ylim([0.0, 1.0])
    pyplot.xlabel('False positive rate')
    pyplot.ylabel('True positive rate')
    pyplot.title('%s (AUROC=%0.3f) '  % (name,score))
    pyplot.fill_between(fpr, tpr,alpha='0.5')
    pyplot.grid(True, linestyle='-',color='0.5')
    pyplot.plot(fpr, tpr, lw=1)
    pyplot.savefig('../figure/' + name + ".png")

    pyplot.close()