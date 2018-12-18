# -*- coding: utf-8 -*-

from optparse import OptionParser

from sklearn.cross_validation import train_test_split, StratifiedKFold
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from tflearn.activations import relu
import csv
# from sets import Set
from utils import *
from config import *


# parser = OptionParser()
# parser.add_option("-d", "--d", default=500, help="The embedding dimension d")
# parser.add_option("-n","--n",default=1, help="global norm to be clipped")
# parser.add_option("-k","--k",default=512,help="The dimension of project matrices k")
# parser.add_option("-t","--t",default = "o",help="Test scenario")
# parser.add_option("-r","--r",default = "ten",help="positive negative ratio")
#
# (opts, args) = parser.parse_args()








class Model(object):
    # domain adaptation model.
    def __init__(self):
        self._build_model()

    def _build_model(self):
        # inputs


        # inputs
        self.RD = tf.placeholder(tf.float32, [num_R, num_D])
        self.RD_normalize = tf.placeholder(tf.float32, [num_R, num_D])
        self.DR = tf.placeholder(tf.float32, [num_D, num_R])
        self.DR_normalize = tf.placeholder(tf.float32, [num_D, num_R])

        self.RS = tf.placeholder(tf.float32, [num_R, num_R])
        self.DS = tf.placeholder(tf.float32, [num_D, num_D])
        self.RS_normalize = tf.placeholder(tf.float32, [num_R, num_R])
        self.DS_normalize = tf.placeholder(tf.float32, [num_D, num_D])

        self.RD_mask = tf.placeholder(tf.float32, [num_R, num_D])
        self.RSmask = tf.placeholder(tf.float32, [num_R, num_R])
        self.DSmask = tf.placeholder(tf.float32, [num_D, num_D])
        # features
        self.Dembedding = weight_variable([num_D, dim_D])
        self.Rembedding = weight_variable([num_R, dim_R])

        W0 = weight_variable([dim_pass + dim_R, dim_R])
        b0 = bias_variable([dim_R])

        # passing 1 times (can be easily extended to multiple passes)
        R_vector1 = tf.nn.l2_normalize(relu(tf.matmul(
            tf.concat([tf.matmul(self.RD_normalize, a_layer(self.Dembedding, dim_pass)) + \
                       tf.matmul(self.RS_normalize, a_layer(self.Rembedding, dim_pass)), \
                       self.Rembedding], axis=1), W0) + b0), dim=1)

        D_vector1 = tf.nn.l2_normalize(relu(tf.matmul(
            tf.concat([tf.matmul(self.DR_normalize, a_layer(self.Rembedding, dim_pass)) + \
                       tf.matmul(self.DS_normalize, a_layer(self.Dembedding, dim_pass)), \
                       self.Dembedding], axis=1), W0) + b0), dim=1)

        self.R_representation = R_vector1
        self.D_representation = D_vector1

        # reconstructing networks
        self.R_reconstruct = bi_layer(self.R_representation, self.R_representation, sym=True, dim_pred=dim_pred)
        RStmp = (self.R_reconstruct - self.RS)
        RStmp = tf.multiply(self.RSmask, (self.R_reconstruct - self.RS))

        RStempu = tf.multiply((tf.ones((num_R, num_R)) - self.RSmask), (self.R_reconstruct - self.RS))

        self.R_reconstruct_loss = tf.reduce_sum(tf.multiply(RStmp, RStmp)) #+ 0.0001 * tf.reduce_sum(tf.multiply(RStempu, RStempu))

        self.D_reconstruct = bi_layer(self.D_representation, self.D_representation, sym=True, dim_pred=dim_pred)
        DStmp = (self.D_reconstruct - self.DS)
        # DStmp = tf.multiply(self.DSmask, (self.D_reconstruct - self.DS))

        DStempu = tf.multiply((tf.ones((num_D, num_D)) - self.DSmask), (self.D_reconstruct - self.DS))
        self.D_reconstruct_loss = tf.reduce_sum(
            tf.multiply(DStmp, DStmp))  # +0.05*tf.reduce_sum(tf.multiply(DStempu, DStempu))

        self.RD_reconstruct = bi_layer(self.R_representation, self.D_representation, sym=False, dim_pred=dim_pred)
        tmp = tf.multiply(self.RD_mask, (self.RD_reconstruct - self.RD))
        tmpu = tf.multiply((tf.ones((num_R, num_D)) - self.RD_mask), (self.RD_reconstruct - self.RD))

        self.DR_reconstruct_loss = tf.reduce_sum(tf.multiply(tmp, tmp)) + 0.03* tf.reduce_sum(
            tf.multiply(tmpu, tmpu))

        self.loss = self.DR_reconstruct_loss + 1 * (self.R_reconstruct_loss +  self.D_reconstruct_loss)  # + tf.reduce_sum(tf.multiply(W0,W0 ))+ tf.reduce_sum(tf.multiply(b0,b0 ))

    def train_and_evaluate(self, Gt,graph, verbose=True, num_steps=2000):
        lr = 0.001

        best_valid_aupr = 0
        best_valid_auc = 0
        test_aupr = 0
        test_auc = 0

        with tf.Session(graph=graph) as sess:
            tf.initialize_all_variables().run()
            for i in range(num_steps):
                _, tloss, dtiloss, results = sess.run([optimizer, self.loss, self.DR_reconstruct_loss, eval_pred], \
                                                      feed_dict={self.DS_normalize: DS_normalize,
                                                                 self.RS_normalize: RS_normalize, \
                                                                 self.DS: DS, self.RS: RS, \
                                                                 self.DR: DR, self.DR_normalize: DR_normalize, \
                                                                 self.RD: RD, self.RD_normalize: RD_normalize, \
                                                                 self.RD_mask: mask, \
                                                                 self.RSmask: RSmask, self.DSmask: DSmask,
                                                                 learning_rate: lr})
                # every 25 steps of gradient descent, evaluate the performance, other choices of this number are possible
                if i % 25 == 0 and verbose == True:
                    print('step', i, 'total and dtiloss', tloss, dtiloss)
                    #
                    #     pred_list = []
                    #     ground_truth = []
                    #     for ele in DTIvalid:
                    #         pred_list.append(results[ele[0], ele[1]])
                    #         ground_truth.append(ele[2])
                    #     valid_auc = roc_auc_score(ground_truth, pred_list)
                    #     valid_aupr = average_precision_score(ground_truth, pred_list)
                    #     if valid_aupr >= best_valid_aupr:
                    #         best_valid_aupr = valid_aupr
                    #         best_valid_auc = valid_auc
                    #     pred_list = []
                    #     ground_truth = []
                    #     for ele in DTItest:
                    #         pred_list.append(results[ele[0], ele[1]])
                    #         ground_truth.append(ele[2])
                    ground_truth = Gt

                    pred_list = results[:, diseasenum]
                    average_score = sum(pred_list) / 577
                    maxscore = max(pred_list)
                    test_auc = roc_auc_score(ground_truth, pred_list)
                    test_aupr = average_precision_score(ground_truth, pred_list)
                    print('test auc aupr average_score maxscore', test_auc, test_aupr, average_score, maxscore)
                    # localstatistics = [tloss, dtiloss, valid_auc, valid_aupr, test_auc, test_aupr]

        return results

graph = tf.get_default_graph()

with graph.as_default():
    model = Model()
    learning_rate = tf.placeholder(tf.float32, [])
    total_loss = model.loss
    dti_loss = model.DR_reconstruct_loss

    optimize = tf.train.AdamOptimizer(learning_rate)
    gradients, variables = zip(*optimize.compute_gradients(total_loss))
    gradients, _ = tf.clip_by_global_norm(gradients, GN)
    optimizer = optimize.apply_gradients(zip(gradients, variables))

    eval_pred = model.RD_reconstruct
mask = RD
DS_normalize = row_normalize(DS, True)
RS_normalize = row_normalize(RS, True)
DR_normalize = row_normalize(DR, False)
RD_normalize = row_normalize(RD, False)
diseasenum = 46
GTr = RD[:, diseasenum]
ORIALL = model.train_and_evaluate(GTr, graph, num_steps=506)
result1 = ORIALL[:, diseasenum]
results=np.multiply((np.ones(num_R)-GTr),result1)
resultindex = np.argsort(results)
resultindex = resultindex[::-1]
rindex=resultindex[0:50]

Rname=RNAname['name']
print(Rname[rindex])
print(results[resultindex])
finalRNA = Rname[resultindex]

#np.savetxt('../casestudy/'+diseasedict[diseasenum]+'miRNA.txt', Rname[resultindex])
auc = []
auprecision = []


