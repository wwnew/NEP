# -*- coding: utf-8 -*-

from optparse import OptionParser

from sklearn.cross_validation import train_test_split, StratifiedKFold
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from tflearn.activations import relu

# from sets import Set
from utils import *
from config import *



class Model(object):
    #domain adaptation model.
    def __init__(self):
        self._build_model()
    
    def _build_model(self):
        #inputs
 


        # inputs
        self.RD = tf.placeholder(tf.float32, [num_R, num_D])
        self.RD_normalize = tf.placeholder(tf.float32, [num_R, num_D])
        self.DR = tf.placeholder(tf.float32, [num_D, num_R])
        self.DR_normalize = tf.placeholder(tf.float32, [num_D, num_R])

        self.RS = tf.placeholder(tf.float32, [num_R, num_R])
        self.DS = tf.placeholder(tf.float32, [num_D, num_D])
        self.RS_normalize = tf.placeholder(tf.float32, [num_R, num_R])
        self.DS_normalize = tf.placeholder(tf.float32, [num_D, num_D])

        self.RSmask = tf.placeholder(tf.float32, [num_R, num_R])
        self.DSmask = tf.placeholder(tf.float32, [num_D, num_D])
        self.RD_mask = tf.placeholder(tf.float32, [num_R,num_D])
        # features
        self.Dembedding = weight_variable([num_D,dim_D])
        self.Rembedding = weight_variable([num_R,dim_R])

        #features_
       
        #feature passing weights (maybe different types of nodes can use different weights)
        W0 = weight_variable([dim_pass+dim_R, dim_R])
        b0 = bias_variable([dim_R])

        #passing 1 times (can be easily extended to multiple passes)
        R_vector1 = tf.nn.l2_normalize(relu(tf.matmul(
            tf.concat([tf.matmul(self.RD_normalize, a_layer(self.Dembedding, dim_pass)) + \
            tf.matmul(self.RS_normalize, a_layer(self.Rembedding, dim_pass)) , \
            self.Rembedding], axis=1), W0)+b0),dim=1)

        D_vector1 = tf.nn.l2_normalize(relu(tf.matmul(
            tf.concat([tf.matmul(self.DR_normalize, a_layer(self.Rembedding, dim_pass)) + \
                       tf.matmul(self.DS_normalize, a_layer(self.Dembedding, dim_pass)),  \
                       self.Dembedding], axis=1), W0) + b0), dim=1)

      


        self.R_representation = R_vector1
        self.D_representation = D_vector1

        
      
        #reconstructing networks
        self.R_reconstruct = bi_layer(self.R_representation,self.R_representation, sym=True, dim_pred=dim_pred)
        #RStmp = tf.multiply(self.RS,(self.R_reconstruct - self.RS_normalize))
        RStmp =  (self.R_reconstruct - self.RS)
        #RStmp = tf.multiply(self.RSmask, (self.R_reconstruct - self.RS))

        #RStempu = tf.multiply((tf.ones((num_R, num_R)) - self.RSmask), (self.R_reconstruct - self.RS))

        self.R_reconstruct_loss = tf.reduce_sum(tf.multiply(RStmp, RStmp))#+0.21*tf.reduce_sum(tf.multiply(RStempu, RStempu))

        self.D_reconstruct = bi_layer(self.D_representation, self.D_representation, sym=True, dim_pred=dim_pred)
        #DStmp = tf.multiply(self.DS, (self.D_reconstruct - self.DS_normalize))
        DStmp =  (self.D_reconstruct - self.DS)
        DStempu = tf.multiply((tf.ones((num_D, num_D)) - self.DS), (self.D_reconstruct - self.DS))
        self.D_reconstruct_loss = tf.reduce_sum(tf.multiply(DStmp, DStmp))#+0.1*tf.reduce_sum(tf.multiply(DStempu, DStempu))


        self.RD_reconstruct = bi_layer(self.R_representation,self.D_representation, sym=False, dim_pred=dim_pred)

        tmp = tf.multiply(self.RD_mask, (self.RD_reconstruct-self.RD))
        tmp = self.RD_reconstruct - self.RD
        tmpu = tf.multiply((tf.ones((num_R, num_D)) - self.RD_mask), (self.RD_reconstruct-self.RD))

        self.DR_reconstruct_loss = tf.reduce_sum(tf.multiply(tmp, tmp))#+0.03*tf.reduce_sum(tf.multiply(tmpu, tmpu))

        self.loss = self.DR_reconstruct_loss + 1*(self.R_reconstruct_loss+self.D_reconstruct_loss) #+ tf.reduce_sum(tf.multiply(W0,W0 ))+ tf.reduce_sum(tf.multiply(b0,b0 ))

    def train_and_evaluate(self, DTIvalid,DTItest, graph, verbose=True, num_steps=2000):
        lr = 0.001

        best_valid_aupr = 0
        best_valid_auc = 0
        test_aupr = 0
        test_auc = 0

        with tf.Session(graph=graph) as sess:
            tf.initialize_all_variables().run()
            for i in range(num_steps):
                _, tloss, dtiloss, results = sess.run([optimizer, self.loss, self.DR_reconstruct_loss, eval_pred], \
                                                      feed_dict={self.DS_normalize: DS_normalize, self.RS_normalize: RS_normalize, \
                                                                 self.DS: DS, self.RS: RS,\
                                                                 self.DR: DR, self.DR_normalize: DR_normalize, \
                                                                 self.RD: RD, self.RD_normalize: RD_normalize, \
                                                                 self.RD_mask: mask, \
                                                                 self.RSmask: RSmask, self.DSmask: DSmask,
                                                                 learning_rate: lr})
                # every 25 steps of gradient descent, evaluate the performance, other choices of this number are possible
                if i % 25 == 0 and verbose == True:
                    print('step', i, 'total and dtiloss', tloss, dtiloss)

                    pred_list = []
                    ground_truth = []
                    for ele in DTIvalid:
                        pred_list.append(results[ele[0], ele[1]])
                        ground_truth.append(ele[2])
                    valid_auc = roc_auc_score(ground_truth, pred_list)
                    valid_aupr = average_precision_score(ground_truth, pred_list)
                    if valid_aupr >= best_valid_aupr:
                        best_valid_aupr = valid_aupr
                        best_valid_auc = valid_auc
                    pred_list = []
                    ground_truth = []
                    for ele in DTItest:
                        pred_list.append(results[ele[0], ele[1]])
                        ground_truth.append(ele[2])
                    test_auc = roc_auc_score(ground_truth, pred_list)
                    test_aupr = average_precision_score(ground_truth, pred_list)
                    print('valid auc aupr,', valid_auc, valid_aupr, 'test auc aupr', test_auc, test_aupr)
                    localstatistics = [tloss, dtiloss, valid_auc, valid_aupr, test_auc, test_aupr]
                    statistics.append(localstatistics)
        return pred_list, results, test_auc, test_aupr


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

DS_normalize = row_normalize(DS,True)
RS_normalize=row_normalize(RS,True)
for key in diseasedict:
    test_precision_round = []
    test_recall_round = []
    test_auroc_round = []

    diseasenum = key


    for r in range(1):
        print ('round',r+1)
        if testS == 'o':
            dti_o = np.loadtxt(network_path+'RDmat.txt',delimiter=",")
        else:
            dti_o = np.loadtxt(network_path+'mat_drug_protein_'+testS+'.txt')

        whole_positive_index = []
        whole_negative_index = []
        disease_pos_index=[]
        disease_neg_index=[]
        statistics=[]
        for i in range(np.shape(dti_o)[0]):
            for j in range(np.shape(dti_o)[1]):
                if j == diseasenum :
                    if int(dti_o[i][j]) == 1:
                        disease_pos_index.append([i, j])
                    elif int(dti_o[i][j]) == 0:
                        disease_neg_index.append([i, j])
                else:
                    if int(dti_o[i][j]) == 1:
                        whole_positive_index.append([i,j])
                    elif int(dti_o[i][j]) == 0:
                        whole_negative_index.append([i,j])


        if PNR == 'ten':
            negative_sample_index = np.random.choice(np.arange(len(whole_negative_index)),size=10*len(whole_positive_index),replace=False)
        elif PNR == 'all':
            negative_sample_index = np.random.choice(np.arange(len(whole_negative_index)),size=len(whole_negative_index),replace=False)
        else:
            print('wrong positive negative ratio')
            break

        data_set = np.zeros((len(negative_sample_index)+len(whole_positive_index),3),dtype=int)
        disease_data_set = np.zeros((len(disease_pos_index)+len(disease_neg_index),3),dtype=int)
        count = 0
        for i in whole_positive_index:
            data_set[count][0] = i[0]
            data_set[count][1] = i[1]
            data_set[count][2] = 1
            count += 1
        for i in negative_sample_index:
            data_set[count][0] = whole_negative_index[i][0]
            data_set[count][1] = whole_negative_index[i][1]
            data_set[count][2] = 0
            count += 1
        dcount=0
        for i in disease_pos_index:
            disease_data_set[dcount][0] = i[0]
            disease_data_set[dcount][1] = i[1]
            disease_data_set[dcount][2] = 1
            dcount += 1
        for i in disease_neg_index:
            disease_data_set[dcount][0] = i[0]
            disease_data_set[dcount][1] = i[1]
            disease_data_set[dcount][2] = 0
            dcount += 1
        #将疾病分成5组


        test_precision_fold = []
        test_recall_fold = []
        test_auroc_fold = []
        GT_fold=[]
        final_fold=[]
        rs = np.random.randint(0,1000,1)[0]

        # kf = StratifiedKFold(data_set[:,2], n_folds=5, shuffle=True, random_state=rs)
        kf = StratifiedKFold(disease_data_set[0:len(disease_pos_index), 2], n_folds=5, shuffle=True, random_state=10)
        fold = 0
        for train_index, test_index in kf:

            # DTItrain, DTItest = data_set[train_index], data_set[test_index]
            # DTItrain, DTIvalid =  train_test_split(DTItrain, test_size=0.05, random_state=rs)
            DTItrain, DTItest = disease_data_set[train_index], disease_data_set[test_index]
            DTItrain = np.vstack((data_set, DTItrain))
            DTItest = np.vstack((DTItest, disease_data_set[len(disease_pos_index):len(disease_neg_index) + len(disease_pos_index) - 1,:]))
            DTItrain, DTIvalid = train_test_split(DTItrain, test_size=0.01, random_state=5)
            RD = np.zeros((num_R, num_D))
            mask = np.zeros((num_R, num_D))
            for ele in DTItrain:
                RD[ele[0], ele[1]] = ele[2]
                mask[ele[0], ele[1]] = 1
            DR = RD.T

            DR_normalize = row_normalize(DR, False)
            RD_normalize = row_normalize(RD, False)
                #v_auc, v_aupr, t_auc, t_aupr = train_and_evaluate(DTItrain=DTItrain, DTIvalid=DTIvalid, DTItest=DTItest, graph=graph, num_steps=2000)
                #test_auc_fold.append(t_auc)
                #test_aupr_fold.append(t_aupr)

            prediction, results, test_auc, test_aupr = model.train_and_evaluate(DTIvalid,DTItest, graph, num_steps=606)

            ground_truth = []
            for ele in DTItest:
                ground_truth.append(ele[2])
                results[ele[0], ele[1]] = 0
            results = results[:, diseasenum - 1]
            resultindex = np.argsort(results)
            resultindex = resultindex[::-1]

            # for i in range(50):
            #     print(results[resultindex[i]])
            #     print(resultindex[i])

            #np.savetxt('../output/' + diseasedict[diseasenum] + str(fold) + 'predictRNA.txt', resultindex)
            print(prediction)
            print(ground_truth)
            valid_auc = roc_auc_score(ground_truth, prediction)
            print(valid_auc)
            finalprediction = sorted(prediction, reverse=True)
            biprediction = np.array(biresults(finalprediction, 0.5))
            index = np.argsort(prediction)
            index = index[::-1]
            GT = np.array(ground_truth)[index]

            # compute ROC curve
            fpr, tpr, thresholds = metrics.roc_curve(GT, finalprediction)
            roc_auc = metrics.auc(fpr, tpr)
            # np.savetxt('../output/fpr',fpr)
            # np.savetxt('../output/tpr',tpr)

            plotroc(fpr, tpr, roc_auc, diseasedict[diseasenum]  + str(fold))

            # compute precision,recall
            precision = []
            recall = []

            for i in range(10):
                pr = precision_score(GT[0:(i + 1) * 10 - 1], biprediction[0:(i + 1) * 10 - 1])
                rec = sum((GT[0:(i + 1) * 10 - 1] == 1) * (biprediction[0:(i + 1) * 10 - 1] == 1)) / sum(GT == 1)
                precision.append(pr)
                recall.append(rec)

            plotbar(precision, 'Precision', diseasedict[diseasenum]  + str(fold))
            plotbar(recall, 'Recall', diseasedict[diseasenum]  + str(fold))

            test_auroc_fold.append(roc_auc)
            test_precision_fold.append(precision)
            test_recall_fold.append(recall)
            GT_fold.append(GT)
            final_fold.append(finalprediction)
            fold = fold + 1

            # test_precision_round.append(test_precision_fold)
            # test_recall_round.append(test_recall_fold)
            # test_auroc_round.append(test_auroc_fold)

        np.savetxt('../output1/' + diseasedict[diseasenum]  + '_pre.txt', test_precision_fold)
        np.savetxt('../output1/' + diseasedict[diseasenum]  + '_rec.txt', test_recall_fold)
        np.savetxt('../output1/' + diseasedict[diseasenum]  + '_auroc.txt', test_auroc_fold)
        np.save('../output1/' + diseasedict[diseasenum] + '_GT.npy', GT_fold)
        np.save('../output1/' + diseasedict[diseasenum]  + '_finalP.npy', final_fold)
        #np.savetxt('../output/' + diseasedict[diseasenum]  + 'statistics.txt', statistics)

        # np.savetxt('../output/'+diseasedict[diseasenum]+'_pre.txt', test_precision_round)
        # np.savetxt('../output/'+diseasedict[diseasenum]+'_rec.txt', test_recall_round)
        # np.savetxt('../output/'+diseasedict[diseasenum]+'_auroc.txt', test_auroc_round)
