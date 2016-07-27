import sys
import numpy as np
import scipy
import operator
from sklearn.linear_model import LogisticRegression
#from sklearn.cross_validation import train_test_split
from sklearn import metrics

import Multi_logit as LR

#######################################################################################
"""Read data """
def read_data(filename):
    f = open(filename,"r")
    dpoint=0
    x = []
    data_to_pattern = {}
    while 1:
        line = f.readline()
        if not line:
            break;
        data,pattern = line.strip().split("|");
        x_row = []
        
        data = data.split()
        data_str=""
        for i in range(0,len(data)):
            d = float(data[i])
            x_row.append(d)
            data_str  = data_str+str(d)
       
        x.append(x_row)
        if not data_to_pattern.has_key(data_str):
            data_to_pattern[data_str] = pattern
        
        dpoint = dpoint + 1
   
    np.random.shuffle(x)
    
    return np.asarray(x),data_to_pattern

def load_image_list(imagehas):
    hash_={}
    file1 = open(imagehas,"r")
    while 1:
        line = file1.readline()
        if not line:
            break
        line = line.strip()
        if not hash_.has_key(line):
            hash_[line]=1
    file1.close()
    return hash_

def read_item_index(file):
    hash_={}
    file1 = open(file,"r")
    while 1:
        line = file1.readline()
        if not line:
            break
        line = line.strip().split("\t")
        if not hash_.has_key(line[0]):
            hash_[line[0]]= int(line[1])
    file1.close()
    return hash_

#def baseline(x,y,initial_data_size,N,Number_of_class):

#    xtrain = x[0:initial_data_size]
#    ytrain = y[0:initial_data_size]
#    #print xtrain.shape, ytrain.shape
#    #W,b = LR.executeLogit(xtrain.T,ytrain.T,Number_of_class)
#    W= LR.executeLogit(xtrain.T,ytrain.T,Number_of_class)
#    #exit()
#    #print W,b
#    xtest = x[initial_data_size:N]
#    ytest = y[initial_data_size:N]
#    #LR.predict_and_performace(xtest,ytest,W,b)
#    #LR.predict_and_performace(xtest,ytest,W)
#    LR.predict_and_performace_softmax(xtest.T,ytest.T,W,Number_of_class)

#def baseline_crossValidation(x,y,N,Number_of_class):

#    N = x.shape[0]
#    CV = 3
#    CV_step = N/CV
#    Acc = []
#    for i in range(1,CV+1):
#        if i==1:
#            xtest = x[(i-1)*CV_step:(i-1)*CV_step+CV_step]
#            xtrain = x[(i-1)*CV_step+CV_step:N]
#            ytest = y[(i-1)*CV_step:(i-1)*CV_step+CV_step]
#            ytrain = y[(i-1)*CV_step+CV_step:N]
#        else:
#            xtrain2 = x[0:(i-1)*CV_step]
#            xtest = x[(i-1)*CV_step:(i-1)*CV_step+CV_step]
#            xtrain1 = x[(i-1)*CV_step+CV_step:N]
#            xtrain = np.append(xtrain2,xtrain1,axis=0)
#            ytrain2 = y[0:(i-1)*CV_step]
#            ytest = y[(i-1)*CV_step:(i-1)*CV_step+CV_step]
#            ytrain1 = y[(i-1)*CV_step+CV_step:N]
#            ytrain = np.append(ytrain2,ytrain1,axis=0)
#            #print xtrain1.shape, xtrain2.shape, xtrain.shape
#        #print xtrain.shape, xtest.shape, ytrain.shape, ytest.shape, x.shape
#        W= LR.executeLogit(xtrain.T,ytrain.T,Number_of_class)
#        acc = LR.predict_and_performace_softmax(xtest.T,ytest.T,W,Number_of_class)
#        Acc.append(acc)
#    print np.mean(np.asarray(Acc))

#def znormalized(x):
#    rowmean = np.mean(x,axis=0)
#    rowstd = np.std(x,axis=0)
#    x = (x-rowmean)/rowstd
#    x = np.nan_to_num(x)
#    #print x
#    return x

def get_representative_random(x,no_of_data_passed_to_user):
    x_l = []
    x_unl = []
    index = np.random.randint(0,len(x),no_of_data_passed_to_user)
   
    for i in index:
        x_l.append(x[i])
    index = sorted(index, reverse=True)
    for i in index:
        np.delete(x,i,0)
    return np.asarray(x_l),np.asarray(x)
        
def get_representative_k_center(x,k):

    t = np.random.random_integers(0,len(x)-1)
    centers = []
    
    centers.append(x[t])
    
    x = np.delete(x,t,0)
    
    while len(centers)!=k:
        best_dist = 0;
        delete_index = 0;
        for i in range(0,len(x)):
            d=0;
            local_best_dist=99999;
            for c in centers:
                dist = scipy.spatial.distance.jaccard(x[i],c);
                if dist<local_best_dist:
                    local_best_dist=dist;
            d = local_best_dist    
            if d >= best_dist:
                best_dist = d
                delete_index = i;

        centers.append(x[delete_index])
       
        x = np.delete(x,delete_index,0)
            
    c = np.asarray(centers)
  
    return np.asarray(centers), np.asarray(x)

def get_pattern_to_show(x):
    c = [str(dp) for dp in x]
    return c

#if __name__ == '__main__':

#    x,y = read_data(sys.argv[1])
#    #x = znormalized(x)
#    N = x.shape[0]
#    Learning_Rate = 0.03
#    no_of_data_passed_to_user=int(sys.argv[2])
#    batch_size = int(sys.argv[3])   
#    training_data_size = int(sys.argv[4])
#    xtest = x[training_data_size:N]
#    ytest = y[training_data_size:N]
#    Number_of_class = len(np.unique(y))
#    Feature_Size = x.shape[1]
#    Possible_Iteration = training_data_size/batch_size

#    gradient = {}
#    for i in range(0,Possible_Iteration):
#        gradient[i] = []   
#    Learning_Rate = [0]*Possible_Iteration
#    Learning_Rate[0] = 0.0001
#    iteration = 0
#    gamma = 0.01

#    Train_set_x = []
#    Train_set_y = []
#    xtrain = x[0:0+batch_size]
#    ytrain = y[0:0+batch_size]
#    xlabel,ylabel,xunlabel,yunlabel = get_representative_random(xtrain,ytrain,no_of_data_passed_to_user)
#    #print "1st = ", xlabel.shape, ylabel.shape
#    Train_set_x = xlabel
#    Train_set_y = ylabel
#    #for i in range(0,len(Train_set_x)):
#    #    print Train_set_x[i],Train_set_y[i]
#    #exit()
#    W_initial= LR.initiateLogit_collective_train(Train_set_x.T,Train_set_y.T,Number_of_class)
#    #W_initial,grad,error= LR.initiateLogit(Train_set_x.T,Train_set_y.T,Number_of_class,Learning_Rate[iteration])
#    recall,pre,fscore = LR.predict_and_performace_softmax(xtest.T,ytest,W_initial,Number_of_class)
#    #print "cost = ", error, 
#    print "Precision: {0:.2f}".format(pre),"Recall: {0:.2f}".format(recall),"Fscore: {0:.2f}".format(fscore),"Number of Feedback = ",len(Train_set_y)
#    #xtrain = x[50:50+no_of_data_passed_to_user]
#    #ytrain = y[50:50+no_of_data_passed_to_user]
#    
#    #x,y,EGL = LR.expected_gradient_length_driver(W_initial,xtrain.T,ytrain.T,Number_of_class)
#    #print np.asarray(EGL).shape
#    #print "here"

#    for j in range(batch_size,training_data_size,batch_size):
#    #    models = []
#        xtrain = x[j:j+batch_size]
#        ytrain = y[j:j+batch_size]
#        #print "iteration = ",xtrain.shape, ytrain.shape, j, batch_size, 
#        x_ELG,y_EGL,EGL = LR.expected_gradient_length_driver(W_initial,xtrain.T,ytrain.T,Number_of_class)
#        #print "2nd = ", x_ELG.shape, y_EGL.shape
#        x_ELG = x_ELG[0:(batch_size*75)/100]
#        y_EGL = y_EGL[0:(batch_size*75)/100]
#        #print "2nd after = ", x_ELG.shape, y_EGL.shape
#        xlabel,ylabel,xunlabel,yunlabel = get_representative_random(x_ELG,y_EGL,no_of_data_passed_to_user)
#        
#        #print "3rd = ", xlabel.shape, ylabel.shape, Train_set_x.shape, Train_set_y.shape
#        ylabel = ylabel.reshape(ylabel.shape[0],)
#        Train_set_x = np.concatenate((Train_set_x,xlabel),axis=0)
#        Train_set_y = np.concatenate((Train_set_y,ylabel),axis=0)
#        #print "4th = ", xlabel.shape, ylabel.shape, Train_set_x.shape, Train_set_y.shape
#        W_update = LR.executeLogit_collective_train(Train_set_x.T,Train_set_y,W_initial,Number_of_class)
#        #W_update,grad,error = LR.executeLogit(Train_set_x.T,Train_set_y,W_initial,Learning_Rate[iteration],Number_of_class)
#        #norm_old = np.linalg.norm(W_initial)
#        #norm_new = np.linalg.norm(W_update)
#        #diff =   np.fabs(norm_new- norm_old)
#        
#        W_initial = W_update
#        recall,pre,fscore = LR.predict_and_performace_softmax(xtest.T,ytest,W_initial,Number_of_class)
#        #print "diff = ", diff, norm_new, norm_old, error, "Accuracy: {0:.2f}%".format(acc)
#        print "Precision: {0:.2f}".format(pre),"Recall: {0:.2f}".format(recall),"Fscore: {0:.2f}".format(fscore), "Number of Feedback = ",len(Train_set_y)

#    #print W_initial.reshape(Number_of_class,Feature_Size)
#    
#    #print W_initial.shape
#    #output = open("weight_vector_IPM.dat","w")
#    #W_initial = W_initial.reshape((Number_of_class, Feature_Size))
#    #print W_initial
#    #for i in W_initial[1]:
#    #    output.write(str(i)+"\n")
#    #print "Acc = ", LR.predict_and_performace_softmax(xtest.T,ytest,W_initial,Number_of_class)
#    W = LR.softmax_with_optimizer(Train_set_x.T,Train_set_y.T,Number_of_class)
#    print "Acc = ",LR.predict_and_performace_softmax(xtest.T,ytest,W,Number_of_class), Train_set_x.shape

#    classifier = LogisticRegression(C=1,solver='liblinear', max_iter=60, multi_class='ovr')
#    #print y_train
#    classifier.fit(Train_set_x, Train_set_y)
#    print classifier.score(xtest, ytest)

#    #for i in range(0,len(Train_set_x)):
#    #    print Train_set_x[i],Train_set_y[i]
