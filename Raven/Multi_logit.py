import sys
import numpy as np
import math
import time
import scipy
import scipy.optimize

from sklearn.metrics import roc_auc_score
from sklearn import metrics
import copy
from operator import itemgetter

lamda = 1

class Logit(object):

    """ Initialization of MultiLogit object """
    

    def __init__(self, class_size, feature_size,lamda):
    
        
        self.lamda = lamda                  # weight decay parameter
        self.class_size = class_size    # number of classes always 1 for logit
        self.feature_size = feature_size      # number of features
        self.limit0 = 0
        self.limit1 = feature_size
        self.limit2 = feature_size + class_size
        
        
        self.theta = 0.005 * np.random.randn(class_size * feature_size)
   
#    def sigmoid(self, x):
#    
#        return (1 / (1 + np.exp(-x)))

#   
#    def column_normalized(self,x):
#        row_sum = np.sum(x,axis=1)
#        for i in range(0,x.shape[0]):
#            for j in range(0,x.shape[1]):
#                x[i][j] = x[i][j]/row_sum[i]
#        return x

    
#    def cost_logit_regression(self,W,input,label):
#        W1 = W[self.limit0:self.limit1].reshape(self.class_size,self.feature_size) # 1 X D
#       
#        training_size = input.shape[1]
#        print "1______________",W1
#        #ones = np.ones(training_size)
#        #print label.shape
#        #print training_size
#        sum_of_squares_error_part1 =  np.dot(label, np.log(self.sigmoid(np.dot(W1,input))).T) # (1 X N).((1XD DXN)).T
#        sum_of_squares_error_part2 =  np.dot((1-label), np.log(1 - self.sigmoid(np.dot(W1,input))).T)
#        sum_of_squares_error = (sum_of_squares_error_part1 + sum_of_squares_error_part2)/training_size * -1

#        regularizer =  (self.lamda * 0.5 * np.dot(W1,W1.T))

#        cost = sum_of_squares_error + regularizer
#        

#        cost_gradiant = np.dot((self.sigmoid(np.dot(W1,input)) - label),input.T)/training_size + (self.lamda * W1) # (1 X D, D X N) X (N X D) =  1 X D
#        #offset_gradiant = np.sum(self.sigmoid(np.dot(W1,input)+b1)- label)/training_size

#        W1_grad = np.array(cost_gradiant)  
#        #b1_grad = np.array(offset_gradiant)
#        #print W1_grad.shape
#        #print W1_grad
#        #W_grad = np.concatenate((W1_grad.flatten(),b1_grad.flatten()))
#        W_grad = W1_grad.flatten()
#        print "2&&&&&&&&&&&&&&&&",W_grad
#        #print "cost = ",cost, sum_of_squares_error, regularizer, offset_gradiant
#        return [cost, W_grad]
#   
 #######################################################################################
#    """Expected Maximum Gradient Length"""
#    def expected_gradient_length(self, theta, data, labels):

#        m = data.shape[1]
#        theta = theta.reshape(self.class_size, self.feature_size)
#        theta_data = theta.dot(data)
#        theta_data = theta_data - np.max(theta_data)
#        prob_data = np.exp(theta_data) / np.sum(np.exp(theta_data), axis=0)
#        indicator = scipy.sparse.csr_matrix((np.ones(m), (labels, np.array(range(m)))))
#        indicator = np.array(indicator.todense())
#        grad = (-1)*(indicator - prob_data).dot(data.transpose())
#        print grad.shape, prob_data.shape

 #######################################################################################
    """Soft Max"""
    def softmax_cost(self, theta, data, labels):
        
        #print "softmax", data.shape,labels.shape
        m = data.shape[1]
        theta = theta.reshape(self.class_size, self.feature_size)
        theta_data = theta.dot(data)
        theta_data = theta_data - np.max(theta_data)
        prob_data = np.exp(theta_data) / np.sum(np.exp(theta_data), axis=0)
        indicator = scipy.sparse.csr_matrix((np.ones(m), (labels, np.array(range(m)))))
        indicator = np.array(indicator.todense())
        error = (-1 / m) * np.sum(indicator * np.log(prob_data))
        cost = error + (self.lamda / 2) * np.sum(theta * theta)
        
        grad = (-1 / m) * (indicator - prob_data).dot(data.transpose()) + self.lamda * theta

        return cost, grad.flatten()
#######################################################################################
#    """Read data """
#def read_data(filename):
#    f = open(filename,"r")
#    dpoint=0;
#    x = []
#    y = []
#    while 1:
#        line = f.readline()
#        if not line:
#            break;
#        data = line.strip().split(",");
#        x_row = []
#        #x_row.append(float(1))
#        for i in range(0,len(data)-1):
#            x_row.append(float(data[i]))
#        x.append(x_row)
#        y.append(float(data[len(data)-1]))
#        #print dpoint
#        dpoint = dpoint + 1
#    return np.asarray(x).T,np.asarray(y).T 


############################################################################################
#""" Loads data, trains the MultiLogit model and learned weights """
#def initiateLogit(x,y,Number_of_class,Learning_Rate):
#    
#    max_iterations = 600
#    feature_size = x.shape[0]
#    #print "in here ",x.shape, y.shape
#    #print x[0]
#    #N = x.shape[0]
#    """ Initialize the MultiLogit """
#    #xtrain = x[j:j+initial_data_size]
#    #ytrain = y[j:j+initial_data_size]
#    xtrain = x
#    ytrain = y
#    encoder = Logit(Number_of_class, feature_size,lamda)
#    '''
#    #*****************Numerical Gradient Checking********************
#    
#    cost, theta_grad = encoder.cost_logit_regression(encoder.W, xtrain,ytrain)
#    epsilon = 1e-4
#    theta = copy.copy(encoder.W)
#    num_grad = np.zeros(len(encoder.W))
#    for i in range(0,len(encoder.W)):
#        memo = theta[i]
#        theta[i] = memo + epsilon
#        cost_plus,t = encoder.cost_logit_regression(theta, xtrain,ytrain)
#        theta[i] = memo - epsilon
#        cost_minus,t = encoder.cost_logit_regression(theta, xtrain,ytrain)
#        theta[i] = memo
#        num_grad[i] = (cost_plus - cost_minus) / (2 * epsilon);
#    print "numeric gradiant check = ", np.linalg.norm(theta_grad - num_grad)
#    print theta_grad
#    print num_grad
#    exit(1)
#    '''

#    """ Run the L-BFGS algorithm to get the optimal parameter values """
#    #encoder.cost_logit_regression(encoder.W,xtrain,ytrain)
#    #sys.exit()
#    cost,gradient,error = encoder.softmax_cost(encoder.theta,xtrain,ytrain)
#    #print encoder.theta.shape
#    opt_W = encoder.theta - Learning_Rate*gradient
#    #opt_solution  = scipy.optimize.minimize(encoder.softmax_cost, encoder.theta,args = (xtrain,ytrain,), method = 'L-BFGS-B',jac = True, options = {'maxiter': max_iterations,'disp': False})
#    #opt_W     = opt_solution.x
#    #opt_W1        = opt_W[encoder.limit0 : encoder.limit1].reshape(Number_of_class, feature_size)
#    #encoder.W1    = opt_W1
#    #opt_b1        = opt_W[encoder.limit1 : encoder.limit2].reshape(Number_of_class, 1)
#    """ Visualize the obtained optimal W1 weights """
#    print opt_W.shape
#    
#    #return opt_W1,opt_b1
#    return opt_W, gradient,error

###########################################################################################
""" Loads data, trains the MultiLogit model and learned weights """
def initiateLogit_collective_train(x,y,Number_of_class):
    
    max_iterations = 600
    feature_size = x.shape[0]
    #print "in here ",x.shape, y.shape
    #print x[0]
    #N = x.shape[0]
    """ Initialize the MultiLogit """
    #xtrain = x[j:j+initial_data_size]
    #ytrain = y[j:j+initial_data_size]
    xtrain = x
    ytrain = y
    encoder = Logit(Number_of_class, feature_size,lamda)
    '''
    #*****************Numerical Gradient Checking********************
    
    cost, theta_grad = encoder.cost_logit_regression(encoder.W, xtrain,ytrain)
    epsilon = 1e-4
    theta = copy.copy(encoder.W)
    num_grad = np.zeros(len(encoder.W))
    for i in range(0,len(encoder.W)):
        memo = theta[i]
        theta[i] = memo + epsilon
        cost_plus,t = encoder.cost_logit_regression(theta, xtrain,ytrain)
        theta[i] = memo - epsilon
        cost_minus,t = encoder.cost_logit_regression(theta, xtrain,ytrain)
        theta[i] = memo
        num_grad[i] = (cost_plus - cost_minus) / (2 * epsilon);
    print "numeric gradiant check = ", np.linalg.norm(theta_grad - num_grad)
    print theta_grad
    print num_grad
    exit(1)
    '''

    """ Run the L-BFGS algorithm to get the optimal parameter values """
    opt_solution  = scipy.optimize.minimize(encoder.softmax_cost, encoder.theta,args = (xtrain,ytrain,), method = 'L-BFGS-B',jac = True, options = {'maxiter': max_iterations,'disp': False})
    opt_W     = opt_solution.x
    
    #print opt_W.shape
    
    #return opt_W1,opt_b1
    return opt_W

#def executeLogit(x,y,theta,Learning_Rate,Number_of_class):

#    """ Run the L-BFGS algorithm to get the optimal parameter values """
#   
#    #lamda = 0.0001
#    max_iterations = 600
#    #Number_of_class = len(np.unique(y))
#    feature_size = x.shape[0]
#    xtrain = x
#    ytrain = y
#    encoder = Logit(Number_of_class, feature_size,lamda)
#    cost,gradient,error = encoder.softmax_cost(theta,xtrain,ytrain)
#    #print encoder.theta.shape
#    opt_W = encoder.theta - Learning_Rate*gradient
#    #print theta.shape
#    #opt_solution  = scipy.optimize.minimize(encoder.softmax_cost, theta,args = (xtrain,ytrain,), method = 'L-BFGS-B',jac = True, options = {'maxiter': max_iterations,'disp': False})
#    #opt_W     = opt_solution.x
#    #opt_W1        = opt_W[encoder.limit0 : encoder.limit1].reshape(Number_of_class, feature_size)
#    #encoder.W1    = opt_W1
#    #opt_b1        = opt_W[encoder.limit1 : encoder.limit2].reshape(Number_of_class, 1)
#    """ Visualize the obtained optimal W1 weights """
#    #print opt_W.shape
#    #print np.linalg.norm(theta-opt_W)
#    #return opt_W1,opt_b1
#    return opt_W,gradient,error

def executeLogit_collective_train(x,y,theta,Number_of_class):

    """ Run the L-BFGS algorithm to get the optimal parameter values """
   
    #lamda = 0.0001
    max_iterations = 200
    #Number_of_class = len(np.unique(y))
    feature_size = x.shape[0]
    xtrain = x
    ytrain = y
    encoder = Logit(Number_of_class, feature_size,lamda)
    
    opt_solution  = scipy.optimize.minimize(encoder.softmax_cost, encoder.theta,args = (xtrain,ytrain,), method = 'L-BFGS-B',jac = True, options = {'maxiter': max_iterations,'disp': False})
    opt_W     = opt_solution.x
    #opt_W1        = opt_W[encoder.limit0 : encoder.limit1].reshape(Number_of_class, feature_size)
    #encoder.W1    = opt_W1
    #opt_b1        = opt_W[encoder.limit1 : encoder.limit2].reshape(Number_of_class, 1)
    """ Visualize the obtained optimal W1 weights """
    #print opt_W.shape
    #print np.linalg.norm(theta-opt_W)
    #return opt_W1,opt_b1
    return opt_W

def expected_gradient_length_driver(theta,data,class_size):
    m = data.shape[1]
    #gradient_length = np.zeros((m,1)) #old way
    gradient_length = m*[0]
    theta = theta.reshape(class_size, data.shape[0])
    theta_data = theta.dot(data)
    #print theta_data.shape, data.shape, theta.shape
    theta_data = theta_data - np.max(theta_data)
    prob_data = np.exp(theta_data) / np.sum(np.exp(theta_data), axis=0)

    for C in range(0,class_size):
        labels_grad = np.full((m,),C)
        #print labels_grad.shape, labels_grad
        indicator = np.zeros((class_size,m))
        indicator[C,:] = 1
        #indicator = scipy.sparse.csr_matrix((np.ones(m), (labels, np.array(range(m)))))
        #indicator = np.array(indicator.todense())
        #print "indicator = ", indicator
    #print "theta = ", theta
        grad = (-1 / m) * (indicator - prob_data).dot(data.transpose()) + lamda * theta
    #grad = (-1)*(indicator - prob_data).dot(data.transpose())
        #print grad.shape, prob_data.shape
        #print grad[C], prob_data[C]
        #print np.linalg.norm(grad[C])
        #print prob_data[C]*np.linalg.norm(grad[C])
        #exit()
        #grad_norm = np.linalg.norm(grad[C],axis=1).reshape((class_size,1))
        grad_norm = np.linalg.norm(grad[C])
        #print "grad_norm = ",C , grad_norm.shape, prob_data[C].shape, prob_data[C]*grad_norm
    #print "probdata", prob_data, prob_data.shape
        gradient_length = np.add(gradient_length , prob_data[C]*grad_norm)
        #print "next = ",C,len(gradient_length), gradient_length
    
    gradient_length = gradient_length.reshape(m,1)
    #print gradient_length.shape
    #print data, gradient_length
    #labels = labels.reshape(1,m)
    #print labels.shape, data.shape
    #xy = np.concatenate((data,labels),axis=0)
    EGL = np.concatenate((data,gradient_length.T),axis=0).T
    #print EGL
    sorted_Col = EGL.shape[1]-1
    EGL = EGL[EGL[:,sorted_Col].argsort()[::-1]]
    
    data,EGL = np.hsplit(EGL, np.array([sorted_Col]))
    #x,y = np.hsplit(data, np.array([sorted_Col-1]))
    #print x,y,EGL
    return data,EGL
    #return gradient_length.item(0,0)

def get_top_houses(opt_theta,x,tran_to_pattern,class_size,choice_items):
    feature_size = x.shape[0]
    #print opt_theta.shape, feature_size, class_size
    #print opt_theta
    opt_theta = opt_theta.reshape((class_size, feature_size))
    prod = opt_theta.dot(x)
    pred = np.exp(prod) / np.sum(np.exp(prod), axis=0)
    #pred = pred.argmax(axis=0)
    class_0_pred = pred[0]
    class_1_pred = pred[1]
    tran = x.T
    house_pred = []
    for index in range(0,tran.shape[0]):
        if class_1_pred[index] > class_0_pred[index]:
            x_str = "".join([str(i) for i in tran[index]])
            house_pred.append((x_str,class_1_pred[index]))
    top_houses = sorted(house_pred, key=itemgetter(1),reverse=True)
    if len(top_houses) >= 50:
        top_k = (len(top_houses)*10)/100
    elif len(top_houses) > 10 and len(top_houses) < 50:
        top_k = (len(top_houses)*30)/100
    else:
        top_k = len(top_houses)
    print top_k, len(top_houses)
    houses = []
    for i in range(0,top_k):
        pat = tran_to_pattern[top_houses[i][0]]
        pat = pat.split("-")
        keywords = pat[0].split(",")
        in_key = []
        out_key = []
        for j in range(0,len(keywords)):
            if keywords[j] in choice_items:
                in_key.append(keywords[j])
                #keywords[i] = "<u>"+keywords[i]+"</u>"
            else:
                out_key.append(keywords[j])
        keywords = []
        keywords.append(in_key)
        keywords.append(out_key)
        address = pat[2]
        address = address.split()
        if len(address)>=3:
            address = address[len(address)-3:len(address)]
            address = " ".join(address)
        else:
            address = " ".join(address)
        pat[2] = address
        pat[0] = keywords
        houses.append((pat,top_houses[i][1]))
    if len(top_houses) == 0:
        pat = []
        pat.append([[],["No House found"]])
        pat.append("")
        pat.append("")
        houses.append((pat,0.0))
    #print houses
    return houses
    
#def predict_and_performace_softmax(x,y,opt_theta,class_size):
#    
#    feature_size = x.shape[0]
#    #print opt_theta.shape, feature_size, class_size
#    #print opt_theta
#    opt_theta = opt_theta.reshape((class_size, feature_size))
#    prod = opt_theta.dot(x)
#    pred = np.exp(prod) / np.sum(np.exp(prod), axis=0)
#    pred = pred.argmax(axis=0)
#    #print pred
#    #print y
#    #print "Accuracy: {0:.2f}%".format(100 * np.sum(pred == y, dtype=np.float64) / y.shape[0]), y.shape[0]
#    acc = 100 * np.sum(pred == y, dtype=np.float64) / y.shape[0]
#    precision = metrics.precision_score(y, pred)
#    recall = metrics.recall_score(y, pred)
#    F_score = metrics.f1_score(y, pred)
#    #print precision, recall, F_score
#    return recall, precision, F_score
    '''
    test = float(x.shape[0])
    encoder = Logit(1,1,0)
    print W
    prediction = encoder.sigmoid(np.dot(W,x.T)).T
    correct=0.0
    TP=0.01
    FP=0.01
    FN = 0.01
    pos_cls=1.0
    neg_cls=0.0
    actual=0.0;
    acuracy_1 = 0.0
    for i in range(0,y.shape[0]):
        if(prediction[i] > 0.5):
            pred_c = 1.0
        else:
            pred_c = 0.0
        actual = y[i]
        if pred_c == y[i]:
            acuracy_1 = acuracy_1 + 1.0
        if actual == pos_cls and pred_c==pos_cls:
            TP = TP + 1.0
        if actual == neg_cls and pred_c==pos_cls:
            FP = FP + 1.0
        if actual == pos_cls and pred_c==neg_cls:
            FN = FN + 1.0
        #if(y[i]-1 == pred_c):
        #    correct = correct + 1
        #print y[i]-1,pred_c
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    F1 = (2*precision*recall)/(precision+recall)
    acuracy_1 = (acuracy_1/len(y))*100
    pos_cls=0.0
    neg_cls=1.0
    actual=0.0;
    acuracy_2 = 0.0
    for i in range(0,y.shape[0]):
        if(prediction[i] > 0.5):
            pred_c = 0.0
        else:
            pred_c = 1.0
        actual = y[i]
        if actual == pos_cls and pred_c==pos_cls:
            TP = TP + 1.0
        if actual == neg_cls and pred_c==pos_cls:
            FP = FP + 1.0
        if actual == pos_cls and pred_c==neg_cls:
            FN = FN + 1.0
        if(y[i] == pred_c):
            acuracy_2 = acuracy_2 + 1.0
        #print y[i]-1,pred_c
    precision = precision + TP/(TP+FP)
    recall = recall + TP/(TP+FN)
    F1 = F1 + (2*precision*recall)/(precision+recall)
    acuracy_2 = (acuracy_2/len(y))*100
    true_label = []
    for i in y:
        true_label.append(int(i))
    #print true_label
    print "precision = ", precision/2,"recall = " , recall/2, "F1score = ", F1/2, "auc = ",roc_auc_score(true_label, prediction),acuracy_1, acuracy_2
    for j in range(0,len(y)):
        print prediction[j], true_label[j]
#executeLogit(
    '''

#""" Loads data, trains the MultiLogit model and learned weights """
#def get_error_softmax(x,y,theta,Number_of_class,Learning_Rate):
#    feature_size = x.shape[0]
#    xtrain = x
#    ytrain = y
#    encoder = Logit(Number_of_class, feature_size,lamda)
#    cost,gradient,error = encoder.softmax_cost(theta,xtrain,ytrain)
#    return error
#def softmax_with_optimizer(x,y,Number_of_class):
#    
#    max_iterations = 600
#    feature_size = x.shape[0]
#    #print "in here ",x.shape, y.shape
#    #print x[0]
#    #N = x.shape[0]
#    """ Initialize the MultiLogit """
#    #xtrain = x[j:j+initial_data_size]
#    #ytrain = y[j:j+initial_data_size]
#    xtrain = x
#    ytrain = y
#    encoder = Logit(Number_of_class, feature_size,lamda)
#    
#    """ Run the L-BFGS algorithm to get the optimal parameter values """
#    #encoder.cost_logit_regression(encoder.W,xtrain,ytrain)
#    #sys.exit()
#    #cost,gard = encoder.softmax_cost(encoder.theta,xtrain,ytrain)
#    #print encoder.theta.shape
#    #opt_W = encoder.theta - Learning_Rate*cost*grad
#    opt_solution  = scipy.optimize.minimize(encoder.softmax_cost, encoder.theta,args = (xtrain,ytrain,), method = 'L-BFGS-B',jac = True, options = {'maxiter': max_iterations,'disp': False})
#    opt_W     = opt_solution.x
#    #opt_W1        = opt_W[encoder.limit0 : encoder.limit1].reshape(Number_of_class, feature_size)
#    #encoder.W1    = opt_W1
#    #opt_b1        = opt_W[encoder.limit1 : encoder.limit2].reshape(Number_of_class, 1)
#    """ Visualize the obtained optimal W1 weights """
#    print opt_W.shape
#    
#    #return opt_W1,opt_b1
#    return opt_W

##executeLogit(
