from django.shortcuts import render
from django.http import HttpResponse
#import Interactive_mining as IM
import IPM_Driver as IPMD
from forms import CategoryForm, BlankForm
import Multi_logit as LR
import numpy as np
import json
from models import Pattern
import os

module_dir = os.path.dirname(__file__)  # get current directory
housefile = os.path.join(module_dir, 'trulia_transaction_WebDemo.data')
#datafile = os.path.join(module_dir, 'trulia_25_maximal_WebDemo.data')
imagehas = os.path.join(module_dir, 'images')
item_index_file = os.path.join(module_dir, 'keyword_indexing_25')

#housefile = "/media/rabbi/492827b3-643d-4801-af80-b0d24080381a/home/rabbi/work_from_Nov2016/workspace/code/IPM_demo/IPM/trulia_transaction_WebDemo.data"
#datafile = "/media/rabbi/492827b3-643d-4801-af80-b0d24080381a/home/rabbi/work_from_Nov2016/workspace/code/IPM_demo/IPM/trulia_15_WebDemo.data"
#imagehas = "/media/rabbi/492827b3-643d-4801-af80-b0d24080381a/home/rabbi/work_from_Nov2016/workspace/code/IPM_demo/IPM/images"
#item_index_file = "/media/rabbi/492827b3-643d-4801-af80-b0d24080381a/home/rabbi/work_from_Nov2016/workspace/code/IPM_demo/IPM/keywords_indexing"
iteration=0
Data = {}
x = []
no_of_data_passed_to_user=5
batch_size = 2000   
training_data_size = 30000
Number_of_class = 2
Train_set_x = []
Train_set_y = []
initialize_flag=0;
W_initial=[]
W_update = []
batch_count=batch_size;
Possible_Iteration = 10
model_update_threshold = 0.00001
model_diff=100000
data_to_pattern = {}
tran_to_pattern = {}
select_user_patterns_keywords = set()
price = []
support = []
pattern = []
keywordtoIndex = {}
imgaes = {}
content = []
pattern_index = []

def initialize_data_structures():
    global iteration, Data,x,batch_size,initialize_flag,no_of_data_passed_to_user,Train_set_x,Train_set_y,\
    batch_count,Possible_Iteration,W_initial, data_to_pattern, model_diff, model_update_threshold,tran_to_pattern,\
    select_user_patterns_keywords,Number_of_class,price,pattern,support,x_test,images,content,pattern_index
    iteration=0
    Data = {}
    x = []
    no_of_data_passed_to_user=5
    batch_size = 2000   
    training_data_size = 30000
    Number_of_class = 2
    Train_set_x = []
    Train_set_y = []
    initialize_flag=0;
    W_initial=[]
    W_update = []
    batch_count=batch_size;
    Possible_Iteration = 10
    model_update_threshold = 0.00001
    model_diff=100000
    data_to_pattern = {}
    tran_to_pattern = {}
    select_user_patterns_keywords = set()
    content = []
    price = []
    support = []
    pattern = []
    pattern_index = range(1,38176)
    np.random.shuffle(pattern_index)        
    keywordtoIndex = {}
    imgaes = {}

def collect_interesting_items(p):
    global select_user_patterns_keywords
    key_str = ""
    if p[0]:
        key_str = ",".join(p[0])
    if p[1]:
        if p[0]:
            key_str = key_str+","+ ",".join(p[1])
        else:
            key_str = key_str+ ",".join(p[1])
    p = key_str.split(",")
    print "select = ",p
    for word in p:
        select_user_patterns_keywords.add(word)
        
def get_current_batch(start,end):
    global pattern_index, data_to_pattern
    records = Pattern.objects.filter(pindex__in=pattern_index[start:end])
    data_to_pattern = []
    X = []
    data_to_pattern = {}
    for line in records:
        line =  str(line)
        data,pattern = line.strip().split("|");
        x_row = []
        
        data = data.split()
        data_str=""
        for i in range(0,len(data)):
            d = float(data[i])
            x_row.append(d)
            data_str  = data_str+str(d)
        X.append(x_row)
        if not data_to_pattern.has_key(data_str):
            data_to_pattern[data_str] = pattern
    return np.asarray(X),data_to_pattern

def get_pattern_from_data(xlabel):
    global data_to_pattern, price,support,pattern,images
    pattern = []
    price = []
    support = []
    
    for dx in xlabel:
        hasimage=[]
        noimage =[]
        temp = []
        x_str=""
        x_str = "".join([str(dxi) for dxi in dx])
        print data_to_pattern[x_str]
        pat,price_data = data_to_pattern[x_str].split("-")
        price_data = price_data.split(":")
        price_data_home = price_data[0:len(price_data)-1]
        price.append(price_data_home)
        support.append(price_data[len(price_data)-1])
        pat = pat.split(",")
        for p in pat:
            if images.has_key(p):
                hasimage.append(p)
            else:
                noimage.append(p)
        temp.append(hasimage)
        temp.append(noimage)
        pattern.append(temp)
    return pattern,price,support
def index(request):
    global iteration, Data,x,batch_size,initialize_flag,no_of_data_passed_to_user,Train_set_x,Train_set_y,\
    batch_count,Possible_Iteration,W_initial, data_to_pattern, model_diff, model_update_threshold,tran_to_pattern,\
    select_user_patterns_keywords,Number_of_class,price,pattern,support,x_test,images,pattern_index
    
    if request.is_ajax():
        if request.method == 'POST':
            
            item_index = IPMD.read_item_index(item_index_file)
            #print "hereeeeeeeeeee ", iteration
            json_data = request.POST['key']
           
            #print json_data
            json_data =  json_data.replace("\"","").split(",")
            json_data = json_data[0:len(json_data)-1]
            #print json_data
            #print item_index
            t =[]
            w = []
            F = x_test.shape[1]
            W1 = W_initial.reshape((Number_of_class,F))
            for i in json_data:
                t.append(item_index[i])
                w.append(W_initial[item_index[i]-1])
                w.append(W1[0][item_index[i]-1])
                w.append(W_initial[item_index[i]-1+F])
                w.append(W1[1][item_index[i]-1])
                W_initial[item_index[i]-1] = 0.0
                W_initial[item_index[i]-1+F] = 0.0
            #print t, W_initial.shape,w,x_test.shape
            #data = [i.replace("\"","\'")for i in json_data]
            #print data
            #print [i.replace("\\t","").replace("\\n","") for i in json_data.split(",")]
            #print [i.strip() for i in json_data.split(",")]
            #x_test,tran_to_pattern = IPMD.read_data(housefile)
            content = list(select_user_patterns_keywords)
            #print content
            temp3 = [x for x in content if x not in json_data]
            #a = select_user_patterns_keywords - set(json_data)
            #print temp3
            top_houses = LR.get_top_houses(W_initial,x_test.T,tran_to_pattern,Number_of_class,temp3)
         
            initialize_data_structures()
            
            return render(request, 'Raven/tophouses_final.html',{'content': temp3,'tophouses':top_houses})  
    
    
    if iteration==0 and initialize_flag==0:
        initialize_data_structures()
        #x,data_to_pattern = IPMD.read_data(datafile)
        xtrain,data_to_pattern = get_current_batch(0,batch_size) 
        #print "starting         :",x.shape
        #np.random.shuffle(x)
        images = IPMD.load_image_list(imagehas)
        #xtrain = x[0:0+batch_size]
        #xlabel,xunlabel = IPMD.get_representative_random(xtrain,no_of_data_passed_to_user)
        xlabel,xunlabel = IPMD.get_representative_k_center(xtrain,no_of_data_passed_to_user)
        Train_set_x = xlabel
        #pattern = []
        #for dx in xlabel:
        #    x_str = "".join(dx)
        #    pattern.append(data_to_pattern[x_str])
        pattern,price,support = get_pattern_from_data(xlabel)
        Data[iteration]=pattern
        initialize_flag=1
    #content.append(str(i))
        #i=i+1
    if request.method == 'POST':
        form = CategoryForm(request.POST)
            #form.name
        # Have we been provided with a valid form?
        if form.is_valid():
            # Save the new category to the database.
            #for key in form.cleaned_data:
            #    if BooleanField()[key]==1:
            #        Train_set_y.append[1]
            #    else:
            #        Train_set_y.append[0]
            if iteration==0 and form.cleaned_data['fb6']==False:
                FB = [0]*5
                if form.cleaned_data['fb1']==True:
                    p = Data[iteration][0]
                    collect_interesting_items(p)
                    FB[0]=1 
                if form.cleaned_data['fb2']==True:
                    p = Data[iteration][1]
                    collect_interesting_items(p)
                    FB[1]=1
                if form.cleaned_data['fb3']==True:
                    p = Data[iteration][2]
                    collect_interesting_items(p)
                    FB[2]=1    
                if form.cleaned_data['fb4']==True:
                    p = Data[iteration][3]
                    collect_interesting_items(p)
                    FB[3]=1
                if form.cleaned_data['fb5']==True:
                    p = Data[iteration][4]
                    collect_interesting_items(p)
                    FB[4]=1
                FB = np.asarray(FB)
                print "int = ", select_user_patterns_keywords
                
                Train_set_y = np.concatenate((Train_set_y,FB),axis=0)
            #Train_set_y = Train_set_y + np.asarray(form.cleaned_data.values())
                W_initial= LR.initiateLogit_collective_train(Train_set_x.T,Train_set_y.T,Number_of_class)
           
                iteration+=1
                #xtrain = x[batch_count:batch_count+batch_size]
                xtrain,data_to_pattern = get_current_batch(batch_count,batch_count+batch_size) 
                x_EGL,EGL = LR.expected_gradient_length_driver(W_initial,xtrain.T,Number_of_class)
                x_EGL = x_EGL[0:(batch_size*50)/100]
                xlabel,xunlabel = IPMD.get_representative_k_center(xtrain,no_of_data_passed_to_user)
                #xlabel,xunlabel = IPMD.get_representative_random(x_EGL,no_of_data_passed_to_user)
                Train_set_x = np.concatenate((Train_set_x,xlabel),axis=0)
                pattern,price,support = get_pattern_from_data(xlabel)
                Data[iteration] = pattern
                
                form = CategoryForm()
                return render(request, 'Raven/index.html',{'content': Data[iteration],'price':price,'form':form})
            
            elif iteration > 0 and iteration < Possible_Iteration - 1 and form.cleaned_data['fb6']==False: #and model_diff <= model_update_threshold:
                FB = [0]*5
                if form.cleaned_data['fb1']==True:
                    p = Data[iteration][0]
                    collect_interesting_items(p)
                    FB[0]=1 
                if form.cleaned_data['fb2']==True:
                    p = Data[iteration][1]
                    collect_interesting_items(p)
                    FB[1]=1
                if form.cleaned_data['fb3']==True:
                    p = Data[iteration][2]
                    collect_interesting_items(p)
                    FB[2]=1    
                if form.cleaned_data['fb4']==True:
                    p = Data[iteration][3]
                    collect_interesting_items(p)
                    FB[3]=1
                if form.cleaned_data['fb5']==True:
                    p = Data[iteration][4]
                    collect_interesting_items(p)
                    FB[4]=1
                FB = np.asarray(FB)
                print "int 1= ", select_user_patterns_keywords
                Train_set_y = np.concatenate((Train_set_y,FB),axis=0)
                W_update = LR.executeLogit_collective_train(Train_set_x.T,Train_set_y,W_initial,Number_of_class)
                norm_old = np.linalg.norm(W_initial)
                norm_new = np.linalg.norm(W_update)
                model_diff = np.fabs(norm_new- norm_old)
                W_initial = W_update
                batch_count+=batch_size
                iteration+=1
                #xtrain = x[batch_count:batch_count+batch_size]
                xtrain,data_to_pattern = get_current_batch(batch_count,batch_count+batch_size)
                x_EGL,EGL = LR.expected_gradient_length_driver(W_initial,xtrain.T,Number_of_class)
                x_EGL = x_EGL[0:(batch_size*50)/100]
                #xlabel,xunlabel = IPMD.get_representative_random(x_EGL,no_of_data_passed_to_user)
                xlabel,xunlabel = IPMD.get_representative_k_center(xtrain,no_of_data_passed_to_user)
                Train_set_x = np.concatenate((Train_set_x,xlabel),axis=0)
                pattern,price,support = get_pattern_from_data(xlabel)
                Data[iteration] = pattern
                form = CategoryForm()
                return render(request, 'Raven/index.html',{'content': Data[iteration],'price':price,'form':form})
            else:
                FB = [0]*5
                if form.cleaned_data['fb1']==True:
                    p = Data[iteration][0]
                    collect_interesting_items(p)
                    FB[0]=1 
                if form.cleaned_data['fb2']==True:
                    p = Data[iteration][1]
                    collect_interesting_items(p)
                    FB[1]=1
                if form.cleaned_data['fb3']==True:
                    p = Data[iteration][2]
                    collect_interesting_items(p)
                    FB[2]=1    
                if form.cleaned_data['fb4']==True:
                    p = Data[iteration][3]
                    collect_interesting_items(p)
                    FB[3]=1
                if form.cleaned_data['fb5']==True:
                    p = Data[iteration][4]
                    collect_interesting_items(p)
                    FB[4]=1
                FB = np.asarray(FB)
                #print "int 2= ", select_user_patterns_keywords
                Train_set_y = np.concatenate((Train_set_y,FB),axis=0)
                W_update = LR.executeLogit_collective_train(Train_set_x.T,Train_set_y,W_initial,Number_of_class)
                W_initial = W_update
                x_test,tran_to_pattern = IPMD.read_data(housefile)
                content = list(select_user_patterns_keywords)
                #top_houses = LR.get_top_houses(W_initial,x_test.T,tran_to_pattern,Number_of_class,content)
                #form = BlankForm()
                
                #initialize_data_structures()
                
                iteration=0
                initialize_flag=0;
              
                return render(request, 'Raven/tophouses.html',{'content': content})
           
        else:
            
            print form.errors
    else:
       
        form = CategoryForm()
        return render(request, 'Raven/index.html',{'content': Data[iteration],'price':price, 'form':form})
            
def about(request):
    return render(request, 'Raven/Documentation.html')
def documentation(request):
    return render(request, 'Raven/Documentation.html')

