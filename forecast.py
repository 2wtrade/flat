# -*- coding: utf-8 -*-
"""

Данный модуль предназначен для прогнозирования цены ДДУ на основе предтренированной 
нейронной сети, для которой средняя абсолютная точность прогнозирования не ниже 95%.
При прогнозировании важным условием является соответствие наименований столбцов 
в обучающих и новых данных! Если это не выполняется, необходимо отредактировать 
данные для прогнозирования или обучающие данные. В крайнем случае можно переобучить модель.

На выходе формируется файл с исходными данными + прогнозируемой ценой по ДДУ для
каждой позиции.

10 октября 2019

"""

# load required libraries
import pandas as pd # for convinient loading and working with data sets
import numpy as np # for working with values
#from sklearn.experimental import enable_iterative_imputer # for IterativeImputer
#from sklearn import impute # for handle missing values
#from sklearn import preprocessing # for preprocessiong data
import pickle # for save any models
import torch
#import  torch.nn.functional as F
#import torch.nn as nn#
#import random
import sys
import dnn_class as dnn


# Output file
out_file='output.xlsx'

# load data for forecasting
df=pd.read_excel('input_example.xlsx',dtype=object)

# load all required names
r_names=pd.read_csv('./sd/r_names.csv',index_col=0)

# check names in file with required names
if list(r_names.r_names)==list(df.columns):
    print("The data have been loaded successfully!")
else:
    sys.exit("The column names in predicted data is not matched with name columns of training data!")

# preparing data set    
X=df.copy()

X=X.astype(str)
X["Площадь"]=X["Площадь"].astype(float) # in this case the correlation is better

# create new variable: square / rooms
X['Квадрат_комн']=X['Площадь']/pd.to_numeric(X['Количество спален'],errors='coerce', downcast = 'integer')
X.loc[list(np.isnan(X['Квадрат_комн'])),'Квадрат_комн']=np.nan
X.loc[list(np.isinf(X['Квадрат_комн'])),'Квадрат_комн']=np.nan

X["Площадь"]=np.log(X["Площадь"])
df['Квадрат_комн']=X['Квадрат_комн'].copy()

# split numeric and categorical variables for imputing 
flag_num=X.dtypes!="object"
nX=X.iloc[:,np.array(flag_num)]
flag_obj=X.dtypes=="object"
cX=X.iloc[:,np.array(flag_obj)]

# load models for fixing numeric missing values
imp = pickle.load(open('./sd/imp_model.sav', 'rb'))
n_col=X.columns[flag_num]
nX = imp.transform(nX)
nX = pd.DataFrame(nX,columns=n_col)
# prepare missing values of categorical data
cX=cX.fillna('missing_values')
cX=cX.astype(str)

# load model for dummy coding
dm = pickle.load(open('./sd/dummy_model.sav', 'rb'))
# transform to dummy codes the categorical data
cX = dm.transform(cX)
cX = pd.DataFrame(cX,columns=dm.get_feature_names())
# combine all data sets (dummy and continious)
fX = pd.concat([nX,cX],axis=1)
# Normalize data
nrm = pickle.load(open('./sd/nrm_model_X.sav', 'rb'))
# get normalized predictor matrix
nfX = nrm.transform(fX)
# fet denominator for y
d = pickle.load(open('./sd/nrm_model_y.sav', 'rb'))
# Transform predictors to tensors
nfX=torch.from_numpy(nfX)
nfX=nfX.float()

# Input size
inp_s = nfX.size()[1]
# initialize the model
model = dnn.DNN(inp_s)
# Load pretrained waights
model.load_state_dict(torch.load('./sd/dnn_model.pth'))
# Set model to eval mode
model.eval()

# Get predictions for loaded data set
pr = model(nfX)*d
pr=pr[:,0].detach().numpy()

# Save results
df['Цена ДДУ(прогноз по данным)']=pr
df.to_excel(out_file)

print("The forecasting has been made successfuly! \nThe output file is:",out_file)


