# -*- coding: utf-8 -*-
"""

Данный модуль предназначен для глубокого обучения нейронной сети по исходным (обучающим) данным, 
где оптимизируемая функция – среднее абсолютное отклонение прогнозируемой величины от известной 
величины (цена ДДУ) в процентах (точность). Обучение прекращается при достижении точности 95% и выше.
Измерения точности проводятся методом Repeated random sub-sampling validation.
Архитектура применяемой сети - нейронная сеть Ворда.
В исходных данных допускаются пропущенные значения. Дополнительно применяются такие методы как 
dummy-coding, robust scaling, fixing missing values. Все модели и служебные данные сохраняются 
в папке sd. 

При прогнозировании важным условием является соответствие наименований столбцов 
в обучающих и новых данных!

10 октябя 2019

"""
# load required libraries
import pandas as pd # for convinient loading and working with data sets
import numpy as np # for working with values
from sklearn.experimental import enable_iterative_imputer # for IterativeImputer
from sklearn import impute # for handle missing values
from sklearn import preprocessing # for preprocessiong data
import pickle # for save any models
import torch
import random
import dnn_class as dnn

# load data sets
df1=pd.read_excel(io='data2.xlsx',sheet_name=0, parse_dates=True, dtype=object)
print(df1.shape)
df2=pd.read_excel(io='data2.xlsx',sheet_name=1, parse_dates=True, dtype=object)
print(df2.shape)

#  merge data frames
df1.rename(columns={'ID ЖК':'ID_ЖК'}, inplace=True) # rename column with ID_ЖК
df_m = pd.merge(left=df2, right=df1, on='ID_ЖК', how="outer") 
print(df_m.shape)

# we will delete ID_ЖК variable (it will not use for forecasting)
df_m=df_m.drop(['ID_ЖК'],axis=1)
print(df_m.shape)

# we have 45 variables.

# split into predictor matrix and response
#y=df_m['Цена ДДУ'].astype(float)
y=df_m['Цена ДДУ за кв. метр'].astype(float)
X=df_m.drop(['Цена ДДУ','Цена ДДУ за кв. метр'],axis=1)
# add new predictor

# Save the names of variable to file for checking when we will forecast new data
r_names=pd.DataFrame({'r_names':X.columns})
r_names.to_csv('./sd/r_names.csv')

# get indexes of NaN flags for y and delete all these observations
nan_flags=np.isfinite(y)
y=y[nan_flags]
X=X[nan_flags]
print(X.shape)

# generate input file example
X.iloc[[100,4000],:].to_excel('input_example.xlsx',index=False, na_rep='')

# Ok. Everything is ready. Look at the types of columns
X=X.astype(str)
X["Площадь"]=X["Площадь"].astype(float) # in this case the correlation is better

# create new variable: square / rooms
X['Квадрат_комн']=X['Площадь']/pd.to_numeric(X['Количество спален'],errors='coerce', downcast = 'integer')
X.loc[list(np.isnan(X['Квадрат_комн'])),'Квадрат_комн']=np.nan
X.loc[list(np.isinf(X['Квадрат_комн'])),'Квадрат_комн']=np.nan

X["Площадь"]=np.log(X["Площадь"])

X.dtypes

# Now we will check the missing values
# count the number of NaN values in each column
print(X.isnull().sum())
# ok. There are a lot of missing values for some variables

# split numeric and categorical variables for imputing 
flag_num=X.dtypes!="object"
nX=X.iloc[:,np.array(flag_num)]
print(nX.isnull().sum())
flag_obj=X.dtypes=="object"
cX=X.iloc[:,np.array(flag_obj)]
print(cX.isnull().sum())

# fix missing values in numerical predictors

# fit model for fixing
imp = impute.SimpleImputer()
imp.fit(nX)
# save to file this model
pickle.dump(imp, open('./sd/imp_model.sav', 'wb'))

# imputing missing values for numeric predictors
n_col=X.columns[flag_num]
nX = imp.transform(nX)
nX = pd.DataFrame(nX,columns=n_col)
print(nX.isnull().sum())
# ok. The numeric data is ready for model

# Some variables are categorical. We will transform all categorical 
# varianles to dummies codes. The missing values will be consider as 
# additional case of value with name 'missing_value'

# ok.
# conver all values to sstrings
cX=cX.astype(str)
  
# fit model for dummy coding
dm = preprocessing.OneHotEncoder(#handle_unknown='ignore', 
                                 sparse=False,
                                 drop='first')
dm.fit(cX)

# save this model
pickle.dump(dm, open('./sd/dummy_model.sav', 'wb'))
# transform categorical variables
cX = dm.transform(cX)
cX = pd.DataFrame(cX,columns=dm.get_feature_names())
# Ok. all dummy codes has been created
print(cX.isnull().sum())
# ok.
# combine numeric and dummy data frame into one
fX = pd.concat([nX,cX],axis=1)
print(fX.shape)

# Now we have 212 features and try to robust normalize all of them
nrm = preprocessing.RobustScaler().fit(fX)

# save this model to the file
pickle.dump(nrm, open('./sd/nrm_model_X.sav', 'wb'))

# get normalized predictor matrix
nfX = nrm.transform(fX)

# get normilized y
d=np.mean(y)
ny=y/d
pickle.dump(d, open('./sd/nrm_model_y.sav', 'wb'))

# Transform all variables to tensors
nfX=torch.from_numpy(nfX)
nfX=nfX.float()
ny=torch.from_numpy(np.asarray(ny))
ny=ny.float()
ny=ny.unsqueeze(1)
# ok. 

# Now we are ready to search the best forecasting Neural Network Model.

# Input size
inp_s = nfX.size()[1]
# initialize the model
model =dnn.DNN(inp_s)
# set optimizer
optimizer = torch.optim.Adamax(model.parameters(), lr=0.01)
# set the train mode
model.train()
# set loss function
criterion = dnn.mape_loss
# validation fraction
validation_fraction=0.2 # 20% for validation
# Fitting process
max_iter=20000
# Validation accuracy vector
avs=10
val_ac_V=np.zeros(avs)
k=0
av_ac_val=0
for epoch in range(max_iter):
    # split data into taining and validation sets by random
    n = nfX.size()[0]
    r_ind = random.sample(range(n), n)
    m = round(n*validation_fraction)
    train_ind = r_ind[0:(n-m)]
    val_ind = r_ind[(n-m):n]
    X_train = nfX[train_ind,:]
    y_train = ny[train_ind]
    X_val = nfX[val_ind,:]
    y_val = ny[val_ind]
    # Set fradients to zero
    optimizer.zero_grad()    
    # Forward pass
    #pr = model(nfX) # get predictions
    pr_train = model(X_train) # get predictions for training data set
    pr_val = model(X_val) # get predictions for validation data set
    # Compute Loss
    train_loss = criterion(pr_train, y_train) # loass function
    val_loss = criterion(pr_val, y_val) # loass function
    # Backward pass
    train_loss.backward() # estimate gradient for training loss
    optimizer.step() # parameters update
    # Set max ac on validatin data set
    #if max_ac_val<get_ac(val_loss):
    #    max_ac_val = get_ac(val_loss)
    #    max_epoch = epoch
    val_ac_V[k]=dnn.get_ac(val_loss)  
    av_ac_val=round(np.mean(val_ac_V),1)
    k+=1
    if k==avs: 
        k=0
    # print results    
    if(epoch%10==0):
        print("\nCurrent epoch:",epoch)
        print("Accuracy on training data set:",dnn.get_ac(train_loss),"%")
        print("Accuracy on validation data set:",dnn.get_ac(val_loss),"%")      
        print("Averaged accuracy on validation data set:",av_ac_val,"%")
    # stop when the accuracy on validation data set >=95%
    if av_ac_val>95.0: 
        print("\n")
        print("The algorithm has been finished!")
        print("Final averaged accuracy on validation data set:",av_ac_val,"%")
        break

# Save model
torch.save(obj=model.state_dict(),f="./sd/dnn_model.pth")

# Save true value and prediction
# Set model to eval mode
model.eval()

# Get predictions for loaded data set
predictions = model(nfX)*d
predictions=predictions[:,0].detach().numpy()
res=pd.DataFrame({'Цена ДДУ за кв. метр':y.to_numpy(),
                  'Цена ДДУ за кв. метр(прогноз по данным)':predictions})
# save to file
res.to_csv('trained_results.csv')


