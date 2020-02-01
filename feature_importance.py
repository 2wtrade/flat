# -*- coding: utf-8 -*-
"""

Оценка значимости предикторов на основе модели случайного леса.

15 октября 2019

"""

# load required libraries
import pandas as pd # for convinient loading and working with data sets
import numpy as np # for working with values
import pickle # for save any models
import sys
from sklearn.ensemble import RandomForestRegressor

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

# split into predictor matrix and response
#y=df_m['Цена ДДУ'].astype(float)
y=df_m['Цена ДДУ за кв. метр'].astype(float)
X=df_m.drop(['Цена ДДУ','Цена ДДУ за кв. метр'],axis=1)

# get indexes of NaN flags for y and delete all these observations
nan_flags=np.isfinite(y)
y=y[nan_flags]
X=X[nan_flags]

# transform "Площадь " using logarithmic transformation
X["Площадь"]=X["Площадь"].astype(float) # in this case the correlation is better

# create new variable: square / rooms
X['Квадрат_комн']=X['Площадь']/pd.to_numeric(X['Количество спален'],errors='coerce', downcast = 'integer')
X.loc[list(np.isnan(X['Квадрат_комн'])),'Квадрат_комн']=np.nan
X.loc[list(np.isinf(X['Квадрат_комн'])),'Квадрат_комн']=np.nan

X["Площадь"]=np.log(X["Площадь"])

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

# load model for dummy coding
dm = pickle.load(open('./sd/dummy_model.sav', 'rb'))
# transform to dummy codes the categorical data
cX=cX.astype(str)
cX = dm.transform(cX)
cX = pd.DataFrame(cX,columns=dm.get_feature_names(X.iloc[:,np.array(flag_obj)].columns))
# combine all data sets (dummy and continious)
fX = pd.concat([nX,cX],axis=1)
# Normalize data
nrm = pickle.load(open('./sd/nrm_model_X.sav', 'rb'))
# get normalized predictor matrix
nfX = nrm.transform(fX)
nfX = pd.DataFrame(nfX, columns=fX.columns)

# fet denominator for y
d = pickle.load(open('./sd/nrm_model_y.sav', 'rb'))
ny=y/d
    
# Init Random Forest model
model = RandomForestRegressor(n_estimators=500, min_samples_leaf=5, random_state=0)
# Fit the model to the data.
model.fit(nfX,ny)
# get feature importance
importances = model.feature_importances_
# sorting by inexes
ind = np.argsort(-importances)
# get the importance in %
importances_p=100*importances/np.sum(importances)
# sorting
importances_p=np.round(importances_p[ind],3)
importances=np.round(importances[ind],3)
# save to file
imp_df=pd.DataFrame({'Feature_Name':nfX.columns[ind],"Importance":importances,"Importance_in_per":importances_p})
imp_df.to_csv('feature_importance.csv')

