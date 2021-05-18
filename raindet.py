#Importing all the reqired libraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statistics import mean 

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import pickle
data=pd.read_csv("temphumdetails.csv")
data=pd.DataFrame(data)
def fahr_to_celsius(temp_fahr):
    """Convert Fahrenheit to Celsius
    
    Return Celsius conversion of input"""
    temp_celsius = (temp_fahr - 32) * 5 / 9
    return temp_celsius

data["tempC"] = fahr_to_celsius(data["TempAvgF"])
data['Events'].replace(' ',np.nan, inplace=True)
data["Events"].replace([np.nan,"Fog"],0,inplace=True)
data["Events"].replace(["Rain , Thunderstorm","Rain","Rain , Snow","Fog , Rain , Thunderstorm","Thunderstorm","Fog , Thunderstorm","Fog , Rain"],1,inplace=True)

data1=pd.DataFrame(data[['tempC',"HumidityAvgPercent","Events"]])

data1['HumidityAvgPercent']=data1['HumidityAvgPercent'].astype(str)

def convert(data):
    number = preprocessing.LabelEncoder()
    data['HumidityAvgPercent'] = number.fit_transform(data['HumidityAvgPercent'])
    data=data.fillna(-999) # fill holes with default value
    return data


data1=convert(data1)

data1=np.array(data1)

X = data1[0:, 0:2]
y = data1[0:, -1]

y = y.astype('int')
X = X.astype('float')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

lr=LogisticRegression()
lr.fit(X_train,y_train)

file=open('logisticrefrain.pkl','wb')
pickle.dump(lr,file)
