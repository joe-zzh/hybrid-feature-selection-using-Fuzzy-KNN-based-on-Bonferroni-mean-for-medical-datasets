import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
encoder = preprocessing.LabelEncoder()

def data_pro():
    path= 'C:\算法\复现\Covid-19-Patient-Health-Analytics-master\cvd\data.csv'
    data= pd.read_csv(path)
    data=min_max_normalization(data)
    return data

def min_max_normalization(data):
    data = data.drop('id',axis=1)
    data = data.fillna(np.nan,axis=0)
    data['age'] = data['age'].fillna(value=data['age'].mean())
    data['from_wuhan'] = data['from_wuhan'].fillna(0.0)
    data['location'] = encoder.fit_transform(data['location'].astype(str))
    data['country'] = encoder.fit_transform(data['country'].astype(str))
    data['gender'] = encoder.fit_transform(data['gender'].astype(str))
    data['symptom1'] = encoder.fit_transform(data['symptom1'].astype(str))
    data['symptom2'] = encoder.fit_transform(data['symptom2'].astype(str))
    data['symptom3'] = encoder.fit_transform(data['symptom3'].astype(str))
    data['symptom4'] = encoder.fit_transform(data['symptom4'].astype(str))
    data['symptom5'] = encoder.fit_transform(data['symptom5'].astype(str))
    data['symptom6'] = encoder.fit_transform(data['symptom6'].astype(str))

    data['sym_on'] = pd.to_datetime(data['sym_on'],errors='coerce')
    data['hosp_vis'] = pd.to_datetime(data['hosp_vis'],errors='coerce')
    data['sym_on']= data['sym_on'].map(dt.datetime.toordinal)
    data['hosp_vis']= data['hosp_vis'].map(dt.datetime.toordinal)
    data['diff_sym_hos']= data['hosp_vis'] - data['sym_on']
    data = data.drop(['sym_on','hosp_vis'],axis=1)
    scaler = MinMaxScaler()
    # 转换数据
    X_scaled = scaler.fit_transform(data)
    X_scaled_df=pd.DataFrame(X_scaled,columns=data.columns)
    return X_scaled_df

#print(data_pro().head())