import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def data_pro():
    path = 'Breast-Cancer-Wisconsin/wdbc.data'
    # 读取数据，不指定列名，因为数据文件没有表头
    data = pd.read_csv(path, header=None)
    
    # 设置列名
    columns = ['id', 'diagnosis']
    feature_names = ['radius', 'texture', 'perimeter', 'area', 'smoothness', 
                    'compactness', 'concavity', 'concave_points', 'symmetry', 'fractal_dimension']
    
    # 添加特征列名
    for feature in feature_names:
        columns.extend([f'{feature}_mean', f'{feature}_se', f'{feature}_worst'])
    
    data.columns = columns
    
    # 将诊断结果转换为数值（M=1, B=0）
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    
    # 删除ID列
    data = data.drop('id', axis=1)
    
    # 数据标准化
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(data)
    X_scaled_df = pd.DataFrame(X_scaled, columns=data.columns)
    
    return X_scaled_df 