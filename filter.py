import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from skrebate import ReliefF
from sklearn import preprocessing
encoder = preprocessing.LabelEncoder()


# 模糊熵Fuzzy_entropy
def fuzzy_entropy(features, labels, n):
    """
    计算模糊熵 (Fuzzy Entropy)
    
    参数:
        features: 特征矩阵 (numpy数组, 形状为 (n_samples, n_features))
        labels: 标签数组 (numpy数组, 形状为 (n_samples,))
        n : 保留的特征数量
    返回:
       result: 熵最小的n个特征
    """
    if isinstance(features, pd.DataFrame):
        feature_names = features.columns
        features = features.values
    else:
        feature_names = [f'feature_{i}' for i in range(features.shape[1])]
    
    labels = labels.values
    n_samples, n_features = features.shape
    n_classes = len(np.unique(labels))
    entropy_values = np.zeros(n_features)
    
    # 计算每个特征的模糊熵
    for f in range(n_features):
        feature_values = features[:, f]
        class_entropy = 0
        # 计算每个类别的理想向量
        for c in range(n_classes):
            class_indices = np.where(labels == c)[0]
            class_mean = np.mean(feature_values[class_indices])
            # 计算相似度
            similarity = 1 - np.abs(feature_values - class_mean)

            # 计算模糊熵
            membership = np.clip(similarity, 0, 1)

            entropy = -np.sum(membership * np.log(membership + 1e-10) + 
                              (1 - membership) * np.log(1 - membership + 1e-10))

            class_entropy += entropy
        
        entropy_values[f] = class_entropy / n_classes
    
    entropy_values = pd.DataFrame({
        'Feature': feature_names,
        'fuzzy_entropy': entropy_values
    })
    
    # 移除熵最大的特征
    sorted_entropy = entropy_values.sort_values(by='fuzzy_entropy', ascending=False)
    n_feature = features.shape[1]
    n_remove = n_feature - n
    result = sorted_entropy.drop(sorted_entropy.index[:n_remove])

    return result

# reliefF算法
def reliefF(X, Y, n):
    relieff = ReliefF()
    relieff.fit(X.values, Y.values)

    feature_scores = relieff.feature_importances_
    scores_df = pd.DataFrame({'Feature': X.columns, 'Score': feature_scores})
    # 对结果排序
    sorted_scores = scores_df.sort_values(by='Score', ascending=False)
    # 保留最好的结果
    result = sorted_scores.head(n)
    return result

def hybrid_filter(X, Y, n):
    reliefF_res = reliefF(X, Y, n)
    entropy_res = fuzzy_entropy(X, Y, n)
    reliefF_feature = reliefF_res['Feature']
    entropy_feature = entropy_res['Feature']
    # 对结果取交集
    union_feature = pd.concat([reliefF_feature, entropy_feature]).unique()
    return union_feature

#print(hybrid_filter(X,Y,n))

