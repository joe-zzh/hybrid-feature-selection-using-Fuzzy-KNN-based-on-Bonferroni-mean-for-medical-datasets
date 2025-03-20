import numpy as np
import pandas as pd
import Covid_Data_process
from sklearn.model_selection import train_test_split
# Bonferroni 均值计算函数
def bonferroni_mean(Z, p=1, q=1):
    n = len(Z)
    if n < 2:
        raise ValueError("输入向量的长度必须至少为 2")
    
    Z_p = Z ** p
    Z_q = Z ** q
    Z_q_mean = (np.sum(Z_q) - Z_q) / (n - 1)
    
    bonferroni_mean = (np.mean(Z_p * Z_q_mean)) ** (1 / (p + q))
    return bonferroni_mean

# Bonferroni k-NN 分类器
def bonferroni_knn_classifier(X_train, Y_train, X_test, k=100, p=1, q=1):
    n_test_samples = X_test.shape[0]
    y_pred = np.zeros(n_test_samples)

    for i in range(n_test_samples):
        distances = np.linalg.norm(X_train - X_test[i], axis=1)
        # 处理距离为0的情况
        distances = np.where(distances < 1e-10, 1e-10, distances)
        
        nearest_indices = np.argsort(distances)[:k]
        nearest_labels = Y_train[nearest_indices]
        
        unique_labels = np.unique(nearest_labels)
        # 计算每个类别的Bonferroni均值和对应的隶属度
        membership_degrees = {}
        
        for label in unique_labels:
            class_indices = nearest_labels == label
            if np.sum(class_indices) < 2:
                membership_degrees[label] = 0
                continue
                
            # 计算Bonferroni均值
            Br = bonferroni_mean(X_train[nearest_indices][class_indices], p=p, q=q)
            
            # 计算样本到理想向量的距离
            deu = np.linalg.norm(X_test[i] - Br)
            
            # 计算隶属度
            distances_to_class = distances[nearest_indices][class_indices]
            m = p + q  # 模糊因子
            
            # 防止出现除零或无效值
            power = 2 / (m - 1) if m != 1 else 2
            numerator = np.sum(1 / (distances_to_class ** power))
            denominator = np.sum(1 / (distances[nearest_indices] ** power))
            
            if denominator == 0:
                membership_degrees[label] = 0
            else:
                membership_degrees[label] = numerator / denominator
            
        # 选择具有最大隶属度的类别
        y_pred[i] = max(membership_degrees.items(), key=lambda x: x[1])[0]

    return y_pred

# 特征重要性评估函数
def evaluate_feature_importance(X_train, Y_train, X_test, Y_test, feature_idx, current_features, k=100, p=1, q=1):
    reduced_features = [f for f in current_features if f != feature_idx]
    
    X_train_reduced = X_train[:, reduced_features]
    X_test_reduced = X_test[:, reduced_features]
    
    y_pred_reduced = bonferroni_knn_classifier(X_train_reduced, Y_train, X_test_reduced, k=k, p=p, q=q)
    accuracy_reduced = np.mean(y_pred_reduced == Y_test)
    
    y_pred_full = bonferroni_knn_classifier(X_train[:, current_features], Y_train, X_test[:, current_features], k=k, p=p, q=q)
    accuracy_full = np.mean(y_pred_full == Y_test)

    return accuracy_full - accuracy_reduced

# RFE 算法
def rfe(X_train, Y_train, X_test, Y_test, n_features_to_select, k=100, p=1, q=1):
    selected_features = list(range(X_train.shape[1]))
    
    while len(selected_features) > n_features_to_select:
        feature_importance = []
        for feature_idx in selected_features:
            importance = evaluate_feature_importance(X_train, Y_train, X_test, Y_test, 
                                                     feature_idx, selected_features, k=k, p=p, q=q)
            feature_importance.append(importance)
        
        least_important_feature = selected_features[np.argmin(feature_importance)]
        selected_features.remove(least_important_feature)
        print(selected_features)
    return selected_features

