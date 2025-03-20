import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from skrebate import ReliefF
import Breast_Cancer_process
import filter
import RFE

# 数据预处理
data = Breast_Cancer_process.data_pro()
X = data.drop('diagnosis', axis=1)
Y = data['diagnosis']

# 使用混合过滤器选择特征
n = 7  # 每种filter保留的特征数量
filted_feature = filter.hybrid_filter(X, Y, n)
print("Filter选择的特征:", filted_feature)

# 只使用过滤后的特征
X = X[filted_feature]
X = X.values
Y = Y.values

# 划分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=22)

# 参数设置
k = 5  # KNN的k值
p = 1
q = 1
n_features_to_select = 10  # 最终选择的特征数量

# 使用全部过滤后的特征进行初始分类
y_pred = RFE.bonferroni_knn_classifier(X_train, Y_train, X_test, k=k, p=p, q=q)
accuracy = np.mean(y_pred == Y_test)
print(f"\n使用Filter后特征的准确率: {accuracy:.4f}")

# 使用 RFE 进一步选择特征
selected_features = RFE.rfe(X_train, Y_train, X_test, Y_test, n_features_to_select, k=k, p=p, q=q)

# 将数值索引转换为特征名称
selected_feature_names = [filted_feature[i] for i in selected_features]

# 使用RFE选择的特征重新训练和测试
X_train_reduced = X_train[:, selected_features]
X_test_reduced = X_test[:, selected_features]
y_pred = RFE.bonferroni_knn_classifier(X_train_reduced, Y_train, X_test_reduced, k=k, p=p, q=q)

# 计算最终评价指标
accuracy = np.mean(y_pred == Y_test)
precision = precision_score(Y_test, y_pred, average='weighted')
recall = recall_score(Y_test, y_pred, average='weighted')
f1 = f1_score(Y_test, y_pred, average='weighted')

print("\n最终选择的特征:", selected_feature_names)
print("\n模型评估结果:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
