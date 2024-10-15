import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

# 加载数据
train_data = pd.read_csv('train1_icu_data.csv')
train_labels = pd.read_csv('train1_icu_label.csv', header=None)

test_data = pd.read_csv('test1_icu_data.csv')
test_labels = pd.read_csv('test1_icu_label.csv', header=None)

# 将标签转换为数值
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_labels.iloc[:, 0])
test_labels = label_encoder.transform(test_labels.iloc[:, 0])

# 选择数据分布
# 假设所有特征都是连续的，使用高斯朴素贝叶斯
model = GaussianNB()

# 训练模型
model.fit(train_data, train_labels[1:])

# 计算训练误差和交叉验证误差
train_predictions = model.predict(train_data)
train_error = 1 - accuracy_score(train_labels[1:], train_predictions)

cv_scores = cross_val_score(model, train_data, train_labels[1:], cv=5)
cv_error = 1 - cv_scores.mean()

print(f"Training Error: {train_error}")
print(f"Cross-validation Error: {cv_error}")

# 应用模型到测试集
test_probabilities = model.predict_proba(test_data)
test_predictions = model.predict(test_data)
test_error = 1 - accuracy_score(test_labels[1:], test_predictions)

print(f"Test Error: {test_error}")

# 打印分类报告
print(classification_report(test_labels[1:], test_predictions))
C_FN = 5  # 假阴性的成本
C_FP = 1  # 假阳性的成本

# 应用贝叶斯决策标准
def bayes_decision(probability, C_FN, C_FP):
    # 计算阈值
    threshold = C_FP / (C_FP + C_FN)
    return 1 if probability > threshold else 0

# 使用贝叶斯决策标准重新分类测试集
bayes_test_predictions = [bayes_decision(prob[1], C_FN, C_FP) for prob in test_probabilities]
bayes_test_error = 1 - accuracy_score(test_labels[1:], bayes_test_predictions)

print(f"Test Error with Bayes Decision: {bayes_test_error}")