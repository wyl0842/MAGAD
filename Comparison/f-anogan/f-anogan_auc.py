import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import pandas as pd
import numpy as np

plt.close()
df = pd.read_csv("score.csv")# 读csv
#df = pd.to_csv("score.csv")
trainig_label = 1
# labels = np.where(df["label"].values == trainig_label, 0, 1) # 라벨이 1이면 0을 반환, 아니면 1을 반환.
labels = np.where(df["label"].values == trainig_label, 1, 0)
anomaly_score = df["anomaly_score"].values# 把一整列提取出来
img_distance = df["img_distance"].values
z_distance = df["z_distance"].values
fpr, tpr, _ = roc_curve(labels, img_distance)#绘制ROC曲线
precision, recall, _ = precision_recall_curve(labels, img_distance)#计算准确率和召回率
roc_auc = auc(fpr, tpr)
pr_auc =  auc(recall, precision)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("ROC-AUC")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.savefig("ROC_AUC3.png")
plt.close()
#plt.show()