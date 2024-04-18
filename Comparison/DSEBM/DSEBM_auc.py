import numpy as np
import matplotlib.pylab as plt  
from sklearn.metrics import roc_curve,auc
# np.savetxt('sorted_labels.txt', sorted_labels, fmt="%d", delimiter=" ")
# np.savetxt('sorted_energies.txt', sorted_energies, fmt="%d", delimiter=" ")
a = np.loadtxt('sorted_labels.txt')
b = np.loadtxt('sorted_energies.txt')
fpr, tpr, _ = roc_curve(a, b)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("ROC-AUC2")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.savefig("ROC_AUC2.png")
plt.close()