import numpy as np
import lightgbm
import pandas as pd
import joblib
import torch
import shap
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

from skmultilearn.problem_transform import LabelPowerset
from utility import metrics, readfile, readname, readstructure
from sklearn.metrics import roc_curve, accuracy_score, auc, matthews_corrcoef, confusion_matrix

def getMetrics(y_true, y_pred, y_proba):
    ACC = accuracy_score(y_true, y_pred)
    MCC = matthews_corrcoef(y_true, y_pred)
    CM = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = CM.ravel()
    Sn = tp / (tp + fn)
    Sp = tn / (tn + fp)
    FPR, TPR, thresholds_ = roc_curve(y_true, y_proba)
    AUC = auc(FPR, TPR)

    Results = np.array([ACC, MCC, Sn, Sp, AUC]).reshape(-1, 5)
    Metrics_ = pd.DataFrame(Results, columns=["ACC", "MCC", "Sn", "Sp", "AUC"])

    return Metrics_

def readesm(directory, file):
    files = readname(file)
    results = []
    for sequence in files:
        name = sequence[1:]
        cur = torch.load(directory + '/' + name + '.pt')['representations'][33]
        # if cur.shape[0] < 50:
        #     cur = torch.cat((cur, torch.zeros(50 - cur.shape[0], 1280)), 0)
        results.append(cur)
    return results


sequence, label = readfile('testing.fasta')

embedding = pd.read_csv("testing.csv", header=0)
property = embedding.iloc[:, 0:188].values.tolist()
prot = [p for p in property]

column = []
categories = []

for i in range(0, 1280):
    column.append('esm' + str(i))

for i in range(0, 188):
    column.append('svmprot' + str(i))

for i in range(0, 50):
    categories.append('residue' + str(i))
    column.append('residue' + str(i))
    column.append('alpha' + str(i))
    column.append('beta' + str(i))
    column.append('coil' + str(i))
    column.append('rsa' + str(i))

esm = readesm('testesm', 'testing.fasta')
node = [item for item in esm]

strcuture = readstructure(sequence, 'testing.txt')

test = []
for i in range(0, len(sequence)):
    avg = node[i].numpy().mean(axis=0)
    test.append(avg.tolist())
    test[i].extend(prot[i])
    test[i].extend(strcuture[i])

label = np.array(label)
test_data = pd.DataFrame(test, columns=column)
test_data[categories] = test_data[categories].astype('category')

probability = np.zeros((len(label), 9), dtype=float)
predictions = np.zeros((len(label), 9), dtype=int)

for i in range(1, 10):
    lgb_model = joblib.load('lgbm{}.job'.format(i))
    cur = lgb_model.predict_proba(test_data, categorical_feature=categories)
    #positive = []

    for j in range(len(label)):
        probability[j][i - 1] = cur[j][1]
        if cur[j][1] > 0.5:
            predictions[j][i - 1] = 1
        #if label[j][i - 1] == 1:
        #    positive.append(cur[j])
    print(getMetrics(label[:, i - 1], predictions[:, i - 1], probability[:, i - 1]))
    #print(max(positive), min(positive), sum(positive) / len(positive))

    # explainer = shap.TreeExplainer(lgb_model)
    # shap_values = explainer.shap_values(test_data)
    # fig = shap.summary_plot(shap_values, test_data, plot_type="bar", max_display=10, class_names=['Negative', 'Positive'], title='Feature importance for Human acute monocytic leukemia')
    # plt.savefig('figure6_{}.png'.format(i))

    file = open('LGBM{}.txt'.format(i), 'w')
    for val in probability[:, i - 1]:
        file.write(str(val))
        file.write('\n')
    file.close()

print(metrics(label, predictions, probability))

# file = open('LGBM.txt', 'w')
# for sub in probability:
#     for val in sub:
#         file.write(str(val) + ' ')
#     file.write('\n')