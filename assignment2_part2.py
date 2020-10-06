# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 20:47:02 2020

@author: Ask
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix as cm
from joblib import dump #, load

from sklearn import svm

def train(c, g):
    svc = svm.SVC(kernel='rbf', C=c, gamma=g)
    svc.fit(X_train, y_train)
    return svc

X, y = datasets.load_digits(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

c_range = np.arange(-3, 9, 0.5)
gamma_range = np.arange(-9,-2.0,0.25)
scores_1d = []
scores_2d = []

#Not a while loop, but does the same thing...
for c in c_range:
    row = []
    for g in gamma_range:
        score = train(10**c, 10**g).score(X_test, y_test)
        scores_1d.append((c,g,score))
        row.append(score)
        print(c,g,score)
    scores_2d.append(row)

best = max(scores_1d, key=lambda e:e[2])
svc = train(10**(best[0]), 10**(best[1]))
pred = svc.predict(X_test)

#graphics
adjusted_arr = [[1/(1-s) for s in row] for row in scores_2d]
fig, ax = plt.subplots()
im = ax.imshow(np.array(adjusted_arr), cmap="hot")
ax.set_xticks(np.arange(len(gamma_range))[::2])
ax.set_yticks(np.arange(len(c_range))[::2])
ax.set_xticklabels([f"10^{round(g,1)}" for g in gamma_range][::2])
ax.set_yticklabels([f"10^{round(c,1)}" for c in c_range][::2])
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

print(f"best parameters:")
print(f"c:       10^{round(best[0],1)}")
print(f"gamma:   10^{round(best[1],1)}")
print(f"accuracy: {round(best[2],4)}")

print(cm(pred, y_test))

dump(svc,"RBF_best_model.joblib")
