import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gc, math, os, re
from tqdm import tqdm
from collections import Counter

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score


def lgbc(n_estimators=1000, learning_rate=0.1):
    return LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        objective="binary",
        metric="auc",
        num_leaves=2 ** 5 - 1,
        max_depth=-1,
        min_child_weight=5,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.2,
        reg_lambda=0.7,
        n_jobs=-1,
        random_state=2020,
        is_unbalance=True,
    )


xTrain = pd.read_csv("output/train2.csv", low_memory=False).values
yTrain = pd.read_csv("input/rematch/train_label.csv")["Label"].values
importances = np.load("output/importances2.npy").mean(axis=0)
featureSort = list(np.argsort(-importances)[:1000])

initFeatures = featureSort[:10]
otherFeatures = featureSort[10:]

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2020)

scores = []
for indexTrain, indexTest in kfold.split(xTrain, yTrain):
    x1, y1 = xTrain[:, initFeatures][indexTrain], yTrain[indexTrain]
    x2, y2 = xTrain[:, initFeatures][indexTest], yTrain[indexTest]
    model = lgbc(1000, 0.1)
    model.fit(x1, y1, eval_set=[(x1, y1), (x2, y2)], verbose=False, early_stopping_rounds=50)
    scores.append(roc_auc_score(y2, model.predict_proba(x2)[:, 1]))
    del x1, x2, y1, y2, model
maxScore = sum(scores) / 5
print("base------%f------initFeatures" % maxScore)
for i, f in enumerate(otherFeatures):
    scores = []
    for indexTrain, indexTest in kfold.split(xTrain, yTrain):
        x1, y1 = xTrain[:, initFeatures + [f]][indexTrain], yTrain[indexTrain]
        x2, y2 = xTrain[:, initFeatures + [f]][indexTest], yTrain[indexTest]
        model = lgbc(1000, 0.1)
        model.fit(x1, y1, eval_set=[(x1, y1), (x2, y2)], verbose=False, early_stopping_rounds=50)
        scores.append(roc_auc_score(y2, model.predict_proba(x2)[:, 1]))
        del x1, x2, y1, y2, model
    currentScore = sum(scores) / 5
    print("%d------%f------%f------%s" % (i, maxScore, currentScore, f))
    if currentScore > maxScore:
        maxScore = currentScore
        initFeatures.append(f)
    np.save("output/features", np.array(initFeatures))
