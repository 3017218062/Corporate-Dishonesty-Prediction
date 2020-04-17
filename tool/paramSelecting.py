import json
import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

X = pd.read_csv("../output/train2.csv", low_memory=False).values
Y = pd.read_csv("../input/rematch/train_label.csv")["Label"].values
importances = np.load("../output/importances2.npy").mean(axis=0)
importances = np.where(importances > 0.03)[0]
X = X[:, importances]
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=2020)


def evaluate(min_child_weight, subsample, colsample_bytree, reg_alpha, reg_lambda):
    params = {
        "n_estimators": 300, "learning_rate": 0.1,
        "objective": "binary", "metric": "auc",
        "num_leaves": 2 ** 5 - 1, "max_depth": -1, "min_child_weight": int(min_child_weight),
        "subsample": float(subsample), "colsample_bytree": float(colsample_bytree),
        "reg_alpha": float(reg_alpha), "reg_lambda": float(reg_lambda),
        "n_jobs": -1, "random_state": 2020, "is_unbalance": True,
    }
    true, pred = [], []
    for i, (indexTrain, indexTest) in enumerate(kfold.split(X, Y)):
        model = LGBMClassifier(**params)
        x1, y1 = X[indexTrain], Y[indexTrain]
        x2, y2 = X[indexTest], Y[indexTest]
        model.fit(
            x1, y1,
            eval_set=[(x1, y1), (x2, y2)],
            verbose=False, early_stopping_rounds=50,
        )
        true += list(y2)
        pred += list(model.predict_proba(x2)[:, 1])
    return roc_auc_score(true, pred)


def saveResult(res):
    columns = ["target"] + list(res[0]["params"].keys())
    results = []
    for i in res:
        result = []
        result.append(i["target"])
        for j in i["params"].values():
            result.append(j)
        results.append(result)
    results = pd.DataFrame(np.asarray(results), columns=columns)
    print(results)
    results.to_csv("../output/params.csv", index=False)


def getBestParams(res):
    bestScore, bestParams = 0, None
    for i in res:
        if i["target"] > bestScore:
            bestScore = i["target"]
            bestParams = i["params"]
    print("The best score is %f" % bestScore)
    return bestParams


optimizer = BayesianOptimization(
    evaluate,
    {
        "min_child_weight": (1, 5),
        "subsample": (0.5, 1.0),
        "colsample_bytree": (0.5, 1.0),
        "reg_alpha": (0.0, 0.5),
        "reg_lambda": (0.5, 1.0),
    },
    random_state=2020,
)
optimizer.maximize(init_points=5, n_iter=50)
saveResult(optimizer.res)
bestParams = getBestParams(optimizer.res)
json.dump(bestParams, open("../output/bestParams.json", 'w'))
