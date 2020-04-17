import numpy as np
import pandas as pd
import gc, math, os, sys
from collections import Counter

from lightgbm import LGBMClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score


class myModel:
    def __init__(self):
        print("1. Loading ...")
        self.dataPreprocessing()
        print("2. Transforming ...")
        self.featureEngineering()
        print("3. Building ...")
        self.modelBuilding()
        print("4. Choosing ...")
        self.featureChoosing()
        print("5. Fitting ...")
        self.modelFitting()
        print("6. Ensembling ...")
        self.modelEnsembling()
        print("7. Predicting ...")
        self.modelPredicting()
        print("8. Adjusting ...")
        self.ruleProcessing()

    def dataPreprocessing(self):
        self.path = "input/rematch/"
        self.xTrain, self.xPred, self.yTrain = None, None, None
        self.transformed = True
        if os.path.exists("output/script/train.csv") and os.path.exists("output/script/test.csv"):
            print("\033[32m  Open train data.\033[0m")
            self.xTrain = pd.read_csv("output/script/train.csv", low_memory=False)
            print("\033[32m  Open test data.\033[0m")
            self.xPred = pd.read_csv("output/script/test.csv", low_memory=False)
            self.transformed = False
        else:
            print("\033[32m  Open train data.\033[0m")
            self.xTrain = pd.read_csv(self.path + "train_stage2_update_20200320.csv", low_memory=False).drop(["ID"],
                                                                                                             axis=1)
            print("\033[32m  Open test data.\033[0m")
            self.xPred = pd.read_csv(self.path + "test_stage2_update_20200320.csv", low_memory=False).drop(["ID"],
                                                                                                           axis=1)
        self.yTrain = pd.read_csv(self.path + "train_label.csv")["Label"].values
        self.K, self.ensembleNum = 10, 10
        self.features = []

    def featureEngineering(self):
        def dataClearing(data):
            print(signal[0] % "dataClearing")
            thresholds = {
                "投资总额": 10000,
                "注册资本": 5000,
                "增值税": 1000,
                "企业所得税": 500,
                "教育费": 40,
                "城建税": 80,
                "诉讼总数量": 100,
                "最新参保人数": 500,
                "年度参保总额": 1500,
            }

            indices = []
            for i in thresholds.keys():
                index = data[i].fillna(0).values >= thresholds[i]
                print(signal[1] % (i, index.sum()))
                indices += list(np.where(index)[0])
            indices = list(set(range(data.shape[0])) - set(indices))
            indices.sort()
            np.save("output/script/indices", np.array(indices))
            data = data.iloc[indices, :]

            gc.collect()
            print(signal[-1] % "dataClearing")
            return data.iloc[:, :-1], data["Label"].values

        def dataEncoding(data):
            def getTime(x):
                return float(x.split(":")[0]) * 60 + float(x.split(":")[1]) if type(x) == str else x

            print(signal[0] % "dataEncoding")
            catFeatures = ["企业类型", "登记机关", "企业状态", "行业代码", "行业门类", "企业类别", "管辖机关", "诉讼总数量", "资本变更次数", "纳税人状态代码",
                           "登记注册类型代码"]
            timeFeatures = ["经营期限自", "经营期限至", "成立日期", "核准日期", "注销时间"]

            print(signal[2] % "category")
            for i in catFeatures:
                data[i] = LabelEncoder().fit_transform(data[i])

            print(signal[2] % "time")
            for i in timeFeatures:
                data[i] = data[i].apply(lambda x: getTime(x))

            print(signal[2] % "special")
            data["邮政编码"] = data["邮政编码"].apply(
                lambda x: float(x) if str(x)[:6].isdigit() and float(x) >= 1e+5 else np.nan)
            data["省"] = data["邮政编码"].apply(lambda x: x // 1e+4 if x else np.nan)
            data["邮区"] = data["邮政编码"].apply(lambda x: x // 1e+3 if x else np.nan)
            data["邮局"] = data["邮政编码"].apply(lambda x: x // 1e+2 if x else np.nan)
            data["投递局"] = data["邮政编码"].apply(lambda x: x % 1e+2 if x else np.nan)
            data["邮政编码"] = LabelEncoder().fit_transform(data["邮政编码"])
            data["省"] = LabelEncoder().fit_transform(data["省"])
            data["邮区"] = LabelEncoder().fit_transform(data["邮区"])
            data["邮局"] = LabelEncoder().fit_transform(data["邮局"])
            data["投递局"] = LabelEncoder().fit_transform(data["投递局"])
            data["new经营范围"] = data["经营范围"].apply(lambda x: np.asarray([int(i) for i in str(x)[1:-1].split(", ")]))
            data["经营范围"] = data["new经营范围"].apply(lambda x: len(x))
            data["经营范围"] = LabelEncoder().fit_transform(data["经营范围"])

            gc.collect()
            print(signal[-1] % "dataEncoding")
            return data

        def featureCreating(data):
            print(signal[0] % "featureCreating")
            allFeatures = list(data.columns)
            catFeatures = ["企业类型", "登记机关", "企业状态", "行业代码", "行业门类", "企业类别", "管辖机关", "诉讼总数量", "资本变更次数", "纳税人状态代码",
                           "登记注册类型代码"]
            timeFeatures = ["经营期限自", "经营期限至", "成立日期", "核准日期", "注销时间"]
            taxFeatures = ["印花税", "企业所得税", "增值税", "城建税", "教育费"]
            specialFeatures = ["邮政编码", "省", "邮区", "邮局", "投递局", "经营范围", "new经营范围"]
            deleteFeatures = ["长期负债合计_年初数", "其他负债（或长期负债）_年初数", "其他应交款_年初数", "应付福利费_年初数", "预提费用_年初数", "长期负债合计_年末数",
                              "其他负债（或长期负债）_年末数", "其他应交款_年末数", "应付福利费_年末数", "预提费用_年末数", "待摊费用_年初数", "应收补贴款_年初数",
                              "长期投资合计_年末数", "待摊费用_年末数", "固定资产净额_年末数", "固定资产净值_年末数", "无形资产及其他资产合计_年末数", "应收补贴款_年末数"]
            fillFeatures = ["%d月案件数" % i for i in range(1, 13)] + \
                           ["%d类诉讼数量" % i for i in range(1, 452)] + \
                           ["处罚程度_%d" % i for i in range(3)] + \
                           ["处罚类型_%d" % i for i in range(3)] + \
                           ["未知月份案件数", "处罚程度_未知", "处罚类型_未知"]
            otherFeatures = list(set(allFeatures) - set(catFeatures) - set(timeFeatures) - set(taxFeatures) - \
                                 set(specialFeatures) - set(deleteFeatures) - set(fillFeatures))

            # data["nan_num"] = data.isna().sum(axis=1).values

            print(signal[3] % "tax")
            tax = data[taxFeatures].fillna(0)
            data["总税"] = tax[taxFeatures].sum(axis=1)
            for i in range(len(taxFeatures)):
                f1 = taxFeatures[i]
                data[f1 + "exp"] = tax[f1].apply(lambda x: math.exp(x / 100))
                data[f1 + "ln"] = tax[f1].apply(lambda x: math.log(x) if x > 0 else 0)
                data[f1 + "/" + "总税"] = tax[f1] / (data["总税"] + 1e-7)
                data["总税" + "-" + f1] = data["总税"] - tax[f1]
                for j in range(i + 1, len(taxFeatures)):
                    f2 = taxFeatures[j]
                    data[f1 + "/" + f2] = tax[f1] / (tax[f2] + 1e-7)
                    data[f2 + "/" + f1] = tax[f2] / (tax[f1] + 1e-7)
                    data[f1 + "*" + f2] = tax[f1] * tax[f2]
                    data[f1 + "-" + f2] = tax[f1] - tax[f2]
                    data[f1 + "+" + f2] = tax[f1] + tax[f2]
                    for k in range(j + 1, len(taxFeatures)):
                        f3 = taxFeatures[k]
                        data[f1 + "*" + f2 + "*" + f3] = tax[f1] * tax[f2] * tax[f3]
                        data[f1 + "+" + f2 + "+" + f3] = tax[f1] + tax[f2] + tax[f3]

            print(signal[3] % "special")
            data["是否全资"] = data["注册资本"] >= data["投资总额"]
            data["资本变更"] = data["资本变更后"] - data["资本变更前"]
            data["注册资本/投资总额"] = data["注册资本"] / (data["投资总额"] + 1e-7)
            data["参保人数/参保总额"] = data["最新参保人数"] / (data["年度参保总额"] + 1e-7)
            scopeNum, mergeNum = 0, 100
            businessScope = np.zeros((data.shape[0], 30000 // mergeNum), dtype=np.uint8)
            for i, j in enumerate(data["new经营范围"].values):
                currentMax = j.max()
                scopeNum = currentMax if currentMax > scopeNum else scopeNum
                tmp = Counter(j // mergeNum)
                businessScope[i, list(tmp.keys())] = list(tmp.values())
            scopeNum = (scopeNum + 1) // mergeNum + 1
            businessScope = businessScope[:, :scopeNum]
            data = pd.concat([data, pd.DataFrame(
                businessScope, columns=np.array(["经营范围_%d" % i for i in range(scopeNum)]), index=data.index
            )], axis=1)
            del data["new经营范围"]

            print(signal[3] % "delete")
            for i in deleteFeatures:
                del data[i]

            print(signal[3] % "fill")
            data = data.fillna(dict(zip(fillFeatures, [0] * len(fillFeatures))))

            print(signal[3] % "other", end="")
            uselessFeatures = []
            for i in otherFeatures:
                if (data[i].fillna(0).values == 0).sum() > data.shape[0] * 0.9:
                    uselessFeatures.append(i)
            otherFeatures = list(set(otherFeatures) - set(uselessFeatures))
            otherFeatures.sort()
            np.random.seed(2020)
            np.random.shuffle(otherFeatures)
            for i in range(len(otherFeatures)):
                f1 = otherFeatures[i]
                data[f1 + "exp"] = data[f1].apply(lambda x: math.exp(x / 10000) if x else np.nan)
                print("\r" + signal[3] % ("%s / %s" % (i + 1, len(otherFeatures))), end="")
                for j in range(i + 1, len(otherFeatures)):
                    f2 = otherFeatures[j]
                    data[f1 + "+" + f2] = data[f1] + data[f2]
                    data[f1 + "-" + f2] = data[f1] - data[f2]
                    data[f1 + "*" + f2] = data[f1] * data[f2]
                    data[f1 + "/" + f2] = data[f1] / (data[f2] + 1e-7)
            print("\r" + signal[3] % "other")

            gc.collect()
            print(signal[-1] % "featureCreating")
            return data

        def creditCardRating(splitLine, data):
            def optimalBinningBoundary(x, y):
                clf = DecisionTreeClassifier(min_samples_leaf=0.001, max_leaf_nodes=32)
                clf.fit(x.reshape(-1, 1), y)

                nodeNum = clf.tree_.node_count
                childrenLeft, childrenRight = clf.tree_.children_left, clf.tree_.children_right
                threshold = clf.tree_.threshold

                boundary = []
                for i in range(nodeNum):
                    if childrenLeft[i] != childrenRight[i]:
                        boundary.append(threshold[i])

                boundary.sort()
                return [min(x)] + boundary + [max(x) + 0.1]

            def caculateWoeIv(x, y, boundary):
                table = pd.concat([pd.Series(x), pd.Series(y)], axis=1)
                table.columns = ["x", "y"]
                table["bins"] = pd.cut(x=x, bins=boundary, right=False)
                grouped = table.groupby("bins")["y"]
                result = grouped.agg([
                    ("good", lambda y: (y == 0).sum()),
                    ("bad", lambda y: (y == 1).sum()),
                    ("total", "count")
                ])

                result["good_prop"] = result["good"] / result["good"].sum()
                result["bad_prop"] = result["bad"] / result["bad"].sum()
                result["total_prop"] = result["total"] / result["total"].sum()
                result["bad_rate"] = result["bad"] / result["total"]

                result["woe"] = np.log(result["good_prop"] / result["bad_prop"] + 1e-7)
                result["iv"] = (result["good_prop"] - result["bad_prop"]) * result["woe"]
                result = result.reset_index()
                del result["bins"]
                return result

            def mapWoeIv(x, f, boundary):
                x[x > boundary[-1]] = boundary[-1] - 1e-3
                x[x < boundary[0]] = boundary[0] + 1e-3

                table = pd.DataFrame({"x": x}, columns=["x"])
                table["bins"] = pd.cut(x=x, bins=boundary, right=False)
                group = table.groupby("bins")["x"]
                indexMap = []
                for j, k in enumerate(group):
                    if k[-1].shape[0]:
                        indexMap.append(pd.Series([j] * k[-1].shape[0], index=k[-1].index))
                indexMap = pd.concat(indexMap, axis=0).sort_index()
                woeIvs = []
                for name in woeIvTable.columns:
                    woeIvs.append(indexMap.apply(lambda x: woeIvTable[name].iloc[x]))
                woeIvs = pd.concat(woeIvs, axis=1)
                woeIvs.columns = map(lambda x: f + "_" + x, list(woeIvTable.columns))
                return woeIvs

            print(signal[0] % "creditCardRating")
            catFeatures = ["企业类型", "登记机关", "企业状态", "行业代码", "行业门类", "企业类别", "管辖机关", "纳税人状态代码", "登记注册类型代码", "诉讼总数量"]
            timeFeatures = ["经营期限自", "经营期限至", "成立日期", "核准日期", "注销时间"]
            specialFeatures = ["邮政编码", "省", "邮区", "邮局", "投递局", "经营范围"]
            features = catFeatures + timeFeatures + specialFeatures
            _data = data[features].fillna(-999.)
            train, test = _data.iloc[:splitLine, :].copy(), _data.iloc[splitLine:, :].copy()
            train.index, test.index = range(train.shape[0]), range(test.shape[0])
            for f in features:
                print("\r" + signal[4] % f, end="")
                kfold = StratifiedKFold(n_splits=5)
                _train, _test = [], None
                for i, (indexTrain, indexTest) in enumerate(kfold.split(train.values, self.yTrain)):
                    x1, y1 = train.iloc[indexTrain], self.yTrain[indexTrain]
                    x2, y2 = train.iloc[indexTest], self.yTrain[indexTest]
                    boundary = optimalBinningBoundary(x1[f].values, y1)
                    woeIvTable = caculateWoeIv(x1[f].values, y1, boundary)

                    woeIv = mapWoeIv(x2[f].values, f, boundary)
                    woeIv.index = indexTest
                    _train.append(woeIv)

                    woeIv = mapWoeIv(test[f].values, f, boundary)
                    if _test is None:
                        _test = woeIv
                    else:
                        _test += woeIv
                _train, _test = pd.concat(_train, axis=0).sort_index(), _test / 5.
                data = pd.concat([data, pd.concat([_train, _test], axis=0, ignore_index=True)], axis=1)

            print("\r" + signal[4] % "features")
            gc.collect()
            print(signal[-1] % "creditCardRating")
            return data

        signal = [
            "\033[33m  (%s)Begin:\033[0m",
            "\033[33m     Clear %s features(del %d),\033[0m",
            "\033[33m     Encode %s features,\033[0m",
            "\033[33m     Deal %s features,\033[0m",
            "\033[33m     Caculate woe/iv of %s,\033[0m",
            "\033[33m  (%s)End.\033[0m"
        ]
        if self.transformed:
            splitLine = self.xTrain.shape[0]
            X = pd.concat([self.xTrain, self.xPred], axis=0, ignore_index=True)
            X = dataEncoding(X)
            X = featureCreating(X)
            X = creditCardRating(splitLine, X)
            self.xTrain, self.xPred = X.iloc[:splitLine, :], X.iloc[splitLine:, :]
            self.xTrain.to_csv("output/script/train.csv", index=False)
            self.xPred.to_csv("output/script/test.csv", index=False)
        self.features = list(self.xTrain.columns)
        self.xTrain = self.xTrain.values
        self.xPred = self.xPred.values
        print("\033[33m  The number of all feature is %d.\033[0m" % len(self.features))

    def modelBuilding(self):
        def selector():
            return LGBMClassifier(
                n_estimators=1000,
                learning_rate=0.1,
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

        def lgbc(n_estimators=1000, learning_rate=0.1):
            return LGBMClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                objective="binary",
                metric="auc",
                num_leaves=2 ** 5 - 1,
                max_depth=-1,
                min_child_weight=2,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.3,
                reg_lambda=0.7,
                n_jobs=-1,
                random_state=2020,
                is_unbalance=True,
            )

        self.selectors = []
        self.models = []
        self.ensembles = []
        for i in range(self.K):
            print("\033[34m  %x) Build selector/model/stack\033[0m" % (i + 1))
            self.selectors.append(selector())
            self.models.append(lgbc(1000, 0.1))
            for i in range(self.ensembleNum):
                self.ensembles.append(lgbc(1000, 0.1))

    def featureChoosing(self):
        importances = []
        if os.path.exists("output/script/importances.npy"):
            importances = np.load("output/script/importances.npy")
        else:
            true, pred = [], []
            kfold = StratifiedKFold(n_splits=self.K, shuffle=True, random_state=2020)
            for i, (indexTrain, indexTest) in enumerate(kfold.split(self.xTrain, self.yTrain)):
                x1, y1 = self.xTrain[indexTrain], self.yTrain[indexTrain]
                x2, y2 = self.xTrain[indexTest], self.yTrain[indexTest]
                self.selectors[i].fit(
                    x1, y1,
                    eval_set=[(x1, y1), (x2, y2)],
                    verbose=False, early_stopping_rounds=50,
                )
                true += list(y2)
                p = self.selectors[i].predict_proba(x2)[:, 1]
                pred += list(p)

                fi = np.asarray(self.selectors[i].feature_importances_)
                a, b = max(fi), min(fi)
                fi = (fi - b) / (a - b)
                importances.append(fi)
                sys.stdout.write("\r\033[35m  step %d/%d score: %.06f\033[0m" % (i + 1, self.K, roc_auc_score(y2, p)))
                gc.collect()
            score = roc_auc_score(true, pred)
            print("\r\033[35m     %.06f\033[0m" % score)
            importances = np.asarray(importances)
            np.save("output/script/importances", importances)
        importances = importances.mean(axis=0)
        importances = np.where(importances > 0.03)[0]
        self.xTrain = self.xTrain[:, importances]
        self.xPred = self.xPred[:, importances]

    def modelFitting(self):
        true, pred = [], []
        kfold = StratifiedKFold(n_splits=self.K, shuffle=True, random_state=2020)
        for i, (indexTrain, indexTest) in enumerate(kfold.split(self.xTrain, self.yTrain)):
            x1, y1 = self.xTrain[indexTrain], self.yTrain[indexTrain]
            x2, y2 = self.xTrain[indexTest], self.yTrain[indexTest]
            self.models[i].fit(
                x1, y1,
                eval_set=[(x1, y1), (x2, y2)],
                verbose=False, early_stopping_rounds=50,
            )
            true += list(y2)
            p = self.models[i].predict_proba(x2)[:, 1]
            pred += list(p)
            sys.stdout.write("\r\033[31m  step %d/%d score: %.06f\033[0m" % (i + 1, self.K, roc_auc_score(y2, p)))
        score = roc_auc_score(true, pred)
        print("\r\033[31m     %.06f\033[0m" % score)
        gc.collect()

    def modelEnsembling(self):
        self.scores = []
        for i in range(self.ensembleNum):
            true, pred = [], []
            kfold = StratifiedKFold(n_splits=self.K, shuffle=True, random_state=i)
            for j, (indexTrain, indexTest) in enumerate(kfold.split(self.xTrain, self.yTrain)):
                x1, y1 = self.xTrain[indexTrain], self.yTrain[indexTrain]
                x2, y2 = self.xTrain[indexTest], self.yTrain[indexTest]
                index = i * self.K + j
                self.ensembles[index].fit(
                    x1, y1,
                    eval_set=[(x1, y1), (x2, y2)],
                    verbose=False, early_stopping_rounds=50,
                )
                true += list(y2)
                p = self.ensembles[index].predict_proba(x2)[:, 1]
                pred += list(p)
                self.scores.append(roc_auc_score(y2, p))
                sys.stdout.write("\r\033[36m  epoch %d/%d: step %d/%d: score: %.06f\033[0m" % (
                    i + 1, self.ensembleNum, j + 1, self.K, roc_auc_score(y2, p)))
                gc.collect()
            score = roc_auc_score(true, pred)
            print("\r\033[36m  epoch %d/%d: %.06f\033[0m" % (i + 1, self.ensembleNum, score))

    def modelPredicting(self):
        yPred = []
        for i, model in enumerate(self.ensembles):
            pred = model.predict_proba(self.xPred)[:, 1]
            yPred.append(pred * self.scores[i])
        yPred = np.asarray(yPred).mean(axis=0)

        self.result = pd.read_csv(self.path + "submission.csv")
        self.result["Label"] = yPred
        self.result.to_csv("output/script/result.csv", index=False)

    def ruleProcessing(self):
        thresholds = {
            "投资总额": 10000,
            "增值税": 1000,
            "企业所得税": 2000,
            "教育费": 60,
            "城建税": 100,
            "诉讼总数量": 100,
            "最新参保人数": 500,
            "年度参保总额": 1500,
        }
        table = pd.read_csv("input/rematch/test_stage2_update_20200320.csv", low_memory=False).fillna(0)
        self.result = pd.read_csv("output/script/result.csv")
        indices = []
        for name, thre in thresholds.items():
            indices.extend(list(np.where(table[name].values > thre)[0]))
        indices = list(set(indices))
        newLabel = self.result["Label"].values
        newLabel[indices] = 0.0
        newLabel = np.around(newLabel, 6)
        self.result["Label"] = newLabel
        self.result.to_csv("output/script/result2.csv", index=False)


if __name__ == "__main__":
    model = myModel()
