from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# setting up default plotting parameters
from imblearn.ensemble import EasyEnsembleClassifier
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE
from imblearn.under_sampling import (
    NearMiss,
    CondensedNearestNeighbour,
    EditedNearestNeighbours,
    RepeatedEditedNearestNeighbours,
    TomekLinks,
)
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    recall_score,
    precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.utils import resample

plt.rcParams["figure.figsize"] = [20.0, 7.0]
plt.rcParams.update({"font.size": 22})

sns.set_palette("viridis")
sns.set_style("white")
sns.set_context("talk", font_scale=0.8)

# read in data
df = pd.read_csv("creditcard.csv")

print(df.shape)
# print(df.head())

print(df.Class.value_counts())

# using seaborns countplot to show distribution of questions in dataset
fig, ax = plt.subplots()
g = sns.countplot(df.Class, palette="viridis")
g.set_xticklabels(["Not Fraud", "Fraud"])
g.set_yticklabels([])


# function to print data about the method
def print_evaluation(name, value_set, x_train, ground_truth, predicted_label):
    print("\n")
    print(name.upper())

    print("Counts")
    print(value_set.value_counts())

    # summarize the new class distribution
    counter = Counter(value_set)

    # scatter plot of examples by class label
    for label, _ in counter.items():
        if label == 0:
            color = "#e67e22"
        else:
            color = "#1abc9c"

        row_ix = np.where(value_set == label)[0]
        plt.scatter(
            x_train.iloc[row_ix, 0],
            x_train.iloc[row_ix, 1],
            label=str(label),
            color=color,
        )
    plt.title(name)
    plt.savefig(name.replace(" ", "") + ".png")
    plt.clf()
    # plt.show()

    print("\nConfusion matrix")
    print(pd.DataFrame(confusion_matrix(ground_truth, predicted_label)))

    print("\nAccuracy: ", round(accuracy_score(ground_truth, predicted_label), 2))
    print("Recall: ", round(recall_score(ground_truth, predicted_label), 2))
    print("Precision: ", round(precision_score(ground_truth, predicted_label), 2))
    print("f1 score: ", round(f1_score(ground_truth, predicted_label), 2))


# function to show values on bars
def show_values_on_bars(axs):
    def _show_on_single_plot(ax):
        for p in ax.patches:
            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height()
            value = "{:.0f}".format(p.get_height())
            ax.text(_x, _y, value, ha="center")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)


show_values_on_bars(ax)

sns.despine(left=True, bottom=True)
plt.xlabel("")
plt.ylabel("")
plt.title("Distribution of Transactions", fontsize=30)
plt.tick_params(axis="x", which="major", labelsize=15)
plt.show()

# print percentage of questions where target == 1
print("\n")
print(
    f"Ratio of the size of the minority class to the majority class: "
    f"{round((len(df.loc[df.Class == 1])) / (len(df.loc[df.Class == 0])) * 100, 2)}%"
)

# Prepare data for modeling
# Separate input features and target
y = df.Class
X = df.drop("Class", axis=1)

# setting up testing and training sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=27
)

dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
dummy_pred = dummy.predict(X_test)

# checking unique labels
print("Unique predicted labels: ", (np.unique(dummy_pred)))

# checking accuracy
print("Accuracy: ", round(accuracy_score(y_test, dummy_pred), 2))

# Modeling the data as is
# Train model
lr = LogisticRegression(solver="liblinear").fit(X_train, y_train)

# Predict on training set
lr_pred = lr.predict(X_test)

# Checking unique values
predictions = pd.DataFrame(lr_pred)

print_evaluation("Logistic regression", predictions[0], X_train, y_test, lr_pred)

# train model
rfc = RandomForestClassifier(n_estimators=10).fit(X_train, y_train)

# predict on test set
rfc_pred = rfc.predict(X_test)

print_evaluation("Rain Forest Classifier", predictions[0], X_train, y_test, rfc_pred)

# OVERSAMPLING TECHNIQUES
# Separate input features and target
y = df.Class
X = df.drop("Class", axis=1)

# setting up testing and training sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=27
)

# concatenate our training data back together
X = pd.concat([X_train, y_train], axis=1)
# print(X.head())

# separate minority and majority classes
not_fraud = X[X.Class == 0]
fraud = X[X.Class == 1]

# upsample minority
fraud_upsampled = resample(
    fraud,
    replace=True,  # sample with replacement
    n_samples=len(not_fraud),  # match number in majority class
    random_state=27,
)  # reproducible results

# combine majority and upsampled minority
upsampled = pd.concat([not_fraud, fraud_upsampled])
dataset = upsampled

# trying logistic regression again with the balanced dataset
y_train = upsampled.Class
X_train = upsampled.drop("Class", axis=1)

upsampled = LogisticRegression(solver="liblinear").fit(X_train, y_train)

upsampled_pred = upsampled.predict(X_test)

print_evaluation(
    "Random oversample",
    dataset.Class,
    X_train,
    y_test,
    upsampled_pred,
)

# UNDERSAMPLING
# still using our separated classes fraud and not_fraud from above

# downsample majority
not_fraud_downsampled = resample(
    not_fraud,
    replace=False,  # sample without replacement
    n_samples=len(fraud),  # match minority n
    random_state=27,
)  # reproducible results

# combine minority and downsampled majority
downsampled = pd.concat([not_fraud_downsampled, fraud])
dataset = downsampled
# checking counts
# print(downsampled.Class.value_counts())

# trying logistic regression again with the undersampled dataset

y_train = downsampled.Class
X_train = downsampled.drop("Class", axis=1)

undersampled = LogisticRegression(solver="liblinear").fit(X_train, y_train)

undersampled_pred = undersampled.predict(X_test)

print_evaluation(
    "Random undersample",
    dataset.Class,
    X_train,
    y_test,
    undersampled_pred,
)

# SMOTE - oversampling
# Separate input features and target
y = df.Class
X = df.drop("Class", axis=1)

# setting up testing and training sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=27
)

sm = SMOTE(random_state=27)
X_train, y_train = sm.fit_sample(X_train, y_train)

smote = LogisticRegression(solver="liblinear").fit(X_train, y_train)

smote_pred = smote.predict(X_test)

print_evaluation("SMOTE", y_train, X_train, y_test, smote_pred)

# BorderlineSMOTE1
# Separate input features and target
y = df.Class
X = df.drop("Class", axis=1)

# setting up testing and training sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=27
)

b1sm = BorderlineSMOTE(kind="borderline-1", random_state=27)
X_train, y_train = b1sm.fit_sample(X_train, y_train)

b1smote = LogisticRegression(solver="liblinear").fit(X_train, y_train)

b1smote_pred = b1smote.predict(X_test)

print_evaluation("Borderline1-SMOTE", y_train, X_train, y_test, b1smote_pred)

# BorderlineSMOTE2
# Separate input features and target
y = df.Class
X = df.drop("Class", axis=1)

# setting up testing and training sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=27
)

b2sm = BorderlineSMOTE(kind="borderline-2", random_state=27)
X_train, y_train = b2sm.fit_sample(X_train, y_train)

b2smote = LogisticRegression(solver="liblinear").fit(X_train, y_train)

b2smote_pred = b2smote.predict(X_test)

print_evaluation("Borderline2-SMOTE", y_train, X_train, y_test, b2smote_pred)

# SVM
# Separate input features and target
y = df.Class
X = df.drop("Class", axis=1)

# setting up testing and training sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=27
)

svm = SVC(kernel="linear", random_state=27)
svm.fit(X_train, y_train)

svm_pred = svm.predict(X_test)

print_evaluation("SVM", y_train, X_train, y_test, svm_pred)

# BorderlineSMOTE-SVM
# Separate input features and target
y = df.Class
X = df.drop("Class", axis=1)

# setting up testing and training sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=27
)

svmsm = SVMSMOTE(random_state=27)
X_train, y_train = svmsm.fit_sample(X_train, y_train)

svmsmote = LogisticRegression(solver="liblinear").fit(X_train, y_train)

svmsmote_pred = svmsmote.predict(X_test)

print_evaluation("SVM-SMOTE", y_train, X_train, y_test, svmsmote_pred)

# NearMiss-1
y = df.Class
X = df.drop("Class", axis=1)

# setting up testing and training sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=27
)

nr1 = NearMiss(version=1)
X_train, y_train = nr1.fit_sample(X_train, y_train)

nr1 = LogisticRegression(solver="liblinear").fit(X_train, y_train)

nr1_pred = nr1.predict(X_test)

print_evaluation(
    "NearMiss-1", y_train, X_train, y_test, nr1_pred
)

# NearMiss-2
y = df.Class
X = df.drop("Class", axis=1)

# setting up testing and training sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=27
)

nr2 = NearMiss(version=2)
X_train, y_train = nr2.fit_sample(X_train, y_train)

nr2 = LogisticRegression(solver="liblinear").fit(X_train, y_train)

nr2_pred = nr2.predict(X_test)

print_evaluation(
    "NearMiss-2", y_train, X_train, y_test, nr2_pred
)

# NearMiss-3
y = df.Class
X = df.drop("Class", axis=1)

# setting up testing and training sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=27
)

nr3 = NearMiss(version=3)
X_train, y_train = nr3.fit_sample(X_train, y_train)

nr3 = LogisticRegression(solver="liblinear").fit(X_train, y_train)

nr3_pred = nr3.predict(X_test)

print_evaluation(
    "NearMiss-3", y_train, X_train, y_test, nr3_pred
)

# EditedNearestNeighbour
y = df.Class
X = df.drop("Class", axis=1)

# setting up testing and training sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=27
)

enn = EditedNearestNeighbours(n_neighbors=3)
X_train, y_train = enn.fit_sample(X_train, y_train)

enn = LogisticRegression(solver="liblinear").fit(X_train, y_train)

enn_pred = enn.predict(X_test)

print_evaluation(
    "Edited Nearest Neighbour",
    y_train,
    X_train,
    y_test,
    enn_pred,
)

# RepeatedEditedNearestNeighbour
y = df.Class
X = df.drop("Class", axis=1)

# setting up testing and training sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=27
)

renn = RepeatedEditedNearestNeighbours(n_neighbors=3)
X_train, y_train = renn.fit_sample(X_train, y_train)

renn = LogisticRegression(solver="liblinear").fit(X_train, y_train)

renn_pred = renn.predict(X_test)

print_evaluation(
    "Repeated Edited Nearest Neighbour",
    y_train,
    X_train,
    y_test,
    renn_pred,
)

# TomekLinksRemoval
y = df.Class
X = df.drop("Class", axis=1)

# setting up testing and training sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=27
)

tl = TomekLinks()
X_train, y_train = tl.fit_sample(X_train, y_train)

tl = LogisticRegression(solver="liblinear").fit(X_train, y_train)

tl_pred = tl.predict(X_test)

print_evaluation(
    "Tomek Links Removal", y_train, X_train, y_test, tl_pred
)

# EasyEnsemble
y = df.Class
X = df.drop("Class", axis=1)

# setting up testing and training sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=27
)

ens = EasyEnsembleClassifier()

ens = ens.fit(X_train, y_train)

ens_pred = ens.predict(X_test)

print_evaluation(
    "Easy Ensamble", y_train, X_train, y_test, ens_pred,
)

# CondensedNearestNeighbor
y = df.Class
X = df.drop("Class", axis=1)

# setting up testing and training sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=27
)

cnn = CondensedNearestNeighbour(n_neighbors=3)
X_train, y_train = cnn.fit_sample(X_train, y_train)

cnn = LogisticRegression(solver="liblinear").fit(X_train, y_train)

cnn_pred = cnn.predict(X_test)

print_evaluation(
    "Condensed Nearest Neighbour",
    y_train,
    X_train,
    y_test,
    cnn_pred,
)
