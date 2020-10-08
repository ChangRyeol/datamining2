# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + id="qYkza4YQtiBx" executionInfo={"status": "ok", "timestamp": 1602135428641, "user_tz": -540, "elapsed": 19861, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="f8879ed5-a046-44d9-ee41-c8f948f1bf88" colab={"base_uri": "https://localhost:8080/", "height": 55}
from google.colab import drive # import drive from google colab

ROOT = "/content/drive"     # default location for the drive
print(ROOT)                 # print content of ROOT (Optional)

drive.mount(ROOT)           # we mount the google drive at /content/drive

# + id="HRroFerPtpqr" executionInfo={"status": "ok", "timestamp": 1602135498285, "user_tz": -540, "elapsed": 683, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="8b139819-a8ab-4f65-bb5d-9757055cd0fd" colab={"base_uri": "https://localhost:8080/", "height": 36}
# %cd 'drive/My Drive/Colab Notebooks/datamining2/src/class/' 

# + id="1qaAMg1mt7Ul" executionInfo={"status": "ok", "timestamp": 1602135512753, "user_tz": -540, "elapsed": 6925, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="fa9b5d99-253c-4fc4-8e35-259e44ae1a54" colab={"base_uri": "https://localhost:8080/", "height": 467}
pip install jupytext #jupytext 설치 

# + id="HQLIDUVNt-Tf" executionInfo={"status": "ok", "timestamp": 1602135533154, "user_tz": -540, "elapsed": 1684, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="c84bb905-98cb-4dd6-af3d-de97622253f3" colab={"base_uri": "https://localhost:8080/", "height": 223}
## Pair a notebook to a light script
# !jupytext --set-formats ipynb,py:light 07_ensemble_learning_and_random_forests.ipynb  

# + id="XVfDe8XguDnt"
# Sync the two representations
# !jupytext --sync 06_decision_trees.ipynb

# + [markdown] id="LZGPMjGptMF2"
# **Chapter 7 – Ensemble Learning and Random Forests**

# + [markdown] id="KG7eScXftMF4"
# _This notebook contains all the sample code and solutions to the exercises in chapter 7._

# + [markdown] id="eg7ZfdJotMF5"
# <table align="left">
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/ageron/handson-ml2/blob/master/07_ensemble_learning_and_random_forests.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
#   </td>
# </table>

# + [markdown] id="Ofbk01dmtMF5"
# # Setup

# + [markdown] id="ey07aPjntMF7"
# First, let's import a few common modules, ensure MatplotLib plots figures inline and prepare a function to save the figures. We also check that Python 3.5 or later is installed (although Python 2.x may work, it is deprecated so we strongly recommend you use Python 3 instead), as well as Scikit-Learn ≥0.20.

# + id="5DNlYnnAtMF8"
# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# Common imports
import numpy as np
import os

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
# %matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "ensembles"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


# + [markdown] id="3hetr3M4tMF_"
# # Voting classifiers

# + id="KpvKGkeltMGA"
heads_proba = 0.51
coin_tosses = (np.random.rand(10000, 10) < heads_proba).astype(np.int32)
cumulative_heads_ratio = np.cumsum(coin_tosses, axis=0) / np.arange(1, 10001).reshape(-1, 1)

# + id="vYmEpr-atMGE" outputId="caef2ca1-1d79-4adf-ec6f-5095b2ea5343"
plt.figure(figsize=(8,3.5))
plt.plot(cumulative_heads_ratio)
plt.plot([0, 10000], [0.51, 0.51], "k--", linewidth=2, label="51%")
plt.plot([0, 10000], [0.5, 0.5], "k-", label="50%")
plt.xlabel("Number of coin tosses")
plt.ylabel("Heads ratio")
plt.legend(loc="lower right")
plt.axis([0, 10000, 0.42, 0.58])
save_fig("law_of_large_numbers_plot")
plt.show()

# + id="eWyYiuEXtMGH"
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# + [markdown] id="SfFD3JNHtMGL"
# **Note**: to be future-proof, we set `solver="lbfgs"`, `n_estimators=100`, and `gamma="scale"` since these will be the default values in upcoming Scikit-Learn versions.

# + id="fkrOLpu2tMGL"
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

log_clf = LogisticRegression(solver="lbfgs", random_state=42)
rnd_clf = RandomForestClassifier(n_estimators=100, random_state=42)
svm_clf = SVC(gamma="scale", random_state=42)

voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
    voting='hard')

# + id="8P1ty_nItMGO" outputId="0f8b5ff4-e401-45ff-c58c-0399da32be25"
voting_clf.fit(X_train, y_train)

# + id="d0ih4I9stMGS" outputId="606585da-1d1f-4444-901d-a95ebd75621a"
from sklearn.metrics import accuracy_score

for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

# + [markdown] id="9WwdSXmVtMGX"
# Soft voting:

# + id="aRg9B8Q1tMGX" outputId="2d3f4398-a15f-4986-bb22-52a2901908c7"
log_clf = LogisticRegression(solver="lbfgs", random_state=42)
rnd_clf = RandomForestClassifier(n_estimators=100, random_state=42)
svm_clf = SVC(gamma="scale", probability=True, random_state=42)

voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
    voting='soft')
voting_clf.fit(X_train, y_train)

# + id="_utrruObtMGa" outputId="6e6555e7-b90d-499f-e74f-61f51cd65760"
from sklearn.metrics import accuracy_score

for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

# + [markdown] id="4i3TR-z8tMGd"
# # Bagging ensembles

# + id="oTxGy2pUtMGe"
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag_clf = BaggingClassifier(
    DecisionTreeClassifier(random_state=42), n_estimators=500,
    max_samples=100, bootstrap=True, random_state=42)
bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)

# + id="eRJ142dftMGg" outputId="3d4cf50b-8dbe-4b07-e638-e81c99cd6e29"
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

# + id="jvQ3nx1-tMGm" outputId="455e5ffb-6b89-4152-c672-a55187ac9bf9"
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)
y_pred_tree = tree_clf.predict(X_test)
print(accuracy_score(y_test, y_pred_tree))

# + id="frFD7-KltMGp"
from matplotlib.colors import ListedColormap

def plot_decision_boundary(clf, X, y, axes=[-1.5, 2.45, -1, 1.5], alpha=0.5, contour=True):
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
    if contour:
        custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", alpha=alpha)
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", alpha=alpha)
    plt.axis(axes)
    plt.xlabel(r"$x_1$", fontsize=18)
    plt.ylabel(r"$x_2$", fontsize=18, rotation=0)


# + id="N7xIFQE7tMGr" outputId="4e184453-4ec4-473e-c304-cceffe8b8f1c"
fix, axes = plt.subplots(ncols=2, figsize=(10,4), sharey=True)
plt.sca(axes[0])
plot_decision_boundary(tree_clf, X, y)
plt.title("Decision Tree", fontsize=14)
plt.sca(axes[1])
plot_decision_boundary(bag_clf, X, y)
plt.title("Decision Trees with Bagging", fontsize=14)
plt.ylabel("")
save_fig("decision_tree_without_and_with_bagging_plot")
plt.show()

# + [markdown] id="v5DjOn-ktMGw"
# # Random Forests

# + id="KnwOLr55tMGw"
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(splitter="random", max_leaf_nodes=16, random_state=42),
    n_estimators=500, max_samples=1.0, bootstrap=True, random_state=42)

# + id="VBwF4UqStMGz"
bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)

# + id="nCMyTpzotMG3"
from sklearn.ensemble import RandomForestClassifier

rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, random_state=42)
rnd_clf.fit(X_train, y_train)

y_pred_rf = rnd_clf.predict(X_test)

# + id="2_B0RRLxtMG6" outputId="ca176efc-02a9-47c4-ad37-4f15443a85f2"
np.sum(y_pred == y_pred_rf) / len(y_pred)  # almost identical predictions

# + id="PyPn8rF2tMG9" outputId="d5bd4c3d-4c4f-4af1-f48c-f1496d707b3c"
from sklearn.datasets import load_iris
iris = load_iris()
rnd_clf = RandomForestClassifier(n_estimators=500, random_state=42)
rnd_clf.fit(iris["data"], iris["target"])
for name, score in zip(iris["feature_names"], rnd_clf.feature_importances_):
    print(name, score)

# + id="eQKg-gTOtMHB" outputId="e956824c-c2a5-4719-f00c-5d444d2f4291"
rnd_clf.feature_importances_

# + id="Q1uYXUAZtMHF" outputId="d7c38992-5295-4d1e-c980-b0f6e2b71d90"
plt.figure(figsize=(6, 4))

for i in range(15):
    tree_clf = DecisionTreeClassifier(max_leaf_nodes=16, random_state=42 + i)
    indices_with_replacement = np.random.randint(0, len(X_train), len(X_train))
    tree_clf.fit(X[indices_with_replacement], y[indices_with_replacement])
    plot_decision_boundary(tree_clf, X, y, axes=[-1.5, 2.45, -1, 1.5], alpha=0.02, contour=False)

plt.show()

# + [markdown] id="Z9LwBGfntMHH"
# ## Out-of-Bag evaluation

# + id="oxBIYormtMHI" outputId="0c8fa4e5-2a5a-4b34-e76c-6db43368f15c"
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(random_state=42), n_estimators=500,
    bootstrap=True, oob_score=True, random_state=40)
bag_clf.fit(X_train, y_train)
bag_clf.oob_score_

# + id="J0A7rMIqtMHL" outputId="2acb5d4d-4301-48de-d8f2-a88ca9168daf"
bag_clf.oob_decision_function_

# + id="gkDiFWgLtMHP" outputId="2dd98a89-1b60-4031-8b87-4339562213b5"
from sklearn.metrics import accuracy_score
y_pred = bag_clf.predict(X_test)
accuracy_score(y_test, y_pred)

# + [markdown] id="9Uck3v5otMHS"
# ## Feature importance

# + id="YUe6ys1htMHS"
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1)
mnist.target = mnist.target.astype(np.uint8)

# + id="CpgaUoAEtMHW" outputId="68e23221-d689-41b2-9a5d-dab14115d93d"
rnd_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rnd_clf.fit(mnist["data"], mnist["target"])


# + id="JULwYuvytMHZ"
def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap = mpl.cm.hot,
               interpolation="nearest")
    plt.axis("off")


# + id="5lWvgakWtMHc" outputId="23bbfd03-b0cc-4656-8260-58b9b54eb153"
plot_digit(rnd_clf.feature_importances_)

cbar = plt.colorbar(ticks=[rnd_clf.feature_importances_.min(), rnd_clf.feature_importances_.max()])
cbar.ax.set_yticklabels(['Not important', 'Very important'])

save_fig("mnist_feature_importance_plot")
plt.show()

# + [markdown] id="evZfxEbOtMHf"
# # AdaBoost

# + id="R8k47JJAtMHf" outputId="c9ce9358-cfe3-4340-adc7-7d32230b62c6"
from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), n_estimators=200,
    algorithm="SAMME.R", learning_rate=0.5, random_state=42)
ada_clf.fit(X_train, y_train)

# + id="0V8k5e0utMHh" outputId="f8f4497d-8f57-48e6-ed6a-b55a36fb0209"
plot_decision_boundary(ada_clf, X, y)

# + id="0eHBhATrtMHk" outputId="5880b341-5669-4f8b-fef4-1ff6fef0dc62"
m = len(X_train)

fix, axes = plt.subplots(ncols=2, figsize=(10,4), sharey=True)
for subplot, learning_rate in ((0, 1), (1, 0.5)):
    sample_weights = np.ones(m)
    plt.sca(axes[subplot])
    for i in range(5):
        svm_clf = SVC(kernel="rbf", C=0.05, gamma="scale", random_state=42)
        svm_clf.fit(X_train, y_train, sample_weight=sample_weights)
        y_pred = svm_clf.predict(X_train)
        sample_weights[y_pred != y_train] *= (1 + learning_rate)
        plot_decision_boundary(svm_clf, X, y, alpha=0.2)
        plt.title("learning_rate = {}".format(learning_rate), fontsize=16)
    if subplot == 0:
        plt.text(-0.7, -0.65, "1", fontsize=14)
        plt.text(-0.6, -0.10, "2", fontsize=14)
        plt.text(-0.5,  0.10, "3", fontsize=14)
        plt.text(-0.4,  0.55, "4", fontsize=14)
        plt.text(-0.3,  0.90, "5", fontsize=14)
    else:
        plt.ylabel("")

save_fig("boosting_plot")
plt.show()

# + id="iXgejDwBtMHo" outputId="f75503ca-235b-4d63-d569-d4eccf3082ab"
list(m for m in dir(ada_clf) if not m.startswith("_") and m.endswith("_"))

# + [markdown] id="_uk4qIDotMHq"
# # Gradient Boosting

# + id="F-SdOkVCtMHr"
np.random.seed(42)
X = np.random.rand(100, 1) - 0.5
y = 3*X[:, 0]**2 + 0.05 * np.random.randn(100)

# + id="-eHgyT1-tMHt" outputId="3798aa18-44bf-4861-9d49-d51867246c6a"
from sklearn.tree import DecisionTreeRegressor

tree_reg1 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg1.fit(X, y)

# + id="BxMrFFNgtMHv" outputId="4e3fefef-bd08-480f-d92c-c94ad3a04def"
y2 = y - tree_reg1.predict(X)
tree_reg2 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg2.fit(X, y2)

# + id="NulUjBZCtMHy" outputId="09e1610b-2e44-4fbc-a072-8a82ec27a949"
y3 = y2 - tree_reg2.predict(X)
tree_reg3 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg3.fit(X, y3)

# + id="VOchun8ytMH1"
X_new = np.array([[0.8]])

# + id="xaTCgzTNtMH5"
y_pred = sum(tree.predict(X_new) for tree in (tree_reg1, tree_reg2, tree_reg3))

# + id="I79EKb1itMIA" outputId="c65a56ba-e400-4166-8a1e-d8f939a78988"
y_pred


# + id="ydTxR10ItMIL"
def plot_predictions(regressors, X, y, axes, label=None, style="r-", data_style="b.", data_label=None):
    x1 = np.linspace(axes[0], axes[1], 500)
    y_pred = sum(regressor.predict(x1.reshape(-1, 1)) for regressor in regressors)
    plt.plot(X[:, 0], y, data_style, label=data_label)
    plt.plot(x1, y_pred, style, linewidth=2, label=label)
    if label or data_label:
        plt.legend(loc="upper center", fontsize=16)
    plt.axis(axes)


# + id="3X0bNR3ttMIP" outputId="824392b4-c601-4428-dd53-58c572103382"
plt.figure(figsize=(11,11))

plt.subplot(321)
plot_predictions([tree_reg1], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="$h_1(x_1)$", style="g-", data_label="Training set")
plt.ylabel("$y$", fontsize=16, rotation=0)
plt.title("Residuals and tree predictions", fontsize=16)

plt.subplot(322)
plot_predictions([tree_reg1], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="$h(x_1) = h_1(x_1)$", data_label="Training set")
plt.ylabel("$y$", fontsize=16, rotation=0)
plt.title("Ensemble predictions", fontsize=16)

plt.subplot(323)
plot_predictions([tree_reg2], X, y2, axes=[-0.5, 0.5, -0.5, 0.5], label="$h_2(x_1)$", style="g-", data_style="k+", data_label="Residuals")
plt.ylabel("$y - h_1(x_1)$", fontsize=16)

plt.subplot(324)
plot_predictions([tree_reg1, tree_reg2], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="$h(x_1) = h_1(x_1) + h_2(x_1)$")
plt.ylabel("$y$", fontsize=16, rotation=0)

plt.subplot(325)
plot_predictions([tree_reg3], X, y3, axes=[-0.5, 0.5, -0.5, 0.5], label="$h_3(x_1)$", style="g-", data_style="k+")
plt.ylabel("$y - h_1(x_1) - h_2(x_1)$", fontsize=16)
plt.xlabel("$x_1$", fontsize=16)

plt.subplot(326)
plot_predictions([tree_reg1, tree_reg2, tree_reg3], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="$h(x_1) = h_1(x_1) + h_2(x_1) + h_3(x_1)$")
plt.xlabel("$x_1$", fontsize=16)
plt.ylabel("$y$", fontsize=16, rotation=0)

save_fig("gradient_boosting_plot")
plt.show()

# + id="rNtoEos8tMIV" outputId="bf2de466-b202-4a1d-c153-a8169b69a926"
from sklearn.ensemble import GradientBoostingRegressor

gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=1.0, random_state=42)
gbrt.fit(X, y)

# + id="qcxNxKxitMIa" outputId="1400ed5e-8244-46ca-8bef-58fd62667157"
gbrt_slow = GradientBoostingRegressor(max_depth=2, n_estimators=200, learning_rate=0.1, random_state=42)
gbrt_slow.fit(X, y)

# + id="FvfNp2RStMIf" outputId="d454efb3-872c-46a4-f8ba-884a5f994e7d"
fix, axes = plt.subplots(ncols=2, figsize=(10,4), sharey=True)

plt.sca(axes[0])
plot_predictions([gbrt], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="Ensemble predictions")
plt.title("learning_rate={}, n_estimators={}".format(gbrt.learning_rate, gbrt.n_estimators), fontsize=14)
plt.xlabel("$x_1$", fontsize=16)
plt.ylabel("$y$", fontsize=16, rotation=0)

plt.sca(axes[1])
plot_predictions([gbrt_slow], X, y, axes=[-0.5, 0.5, -0.1, 0.8])
plt.title("learning_rate={}, n_estimators={}".format(gbrt_slow.learning_rate, gbrt_slow.n_estimators), fontsize=14)
plt.xlabel("$x_1$", fontsize=16)

save_fig("gbrt_learning_rate_plot")
plt.show()

# + [markdown] id="jGwqcaFCtMIl"
# ## Gradient Boosting with Early stopping

# + id="3A6RSZhYtMIm" outputId="76ffb891-2911-4c34-8633-ea8a1f8cdec1"
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=49)

gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=120, random_state=42)
gbrt.fit(X_train, y_train)

errors = [mean_squared_error(y_val, y_pred)
          for y_pred in gbrt.staged_predict(X_val)]
bst_n_estimators = np.argmin(errors) + 1

gbrt_best = GradientBoostingRegressor(max_depth=2, n_estimators=bst_n_estimators, random_state=42)
gbrt_best.fit(X_train, y_train)

# + id="tx-J6uwctMIq"
min_error = np.min(errors)

# + id="sYlWLS3WtMIt" outputId="87675e45-a9b2-4023-a8d6-923944815a76"
plt.figure(figsize=(10, 4))

plt.subplot(121)
plt.plot(errors, "b.-")
plt.plot([bst_n_estimators, bst_n_estimators], [0, min_error], "k--")
plt.plot([0, 120], [min_error, min_error], "k--")
plt.plot(bst_n_estimators, min_error, "ko")
plt.text(bst_n_estimators, min_error*1.2, "Minimum", ha="center", fontsize=14)
plt.axis([0, 120, 0, 0.01])
plt.xlabel("Number of trees")
plt.ylabel("Error", fontsize=16)
plt.title("Validation error", fontsize=14)

plt.subplot(122)
plot_predictions([gbrt_best], X, y, axes=[-0.5, 0.5, -0.1, 0.8])
plt.title("Best model (%d trees)" % bst_n_estimators, fontsize=14)
plt.ylabel("$y$", fontsize=16, rotation=0)
plt.xlabel("$x_1$", fontsize=16)

save_fig("early_stopping_gbrt_plot")
plt.show()

# + id="2lRt4hx6tMIy"
gbrt = GradientBoostingRegressor(max_depth=2, warm_start=True, random_state=42)

min_val_error = float("inf")
error_going_up = 0
for n_estimators in range(1, 120):
    gbrt.n_estimators = n_estimators
    gbrt.fit(X_train, y_train)
    y_pred = gbrt.predict(X_val)
    val_error = mean_squared_error(y_val, y_pred)
    if val_error < min_val_error:
        min_val_error = val_error
        error_going_up = 0
    else:
        error_going_up += 1
        if error_going_up == 5:
            break  # early stopping

# + id="CoYlLZovtMI1" outputId="a49068c6-11c5-4db3-d444-3ce94c7a31d0"
print(gbrt.n_estimators)

# + id="oRUJapc9tMI4" outputId="7ba6c8d0-2db1-48c9-c6f5-177bc8ac2eb6"
print("Minimum validation MSE:", min_val_error)

# + [markdown] id="bP92nAKztMI9"
# ## Using XGBoost

# + id="PPpBH3TrtMI-"
try:
    import xgboost
except ImportError as ex:
    print("Error: the xgboost library is not installed.")
    xgboost = None

# + id="m95zwLTUtMJA" outputId="877f1b51-5931-46ea-938e-fae13577f120"
if xgboost is not None:  # not shown in the book
    xgb_reg = xgboost.XGBRegressor(random_state=42)
    xgb_reg.fit(X_train, y_train)
    y_pred = xgb_reg.predict(X_val)
    val_error = mean_squared_error(y_val, y_pred) # Not shown
    print("Validation MSE:", val_error)           # Not shown

# + id="x4y0zrAYtMJC" outputId="378760fe-bfff-40e0-9013-c942aa1640a2"
if xgboost is not None:  # not shown in the book
    xgb_reg.fit(X_train, y_train,
                eval_set=[(X_val, y_val)], early_stopping_rounds=2)
    y_pred = xgb_reg.predict(X_val)
    val_error = mean_squared_error(y_val, y_pred)  # Not shown
    print("Validation MSE:", val_error)            # Not shown

# + id="VQ0BQJgrtMJE" outputId="fc3f1ce4-5c3e-4d2c-f552-1d1f20735e45"
# %timeit xgboost.XGBRegressor().fit(X_train, y_train) if xgboost is not None else None

# + id="KrLzdkBntMJI" outputId="603b4894-a912-469d-8414-b83155de793b"
# %timeit GradientBoostingRegressor().fit(X_train, y_train)

# + [markdown] id="gl_bp_wbtMJL"
# # Exercise solutions

# + [markdown] id="b307iAvNtMJM"
# ## 1. to 7.

# + [markdown] id="buIb_fR0tMJM"
# See Appendix A.

# + [markdown] id="sxtdH0-ttMJN"
# ## 8. Voting Classifier

# + [markdown] id="avBRpDA1tMJN"
# Exercise: _Load the MNIST data and split it into a training set, a validation set, and a test set (e.g., use 50,000 instances for training, 10,000 for validation, and 10,000 for testing)._

# + [markdown] id="b7UcMZ-QtMJO"
# The MNIST dataset was loaded earlier.

# + id="nByCvjGDtMJO"
from sklearn.model_selection import train_test_split

# + id="vBvS8iaotMJS"
X_train_val, X_test, y_train_val, y_test = train_test_split(
    mnist.data, mnist.target, test_size=10000, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=10000, random_state=42)

# + [markdown] id="D190MtnVtMJU"
# Exercise: _Then train various classifiers, such as a Random Forest classifier, an Extra-Trees classifier, and an SVM._

# + id="e7UIaelBtMJU"
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier

# + id="Uxtgcel6tMJW"
random_forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
extra_trees_clf = ExtraTreesClassifier(n_estimators=100, random_state=42)
svm_clf = LinearSVC(random_state=42)
mlp_clf = MLPClassifier(random_state=42)

# + id="jYHZnkwTtMJY" outputId="9fd98c4f-3429-47bb-bcd6-ef3e77d12780"
estimators = [random_forest_clf, extra_trees_clf, svm_clf, mlp_clf]
for estimator in estimators:
    print("Training the", estimator)
    estimator.fit(X_train, y_train)

# + id="OY_OaAiTtMJb" outputId="eb78b6e9-e424-442a-b3f3-a284a1f2e13b"
[estimator.score(X_val, y_val) for estimator in estimators]

# + [markdown] id="MET63D5CtMJd"
# The linear SVM is far outperformed by the other classifiers. However, let's keep it for now since it may improve the voting classifier's performance.

# + [markdown] id="UBweoEpLtMJe"
# Exercise: _Next, try to combine them into an ensemble that outperforms them all on the validation set, using a soft or hard voting classifier._

# + id="BfsOvMIjtMJe"
from sklearn.ensemble import VotingClassifier

# + id="xtrqS8_ctMJf"
named_estimators = [
    ("random_forest_clf", random_forest_clf),
    ("extra_trees_clf", extra_trees_clf),
    ("svm_clf", svm_clf),
    ("mlp_clf", mlp_clf),
]

# + id="uWXXGnaotMJh"
voting_clf = VotingClassifier(named_estimators)

# + id="9BVz6NnotMJi" outputId="17c51a67-183f-40f8-804d-afc4229798aa"
voting_clf.fit(X_train, y_train)

# + id="QU4h3PivtMJk" outputId="4157872f-3726-4676-c7ff-34600f39b5ba"
voting_clf.score(X_val, y_val)

# + id="vXU3rhEztMJo" outputId="7a3a37aa-6463-42da-dd3b-00cb020a9e90"
[estimator.score(X_val, y_val) for estimator in voting_clf.estimators_]

# + [markdown] id="NTkUGhTdtMJr"
# Let's remove the SVM to see if performance improves. It is possible to remove an estimator by setting it to `None` using `set_params()` like this:

# + id="AN_uRcbvtMJr" outputId="8840a7dc-25db-4e08-f79d-9e347d0bce91"
voting_clf.set_params(svm_clf=None)

# + [markdown] id="rcoK-ATrtMJu"
# This updated the list of estimators:

# + id="8-iNdxLCtMJu" outputId="27d0ceaa-0852-4142-a2c0-7063fbb6ca4b"
voting_clf.estimators

# + [markdown] id="MKBRV6x6tMJz"
# However, it did not update the list of _trained_ estimators:

# + id="LS46esQYtMJz" outputId="f0ef0b1b-c2d4-4b86-c063-2d0ac0a07a8c"
voting_clf.estimators_

# + [markdown] id="xxE_hMXJtMJ2"
# So we can either fit the `VotingClassifier` again, or just remove the SVM from the list of trained estimators:

# + id="nFye2UgRtMJ3"
del voting_clf.estimators_[2]

# + [markdown] id="UcdV1MvCtMJ5"
# Now let's evaluate the `VotingClassifier` again:

# + id="W0b9w9v4tMJ5" outputId="7c65d089-0ffe-42bc-9d50-0cc85984cadc"
voting_clf.score(X_val, y_val)

# + [markdown] id="EiMRIIiatMJ8"
# A bit better! The SVM was hurting performance. Now let's try using a soft voting classifier. We do not actually need to retrain the classifier, we can just set `voting` to `"soft"`:

# + id="LhR5dfnctMJ8"
voting_clf.voting = "soft"

# + id="PCHBQBqztMJ-" outputId="52b23bf1-e681-4d6b-9e7c-4845c2a667c7"
voting_clf.score(X_val, y_val)

# + [markdown] id="bufXhfo9tMKA"
# Nope, hard voting wins in this case.

# + [markdown] id="OkNY-Yg2tMKA"
# _Once you have found one, try it on the test set. How much better does it perform compared to the individual classifiers?_

# + id="cpelrrmQtMKA" outputId="755e9c91-12ff-45a0-af96-2d35a0b034e7"
voting_clf.voting = "hard"
voting_clf.score(X_test, y_test)

# + id="fuG2EjkJtMKC" outputId="2e64e9b5-8ffc-4980-c346-8488307eba5a"
[estimator.score(X_test, y_test) for estimator in voting_clf.estimators_]

# + [markdown] id="zWU5fsTrtMKF"
# The voting classifier only very slightly reduced the error rate of the best model in this case.

# + [markdown] id="CCt5VW-ZtMKG"
# ## 9. Stacking Ensemble

# + [markdown] id="kRYMYhvrtMKG"
# Exercise: _Run the individual classifiers from the previous exercise to make predictions on the validation set, and create a new training set with the resulting predictions: each training instance is a vector containing the set of predictions from all your classifiers for an image, and the target is the image's class. Train a classifier on this new training set._

# + id="xvR6zSmgtMKH"
X_val_predictions = np.empty((len(X_val), len(estimators)), dtype=np.float32)

for index, estimator in enumerate(estimators):
    X_val_predictions[:, index] = estimator.predict(X_val)

# + id="TNoupwGJtMKI" outputId="eccd5029-bc63-4cbe-d8a4-c4132bac29ee"
X_val_predictions

# + id="CeVW9eaJtMKK" outputId="e71ea329-1af3-4e09-926a-2a35ae413543"
rnd_forest_blender = RandomForestClassifier(n_estimators=200, oob_score=True, random_state=42)
rnd_forest_blender.fit(X_val_predictions, y_val)

# + id="UY-gc1D9tMKN" outputId="3149e523-93de-479c-d926-542a328bac3e"
rnd_forest_blender.oob_score_

# + [markdown] id="XcN79dFLtMKP"
# You could fine-tune this blender or try other types of blenders (e.g., an `MLPClassifier`), then select the best one using cross-validation, as always.

# + [markdown] id="SOAvYKKJtMKP"
# Exercise: _Congratulations, you have just trained a blender, and together with the classifiers they form a stacking ensemble! Now let's evaluate the ensemble on the test set. For each image in the test set, make predictions with all your classifiers, then feed the predictions to the blender to get the ensemble's predictions. How does it compare to the voting classifier you trained earlier?_

# + id="HGpyGAwttMKP"
X_test_predictions = np.empty((len(X_test), len(estimators)), dtype=np.float32)

for index, estimator in enumerate(estimators):
    X_test_predictions[:, index] = estimator.predict(X_test)

# + id="7paR_m8btMKR"
y_pred = rnd_forest_blender.predict(X_test_predictions)

# + id="QaPLsLv-tMKS"
from sklearn.metrics import accuracy_score

# + id="j5x92ibutMKT" outputId="9a2e39e8-86b0-4977-ebfc-d0239963a3be"
accuracy_score(y_test, y_pred)

# + [markdown] id="5wV8XakDtMKV"
# This stacking ensemble does not perform as well as the voting classifier we trained earlier, it's not quite as good as the best individual classifier.

# + id="2POc0PEqtMKV"

