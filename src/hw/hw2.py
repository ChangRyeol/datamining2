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

# + [markdown] id="ABh1_YZJyAnt"
# #데이터마이닝(2) 두번째 과제 

# + [markdown] id="Zqg6B8vyHTa8"
# - 기본 셋팅

# + id="q62qlQzJyNtv" executionInfo={"status": "ok", "timestamp": 1602587757349, "user_tz": -540, "elapsed": 957, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}}
# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# Common imports
import numpy as np
import os


# + [markdown] id="G_jiV2RCm4mg"
# # Chap 5 과제 

# + [markdown] id="yBHg1xSTfEdU"
# _Exercise: train a `LinearSVC` on a linearly separable dataset. Then train an `SVC` and a `SGDClassifier` on the same dataset. See if you can get them to produce roughly the same model._

# + [markdown] id="GJPpyXP6m4mh"
# <table align="left">
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/ageron/handson-ml2/blob/master/05_support_vector_machines.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
#   </td>
# </table>

# + id="v-0Tebc4o95a" executionInfo={"status": "ok", "timestamp": 1602587778983, "user_tz": -540, "elapsed": 22559, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="566fc018-1e62-4706-edd8-1ba4218a6ce8" colab={"base_uri": "https://localhost:8080/", "height": 55}
from google.colab import drive # import drive from google colab

ROOT = "/content/drive"     # default location for the drive
print(ROOT)                 # print content of ROOT (Optional)

drive.mount(ROOT)           # we mount the google drive at /content/drive

# + id="ZaxnFEa8pB6p" executionInfo={"status": "ok", "timestamp": 1602587785219, "user_tz": -540, "elapsed": 1289, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="112945ae-38e0-46c4-b3f1-09e55ad7c5ac" colab={"base_uri": "https://localhost:8080/", "height": 36}
# %cd 'drive/My Drive/Colab Notebooks/datamining2/src/hw/' 

# + id="wHElg7mfpTdo" executionInfo={"status": "ok", "timestamp": 1602381885112, "user_tz": -540, "elapsed": 7386, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="255fce06-9ce7-48db-fb88-3eaa61c0457c" colab={"base_uri": "https://localhost:8080/", "height": 467}
pip install jupytext #jupytext 설치 

# + id="WzejPHBbpX8C" executionInfo={"status": "ok", "timestamp": 1602381891622, "user_tz": -540, "elapsed": 3800, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="302fdb77-5bb9-4410-b4d2-32eb172563bf" colab={"base_uri": "https://localhost:8080/", "height": 92}
## Pair a notebook to a light script
# !jupytext --set-formats ipynb,py:light hw2.ipynb  


# + id="Z4CHcvQ5pfzh" executionInfo={"status": "ok", "timestamp": 1602381904902, "user_tz": -540, "elapsed": 1941, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="fd2586e5-a039-4bb4-f40b-a8fe810c0bb3" colab={"base_uri": "https://localhost:8080/", "height": 92}
# Sync the two representations
# !jupytext --sync hw2.ipynb

# + [markdown] id="vMp5jVtEqCTB"
# ## 데이터 임의로 생성

# + id="OLQYwGwSjGE_"
np.random.seed(6)
X_train = np.random.rand(800, 2) - 0.5
y_train = (X_train[:, 0] > 0).astype(np.float32) * 2

X_test = np.random.rand(100, 2) - 0.5
y_test = (X_test[:, 0] > 0).astype(np.float32) * 2


# + id="aQ-zE0ianM36" executionInfo={"status": "ok", "timestamp": 1602337660737, "user_tz": -540, "elapsed": 431, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="a7360c47-638e-4f83-872e-8401500a7f3d" colab={"base_uri": "https://localhost:8080/", "height": 36}
y_train.shape

# + [markdown] id="KzOyOPONp_ko"
# ## 모델 학습

# + id="0QiLqjU5r-FG"
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
C = 1

# + [markdown] id="JLE1d90usSBp"
# ## 1. LinearSVC

# + id="-hrTNhoZvYNq" executionInfo={"status": "ok", "timestamp": 1602337776285, "user_tz": -540, "elapsed": 769, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="caa7cf47-ff47-4728-c89c-2659e753cf51" colab={"base_uri": "https://localhost:8080/", "height": 92}
lin_clf = LinearSVC(C=C)
lin_clf.fit(X_train, y_train)

# + [markdown] id="5FjFNL42qrCU"
# ## 2. SVC with linear kernel 

# + id="Vw66NOcQjvtX" executionInfo={"status": "ok", "timestamp": 1602337777134, "user_tz": -540, "elapsed": 474, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="40e843ff-2d70-45db-9070-993c6433d0d9" colab={"base_uri": "https://localhost:8080/", "height": 92}
svm_clf = SVC(kernel="linear", C=C)
svm_clf.fit(X_train, y_train)

# + [markdown] id="bAoj2yI9v2NF"
# ## 3. SGD Classifier 

# + id="7GadygZdv7dM" executionInfo={"status": "ok", "timestamp": 1602337778271, "user_tz": -540, "elapsed": 391, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="ab64f188-b48b-4743-e7d1-5325957154ea" colab={"base_uri": "https://localhost:8080/", "height": 129}
sgd_clf = SGDClassifier(learning_rate="constant", eta0=0.001)
sgd_clf.fit(X_train, y_train)

# + [markdown] id="scf9lmyhwFQr"
# ## 학습결과

# + id="HGNJpFuKlQ2t" executionInfo={"status": "ok", "timestamp": 1602337780904, "user_tz": -540, "elapsed": 635, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="49bd08b4-ad93-4087-b816-a1787a3a203d" colab={"base_uri": "https://localhost:8080/", "height": 73}
y_pred_lin = lin_clf.predict(X_test) 
y_pred_svm = svm_clf.predict(X_test)
y_pred_sgd = sgd_clf.predict(X_test)
print('linear method Accuracy: %.2f' % accuracy_score(y_test, y_pred_lin))
print('svc with linear kernel Accuracy: %.2f' % accuracy_score(y_test, y_pred_svm))
print('sgd classifier Accuracy: %.2f' % accuracy_score(y_test, y_pred_sgd))

# + [markdown] id="LN9_BJWhyuQc"
# # Chap6 과제 

# + [markdown] id="ZUwBCVhLzHpm"
# _Exercise: train and fine-tune a Decision Tree for the moons dataset._

# + [markdown] id="_1dzr188zQHr"
# a. Generate a moons dataset using `make_moons(n_samples=10000, noise=0.4)`.

# + id="dksvsz0TzTev"
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=10000, noise=0.4, random_state=2)

# + [markdown] id="SgCRivbe0O8W"
# b. Split it into a training set and a test set using `train_test_split()`.

# + id="Ai7IU8pI0XBd"
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=2)

# + [markdown] id="MUDPL0pV1kgf"
# c. Use grid search with cross-validation (with the help of the `GridSearchCV` class) to find good hyperparameter values for a `DecisionTreeClassifier`. Hint: try various values for `max_leaf_nodes`.

# + id="qD7_sWOe1lgZ" executionInfo={"status": "ok", "timestamp": 1602339109198, "user_tz": -540, "elapsed": 15075, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="e4266a1d-78a4-44c1-dd3c-2aae57ff2d02" colab={"base_uri": "https://localhost:8080/", "height": 447}
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

params = {'max_leaf_nodes': list(range(2, 50)), 'min_samples_split': [2, 3, 4]}
grid_search_cv = GridSearchCV(DecisionTreeClassifier(random_state=2), params, verbose=1, cv=5)

grid_search_cv.fit(X_train, y_train)

# + [markdown] id="qhuezGO42zgz"
# d. Train it on the full training set using these hyperparameters, and measure your model's performance on the test set. You should get roughly 85% to 87% accuracy.

# + id="uZ6lJ6Eg20kE" executionInfo={"status": "ok", "timestamp": 1602339183189, "user_tz": -540, "elapsed": 734, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="9a267e91-65f3-4f0c-fdb3-4c4453f3ec27" colab={"base_uri": "https://localhost:8080/", "height": 36}
from sklearn.metrics import accuracy_score

y_pred = grid_search_cv.predict(X_test)
accuracy_score(y_test, y_pred)

# + [markdown] id="6goss6lU3APo"
# # Chap 7 과제

# + [markdown] id="H0Lu-s8J7gOK"
# Exercise: _Load the MNIST data and split it into a training set, a validation set, and a test set (e.g., use 50,000 instances for training, 10,000 for validation, and 10,000 for testing)._

# + [markdown] id="v0uXfd6J_ApZ"
# ## 데이터 로드 및 분할(훈련, 검증, 테스트) 

# + [markdown] id="E1QY2DmI_37A"
# - mnist 데이터 생성

# + id="rrnli4bZ_zrA"
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)

# + [markdown] id="MahdtPNs_XGL"
# - 검증과 테스트셋 10000개씩 할당. train set은 50000개 

# + id="ThIRQjUk7I2m"
from sklearn.model_selection import train_test_split
X_train_val, X_test, y_train_val, y_test = train_test_split(
    mnist.data, mnist.target, test_size=10000, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=10000, random_state=1)

# + id="b8AaL0ExAfhh" executionInfo={"status": "ok", "timestamp": 1602341728768, "user_tz": -540, "elapsed": 686, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="6cb734fa-2572-40ae-ded4-0283e518ab70" colab={"base_uri": "https://localhost:8080/", "height": 36}
X_train.shape

# + [markdown] id="odMqYsH1BcUP"
# ## 모델 학습

# + id="pXSwHAiZBmaA"
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import LinearSVC

# + [markdown] id="ibUXrkblBhyb"
# ### 1. 랜덤 포레스트 

# + id="ERQxQU0qAffp" executionInfo={"status": "error", "timestamp": 1602426095773, "user_tz": -540, "elapsed": 1117, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="99626dcd-1ae5-4fde-c7a1-3f699f4d6225" colab={"base_uri": "https://localhost:8080/", "height": 195}
random_forest_clf = RandomForestClassifier(n_estimators=80, random_state=1)
random_forest_clf.fit(X_train, y_train)

# + [markdown] id="24O3SOZNCPGK"
# ### 2. 엑스트라 트리 분류기

# + id="QPeP-h1zCODl" executionInfo={"status": "ok", "timestamp": 1602382291569, "user_tz": -540, "elapsed": 25033, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="ad26cfcd-e4d2-4ff1-c8cb-b6f999b1bac2" colab={"base_uri": "https://localhost:8080/", "height": 167}
extra_trees_clf = ExtraTreesClassifier(n_estimators=80, random_state=1)
extra_trees_clf.fit(X_train, y_train)

# + [markdown] id="jal985fYCcvW"
# ### 3.SVM 분류기 

# + id="3gi5eMVHCOBW" executionInfo={"status": "ok", "timestamp": 1602382472955, "user_tz": -540, "elapsed": 181370, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="687ce577-54ba-4ed4-a3d2-cf4f00f51ea1" colab={"base_uri": "https://localhost:8080/", "height": 129}
chap7_svm_clf = LinearSVC(random_state=1)
chap7_svm_clf.fit(X_train, y_train)

# + id="Hfsg3IhoU9sY"
#from nltk.classify.scikitlearn import SklearnClassifier
#LinearSVC_clf = SklearnClassifier(SVC(kernel='linear',probability=True))
#LinearSVC_clf.fit(X_train, y_train)

chap7_svm_clf = SVC(kernel="linear", C=C)
chap7_svm_clf.fit(X_train, y_train)

# + [markdown] id="W1gY_nT3C7Ee"
# ## 학습결과

# + [markdown] id="NZXTf8ecDiGS"
# - Validataion set 에 검증

# + id="Bgbi4sA4C9de" executionInfo={"status": "error", "timestamp": 1602426063548, "user_tz": -540, "elapsed": 2114, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="2c37fbdb-0aa1-4579-dd97-395b8a9cbe71" colab={"base_uri": "https://localhost:8080/", "height": 251}
pred_random_forest = random_forest_clf.predict(X_val) 
pred_extra_trees = extra_trees_clf.predict(X_val)
pred_svm = chap7_svm_clf.predict(X_val)
print('rf method Accuracy: %.2f' % accuracy_score(y_val, pred_random_forest))
print('et with linear kernel Accuracy: %.2f' % accuracy_score(y_val, pred_extra_trees))
print('svm classifier Accuracy: %.2f' % accuracy_score(y_val, pred_svm))

# + [markdown] id="E3-F69dyHrTW"
# ## soft voting, hard voting 

# + [markdown] id="wMRRI7oyEk01"
# ### voting classifier 

# + id="sbe8LJflEoef"
from sklearn.ensemble import VotingClassifier
named_estimators = [
    ("random_forest_clf", random_forest_clf),
    ("extra_trees_clf", extra_trees_clf),
    ("svm_clf", chap7_svm_clf),
]
voting_clf = VotingClassifier(named_estimators)
#soft_voting_clf = VotingClassifier(named_estimators, voting='soft')
#hard_voting_clf = VotingClassifier(named_estimators, voting='hard')

# + [markdown] id="y5nXrNqWd-_a"
# - hard voting 

# + id="rKxVMzdDG_o-" executionInfo={"status": "ok", "timestamp": 1602382875760, "user_tz": -540, "elapsed": 244891, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="a22fdd6d-e2e6-440e-84f0-e3c1366aea32" colab={"base_uri": "https://localhost:8080/", "height": 73}
### default = "hard" 
voting_clf.fit(X_train, y_train)
voting_clf.score(X_val, y_val)

# + [markdown] id="sPWqAcwwbAaF"
# - soft voting

# + [markdown] id="hejQEIOXbAXB"
#

# + id="fYKLkPMRQ7ki" executionInfo={"status": "error", "timestamp": 1602383000907, "user_tz": -540, "elapsed": 1831, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="c6e0bbec-1c70-4d16-aa11-530fc965612a" colab={"base_uri": "https://localhost:8080/", "height": 349}
voting_clf.voting = "soft"
voting_clf.score(X_val, y_val)

# + id="6dGHbDv6H1M8" executionInfo={"status": "error", "timestamp": 1602345091681, "user_tz": -540, "elapsed": 1757, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="1619d567-250b-4d17-a8bd-e25dcc318b71" colab={"base_uri": "https://localhost:8080/", "height": 405}
# y_pred = clf.predict(X_test)
# print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

pred_soft_voting= soft_voting_clf.predict(X_val) 
# pred_extra_trees = extra_trees_clf.predict(X_val)
# pred_svm = chap7_svm_clf.predict(X_val)
# print('rf method Accuracy: %.2f' % accuracy_score(y_val, pred_random_forest))
# print('et with linear kernel Accuracy: %.2f' % accuracy_score(y_val, pred_extra_trees))
# print('svm classifier Accuracy: %.2f' % accuracy_score(y_val, pred_svm))

# + id="0kmS24RYKwkb"



# + [markdown] id="fDIPDJ5vm4mp"
# # Large margin classification

# + [markdown] id="_T6AP6gCm4mp"
# The next few code cells generate the first figures in chapter 5. The first actual code sample comes after:
