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
#     name: python3
# ---

# + colab_type="code" id="6q2UxwfB3Cn4" colab={"base_uri": "https://localhost:8080/", "height": 127} outputId="dec34773-0dc2-4ab8-c9ae-01380f8ec55d"
from google.colab import drive # import drive from google colab

ROOT = "/content/drive"     # default location for the drive
print(ROOT)                 # print content of ROOT (Optional)

drive.mount(ROOT)           # we mount the google drive at /content/drive

# + colab_type="code" id="R_Xzz80b3waU" colab={"base_uri": "https://localhost:8080/", "height": 36} executionInfo={"status": "ok", "timestamp": 1600932857801, "user_tz": -540, "elapsed": 970, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="b7be13ec-1b19-4a65-8833-0415aa1abb60"
# %cd 'drive/My Drive/Colab Notebooks/datamining2/src/hw/' 

# + colab_type="code" id="NIfSMJtsAoLH" colab={"base_uri": "https://localhost:8080/", "height": 467} executionInfo={"status": "ok", "timestamp": 1600932867930, "user_tz": -540, "elapsed": 7253, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="38a5a260-a0ce-4827-b44a-285ba9a8a948"
pip install jupytext #jupytext 설치 

# + id="yszQG7EyCS-F" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 92} executionInfo={"status": "ok", "timestamp": 1600932903652, "user_tz": -540, "elapsed": 2419, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="114a2e8e-b784-47ef-d541-7f9ead40d088"
## Pair a notebook to a light script
# !jupytext --set-formats ipynb,py:light homework1.ipynb  

# + id="PuH5KZvuCZM8" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 92} executionInfo={"status": "ok", "timestamp": 1600932919501, "user_tz": -540, "elapsed": 1560, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="716d2d92-ff19-473b-d685-ca5626cd76e8"
# Sync the two representations
# !jupytext --sync homework1.ipynb

# + [markdown] id="CoGntZtOCh14" colab_type="text"
# #라이브러리 및 데이터 다운로드 

# + [markdown] id="YkJYTeiUaQgy" colab_type="text"
# 라이브러리, 데이터 다운. 

# + id="EDyyaDgIClZ-" colab_type="code" colab={} executionInfo={"status": "ok", "timestamp": 1600941043485, "user_tz": -540, "elapsed": 722, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}}
# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# Common imports
import numpy as np
import os

import tarfile
import urllib.request
#클라우드에 데이터가 있을때 이런식으로 써서 다운받음 
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

#데이터 받아올 fetch 함수 정의 
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH): #함수 파라미터 기본값에 경로설정 
    if not os.path.isdir(housing_path): #이 디렉토리가 없으면 만들어라 
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz") #경로 병합해서 path 설정 
    urllib.request.urlretrieve(housing_url, tgz_path) #url에 있는 파일 다운 
    housing_tgz = tarfile.open(tgz_path) #tgz파일의 압축을 풀어줌 
    housing_tgz.extractall(path=housing_path) #파일 추출 
    housing_tgz.close()


# + id="_flLpoToEVgZ" colab_type="code" colab={} executionInfo={"status": "ok", "timestamp": 1600941086448, "user_tz": -540, "elapsed": 964, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}}
fetch_housing_data() #정의한 함수 써서 데이터 불러옴. 

# + id="onfnrHIXEdf_" colab_type="code" colab={} executionInfo={"status": "ok", "timestamp": 1600941086451, "user_tz": -540, "elapsed": 576, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}}
import pandas as pd
#csv로 변환하여 pandas로 읽어오는 함수 정의 
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


# + [markdown] id="WqAiziuSaYj9" colab_type="text"
# 다운로드된 데이터 확인

# + id="AtUMn8gLEgY0" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 222} executionInfo={"status": "ok", "timestamp": 1600941086812, "user_tz": -540, "elapsed": 629, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="8602ebf6-5ce5-496f-ffb0-fb30ee2480b9"
#정의한 함수 사용 
housing = load_housing_data()
housing.head()

# + id="Bq1osCYFUF0A" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 294} executionInfo={"status": "ok", "timestamp": 1600941087071, "user_tz": -540, "elapsed": 634, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="91bd7390-022a-43b2-b20e-0756fd7038dc"
housing.describe()

# + [markdown] id="iSr_-xaVGcqS" colab_type="text"
# #데이터 전처리

# + [markdown] id="CluBtfztI4vQ" colab_type="text"
# 파생변수 생성

# + id="_SiUgd14EkOj" colab_type="code" colab={} executionInfo={"status": "ok", "timestamp": 1600941091140, "user_tz": -540, "elapsed": 658, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}}
#기존의 변수보다 더 의미있는 파생변수를 정의 
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]

# + [markdown] id="GKyOyV4BhRg1" colab_type="text"
# 기존데이터와 label로 split

# + id="TQ7BoQushA64" colab_type="code" colab={} executionInfo={"status": "ok", "timestamp": 1600941091989, "user_tz": -540, "elapsed": 427, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}}
housing_labels = housing["median_house_value"].copy() #라벨 변수 생성
housing = housing.drop("median_house_value", axis=1) #원래 데이터셋에 라벨만 드랍. 

# + [markdown] id="O8cyPBjWI6K4" colab_type="text"
# 결측치 처리 

# + id="Xg8VxufCGgYC" colab_type="code" colab={} executionInfo={"status": "ok", "timestamp": 1600941160729, "user_tz": -540, "elapsed": 714, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}}
#sklearn으로 결측치 처리하기 - median으로 대체
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median") #median으로 변경시키는 imputer 정의

# + id="ONEbQvgcVHQU" colab_type="code" colab={} executionInfo={"status": "ok", "timestamp": 1600941161453, "user_tz": -540, "elapsed": 520, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}}
housing_num = housing.drop("ocean_proximity", axis=1) #텍스트 변수는 제외

# + id="tBcTQKH_GxEv" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 55} executionInfo={"status": "ok", "timestamp": 1600941162376, "user_tz": -540, "elapsed": 447, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="e52ce064-5896-4742-9a70-2a7ca320d12b"
imputer.fit(housing_num) #정의한 imputer 하여 fit. 

# + id="aQDEQlfMHA1J" colab_type="code" colab={} executionInfo={"status": "ok", "timestamp": 1600941209777, "user_tz": -540, "elapsed": 908, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}}
X = imputer.transform(housing_num)

# + id="xq9QK8pVIk5Z" colab_type="code" colab={} executionInfo={"status": "ok", "timestamp": 1600941209778, "user_tz": -540, "elapsed": 542, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}}
#리턴값이 numpy형태라서 판다스로 데이터형변경 
housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                          index=housing.index)

# + [markdown] id="NaqhVFwoI0m9" colab_type="text"
# 범주형 데이터 : on-hot encoding 파이프라인 정의 

# + id="_bWsIbEsI-9T" colab_type="code" colab={} executionInfo={"status": "ok", "timestamp": 1600941221024, "user_tz": -540, "elapsed": 687, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}}
#housing_cat = housing[["ocean_proximity"]]

# + id="yv5gXTNpdw34" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 148} executionInfo={"status": "ok", "timestamp": 1600941221398, "user_tz": -540, "elapsed": 424, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="cdec66a8-ae50-4d73-a25c-4991baf0b26c"
#원핫인코딩(더미변수) 방법사용 
# from sklearn.preprocessing import OneHotEncoder
# cat_encoder = OneHotEncoder(sparse=False)
# housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
# housing_cat_1hot

# + [markdown] id="kLFSkil6dn7v" colab_type="text"
# 수치형데이터 : num_pipeline 정의

# + id="Seysj9ezdeN_" colab_type="code" colab={} executionInfo={"status": "ok", "timestamp": 1600941224045, "user_tz": -540, "elapsed": 630, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}}
from sklearn.base import BaseEstimator, TransformerMixin

# column index
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

# + id="cfQT7HvRdHV1" colab_type="code" colab={} executionInfo={"status": "ok", "timestamp": 1600941225301, "user_tz": -540, "elapsed": 932, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}}
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")), #중간값 대체 
        ('attribs_adder', CombinedAttributesAdder()), #파생변수추가. 
        ('std_scaler', StandardScaler()),  #데이터 스케일링함. 
    ])

housing_num_tr = num_pipeline.fit_transform(housing_num)

# + [markdown] id="vWNfBpdmdra-" colab_type="text"
# 수치형 파이프라인과 범주형 파이프라인(one-hot encoding)으로 full pipeline을 만듦. 이후 데이터에 적용하여 변환

# + id="0S68WI0scqEM" colab_type="code" colab={} executionInfo={"status": "ok", "timestamp": 1600941228057, "user_tz": -540, "elapsed": 733, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}}
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

#full pipeline 만듦. 
full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

housing_prepared = full_pipeline.fit_transform(housing)

# + [markdown] id="y2rzSPhjcTDy" colab_type="text"
# 파이프라인을 통과시킨 데이터(housing_prepared)를 준비된 label(housing_labels)에 fit 시킴.

# + id="u9JuB8LKjkEv" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 409} executionInfo={"status": "ok", "timestamp": 1600941750572, "user_tz": -540, "elapsed": 80380, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="2e0a65b3-9db9-4dba-a256-e1f351205b4d"
#하이퍼파라미터 튜닝. 
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

#튜닝할 파라미터들 정의 
param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)

# + id="PDJFqPRskjgq" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 36} executionInfo={"status": "ok", "timestamp": 1600946044853, "user_tz": -540, "elapsed": 34863, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="5b9e18cb-031a-4935-9d76-c6b581e7b759"
# from sklearn.svm import SVR
# from sklearn.metrics import mean_squared_error
# from sklearn.model_selection import GridSearchCV

# svm_reg = SVR(kernel="linear")


# svm_reg.fit(housing_prepared, housing_labels)
# housing_predictions = svm_reg.predict(housing_prepared)
# svm_mse = mean_squared_error(housing_labels, housing_predictions)
# svm_rmse = np.sqrt(svm_mse)
# svm_rmse

# + [markdown] id="sgGAqPTU4vP4" colab_type="text"
# #모델 적합 및 결과 출력 

# + [markdown] id="cTKp-bzD44tg" colab_type="text"
# ##예제 1 grid search 사용하여 모델 선택

# + [markdown] id="jiB-gaYisLH0" colab_type="text"
# Linear kernel 및 5-fold cross validation사용하여 하이퍼파라미터 튜닝.   

# + id="f0w2b7CSklwM" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 185} executionInfo={"status": "ok", "timestamp": 1600948238915, "user_tz": -540, "elapsed": 435034, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="2569cf6f-d997-4e8d-d405-a3bcd1be7f53"
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

svm_linear = SVR(kernel = 'linear') #linear kernel 사용 
param_grid = {'C': [0.01, 0.1, 1, 10, 25]} # 탐색할 하이퍼파라미터 C 정의 

grid_svm_linear = GridSearchCV(svm_linear,
                               param_grid = param_grid, 
                               scoring='neg_mean_squared_error',
                               cv = 5) #모델 생성. 

grid_svm_linear.fit(housing_prepared, housing_labels) #fit

# + id="36_ElaQh9uZf" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 111} executionInfo={"status": "ok", "timestamp": 1600948466632, "user_tz": -540, "elapsed": 1114, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="ce51a5c0-a9fb-41a9-f00f-d1261ddf3ece"
#cv결과 확인
cvres = grid_svm_linear.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

# + id="IfM1Xy5Y3WPH" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 36} executionInfo={"status": "ok", "timestamp": 1600948469341, "user_tz": -540, "elapsed": 777, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="714c413d-f1b0-4d93-9b68-83d73537bddd"
#베스트파라미터 확인
grid_svm_linear.best_params_

# + id="SP0En9eH6EP7" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 185} executionInfo={"status": "ok", "timestamp": 1600955314256, "user_tz": -540, "elapsed": 2248353, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="f68b9a96-abe3-4ecc-e64b-625f987fdb29"
svm_rbf = SVR(kernel = 'rbf') #rbf커널 사용. 

param_grid2 = {'C': [0.01, 0.1, 1, 10], #튜닝할 하이퍼파라미터 c와 gamma 범위 설정. 
              'gamma':[0.01, 0.1, 1, 10]}

grid_svm_rbf = GridSearchCV(svm_rbf,
                            scoring='neg_mean_squared_error',
                            param_grid = param_grid2,
                            cv = 5)

grid_svm_rbf.fit(housing_prepared, housing_labels)


# + id="kDrnFFedFaKy" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 36} executionInfo={"status": "ok", "timestamp": 1600955591245, "user_tz": -540, "elapsed": 709, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="d84fa580-92a1-448d-9169-2cd5d398cf10"
grid_svm_rbf.best_params_

# + id="TnbtteIl9T4O" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 316} executionInfo={"status": "ok", "timestamp": 1600955594350, "user_tz": -540, "elapsed": 779, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="7c99f164-3660-4796-bb08-8240223f1c73"
#cv결과 확인
cvres2 = grid_svm_rbf.cv_results_
for mean_score, params in zip(cvres2["mean_test_score"], cvres2["params"]):
    print(np.sqrt(-mean_score), params)

# + id="pVjonksX3rfl" colab_type="code" colab={}



# + [markdown] id="J7wECVCG5BCb" colab_type="text"
# ## Randomized Search CV 를 사용하여 모델 선택

# + id="r9xMmpM9S8pD" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 241} executionInfo={"status": "ok", "timestamp": 1600963392060, "user_tz": -540, "elapsed": 330, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="809c7edd-f77d-4429-cca1-bdd273f6181c"
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import expon, reciprocal

param_distribs = {
        'kernel': ['linear', 'rbf'],
        'C': reciprocal(20, 200000),
        'gamma': expon(scale=1.0),
    }

svm_reg = SVR()
rnd_search = RandomizedSearchCV(svm_reg, param_distributions=param_distribs,
                                n_iter=50, cv=5, scoring='neg_mean_squared_error')
rnd_search.fit(housing_prepared, housing_labels)

# + id="Vx7vRDWaS8l3" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 176} executionInfo={"status": "error", "timestamp": 1601000781489, "user_tz": -540, "elapsed": 1870, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="a77ad5d1-9703-44a4-9afa-024ac1c5f629"
rnd_search.best_params_
