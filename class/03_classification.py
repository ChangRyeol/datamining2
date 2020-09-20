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

# + [markdown] id="view-in-github" colab_type="text"
# <a href="https://colab.research.google.com/github/ChangRyeol/datamining2/blob/master/class/03_classification.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="M_4dYGwLo6Af" colab_type="text"
# **Chapter 3 – Classification**
#
# _This notebook contains all the sample code and solutions to the exercises in chapter 3._

# + [markdown] id="5AkcwxLyo6Aj" colab_type="text"
# <table align="left">
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/ageron/handson-ml2/blob/master/03_classification.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
#   </td>
# </table>

# + [markdown] id="UWSZCUXDfRSb" colab_type="text"
# #goole drive mounting

# + id="uak15B4jfPP7" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 55} executionInfo={"status": "ok", "timestamp": 1600605052214, "user_tz": -540, "elapsed": 24903, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="296fe188-c433-4e43-b035-106771c3c015"
from google.colab import drive # import drive from google colab

ROOT = "/content/drive"     # default location for the drive
print(ROOT)                 # print content of ROOT (Optional)

drive.mount(ROOT)           # we mount the google drive at /content/drive

# + [markdown] id="K806ELG2fXcu" colab_type="text"
# ##syn py and ipynb 

# + id="Cpon8anzfWRU" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 36} executionInfo={"status": "ok", "timestamp": 1600605001610, "user_tz": -540, "elapsed": 776, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="f01880f2-d88f-48b0-fafc-b480720ecf42"
# %pwd 

# + id="wXCBSKYlfh-0" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 55} executionInfo={"status": "ok", "timestamp": 1600605067727, "user_tz": -540, "elapsed": 623, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="3c24d8c0-6fbe-498d-bcf7-71bb80d32120"
# %cd 'drive/My Drive/Colab Notebooks/datamining2/class' 

# + id="PtDnAwquf9tT" colab_type="code" colab={}



# + id="ymtMPYZLf6xe" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 36} executionInfo={"status": "ok", "timestamp": 1600605133936, "user_tz": -540, "elapsed": 616, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="b4905ae0-2168-4530-e30f-2fe53fd243cd"
## Pair a notebook to a light script
# !jupytext --set-formats ipynb,py:light 03_end_to_end_machine_learning_project.ipynb  


# + [markdown] id="KvY2OWwro6Am" colab_type="text"
# # Setup

# + [markdown] id="TJcB-ODbo6An" colab_type="text"
# First, let's import a few common modules, ensure MatplotLib plots figures inline and prepare a function to save the figures. We also check that Python 3.5 or later is installed (although Python 2.x may work, it is deprecated so we strongly recommend you use Python 3 instead), as well as Scikit-Learn ≥0.20.

# + [markdown] id="FBI_woSSwP-k" colab_type="text"
# # commit test!! 

# + id="8jl0efRQo6Ap" colab_type="code" colab={}
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
CHAPTER_ID = "classification"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


# + [markdown] id="lRu35nK3o6Aw" colab_type="text"
# # MNIST

# + id="yRI9wbcto6Ay" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 35} outputId="10949ee0-68f4-4617-a4d6-00ee49ef3881"
 #mnist가 유명한데이터셋이라 sklearn 등 라이브러리에서 불러올 수 있음. 
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)
mnist.keys() 


# + id="eZvaaemzo6A4" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 35} outputId="08619bc1-0792-47dd-cb55-cb15a130067a"
#mnist의 key에서 data와 target만을 뽑아옴. 
X, y = mnist["data"], mnist["target"] 
X.shape

# + id="erqLJxDno6A8" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 35} outputId="4a165e8c-d561-4ad3-e05d-a06cf6cd761e"
y.shape

# + id="yyuxidrPo6A_" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 35} outputId="5bf72c44-148a-4edc-ae0b-46078befd370"
28 * 28 #28필셀*28픽셀 = 784 

# + id="NDLkiF1vo6BD" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 315} outputId="ea76c1b2-2954-4b83-d077-ad0933c42abd"
# %matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt

some_digit = X[0] #첫번째 digit을 가져와옴. 차원이 784라서 28*28로 reshape 
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap=mpl.cm.binary)
plt.axis("off")

save_fig("some_digit_plot")
plt.show()

# + id="KnWokmxRo6BH" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 35} outputId="0b8aca83-2511-4a24-be72-a16c3cd2cc2a"
y[0]

# + id="CJOVT8sho6BK" colab_type="code" colab={}
y = y.astype(np.uint8)


# + id="0bvFwWnMo6BN" colab_type="code" colab={}
#임의의 데이터에서 28*28로 reshape해주는 함수를 정의 
def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap = mpl.cm.binary,
               interpolation="nearest")
    plt.axis("off")


# + id="V5S3TD8Ko6BQ" colab_type="code" colab={}
## 그림으로 그려주는 함수인데 관심있으면 알아서 더 보기 
# EXTRA
def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = mpl.cm.binary, **options)
    plt.axis("off")


# + id="OuyNZ9N5o6BT" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 675} outputId="4ca08f96-ff2d-4ce8-cb46-9f7189dca7a9"
plt.figure(figsize=(9,9))
example_images = X[:100]
plot_digits(example_images, images_per_row=10)
save_fig("more_digits_plot")
plt.show()

# + id="ElVsayqgo6BY" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 35} outputId="2c70cde9-195a-48fd-b7bf-0d5e1df463fa"
y[0]

# + id="j7-QSIUko6Bd" colab_type="code" colab={}
#train와 test로 나누기 
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# + [markdown] id="cdTw2NH7o6Bg" colab_type="text"
# # Binary classifier

# + id="xmW7zPiNo6Bg" colab_type="code" colab={}
#5번 1 아니면 0 이런식으로 구성 
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

# + [markdown] id="kYGUkZ-Qo6Bk" colab_type="text"
# **Note**: some hyperparameters will have a different defaut value in future versions of Scikit-Learn, such as `max_iter` and `tol`. To be future-proof, we explicitly set these hyperparameters to their future default values. For simplicity, this is not shown in the book.

# + id="oENigBNbo6Bl" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 125} outputId="66e4d5b5-1505-490f-bc8a-a7c37126c081"
#SGDClassifier 만들디 
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
sgd_clf.fit(X_train, y_train_5)

# + id="Ey_MGKW-o6Br" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 35} outputId="67b3e9e2-bc05-4d6c-8c0c-7d231377a1bf"
#some digit넣어보니 5가 맞다! 
sgd_clf.predict([some_digit])

# + id="i8ATSfEvo6Bu" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 35} outputId="f547faaa-0779-46c8-826a-bc41a427489f"
#cv로 돌려보기 
from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
#0.95가 넘으니까 좋은거 아닌가? 아니다! 

# + id="XK5rzYJco6Bx" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 127} outputId="c74a0e9b-6fe3-48e4-e662-d481d34ab302"

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds = StratifiedKFold(n_splits=3, random_state=42)

for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_fold = X_train[test_index]
    y_test_fold = y_train_5[test_index]

    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))

# + id="f-Q032rPo6Bz" colab_type="code" colab={}
#그냥 다 5가 아니라고 하는 분류기를 만들어도 0.9가 넘는다. 
#따라서 class가 unbalance할땐 다른 지표들도 보는게 좋다. 
from sklearn.base import BaseEstimator
class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)


# + id="wWkwOYWco6B1" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 35} outputId="fd45d713-456a-48dd-e214-ffbf07f2793c"
never_5_clf = Never5Classifier()
cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")

# + id="IPwAAUwDo6B4" colab_type="code" colab={}
#sgd로 cv하여 fit해보기 
from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

# + id="1IY8DF8Xo6B7" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 53} outputId="4f67203d-6c66-4423-8e4a-f5adeb507f8f"
#위의 결과를 confusion matrix로 표현 
from sklearn.metrics import confusion_matrix

confusion_matrix(y_train_5, y_train_pred)

# + id="lPT4uTbko6B_" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 53} outputId="b4d75960-4a08-48ba-aa45-8f478c3b5168"
#confusion matrix는 대각선 숫자가 높을수록 예측이 잘된것. 
y_train_perfect_predictions = y_train_5  # pretend we reached perfection
confusion_matrix(y_train_5, y_train_perfect_predictions)

# + id="6zInj51po6CC" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 35} outputId="68784546-cfcb-42aa-e1ae-8d1787c7ad29"
from sklearn.metrics import precision_score, recall_score

precision_score(y_train_5, y_train_pred)

# + id="APe-_70uo6CF" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 35} outputId="0882ca1d-8938-4334-f91a-57913e0788f4"
##precision score를 confusion matrix보고 수기로 계산 
3530 / (3530 + 687)

# + id="WhszspIPo6CI" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 35} outputId="9a2cd9f6-1745-4af9-8b03-1910c178e8be"
recall_score(y_train_5, y_train_pred)

# + id="O584con4o6CN" colab_type="code" colab={} outputId="b5e1400f-485c-49bc-c41a-73321698582d"
#recall도 confusion matirx로 계산가능. 지금 아래는 숫자가 봐뀐듯.. diy 
4096 / (4096 + 1325)

# + id="xGTcn_ego6CR" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 35} outputId="836043a5-ffc3-4e19-9476-cb277d6aba27"
from sklearn.metrics import f1_score

f1_score(y_train_5, y_train_pred)

# + id="dZa8mzEKo6CU" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 35} outputId="c3649b23-2372-4596-aa39-8d88a538a31d"
4096 / (4096 + (1522 + 1325) / 2)

# + id="8b31NC2fEDkB" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 35} outputId="cb1483ad-add5-4cca-a5cc-9255a8dcdfbf"
sgd_clf.predict([some_digit]) #predict함수를 쓰면 t/f 로 반환 

# + id="SNVb90pfo6CY" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 35} outputId="8cdab909-7fd3-413e-9796-bcb22915a09f"
y_scores = sgd_clf.decision_function([some_digit]) 
y_scores #decision_funcion함수를 사용하면 스코어가 값으로 나타남. threshold에 따른 분류를 설정하기 위해서는 스코어값이 필요하기때문에 사용.  

# + id="PG9sRpBIo6Cb" colab_type="code" colab={}
threshold = 0 #threshold를 0을 기준으로 판별
y_some_digit_pred = (y_scores > threshold)

# + id="s7azr3KHo6Ce" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 35} outputId="5c4c662e-7ce0-4e77-9da0-35dccc623665"
y_some_digit_pred

# + id="xfOHce0ro6Ch" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 35} outputId="be550b87-5a72-44df-fa8f-8cf0b2f4e919"
#threshold를 변경하면 결과가 달라진다. 
threshold = 8000
y_some_digit_pred = (y_scores > threshold)
y_some_digit_pred

# + id="gD_TQEhvo6Ck" colab_type="code" colab={}
#cv로 위에것을 다시해보는 코드. 
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
                             method="decision_function")

# + id="lfCGrC2_o6Cn" colab_type="code" colab={}
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)


# + id="xSrez9a6o6Cs" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 315} outputId="27bdc79f-bee9-4801-dd37-38de89792569"
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.legend(loc="center right", fontsize=16) # Not shown in the book
    plt.xlabel("Threshold", fontsize=16)        # Not shown
    plt.grid(True)                              # Not shown
    plt.axis([-50000, 50000, 0, 1])             # Not shown



recall_90_precision = recalls[np.argmax(precisions >= 0.90)]
threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]


plt.figure(figsize=(8, 4))                                                                  # Not shown
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.plot([threshold_90_precision, threshold_90_precision], [0., 0.9], "r:")                 # Not shown
plt.plot([-50000, threshold_90_precision], [0.9, 0.9], "r:")                                # Not shown
plt.plot([-50000, threshold_90_precision], [recall_90_precision, recall_90_precision], "r:")# Not shown
plt.plot([threshold_90_precision], [0.9], "ro")                                             # Not shown
plt.plot([threshold_90_precision], [recall_90_precision], "ro")                             # Not shown
save_fig("precision_recall_vs_threshold_plot")                                              # Not shown
plt.show()

# + id="ck0JeWMKo6Cx" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 35} outputId="c39e5a43-6ac1-4ccf-8494-d0d93c3901e7"
(y_train_pred == (y_scores > 0)).all()


# + id="l-umKn7eo6Cz" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 459} outputId="174e4467-34cb-46cc-f124-cc92ddf7c436"
def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])
    plt.grid(True)

plt.figure(figsize=(8, 6))
plot_precision_vs_recall(precisions, recalls)
plt.plot([0.4368, 0.4368], [0., 0.9], "r:")
plt.plot([0.0, 0.4368], [0.9, 0.9], "r:")
plt.plot([0.4368], [0.9], "ro")
save_fig("precision_vs_recall_plot")
plt.show()

# + id="j7FNWAaso6C3" colab_type="code" colab={}
#threshold를 설정하는 전략. 
threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]

# + id="Mw0EOPsjo6C7" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 35} outputId="2e801dce-a5b5-44e3-baeb-f5603694254e"
threshold_90_precision

# + id="W4ZDNCn0o6DD" colab_type="code" colab={}
y_train_pred_90 = (y_scores >= threshold_90_precision)

# + id="uljpTpRzo6DG" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 35} outputId="ee44c175-0d37-4b2e-bd55-3e7a7cd28871"
precision_score(y_train_5, y_train_pred_90)

# + id="_KJKnFzFo6DL" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 35} outputId="c1ab3ee5-5a87-4ad2-eb37-1ca000913bc5"
recall_score(y_train_5, y_train_pred_90)

# + [markdown] id="YWP_-5nTo6DR" colab_type="text"
# # ROC curves

# + id="av6M5uI7o6DR" colab_type="code" colab={}
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)


# + id="r2xopy1Oo6DW" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 459} outputId="32d4a17c-08d7-4ce0-e102-cf39fc29df93"
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal
    plt.axis([0, 1, 0, 1])                                    # Not shown in the book
    plt.xlabel('False Positive Rate (Fall-Out)', fontsize=16) # Not shown
    plt.ylabel('True Positive Rate (Recall)', fontsize=16)    # Not shown
    plt.grid(True)                                            # Not shown

plt.figure(figsize=(8, 6))                         # Not shown
plot_roc_curve(fpr, tpr)
plt.plot([4.837e-3, 4.837e-3], [0., 0.4368], "r:") # Not shown
plt.plot([0.0, 4.837e-3], [0.4368, 0.4368], "r:")  # Not shown
plt.plot([4.837e-3], [0.4368], "ro")               # Not shown
save_fig("roc_curve_plot")                         # Not shown
plt.show()

# + id="axaTtRPno6DZ" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 35} outputId="11b7a3c6-0f54-41b9-8311-1c360d091901"
from sklearn.metrics import roc_auc_score

roc_auc_score(y_train_5, y_scores)

# + [markdown] id="EYZkqBzzo6De" colab_type="text"
# **Note**: we set `n_estimators=100` to be future-proof since this will be the default value in Scikit-Learn 0.22.

# + id="uH_9NmlBo6De" colab_type="code" colab={}
from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,
                                    method="predict_proba")

# + id="mybD6APgo6Dh" colab_type="code" colab={}
y_scores_forest = y_probas_forest[:, 1] # score = proba of positive class
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5,y_scores_forest)

# + id="XghkJ612o6Dk" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 459} outputId="745d5795-c6d7-4432-ac6f-b9e06365a1bf"
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, "b:", linewidth=2, label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.plot([4.837e-3, 4.837e-3], [0., 0.4368], "r:")
plt.plot([0.0, 4.837e-3], [0.4368, 0.4368], "r:")
plt.plot([4.837e-3], [0.4368], "ro")
plt.plot([4.837e-3, 4.837e-3], [0., 0.9487], "r:")
plt.plot([4.837e-3], [0.9487], "ro")
plt.grid(True)
plt.legend(loc="lower right", fontsize=16)
save_fig("roc_curve_comparison_plot")
plt.show()
#rf가 더 좋은모형이란것을 시각적으로 알 수 있음. 

# + id="CKBcHusso6Do" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 35} outputId="e653cd16-18bb-4349-caf2-76f2cd9cadbc"
roc_auc_score(y_train_5, y_scores_forest)

# + id="unrSJctco6Dt" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 35} outputId="d57bb59e-512e-4bc3-dda0-3d19857a3e89"
y_train_pred_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3)
precision_score(y_train_5, y_train_pred_forest)

# + id="HLVzBp6uo6Dx" colab_type="code" colab={} outputId="7fc39653-07c5-4fac-b0af-7d6cbc0fb276"
recall_score(y_train_5, y_train_pred_forest)

# + [markdown] id="03hMSgy5o6Dz" colab_type="text"
# # Multiclass classification

# + id="LKqc4Py9o6D0" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 35} outputId="bfeaa2a2-22aa-4775-c349-843b8919a832"
from sklearn.svm import SVC

svm_clf = SVC(gamma="auto", random_state=42)
svm_clf.fit(X_train[:1000], y_train[:1000]) # y_train, not y_train_5
svm_clf.predict([some_digit])

# + id="TusFauKNo6D4" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 53} outputId="38752153-4ccd-421a-9be1-5b995f11b19d"
some_digit_scores = svm_clf.decision_function([some_digit])
some_digit_scores #0부터 9까지중 5에서 score가 가장큼 

# + id="ulMuUr3Yo6D6" colab_type="code" colab={} outputId="4c7c894a-fb98-4a99-9735-ae0200fd9418"
np.argmax(some_digit_scores) #max를 찾으면 5

# + id="lv4V3Ry2o6D-" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 35} outputId="bcc604f8-cedc-4afc-938a-6a769d1e786c"
svm_clf.classes_ #클래스는 0부터 9까지 

# + id="sTBygyjDo6EA" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 35} outputId="76e92381-4550-4bb2-f28a-13ec7093b9e2"
svm_clf.classes_[5] #다섯번째 위치에 5가 있다. 

# + id="6XHQHboHo6ED" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 35} outputId="1358267b-573c-414e-857c-085343d73b67"
# OneVsRestClassifier를 사용하여 분류 
from sklearn.multiclass import OneVsRestClassifier
ovr_clf = OneVsRestClassifier(SVC(gamma="auto", random_state=42))
ovr_clf.fit(X_train[:1000], y_train[:1000])
ovr_clf.predict([some_digit])

# + id="hUAQldzxo6EG" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 35} outputId="fe8e0fe6-bcf3-4df7-eeec-ea930391515f"
len(ovr_clf.estimators_)

# + id="2fsKyQvto6EJ" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 35} outputId="d9cf86e2-ac5a-4897-9f90-7300b4fcac65"
#sgd로 predict해도 5. 
sgd_clf.fit(X_train, y_train)
sgd_clf.predict([some_digit])

# + id="9lG6--G1o6EL" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 89} outputId="4d65e689-1f65-4eb0-b643-0b31c8b9403a"
sgd_clf.decision_function([some_digit])

# + id="-xUkmHdJo6EN" colab_type="code" colab={}
cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")

# + id="2jnT0x5So6EP" colab_type="code" colab={}
#정규화했더니 성능이 더 좋아지더라 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")

# + id="fyG9Mzrio6ES" colab_type="code" colab={}
y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)
conf_mx


# + id="M5Z6Wp-Io6EV" colab_type="code" colab={}
# since sklearn 0.22, you can use sklearn.metrics.plot_confusion_matrix()
def plot_confusion_matrix(matrix):
    """If you prefer color and a colorbar"""
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix)
    fig.colorbar(cax)


# + id="e6BB5WITo6EX" colab_type="code" colab={}
plt.matshow(conf_mx, cmap=plt.cm.gray)
save_fig("confusion_matrix_plot", tight_layout=False)
plt.show()

# + id="AQo6QEd4o6Ea" colab_type="code" colab={}
row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums

# + id="CaVr-v-jo6Ed" colab_type="code" colab={}
np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
save_fig("confusion_matrix_errors_plot", tight_layout=False)
plt.show()

# + id="pWSXCvaro6Eg" colab_type="code" colab={} outputId="4d02cf76-550e-4a6c-c20c-9538801d2fdb"
##3인데 5라고 분류한것, 5인데 3이라고 분류한거 등을 보여줌 
cl_a, cl_b = 3, 5
X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]

plt.figure(figsize=(8,8))
plt.subplot(221); plot_digits(X_aa[:25], images_per_row=5)
plt.subplot(222); plot_digits(X_ab[:25], images_per_row=5)
plt.subplot(223); plot_digits(X_ba[:25], images_per_row=5)
plt.subplot(224); plot_digits(X_bb[:25], images_per_row=5)
save_fig("error_analysis_digits_plot")
plt.show()

# + [markdown] id="vy_zw8vko6El" colab_type="text"
# # Multilabel classification

# + id="T7qKRzRPo6El" colab_type="code" colab={}
#knn으로 분류 
from sklearn.neighbors import KNeighborsClassifier

y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)

# + id="YKArv-Lio6En" colab_type="code" colab={}
knn_clf.predict([some_digit])

# + [markdown] id="YDjHEsJzo6Ep" colab_type="text"
# **Warning**: the following cell may take a very long time (possibly hours depending on your hardware).

# + id="AZj16CFUo6Eq" colab_type="code" colab={}
#knn의 f1구하기 
y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)
f1_score(y_multilabel, y_train_knn_pred, average="macro")

# + [markdown] id="3X6ZUeUbo6Es" colab_type="text"
# # Multioutput classification

# + id="BEyQczKfo6Et" colab_type="code" colab={}
#noise가 추가되었을 때, 이를 training으로 쓰고 noise없는 것을 label로 씀. 
noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise
noise = np.random.randint(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise
y_train_mod = X_train
y_test_mod = X_test

# + id="2R9xQor-o6Ev" colab_type="code" colab={} outputId="25211551-5f9c-4f58-9217-867af409bcdd"
some_index = 0
plt.subplot(121); plot_digit(X_test_mod[some_index])
plt.subplot(122); plot_digit(y_test_mod[some_index])
save_fig("noisy_digit_example_plot")
plt.show()

# + id="oz1XxGC-o6Ez" colab_type="code" colab={} outputId="d954bc3e-7d9b-45d5-f114-a2443273d50c"
knn_clf.fit(X_train_mod, y_train_mod)
clean_digit = knn_clf.predict([X_test_mod[some_index]])
plot_digit(clean_digit)
save_fig("cleaned_digit_example_plot")

# + [markdown] id="XzcVuO2do6E2" colab_type="text"
# # Extra material

# + [markdown] id="Kvr8JRNUo6E2" colab_type="text"
# ## Dummy (ie. random) classifier

# + id="1Ik6-0oJo6E3" colab_type="code" colab={}
from sklearn.dummy import DummyClassifier
dmy_clf = DummyClassifier()
y_probas_dmy = cross_val_predict(dmy_clf, X_train, y_train_5, cv=3, method="predict_proba")
y_scores_dmy = y_probas_dmy[:, 1]

# + id="_DJlTes4o6E6" colab_type="code" colab={} outputId="b3ff479f-2374-49bf-b3da-fac8ee5308db"
fprr, tprr, thresholdsr = roc_curve(y_train_5, y_scores_dmy)
plot_roc_curve(fprr, tprr)

# + [markdown] id="hi1brcFPo6E9" colab_type="text"
# ## KNN classifier

# + id="ng5hWEcAo6E9" colab_type="code" colab={} outputId="d1efdf08-45c6-4b1f-b82e-d6a9e4412f88"
from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier(weights='distance', n_neighbors=4)
knn_clf.fit(X_train, y_train)

# + id="ww2bLelbo6FA" colab_type="code" colab={}
y_knn_pred = knn_clf.predict(X_test)

# + id="4RrjHE-Xo6FC" colab_type="code" colab={} outputId="1c013741-ebdc-4735-afec-40f650f607dc"
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_knn_pred)

# + id="Z_XFCAuyo6FG" colab_type="code" colab={} outputId="e091d203-7211-46f8-b5f6-c170a057c8a0"
from scipy.ndimage.interpolation import shift
def shift_digit(digit_array, dx, dy, new=0):
    return shift(digit_array.reshape(28, 28), [dy, dx], cval=new).reshape(784)

plot_digit(shift_digit(some_digit, 5, 1, new=100))

# + id="j9HA3Afbo6FI" colab_type="code" colab={} outputId="e6f2cfdf-1211-447a-cb0c-3f507ec3a337"
X_train_expanded = [X_train]
y_train_expanded = [y_train]
for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
    shifted_images = np.apply_along_axis(shift_digit, axis=1, arr=X_train, dx=dx, dy=dy)
    X_train_expanded.append(shifted_images)
    y_train_expanded.append(y_train)

X_train_expanded = np.concatenate(X_train_expanded)
y_train_expanded = np.concatenate(y_train_expanded)
X_train_expanded.shape, y_train_expanded.shape

# + id="UqTQh4Fxo6FL" colab_type="code" colab={} outputId="2057bb5e-c27f-466d-ffba-729445662438"
knn_clf.fit(X_train_expanded, y_train_expanded)

# + id="BkVmhudHo6FN" colab_type="code" colab={}
y_knn_expanded_pred = knn_clf.predict(X_test)

# + id="hET21XLso6FP" colab_type="code" colab={} outputId="a8e50ceb-9e24-43b8-8066-453b24cbd593"
accuracy_score(y_test, y_knn_expanded_pred)

# + id="EVw__PpCo6FR" colab_type="code" colab={} outputId="f7a6b7fd-eb00-42f8-8f6e-3b36b2ec4746"
ambiguous_digit = X_test[2589]
knn_clf.predict_proba([ambiguous_digit])

# + id="Ynq1hz8xo6FT" colab_type="code" colab={} outputId="d227f829-cb5d-4105-dee3-1350c4f9674a"
plot_digit(ambiguous_digit)

# + [markdown] id="132EeLW7o6FV" colab_type="text"
# # Exercise solutions

# + [markdown] id="QIs4fS5No6FV" colab_type="text"
# ## 1. An MNIST Classifier With Over 97% Accuracy

# + [markdown] id="o0cP7FSXo6FV" colab_type="text"
# **Warning**: the next cell may take hours to run, depending on your hardware.

# + id="e8XHX0-Ho6FV" colab_type="code" colab={} outputId="9763edd0-96af-421f-cf8e-0361caf02399"
from sklearn.model_selection import GridSearchCV

param_grid = [{'weights': ["uniform", "distance"], 'n_neighbors': [3, 4, 5]}]

knn_clf = KNeighborsClassifier()
grid_search = GridSearchCV(knn_clf, param_grid, cv=5, verbose=3)
grid_search.fit(X_train, y_train)

# + id="OD3hWvLOo6FX" colab_type="code" colab={} outputId="607f798c-901d-4df1-b756-b3b6b4e4ebc2"
grid_search.best_params_

# + id="VI0w2c37o6FZ" colab_type="code" colab={} outputId="6003b9f7-b262-41c3-acae-0225e6d7df48"
grid_search.best_score_

# + id="jnW8qm2Mo6Fb" colab_type="code" colab={} outputId="9c7ce45a-5493-4ce0-f628-6e0785c9ba45"
from sklearn.metrics import accuracy_score

y_pred = grid_search.predict(X_test)
accuracy_score(y_test, y_pred)

# + [markdown] id="v7RHs_XMo6Fd" colab_type="text"
# ## 2. Data Augmentation

# + id="OSaEknFQo6Fd" colab_type="code" colab={}
from scipy.ndimage.interpolation import shift


# + id="JzTEwUapo6Fg" colab_type="code" colab={}
def shift_image(image, dx, dy):
    image = image.reshape((28, 28))
    shifted_image = shift(image, [dy, dx], cval=0, mode="constant")
    return shifted_image.reshape([-1])


# + id="P-hYDkQ_o6Fi" colab_type="code" colab={} outputId="c58e34f9-f2f8-40ca-ea0b-122412864162"
image = X_train[1000]
shifted_image_down = shift_image(image, 0, 5)
shifted_image_left = shift_image(image, -5, 0)

plt.figure(figsize=(12,3))
plt.subplot(131)
plt.title("Original", fontsize=14)
plt.imshow(image.reshape(28, 28), interpolation="nearest", cmap="Greys")
plt.subplot(132)
plt.title("Shifted down", fontsize=14)
plt.imshow(shifted_image_down.reshape(28, 28), interpolation="nearest", cmap="Greys")
plt.subplot(133)
plt.title("Shifted left", fontsize=14)
plt.imshow(shifted_image_left.reshape(28, 28), interpolation="nearest", cmap="Greys")
plt.show()

# + id="6ZrewO4_o6Fm" colab_type="code" colab={}
X_train_augmented = [image for image in X_train]
y_train_augmented = [label for label in y_train]

for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
    for image, label in zip(X_train, y_train):
        X_train_augmented.append(shift_image(image, dx, dy))
        y_train_augmented.append(label)

X_train_augmented = np.array(X_train_augmented)
y_train_augmented = np.array(y_train_augmented)

# + id="6ti2CIG5o6Fo" colab_type="code" colab={}
shuffle_idx = np.random.permutation(len(X_train_augmented))
X_train_augmented = X_train_augmented[shuffle_idx]
y_train_augmented = y_train_augmented[shuffle_idx]

# + id="4-PfPjOco6Fr" colab_type="code" colab={}
knn_clf = KNeighborsClassifier(**grid_search.best_params_)

# + id="V1qkNAfeo6Fs" colab_type="code" colab={} outputId="cbbef0d2-ce9e-4b00-9989-90bb76a59926"
knn_clf.fit(X_train_augmented, y_train_augmented)

# + id="7YtQWkcRo6Fv" colab_type="code" colab={} outputId="c561aaeb-93d3-46dd-96f2-b32890b5d0a5"
y_pred = knn_clf.predict(X_test)
accuracy_score(y_test, y_pred)

# + [markdown] id="CdBSEG8Po6Fw" colab_type="text"
# By simply augmenting the data, we got a 0.5% accuracy boost. :)

# + [markdown] id="73P4u9UGo6Fx" colab_type="text"
# ## 3. Tackle the Titanic dataset

# + [markdown] id="43SgRjXIo6Fx" colab_type="text"
# The goal is to predict whether or not a passenger survived based on attributes such as their age, sex, passenger class, where they embarked and so on.

# + [markdown] id="YcMCG_GRo6Fx" colab_type="text"
# First, login to [Kaggle](https://www.kaggle.com/) and go to the [Titanic challenge](https://www.kaggle.com/c/titanic) to download `train.csv` and `test.csv`. Save them to the `datasets/titanic` directory.

# + [markdown] id="qMmS2OJqo6Fx" colab_type="text"
# Next, let's load the data:

# + id="BHf8YHp8o6Fy" colab_type="code" colab={}
import os

TITANIC_PATH = os.path.join("datasets", "titanic")

# + id="e3gajNsOo6Fz" colab_type="code" colab={}
import pandas as pd

def load_titanic_data(filename, titanic_path=TITANIC_PATH):
    csv_path = os.path.join(titanic_path, filename)
    return pd.read_csv(csv_path)


# + id="rc-MHTnPo6F1" colab_type="code" colab={}
train_data = load_titanic_data("train.csv")
test_data = load_titanic_data("test.csv")

# + [markdown] id="Rb975q_Po6F2" colab_type="text"
# The data is already split into a training set and a test set. However, the test data does *not* contain the labels: your goal is to train the best model you can using the training data, then make your predictions on the test data and upload them to Kaggle to see your final score.

# + [markdown] id="QJ4n83rgo6F2" colab_type="text"
# Let's take a peek at the top few rows of the training set:

# + id="TGTpKGhDo6F3" colab_type="code" colab={} outputId="c41d8c13-fc78-4633-f5a8-dd21a3c10ce2"
train_data.head()

# + [markdown] id="YPTG-M68o6F4" colab_type="text"
# The attributes have the following meaning:
# * **Survived**: that's the target, 0 means the passenger did not survive, while 1 means he/she survived.
# * **Pclass**: passenger class.
# * **Name**, **Sex**, **Age**: self-explanatory
# * **SibSp**: how many siblings & spouses of the passenger aboard the Titanic.
# * **Parch**: how many children & parents of the passenger aboard the Titanic.
# * **Ticket**: ticket id
# * **Fare**: price paid (in pounds)
# * **Cabin**: passenger's cabin number
# * **Embarked**: where the passenger embarked the Titanic

# + [markdown] id="7nDyThzzo6F5" colab_type="text"
# Let's get more info to see how much data is missing:

# + id="gDfEOQTYo6F5" colab_type="code" colab={} outputId="93248834-ba21-4eba-d4f1-a3bdd45b32ad"
train_data.info()

# + [markdown] id="CJInA7Ozo6F7" colab_type="text"
# Okay, the **Age**, **Cabin** and **Embarked** attributes are sometimes null (less than 891 non-null), especially the **Cabin** (77% are null). We will ignore the **Cabin** for now and focus on the rest. The **Age** attribute has about 19% null values, so we will need to decide what to do with them. Replacing null values with the median age seems reasonable.

# + [markdown] id="tZex80SBo6F7" colab_type="text"
# The **Name** and **Ticket** attributes may have some value, but they will be a bit tricky to convert into useful numbers that a model can consume. So for now, we will ignore them.

# + [markdown] id="M-yD4RWAo6F7" colab_type="text"
# Let's take a look at the numerical attributes:

# + id="z70-s6amo6F9" colab_type="code" colab={} outputId="33b28208-cd58-42ab-92b5-f271a37f4555"
train_data.describe()

# + [markdown] id="9rFZp1G2o6GA" colab_type="text"
# * Yikes, only 38% **Survived**. :(  That's close enough to 40%, so accuracy will be a reasonable metric to evaluate our model.
# * The mean **Fare** was £32.20, which does not seem so expensive (but it was probably a lot of money back then).
# * The mean **Age** was less than 30 years old.

# + [markdown] id="b6QY97gbo6GB" colab_type="text"
# Let's check that the target is indeed 0 or 1:

# + id="YayMYYSpo6GB" colab_type="code" colab={} outputId="3f8233d2-03dc-47be-d0fa-baf4db293243"
train_data["Survived"].value_counts()

# + [markdown] id="mi8-Hf_3o6GE" colab_type="text"
# Now let's take a quick look at all the categorical attributes:

# + id="Rt4QJuVuo6GE" colab_type="code" colab={} outputId="dcb89487-125a-4233-d43c-4f3e53c4a8a1"
train_data["Pclass"].value_counts()

# + id="_Y4iyKpTo6GI" colab_type="code" colab={} outputId="f26b6d2c-324d-4406-e177-b912830539f2"
train_data["Sex"].value_counts()

# + id="iaNeK-dEo6GJ" colab_type="code" colab={} outputId="e122043b-df50-4d5e-dfa5-5b598d3a0877"
train_data["Embarked"].value_counts()

# + [markdown] id="7IOd-2Nxo6GK" colab_type="text"
# The Embarked attribute tells us where the passenger embarked: C=Cherbourg, Q=Queenstown, S=Southampton.

# + [markdown] id="P9D6GpBqo6GL" colab_type="text"
# **Note**: the code below uses a mix of `Pipeline`, `FeatureUnion` and a custom `DataFrameSelector` to preprocess some columns differently.  Since Scikit-Learn 0.20, it is preferable to use a `ColumnTransformer`, like in the previous chapter.

# + [markdown] id="5DgpsNGno6GL" colab_type="text"
# Now let's build our preprocessing pipelines. We will reuse the `DataframeSelector` we built in the previous chapter to select specific attributes from the `DataFrame`:

# + id="qMYTb2CEo6GL" colab_type="code" colab={}
from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names]


# + [markdown] id="u4U5Bxsao6GN" colab_type="text"
# Let's build the pipeline for the numerical attributes:

# + id="8YBHRtElo6GN" colab_type="code" colab={}
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

num_pipeline = Pipeline([
        ("select_numeric", DataFrameSelector(["Age", "SibSp", "Parch", "Fare"])),
        ("imputer", SimpleImputer(strategy="median")),
    ])

# + id="iSv5Rh9go6GO" colab_type="code" colab={} outputId="d6fdec1e-0b85-4ed4-a92e-8b888a1e06d9"
num_pipeline.fit_transform(train_data)


# + [markdown] id="R1bcRk6uo6GP" colab_type="text"
# We will also need an imputer for the string categorical columns (the regular `SimpleImputer` does not work on those):

# + id="rTiijB0ko6GP" colab_type="code" colab={}
# Inspired from stackoverflow.com/questions/25239958
class MostFrequentImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],
                                        index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.most_frequent_)


# + id="uCUD4YODo6GR" colab_type="code" colab={}
from sklearn.preprocessing import OneHotEncoder

# + [markdown] id="xqvCf4zSo6GS" colab_type="text"
# Now we can build the pipeline for the categorical attributes:

# + id="mBi3LUWQo6GS" colab_type="code" colab={}
cat_pipeline = Pipeline([
        ("select_cat", DataFrameSelector(["Pclass", "Sex", "Embarked"])),
        ("imputer", MostFrequentImputer()),
        ("cat_encoder", OneHotEncoder(sparse=False)),
    ])

# + id="s5CgUV-ko6GU" colab_type="code" colab={} outputId="47a18e2c-01ca-4fbf-8aee-9c4f72f4c3e3"
cat_pipeline.fit_transform(train_data)

# + [markdown] id="_QG03ScWo6GV" colab_type="text"
# Finally, let's join the numerical and categorical pipelines:

# + id="Ed6CxjL_o6GV" colab_type="code" colab={}
from sklearn.pipeline import FeatureUnion
preprocess_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])

# + [markdown] id="7MDlC2F1o6Ga" colab_type="text"
# Cool! Now we have a nice preprocessing pipeline that takes the raw data and outputs numerical input features that we can feed to any Machine Learning model we want.

# + id="kFvPYSnuo6Ga" colab_type="code" colab={} outputId="12b34b09-d4db-4aec-91ae-0c88affec4d5"
X_train = preprocess_pipeline.fit_transform(train_data)
X_train

# + [markdown] id="4U31fFfGo6Gb" colab_type="text"
# Let's not forget to get the labels:

# + id="fl2dy1SPo6Gb" colab_type="code" colab={}
y_train = train_data["Survived"]

# + [markdown] id="39rOYlY3o6Gd" colab_type="text"
# We are now ready to train a classifier. Let's start with an `SVC`:

# + id="_fMdw-SOo6Gd" colab_type="code" colab={} outputId="40ccbe36-5313-46ab-f7c5-377cdbf334f4"
from sklearn.svm import SVC

svm_clf = SVC(gamma="auto")
svm_clf.fit(X_train, y_train)

# + [markdown] id="nLwdIa-po6Ge" colab_type="text"
# Great, our model is trained, let's use it to make predictions on the test set:

# + id="kDoch5u_o6Ge" colab_type="code" colab={}
X_test = preprocess_pipeline.transform(test_data)
y_pred = svm_clf.predict(X_test)

# + [markdown] id="rpQOTOlKo6Gf" colab_type="text"
# And now we could just build a CSV file with these predictions (respecting the format excepted by Kaggle), then upload it and hope for the best. But wait! We can do better than hope. Why don't we use cross-validation to have an idea of how good our model is?

# + id="-4WbIhzeo6Gg" colab_type="code" colab={} outputId="36dfa52a-904e-4f05-8c79-45acd48ebec7"
from sklearn.model_selection import cross_val_score

svm_scores = cross_val_score(svm_clf, X_train, y_train, cv=10)
svm_scores.mean()

# + [markdown] id="kcG2z3aao6Gh" colab_type="text"
# Okay, over 73% accuracy, clearly better than random chance, but it's not a great score. Looking at the [leaderboard](https://www.kaggle.com/c/titanic/leaderboard) for the Titanic competition on Kaggle, you can see that you need to reach above 80% accuracy to be within the top 10% Kagglers. Some reached 100%, but since you can easily find the [list of victims](https://www.encyclopedia-titanica.org/titanic-victims/) of the Titanic, it seems likely that there was little Machine Learning involved in their performance! ;-) So let's try to build a model that reaches 80% accuracy.

# + [markdown] id="8aIymzCqo6Gh" colab_type="text"
# Let's try a `RandomForestClassifier`:

# + id="HC6-AsPOo6Gh" colab_type="code" colab={} outputId="c2f96554-fb6b-432f-e3f0-9b0b6e54ea97"
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
forest_scores = cross_val_score(forest_clf, X_train, y_train, cv=10)
forest_scores.mean()

# + [markdown] id="hMyKnLY0o6Gj" colab_type="text"
# That's much better!

# + [markdown] id="5nfFO7fao6Gk" colab_type="text"
# Instead of just looking at the mean accuracy across the 10 cross-validation folds, let's plot all 10 scores for each model, along with a box plot highlighting the lower and upper quartiles, and "whiskers" showing the extent of the scores (thanks to Nevin Yilmaz for suggesting this visualization). Note that the `boxplot()` function detects outliers (called "fliers") and does not include them within the whiskers. Specifically, if the lower quartile is $Q_1$ and the upper quartile is $Q_3$, then the interquartile range $IQR = Q_3 - Q_1$ (this is the box's height), and any score lower than $Q_1 - 1.5 \times IQR$ is a flier, and so is any score greater than $Q3 + 1.5 \times IQR$.

# + id="A48NMXbGo6Gl" colab_type="code" colab={} outputId="257a4244-be0b-47d8-ff99-3b8671aff5a4"
plt.figure(figsize=(8, 4))
plt.plot([1]*10, svm_scores, ".")
plt.plot([2]*10, forest_scores, ".")
plt.boxplot([svm_scores, forest_scores], labels=("SVM","Random Forest"))
plt.ylabel("Accuracy", fontsize=14)
plt.show()

# + [markdown] id="uVLebMDVo6Gn" colab_type="text"
# To improve this result further, you could:
# * Compare many more models and tune hyperparameters using cross validation and grid search,
# * Do more feature engineering, for example:
#   * replace **SibSp** and **Parch** with their sum,
#   * try to identify parts of names that correlate well with the **Survived** attribute (e.g. if the name contains "Countess", then survival seems more likely),
# * try to convert numerical attributes to categorical attributes: for example, different age groups had very different survival rates (see below), so it may help to create an age bucket category and use it instead of the age. Similarly, it may be useful to have a special category for people traveling alone since only 30% of them survived (see below).

# + id="kOHcoDt8o6Go" colab_type="code" colab={} outputId="97622314-f461-4164-b8de-f0f324197b0e"
train_data["AgeBucket"] = train_data["Age"] // 15 * 15
train_data[["AgeBucket", "Survived"]].groupby(['AgeBucket']).mean()

# + id="okUYrpSro6Gp" colab_type="code" colab={} outputId="3a41fe27-b877-467c-bef7-13ef26d958ed"
train_data["RelativesOnboard"] = train_data["SibSp"] + train_data["Parch"]
train_data[["RelativesOnboard", "Survived"]].groupby(['RelativesOnboard']).mean()

# + [markdown] id="qqXBEXvto6Gr" colab_type="text"
# ## 4. Spam classifier

# + [markdown] id="ruk8rrTdo6Gr" colab_type="text"
# First, let's fetch the data:

# + id="NpninCtEo6Gr" colab_type="code" colab={}
import os
import tarfile
import urllib

DOWNLOAD_ROOT = "http://spamassassin.apache.org/old/publiccorpus/"
HAM_URL = DOWNLOAD_ROOT + "20030228_easy_ham.tar.bz2"
SPAM_URL = DOWNLOAD_ROOT + "20030228_spam.tar.bz2"
SPAM_PATH = os.path.join("datasets", "spam")

def fetch_spam_data(spam_url=SPAM_URL, spam_path=SPAM_PATH):
    if not os.path.isdir(spam_path):
        os.makedirs(spam_path)
    for filename, url in (("ham.tar.bz2", HAM_URL), ("spam.tar.bz2", SPAM_URL)):
        path = os.path.join(spam_path, filename)
        if not os.path.isfile(path):
            urllib.request.urlretrieve(url, path)
        tar_bz2_file = tarfile.open(path)
        tar_bz2_file.extractall(path=SPAM_PATH)
        tar_bz2_file.close()


# + id="upNlGDsoo6Gu" colab_type="code" colab={}
fetch_spam_data()

# + [markdown] id="9xhkQPi1o6Gv" colab_type="text"
# Next, let's load all the emails:

# + id="o2G0jBjco6Gv" colab_type="code" colab={}
HAM_DIR = os.path.join(SPAM_PATH, "easy_ham")
SPAM_DIR = os.path.join(SPAM_PATH, "spam")
ham_filenames = [name for name in sorted(os.listdir(HAM_DIR)) if len(name) > 20]
spam_filenames = [name for name in sorted(os.listdir(SPAM_DIR)) if len(name) > 20]

# + id="10wrL-e3o6Gy" colab_type="code" colab={} outputId="6014e260-8f75-4f2a-c754-9fde2f3e676d"
len(ham_filenames)

# + id="Ik5PjXvqo6Gz" colab_type="code" colab={} outputId="b2fb7235-bf12-4311-9df7-75318320e79b"
len(spam_filenames)

# + [markdown] id="7Cfjwfsqo6G1" colab_type="text"
# We can use Python's `email` module to parse these emails (this handles headers, encoding, and so on):

# + id="AsN9YPpeo6G1" colab_type="code" colab={}
import email
import email.policy

def load_email(is_spam, filename, spam_path=SPAM_PATH):
    directory = "spam" if is_spam else "easy_ham"
    with open(os.path.join(spam_path, directory, filename), "rb") as f:
        return email.parser.BytesParser(policy=email.policy.default).parse(f)


# + id="8BnAevbco6G2" colab_type="code" colab={}
ham_emails = [load_email(is_spam=False, filename=name) for name in ham_filenames]
spam_emails = [load_email(is_spam=True, filename=name) for name in spam_filenames]

# + [markdown] id="eqRW9EVro6G3" colab_type="text"
# Let's look at one example of ham and one example of spam, to get a feel of what the data looks like:

# + id="3UcQ7Hkmo6G3" colab_type="code" colab={} outputId="8c0093bd-0511-49f3-a92c-0ae02cdb9c0d"
print(ham_emails[1].get_content().strip())

# + id="MTXk1il3o6G7" colab_type="code" colab={} outputId="be9e3f5c-7ac8-40af-dab6-45ae5d401d2b"
print(spam_emails[6].get_content().strip())


# + [markdown] id="YRiV-5Cfo6G8" colab_type="text"
# Some emails are actually multipart, with images and attachments (which can have their own attachments). Let's look at the various types of structures we have:

# + id="XcAXGFE7o6G8" colab_type="code" colab={}
def get_email_structure(email):
    if isinstance(email, str):
        return email
    payload = email.get_payload()
    if isinstance(payload, list):
        return "multipart({})".format(", ".join([
            get_email_structure(sub_email)
            for sub_email in payload
        ]))
    else:
        return email.get_content_type()


# + id="_Mw2NEZ9o6G9" colab_type="code" colab={}
from collections import Counter

def structures_counter(emails):
    structures = Counter()
    for email in emails:
        structure = get_email_structure(email)
        structures[structure] += 1
    return structures


# + id="nKmKNaPNo6G_" colab_type="code" colab={} outputId="2e930e79-198c-4a68-a9a0-d55998152ad3"
structures_counter(ham_emails).most_common()

# + id="KazTJBvdo6HD" colab_type="code" colab={} outputId="f2e5c410-0089-4a3e-fa9d-3b1787084de4"
structures_counter(spam_emails).most_common()

# + [markdown] id="r_IsouNgo6HG" colab_type="text"
# It seems that the ham emails are more often plain text, while spam has quite a lot of HTML. Moreover, quite a few ham emails are signed using PGP, while no spam is. In short, it seems that the email structure is useful information to have.

# + [markdown] id="Qq_wDKcZo6HG" colab_type="text"
# Now let's take a look at the email headers:

# + id="M_eAStuSo6HH" colab_type="code" colab={} outputId="74bbe9cb-098c-4540-8f12-797d98dbbe3d"
for header, value in spam_emails[0].items():
    print(header,":",value)

# + [markdown] id="Ja1YBQVHo6HI" colab_type="text"
# There's probably a lot of useful information in there, such as the sender's email address (12a1mailbot1@web.de looks fishy), but we will just focus on the `Subject` header:

# + id="F69qBBqoo6HI" colab_type="code" colab={} outputId="08439689-c65e-45d7-87c0-d50c59708fd9"
spam_emails[0]["Subject"]

# + [markdown] id="ErFswQP1o6HM" colab_type="text"
# Okay, before we learn too much about the data, let's not forget to split it into a training set and a test set:

# + id="2bsv0dg0o6HM" colab_type="code" colab={}
import numpy as np
from sklearn.model_selection import train_test_split

X = np.array(ham_emails + spam_emails)
y = np.array([0] * len(ham_emails) + [1] * len(spam_emails))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# + [markdown] id="QJSuSnuKo6HP" colab_type="text"
# Okay, let's start writing the preprocessing functions. First, we will need a function to convert HTML to plain text. Arguably the best way to do this would be to use the great [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/) library, but I would like to avoid adding another dependency to this project, so let's hack a quick & dirty solution using regular expressions (at the risk of [un̨ho͞ly radiańcé destro҉ying all enli̍̈́̂̈́ghtenment](https://stackoverflow.com/a/1732454/38626)). The following function first drops the `<head>` section, then converts all `<a>` tags to the word HYPERLINK, then it gets rid of all HTML tags, leaving only the plain text. For readability, it also replaces multiple newlines with single newlines, and finally it unescapes html entities (such as `&gt;` or `&nbsp;`):

# + id="umZeFRKwo6HP" colab_type="code" colab={}
import re
from html import unescape

def html_to_plain_text(html):
    text = re.sub('<head.*?>.*?</head>', '', html, flags=re.M | re.S | re.I)
    text = re.sub('<a\s.*?>', ' HYPERLINK ', text, flags=re.M | re.S | re.I)
    text = re.sub('<.*?>', '', text, flags=re.M | re.S)
    text = re.sub(r'(\s*\n)+', '\n', text, flags=re.M | re.S)
    return unescape(text)


# + [markdown] id="zfgsYvhGo6HQ" colab_type="text"
# Let's see if it works. This is HTML spam:

# + id="n3xamorSo6HR" colab_type="code" colab={} outputId="b3003702-cec0-44d6-c69a-b7b536a51cb2"
html_spam_emails = [email for email in X_train[y_train==1]
                    if get_email_structure(email) == "text/html"]
sample_html_spam = html_spam_emails[7]
print(sample_html_spam.get_content().strip()[:1000], "...")

# + [markdown] id="vhVnivyto6HS" colab_type="text"
# And this is the resulting plain text:

# + id="7l8UEIe0o6HS" colab_type="code" colab={} outputId="e97e5a81-2790-4bf1-d3d3-42fc9c11a07a"
print(html_to_plain_text(sample_html_spam.get_content())[:1000], "...")


# + [markdown] id="wE-qJHP8o6HU" colab_type="text"
# Great! Now let's write a function that takes an email as input and returns its content as plain text, whatever its format is:

# + id="p1enz2gko6HU" colab_type="code" colab={}
def email_to_text(email):
    html = None
    for part in email.walk():
        ctype = part.get_content_type()
        if not ctype in ("text/plain", "text/html"):
            continue
        try:
            content = part.get_content()
        except: # in case of encoding issues
            content = str(part.get_payload())
        if ctype == "text/plain":
            return content
        else:
            html = content
    if html:
        return html_to_plain_text(html)


# + id="aAR1pbMjo6HW" colab_type="code" colab={} outputId="f919299e-c096-4e33-905f-72cf535bbd84"
print(email_to_text(sample_html_spam)[:100], "...")

# + [markdown] id="ROCDKcSzo6HX" colab_type="text"
# Let's throw in some stemming! For this to work, you need to install the Natural Language Toolkit ([NLTK](http://www.nltk.org/)). It's as simple as running the following command (don't forget to activate your virtualenv first; if you don't have one, you will likely need administrator rights, or use the `--user` option):
#
# `$ pip3 install nltk`

# + id="KqDiEii1o6HX" colab_type="code" colab={} outputId="d98e355f-baee-4194-eb24-179f6796c201"
try:
    import nltk

    stemmer = nltk.PorterStemmer()
    for word in ("Computations", "Computation", "Computing", "Computed", "Compute", "Compulsive"):
        print(word, "=>", stemmer.stem(word))
except ImportError:
    print("Error: stemming requires the NLTK module.")
    stemmer = None

# + [markdown] id="3rTUG9_Bo6HY" colab_type="text"
# We will also need a way to replace URLs with the word "URL". For this, we could use hard core [regular expressions](https://mathiasbynens.be/demo/url-regex) but we will just use the [urlextract](https://github.com/lipoja/URLExtract) library. You can install it with the following command (don't forget to activate your virtualenv first; if you don't have one, you will likely need administrator rights, or use the `--user` option):
#
# `$ pip3 install urlextract`

# + id="EtuchK5Oo6HY" colab_type="code" colab={}
# if running this notebook on Colab, we just pip install urlextract
try:
    import google.colab
    # !pip install -q -U urlextract
except ImportError:
    pass # not running on Colab

# + id="b3yXi_uso6HZ" colab_type="code" colab={} outputId="9809fae6-5ba4-4d9f-e4bd-6a3b2858713a"
try:
    import urlextract # may require an Internet connection to download root domain names
    
    url_extractor = urlextract.URLExtract()
    print(url_extractor.find_urls("Will it detect github.com and https://youtu.be/7Pq-S557XQU?t=3m32s"))
except ImportError:
    print("Error: replacing URLs requires the urlextract module.")
    url_extractor = None

# + [markdown] id="wK-xOT8go6Hb" colab_type="text"
# We are ready to put all this together into a transformer that we will use to convert emails to word counters. Note that we split sentences into words using Python's `split()` method, which uses whitespaces for word boundaries. This works for many written languages, but not all. For example, Chinese and Japanese scripts generally don't use spaces between words, and Vietnamese often uses spaces even between syllables. It's okay in this exercise, because the dataset is (mostly) in English.

# + id="4IhymB6ho6Hb" colab_type="code" colab={}
from sklearn.base import BaseEstimator, TransformerMixin

class EmailToWordCounterTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, strip_headers=True, lower_case=True, remove_punctuation=True,
                 replace_urls=True, replace_numbers=True, stemming=True):
        self.strip_headers = strip_headers
        self.lower_case = lower_case
        self.remove_punctuation = remove_punctuation
        self.replace_urls = replace_urls
        self.replace_numbers = replace_numbers
        self.stemming = stemming
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X_transformed = []
        for email in X:
            text = email_to_text(email) or ""
            if self.lower_case:
                text = text.lower()
            if self.replace_urls and url_extractor is not None:
                urls = list(set(url_extractor.find_urls(text)))
                urls.sort(key=lambda url: len(url), reverse=True)
                for url in urls:
                    text = text.replace(url, " URL ")
            if self.replace_numbers:
                text = re.sub(r'\d+(?:\.\d*(?:[eE]\d+))?', 'NUMBER', text)
            if self.remove_punctuation:
                text = re.sub(r'\W+', ' ', text, flags=re.M)
            word_counts = Counter(text.split())
            if self.stemming and stemmer is not None:
                stemmed_word_counts = Counter()
                for word, count in word_counts.items():
                    stemmed_word = stemmer.stem(word)
                    stemmed_word_counts[stemmed_word] += count
                word_counts = stemmed_word_counts
            X_transformed.append(word_counts)
        return np.array(X_transformed)


# + [markdown] id="lWXPaRPRo6Hc" colab_type="text"
# Let's try this transformer on a few emails:

# + id="DdbRTh7Do6Hc" colab_type="code" colab={} outputId="b5837fed-9310-4736-ca85-2ae2e348139b"
X_few = X_train[:3]
X_few_wordcounts = EmailToWordCounterTransformer().fit_transform(X_few)
X_few_wordcounts

# + [markdown] id="_djdQRc3o6He" colab_type="text"
# This looks about right!

# + [markdown] id="yP9yi1Zro6He" colab_type="text"
# Now we have the word counts, and we need to convert them to vectors. For this, we will build another transformer whose `fit()` method will build the vocabulary (an ordered list of the most common words) and whose `transform()` method will use the vocabulary to convert word counts to vectors. The output is a sparse matrix.

# + id="C9qDHR_xo6He" colab_type="code" colab={}
from scipy.sparse import csr_matrix

class WordCounterToVectorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, vocabulary_size=1000):
        self.vocabulary_size = vocabulary_size
    def fit(self, X, y=None):
        total_count = Counter()
        for word_count in X:
            for word, count in word_count.items():
                total_count[word] += min(count, 10)
        most_common = total_count.most_common()[:self.vocabulary_size]
        self.most_common_ = most_common
        self.vocabulary_ = {word: index + 1 for index, (word, count) in enumerate(most_common)}
        return self
    def transform(self, X, y=None):
        rows = []
        cols = []
        data = []
        for row, word_count in enumerate(X):
            for word, count in word_count.items():
                rows.append(row)
                cols.append(self.vocabulary_.get(word, 0))
                data.append(count)
        return csr_matrix((data, (rows, cols)), shape=(len(X), self.vocabulary_size + 1))


# + id="Zv01wTYWo6Hg" colab_type="code" colab={} outputId="18cef946-ece2-4a5b-f869-0cc846013a6c"
vocab_transformer = WordCounterToVectorTransformer(vocabulary_size=10)
X_few_vectors = vocab_transformer.fit_transform(X_few_wordcounts)
X_few_vectors

# + id="XlN0apsoo6Hh" colab_type="code" colab={} outputId="735efd6f-70a0-4c13-aaed-95e1c476be57"
X_few_vectors.toarray()

# + [markdown] id="ANurAttTo6Hi" colab_type="text"
# What does this matrix mean? Well, the 99 in the second row, first column, means that the second email contains 99 words that are not part of the vocabulary. The 11 next to it means that the first word in the vocabulary is present 11 times in this email. The 9 next to it means that the second word is present 9 times, and so on. You can look at the vocabulary to know which words we are talking about. The first word is "the", the second word is "of", etc.

# + id="M5FKF339o6Hi" colab_type="code" colab={} outputId="fc63ec2c-6923-46d7-a4a4-ece766e90664"
vocab_transformer.vocabulary_

# + [markdown] id="bA66U1RTo6Hj" colab_type="text"
# We are now ready to train our first spam classifier! Let's transform the whole dataset:

# + id="FSIMvApRo6Hk" colab_type="code" colab={}
from sklearn.pipeline import Pipeline

preprocess_pipeline = Pipeline([
    ("email_to_wordcount", EmailToWordCounterTransformer()),
    ("wordcount_to_vector", WordCounterToVectorTransformer()),
])

X_train_transformed = preprocess_pipeline.fit_transform(X_train)

# + [markdown] id="_hZfr7u6o6Hl" colab_type="text"
# **Note**: to be future-proof, we set `solver="lbfgs"` since this will be the default value in Scikit-Learn 0.22.

# + id="Zy9I_a9jo6Hl" colab_type="code" colab={} outputId="455f6618-6208-42e9-de99-d40d8bf0b64b"
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

log_clf = LogisticRegression(solver="lbfgs", random_state=42)
score = cross_val_score(log_clf, X_train_transformed, y_train, cv=3, verbose=3)
score.mean()

# + [markdown] id="PT_SEaGDo6Hn" colab_type="text"
# Over 98.7%, not bad for a first try! :) However, remember that we are using the "easy" dataset. You can try with the harder datasets, the results won't be so amazing. You would have to try multiple models, select the best ones and fine-tune them using cross-validation, and so on.
#
# But you get the picture, so let's stop now, and just print out the precision/recall we get on the test set:

# + id="yeytsljPo6Hn" colab_type="code" colab={} outputId="a6a5c820-fd64-4d77-b783-92bde2358b01"
from sklearn.metrics import precision_score, recall_score

X_test_transformed = preprocess_pipeline.transform(X_test)

log_clf = LogisticRegression(solver="lbfgs", random_state=42)
log_clf.fit(X_train_transformed, y_train)

y_pred = log_clf.predict(X_test_transformed)

print("Precision: {:.2f}%".format(100 * precision_score(y_test, y_pred)))
print("Recall: {:.2f}%".format(100 * recall_score(y_test, y_pred)))

# + id="Fld_rjKzo6Hp" colab_type="code" colab={}

