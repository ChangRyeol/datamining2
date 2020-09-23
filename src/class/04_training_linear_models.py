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

# + [markdown] id="nR3MECfU3T2d" colab_type="text"
# **Chapter 4 – Training Linear Models**

# + [markdown] id="Ahm0Y1GL3T2e" colab_type="text"
# _This notebook contains all the sample code and solutions to the exercises in chapter 4._

# + [markdown] id="GOvJQcrT3T2g" colab_type="text"
# <table align="left">
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/ageron/handson-ml2/blob/master/04_training_linear_models.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
#   </td>
# </table>

# + id="JRH7q7L9l-lI" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 179} executionInfo={"status": "ok", "timestamp": 1600674393949, "user_tz": -540, "elapsed": 5425, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="5fb818e3-0387-4ffa-e56d-83a9e91fc8e2"
# !pip install import_ipynb

# + id="7UGWv4rbj4-B" colab_type="code" colab={} executionInfo={"status": "ok", "timestamp": 1600674427126, "user_tz": -540, "elapsed": 1236, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}}
import import_ipynb

# + id="HIGlCVG1pAKa" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 208} executionInfo={"status": "error", "timestamp": 1600674665164, "user_tz": -540, "elapsed": 938, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="34e48dc7-0a6a-4f4e-faf0-c04eb1e12be4"
import os 
import sys 
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

# + id="4EDnilkol1ii" colab_type="code" colab={}
import datamining2.config.ibynb

# + id="6PtYtRnt3ytd" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 53} executionInfo={"status": "ok", "timestamp": 1600675063732, "user_tz": -540, "elapsed": 27789, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="6320a734-12ee-4b4d-d8e1-34bec19f1a97"
from google.colab import drive # import drive from google colab

ROOT = "/content/drive"     # default location for the drive
print(ROOT)                 # print content of ROOT (Optional)

drive.mount(ROOT)           # we mount the google drive at /content/drive

# + id="v6vIA6y83_lw" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 36} executionInfo={"status": "ok", "timestamp": 1600661750672, "user_tz": -540, "elapsed": 780, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="0afaa227-4ee7-45d9-9f1d-001ff19947aa"
# %cd 'drive/My Drive/Colab Notebooks/datamining2/class' 

# + id="E6sbl6Qb4AhX" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 431} executionInfo={"status": "ok", "timestamp": 1600675078221, "user_tz": -540, "elapsed": 5751, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="e60acded-981b-47a5-89dc-a6abd8947cf2"
pip install jupytext #jupytext 설치 

# + id="2E60tHvQ4EeZ" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 46} outputId="e7fd6b87-d069-4ff9-bb24-42518643dc02"
## Pair a notebook to a light script
# !jupytext --set-formats ipynb,py:light 04_training_linear_models.ipynb  


# + id="HrV7zTda5Umb" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 89} executionInfo={"status": "ok", "timestamp": 1600670804065, "user_tz": -540, "elapsed": 1529, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="db01c0e4-cfa8-400f-a0a6-f2ed7603b15c"
# Sync the two representations
# !jupytext --sync 04_training_linear_models.ipynb

# + [markdown] id="ih1CTPrJ3T2h" colab_type="text"
# # Setup

# + [markdown] id="Z0xlfPfc3T2i" colab_type="text"
# First, let's import a few common modules, ensure MatplotLib plots figures inline and prepare a function to save the figures. We also check that Python 3.5 or later is installed (although Python 2.x may work, it is deprecated so we strongly recommend you use Python 3 instead), as well as Scikit-Learn ≥0.20.

# + id="BjNlbNfW3T2j" colab_type="code" colab={}
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
CHAPTER_ID = "training_linear_models"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

#plot save 함수. 

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# Ignore useless warnings (see SciPy issue #5998)
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")

# + [markdown] id="BYGC3ogX3T2o" colab_type="text"
# # Linear regression using the Normal Equation

# + id="TlH5f1qH3T2p" colab_type="code" colab={}
import numpy as np

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# + id="Jd5BamrwBfHS" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 35} executionInfo={"status": "ok", "timestamp": 1600664243563, "user_tz": -540, "elapsed": 696, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="7258414a-e112-4adb-f530-0849a897b9a3"
X.shape  # X에 난수 추출하여 2 곱함

# + id="RvCWDuJHBmPU" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 107} executionInfo={"status": "ok", "timestamp": 1600664280708, "user_tz": -540, "elapsed": 665, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="c8085f96-ea64-4580-dfa7-4d45ca2063d4"
X[:5]

# + id="SmBQ3F-W3T2t" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 315} executionInfo={"status": "ok", "timestamp": 1600664292094, "user_tz": -540, "elapsed": 1421, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="8b7c1ce0-e23b-4544-a21e-2f7d8752084e"
#X와 y PLOT  
plt.plot(X, y, "b.")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.axis([0, 2, 0, 15])
save_fig("generated_data_plot")
plt.show()

# + id="Uy4FItMz3T2z" colab_type="code" colab={}
#파라미터 찾기 
X_b = np.c_[np.ones((100, 1)), X]  # add x0 = 1 to each instance
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

# + id="SCwf46Px3T23" colab_type="code" colab={} outputId="d2b1aac0-dbbd-43c8-a86b-b6df42de4d16"
theta_best

# + id="qaafXLAE3T29" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 53} executionInfo={"status": "ok", "timestamp": 1600664328282, "user_tz": -540, "elapsed": 742, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="75ec1b03-e122-4652-a7e1-41111b921ad8"
#prediction. 
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]  # add x0 = 1 to each instance
y_predict = X_new_b.dot(theta_best)
y_predict

# + id="_MPXEdx63T3E" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 268} executionInfo={"status": "ok", "timestamp": 1600664396097, "user_tz": -540, "elapsed": 730, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="05d16500-891c-4eb6-fa1c-66edd4d79022"
#plot 
plt.plot(X_new, y_predict, "r-")
plt.plot(X, y, "b.")
plt.axis([0, 2, 0, 15])
plt.show()

# + [markdown] id="I1AqoeP-3T3M" colab_type="text"
# The figure in the book actually corresponds to the following code, with a legend and axis labels:

# + id="BtnAIoij3T3O" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 315} executionInfo={"status": "ok", "timestamp": 1600664402341, "user_tz": -540, "elapsed": 1337, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="baf418e4-86c7-449d-d7f8-72c27aff6f39"
plt.plot(X_new, y_predict, "r-", linewidth=2, label="Predictions")
plt.plot(X, y, "b.")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.legend(loc="upper left", fontsize=14)
plt.axis([0, 2, 0, 15])
save_fig("linear_model_predictions_plot")
plt.show()

# + id="cqQSdQDK3T3X" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 35} executionInfo={"status": "ok", "timestamp": 1600664420232, "user_tz": -540, "elapsed": 850, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="91738efe-bf3a-4138-afa9-50b52ad64425"
#sklearn 으로 간단히 선형회귀 사용가능. 
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, y)
lin_reg.intercept_, lin_reg.coef_

# + id="tO1E-Okv3T3e" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 53} executionInfo={"status": "ok", "timestamp": 1600664496059, "user_tz": -540, "elapsed": 938, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="7e16ab3e-a3ef-4889-8673-9aa4e7605a9b"
lin_reg.predict(X_new)

# + [markdown] id="Fdu_yds53T3m" colab_type="text"
# The `LinearRegression` class is based on the `scipy.linalg.lstsq()` function (the name stands for "least squares"), which you could call directly:

# + id="I4unqVwK3T3n" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 53} executionInfo={"status": "ok", "timestamp": 1600664499481, "user_tz": -540, "elapsed": 785, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="3bbb31b7-7709-4593-abe5-951abb6ad4f3"
theta_best_svd, residuals, rank, s = np.linalg.lstsq(X_b, y, rcond=1e-6)
theta_best_svd

# + [markdown] id="8cGSUreL3T3t" colab_type="text"
# This function computes $\mathbf{X}^+\mathbf{y}$, where $\mathbf{X}^{+}$ is the _pseudoinverse_ of $\mathbf{X}$ (specifically the Moore-Penrose inverse). You can use `np.linalg.pinv()` to compute the pseudoinverse directly:

# + id="9l5ch8QF3T3u" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 53} executionInfo={"status": "ok", "timestamp": 1600664504398, "user_tz": -540, "elapsed": 627, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="d9f2f8e1-78d2-4f86-9823-ddd21c372272"
np.linalg.pinv(X_b).dot(y)

# + [markdown] id="__yWIxwa3T30" colab_type="text"
# # Linear regression using batch gradient descent

# + id="IxzwPGW33T30" colab_type="code" colab={}
#앞의 내용을 gradient방식으로 적용 

eta = 0.1  # learning rate
n_iterations = 1000
m = 100

theta = np.random.randn(2,1)  # random initialization

for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients

# + id="SYj4sAyF3T35" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 53} executionInfo={"status": "ok", "timestamp": 1600665468173, "user_tz": -540, "elapsed": 1198, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="968d359f-4b2a-48c5-c4b7-bf9d6818be25"
theta

# + id="jKKBghiY3T38" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 53} executionInfo={"status": "ok", "timestamp": 1600665475326, "user_tz": -540, "elapsed": 731, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="65737350-c569-43bb-86ba-d550c4f2e548"
X_new_b.dot(theta)

# + id="IL0IhYJv3T4A" colab_type="code" colab={}

theta_path_bgd = []

def plot_gradient_descent(theta, eta, theta_path=None):
    m = len(X_b)
    plt.plot(X, y, "b.")
    n_iterations = 1000
    for iteration in range(n_iterations):
        if iteration < 10:
            y_predict = X_new_b.dot(theta)
            style = "b-" if iteration > 0 else "r--"
            plt.plot(X_new, y_predict, style)
        gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
        theta = theta - eta * gradients
        if theta_path is not None:
            theta_path.append(theta) #theta path에 저장하여 learing rate에 따른 plot을 보기. 
    plt.xlabel("$x_1$", fontsize=18)
    plt.axis([0, 2, 0, 15])
    plt.title(r"$\eta = {}$".format(eta), fontsize=16)


# + id="dYV_5hLN3T4E" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 315} executionInfo={"status": "ok", "timestamp": 1600665523762, "user_tz": -540, "elapsed": 2094, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="ef528d53-42a9-4ed0-b28d-9cc79b73b685"
np.random.seed(42)
theta = np.random.randn(2,1)  # random initialization

plt.figure(figsize=(10,4))
plt.subplot(131); plot_gradient_descent(theta, eta=0.02)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.subplot(132); plot_gradient_descent(theta, eta=0.1, theta_path=theta_path_bgd)
plt.subplot(133); plot_gradient_descent(theta, eta=0.5)

save_fig("gradient_descent_plot")
plt.show()

# + [markdown] id="2Oi1pTYP3T4H" colab_type="text"
# # Stochastic Gradient Descent

# + id="uf-2O0rL3T4H" colab_type="code" colab={}
theta_path_sgd = []
m = len(X_b)
np.random.seed(42)

# + id="JRHvwNlr3T4K" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 315} executionInfo={"status": "ok", "timestamp": 1600665540086, "user_tz": -540, "elapsed": 1372, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="d3f2b208-cd39-4838-c577-d8d6bfb81851"
n_epochs = 50
t0, t1 = 5, 50  # learning schedule hyperparameters

def learning_schedule(t):
    return t0 / (t + t1)

theta = np.random.randn(2,1)  # random initialization

for epoch in range(n_epochs):
    for i in range(m):
        if epoch == 0 and i < 20:                    # not shown in the book
            y_predict = X_new_b.dot(theta)           # not shown
            style = "b-" if i > 0 else "r--"         # not shown
            plt.plot(X_new, y_predict, style)        # not shown
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch * m + i)
        theta = theta - eta * gradients
        theta_path_sgd.append(theta)                 # not shown

plt.plot(X, y, "b.")                                 # not shown
plt.xlabel("$x_1$", fontsize=18)                     # not shown
plt.ylabel("$y$", rotation=0, fontsize=18)           # not shown
plt.axis([0, 2, 0, 15])                              # not shown
save_fig("sgd_plot")                                 # not shown
plt.show()                                           # not shown

# + id="v_LHgSie3T4N" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 53} executionInfo={"status": "ok", "timestamp": 1600665554845, "user_tz": -540, "elapsed": 759, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="24d198ed-0813-4376-db44-358d4362b3d8"
theta

# + id="yY6SqK8v3T4R" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 125} executionInfo={"status": "ok", "timestamp": 1600665652710, "user_tz": -540, "elapsed": 680, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="cca2da94-000f-4a4a-f86e-1c27158053ad"
#위에서는 sgd를 직접구현했다면, sklearn을 통해서 간단히 구현할 수 있다. 
from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.1, random_state=42)
sgd_reg.fit(X, y.ravel())

# + id="LCDJ9EJi3T4T" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 35} executionInfo={"status": "ok", "timestamp": 1600665656373, "user_tz": -540, "elapsed": 1028, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="29e19118-9a0d-4d3b-de11-eecf35b5d0c4"
sgd_reg.intercept_, sgd_reg.coef_

# + [markdown] id="gGpMl7hb3T4Z" colab_type="text"
# # Mini-batch gradient descent

# + id="dMiZsVxa3T4Z" colab_type="code" colab={}
theta_path_mgd = []

n_iterations = 50
minibatch_size = 20  #minibatch size를 정한다. 

np.random.seed(42)
theta = np.random.randn(2,1)  # random initialization

t0, t1 = 200, 1000

#learning_rate을 학습이 진행될수록 줄어들도록 설정하는 함수. 즉 초반부에는 크게 학습하다 해에 가까워질수록 학습률을 낮춘다. 
def learning_schedule(t):
    return t0 / (t + t1)

t = 0
for epoch in range(n_iterations):
    shuffled_indices = np.random.permutation(m)
    X_b_shuffled = X_b[shuffled_indices]
    y_shuffled = y[shuffled_indices]
    for i in range(0, m, minibatch_size):
        t += 1
        xi = X_b_shuffled[i:i+minibatch_size] #minibatch 만큼. 
        yi = y_shuffled[i:i+minibatch_size]
        gradients = 2/minibatch_size * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(t)
        theta = theta - eta * gradients
        theta_path_mgd.append(theta)

# + id="qj43fMVR3T4c" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 53} executionInfo={"status": "ok", "timestamp": 1600665714668, "user_tz": -540, "elapsed": 880, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="f1f121d5-9b9d-433c-898c-6d9b290641a8"
theta

# + id="uspngCl23T4f" colab_type="code" colab={}
theta_path_bgd = np.array(theta_path_bgd)
theta_path_sgd = np.array(theta_path_sgd)
theta_path_mgd = np.array(theta_path_mgd)

# + id="N_bbQKAf3T4j" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 315} executionInfo={"status": "ok", "timestamp": 1600665988750, "user_tz": -540, "elapsed": 1638, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="ab0f5d79-c4d8-46e1-e210-301e66cd6fd5"
plt.figure(figsize=(7,4))
plt.plot(theta_path_sgd[:, 0], theta_path_sgd[:, 1], "r-s", linewidth=1, label="Stochastic")
plt.plot(theta_path_mgd[:, 0], theta_path_mgd[:, 1], "g-+", linewidth=2, label="Mini-batch")
plt.plot(theta_path_bgd[:, 0], theta_path_bgd[:, 1], "b-o", linewidth=3, label="Batch")
plt.legend(loc="upper left", fontsize=16)
plt.xlabel(r"$\theta_0$", fontsize=20)
plt.ylabel(r"$\theta_1$   ", fontsize=20, rotation=0)
plt.axis([2.5, 4.5, 2.3, 3.9])
save_fig("gradient_descent_paths_plot")
plt.show()

# + [markdown] id="g04QXLSq3T4l" colab_type="text"
# # Polynomial regression

# + id="lnvWkTf13T4l" colab_type="code" colab={}
import numpy as np
import numpy.random as rnd

np.random.seed(42)

# + id="lKGJ_Aez3T4o" colab_type="code" colab={}
#2차항 추가. 
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

# + id="o00apkph3T4s" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 315} executionInfo={"status": "ok", "timestamp": 1600667146289, "user_tz": -540, "elapsed": 1328, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="e72198b7-cf7e-47c5-add8-dd1e01009b94"
plt.plot(X, y, "b.")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.axis([-3, 3, 0, 10])
save_fig("quadratic_data_plot")
plt.show()

# + id="h57KDZJt3T4v" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 35} executionInfo={"status": "ok", "timestamp": 1600667186958, "user_tz": -540, "elapsed": 644, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="36b1e2b7-a325-412f-9e5e-2c4700b21149"
#sklearn으로 poly구현. 
from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=2, include_bias=False) #2차항
X_poly = poly_features.fit_transform(X)
X[0]

# + id="q2V-gace3T4y" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 35} executionInfo={"status": "ok", "timestamp": 1600667195987, "user_tz": -540, "elapsed": 828, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="715946c6-90e4-4d8e-8ccc-4aea1341aa2a"
X_poly[0]  #x^2 = 0.56664654

# + id="Jyc4bxDF3T41" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 35} executionInfo={"status": "ok", "timestamp": 1600667217592, "user_tz": -540, "elapsed": 925, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="5fd0d1fc-d8f3-416a-eb71-75db79ee7621"
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
lin_reg.intercept_, lin_reg.coef_

# + id="guZ_7oiC3T44" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 315} executionInfo={"status": "ok", "timestamp": 1600667301789, "user_tz": -540, "elapsed": 1240, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="e6d382a5-55dc-4564-bf37-4fe7e8f16715"
#perdiction을 plot
X_new=np.linspace(-3, 3, 100).reshape(100, 1) #-3에서 3사이에 100개의 점을 뽑음. 
X_new_poly = poly_features.transform(X_new) #poly feature로 변환 
y_new = lin_reg.predict(X_new_poly) 
plt.plot(X, y, "b.") 
plt.plot(X_new, y_new, "r-", linewidth=2, label="Predictions")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.legend(loc="upper left", fontsize=14)
plt.axis([-3, 3, 0, 10])
save_fig("quadratic_predictions_plot")
plt.show()

# + id="x67rRn3M3T46" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 315} executionInfo={"status": "ok", "timestamp": 1600667316868, "user_tz": -540, "elapsed": 1489, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="3e58040a-734a-4645-ee95-9a4eeb2a9a7e"
#300차로 오버피티의 단점을 시각화해서 보기
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

for style, width, degree in (("g-", 1, 300), ("b--", 2, 2), ("r-+", 2, 1)):
    polybig_features = PolynomialFeatures(degree=degree, include_bias=False)
    std_scaler = StandardScaler()
    lin_reg = LinearRegression()
    polynomial_regression = Pipeline([
            ("poly_features", polybig_features),
            ("std_scaler", std_scaler),
            ("lin_reg", lin_reg),
        ])
    polynomial_regression.fit(X, y)
    y_newbig = polynomial_regression.predict(X_new)
    plt.plot(X_new, y_newbig, style, label=str(degree), linewidth=width)

plt.plot(X, y, "b.", linewidth=3)
plt.legend(loc="upper left")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.axis([-3, 3, 0, 10])
save_fig("high_degree_polynomials_plot")
plt.show()

# + id="LMuuAcxh3T49" colab_type="code" colab={}
#각각의 mse는 어떻게될까 
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))

    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    plt.legend(loc="upper right", fontsize=14)   # not shown in the book
    plt.xlabel("Training set size", fontsize=14) # not shown
    plt.ylabel("RMSE", fontsize=14)              # not shown


# + id="8g0DFnhU3T5A" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 315} executionInfo={"status": "ok", "timestamp": 1600667357945, "user_tz": -540, "elapsed": 1123, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="5360e9b8-958c-426b-dfc1-75731ed154ef"
#선형회귀때 mse 
lin_reg = LinearRegression()
plot_learning_curves(lin_reg, X, y)
plt.axis([0, 80, 0, 3])                         # not shown in the book
save_fig("underfitting_learning_curves_plot")   # not shown
plt.show()                                      # not shown

# + id="MVWOXrcL3T5E" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 315} executionInfo={"status": "ok", "timestamp": 1600667421713, "user_tz": -540, "elapsed": 1389, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="e2a3a717-5f16-400a-b974-9d546003d56d"
#poly degree가 10일때 mse. 
from sklearn.pipeline import Pipeline

polynomial_regression = Pipeline([
        ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
        ("lin_reg", LinearRegression()),
    ])

plot_learning_curves(polynomial_regression, X, y)
plt.axis([0, 80, 0, 3])           # not shown
save_fig("learning_curves_plot")  # not shown
plt.show()                        # not shown

# + [markdown] id="kqC5TnrT3T5G" colab_type="text"
# # Regularized models

# + id="JD7PQWYG3T5H" colab_type="code" colab={}
np.random.seed(42)
m = 20
X = 3 * np.random.rand(m, 1)
y = 1 + 0.5 * X + np.random.randn(m, 1) / 1.5
X_new = np.linspace(0, 3, 100).reshape(100, 1)

# + id="lPIHUkSV3T5K" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 35} executionInfo={"status": "ok", "timestamp": 1600668808785, "user_tz": -540, "elapsed": 1182, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="ba0268e8-8385-4a7d-83eb-ebc8506b17c1"
from sklearn.linear_model import Ridge
ridge_reg = Ridge(alpha=1, solver="cholesky", random_state=42)
ridge_reg.fit(X, y)
ridge_reg.predict([[1.5]])

# + id="lAKLbYRP3T5O" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 35} executionInfo={"status": "ok", "timestamp": 1600668811598, "user_tz": -540, "elapsed": 946, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="6eb71fa8-c7d3-47a0-da74-ef2bedb0fad0"
ridge_reg = Ridge(alpha=1, solver="sag", random_state=42)
ridge_reg.fit(X, y)
ridge_reg.predict([[1.5]])

# + id="sgsyCOlt3T5R" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 315} executionInfo={"status": "ok", "timestamp": 1600668823094, "user_tz": -540, "elapsed": 1954, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="c1cb20b1-78bc-46ff-ee30-afc86c1f3d2a"
from sklearn.linear_model import Ridge

def plot_model(model_class, polynomial, alphas, **model_kargs):
    for alpha, style in zip(alphas, ("b-", "g--", "r:")):
        model = model_class(alpha, **model_kargs) if alpha > 0 else LinearRegression()
        if polynomial:
            model = Pipeline([
                    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
                    ("std_scaler", StandardScaler()),
                    ("regul_reg", model),
                ])
        model.fit(X, y)
        y_new_regul = model.predict(X_new)
        lw = 2 if alpha > 0 else 1
        plt.plot(X_new, y_new_regul, style, linewidth=lw, label=r"$\alpha = {}$".format(alpha))
    plt.plot(X, y, "b.", linewidth=3)
    plt.legend(loc="upper left", fontsize=15)
    plt.xlabel("$x_1$", fontsize=18)
    plt.axis([0, 3, 0, 4])

plt.figure(figsize=(8,4))
plt.subplot(121)
plot_model(Ridge, polynomial=False, alphas=(0, 10, 100), random_state=42)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.subplot(122)
plot_model(Ridge, polynomial=True, alphas=(0, 10**-5, 1), random_state=42)

save_fig("ridge_regression_plot")
plt.show()

# + [markdown] id="TrWc8KkU3T5W" colab_type="text"
# **Note**: to be future-proof, we set `max_iter=1000` and `tol=1e-3` because these will be the default values in Scikit-Learn 0.21.

# + id="MuPoZhIp3T5W" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 35} executionInfo={"status": "ok", "timestamp": 1600668887471, "user_tz": -540, "elapsed": 1115, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="c775db86-1ce3-4916-c612-8032c7d830d9"
sgd_reg = SGDRegressor(penalty="l2", max_iter=1000, tol=1e-3, random_state=42) #L1으로 봐꾸면 LASSO가됨 (?)
sgd_reg.fit(X, y.ravel())
sgd_reg.predict([[1.5]])

# + id="IdGbaika3T5Z" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 371} executionInfo={"status": "ok", "timestamp": 1600668941628, "user_tz": -540, "elapsed": 2296, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="92f72231-4b01-4caf-ca6a-eaf880bc9c29"
from sklearn.linear_model import Lasso

plt.figure(figsize=(8,4))
plt.subplot(121)
plot_model(Lasso, polynomial=False, alphas=(0, 0.1, 1), random_state=42)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.subplot(122)
plot_model(Lasso, polynomial=True, alphas=(0, 10**-7, 1), random_state=42)

save_fig("lasso_regression_plot")
plt.show()

# + id="wD-h8_zJ3T5e" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 36} executionInfo={"status": "ok", "timestamp": 1600668947664, "user_tz": -540, "elapsed": 979, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="6349ae70-599c-4ce8-e350-34c787e426cd"
from sklearn.linear_model import Lasso
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X, y)
lasso_reg.predict([[1.5]])

# + id="4RERUuw_3T5g" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 35} executionInfo={"status": "ok", "timestamp": 1600668969876, "user_tz": -540, "elapsed": 1086, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="fa906b35-fb7a-4730-c767-200ee96f735a"
#Elastic net은 L1과 L2모두 고려한 모형임. 
from sklearn.linear_model import ElasticNet
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
elastic_net.fit(X, y)
elastic_net.predict([[1.5]])

# + id="vUxWZbxt3T5j" colab_type="code" colab={}
#데이터를 만들어서 TRAIN과 val로 분리. 이 데이터로 ERALY STOPPING 설명. 
np.random.seed(42)
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 2 + X + 0.5 * X**2 + np.random.randn(m, 1)

X_train, X_val, y_train, y_val = train_test_split(X[:50], y[:50].ravel(), test_size=0.5, random_state=10)

# + [markdown] id="iiQrngIH3T5m" colab_type="text"
# Early stopping example:

# + id="9MPXYLN23T5m" colab_type="code" colab={}
from copy import deepcopy

poly_scaler = Pipeline([
        ("poly_features", PolynomialFeatures(degree=90, include_bias=False)),
        ("std_scaler", StandardScaler())
    ])

X_train_poly_scaled = poly_scaler.fit_transform(X_train)
X_val_poly_scaled = poly_scaler.transform(X_val)

sgd_reg = SGDRegressor(max_iter=1, tol=-np.infty, warm_start=True,
                       penalty=None, learning_rate="constant", eta0=0.0005, random_state=42)

minimum_val_error = float("inf")
best_epoch = None
best_model = None
for epoch in range(1000):
    sgd_reg.fit(X_train_poly_scaled, y_train)  # continues where it left off
    y_val_predict = sgd_reg.predict(X_val_poly_scaled)
    val_error = mean_squared_error(y_val, y_val_predict)
    if val_error < minimum_val_error:
        minimum_val_error = val_error
        best_epoch = epoch
        best_model = deepcopy(sgd_reg)

# + [markdown] id="us_pd6ao3T5p" colab_type="text"
# Create the graph:

# + id="IoKtJ_EJ3T5q" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 315} executionInfo={"status": "ok", "timestamp": 1600669294897, "user_tz": -540, "elapsed": 2101, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="fe4dd55e-4f51-42e3-b76a-2e0eab4be569"
sgd_reg = SGDRegressor(max_iter=1, tol=-np.infty, warm_start=True,
                       penalty=None, learning_rate="constant", eta0=0.0005, random_state=42)
#500번 돌린 후 best 모델 찾기
n_epochs = 500
train_errors, val_errors = [], []
for epoch in range(n_epochs):
    sgd_reg.fit(X_train_poly_scaled, y_train)
    y_train_predict = sgd_reg.predict(X_train_poly_scaled)
    y_val_predict = sgd_reg.predict(X_val_poly_scaled)
    train_errors.append(mean_squared_error(y_train, y_train_predict))
    val_errors.append(mean_squared_error(y_val, y_val_predict))

best_epoch = np.argmin(val_errors)
best_val_rmse = np.sqrt(val_errors[best_epoch])

plt.annotate('Best model',
             xy=(best_epoch, best_val_rmse),
             xytext=(best_epoch, best_val_rmse + 1),
             ha="center",
             arrowprops=dict(facecolor='black', shrink=0.05),
             fontsize=16,
            )

best_val_rmse -= 0.03  # just to make the graph look better
plt.plot([0, n_epochs], [best_val_rmse, best_val_rmse], "k:", linewidth=2)
plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="Validation set")
plt.plot(np.sqrt(train_errors), "r--", linewidth=2, label="Training set")
plt.legend(loc="upper right", fontsize=14)
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("RMSE", fontsize=14)
save_fig("early_stopping_plot")
plt.show()

# + id="pnXMqhzV3T5u" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 143} executionInfo={"status": "ok", "timestamp": 1600669299911, "user_tz": -540, "elapsed": 984, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="ec8b300a-afd7-46f2-bc2c-5048842a84f0"
best_epoch, best_model #best model

# + id="BljF3VMO3T51" colab_type="code" colab={}
# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

# + id="IEjJUvwH3T55" colab_type="code" colab={}
t1a, t1b, t2a, t2b = -1, 3, -1.5, 1.5

t1s = np.linspace(t1a, t1b, 500)
t2s = np.linspace(t2a, t2b, 500)
t1, t2 = np.meshgrid(t1s, t2s)
T = np.c_[t1.ravel(), t2.ravel()]
Xr = np.array([[1, 1], [1, -1], [1, 0.5]])
yr = 2 * Xr[:, :1] + 0.5 * Xr[:, 1:]

J = (1/len(Xr) * np.sum((T.dot(Xr.T) - yr.T)**2, axis=1)).reshape(t1.shape)

N1 = np.linalg.norm(T, ord=1, axis=1).reshape(t1.shape)
N2 = np.linalg.norm(T, ord=2, axis=1).reshape(t1.shape)

t_min_idx = np.unravel_index(np.argmin(J), J.shape)
t1_min, t2_min = t1[t_min_idx], t2[t_min_idx]

t_init = np.array([[0.25], [-1]])


# + id="MxsV2S0m3T59" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 603} executionInfo={"status": "ok", "timestamp": 1600669316908, "user_tz": -540, "elapsed": 3845, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="51b814bd-62fb-4f66-d96f-e9dbd997ab3d"
def bgd_path(theta, X, y, l1, l2, core = 1, eta = 0.05, n_iterations = 200):
    path = [theta]
    for iteration in range(n_iterations):
        gradients = core * 2/len(X) * X.T.dot(X.dot(theta) - y) + l1 * np.sign(theta) + l2 * theta
        theta = theta - eta * gradients
        path.append(theta)
    return np.array(path)

fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10.1, 8))
for i, N, l1, l2, title in ((0, N1, 2., 0, "Lasso"), (1, N2, 0,  2., "Ridge")):
    JR = J + l1 * N1 + l2 * 0.5 * N2**2
    
    tr_min_idx = np.unravel_index(np.argmin(JR), JR.shape)
    t1r_min, t2r_min = t1[tr_min_idx], t2[tr_min_idx]

    levelsJ=(np.exp(np.linspace(0, 1, 20)) - 1) * (np.max(J) - np.min(J)) + np.min(J)
    levelsJR=(np.exp(np.linspace(0, 1, 20)) - 1) * (np.max(JR) - np.min(JR)) + np.min(JR)
    levelsN=np.linspace(0, np.max(N), 10)
    
    path_J = bgd_path(t_init, Xr, yr, l1=0, l2=0)
    path_JR = bgd_path(t_init, Xr, yr, l1, l2)
    path_N = bgd_path(np.array([[2.0], [0.5]]), Xr, yr, np.sign(l1)/3, np.sign(l2), core=0)

    ax = axes[i, 0]
    ax.grid(True)
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    ax.contourf(t1, t2, N / 2., levels=levelsN)
    ax.plot(path_N[:, 0], path_N[:, 1], "y--")
    ax.plot(0, 0, "ys")
    ax.plot(t1_min, t2_min, "ys")
    ax.set_title(r"$\ell_{}$ penalty".format(i + 1), fontsize=16)
    ax.axis([t1a, t1b, t2a, t2b])
    if i == 1:
        ax.set_xlabel(r"$\theta_1$", fontsize=16)
    ax.set_ylabel(r"$\theta_2$", fontsize=16, rotation=0)

    ax = axes[i, 1]
    ax.grid(True)
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    ax.contourf(t1, t2, JR, levels=levelsJR, alpha=0.9)
    ax.plot(path_JR[:, 0], path_JR[:, 1], "w-o")
    ax.plot(path_N[:, 0], path_N[:, 1], "y--")
    ax.plot(0, 0, "ys")
    ax.plot(t1_min, t2_min, "ys")
    ax.plot(t1r_min, t2r_min, "rs")
    ax.set_title(title, fontsize=16)
    ax.axis([t1a, t1b, t2a, t2b])
    if i == 1:
        ax.set_xlabel(r"$\theta_1$", fontsize=16)

save_fig("lasso_vs_ridge_plot")
plt.show()
#L1 NORM은 파라미터를 0으로 설정되기도 해서 파마리터 셀렉션의 역할을 하기도 함. 

# + [markdown] id="GcPAfQII3T6C" colab_type="text"
# # Logistic regression

# + id="X_pQdxP83T6D" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 243} executionInfo={"status": "ok", "timestamp": 1600669821134, "user_tz": -540, "elapsed": 1209, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="f2270f61-1545-434e-f62b-e7119b077554"
t = np.linspace(-10, 10, 100)
sig = 1 / (1 + np.exp(-t))
plt.figure(figsize=(9, 3))
plt.plot([-10, 10], [0, 0], "k-")
plt.plot([-10, 10], [0.5, 0.5], "k:")
plt.plot([-10, 10], [1, 1], "k:")
plt.plot([0, 0], [-1.1, 1.1], "k-")
plt.plot(t, sig, "b-", linewidth=2, label=r"$\sigma(t) = \frac{1}{1 + e^{-t}}$")
plt.xlabel("t")
plt.legend(loc="upper left", fontsize=20)
plt.axis([-10, 10, -0.1, 1.1])
save_fig("logistic_function_plot")
plt.show()
#0에서 1값을 갖도록 모델링 

# + id="PUP5SFKL3T6I" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 35} executionInfo={"status": "ok", "timestamp": 1600669847482, "user_tz": -540, "elapsed": 860, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="30f8834d-b843-4628-b8b4-840eac94afb2"
from sklearn import datasets
iris = datasets.load_iris()
list(iris.keys())

# + id="Vn9yg-LA3T6K" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"status": "ok", "timestamp": 1600669857088, "user_tz": -540, "elapsed": 792, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="2b958457-b2df-48cf-a5d2-f87b82dd7122"
print(iris.DESCR)

# + id="6jC5BYn83T6L" colab_type="code" colab={}
X = iris["data"][:, 3:]  # petal width
y = (iris["target"] == 2).astype(np.int)  # 1 if Iris virginica, else 0

# + [markdown] id="R1to_Dg13T6O" colab_type="text"
# **Note**: To be future-proof we set `solver="lbfgs"` since this will be the default value in Scikit-Learn 0.22.

# + id="7rWHg08O3T6O" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 107} executionInfo={"status": "ok", "timestamp": 1600669960435, "user_tz": -540, "elapsed": 885, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="f91729fe-603f-4b73-ddb3-9552473ae9ad"
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(solver="lbfgs", random_state=42)
log_reg.fit(X, y)

# + id="RNQMf68R3T6R" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 286} executionInfo={"status": "ok", "timestamp": 1600669964542, "user_tz": -540, "elapsed": 837, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="30c2fb1f-fe99-45ad-e2ef-f0998f4bd8cd"
X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(X_new)

plt.plot(X_new, y_proba[:, 1], "g-", linewidth=2, label="Iris virginica")
plt.plot(X_new, y_proba[:, 0], "b--", linewidth=2, label="Not Iris virginica")

# + [markdown] id="iHl65O6P3T6T" colab_type="text"
# The figure in the book actually is actually a bit fancier:

# + id="Ou5E3CK23T6U" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 243} executionInfo={"status": "ok", "timestamp": 1600669976250, "user_tz": -540, "elapsed": 1647, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="18e8e651-4f3c-423e-b150-a42dc0ab600c"
X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(X_new)
decision_boundary = X_new[y_proba[:, 1] >= 0.5][0]

plt.figure(figsize=(8, 3))
plt.plot(X[y==0], y[y==0], "bs")
plt.plot(X[y==1], y[y==1], "g^")
plt.plot([decision_boundary, decision_boundary], [-1, 2], "k:", linewidth=2)
plt.plot(X_new, y_proba[:, 1], "g-", linewidth=2, label="Iris virginica")
plt.plot(X_new, y_proba[:, 0], "b--", linewidth=2, label="Not Iris virginica")
plt.text(decision_boundary+0.02, 0.15, "Decision  boundary", fontsize=14, color="k", ha="center")
plt.arrow(decision_boundary, 0.08, -0.3, 0, head_width=0.05, head_length=0.1, fc='b', ec='b')
plt.arrow(decision_boundary, 0.92, 0.3, 0, head_width=0.05, head_length=0.1, fc='g', ec='g')
plt.xlabel("Petal width (cm)", fontsize=14)
plt.ylabel("Probability", fontsize=14)
plt.legend(loc="center left", fontsize=14)
plt.axis([0, 3, -0.02, 1.02])
save_fig("logistic_regression_plot")
plt.show()

# + id="B_lN7SRt3T6W" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 35} executionInfo={"status": "ok", "timestamp": 1600669995724, "user_tz": -540, "elapsed": 1096, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="26d3e5f8-95f0-48dc-b470-fb0640a8383b"
decision_boundary

# + id="YswsX6-G3T6Y" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 35} executionInfo={"status": "ok", "timestamp": 1600670001096, "user_tz": -540, "elapsed": 1183, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="741cba52-0575-4cc0-f5fc-d5c7339a6cd5"
log_reg.predict([[1.7], [1.5]])

# + id="YCmWX52k3T6a" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 315} executionInfo={"status": "ok", "timestamp": 1600670006897, "user_tz": -540, "elapsed": 2033, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="60a10f65-d1da-47ab-b5a4-c81580a43f72"
from sklearn.linear_model import LogisticRegression

X = iris["data"][:, (2, 3)]  # petal length, petal width
y = (iris["target"] == 2).astype(np.int)

log_reg = LogisticRegression(solver="lbfgs", C=10**10, random_state=42)
log_reg.fit(X, y)

x0, x1 = np.meshgrid(
        np.linspace(2.9, 7, 500).reshape(-1, 1),
        np.linspace(0.8, 2.7, 200).reshape(-1, 1),
    )
X_new = np.c_[x0.ravel(), x1.ravel()]

y_proba = log_reg.predict_proba(X_new)

plt.figure(figsize=(10, 4))
plt.plot(X[y==0, 0], X[y==0, 1], "bs")
plt.plot(X[y==1, 0], X[y==1, 1], "g^")

zz = y_proba[:, 1].reshape(x0.shape)
contour = plt.contour(x0, x1, zz, cmap=plt.cm.brg)


left_right = np.array([2.9, 7])
boundary = -(log_reg.coef_[0][0] * left_right + log_reg.intercept_[0]) / log_reg.coef_[0][1]

plt.clabel(contour, inline=1, fontsize=12)
plt.plot(left_right, boundary, "k--", linewidth=3)
plt.text(3.5, 1.5, "Not Iris virginica", fontsize=14, color="b", ha="center")
plt.text(6.5, 2.3, "Iris virginica", fontsize=14, color="g", ha="center")
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.axis([2.9, 7, 0.8, 2.7])
save_fig("logistic_regression_contour_plot")
plt.show()

# + id="7mpqdAz-3T6c" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 107} executionInfo={"status": "ok", "timestamp": 1600670016430, "user_tz": -540, "elapsed": 660, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="16ea18fd-fa34-4d11-ebe4-ae356274e1df"
X = iris["data"][:, (2, 3)]  # petal length, petal width
y = iris["target"]

softmax_reg = LogisticRegression(multi_class="multinomial",solver="lbfgs", C=10, random_state=42)
softmax_reg.fit(X, y)

# + id="Z8wjt4B_3T6e" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 315} executionInfo={"status": "ok", "timestamp": 1600670052368, "user_tz": -540, "elapsed": 1843, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="e864da9b-51b5-4d0e-ab95-761b74001116"
x0, x1 = np.meshgrid(
        np.linspace(0, 8, 500).reshape(-1, 1),
        np.linspace(0, 3.5, 200).reshape(-1, 1),
    )
X_new = np.c_[x0.ravel(), x1.ravel()]


y_proba = softmax_reg.predict_proba(X_new)
y_predict = softmax_reg.predict(X_new)

zz1 = y_proba[:, 1].reshape(x0.shape)
zz = y_predict.reshape(x0.shape)

plt.figure(figsize=(10, 4))
plt.plot(X[y==2, 0], X[y==2, 1], "g^", label="Iris virginica")
plt.plot(X[y==1, 0], X[y==1, 1], "bs", label="Iris versicolor")
plt.plot(X[y==0, 0], X[y==0, 1], "yo", label="Iris setosa")

from matplotlib.colors import ListedColormap
custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])

plt.contourf(x0, x1, zz, cmap=custom_cmap)
contour = plt.contour(x0, x1, zz1, cmap=plt.cm.brg)
plt.clabel(contour, inline=1, fontsize=12)
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.legend(loc="center left", fontsize=14)
plt.axis([0, 7, 0, 3.5])
save_fig("softmax_regression_contour_plot")
plt.show()

# + id="0kcxLtka3T6f" colab_type="code" colab={} outputId="75b1acf3-ae7d-4fac-ada8-40eced29a468"
softmax_reg.predict([[5, 2]])

# + id="XZF-0QJc3T6h" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 35} executionInfo={"status": "ok", "timestamp": 1600670078914, "user_tz": -540, "elapsed": 834, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="bfba5ffe-5f61-4fe6-f74e-62dc1c166beb"
softmax_reg.predict_proba([[5, 2]])

# + [markdown] id="KDw1dkM03T6i" colab_type="text"
# # Exercise solutions

# + [markdown] id="eXOqMGzc3T6i" colab_type="text"
# ## 1. to 11.

# + [markdown] id="eY_M_bT53T6j" colab_type="text"
# See appendix A.

# + [markdown] id="dXlPTZuO3T6l" colab_type="text"
# ## 12. Batch Gradient Descent with early stopping for Softmax Regression
# (without using Scikit-Learn)

# + [markdown] id="VfV4xNwM3T6m" colab_type="text"
# Let's start by loading the data. We will just reuse the Iris dataset we loaded earlier.

# + id="J319EJuE3T6m" colab_type="code" colab={}
X = iris["data"][:, (2, 3)]  # petal length, petal width
y = iris["target"]

# + [markdown] id="bTESzmAn3T6o" colab_type="text"
# We need to add the bias term for every instance ($x_0 = 1$):

# + id="N4bVI5W73T6o" colab_type="code" colab={}
X_with_bias = np.c_[np.ones([len(X), 1]), X]

# + [markdown] id="A6d89FxC3T6q" colab_type="text"
# And let's set the random seed so the output of this exercise solution is reproducible:

# + id="pMds3rY13T6r" colab_type="code" colab={}
np.random.seed(2042)

# + [markdown] id="gT8cGD_F3T6s" colab_type="text"
# The easiest option to split the dataset into a training set, a validation set and a test set would be to use Scikit-Learn's `train_test_split()` function, but the point of this exercise is to try understand the algorithms by implementing them manually. So here is one possible implementation:

# + id="m9wxQk-l3T6s" colab_type="code" colab={}
test_ratio = 0.2
validation_ratio = 0.2
total_size = len(X_with_bias)

test_size = int(total_size * test_ratio)
validation_size = int(total_size * validation_ratio)
train_size = total_size - test_size - validation_size

rnd_indices = np.random.permutation(total_size)

X_train = X_with_bias[rnd_indices[:train_size]]
y_train = y[rnd_indices[:train_size]]
X_valid = X_with_bias[rnd_indices[train_size:-test_size]]
y_valid = y[rnd_indices[train_size:-test_size]]
X_test = X_with_bias[rnd_indices[-test_size:]]
y_test = y[rnd_indices[-test_size:]]


# + [markdown] id="J8XzsIxX3T6u" colab_type="text"
# The targets are currently class indices (0, 1 or 2), but we need target class probabilities to train the Softmax Regression model. Each instance will have target class probabilities equal to 0.0 for all classes except for the target class which will have a probability of 1.0 (in other words, the vector of class probabilities for ay given instance is a one-hot vector). Let's write a small function to convert the vector of class indices into a matrix containing a one-hot vector for each instance:

# + id="XyRZRk403T6u" colab_type="code" colab={}
def to_one_hot(y):
    n_classes = y.max() + 1
    m = len(y)
    Y_one_hot = np.zeros((m, n_classes))
    Y_one_hot[np.arange(m), y] = 1
    return Y_one_hot


# + [markdown] id="QAHPynx63T6x" colab_type="text"
# Let's test this function on the first 10 instances:

# + id="6Baylzme3T6x" colab_type="code" colab={} outputId="dc40881b-5fe2-4c54-f254-90df4c214778"
y_train[:10]

# + id="QYcp1bvu3T6z" colab_type="code" colab={} outputId="0dd0f937-082c-484f-ff4c-b713ad243944"
to_one_hot(y_train[:10])

# + [markdown] id="yJjKGFxX3T61" colab_type="text"
# Looks good, so let's create the target class probabilities matrix for the training set and the test set:

# + id="r4vvj56q3T61" colab_type="code" colab={}
Y_train_one_hot = to_one_hot(y_train)
Y_valid_one_hot = to_one_hot(y_valid)
Y_test_one_hot = to_one_hot(y_test)


# + [markdown] id="6P4_9Gbh3T64" colab_type="text"
# Now let's implement the Softmax function. Recall that it is defined by the following equation:
#
# $\sigma\left(\mathbf{s}(\mathbf{x})\right)_k = \dfrac{\exp\left(s_k(\mathbf{x})\right)}{\sum\limits_{j=1}^{K}{\exp\left(s_j(\mathbf{x})\right)}}$

# + id="WAX-jysD3T64" colab_type="code" colab={}
def softmax(logits):
    exps = np.exp(logits)
    exp_sums = np.sum(exps, axis=1, keepdims=True)
    return exps / exp_sums


# + [markdown] id="KaZKKIqk3T67" colab_type="text"
# We are almost ready to start training. Let's define the number of inputs and outputs:

# + id="0y8BHlBH3T67" colab_type="code" colab={}
n_inputs = X_train.shape[1] # == 3 (2 features plus the bias term)
n_outputs = len(np.unique(y_train))   # == 3 (3 iris classes)

# + [markdown] id="i7FIpk4-3T6_" colab_type="text"
# Now here comes the hardest part: training! Theoretically, it's simple: it's just a matter of translating the math equations into Python code. But in practice, it can be quite tricky: in particular, it's easy to mix up the order of the terms, or the indices. You can even end up with code that looks like it's working but is actually not computing exactly the right thing. When unsure, you should write down the shape of each term in the equation and make sure the corresponding terms in your code match closely. It can also help to evaluate each term independently and print them out. The good news it that you won't have to do this everyday, since all this is well implemented by Scikit-Learn, but it will help you understand what's going on under the hood.
#
# So the equations we will need are the cost function:
#
# $J(\mathbf{\Theta}) =
# - \dfrac{1}{m}\sum\limits_{i=1}^{m}\sum\limits_{k=1}^{K}{y_k^{(i)}\log\left(\hat{p}_k^{(i)}\right)}$
#
# And the equation for the gradients:
#
# $\nabla_{\mathbf{\theta}^{(k)}} \, J(\mathbf{\Theta}) = \dfrac{1}{m} \sum\limits_{i=1}^{m}{ \left ( \hat{p}^{(i)}_k - y_k^{(i)} \right ) \mathbf{x}^{(i)}}$
#
# Note that $\log\left(\hat{p}_k^{(i)}\right)$ may not be computable if $\hat{p}_k^{(i)} = 0$. So we will add a tiny value $\epsilon$ to $\log\left(\hat{p}_k^{(i)}\right)$ to avoid getting `nan` values.

# + id="ECJna8vT3T6_" colab_type="code" colab={} outputId="d8fa1dd6-c369-4be2-97ff-51c6c98f29ec"
eta = 0.01
n_iterations = 5001
m = len(X_train)
epsilon = 1e-7

Theta = np.random.randn(n_inputs, n_outputs)

for iteration in range(n_iterations):
    logits = X_train.dot(Theta)
    Y_proba = softmax(logits)
    loss = -np.mean(np.sum(Y_train_one_hot * np.log(Y_proba + epsilon), axis=1))
    error = Y_proba - Y_train_one_hot
    if iteration % 500 == 0:
        print(iteration, loss)
    gradients = 1/m * X_train.T.dot(error)
    Theta = Theta - eta * gradients

# + [markdown] id="Tlv3B8Rq3T7B" colab_type="text"
# And that's it! The Softmax model is trained. Let's look at the model parameters:

# + id="F1BiJddI3T7C" colab_type="code" colab={} outputId="8e4fa1d5-4ae1-456d-d381-3170bbc23edc"
Theta

# + [markdown] id="eYK_6bst3T7E" colab_type="text"
# Let's make predictions for the validation set and check the accuracy score:

# + id="dNABZXr53T7F" colab_type="code" colab={} outputId="759f0b2b-7900-4705-c19a-36b71feb0854"
logits = X_valid.dot(Theta)
Y_proba = softmax(logits)
y_predict = np.argmax(Y_proba, axis=1)

accuracy_score = np.mean(y_predict == y_valid)
accuracy_score

# + [markdown] id="nH5bd62m3T7G" colab_type="text"
# Well, this model looks pretty good. For the sake of the exercise, let's add a bit of $\ell_2$ regularization. The following training code is similar to the one above, but the loss now has an additional $\ell_2$ penalty, and the gradients have the proper additional term (note that we don't regularize the first element of `Theta` since this corresponds to the bias term). Also, let's try increasing the learning rate `eta`.

# + id="rHf-HHBV3T7H" colab_type="code" colab={} outputId="a9e3241e-d126-440d-a77c-f1a10a2a0684"
eta = 0.1
n_iterations = 5001
m = len(X_train)
epsilon = 1e-7
alpha = 0.1  # regularization hyperparameter

Theta = np.random.randn(n_inputs, n_outputs)

for iteration in range(n_iterations):
    logits = X_train.dot(Theta)
    Y_proba = softmax(logits)
    xentropy_loss = -np.mean(np.sum(Y_train_one_hot * np.log(Y_proba + epsilon), axis=1))
    l2_loss = 1/2 * np.sum(np.square(Theta[1:]))
    loss = xentropy_loss + alpha * l2_loss
    error = Y_proba - Y_train_one_hot
    if iteration % 500 == 0:
        print(iteration, loss)
    gradients = 1/m * X_train.T.dot(error) + np.r_[np.zeros([1, n_outputs]), alpha * Theta[1:]]
    Theta = Theta - eta * gradients

# + [markdown] id="ajp8bFMm3T7J" colab_type="text"
# Because of the additional $\ell_2$ penalty, the loss seems greater than earlier, but perhaps this model will perform better? Let's find out:

# + id="ov0QdVkI3T7J" colab_type="code" colab={} outputId="c8ea4483-5ec0-42e0-bec4-291aa190ae87"
logits = X_valid.dot(Theta)
Y_proba = softmax(logits)
y_predict = np.argmax(Y_proba, axis=1)

accuracy_score = np.mean(y_predict == y_valid)
accuracy_score

# + [markdown] id="MjAU9ugV3T7L" colab_type="text"
# Cool, perfect accuracy! We probably just got lucky with this validation set, but still, it's pleasant.

# + [markdown] id="PXLPOej63T7L" colab_type="text"
# Now let's add early stopping. For this we just need to measure the loss on the validation set at every iteration and stop when the error starts growing.

# + id="ml5MWZjZ3T7L" colab_type="code" colab={} outputId="e994c45a-92cb-4249-9d12-93b20c900f0c"
eta = 0.1 
n_iterations = 5001
m = len(X_train)
epsilon = 1e-7
alpha = 0.1  # regularization hyperparameter
best_loss = np.infty

Theta = np.random.randn(n_inputs, n_outputs)

for iteration in range(n_iterations):
    logits = X_train.dot(Theta)
    Y_proba = softmax(logits)
    xentropy_loss = -np.mean(np.sum(Y_train_one_hot * np.log(Y_proba + epsilon), axis=1))
    l2_loss = 1/2 * np.sum(np.square(Theta[1:]))
    loss = xentropy_loss + alpha * l2_loss
    error = Y_proba - Y_train_one_hot
    gradients = 1/m * X_train.T.dot(error) + np.r_[np.zeros([1, n_outputs]), alpha * Theta[1:]]
    Theta = Theta - eta * gradients

    logits = X_valid.dot(Theta)
    Y_proba = softmax(logits)
    xentropy_loss = -np.mean(np.sum(Y_valid_one_hot * np.log(Y_proba + epsilon), axis=1))
    l2_loss = 1/2 * np.sum(np.square(Theta[1:]))
    loss = xentropy_loss + alpha * l2_loss
    if iteration % 500 == 0:
        print(iteration, loss)
    if loss < best_loss:
        best_loss = loss
    else:
        print(iteration - 1, best_loss)
        print(iteration, loss, "early stopping!")
        break

# + id="au17w8Sv3T7N" colab_type="code" colab={} outputId="c08c6837-6f5b-4b5c-e344-2146d5443dd7"
logits = X_valid.dot(Theta)
Y_proba = softmax(logits)
y_predict = np.argmax(Y_proba, axis=1)

accuracy_score = np.mean(y_predict == y_valid)
accuracy_score

# + [markdown] id="ywUCer693T7P" colab_type="text"
# Still perfect, but faster.

# + [markdown] id="FF34ifoJ3T7P" colab_type="text"
# Now let's plot the model's predictions on the whole dataset:

# + id="P6aFD_gZ3T7Q" colab_type="code" colab={} outputId="c6f03a98-6a83-4d28-a5c5-50dccfa36481"
x0, x1 = np.meshgrid(
        np.linspace(0, 8, 500).reshape(-1, 1),
        np.linspace(0, 3.5, 200).reshape(-1, 1),
    )
X_new = np.c_[x0.ravel(), x1.ravel()]
X_new_with_bias = np.c_[np.ones([len(X_new), 1]), X_new]

logits = X_new_with_bias.dot(Theta)
Y_proba = softmax(logits)
y_predict = np.argmax(Y_proba, axis=1)

zz1 = Y_proba[:, 1].reshape(x0.shape)
zz = y_predict.reshape(x0.shape)

plt.figure(figsize=(10, 4))
plt.plot(X[y==2, 0], X[y==2, 1], "g^", label="Iris virginica")
plt.plot(X[y==1, 0], X[y==1, 1], "bs", label="Iris versicolor")
plt.plot(X[y==0, 0], X[y==0, 1], "yo", label="Iris setosa")

from matplotlib.colors import ListedColormap
custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])

plt.contourf(x0, x1, zz, cmap=custom_cmap)
contour = plt.contour(x0, x1, zz1, cmap=plt.cm.brg)
plt.clabel(contour, inline=1, fontsize=12)
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.legend(loc="upper left", fontsize=14)
plt.axis([0, 7, 0, 3.5])
plt.show()

# + [markdown] id="YdvDJ-Ih3T7S" colab_type="text"
# And now let's measure the final model's accuracy on the test set:

# + id="aKn2YL3k3T7S" colab_type="code" colab={} outputId="f15e32bc-69ba-4ea8-990b-195aceac0ca5"
logits = X_test.dot(Theta)
Y_proba = softmax(logits)
y_predict = np.argmax(Y_proba, axis=1)

accuracy_score = np.mean(y_predict == y_test)
accuracy_score

# + [markdown] id="LFA87Fsr3T7U" colab_type="text"
# Our perfect model turns out to have slight imperfections. This variability is likely due to the very small size of the dataset: depending on how you sample the training set, validation set and the test set, you can get quite different results. Try changing the random seed and running the code again a few times, you will see that the results will vary.

# + id="k1eyWyeR3T7U" colab_type="code" colab={}

