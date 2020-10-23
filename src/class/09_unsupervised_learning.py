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

# + [markdown] id="6fKhENKV-Xck"
# **Chapter 9 – Unsupervised Learning**
#
# _This notebook contains all the sample code in chapter 9._

# + [markdown] id="0t00WQYe-Xcl"
# <table align="left">
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/ageron/handson-ml2/blob/master/09_unsupervised_learning.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
#   </td>
# </table>

# + id="hnF_0FANZe3a" executionInfo={"status": "ok", "timestamp": 1603455581844, "user_tz": -540, "elapsed": 27740, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="5da42be8-6e70-47d1-d2c0-dab0075eb18f" colab={"base_uri": "https://localhost:8080/", "height": 55}
from google.colab import drive # import drive from google colab

ROOT = "/content/drive"     # default location for the drive
print(ROOT)                 # print content of ROOT (Optional)

drive.mount(ROOT)           #  we mount the google drive at /content/drive

# + id="ea2epz1GZhY2" executionInfo={"status": "ok", "timestamp": 1603455661266, "user_tz": -540, "elapsed": 1538, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="62f318da-5ab5-4ef3-a528-ff79beeefa63" colab={"base_uri": "https://localhost:8080/", "height": 36}
# %cd 'drive/My Drive/Colab Notebooks/datamining2/src/class/' 

# + id="UcNwduDEZ9iu" executionInfo={"status": "ok", "timestamp": 1603455680186, "user_tz": -540, "elapsed": 7842, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="3771354c-1566-4227-80b6-e8a828f1addd" colab={"base_uri": "https://localhost:8080/", "height": 467}
pip install jupytext #jupytext 설치 

# + id="PRJ8ITj8aAJN" executionInfo={"status": "ok", "timestamp": 1603455692679, "user_tz": -540, "elapsed": 3307, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="a4d7b97c-dc52-4978-d06b-282cd8a7e941" colab={"base_uri": "https://localhost:8080/", "height": 92}
## Pair a notebook to a light script
# !jupytext --set-formats ipynb,py:light 09_unsupervised_learning.ipynb  

# + id="A1R60G3PaEWe" executionInfo={"status": "ok", "timestamp": 1603455706364, "user_tz": -540, "elapsed": 2503, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="d2ed06ad-ae8b-483a-9156-c401e9f2f165" colab={"base_uri": "https://localhost:8080/", "height": 92}
# Sync the two representations
# !jupytext --sync 09_unsupervised_learning.ipynb

# + [markdown] id="p1ahZ84g-Xcn"
# # Setup

# + [markdown] id="eMGE3hJ0-Xco"
# First, let's import a few common modules, ensure MatplotLib plots figures inline and prepare a function to save the figures. We also check that Python 3.5 or later is installed (although Python 2.x may work, it is deprecated so we strongly recommend you use Python 3 instead), as well as Scikit-Learn ≥0.20.

# + id="_QbICwgS-Xcp" executionInfo={"status": "ok", "timestamp": 1603457227645, "user_tz": -540, "elapsed": 1638, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}}
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
CHAPTER_ID = "unsupervised_learning"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# Ignore useless warnings (see SciPy issue #5998)
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")

# + [markdown] id="rBgqqJlq-Xcu"
# # Clustering

# + [markdown] id="Xg-QV8Qx-Xcv"
# ## Introduction – Classification _vs_ Clustering

# + id="DHehewTu-Xcw" executionInfo={"status": "ok", "timestamp": 1603457234481, "user_tz": -540, "elapsed": 1059, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}}
from sklearn.datasets import load_iris

# + id="i-2YQNZv-Xc0" executionInfo={"status": "ok", "timestamp": 1603457234484, "user_tz": -540, "elapsed": 623, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="0f761404-13e9-4501-d0c4-500169b0b60f" colab={"base_uri": "https://localhost:8080/", "height": 36}
data = load_iris()
X = data.data
y = data.target
data.target_names

# + id="A0IRUJ9X-Xc6" executionInfo={"status": "ok", "timestamp": 1603457243589, "user_tz": -540, "elapsed": 1810, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="fae5ebb2-fce6-4eb2-d976-711cd7108f10" colab={"base_uri": "https://localhost:8080/", "height": 280}
plt.figure(figsize=(9, 3.5))

plt.subplot(121)
plt.plot(X[y==0, 2], X[y==0, 3], "yo", label="Iris setosa")
plt.plot(X[y==1, 2], X[y==1, 3], "bs", label="Iris versicolor")
plt.plot(X[y==2, 2], X[y==2, 3], "g^", label="Iris virginica")
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.legend(fontsize=12)

plt.subplot(122)
plt.scatter(X[:, 2], X[:, 3], c="k", marker=".")
plt.xlabel("Petal length", fontsize=14)
plt.tick_params(labelleft=False)

save_fig("classification_vs_clustering_plot")
plt.show()

# + [markdown] id="zmHUxk5M-Xc-"
# A Gaussian mixture model (explained below) can actually separate these clusters pretty well (using all 4 features: petal length & width, and sepal length & width).

# + id="TIsvIv6Z-Xc-" executionInfo={"status": "ok", "timestamp": 1603457260131, "user_tz": -540, "elapsed": 1518, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}}
####나중에 배울거라 지금은 그냥 넘어가자 


from sklearn.mixture import GaussianMixture

# + id="Eq35HVWP-XdC" executionInfo={"status": "ok", "timestamp": 1603457260132, "user_tz": -540, "elapsed": 1048, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}}
y_pred = GaussianMixture(n_components=3, random_state=42).fit(X).predict(X)
mapping = np.array([2, 0, 1])
y_pred = np.array([mapping[cluster_id] for cluster_id in y_pred])

# + id="pQSzpkFh-XdH" executionInfo={"status": "ok", "timestamp": 1603457261385, "user_tz": -540, "elapsed": 1970, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="7d4a4778-ee63-45f7-9f2f-01241f9a6210" colab={"base_uri": "https://localhost:8080/", "height": 286}
plt.plot(X[y_pred==0, 2], X[y_pred==0, 3], "yo", label="Cluster 1")
plt.plot(X[y_pred==1, 2], X[y_pred==1, 3], "bs", label="Cluster 2")
plt.plot(X[y_pred==2, 2], X[y_pred==2, 3], "g^", label="Cluster 3")
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.legend(loc="upper left", fontsize=12)
plt.show()

# + id="RWZ9DCTT-XdL" executionInfo={"status": "ok", "timestamp": 1603457261389, "user_tz": -540, "elapsed": 1746, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="00e59d97-b2dd-4e71-9669-0f18078ee426" colab={"base_uri": "https://localhost:8080/", "height": 36}
np.sum(y_pred==y)

# + id="zYhEQsRo-XdO" executionInfo={"status": "ok", "timestamp": 1603457261391, "user_tz": -540, "elapsed": 1580, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="245c22ab-de46-426d-98ec-aac030f50b5d" colab={"base_uri": "https://localhost:8080/", "height": 36}
np.sum(y_pred==y) / len(y_pred)

# + [markdown] id="2n--zFC2-XdR"
# ## K-Means

# + [markdown] id="HCaLMWbX-XdS"
# Let's start by generating some blobs:

# + id="2Z2wjKSY-XdT" executionInfo={"status": "ok", "timestamp": 1603457524937, "user_tz": -540, "elapsed": 1117, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}}

from sklearn.datasets import make_blobs

# + id="pOb4c5Bi-XdX" executionInfo={"status": "ok", "timestamp": 1603457565559, "user_tz": -540, "elapsed": 1238, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}}
#5개의 센터를 만들어서 데이터를 만듦. 
blob_centers = np.array(
    [[ 0.2,  2.3],
     [-1.5 ,  2.3],
     [-2.8,  1.8],
     [-2.8,  2.8],
     [-2.8,  1.3]])
blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])

# + id="7FYWZGMs-Xdc" executionInfo={"status": "ok", "timestamp": 1603457566191, "user_tz": -540, "elapsed": 1813, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}}
X, y = make_blobs(n_samples=2000, centers=blob_centers,
                  cluster_std=blob_std, random_state=7)


# + [markdown] id="R6d4fR0D-Xdg"
# Now let's plot them:

# + id="9WP1cnAr-Xdg" executionInfo={"status": "ok", "timestamp": 1603457566193, "user_tz": -540, "elapsed": 1764, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}}
def plot_clusters(X, y=None):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=1)
    plt.xlabel("$x_1$", fontsize=14)
    plt.ylabel("$x_2$", fontsize=14, rotation=0)


# + id="g1rZDtHd-Xdl" executionInfo={"status": "ok", "timestamp": 1603457566812, "user_tz": -540, "elapsed": 2354, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="906e70f0-3bbb-4baa-9501-15d96b44489a" colab={"base_uri": "https://localhost:8080/", "height": 316}
plt.figure(figsize=(8, 4))
plot_clusters(X)
save_fig("blobs_plot")
plt.show()

# + [markdown] id="ukVT8Hb6-Xdo"
# ### Fit and Predict

# + [markdown] id="hy5UWaRg-Xdp"
# Let's train a K-Means clusterer on this dataset. It will try to find each blob's center and assign each instance to the closest blob:

# + id="Vk2_bgF--Xdq" executionInfo={"status": "ok", "timestamp": 1603457570994, "user_tz": -540, "elapsed": 968, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}}
from sklearn.cluster import KMeans

# + id="Tdx6Uf1p-Xdu" executionInfo={"status": "ok", "timestamp": 1603457752573, "user_tz": -540, "elapsed": 958, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}}
#k는 클러스터 갯수. 최초 포인트가 랜덤하게 결정되기 때문에 매번 결과가 달라진다. 따라서 random_state를 지정해주자. 
k = 5
kmeans = KMeans(n_clusters=k, random_state=42)  #n_init 파라미터가 크면 그래도 결과가 매번 달라지는 편차가 좀 줄어듬. 
#n_init이 10이면 10번을 클러스터링 해서 그 평균으로 최초 지점을 잡으라는 의미. 1로 지정하면 결과가 많이 달라짐. 
y_pred = kmeans.fit_predict(X)

# + [markdown] id="7LpaV5iC-Xdz"
# Each instance was assigned to one of the 5 clusters:

# + id="xbZttWYw-Xd0" executionInfo={"status": "ok", "timestamp": 1603457757421, "user_tz": -540, "elapsed": 903, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="004e439e-d26d-4677-b492-30a334990a37" colab={"base_uri": "https://localhost:8080/", "height": 35}
y_pred

# + id="5nKlaOgI-Xd3" executionInfo={"status": "ok", "timestamp": 1603457757872, "user_tz": -540, "elapsed": 562, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="d971e015-eac7-47a4-b8a8-b73c3f0ba570" colab={"base_uri": "https://localhost:8080/", "height": 36}
y_pred is kmeans.labels_

# + [markdown] id="30u_M7Ta-Xd-"
# And the following 5 _centroids_ (i.e., cluster centers) were estimated:

# + id="n__I_37Y-XeA" executionInfo={"status": "ok", "timestamp": 1603457759412, "user_tz": -540, "elapsed": 629, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="022b3291-4f09-4957-905c-ad33fc3451bb" colab={"base_uri": "https://localhost:8080/", "height": 111}
kmeans.cluster_centers_

# + [markdown] id="t6-J2bxz-XeF"
# Note that the `KMeans` instance preserves the labels of the instances it was trained on. Somewhat confusingly, in this context, the _label_ of an instance is the index of the cluster that instance gets assigned to:

# + id="eHe0XBWp-XeG" executionInfo={"status": "ok", "timestamp": 1603457760499, "user_tz": -540, "elapsed": 830, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="4d2976c8-5991-46d5-b96a-90e097a74bc9" colab={"base_uri": "https://localhost:8080/", "height": 36}
kmeans.labels_

# + [markdown] id="X7kkG564-XeK"
# Of course, we can predict the labels of new instances:

# + id="yvYz91cp-XeL" executionInfo={"status": "ok", "timestamp": 1603457762767, "user_tz": -540, "elapsed": 1465, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="386a2d49-3993-4953-a52b-6c67440730cc" colab={"base_uri": "https://localhost:8080/", "height": 35}
X_new = np.array([[0, 2], [3, 2], [-3, 3], [-3, 2.5]])
kmeans.predict(X_new)


# + [markdown] id="1WQRce2x-XeS"
# ### Decision Boundaries

# + [markdown] id="HagMbsue-XeT"
# Let's plot the model's decision boundaries. This gives us a _Voronoi diagram_:

# + id="SUF41fMz-XeT" executionInfo={"status": "ok", "timestamp": 1603457765191, "user_tz": -540, "elapsed": 1066, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}}
def plot_data(X):
    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)

def plot_centroids(centroids, weights=None, circle_color='w', cross_color='k'):
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='o', s=30, linewidths=8,
                color=circle_color, zorder=10, alpha=0.9)
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=50, linewidths=50,
                color=cross_color, zorder=11, alpha=1)

def plot_decision_boundaries(clusterer, X, resolution=1000, show_centroids=True,
                             show_xlabels=True, show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                cmap="Pastel2")
    plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                linewidths=1, colors='k')
    plot_data(X)
    if show_centroids:
        plot_centroids(clusterer.cluster_centers_)

    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)


# + id="uyX-rnPr-Xea" executionInfo={"status": "ok", "timestamp": 1603457766994, "user_tz": -540, "elapsed": 2425, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="23aabe78-a605-4927-e307-207d105d0c25" colab={"base_uri": "https://localhost:8080/", "height": 316}
plt.figure(figsize=(8, 4))
plot_decision_boundaries(kmeans, X)
save_fig("voronoi_plot")
plt.show()

# + [markdown] id="r02hwvrd-Xeh"
# Not bad! Some of the instances near the edges were probably assigned to the wrong cluster, but overall it looks pretty good.

# + [markdown] id="kUwZcQdE-Xei"
# ### Hard Clustering _vs_ Soft Clustering

# + [markdown] id="FWu98SRl-Xei"
# Rather than arbitrarily choosing the closest cluster for each instance, which is called _hard clustering_, it might be better measure the distance of each instance to all 5 centroids. This is what the `transform()` method does:

# + id="nKKfVAeN-Xej" executionInfo={"status": "ok", "timestamp": 1603458140012, "user_tz": -540, "elapsed": 1172, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="877a9170-62a1-4d4e-aff9-3e31c9363eb6" colab={"base_uri": "https://localhost:8080/", "height": 92}
kmeans.transform(X_new)

# + [markdown] id="Wbb1YFGj-Xen"
# You can verify that this is indeed the Euclidian distance between each instance and each centroid:

# + id="DLsueO22-Xeo" executionInfo={"status": "ok", "timestamp": 1603458142364, "user_tz": -540, "elapsed": 853, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="a82a677a-171d-4a96-e70b-37825b89e2d1" colab={"base_uri": "https://localhost:8080/", "height": 92}
np.linalg.norm(np.tile(X_new, (1, k)).reshape(-1, k, 2) - kmeans.cluster_centers_, axis=2)

# + [markdown] id="eu07Ll12-Xet"
# ### K-Means Algorithm

# + [markdown] id="FNjn3gLV-Xeu"
# The K-Means algorithm is one of the fastest clustering algorithms, but also one of the simplest:
# * First initialize $k$ centroids randomly: $k$ distinct instances are chosen randomly from the dataset and the centroids are placed at their locations.
# * Repeat until convergence (i.e., until the centroids stop moving):
#     * Assign each instance to the closest centroid.
#     * Update the centroids to be the mean of the instances that are assigned to them.

# + [markdown] id="4AeJF1Dp-Xew"
# The `KMeans` class applies an optimized algorithm by default. To get the original K-Means algorithm (for educational purposes only), you must set `init="random"`, `n_init=1`and `algorithm="full"`. These hyperparameters will be explained below.

# + [markdown] id="SXO-Z1vu-Xex"
# Let's run the K-Means algorithm for 1, 2 and 3 iterations, to see how the centroids move around:

# + id="vRlO5DUh-Xey" executionInfo={"status": "ok", "timestamp": 1603458149402, "user_tz": -540, "elapsed": 1492, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="90b21d24-0f89-485c-b074-c66f9d806146" colab={"base_uri": "https://localhost:8080/", "height": 71}
kmeans_iter1 = KMeans(n_clusters=5, init="random", n_init=1,
                     algorithm="full", max_iter=1, random_state=1)
kmeans_iter2 = KMeans(n_clusters=5, init="random", n_init=1,
                     algorithm="full", max_iter=2, random_state=1)
kmeans_iter3 = KMeans(n_clusters=5, init="random", n_init=1,
                     algorithm="full", max_iter=3, random_state=1)
kmeans_iter1.fit(X)
kmeans_iter2.fit(X)
kmeans_iter3.fit(X)

# + [markdown] id="3F75q5kM-Xe5"
# And let's plot this:

# + id="FaK5UKph-Xe6" executionInfo={"status": "ok", "timestamp": 1603458155946, "user_tz": -540, "elapsed": 6729, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="2905e1e3-e77b-49c4-db12-9dff1ccd3328" colab={"base_uri": "https://localhost:8080/", "height": 604}
plt.figure(figsize=(10, 8))

plt.subplot(321)
plot_data(X)
plot_centroids(kmeans_iter1.cluster_centers_, circle_color='r', cross_color='w')
plt.ylabel("$x_2$", fontsize=14, rotation=0)
plt.tick_params(labelbottom=False)
plt.title("Update the centroids (initially randomly)", fontsize=14)

plt.subplot(322)
plot_decision_boundaries(kmeans_iter1, X, show_xlabels=False, show_ylabels=False)
plt.title("Label the instances", fontsize=14)

plt.subplot(323)
plot_decision_boundaries(kmeans_iter1, X, show_centroids=False, show_xlabels=False)
plot_centroids(kmeans_iter2.cluster_centers_)

plt.subplot(324)
plot_decision_boundaries(kmeans_iter2, X, show_xlabels=False, show_ylabels=False)

plt.subplot(325)
plot_decision_boundaries(kmeans_iter2, X, show_centroids=False)
plot_centroids(kmeans_iter3.cluster_centers_)

plt.subplot(326)
plot_decision_boundaries(kmeans_iter3, X, show_ylabels=False)

save_fig("kmeans_algorithm_plot")
plt.show()


# + [markdown] id="jWkCwKF9-Xe_"
# ### K-Means Variability

# + [markdown] id="aObVdOla-XfA"
# In the original K-Means algorithm, the centroids are just initialized randomly, and the algorithm simply runs a single iteration to gradually improve the centroids, as we saw above.
#
# However, one major problem with this approach is that if you run K-Means multiple times (or with different random seeds), it can converge to very different solutions, as you can see below:

# + id="12LOxuKV-XfA"
def plot_clusterer_comparison(clusterer1, clusterer2, X, title1=None, title2=None):
    clusterer1.fit(X)
    clusterer2.fit(X)

    plt.figure(figsize=(10, 3.2))

    plt.subplot(121)
    plot_decision_boundaries(clusterer1, X)
    if title1:
        plt.title(title1, fontsize=14)

    plt.subplot(122)
    plot_decision_boundaries(clusterer2, X, show_ylabels=False)
    if title2:
        plt.title(title2, fontsize=14)


# + id="69j-AGwo-XfD" outputId="9d2a8884-3c00-400d-e888-a4453ea63c21"
kmeans_rnd_init1 = KMeans(n_clusters=5, init="random", n_init=1,
                         algorithm="full", random_state=11)
kmeans_rnd_init2 = KMeans(n_clusters=5, init="random", n_init=1,
                         algorithm="full", random_state=19)

plot_clusterer_comparison(kmeans_rnd_init1, kmeans_rnd_init2, X,
                          "Solution 1", "Solution 2 (with a different random init)")

save_fig("kmeans_variability_plot")
plt.show()

# + [markdown] id="wHoJFKm5-XfI"
# ### Inertia

# + [markdown] id="hLtSQAQo-XfJ"
# To select the best model, we will need a way to evaluate a K-Mean model's performance. Unfortunately, clustering is an unsupervised task, so we do not have the targets. But at least we can measure the distance between each instance and its centroid. This is the idea behind the _inertia_ metric:

# + id="Kr_1KSqB-XfJ" outputId="e52ac0c3-52f7-4750-b62f-95428d40eb50"
kmeans.inertia_

# + [markdown] id="qrCIhObE-XfO"
# As you can easily verify, inertia is the sum of the squared distances between each training instance and its closest centroid:

# + id="vap54Tv4-XfP" outputId="87fe312e-eaff-44f4-8a94-ff117f400004"
X_dist = kmeans.transform(X)
np.sum(X_dist[np.arange(len(X_dist)), kmeans.labels_]**2)

# + [markdown] id="oIEIn917-XfU"
# The `score()` method returns the negative inertia. Why negative? Well, it is because a predictor's `score()` method must always respect the "_great is better_" rule.

# + id="ADzmzCVJ-XfV" outputId="e520e64b-3f8f-4190-9af5-798ff11f2a95"
kmeans.score(X)

# + [markdown] id="9o-mZubK-Xfa"
# ### Multiple Initializations

# + [markdown] id="BLsQ4bKO-Xfb"
# So one approach to solve the variability issue is to simply run the K-Means algorithm multiple times with different random initializations, and select the solution that minimizes the inertia. For example, here are the inertias of the two "bad" models shown in the previous figure:

# + id="1Ic3pyfI-Xfc" outputId="0c9c9aaf-9400-444b-cc78-8553fb0f1e07"
kmeans_rnd_init1.inertia_

# + id="Cyam_ge1-Xfg" outputId="986ab02f-676a-4739-feba-ea88509863f6"
kmeans_rnd_init2.inertia_

# + [markdown] id="Q7KmvMa1-Xfm"
# As you can see, they have a higher inertia than the first "good" model we trained, which means they are probably worse.

# + [markdown] id="LQn5T4UR-Xfn"
# When you set the `n_init` hyperparameter, Scikit-Learn runs the original algorithm `n_init` times, and selects the solution that minimizes the inertia. By default, Scikit-Learn sets `n_init=10`.

# + id="PBbt0Mp8-Xfn" outputId="5f3fbd21-f95e-4d5d-bf2d-9cf4495628e0"
kmeans_rnd_10_inits = KMeans(n_clusters=5, init="random", n_init=10,
                              algorithm="full", random_state=11)
kmeans_rnd_10_inits.fit(X)

# + [markdown] id="xF6DYtjL-Xft"
# As you can see, we end up with the initial model, which is certainly the optimal K-Means solution (at least in terms of inertia, and assuming $k=5$).

# + id="gu55YBCJ-Xfu" outputId="14b4301a-fdc0-49b4-de62-794687035fa6"
plt.figure(figsize=(8, 4))
plot_decision_boundaries(kmeans_rnd_10_inits, X)
plt.show()

# + [markdown] id="Xh3PbRft-Xf0"
# ### K-Means++

# + [markdown] id="SIJ7ToHG-Xf0"
# Instead of initializing the centroids entirely randomly, it is preferable to initialize them using the following algorithm, proposed in a [2006 paper](https://goo.gl/eNUPw6) by David Arthur and Sergei Vassilvitskii:
# * Take one centroid $c_1$, chosen uniformly at random from the dataset.
# * Take a new center $c_i$, choosing an instance $\mathbf{x}_i$ with probability: $D(\mathbf{x}_i)^2$ / $\sum\limits_{j=1}^{m}{D(\mathbf{x}_j)}^2$ where $D(\mathbf{x}_i)$ is the distance between the instance $\mathbf{x}_i$ and the closest centroid that was already chosen. This probability distribution ensures that instances that are further away from already chosen centroids are much more likely be selected as centroids.
# * Repeat the previous step until all $k$ centroids have been chosen.

# + [markdown] id="7xHY5_P4-Xf1"
# The rest of the K-Means++ algorithm is just regular K-Means. With this initialization, the K-Means algorithm is much less likely to converge to a suboptimal solution, so it is possible to reduce `n_init` considerably. Most of the time, this largely compensates for the additional complexity of the initialization process.

# + [markdown] id="Mgt9AhiL-Xf1"
# To set the initialization to K-Means++, simply set `init="k-means++"` (this is actually the default):

# + id="4s39RMus-Xf2" outputId="0e6b29a8-86b6-4c5d-ba99-196ead0bc371"
KMeans()

# + id="5qJukx8o-Xf-" outputId="77d19fb8-ac5d-43b4-d439-bccb44b6df3b"
good_init = np.array([[-3, 3], [-3, 2], [-3, 1], [-1, 2], [0, 2]])
kmeans = KMeans(n_clusters=5, init=good_init, n_init=1, random_state=42)
kmeans.fit(X)
kmeans.inertia_

# + [markdown] id="cBqcV7O--XgF"
# ### Accelerated K-Means

# + [markdown] id="hJTkEq3V-XgG"
# The K-Means algorithm can be significantly accelerated by avoiding many unnecessary distance calculations: this is achieved by exploiting the triangle inequality (given three points A, B and C, the distance AC is always such that AC ≤ AB + BC) and by keeping track of lower and upper bounds for distances between instances and centroids (see this [2003 paper](https://www.aaai.org/Papers/ICML/2003/ICML03-022.pdf) by Charles Elkan for more details).

# + [markdown] id="9MRKdCzD-XgH"
# To use Elkan's variant of K-Means, just set `algorithm="elkan"`. Note that it does not support sparse data, so by default, Scikit-Learn uses `"elkan"` for dense data, and `"full"` (the regular K-Means algorithm) for sparse data.

# + id="U2_9dQ_L-XgI" outputId="b47e55b9-e467-42a0-9233-0d298857fe08"
# %timeit -n 50 KMeans(algorithm="elkan").fit(X)

# + id="QaZbEcgb-XgT" outputId="3bb0a823-b3fc-451f-e97b-c7cc7f34c533"
# %timeit -n 50 KMeans(algorithm="full").fit(X)

# + [markdown] id="ZGSvvaRJ-XgZ"
# ### Mini-Batch K-Means

# + [markdown] id="5zs2YHNG-Xga"
# Scikit-Learn also implements a variant of the K-Means algorithm that supports mini-batches (see [this paper](http://www.eecs.tufts.edu/~dsculley/papers/fastkmeans.pdf)):

# + id="-LrKmB8D-Xgb"
from sklearn.cluster import MiniBatchKMeans

# + id="NxVzf0mu-Xge" outputId="dd036051-e435-4f09-c4cf-b299e7dfbfb4"
minibatch_kmeans = MiniBatchKMeans(n_clusters=5, random_state=42)
minibatch_kmeans.fit(X)

# + id="rdDNwzsP-Xgm" outputId="7fee4e1f-241a-450b-def2-446824557e7f"
minibatch_kmeans.inertia_

# + [markdown] id="5iYMcOh2-Xgo"
# If the dataset does not fit in memory, the simplest option is to use the `memmap` class, just like we did for incremental PCA in the previous chapter. First let's load MNIST:

# + id="BKu4VPfV-Xgr"
import urllib
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1)
mnist.target = mnist.target.astype(np.int64)

# + id="bVIyUOHr-Xgw"
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    mnist["data"], mnist["target"], random_state=42)

# + [markdown] id="4FxKJlj_-Xg7"
# Next, let's write it to a `memmap`:

# + id="MZ_pNIku-Xg8"
filename = "my_mnist.data"
X_mm = np.memmap(filename, dtype='float32', mode='write', shape=X_train.shape)
X_mm[:] = X_train

# + id="eSXnuWpj-XhA" outputId="0981307f-b4b3-474a-ce08-aa4d74aafbfb"
minibatch_kmeans = MiniBatchKMeans(n_clusters=10, batch_size=10, random_state=42)
minibatch_kmeans.fit(X_mm)


# + [markdown] id="jI8m-QrE-XhG"
# If your data is so large that you cannot use `memmap`, things get more complicated. Let's start by writing a function to load the next batch (in real life, you would load the data from disk):

# + id="fNEcfGEz-XhH"
def load_next_batch(batch_size):
    return X[np.random.choice(len(X), batch_size, replace=False)]


# + [markdown] id="hNNRkHPU-XhJ"
# Now we can train the model by feeding it one batch at a time. We also need to implement multiple initializations and keep the model with the lowest inertia:

# + id="pRtY2fzw-XhK"
np.random.seed(42)

# + id="yeqMr1m1-XhM"
k = 5
n_init = 10
n_iterations = 100
batch_size = 100
init_size = 500  # more data for K-Means++ initialization
evaluate_on_last_n_iters = 10

best_kmeans = None

for init in range(n_init):
    minibatch_kmeans = MiniBatchKMeans(n_clusters=k, init_size=init_size)
    X_init = load_next_batch(init_size)
    minibatch_kmeans.partial_fit(X_init)

    minibatch_kmeans.sum_inertia_ = 0
    for iteration in range(n_iterations):
        X_batch = load_next_batch(batch_size)
        minibatch_kmeans.partial_fit(X_batch)
        if iteration >= n_iterations - evaluate_on_last_n_iters:
            minibatch_kmeans.sum_inertia_ += minibatch_kmeans.inertia_

    if (best_kmeans is None or
        minibatch_kmeans.sum_inertia_ < best_kmeans.sum_inertia_):
        best_kmeans = minibatch_kmeans

# + id="faBsAG-w-XhQ" outputId="e17154f2-51f7-442c-a6ff-2e7389b268c1"
best_kmeans.score(X)

# + [markdown] id="raDmDM7F-XhX"
# Mini-batch K-Means is much faster than regular K-Means:

# + id="AJq_aGR2-XhY" outputId="12eeb36e-e7b2-4fa8-836c-13b550fa7097"
# %timeit KMeans(n_clusters=5).fit(X)

# + id="4OgyZTX0-Xhc" outputId="1ef666b6-aebd-4c09-dcfc-0971b448a504"
# %timeit MiniBatchKMeans(n_clusters=5).fit(X)

# + [markdown] id="KQUK55ju-Xhj"
# That's *much* faster! However, its performance is often lower (higher inertia), and it keeps degrading as _k_ increases. Let's plot the inertia ratio and the training time ratio between Mini-batch K-Means and regular K-Means:

# + id="kICHkFtB-Xhl"
from timeit import timeit

# + id="8FXHwunG-Xhq" outputId="bd0d16fe-c92e-42bd-cf04-fc98d78f6e8f"
times = np.empty((100, 2))
inertias = np.empty((100, 2))
for k in range(1, 101):
    kmeans_ = KMeans(n_clusters=k, random_state=42)
    minibatch_kmeans = MiniBatchKMeans(n_clusters=k, random_state=42)
    print("\r{}/{}".format(k, 100), end="")
    times[k-1, 0] = timeit("kmeans_.fit(X)", number=10, globals=globals())
    times[k-1, 1]  = timeit("minibatch_kmeans.fit(X)", number=10, globals=globals())
    inertias[k-1, 0] = kmeans_.inertia_
    inertias[k-1, 1] = minibatch_kmeans.inertia_

# + id="d9SplAWC-Xht" outputId="054a221c-b395-4f36-8313-b74b8eb09554"
plt.figure(figsize=(10,4))

plt.subplot(121)
plt.plot(range(1, 101), inertias[:, 0], "r--", label="K-Means")
plt.plot(range(1, 101), inertias[:, 1], "b.-", label="Mini-batch K-Means")
plt.xlabel("$k$", fontsize=16)
plt.title("Inertia", fontsize=14)
plt.legend(fontsize=14)
plt.axis([1, 100, 0, 100])

plt.subplot(122)
plt.plot(range(1, 101), times[:, 0], "r--", label="K-Means")
plt.plot(range(1, 101), times[:, 1], "b.-", label="Mini-batch K-Means")
plt.xlabel("$k$", fontsize=16)
plt.title("Training time (seconds)", fontsize=14)
plt.axis([1, 100, 0, 6])

save_fig("minibatch_kmeans_vs_kmeans")
plt.show()

# + [markdown] id="bCZKVMeY-Xhw"
# ### Finding the optimal number of clusters

# + [markdown] id="lYFWe5Mq-Xhx"
# What if the number of clusters was set to a lower or greater value than 5?

# + id="vGe6OlUq-Xhx" outputId="d05e5efb-2933-4ad8-a8b4-6d30194380e1"
kmeans_k3 = KMeans(n_clusters=3, random_state=42)
kmeans_k8 = KMeans(n_clusters=8, random_state=42)

plot_clusterer_comparison(kmeans_k3, kmeans_k8, X, "$k=3$", "$k=8$")
save_fig("bad_n_clusters_plot")
plt.show()

# + [markdown] id="IANA3l0Z-Xh1"
# Ouch, these two models don't look great. What about their inertias?

# + id="vUAf27Qr-Xh2" outputId="8cb07837-144b-41fa-9987-a6bd99539535"
kmeans_k3.inertia_

# + id="cNDta4zk-Xh6" outputId="9d881681-f150-4c43-bd2d-779539f70e3a"
kmeans_k8.inertia_

# + [markdown] id="Nii6L_Ma-Xh-"
# No, we cannot simply take the value of $k$ that minimizes the inertia, since it keeps getting lower as we increase $k$. Indeed, the more clusters there are, the closer each instance will be to its closest centroid, and therefore the lower the inertia will be. However, we can plot the inertia as a function of $k$ and analyze the resulting curve:

# + id="C-60VUsv-Xh-"
kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(X)
                for k in range(1, 10)]
inertias = [model.inertia_ for model in kmeans_per_k]

# + id="AvANNN_y-XiA" outputId="a0f01e41-abea-4e42-9d3e-5d88a80aae51"
plt.figure(figsize=(8, 3.5))
plt.plot(range(1, 10), inertias, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Inertia", fontsize=14)
plt.annotate('Elbow',
             xy=(4, inertias[3]),
             xytext=(0.55, 0.55),
             textcoords='figure fraction',
             fontsize=16,
             arrowprops=dict(facecolor='black', shrink=0.1)
            )
plt.axis([1, 8.5, 0, 1300])
save_fig("inertia_vs_k_plot")
plt.show()

# + [markdown] id="8dYRLwAP-XiI"
# As you can see, there is an elbow at $k=4$, which means that less clusters than that would be bad, and more clusters would not help much and might cut clusters in half. So $k=4$ is a pretty good choice. Of course in this example it is not perfect since it means that the two blobs in the lower left will be considered as just a single cluster, but it's a pretty good clustering nonetheless.

# + id="y6I_uYI0-XiI" outputId="d3df14c0-96ea-4092-aa30-b47447fbf2cc"
plot_decision_boundaries(kmeans_per_k[4-1], X)
plt.show()

# + [markdown] id="52q62HjX-XiL"
# Another approach is to look at the _silhouette score_, which is the mean _silhouette coefficient_ over all the instances. An instance's silhouette coefficient is equal to $(b - a)/\max(a, b)$ where $a$ is the mean distance to the other instances in the same cluster (it is the _mean intra-cluster distance_), and $b$ is the _mean nearest-cluster distance_, that is the mean distance to the instances of the next closest cluster (defined as the one that minimizes $b$, excluding the instance's own cluster). The silhouette coefficient can vary between -1 and +1: a coefficient close to +1 means that the instance is well inside its own cluster and far from other clusters, while a coefficient close to 0 means that it is close to a cluster boundary, and finally a coefficient close to -1 means that the instance may have been assigned to the wrong cluster.

# + [markdown] id="wMqS3Nnj-XiL"
# Let's plot the silhouette score as a function of $k$:

# + id="HeA-L3nU-XiL"
from sklearn.metrics import silhouette_score

# + id="GhC7DggH-XiO" outputId="94ae5cfb-f904-4934-805f-c3f9c420861c"
silhouette_score(X, kmeans.labels_)

# + id="Tcwj__Gh-XiW"
silhouette_scores = [silhouette_score(X, model.labels_)
                     for model in kmeans_per_k[1:]]

# + id="1vNEa3sj-Xib" outputId="503c3e5b-4a32-4a7d-cdd4-e1d3d0483cb9"
plt.figure(figsize=(8, 3))
plt.plot(range(2, 10), silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score", fontsize=14)
plt.axis([1.8, 8.5, 0.55, 0.7])
save_fig("silhouette_score_vs_k_plot")
plt.show()

# + [markdown] id="cN3BsQjB-Xig"
# As you can see, this visualization is much richer than the previous one: in particular, although it confirms that $k=4$ is a very good choice, but it also underlines the fact that $k=5$ is quite good as well.

# + [markdown] id="iE_ebc9n-Xih"
# An even more informative visualization is given when you plot every instance's silhouette coefficient, sorted by the cluster they are assigned to and by the value of the coefficient. This is called a _silhouette diagram_:

# + id="IPlN3VyQ-Xii" outputId="2429ef10-3f85-40c6-aedd-7142acc66ec1"
from sklearn.metrics import silhouette_samples
from matplotlib.ticker import FixedLocator, FixedFormatter

plt.figure(figsize=(11, 9))

for k in (3, 4, 5, 6):
    plt.subplot(2, 2, k - 2)
    
    y_pred = kmeans_per_k[k - 1].labels_
    silhouette_coefficients = silhouette_samples(X, y_pred)

    padding = len(X) // 30
    pos = padding
    ticks = []
    for i in range(k):
        coeffs = silhouette_coefficients[y_pred == i]
        coeffs.sort()

        color = mpl.cm.Spectral(i / k)
        plt.fill_betweenx(np.arange(pos, pos + len(coeffs)), 0, coeffs,
                          facecolor=color, edgecolor=color, alpha=0.7)
        ticks.append(pos + len(coeffs) // 2)
        pos += len(coeffs) + padding

    plt.gca().yaxis.set_major_locator(FixedLocator(ticks))
    plt.gca().yaxis.set_major_formatter(FixedFormatter(range(k)))
    if k in (3, 5):
        plt.ylabel("Cluster")
    
    if k in (5, 6):
        plt.gca().set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        plt.xlabel("Silhouette Coefficient")
    else:
        plt.tick_params(labelbottom=False)

    plt.axvline(x=silhouette_scores[k - 2], color="red", linestyle="--")
    plt.title("$k={}$".format(k), fontsize=16)

save_fig("silhouette_analysis_plot")
plt.show()

# + [markdown] id="j-oaomXB-Xik"
# ### Limits of K-Means

# + id="89WDB7zX-Xil"
X1, y1 = make_blobs(n_samples=1000, centers=((4, -4), (0, 0)), random_state=42)
X1 = X1.dot(np.array([[0.374, 0.95], [0.732, 0.598]]))
X2, y2 = make_blobs(n_samples=250, centers=1, random_state=42)
X2 = X2 + [6, -8]
X = np.r_[X1, X2]
y = np.r_[y1, y2]

# + id="sotjz_pH-Xit" outputId="a01b6453-3dea-46d3-bb11-e6bc158d6127"
plot_clusters(X)

# + id="PQhBXjU7-Xix" outputId="bc446453-0059-4bce-96ba-8a0682bc7781"
kmeans_good = KMeans(n_clusters=3, init=np.array([[-1.5, 2.5], [0.5, 0], [4, 0]]), n_init=1, random_state=42)
kmeans_bad = KMeans(n_clusters=3, random_state=42)
kmeans_good.fit(X)
kmeans_bad.fit(X)

# + id="5L0GeeMN-Xi2" outputId="47d6d236-0e9c-4ad1-ba7f-66f42266b598"
plt.figure(figsize=(10, 3.2))

plt.subplot(121)
plot_decision_boundaries(kmeans_good, X)
plt.title("Inertia = {:.1f}".format(kmeans_good.inertia_), fontsize=14)

plt.subplot(122)
plot_decision_boundaries(kmeans_bad, X, show_ylabels=False)
plt.title("Inertia = {:.1f}".format(kmeans_bad.inertia_), fontsize=14)

save_fig("bad_kmeans_plot")
plt.show()

# + [markdown] id="7ZLHbQxm-Xi8"
# ### Using clustering for image segmentation

# + id="c871dowi-Xi9"
# Download the ladybug image
images_path = os.path.join(PROJECT_ROOT_DIR, "images", "unsupervised_learning")
os.makedirs(images_path, exist_ok=True)
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
filename = "ladybug.png"
print("Downloading", filename)
url = DOWNLOAD_ROOT + "images/unsupervised_learning/" + filename
urllib.request.urlretrieve(url, os.path.join(images_path, filename))

# + id="Qnk5EGAK-XjB" outputId="1e61fdf4-bfd0-4f0c-e64a-c6eb5de3beef"
from matplotlib.image import imread
image = imread(os.path.join(images_path, filename))
image.shape

# + id="A5SQ3wEN-XjE"
X = image.reshape(-1, 3)
kmeans = KMeans(n_clusters=8, random_state=42).fit(X)
segmented_img = kmeans.cluster_centers_[kmeans.labels_]
segmented_img = segmented_img.reshape(image.shape)

# + id="yNC-pr16-XjG"
segmented_imgs = []
n_colors = (10, 8, 6, 4, 2)
for n_clusters in n_colors:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
    segmented_img = kmeans.cluster_centers_[kmeans.labels_]
    segmented_imgs.append(segmented_img.reshape(image.shape))

# + id="CcurC2YY-XjM" outputId="6922d0e2-7cd5-4c19-efb4-f85d9b4a2249"
plt.figure(figsize=(10,5))
plt.subplots_adjust(wspace=0.05, hspace=0.1)

plt.subplot(231)
plt.imshow(image)
plt.title("Original image")
plt.axis('off')

for idx, n_clusters in enumerate(n_colors):
    plt.subplot(232 + idx)
    plt.imshow(segmented_imgs[idx])
    plt.title("{} colors".format(n_clusters))
    plt.axis('off')

save_fig('image_segmentation_diagram', tight_layout=False)
plt.show()

# + [markdown] id="P8Ugw3I6-XjO"
# ### Using Clustering for Preprocessing

# + [markdown] id="ux8tbKoQ-XjP"
# Let's tackle the _digits dataset_ which is a simple MNIST-like dataset containing 1,797 grayscale 8×8 images representing digits 0 to 9.

# + id="wGPtkpVs-XjP"
from sklearn.datasets import load_digits

# + id="X3fOt0P9-XjS"
X_digits, y_digits = load_digits(return_X_y=True)

# + [markdown] id="1Pq0DBKz-XjU"
# Let's split it into a training set and a test set:

# + id="pAr6XSDp-XjV"
from sklearn.model_selection import train_test_split

# + id="EWvmSFme-XjW"
X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits, random_state=42)

# + [markdown] id="tqI3QjJt-Xjc"
# Now let's fit a Logistic Regression model and evaluate it on the test set:

# + id="wanUH_bs-Xjd"
from sklearn.linear_model import LogisticRegression

# + id="Po5ZfQH6-Xjg" outputId="04920ef8-c677-4176-cdc8-fed18c3a58a8"
log_reg = LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=5000, random_state=42)
log_reg.fit(X_train, y_train)

# + id="UXUNDiDe-Xjj" outputId="f1a1c20d-2b9d-402c-e34b-e9647f7727c0"
log_reg.score(X_test, y_test)

# + [markdown] id="C9N3GByP-Xjr"
# Okay, that's our baseline: 96.89% accuracy. Let's see if we can do better by using K-Means as a preprocessing step. We will create a pipeline that will first cluster the training set into 50 clusters and replace the images with their distances to the 50 clusters, then apply a logistic regression model:

# + id="4_zfDYDc-Xjs"
from sklearn.pipeline import Pipeline

# + id="Bu1CN4x0-Xjz" outputId="cd4459ac-1c17-4d8c-808e-2e38beb91f21"
pipeline = Pipeline([
    ("kmeans", KMeans(n_clusters=50, random_state=42)),
    ("log_reg", LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=5000, random_state=42)),
])
pipeline.fit(X_train, y_train)

# + id="cIvkqTx4-Xj3" outputId="a97c3c80-4968-4990-e34d-8074db4d8324"
pipeline.score(X_test, y_test)

# + id="zibHr5Of-Xj_" outputId="200572ec-3fe5-41fc-befd-f00d160c455a"
1 - (1 - 0.977777) / (1 - 0.968888)

# + [markdown] id="rATL6X8v-XkC"
# How about that? We reduced the error rate by over 28%! But we chose the number of clusters $k$ completely arbitrarily, we can surely do better. Since K-Means is just a preprocessing step in a classification pipeline, finding a good value for $k$ is much simpler than earlier: there's no need to perform silhouette analysis or minimize the inertia, the best value of $k$ is simply the one that results in the best classification performance.

# + id="lnmBqfDY-XkC"
from sklearn.model_selection import GridSearchCV

# + id="8iBcp9tO-XkE" outputId="39b17df5-30fe-4f55-cd93-a3ed37f36e82"
param_grid = dict(kmeans__n_clusters=range(2, 100))
grid_clf = GridSearchCV(pipeline, param_grid, cv=3, verbose=2)
grid_clf.fit(X_train, y_train)

# + [markdown] id="-ttoDSTM-XkG"
# Let's see what the best number of clusters is:

# + id="WCRVUT2t-XkH" outputId="b981a418-4c9f-4c70-9083-32d5ae304fc3"
grid_clf.best_params_

# + id="i6Th8L10-XkM" outputId="9737a272-dc0f-4f28-839d-ef03cf2e6b49"
grid_clf.score(X_test, y_test)

# + [markdown] id="QROMTw2M-XkS"
# ### Clustering for Semi-supervised Learning

# + [markdown] id="rZAu7vDz-XkU"
# Another use case for clustering is in semi-supervised learning, when we have plenty of unlabeled instances and very few labeled instances.

# + [markdown] id="1s5NMfXi-XkV"
# Let's look at the performance of a logistic regression model when we only have 50 labeled instances:

# + id="ug5L2gbR-XkW"
n_labeled = 50

# + id="tPA4euF4-XkY" outputId="43ba60da-71e5-4bc9-9538-2b066eb722db"
log_reg = LogisticRegression(multi_class="ovr", solver="lbfgs", random_state=42)
log_reg.fit(X_train[:n_labeled], y_train[:n_labeled])
log_reg.score(X_test, y_test)

# + [markdown] id="SH6B3wvp-Xki"
# It's much less than earlier of course. Let's see how we can do better. First, let's cluster the training set into 50 clusters, then for each cluster let's find the image closest to the centroid. We will call these images the representative images:

# + id="4MHZ4PRH-Xkk"
k = 50

# + id="OzDXZ4vs-Xkp"
kmeans = KMeans(n_clusters=k, random_state=42)
X_digits_dist = kmeans.fit_transform(X_train)
representative_digit_idx = np.argmin(X_digits_dist, axis=0)
X_representative_digits = X_train[representative_digit_idx]

# + [markdown] id="xIhgfJOn-Xkz"
# Now let's plot these representative images and label them manually:

# + id="urY_WnBF-Xk0" outputId="c6adfd3e-6e24-4ccf-a97d-eff9419b968c"
plt.figure(figsize=(8, 2))
for index, X_representative_digit in enumerate(X_representative_digits):
    plt.subplot(k // 10, 10, index + 1)
    plt.imshow(X_representative_digit.reshape(8, 8), cmap="binary", interpolation="bilinear")
    plt.axis('off')

save_fig("representative_images_diagram", tight_layout=False)
plt.show()

# + id="I07Xu0MD-Xk4"
y_representative_digits = np.array([
    4, 8, 0, 6, 8, 3, 7, 7, 9, 2,
    5, 5, 8, 5, 2, 1, 2, 9, 6, 1,
    1, 6, 9, 0, 8, 3, 0, 7, 4, 1,
    6, 5, 2, 4, 1, 8, 6, 3, 9, 2,
    4, 2, 9, 4, 7, 6, 2, 3, 1, 1])

# + [markdown] id="8DfKRcii-Xk7"
# Now we have a dataset with just 50 labeled instances, but instead of being completely random instances, each of them is a representative image of its cluster. Let's see if the performance is any better:

# + id="jgf_JY0q-Xk7" outputId="1a874c1c-885a-4279-ba50-0bb99191b9e8"
log_reg = LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=5000, random_state=42)
log_reg.fit(X_representative_digits, y_representative_digits)
log_reg.score(X_test, y_test)

# + [markdown] id="FksqbRZq-XlH"
# Wow! We jumped from 83.3% accuracy to 92.2%, although we are still only training the model on 50 instances. Since it's often costly and painful to label instances, especially when it has to be done manually by experts, it's a good idea to make them label representative instances rather than just random instances.

# + [markdown] id="jSboqzWx-XlI"
# But perhaps we can go one step further: what if we propagated the labels to all the other instances in the same cluster?

# + id="1Zr5UiGZ-XlI"
y_train_propagated = np.empty(len(X_train), dtype=np.int32)
for i in range(k):
    y_train_propagated[kmeans.labels_==i] = y_representative_digits[i]

# + id="sdISaFx9-XlK" outputId="bfce9060-2195-4108-a08a-ac997471e001"
log_reg = LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=5000, random_state=42)
log_reg.fit(X_train, y_train_propagated)

# + id="bFNpD1pY-XlN" outputId="dae35ba7-27bb-4901-cdf0-d74c56301526"
log_reg.score(X_test, y_test)

# + [markdown] id="F94wbXV_-XlS"
# We got a tiny little accuracy boost. Better than nothing, but we should probably have propagated the labels only to the instances closest to the centroid, because by propagating to the full cluster, we have certainly included some outliers. Let's only propagate the labels to the 20th percentile closest to the centroid:

# + id="tkSktsLR-XlS"
percentile_closest = 20

X_cluster_dist = X_digits_dist[np.arange(len(X_train)), kmeans.labels_]
for i in range(k):
    in_cluster = (kmeans.labels_ == i)
    cluster_dist = X_cluster_dist[in_cluster]
    cutoff_distance = np.percentile(cluster_dist, percentile_closest)
    above_cutoff = (X_cluster_dist > cutoff_distance)
    X_cluster_dist[in_cluster & above_cutoff] = -1

# + id="CWaeUIGf-XlV"
partially_propagated = (X_cluster_dist != -1)
X_train_partially_propagated = X_train[partially_propagated]
y_train_partially_propagated = y_train_propagated[partially_propagated]

# + id="ojEIzJHL-XlY" outputId="e04506be-0118-47d9-b06a-6a7683384001"
log_reg = LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=5000, random_state=42)
log_reg.fit(X_train_partially_propagated, y_train_partially_propagated)

# + id="rAzNKrqC-Xla" outputId="d1bc8a5b-a905-4861-c2a2-b44ed77fc5b0"
log_reg.score(X_test, y_test)

# + [markdown] id="y4J25hP6-Xlb"
# Nice! With just 50 labeled instances (just 5 examples per class on average!), we got 94% performance, which is pretty close to the performance of logistic regression on the fully labeled _digits_ dataset (which was 96.9%).

# + [markdown] id="fpEKOyad-Xlc"
# This is because the propagated labels are actually pretty good: their accuracy is very close to 99%:

# + id="0OfdRUtr-Xld" outputId="72314372-6e8b-4176-fa52-c748695b2bf4"
np.mean(y_train_partially_propagated == y_train[partially_propagated])

# + [markdown] id="-kLuk8aL-Xli"
# You could now do a few iterations of *active learning*:
# 1. Manually label the instances that the classifier is least sure about, if possible by picking them in distinct clusters.
# 2. Train a new model with these additional labels.

# + [markdown] id="dqy9l9Yy-Xlj"
# ## DBSCAN

# + id="97L2VF79-Xlj" executionInfo={"status": "ok", "timestamp": 1603459007760, "user_tz": -540, "elapsed": 2290, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}}
from sklearn.datasets import make_moons

# + id="l2bgJM3C-Xlm" executionInfo={"status": "ok", "timestamp": 1603459007764, "user_tz": -540, "elapsed": 1823, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}}
X, y = make_moons(n_samples=1000, noise=0.05, random_state=42)

# + id="OOtRY6b_-Xlo" executionInfo={"status": "ok", "timestamp": 1603459007774, "user_tz": -540, "elapsed": 1500, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}}
from sklearn.cluster import DBSCAN

# + id="c2ZK2pcd-Xlr" executionInfo={"status": "ok", "timestamp": 1603459007777, "user_tz": -540, "elapsed": 1263, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="3b568189-0726-4577-8387-66ce3cb4ca45" colab={"base_uri": "https://localhost:8080/", "height": 55}
dbscan = DBSCAN(eps=0.05, min_samples=5)   ########### 엡실론 0.05 
dbscan.fit(X)

# + id="PMyxqtE--Xl0" executionInfo={"status": "ok", "timestamp": 1603459051605, "user_tz": -540, "elapsed": 929, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="1809031a-20fa-4276-8964-64fd8834a3c4" colab={"base_uri": "https://localhost:8080/", "height": 35}
dbscan.labels_[:10]  ##-1이 anomaly 

# + id="Z1dda0cd-Xl3" executionInfo={"status": "ok", "timestamp": 1603459052007, "user_tz": -540, "elapsed": 605, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="d93acc0a-3c79-45aa-a235-ed85b6b0751d" colab={"base_uri": "https://localhost:8080/", "height": 35}
len(dbscan.core_sample_indices_)

# + id="kLdQ_4uZ-Xl5" executionInfo={"status": "ok", "timestamp": 1603459054554, "user_tz": -540, "elapsed": 1150, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="514b47b1-ea8a-4ef7-e4b4-b62e8a58e31c" colab={"base_uri": "https://localhost:8080/", "height": 35}
dbscan.core_sample_indices_[:10]

# + id="PldL30fC-Xl7" executionInfo={"status": "ok", "timestamp": 1603459054557, "user_tz": -540, "elapsed": 668, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="46ea659a-4db8-48ec-a606-169cd0d6e5e8" colab={"base_uri": "https://localhost:8080/", "height": 73}
dbscan.components_[:3]

# + id="uzd5auIR-Xl9" executionInfo={"status": "ok", "timestamp": 1603459056785, "user_tz": -540, "elapsed": 1435, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="ec50a98e-2bdd-4409-d314-73e75ac400c9" colab={"base_uri": "https://localhost:8080/", "height": 36}
np.unique(dbscan.labels_)

# + id="yDjHPzyC-XmB" executionInfo={"status": "ok", "timestamp": 1603459059874, "user_tz": -540, "elapsed": 1068, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="1fc835dc-18f6-4c03-f69f-dba934ac21c1" colab={"base_uri": "https://localhost:8080/", "height": 55}
dbscan2 = DBSCAN(eps=0.2) #3####################3 입실론 0.2 
dbscan2.fit(X)


# + id="lOlxYQVM-XmE" executionInfo={"status": "ok", "timestamp": 1603459169058, "user_tz": -540, "elapsed": 1707, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}}
def plot_dbscan(dbscan, X, size, show_xlabels=True, show_ylabels=True):
    core_mask = np.zeros_like(dbscan.labels_, dtype=bool)
    core_mask[dbscan.core_sample_indices_] = True
    anomalies_mask = dbscan.labels_ == -1
    non_core_mask = ~(core_mask | anomalies_mask)

    cores = dbscan.components_
    anomalies = X[anomalies_mask]
    non_cores = X[non_core_mask]
    
    plt.scatter(cores[:, 0], cores[:, 1],
                c=dbscan.labels_[core_mask], marker='o', s=size, cmap="Paired")
    plt.scatter(cores[:, 0], cores[:, 1], marker='*', s=20, c=dbscan.labels_[core_mask])
    plt.scatter(anomalies[:, 0], anomalies[:, 1],
                c="r", marker="x", s=100)
    plt.scatter(non_cores[:, 0], non_cores[:, 1], c=dbscan.labels_[non_core_mask], marker=".")
    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)
    plt.title("eps={:.2f}, min_samples={}".format(dbscan.eps, dbscan.min_samples), fontsize=14)


# + id="GHKpMGKK-XmK" executionInfo={"status": "ok", "timestamp": 1603459173901, "user_tz": -540, "elapsed": 3690, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="bb1ea7e1-c7dc-42f9-86b1-ff9ca80f9faf" colab={"base_uri": "https://localhost:8080/", "height": 258}
plt.figure(figsize=(9, 3.2))

plt.subplot(121)
plot_dbscan(dbscan, X, size=100)

plt.subplot(122)
plot_dbscan(dbscan2, X, size=600, show_ylabels=False)

save_fig("dbscan_plot")
plt.show()
### 파라미터 설정이 특히 중요하다 !!! 

# + id="u5h8ZLQy-XmO" executionInfo={"status": "ok", "timestamp": 1603459245044, "user_tz": -540, "elapsed": 12834, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}}
dbscan = dbscan2

# + [markdown] id="8sVNuOPEnkHX"
# dbscan은 새로운 데이터에 대해서 predict가 안된다. knn으로 판단해서 추가적으로 분류에 적합시켜서 prediction을 실시한다. 

# + id="Qqr0CSLD-XmV" executionInfo={"status": "ok", "timestamp": 1603459299524, "user_tz": -540, "elapsed": 1079, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}}
from sklearn.neighbors import KNeighborsClassifier

# + id="xcbbkhq8-XmY" executionInfo={"status": "ok", "timestamp": 1603459300044, "user_tz": -540, "elapsed": 1219, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="61e2eb16-47b3-4325-b30f-954613e60869" colab={"base_uri": "https://localhost:8080/", "height": 73}
knn = KNeighborsClassifier(n_neighbors=50)
knn.fit(dbscan.components_, dbscan.labels_[dbscan.core_sample_indices_])

# + id="-0iGOhQF-Xme" executionInfo={"status": "ok", "timestamp": 1603459302308, "user_tz": -540, "elapsed": 921, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="bd465c67-f185-4bb8-ef2e-b2e2d91ee86c" colab={"base_uri": "https://localhost:8080/", "height": 36}
X_new = np.array([[-0.5, 0], [0, 0.5], [1, -0.1], [2, 1]])
knn.predict(X_new)

# + id="LwoTVeEj-Xmj" executionInfo={"status": "ok", "timestamp": 1603459304323, "user_tz": -540, "elapsed": 790, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="3697dedd-237c-4691-9d03-3d9975da529c" colab={"base_uri": "https://localhost:8080/", "height": 92}
knn.predict_proba(X_new)

# + id="nExD4LZg-Xmm" executionInfo={"status": "ok", "timestamp": 1603459335980, "user_tz": -540, "elapsed": 32054, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="659b797e-23af-405b-ad9f-1a824b687b86" colab={"base_uri": "https://localhost:8080/", "height": 244}
plt.figure(figsize=(6, 3))
plot_decision_boundaries(knn, X, show_centroids=False)
plt.scatter(X_new[:, 0], X_new[:, 1], c="b", marker="+", s=200, zorder=10)
save_fig("cluster_classification_plot")
plt.show()

# + id="8yj1HMJB-Xmo" executionInfo={"status": "ok", "timestamp": 1603459335983, "user_tz": -540, "elapsed": 31894, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="ce94ca68-fa2d-47dd-ee0e-ead909eb4531" colab={"base_uri": "https://localhost:8080/", "height": 35}
y_dist, y_pred_idx = knn.kneighbors(X_new, n_neighbors=1)
y_pred = dbscan.labels_[dbscan.core_sample_indices_][y_pred_idx]
y_pred[y_dist > 0.2] = -1
y_pred.ravel()

# + [markdown] id="oj3UJl8Bn782"
# other clustering algorithms는 안함. 

# + [markdown] id="zvYA-jDG-Xmp"
# ## Other Clustering Algorithms

# + [markdown] id="WqK3GI2U-Xmq"
# ### Spectral Clustering

# + id="J69QTNKn-Xmq"
from sklearn.cluster import SpectralClustering

# + id="-fI07oNF-Xmu" outputId="8d2e1e61-0775-4824-c8bf-20064184c77e"
sc1 = SpectralClustering(n_clusters=2, gamma=100, random_state=42)
sc1.fit(X)

# + id="WYgylhpK-Xmw" outputId="2410fc95-29fd-4a59-a625-6d1464369d0d"
sc2 = SpectralClustering(n_clusters=2, gamma=1, random_state=42)
sc2.fit(X)

# + id="dkNPu21P-Xm4" outputId="61d97651-2643-4c7c-a874-c27350a944fc"
np.percentile(sc1.affinity_matrix_, 95)


# + id="x8czG74P-Xm7"
def plot_spectral_clustering(sc, X, size, alpha, show_xlabels=True, show_ylabels=True):
    plt.scatter(X[:, 0], X[:, 1], marker='o', s=size, c='gray', cmap="Paired", alpha=alpha)
    plt.scatter(X[:, 0], X[:, 1], marker='o', s=30, c='w')
    plt.scatter(X[:, 0], X[:, 1], marker='.', s=10, c=sc.labels_, cmap="Paired")
    
    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)
    plt.title("RBF gamma={}".format(sc.gamma), fontsize=14)


# + id="621N8pWU-Xm9" outputId="e6c27259-0d59-43ed-f220-c69ef8e0c027"
plt.figure(figsize=(9, 3.2))

plt.subplot(121)
plot_spectral_clustering(sc1, X, size=500, alpha=0.1)

plt.subplot(122)
plot_spectral_clustering(sc2, X, size=4000, alpha=0.01, show_ylabels=False)

plt.show()


# + [markdown] id="_chLKMx--Xm_"
# ### Agglomerative Clustering

# + id="9oA9_sKL-Xm_"
from sklearn.cluster import AgglomerativeClustering

# + id="_kSPX97e-XnF"
X = np.array([0, 2, 5, 8.5]).reshape(-1, 1)
agg = AgglomerativeClustering(linkage="complete").fit(X)


# + id="EasS3a_8-XnI"
def learned_parameters(estimator):
    return [attrib for attrib in dir(estimator)
            if attrib.endswith("_") and not attrib.startswith("_")]


# + id="7vrfVw9S-XnK" outputId="b7beb836-0ac7-4acc-81af-1b98b5630a94"
learned_parameters(agg)

# + id="VWw32Xji-XnM" outputId="fd8e9bd5-a55c-4fed-9a10-802d73f9d71a"
agg.children_

# + [markdown] id="3q5IjG1E-XnO"
# # Gaussian Mixtures

# + id="VVeE8sF5-XnW" executionInfo={"status": "ok", "timestamp": 1603460044245, "user_tz": -540, "elapsed": 1014, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}}
X1, y1 = make_blobs(n_samples=1000, centers=((4, -4), (0, 0)), random_state=42) # 두개의 센터로 blobs생성 
X1 = X1.dot(np.array([[0.374, 0.95], [0.732, 0.598]]))
X2, y2 = make_blobs(n_samples=250, centers=1, random_state=42)
X2 = X2 + [6, -8]  
X = np.r_[X1, X2]
y = np.r_[y1, y2]

# + [markdown] id="f9edNnPw-XnY"
# Let's train a Gaussian mixture model on the previous dataset:

# + id="522vdTqn-XnZ" executionInfo={"status": "ok", "timestamp": 1603460046399, "user_tz": -540, "elapsed": 1980, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}}
from sklearn.mixture import GaussianMixture

# + id="XnJWX-bq-Xnb" executionInfo={"status": "ok", "timestamp": 1603460055484, "user_tz": -540, "elapsed": 914, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="2ca5e39b-cb94-4a11-ddac-ce0edfaaecdb" colab={"base_uri": "https://localhost:8080/", "height": 111}
gm = GaussianMixture(n_components=3, n_init=10, random_state=42) #3개의 가우시안 믹스쳐 
gm.fit(X)

# + [markdown] id="JsyQDv5g-Xnc"
# Let's look at the parameters that the EM algorithm estimated:

# + id="uTW7slOi-Xnd" executionInfo={"status": "ok", "timestamp": 1603460080683, "user_tz": -540, "elapsed": 1319, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="bc0cf16d-fe7e-40e0-bd54-3a23004d3edc" colab={"base_uri": "https://localhost:8080/", "height": 35}
gm.weights_ #각 가우시안 모델의 weight 

# + id="2MIiDLeI-Xnf" executionInfo={"status": "ok", "timestamp": 1603460096736, "user_tz": -540, "elapsed": 1127, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="e50f98b2-a3fb-4670-99e5-a10f7a2099a6" colab={"base_uri": "https://localhost:8080/", "height": 71}
gm.means_ #각 가우시안 분포의 평균  

# + id="lYA0WVFY-Xnl" executionInfo={"status": "ok", "timestamp": 1603460084040, "user_tz": -540, "elapsed": 900, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="b9847f25-6181-4519-c559-6e01e431e7eb" colab={"base_uri": "https://localhost:8080/", "height": 161}
gm.covariances_ 

# + [markdown] id="KFI_FfBb-Xno"
# Did the algorithm actually converge?

# + id="_Dj7qtyp-Xno" executionInfo={"status": "ok", "timestamp": 1603460110604, "user_tz": -540, "elapsed": 1018, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="7cf971d4-396b-4541-a1fb-cd7d69ca743a" colab={"base_uri": "https://localhost:8080/", "height": 35}
gm.converged_

# + [markdown] id="TsdU53uC-Xnq"
# Yes, good. How many iterations did it take?

# + id="NZLqrHLm-Xnq" executionInfo={"status": "ok", "timestamp": 1603460111014, "user_tz": -540, "elapsed": 627, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="dff2b0f1-ae53-4c9b-fc8c-c24f7252a9a0" colab={"base_uri": "https://localhost:8080/", "height": 36}
gm.n_iter_ #모형이 수렴할때까지의 이터레이션 수 

# + [markdown] id="Nhw4x2m4-Xnr"
# You can now use the model to predict which cluster each instance belongs to (hard clustering) or the probabilities that it came from each cluster. For this, just use `predict()` method or the `predict_proba()` method:

# + id="kiit4Jb3-Xns" executionInfo={"status": "ok", "timestamp": 1603460172020, "user_tz": -540, "elapsed": 930, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="fb9d5c4c-5fc0-446b-9d02-3138a1c68a97" colab={"base_uri": "https://localhost:8080/", "height": 36}
gm.predict(X)

# + id="_s_u4ETr-Xnu" executionInfo={"status": "ok", "timestamp": 1603460179816, "user_tz": -540, "elapsed": 552, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="d962aba8-d87a-41b2-8683-74435944dc7b" colab={"base_uri": "https://localhost:8080/", "height": 148}
gm.predict_proba(X)

# + [markdown] id="cSGSItRM-Xnx"
# This is a generative model, so you can sample new instances from it (and get their labels):

# + id="EKtzNqso-Xnx" executionInfo={"status": "ok", "timestamp": 1603460200476, "user_tz": -540, "elapsed": 1151, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="909a7961-071b-41e2-b9d3-a7ba1b8de88c" colab={"base_uri": "https://localhost:8080/", "height": 129}
X_new, y_new = gm.sample(6) #가우시안 믹스쳐에서 sample를 새로 뽑아보기 
X_new

# + id="xGbu4aaK-Xn0" executionInfo={"status": "ok", "timestamp": 1603460201712, "user_tz": -540, "elapsed": 555, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="9be6a675-3f55-427b-d225-f2dfd3da439b" colab={"base_uri": "https://localhost:8080/", "height": 36}
y_new

# + [markdown] id="DDhg_Q5g-Xn4"
# Notice that they are sampled sequentially from each cluster.

# + [markdown] id="6kNyX4ce-Xn4"
# You can also estimate the log of the _probability density function_ (PDF) at any location using the `score_samples()` method:

# + id="J6QW29tb-Xn4" executionInfo={"status": "ok", "timestamp": 1603460203684, "user_tz": -540, "elapsed": 1098, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="346e5603-2ab9-4204-a54a-bd504f798f29" colab={"base_uri": "https://localhost:8080/", "height": 55}
gm.score_samples(X)

# + [markdown] id="PL3ufZQq-Xn5"
# Let's check that the PDF integrates to 1 over the whole space. We just take a large square around the clusters, and chop it into a grid of tiny squares, then we compute the approximate probability that the instances will be generated in each tiny square (by multiplying the PDF at one corner of the tiny square by the area of the square), and finally summing all these probabilities). The result is very close to 1:

# + id="xiyRhcxl-Xn6" executionInfo={"status": "ok", "timestamp": 1603460208117, "user_tz": -540, "elapsed": 2229, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="cd589b51-370c-4280-a481-f55046903e07" colab={"base_uri": "https://localhost:8080/", "height": 35}
resolution = 100
grid = np.arange(-10, 10, 1 / resolution)
xx, yy = np.meshgrid(grid, grid)
X_full = np.vstack([xx.ravel(), yy.ravel()]).T

pdf = np.exp(gm.score_samples(X_full))
pdf_probas = pdf * (1 / resolution) ** 2
pdf_probas.sum()

# + [markdown] id="NOm1Go5q-Xn7"
# Now let's plot the resulting decision boundaries (dashed lines) and density contours:

# + id="MQkN_2q9-Xn8" executionInfo={"status": "ok", "timestamp": 1603460208651, "user_tz": -540, "elapsed": 747, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}}
from matplotlib.colors import LogNorm

def plot_gaussian_mixture(clusterer, X, resolution=1000, show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    Z = -clusterer.score_samples(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z,
                 norm=LogNorm(vmin=1.0, vmax=30.0),
                 levels=np.logspace(0, 2, 12))
    plt.contour(xx, yy, Z,
                norm=LogNorm(vmin=1.0, vmax=30.0),
                levels=np.logspace(0, 2, 12),
                linewidths=1, colors='k')

    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z,
                linewidths=2, colors='r', linestyles='dashed')
    
    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)
    plot_centroids(clusterer.means_, clusterer.weights_)

    plt.xlabel("$x_1$", fontsize=14)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)


# + id="hkV-wAQW-Xn-" executionInfo={"status": "ok", "timestamp": 1603460213854, "user_tz": -540, "elapsed": 3489, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="7dfbfe24-69ba-4a3a-d9df-e6a3876819e8" colab={"base_uri": "https://localhost:8080/", "height": 316}
plt.figure(figsize=(8, 4))

plot_gaussian_mixture(gm, X)

save_fig("gaussian_mixtures_plot")
plt.show()

# + [markdown] id="oBBnSpQy-XoC"
# You can impose constraints on the covariance matrices that the algorithm looks for by setting the `covariance_type` hyperparameter:
# * `"full"` (default): no constraint, all clusters can take on any ellipsoidal shape of any size.
# * `"tied"`: all clusters must have the same shape, which can be any ellipsoid (i.e., they all share the same covariance matrix).
# * `"spherical"`: all clusters must be spherical, but they can have different diameters (i.e., different variances).
# * `"diag"`: clusters can take on any ellipsoidal shape of any size, but the ellipsoid's axes must be parallel to the axes (i.e., the covariance matrices must be diagonal).

# + id="2cv8LhUO-XoC" executionInfo={"status": "ok", "timestamp": 1603460259405, "user_tz": -540, "elapsed": 3503, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="9d616c96-992b-4927-93cf-6344f96d6257" colab={"base_uri": "https://localhost:8080/", "height": 111}
gm_full = GaussianMixture(n_components=3, n_init=10, covariance_type="full", random_state=42)
gm_tied = GaussianMixture(n_components=3, n_init=10, covariance_type="tied", random_state=42)
gm_spherical = GaussianMixture(n_components=3, n_init=10, covariance_type="spherical", random_state=42)
gm_diag = GaussianMixture(n_components=3, n_init=10, covariance_type="diag", random_state=42)
gm_full.fit(X)
gm_tied.fit(X)
gm_spherical.fit(X)
gm_diag.fit(X)


# + id="MDqqFNt7-XoG" executionInfo={"status": "ok", "timestamp": 1603460259968, "user_tz": -540, "elapsed": 2330, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}}
def compare_gaussian_mixtures(gm1, gm2, X):
    plt.figure(figsize=(9, 4))

    plt.subplot(121)
    plot_gaussian_mixture(gm1, X)
    plt.title('covariance_type="{}"'.format(gm1.covariance_type), fontsize=14)

    plt.subplot(122)
    plot_gaussian_mixture(gm2, X, show_ylabels=False)
    plt.title('covariance_type="{}"'.format(gm2.covariance_type), fontsize=14)



# + id="01azIZeC-XoI" executionInfo={"status": "ok", "timestamp": 1603460273293, "user_tz": -540, "elapsed": 4366, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="d52a4099-0e39-432d-f447-b51acac197a7" colab={"base_uri": "https://localhost:8080/", "height": 315}
compare_gaussian_mixtures(gm_tied, gm_spherical, X)

save_fig("covariance_type_plot")
plt.show()

# + id="ZU1j0LBq-XoL" executionInfo={"status": "ok", "timestamp": 1603460276635, "user_tz": -540, "elapsed": 7319, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="2ab1195a-ed9d-4b32-9998-9ecfe0cf432c" colab={"base_uri": "https://localhost:8080/", "height": 297}
compare_gaussian_mixtures(gm_full, gm_diag, X)
plt.tight_layout()
plt.show()

# + [markdown] id="fTUrRMZq-XoM"
# ## Anomaly Detection using Gaussian Mixtures

# + [markdown] id="0BhQGYad-XoM"
# Gaussian Mixtures can be used for _anomaly detection_: instances located in low-density regions can be considered anomalies. You must define what density threshold you want to use. For example, in a manufacturing company that tries to detect defective products, the ratio of defective products is usually well-known. Say it is equal to 4%, then you can set the density threshold to be the value that results in having 4% of the instances located in areas below that threshold density:

# + [markdown] id="t0f5JgTHrkCN"
# 가우시안 믹스쳐에서 특정 threshold보다 낮은경우 anomaly 표시 

# + id="XQIHurDZ-XoM" executionInfo={"status": "ok", "timestamp": 1603460306705, "user_tz": -540, "elapsed": 1157, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}}
densities = gm.score_samples(X)
density_threshold = np.percentile(densities, 4)
anomalies = X[densities < density_threshold]

# + id="s7D8v3CL-XoN" executionInfo={"status": "ok", "timestamp": 1603460310141, "user_tz": -540, "elapsed": 4091, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="c965bdf6-5add-4fc1-abb0-7a9c0e7ce691" colab={"base_uri": "https://localhost:8080/", "height": 316}
plt.figure(figsize=(8, 4))

plot_gaussian_mixture(gm, X)
plt.scatter(anomalies[:, 0], anomalies[:, 1], color='r', marker='*')
plt.ylim(top=5.1)

save_fig("mixture_anomaly_detection_plot")
plt.show()

# + [markdown] id="RpD_y8laruSQ"
# 모델 셀력션 그냥 안하고 넘어감 

# + [markdown] id="iDhrb5Fy-XoT"
# ## Model selection

# + [markdown] id="reA7iLdR-XoU"
# We cannot use the inertia or the silhouette score because they both assume that the clusters are spherical. Instead, we can try to find the model that minimizes a theoretical information criterion such as the Bayesian Information Criterion (BIC) or the Akaike Information Criterion (AIC):
#
# ${BIC} = {\log(m)p - 2\log({\hat L})}$
#
# ${AIC} = 2p - 2\log(\hat L)$
#
# * $m$ is the number of instances.
# * $p$ is the number of parameters learned by the model.
# * $\hat L$ is the maximized value of the likelihood function of the model. This is the conditional probability of the observed data $\mathbf{X}$, given the model and its optimized parameters.
#
# Both BIC and AIC penalize models that have more parameters to learn (e.g., more clusters), and reward models that fit the data well (i.e., models that give a high likelihood to the observed data).

# + id="FGEKR3TX-XoX" outputId="d351aa71-dfa4-4049-d13b-bb035862e99f"
gm.bic(X)

# + id="oYrbExPS-Xoc" outputId="059f957c-f0e6-4647-eadd-152307997a9f"
gm.aic(X)

# + [markdown] id="PSpjkIEU-Xog"
# We could compute the BIC manually like this:

# + id="31S2w7l4-Xoh"
n_clusters = 3
n_dims = 2
n_params_for_weights = n_clusters - 1
n_params_for_means = n_clusters * n_dims
n_params_for_covariance = n_clusters * n_dims * (n_dims + 1) // 2
n_params = n_params_for_weights + n_params_for_means + n_params_for_covariance
max_log_likelihood = gm.score(X) * len(X) # log(L^)
bic = np.log(len(X)) * n_params - 2 * max_log_likelihood
aic = 2 * n_params - 2 * max_log_likelihood

# + id="-o7VinwN-Xoj" outputId="f8e4c648-e420-47b6-ad44-63734de2852f"
bic, aic

# + id="DMz54hYV-Xop" outputId="57eaebd6-27fb-4f76-9bb6-5add6ec3affe"
n_params

# + [markdown] id="BhA0pX-9-Xov"
# There's one weight per cluster, but the sum must be equal to 1, so we have one degree of freedom less, hence the -1. Similarly, the degrees of freedom for an $n \times n$ covariance matrix is not $n^2$, but $1 + 2 + \dots + n = \dfrac{n (n+1)}{2}$.

# + [markdown] id="PNimvyiz-Xoz"
# Let's train Gaussian Mixture models with various values of $k$ and measure their BIC:

# + id="0CjqzLq0-Xo0"
gms_per_k = [GaussianMixture(n_components=k, n_init=10, random_state=42).fit(X)
             for k in range(1, 11)]

# + id="tLZlhryY-Xo6"
bics = [model.bic(X) for model in gms_per_k]
aics = [model.aic(X) for model in gms_per_k]

# + id="S144b-4v-Xo8" outputId="a77bdddf-35e8-485e-aa12-a991e609c489"
plt.figure(figsize=(8, 3))
plt.plot(range(1, 11), bics, "bo-", label="BIC")
plt.plot(range(1, 11), aics, "go--", label="AIC")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Information Criterion", fontsize=14)
plt.axis([1, 9.5, np.min(aics) - 50, np.max(aics) + 50])
plt.annotate('Minimum',
             xy=(3, bics[2]),
             xytext=(0.35, 0.6),
             textcoords='figure fraction',
             fontsize=14,
             arrowprops=dict(facecolor='black', shrink=0.1)
            )
plt.legend()
save_fig("aic_bic_vs_k_plot")
plt.show()

# + [markdown] id="408qy7JX-Xo-"
# Let's search for best combination of values for both the number of clusters and the `covariance_type` hyperparameter:

# + id="FzfJspCW-Xo-"
min_bic = np.infty

for k in range(1, 11):
    for covariance_type in ("full", "tied", "spherical", "diag"):
        bic = GaussianMixture(n_components=k, n_init=10,
                              covariance_type=covariance_type,
                              random_state=42).fit(X).bic(X)
        if bic < min_bic:
            min_bic = bic
            best_k = k
            best_covariance_type = covariance_type

# + id="DH90RHhY-XpA" outputId="60b9fd76-8bbc-42c7-ad1f-b4adf8d057e8"
best_k

# + id="dnEWfCOD-XpC" outputId="888cf2a7-441f-4671-bb88-9246f5b92611"
best_covariance_type

# + [markdown] id="Abd_mmfV-XpE"
# ## Variational Bayesian Gaussian Mixtures

# + [markdown] id="3xEIvTom-XpG"
# Rather than manually searching for the optimal number of clusters, it is possible to use instead the `BayesianGaussianMixture` class which is capable of giving weights equal (or close) to zero to unnecessary clusters. Just set the number of components to a value that you believe is greater than the optimal number of clusters, and the algorithm will eliminate the unnecessary clusters automatically.

# + id="awKDkxAm-XpG" executionInfo={"status": "ok", "timestamp": 1603460342161, "user_tz": -540, "elapsed": 1087, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}}
from sklearn.mixture import BayesianGaussianMixture

# + id="rf3WpD7P-XpH" executionInfo={"status": "ok", "timestamp": 1603460346574, "user_tz": -540, "elapsed": 4639, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="d445884a-7243-45aa-a1f3-ac8a16c25d4c" colab={"base_uri": "https://localhost:8080/", "height": 224}
bgm = BayesianGaussianMixture(n_components=10, n_init=10, random_state=42) #컴포넌트를 10으로 결정하면 알아서 k를 결정해줌 
bgm.fit(X)

# + [markdown] id="RVPNGtCa-XpI"
# The algorithm automatically detected that only 3 components are needed:

# + id="WIMJKluT-XpI" executionInfo={"status": "ok", "timestamp": 1603460346576, "user_tz": -540, "elapsed": 2706, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="994b9ffd-e2bb-494e-c7b1-3775ad34aa01" colab={"base_uri": "https://localhost:8080/", "height": 35}
np.round(bgm.weights_, 2) #안쓰는건 0으로 봐꿔서 최종적으로는 k=3개만 남게됨. 

# + id="8zLX38Jk-XpJ" executionInfo={"status": "ok", "timestamp": 1603460396203, "user_tz": -540, "elapsed": 3961, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="1cf12d50-d7c6-460f-a02d-587f18874cd2" colab={"base_uri": "https://localhost:8080/", "height": 343}
plt.figure(figsize=(8, 5))
plot_gaussian_mixture(bgm, X)
plt.show()

# + id="d2E5asCl-XpK" executionInfo={"status": "ok", "timestamp": 1603460434811, "user_tz": -540, "elapsed": 1902, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="f085f0a7-6b06-4f1f-8e09-f6bded809b7f" colab={"base_uri": "https://localhost:8080/", "height": 161}
#bayesian은 prior에 따라서 모형이 변화한다. 
bgm_low = BayesianGaussianMixture(n_components=10, max_iter=1000, n_init=1,
                                  weight_concentration_prior=0.01, random_state=42)
bgm_high = BayesianGaussianMixture(n_components=10, max_iter=1000, n_init=1,
                                  weight_concentration_prior=10000, random_state=42)
nn = 73
bgm_low.fit(X[:nn])
bgm_high.fit(X[:nn])

# + id="8TtMx5Mb-XpM" executionInfo={"status": "ok", "timestamp": 1603460434812, "user_tz": -540, "elapsed": 1439, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="05ed851e-a55d-4064-f243-696e34b813ca" colab={"base_uri": "https://localhost:8080/", "height": 36}
np.round(bgm_low.weights_, 2)

# + id="7pOo9deY-XpN" executionInfo={"status": "ok", "timestamp": 1603460434823, "user_tz": -540, "elapsed": 1211, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="9b738651-2734-48f7-af71-20ff9ac2f7ab" colab={"base_uri": "https://localhost:8080/", "height": 36}
np.round(bgm_high.weights_, 2)

# + id="iOt8v4lz-XpO" executionInfo={"status": "ok", "timestamp": 1603460444227, "user_tz": -540, "elapsed": 8749, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="1cba5fe4-ac5c-428a-9386-d0ff245c7aef" colab={"base_uri": "https://localhost:8080/", "height": 316}
plt.figure(figsize=(9, 4))

plt.subplot(121)
plot_gaussian_mixture(bgm_low, X[:nn])
plt.title("weight_concentration_prior = 0.01", fontsize=14)

plt.subplot(122)
plot_gaussian_mixture(bgm_high, X[:nn], show_ylabels=False)
plt.title("weight_concentration_prior = 10000", fontsize=14)

save_fig("mixture_concentration_prior_plot")
plt.show()

# + [markdown] id="NMFXrlXB-XpQ"
# Note: the fact that you see only 3 regions in the right plot although there are 4 centroids is not a bug. The weight of the top-right cluster is much larger than the weight of the lower-right cluster, so the probability that any given point in this region belongs to the top right cluster is greater than the probability that it belongs to the lower-right cluster.

# + id="rV8zoyUV-XpQ" executionInfo={"status": "ok", "timestamp": 1603460444974, "user_tz": -540, "elapsed": 1203, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}}
X_moons, y_moons = make_moons(n_samples=1000, noise=0.05, random_state=42)

# + id="omkep1zE-XpR" executionInfo={"status": "ok", "timestamp": 1603460448123, "user_tz": -540, "elapsed": 4100, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="8fb6b639-7f11-4445-a933-ebd0289749c0" colab={"base_uri": "https://localhost:8080/", "height": 167}
bgm = BayesianGaussianMixture(n_components=10, n_init=10, random_state=42)
bgm.fit(X_moons)

# + id="QMeKHvkj-XpV" executionInfo={"status": "ok", "timestamp": 1603460451922, "user_tz": -540, "elapsed": 5414, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="bad02455-e037-4638-e71a-3bcdac4a414c" colab={"base_uri": "https://localhost:8080/", "height": 259}
plt.figure(figsize=(9, 3.2))

plt.subplot(121)
plot_data(X_moons)
plt.xlabel("$x_1$", fontsize=14)
plt.ylabel("$x_2$", fontsize=14, rotation=0)

plt.subplot(122)
plot_gaussian_mixture(bgm, X_moons, show_ylabels=False)

save_fig("moons_vs_bgm_plot")
plt.show()
##이 데이터는 가우시안이랑 모양이 많이 다르기 때문에, 적용하면 이상하게 분류가 된다. 

# + [markdown] id="v1s8nRrQ-XpW"
# Oops, not great... instead of detecting 2 moon-shaped clusters, the algorithm detected 8 ellipsoidal clusters. However, the density plot does not look too bad, so it might be usable for anomaly detection.

# + [markdown] id="2b7jOPI7-XpW"
# ## Likelihood Function

# + id="gpB3F1K0-XpX"
from scipy.stats import norm

# + id="JuqG-ATY-XpY"
xx = np.linspace(-6, 4, 101)
ss = np.linspace(1, 2, 101)
XX, SS = np.meshgrid(xx, ss)
ZZ = 2 * norm.pdf(XX - 1.0, 0, SS) + norm.pdf(XX + 4.0, 0, SS)
ZZ = ZZ / ZZ.sum(axis=1)[:,np.newaxis] / (xx[1] - xx[0])

# + id="FbsWnBTQ-XpZ" outputId="0e9ec3b7-fc1f-49b9-b7ce-0de81da31d0f"
from matplotlib.patches import Polygon

plt.figure(figsize=(8, 4.5))

x_idx = 85
s_idx = 30

plt.subplot(221)
plt.contourf(XX, SS, ZZ, cmap="GnBu")
plt.plot([-6, 4], [ss[s_idx], ss[s_idx]], "k-", linewidth=2)
plt.plot([xx[x_idx], xx[x_idx]], [1, 2], "b-", linewidth=2)
plt.xlabel(r"$x$")
plt.ylabel(r"$\theta$", fontsize=14, rotation=0)
plt.title(r"Model $f(x; \theta)$", fontsize=14)

plt.subplot(222)
plt.plot(ss, ZZ[:, x_idx], "b-")
max_idx = np.argmax(ZZ[:, x_idx])
max_val = np.max(ZZ[:, x_idx])
plt.plot(ss[max_idx], max_val, "r.")
plt.plot([ss[max_idx], ss[max_idx]], [0, max_val], "r:")
plt.plot([0, ss[max_idx]], [max_val, max_val], "r:")
plt.text(1.01, max_val + 0.005, r"$\hat{L}$", fontsize=14)
plt.text(ss[max_idx]+ 0.01, 0.055, r"$\hat{\theta}$", fontsize=14)
plt.text(ss[max_idx]+ 0.01, max_val - 0.012, r"$Max$", fontsize=12)
plt.axis([1, 2, 0.05, 0.15])
plt.xlabel(r"$\theta$", fontsize=14)
plt.grid(True)
plt.text(1.99, 0.135, r"$=f(x=2.5; \theta)$", fontsize=14, ha="right")
plt.title(r"Likelihood function $\mathcal{L}(\theta|x=2.5)$", fontsize=14)

plt.subplot(223)
plt.plot(xx, ZZ[s_idx], "k-")
plt.axis([-6, 4, 0, 0.25])
plt.xlabel(r"$x$", fontsize=14)
plt.grid(True)
plt.title(r"PDF $f(x; \theta=1.3)$", fontsize=14)
verts = [(xx[41], 0)] + list(zip(xx[41:81], ZZ[s_idx, 41:81])) + [(xx[80], 0)]
poly = Polygon(verts, facecolor='0.9', edgecolor='0.5')
plt.gca().add_patch(poly)

plt.subplot(224)
plt.plot(ss, np.log(ZZ[:, x_idx]), "b-")
max_idx = np.argmax(np.log(ZZ[:, x_idx]))
max_val = np.max(np.log(ZZ[:, x_idx]))
plt.plot(ss[max_idx], max_val, "r.")
plt.plot([ss[max_idx], ss[max_idx]], [-5, max_val], "r:")
plt.plot([0, ss[max_idx]], [max_val, max_val], "r:")
plt.axis([1, 2, -2.4, -2])
plt.xlabel(r"$\theta$", fontsize=14)
plt.text(ss[max_idx]+ 0.01, max_val - 0.05, r"$Max$", fontsize=12)
plt.text(ss[max_idx]+ 0.01, -2.39, r"$\hat{\theta}$", fontsize=14)
plt.text(1.01, max_val + 0.02, r"$\log \, \hat{L}$", fontsize=14)
plt.grid(True)
plt.title(r"$\log \, \mathcal{L}(\theta|x=2.5)$", fontsize=14)

save_fig("likelihood_function_plot")
plt.show()

# + id="VHLN3Hv9-Xpa"



# + [markdown] id="sqpjN0TT-Xpc"
# # Exercise solutions

# + [markdown] id="ZTMwcKH2-Xpc"
# ## 1. to 9.

# + [markdown] id="-3tGOnP9-Xpc"
# See Appendix A.

# + [markdown] id="wjaHSjWy-Xpc"
# ## 10. Cluster the Olivetti Faces Dataset

# + [markdown] id="f4_tq0B9-Xpd"
# *Exercise: The classic Olivetti faces dataset contains 400 grayscale 64 × 64–pixel images of faces. Each image is flattened to a 1D vector of size 4,096. 40 different people were photographed (10 times each), and the usual task is to train a model that can predict which person is represented in each picture. Load the dataset using the `sklearn.datasets.fetch_olivetti_faces()` function.*

# + id="DeovA92X-Xpd"
from sklearn.datasets import fetch_olivetti_faces

olivetti = fetch_olivetti_faces()

# + id="raQ3c1T--Xpf" outputId="7ff87cdc-5bd7-48a3-ba6a-5b46e1bb75be"
print(olivetti.DESCR)

# + id="1_xwWaYJ-Xpg" outputId="4dae2adf-c3d8-49fe-c592-fabebd444e04"
olivetti.target

# + [markdown] id="j3eF8ssO-Xpj"
# *Exercise: Then split it into a training set, a validation set, and a test set (note that the dataset is already scaled between 0 and 1). Since the dataset is quite small, you probably want to use stratified sampling to ensure that there are the same number of images per person in each set.*

# + id="jCcBqxW6-Xpj"
from sklearn.model_selection import StratifiedShuffleSplit

strat_split = StratifiedShuffleSplit(n_splits=1, test_size=40, random_state=42)
train_valid_idx, test_idx = next(strat_split.split(olivetti.data, olivetti.target))
X_train_valid = olivetti.data[train_valid_idx]
y_train_valid = olivetti.target[train_valid_idx]
X_test = olivetti.data[test_idx]
y_test = olivetti.target[test_idx]

strat_split = StratifiedShuffleSplit(n_splits=1, test_size=80, random_state=43)
train_idx, valid_idx = next(strat_split.split(X_train_valid, y_train_valid))
X_train = X_train_valid[train_idx]
y_train = y_train_valid[train_idx]
X_valid = X_train_valid[valid_idx]
y_valid = y_train_valid[valid_idx]

# + id="Oit7l87X-Xpk" outputId="39a428ab-c3e8-4c3e-ca1a-1fcbe96aeeab"
print(X_train.shape, y_train.shape)
print(X_valid.shape, y_valid.shape)
print(X_test.shape, y_test.shape)

# + [markdown] id="TMh4UiIJ-Xpm"
# To speed things up, we'll reduce the data's dimensionality using PCA:

# + id="e5S3p5k_-Xpm" outputId="855d350e-5011-4b44-cf06-e6faa8726a86"
from sklearn.decomposition import PCA

pca = PCA(0.99)
X_train_pca = pca.fit_transform(X_train)
X_valid_pca = pca.transform(X_valid)
X_test_pca = pca.transform(X_test)

pca.n_components_

# + [markdown] id="19Zq0GrA-Xpn"
# *Exercise: Next, cluster the images using K-Means, and ensure that you have a good number of clusters (using one of the techniques discussed in this chapter).*

# + id="nPSwmBpk-Xpn" outputId="a44b039b-40bf-4cf8-deec-6feb7fc2fdf0"
from sklearn.cluster import KMeans

k_range = range(5, 150, 5)
kmeans_per_k = []
for k in k_range:
    print("k={}".format(k))
    kmeans = KMeans(n_clusters=k, random_state=42).fit(X_train_pca)
    kmeans_per_k.append(kmeans)

# + id="aiBwyUsN-Xpn" outputId="ba1fe249-7626-4c7e-f887-458a8899431e"
from sklearn.metrics import silhouette_score

silhouette_scores = [silhouette_score(X_train_pca, model.labels_)
                     for model in kmeans_per_k]
best_index = np.argmax(silhouette_scores)
best_k = k_range[best_index]
best_score = silhouette_scores[best_index]

plt.figure(figsize=(8, 3))
plt.plot(k_range, silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score", fontsize=14)
plt.plot(best_k, best_score, "rs")
plt.show()

# + id="U97s4TzK-Xpo" outputId="3a1b941d-fac0-4480-a299-5b1433465a34"
best_k

# + [markdown] id="gREJxvZA-Xpq"
# It looks like the best number of clusters is quite high, at 120. You might have expected it to be 40, since there are 40 different people on the pictures. However, the same person may look quite different on different pictures (e.g., with or without glasses, or simply shifted left or right).

# + id="c205dn4O-Xpq" outputId="33f7c6e4-5f3a-4493-92dc-a0c6a25c6b63"
inertias = [model.inertia_ for model in kmeans_per_k]
best_inertia = inertias[best_index]

plt.figure(figsize=(8, 3.5))
plt.plot(k_range, inertias, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Inertia", fontsize=14)
plt.plot(best_k, best_inertia, "rs")
plt.show()

# + [markdown] id="2YR31SgF-Xpr"
# The optimal number of clusters is not clear on this inertia diagram, as there is no obvious elbow, so let's stick with k=120.

# + id="N3bHljCK-Xpr"
best_model = kmeans_per_k[best_index]


# + [markdown] id="M3kqnVVW-Xpu"
# *Exercise: Visualize the clusters: do you see similar faces in each cluster?*

# + id="G5955U3--Xpu" outputId="85c75b5b-ee78-43a9-cec0-04f35698c996"
def plot_faces(faces, labels, n_cols=5):
    n_rows = (len(faces) - 1) // n_cols + 1
    plt.figure(figsize=(n_cols, n_rows * 1.1))
    for index, (face, label) in enumerate(zip(faces, labels)):
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(face.reshape(64, 64), cmap="gray")
        plt.axis("off")
        plt.title(label)
    plt.show()

for cluster_id in np.unique(best_model.labels_):
    print("Cluster", cluster_id)
    in_cluster = best_model.labels_==cluster_id
    faces = X_train[in_cluster].reshape(-1, 64, 64)
    labels = y_train[in_cluster]
    plot_faces(faces, labels)

# + [markdown] id="Me0JBINW-Xpw"
# About 2 out of 3 clusters are useful: that is, they contain at least 2 pictures, all of the same person. However, the rest of the clusters have either one or more intruders, or they have just a single picture.
#
# Clustering images this way may be too imprecise to be directly useful when training a model (as we will see below), but it can be tremendously useful when labeling images in a new dataset: it will usually make labelling much faster.

# + [markdown] id="c872k9dc-Xpw"
# ## 11. Using Clustering as Preprocessing for Classification

# + [markdown] id="xfUwbjUX-Xpw"
# *Exercise: Continuing with the Olivetti faces dataset, train a classifier to predict which person is represented in each picture, and evaluate it on the validation set.*

# + id="x_gl8f0z-Xpw" outputId="ffe24715-9fdd-47f8-8b49-4add16f9814d"
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=150, random_state=42)
clf.fit(X_train_pca, y_train)
clf.score(X_valid_pca, y_valid)

# + [markdown] id="I20iS8iG-Xpx"
# *Exercise: Next, use K-Means as a dimensionality reduction tool, and train a classifier on the reduced set.*

# + id="HHZ2AVxx-Xpx" outputId="4e6e42b6-a7a8-4ce0-9948-d2c895cd3c21"
X_train_reduced = best_model.transform(X_train_pca)
X_valid_reduced = best_model.transform(X_valid_pca)
X_test_reduced = best_model.transform(X_test_pca)

clf = RandomForestClassifier(n_estimators=150, random_state=42)
clf.fit(X_train_reduced, y_train)
    
clf.score(X_valid_reduced, y_valid)

# + [markdown] id="VeBCCspD-Xpy"
# Yikes! That's not better at all! Let's see if tuning the number of clusters helps.

# + [markdown] id="lY9t9t_D-Xpz"
# *Exercise: Search for the number of clusters that allows the classifier to get the best performance: what performance can you reach?*

# + [markdown] id="T5lW347Y-Xpz"
# We could use a `GridSearchCV` like we did earlier in this notebook, but since we already have a validation set, we don't need K-fold cross-validation, and we're only exploring a single hyperparameter, so it's simpler to just run a loop manually:

# + id="YagtSb4G-Xpz" outputId="ab7cdcf2-a429-48e5-f209-038972d96696"
from sklearn.pipeline import Pipeline

for n_clusters in k_range:
    pipeline = Pipeline([
        ("kmeans", KMeans(n_clusters=n_clusters, random_state=n_clusters)),
        ("forest_clf", RandomForestClassifier(n_estimators=150, random_state=42))
    ])
    pipeline.fit(X_train_pca, y_train)
    print(n_clusters, pipeline.score(X_valid_pca, y_valid))

# + [markdown] id="8jswFiYu-Xp0"
# Oh well, even by tuning the number of clusters, we never get beyond 80% accuracy. Looks like the distances to the cluster centroids are not as informative as the original images.

# + [markdown] id="G3wFidWG-Xp0"
# *Exercise: What if you append the features from the reduced set to the original features (again, searching for the best number of clusters)?*

# + id="c5AEALcO-Xp0"
X_train_extended = np.c_[X_train_pca, X_train_reduced]
X_valid_extended = np.c_[X_valid_pca, X_valid_reduced]
X_test_extended = np.c_[X_test_pca, X_test_reduced]

# + id="vXFYSUxY-Xp2" outputId="e4286e60-ccf3-4685-b962-195d198914d5"
clf = RandomForestClassifier(n_estimators=150, random_state=42)
clf.fit(X_train_extended, y_train)
clf.score(X_valid_extended, y_valid)

# + [markdown] id="3uEiarVY-Xp3"
# That's a bit better, but still worse than without the cluster features. The clusters are not useful to directly train a classifier in this case (but they can still help when labelling new training instances).

# + [markdown] id="GSCRH_9m-Xp3"
# ## 12. A Gaussian Mixture Model for the Olivetti Faces Dataset

# + [markdown] id="hLQ6CX3R-Xp3"
# *Exercise: Train a Gaussian mixture model on the Olivetti faces dataset. To speed up the algorithm, you should probably reduce the dataset's dimensionality (e.g., use PCA, preserving 99% of the variance).*

# + id="L0A34ypS-Xp4"
from sklearn.mixture import GaussianMixture

gm = GaussianMixture(n_components=40, random_state=42)
y_pred = gm.fit_predict(X_train_pca)

# + [markdown] id="l7b99s6T-Xp6"
# *Exercise: Use the model to generate some new faces (using the `sample()` method), and visualize them (if you used PCA, you will need to use its `inverse_transform()` method).*

# + id="bGncMc2t-Xp6"
n_gen_faces = 20
gen_faces_reduced, y_gen_faces = gm.sample(n_samples=n_gen_faces)
gen_faces = pca.inverse_transform(gen_faces_reduced)

# + id="dFyswvXi-Xp7" outputId="c1af4c10-1157-48d9-f38d-3b8a18ab8e7b"
plot_faces(gen_faces, y_gen_faces)

# + [markdown] id="zMSLp82t-Xp8"
# *Exercise: Try to modify some images (e.g., rotate, flip, darken) and see if the model can detect the anomalies (i.e., compare the output of the `score_samples()` method for normal images and for anomalies).*

# + id="tZV4zxHL-Xp8" outputId="14bef2a1-658c-4f4e-e359-948cf04ba6e2"
n_rotated = 4
rotated = np.transpose(X_train[:n_rotated].reshape(-1, 64, 64), axes=[0, 2, 1])
rotated = rotated.reshape(-1, 64*64)
y_rotated = y_train[:n_rotated]

n_flipped = 3
flipped = X_train[:n_flipped].reshape(-1, 64, 64)[:, ::-1]
flipped = flipped.reshape(-1, 64*64)
y_flipped = y_train[:n_flipped]

n_darkened = 3
darkened = X_train[:n_darkened].copy()
darkened[:, 1:-1] *= 0.3
darkened = darkened.reshape(-1, 64*64)
y_darkened = y_train[:n_darkened]

X_bad_faces = np.r_[rotated, flipped, darkened]
y_bad = np.concatenate([y_rotated, y_flipped, y_darkened])

plot_faces(X_bad_faces, y_bad)

# + id="5ndfJ9dS-Xp9"
X_bad_faces_pca = pca.transform(X_bad_faces)

# + id="5HH9XgWQ-Xp-" outputId="54ccfa46-9b0e-4f7e-f2cf-8ff50ae0a99d"
gm.score_samples(X_bad_faces_pca)

# + [markdown] id="ytwokbkQ-Xp_"
# The bad faces are all considered highly unlikely by the Gaussian Mixture model. Compare this to the scores of some training instances:

# + id="SxwPQOUu-Xp_" outputId="72a9d554-3285-4400-e3fa-70dc19c737a3"
gm.score_samples(X_train_pca[:10])

# + [markdown] id="7VVHA3dH-Xp_"
# ## 13. Using Dimensionality Reduction Techniques for Anomaly Detection

# + [markdown] id="g-wuXZaB-XqA"
# *Exercise: Some dimensionality reduction techniques can also be used for anomaly detection. For example, take the Olivetti faces dataset and reduce it with PCA, preserving 99% of the variance. Then compute the reconstruction error for each image. Next, take some of the modified images you built in the previous exercise, and look at their reconstruction error: notice how much larger the reconstruction error is. If you plot a reconstructed image, you will see why: it tries to reconstruct a normal face.*

# + [markdown] id="BtcvHvSF-XqA"
# We already reduced the dataset using PCA earlier:

# + id="O56ITKGC-XqA" outputId="d64bded0-860c-4359-9a8e-50e550ff71c2"
X_train_pca


# + id="NJMrvOtc-XqB"
def reconstruction_errors(pca, X):
    X_pca = pca.transform(X)
    X_reconstructed = pca.inverse_transform(X_pca)
    mse = np.square(X_reconstructed - X).mean(axis=-1)
    return mse


# + id="bTHQZDgD-XqC" outputId="749f6e2a-366d-4c72-ea40-749ba266c1f7"
reconstruction_errors(pca, X_train).mean()

# + id="lqkeIF_F-XqF" outputId="0d6d8ca6-89b4-4b78-c01a-5912bd838729"
reconstruction_errors(pca, X_bad_faces).mean()

# + id="yELku7nc-XqG" outputId="60eab9e8-e8bf-41e3-d56f-25ff9e2acd78"
plot_faces(X_bad_faces, y_gen_faces)

# + id="HveAkXzh-XqH" outputId="ad4849ae-749e-4143-8338-dff70a324ba5"
X_bad_faces_reconstructed = pca.inverse_transform(X_bad_faces_pca)
plot_faces(X_bad_faces_reconstructed, y_gen_faces)

# + id="nUO8lInE-XqI"

