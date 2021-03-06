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

# + id="TpVxT9gaTAVw" executionInfo={"status": "ok", "timestamp": 1603450853904, "user_tz": -540, "elapsed": 931, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="d8c2c891-de28-4f09-a792-c635dee1a86f" colab={"base_uri": "https://localhost:8080/", "height": 55}
from google.colab import drive # import drive from google colab

ROOT = "/content/drive"     # default location for the drive
print(ROOT)                 # print content of ROOT (Optional)

drive.mount(ROOT)           # we mount the google drive at /content/drive

# + id="Zw_YAth8Tuut" executionInfo={"status": "ok", "timestamp": 1603450855274, "user_tz": -540, "elapsed": 1258, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="9665d6de-7035-43bb-e538-e189ea924fd7" colab={"base_uri": "https://localhost:8080/", "height": 55}
# %cd 'drive/My Drive/Colab Notebooks/datamining2/src/class/' 

# + id="Si2GMoeNTxRj" executionInfo={"status": "ok", "timestamp": 1603450857622, "user_tz": -540, "elapsed": 2629, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="35376da1-d4f0-4663-f50b-be0c76881ec4" colab={"base_uri": "https://localhost:8080/", "height": 261}
pip install jupytext #jupytext 설치 

# + id="XqeC-Qk6T1la" executionInfo={"elapsed": 2745, "status": "ok", "timestamp": 1602145454738, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}, "user_tz": -540} outputId="189ae668-b908-4554-8732-eac2efc992b2" colab={"base_uri": "https://localhost:8080/", "height": 92}
## Pair a notebook to a light script
# !jupytext --set-formats ipynb,py:light 08_dimensionality_reduction.ipynb  

# + id="-coqjg36T7v4" executionInfo={"elapsed": 1797, "status": "ok", "timestamp": 1602145477462, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}, "user_tz": -540} outputId="ba7fa35f-c19e-4e97-a4f7-2b3cb58c0f92" colab={"base_uri": "https://localhost:8080/", "height": 92}
# Sync the two representations
# !jupytext --sync 08_dimensionality_reduction.ipynb

# + [markdown] id="v-Fho7EJS4bR"
# **Chapter 8 – Dimensionality Reduction**
#
# _This notebook contains all the sample code and solutions to the exercises in chapter 8._

# + [markdown] id="oC_IdRI3S4bS"
# <table align="left">
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/ageron/handson-ml2/blob/master/08_dimensionality_reduction.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
#   </td>
# </table>

# + [markdown] id="8hvm0o8AS4bT"
# # Setup

# + [markdown] id="dNRFm1PBS4bU"
# First, let's import a few common modules, ensure MatplotLib plots figures inline and prepare a function to save the figures. We also check that Python 3.5 or later is installed (although Python 2.x may work, it is deprecated so we strongly recommend you use Python 3 instead), as well as Scikit-Learn ≥0.20.

# + id="ZFYRALlrS4bV"
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
CHAPTER_ID = "dim_reduction"
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

# + [markdown] id="erP6rwY-S4ba"
# # Projection methods
# Build 3D dataset:

# + id="SDVdpDkMS4ba"
###모의로 데이터를 만들어서 진행. 
np.random.seed(4)
m = 60
w1, w2 = 0.1, 0.3
noise = 0.1

angles = np.random.rand(m) * 3 * np.pi / 2 - 0.5
X = np.empty((m, 3))
X[:, 0] = np.cos(angles) + np.sin(angles)/2 + noise * np.random.randn(m) / 2
X[:, 1] = np.sin(angles) * 0.7 + noise * np.random.randn(m) / 2
X[:, 2] = X[:, 0] * w1 + X[:, 1] * w2 + noise * np.random.randn(m)

# + [markdown] id="lXNkyAXoS4bg"
# ## PCA using SVD decomposition

# + id="wDNes4gqS4bh"
X_centered = X - X.mean(axis=0) #centerd X생성 
U, s, Vt = np.linalg.svd(X_centered) # svd 분해 
c1 = Vt.T[:, 0] #2개의 pc를 뽑아냄 
c2 = Vt.T[:, 1]

# + id="9nTWsujGS4bl"
m, n = X.shape

S = np.zeros(X_centered.shape)
S[:n, :n] = np.diag(s)

# + id="n5Ov3ijlS4bo" executionInfo={"elapsed": 905, "status": "ok", "timestamp": 1602145598086, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}, "user_tz": -540} outputId="5690fafa-d593-48af-b204-78b60b265e76" colab={"base_uri": "https://localhost:8080/", "height": 36}
np.allclose(X_centered, U.dot(S).dot(Vt))

# + id="hl4dRNWSS4bs"
W2 = Vt.T[:, :2]
X2D = X_centered.dot(W2)

# + id="PePemozgS4bx"
X2D_using_svd = X2D  #원래의 3차원 X를 2차원 X2D_using_svd로 만듦. 

# + id="zsK5ZTuAUhwP"



# + [markdown] id="OxGWiNVmS4b3"
# ## PCA using Scikit-Learn

# + [markdown] id="b5-yljznS4b5"
# With Scikit-Learn, PCA is really trivial. It even takes care of mean centering for you:

# + [markdown] id="k9cd5Q49Um0J"
# 위에서 한걸 sklearn으로 만듦. 

# + id="QNv62R8tUmxB"



# + id="UlBJOqDlS4b6"
from sklearn.decomposition import PCA

pca = PCA(n_components = 2)
X2D = pca.fit_transform(X)

# + id="WCP86vI2S4b-" executionInfo={"elapsed": 1479, "status": "ok", "timestamp": 1602145672308, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}, "user_tz": -540} outputId="e197268b-60a6-4464-86b6-769fc91e66ff" colab={"base_uri": "https://localhost:8080/", "height": 111}
X2D[:5]

# + id="hPy5ZI26S4cC" executionInfo={"elapsed": 1073, "status": "ok", "timestamp": 1602145674081, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}, "user_tz": -540} outputId="d6193372-9e7a-4ce2-8d0f-603363cc4c3c" colab={"base_uri": "https://localhost:8080/", "height": 111}
X2D_using_svd[:5] 

# + [markdown] id="UiQBstaRS4cG"
# Notice that running PCA multiple times on slightly different datasets may result in different results. In general the only difference is that some axes may be flipped. In this example, PCA using Scikit-Learn gives the same projection as the one given by the SVD approach, except both axes are flipped:

# + id="mZXH4pQ8S4cI" executionInfo={"elapsed": 1006, "status": "ok", "timestamp": 1602145681831, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}, "user_tz": -540} outputId="85046c61-3bf0-4adf-d214-59c7ce1ec5ac" colab={"base_uri": "https://localhost:8080/", "height": 36}
np.allclose(X2D, -X2D_using_svd)

# + [markdown] id="TSfYbTLtS4cN"
# Recover the 3D points projected on the plane (PCA 2D subspace).

# + id="hfgwb-_VS4cO"
X3D_inv = pca.inverse_transform(X2D)   #다시 3차원으로 보내기 

# + [markdown] id="1gQTn0RvS4cR"
# Of course, there was some loss of information during the projection step, so the recovered 3D points are not exactly equal to the original 3D points:

# + id="AwcNCaGyS4cS" executionInfo={"elapsed": 952, "status": "ok", "timestamp": 1602145695048, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}, "user_tz": -540} outputId="0f6cd02d-5c6a-4d82-fd62-7aefc5f282e1" colab={"base_uri": "https://localhost:8080/", "height": 36}
np.allclose(X3D_inv, X)

# + [markdown] id="3ZKjlXpaS4cX"
# We can compute the reconstruction error:

# + id="XXFB7mtIS4cY" executionInfo={"elapsed": 925, "status": "ok", "timestamp": 1602145717789, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}, "user_tz": -540} outputId="380cd6bf-7963-4561-b11f-ece9bcfe1cee" colab={"base_uri": "https://localhost:8080/", "height": 36}
np.mean(np.sum(np.square(X3D_inv - X), axis=1)) #원래의 3차원과, 복원된 3차원은 약간의 오차가 발생한다. 

# + [markdown] id="EnwEWL64S4cb"
# The inverse transform in the SVD approach looks like this:

# + id="CFrGJS5KS4cb"
X3D_inv_using_svd = X2D_using_svd.dot(Vt[:2, :]) #svd를 사용하여 3차원으로 변환시키기. 

# + [markdown] id="Mo7hpPzDS4cf"
# The reconstructions from both methods are not identical because Scikit-Learn's `PCA` class automatically takes care of reversing the mean centering, but if we subtract the mean, we get the same reconstruction:

# + id="93ecaMS0S4cg" executionInfo={"elapsed": 1331, "status": "ok", "timestamp": 1602145786473, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}, "user_tz": -540} outputId="ca17c5dd-8772-4fbd-f2fc-a81f8785b174" colab={"base_uri": "https://localhost:8080/", "height": 36}
np.allclose(X3D_inv_using_svd, X3D_inv - pca.mean_)

# + [markdown] id="tVQI-JnaS4cj"
# The `PCA` object gives access to the principal components that it computed:

# + id="84UeWFu3S4cj" executionInfo={"elapsed": 906, "status": "ok", "timestamp": 1602145797859, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}, "user_tz": -540} outputId="3a610ee3-f874-485c-cae1-fa193fc336fb" colab={"base_uri": "https://localhost:8080/", "height": 55}
pca.components_ #어떤 pc가 생겼는지 살펴보기 

# + [markdown] id="NWKIVnWQS4cm"
# Compare to the first two principal components computed using the SVD method:

# + id="9D9995QYS4cn" executionInfo={"elapsed": 1616, "status": "ok", "timestamp": 1602145813241, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}, "user_tz": -540} outputId="d190befd-74ea-433c-a15a-3035f6b93bbe" colab={"base_uri": "https://localhost:8080/", "height": 55}
Vt[:2] ##2개의 pc백터 생성. 

# + [markdown] id="8ftekg4dS4cp"
# Notice how the axes are flipped.

# + [markdown] id="OyPKwF_8S4cq"
# Now let's look at the explained variance ratio:

# + id="lUHoSH0uS4cq" executionInfo={"elapsed": 751, "status": "ok", "timestamp": 1602145814746, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}, "user_tz": -540} outputId="731f77a5-6d02-481a-893e-17c3a5537679" colab={"base_uri": "https://localhost:8080/", "height": 36}
pca.explained_variance_ratio_ #2개의 변수가 0.98이상의 분산을 가짐. 

# + [markdown] id="xIMo6PxYS4cu"
# The first dimension explains 84.2% of the variance, while the second explains 14.6%.

# + [markdown] id="ycr0H6NgS4cu"
# By projecting down to 2D, we lost about 1.1% of the variance:

# + id="mbbIJZpYS4cv" executionInfo={"elapsed": 1196, "status": "ok", "timestamp": 1602145833635, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}, "user_tz": -540} outputId="57a91341-eb7f-4cca-a784-55568899041d" colab={"base_uri": "https://localhost:8080/", "height": 36}
1 - pca.explained_variance_ratio_.sum()

# + [markdown] id="MaRaHteXS4cy"
# Here is how to compute the explained variance ratio using the SVD approach (recall that `s` is the diagonal of the matrix `S`):

# + id="OUJR9hiKVdNV" executionInfo={"elapsed": 1537, "status": "ok", "timestamp": 1602145873590, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}, "user_tz": -540} outputId="f0e14326-4184-4c9f-a1b6-37f2e0d6e164" colab={"base_uri": "https://localhost:8080/", "height": 36}
s #s는 대각행렬을 갖는 벡터. 

# + id="qkRxWzd6S4cy" executionInfo={"elapsed": 1168, "status": "ok", "timestamp": 1602145837143, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}, "user_tz": -540} outputId="dbd0c457-fc93-4d06-8e74-43df3d2a55dc" colab={"base_uri": "https://localhost:8080/", "height": 36}
np.square(s) / np.square(s).sum() #각 pc별로 얼마만큼의 분산을 설명하는지 계산. 

# + [markdown] id="YWVJkOUrS4c1"
# Next, let's generate some nice figures! :)

# + [markdown] id="ZVTxA3u3S4c2"
# Utility class to draw 3D arrows (copied from http://stackoverflow.com/questions/11140163)

# + id="4zIOC4MuS4c3"
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)


# + [markdown] id="C1-ts7xKS4c7"
# Express the plane as a function of x and y.

# + id="qgqjoRvJS4c9"
axes = [-1.8, 1.8, -1.3, 1.3, -1.0, 1.0]

x1s = np.linspace(axes[0], axes[1], 10)
x2s = np.linspace(axes[2], axes[3], 10)
x1, x2 = np.meshgrid(x1s, x2s)

C = pca.components_
R = C.T.dot(C)
z = (R[0, 2] * x1 + R[1, 2] * x2) / (1 - R[2, 2])

# + [markdown] id="YJMQsnV2S4dD"
# Plot the 3D dataset, the plane and the projections on that plane.

# + id="8S5usACXS4dF" executionInfo={"elapsed": 2353, "status": "ok", "timestamp": 1602145922941, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}, "user_tz": -540} outputId="b37b9aaa-e9b6-4976-ecc8-b289ebc930cb" colab={"base_uri": "https://localhost:8080/", "height": 302}
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(6, 3.8))
ax = fig.add_subplot(111, projection='3d')

X3D_above = X[X[:, 2] > X3D_inv[:, 2]]
X3D_below = X[X[:, 2] <= X3D_inv[:, 2]]

ax.plot(X3D_below[:, 0], X3D_below[:, 1], X3D_below[:, 2], "bo", alpha=0.5)

ax.plot_surface(x1, x2, z, alpha=0.2, color="k")
np.linalg.norm(C, axis=0)
ax.add_artist(Arrow3D([0, C[0, 0]],[0, C[0, 1]],[0, C[0, 2]], mutation_scale=15, lw=1, arrowstyle="-|>", color="k"))
ax.add_artist(Arrow3D([0, C[1, 0]],[0, C[1, 1]],[0, C[1, 2]], mutation_scale=15, lw=1, arrowstyle="-|>", color="k"))
ax.plot([0], [0], [0], "k.")

for i in range(m):
    if X[i, 2] > X3D_inv[i, 2]:
        ax.plot([X[i][0], X3D_inv[i][0]], [X[i][1], X3D_inv[i][1]], [X[i][2], X3D_inv[i][2]], "k-")
    else:
        ax.plot([X[i][0], X3D_inv[i][0]], [X[i][1], X3D_inv[i][1]], [X[i][2], X3D_inv[i][2]], "k-", color="#505050")
    
ax.plot(X3D_inv[:, 0], X3D_inv[:, 1], X3D_inv[:, 2], "k+")
ax.plot(X3D_inv[:, 0], X3D_inv[:, 1], X3D_inv[:, 2], "k.")
ax.plot(X3D_above[:, 0], X3D_above[:, 1], X3D_above[:, 2], "bo")
ax.set_xlabel("$x_1$", fontsize=18, labelpad=10)
ax.set_ylabel("$x_2$", fontsize=18, labelpad=10)
ax.set_zlabel("$x_3$", fontsize=18, labelpad=10)
ax.set_xlim(axes[0:2])
ax.set_ylim(axes[2:4])
ax.set_zlim(axes[4:6])

# Note: If you are using Matplotlib 3.0.0, it has a bug and does not
# display 3D graphs properly.
# See https://github.com/matplotlib/matplotlib/issues/12239
# You should upgrade to a later version. If you cannot, then you can
# use the following workaround before displaying each 3D graph:
# for spine in ax.spines.values():
#     spine.set_visible(False)

save_fig("dataset_3d_plot")
plt.show()

# + id="ErKTDe_AS4dL" executionInfo={"elapsed": 1448, "status": "ok", "timestamp": 1602145927373, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}, "user_tz": -540} outputId="57d22637-8704-4386-f528-bbdd1d82ee4f" colab={"base_uri": "https://localhost:8080/", "height": 316}
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')

ax.plot(X2D[:, 0], X2D[:, 1], "k+")
ax.plot(X2D[:, 0], X2D[:, 1], "k.")
ax.plot([0], [0], "ko")
ax.arrow(0, 0, 0, 1, head_width=0.05, length_includes_head=True, head_length=0.1, fc='k', ec='k')
ax.arrow(0, 0, 1, 0, head_width=0.05, length_includes_head=True, head_length=0.1, fc='k', ec='k')
ax.set_xlabel("$z_1$", fontsize=18)
ax.set_ylabel("$z_2$", fontsize=18, rotation=0)
ax.axis([-1.5, 1.3, -1.2, 1.2])
ax.grid(True)
save_fig("dataset_2d_plot")

# + [markdown] id="-wxEGfbfS4dQ"
# # Manifold learning
# Swiss roll:

# + id="mR86SJiCS4dR"
from sklearn.datasets import make_swiss_roll
X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)

# + id="MQOhwKamS4dV" outputId="66583461-7dec-4eea-caa0-c01a87347fe6"
axes = [-11.5, 14, -2, 23, -12, 15]

fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=t, cmap=plt.cm.hot)
ax.view_init(10, -70)
ax.set_xlabel("$x_1$", fontsize=18)
ax.set_ylabel("$x_2$", fontsize=18)
ax.set_zlabel("$x_3$", fontsize=18)
ax.set_xlim(axes[0:2])
ax.set_ylim(axes[2:4])
ax.set_zlim(axes[4:6])

save_fig("swiss_roll_plot")
plt.show()

# + id="C1ClEgGYS4dY" outputId="0017c51f-e6c2-4402-cd17-90dd73e82eab"
plt.figure(figsize=(11, 4))

plt.subplot(121)
plt.scatter(X[:, 0], X[:, 1], c=t, cmap=plt.cm.hot)
plt.axis(axes[:4])
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$x_2$", fontsize=18, rotation=0)
plt.grid(True)

plt.subplot(122)
plt.scatter(t, X[:, 1], c=t, cmap=plt.cm.hot)
plt.axis([4, 15, axes[2], axes[3]])
plt.xlabel("$z_1$", fontsize=18)
plt.grid(True)

save_fig("squished_swiss_roll_plot")
plt.show()

# + id="DdPIDBCPS4db" outputId="4918d972-662c-493d-b6eb-312ccea8249e"
from matplotlib import gridspec

axes = [-11.5, 14, -2, 23, -12, 15]

x2s = np.linspace(axes[2], axes[3], 10)
x3s = np.linspace(axes[4], axes[5], 10)
x2, x3 = np.meshgrid(x2s, x3s)

fig = plt.figure(figsize=(6, 5))
ax = plt.subplot(111, projection='3d')

positive_class = X[:, 0] > 5
X_pos = X[positive_class]
X_neg = X[~positive_class]
ax.view_init(10, -70)
ax.plot(X_neg[:, 0], X_neg[:, 1], X_neg[:, 2], "y^")
ax.plot_wireframe(5, x2, x3, alpha=0.5)
ax.plot(X_pos[:, 0], X_pos[:, 1], X_pos[:, 2], "gs")
ax.set_xlabel("$x_1$", fontsize=18)
ax.set_ylabel("$x_2$", fontsize=18)
ax.set_zlabel("$x_3$", fontsize=18)
ax.set_xlim(axes[0:2])
ax.set_ylim(axes[2:4])
ax.set_zlim(axes[4:6])

save_fig("manifold_decision_boundary_plot1")
plt.show()

fig = plt.figure(figsize=(5, 4))
ax = plt.subplot(111)

plt.plot(t[positive_class], X[positive_class, 1], "gs")
plt.plot(t[~positive_class], X[~positive_class, 1], "y^")
plt.axis([4, 15, axes[2], axes[3]])
plt.xlabel("$z_1$", fontsize=18)
plt.ylabel("$z_2$", fontsize=18, rotation=0)
plt.grid(True)

save_fig("manifold_decision_boundary_plot2")
plt.show()

fig = plt.figure(figsize=(6, 5))
ax = plt.subplot(111, projection='3d')

positive_class = 2 * (t[:] - 4) > X[:, 1]
X_pos = X[positive_class]
X_neg = X[~positive_class]
ax.view_init(10, -70)
ax.plot(X_neg[:, 0], X_neg[:, 1], X_neg[:, 2], "y^")
ax.plot(X_pos[:, 0], X_pos[:, 1], X_pos[:, 2], "gs")
ax.set_xlabel("$x_1$", fontsize=18)
ax.set_ylabel("$x_2$", fontsize=18)
ax.set_zlabel("$x_3$", fontsize=18)
ax.set_xlim(axes[0:2])
ax.set_ylim(axes[2:4])
ax.set_zlim(axes[4:6])

save_fig("manifold_decision_boundary_plot3")
plt.show()

fig = plt.figure(figsize=(5, 4))
ax = plt.subplot(111)

plt.plot(t[positive_class], X[positive_class, 1], "gs")
plt.plot(t[~positive_class], X[~positive_class, 1], "y^")
plt.plot([4, 15], [0, 22], "b-", linewidth=2)
plt.axis([4, 15, axes[2], axes[3]])
plt.xlabel("$z_1$", fontsize=18)
plt.ylabel("$z_2$", fontsize=18, rotation=0)
plt.grid(True)

save_fig("manifold_decision_boundary_plot4")
plt.show()

# + [markdown] id="gvxBIRyPS4df"
# # PCA

# + id="hrP1gAv6S4dg" outputId="9c4b62f3-f163-4c3b-bfe9-7b62ef484140"
angle = np.pi / 5
stretch = 5
m = 200

np.random.seed(3)
X = np.random.randn(m, 2) / 10
X = X.dot(np.array([[stretch, 0],[0, 1]])) # stretch
X = X.dot([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]) # rotate

u1 = np.array([np.cos(angle), np.sin(angle)])
u2 = np.array([np.cos(angle - 2 * np.pi/6), np.sin(angle - 2 * np.pi/6)])
u3 = np.array([np.cos(angle - np.pi/2), np.sin(angle - np.pi/2)])

X_proj1 = X.dot(u1.reshape(-1, 1))
X_proj2 = X.dot(u2.reshape(-1, 1))
X_proj3 = X.dot(u3.reshape(-1, 1))

plt.figure(figsize=(8,4))
plt.subplot2grid((3,2), (0, 0), rowspan=3)
plt.plot([-1.4, 1.4], [-1.4*u1[1]/u1[0], 1.4*u1[1]/u1[0]], "k-", linewidth=1)
plt.plot([-1.4, 1.4], [-1.4*u2[1]/u2[0], 1.4*u2[1]/u2[0]], "k--", linewidth=1)
plt.plot([-1.4, 1.4], [-1.4*u3[1]/u3[0], 1.4*u3[1]/u3[0]], "k:", linewidth=2)
plt.plot(X[:, 0], X[:, 1], "bo", alpha=0.5)
plt.axis([-1.4, 1.4, -1.4, 1.4])
plt.arrow(0, 0, u1[0], u1[1], head_width=0.1, linewidth=5, length_includes_head=True, head_length=0.1, fc='k', ec='k')
plt.arrow(0, 0, u3[0], u3[1], head_width=0.1, linewidth=5, length_includes_head=True, head_length=0.1, fc='k', ec='k')
plt.text(u1[0] + 0.1, u1[1] - 0.05, r"$\mathbf{c_1}$", fontsize=22)
plt.text(u3[0] + 0.1, u3[1], r"$\mathbf{c_2}$", fontsize=22)
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$x_2$", fontsize=18, rotation=0)
plt.grid(True)

plt.subplot2grid((3,2), (0, 1))
plt.plot([-2, 2], [0, 0], "k-", linewidth=1)
plt.plot(X_proj1[:, 0], np.zeros(m), "bo", alpha=0.3)
plt.gca().get_yaxis().set_ticks([])
plt.gca().get_xaxis().set_ticklabels([])
plt.axis([-2, 2, -1, 1])
plt.grid(True)

plt.subplot2grid((3,2), (1, 1))
plt.plot([-2, 2], [0, 0], "k--", linewidth=1)
plt.plot(X_proj2[:, 0], np.zeros(m), "bo", alpha=0.3)
plt.gca().get_yaxis().set_ticks([])
plt.gca().get_xaxis().set_ticklabels([])
plt.axis([-2, 2, -1, 1])
plt.grid(True)

plt.subplot2grid((3,2), (2, 1))
plt.plot([-2, 2], [0, 0], "k:", linewidth=2)
plt.plot(X_proj3[:, 0], np.zeros(m), "bo", alpha=0.3)
plt.gca().get_yaxis().set_ticks([])
plt.axis([-2, 2, -1, 1])
plt.xlabel("$z_1$", fontsize=18)
plt.grid(True)

save_fig("pca_best_projection_plot")
plt.show()

# + [markdown] id="tIYRXqs7S4dk"
# # MNIST compression

# + id="HXIDOnjeS4dl"
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1)
mnist.target = mnist.target.astype(np.uint8)

# + id="EONEz9JpS4do"
from sklearn.model_selection import train_test_split

X = mnist["data"]
y = mnist["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y)

# + id="X0pPWApZS4dr"
pca = PCA()
pca.fit(X_train)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1

# + id="WTI-gT-DS4du" outputId="3eb4a2c4-77cc-4ca0-de70-80d48e80009f"
d

# + id="AZk_WtubS4d0" outputId="e6fbc657-ef11-40c2-b9d6-412d446eebd1"
plt.figure(figsize=(6,4))
plt.plot(cumsum, linewidth=3)
plt.axis([0, 400, 0, 1])
plt.xlabel("Dimensions")
plt.ylabel("Explained Variance")
plt.plot([d, d], [0, 0.95], "k:")
plt.plot([0, d], [0.95, 0.95], "k:")
plt.plot(d, 0.95, "ko")
plt.annotate("Elbow", xy=(65, 0.85), xytext=(70, 0.7),
             arrowprops=dict(arrowstyle="->"), fontsize=16)
plt.grid(True)
save_fig("explained_variance_plot")
plt.show()

# + id="OIF0PIv9S4d4"
pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X_train)

# + id="APlCzX6zS4d9" outputId="9830f5c0-6621-46c5-cc08-11dc3a1da282"
pca.n_components_

# + id="_wrcaNWYS4eC" outputId="58b34c21-eaff-4db5-a8d3-2096ba2c185f"
np.sum(pca.explained_variance_ratio_)

# + id="h37zCDltS4eH"
pca = PCA(n_components = 154)
X_reduced = pca.fit_transform(X_train)
X_recovered = pca.inverse_transform(X_reduced)


# + id="NEFng73NS4eL"
def plot_digits(instances, images_per_row=5, **options):
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


# + id="DoBwSfXcS4eQ" outputId="b230d22f-ecd7-4ce1-a265-c9ed99222071"
plt.figure(figsize=(7, 4))
plt.subplot(121)
plot_digits(X_train[::2100])
plt.title("Original", fontsize=16)
plt.subplot(122)
plot_digits(X_recovered[::2100])
plt.title("Compressed", fontsize=16)

save_fig("mnist_compression_plot")

# + id="tDUfMUACS4eU"
X_reduced_pca = X_reduced

# + [markdown] id="Z2ok84E9S4eY"
# ## Incremental PCA

# + id="jup_9r6iS4eY" outputId="407a48f0-ece5-4d35-f49e-e95629f0f209"
from sklearn.decomposition import IncrementalPCA

n_batches = 100
inc_pca = IncrementalPCA(n_components=154)
for X_batch in np.array_split(X_train, n_batches):
    print(".", end="") # not shown in the book
    inc_pca.partial_fit(X_batch)

X_reduced = inc_pca.transform(X_train)

# + id="8p9cVEKMS4eb"
X_recovered_inc_pca = inc_pca.inverse_transform(X_reduced)

# + id="6-_rGQexS4ef" outputId="f1ab2631-f9b6-448b-c533-717497f50fe7"
plt.figure(figsize=(7, 4))
plt.subplot(121)
plot_digits(X_train[::2100])
plt.subplot(122)
plot_digits(X_recovered_inc_pca[::2100])
plt.tight_layout()

# + id="gWZNLzn-S4ej"
X_reduced_inc_pca = X_reduced

# + [markdown] id="W6TQVbcDS4em"
# Let's compare the results of transforming MNIST using regular PCA and incremental PCA. First, the means are equal: 

# + id="XgTC6Y7eS4em" outputId="84c6ce25-b080-4c23-c7a1-6b5025da0fc2"
np.allclose(pca.mean_, inc_pca.mean_)

# + [markdown] id="lhTMn3NpS4ev"
# But the results are not exactly identical. Incremental PCA gives a very good approximate solution, but it's not perfect:

# + id="Qcu-gqftS4ew" outputId="bd99791c-a685-44f8-99c5-a676228963d3"
np.allclose(X_reduced_pca, X_reduced_inc_pca)

# + [markdown] id="xTAvOS9SS4e0"
# ### Using `memmap()`

# + [markdown] id="2UvEpP8BS4e1"
# Let's create the `memmap()` structure and copy the MNIST data into it. This would typically be done by a first program:

# + id="UgS1J9A3S4e2"
filename = "my_mnist.data"
m, n = X_train.shape

X_mm = np.memmap(filename, dtype='float32', mode='write', shape=(m, n))
X_mm[:] = X_train

# + [markdown] id="llcO0u-8S4e5"
# Now deleting the `memmap()` object will trigger its Python finalizer, which ensures that the data is saved to disk.

# + id="uIH3E7RLS4e6"
del X_mm

# + [markdown] id="rLMXwkqzS4fA"
# Next, another program would load the data and use it for training:

# + id="XsOQSH4vS4fB" outputId="819e5215-a4eb-4dc6-b08b-a3031fd1dadf"
X_mm = np.memmap(filename, dtype="float32", mode="readonly", shape=(m, n))

batch_size = m // n_batches
inc_pca = IncrementalPCA(n_components=154, batch_size=batch_size)
inc_pca.fit(X_mm)

# + id="z_b2ZkfMS4fE"
rnd_pca = PCA(n_components=154, svd_solver="randomized", random_state=42)
X_reduced = rnd_pca.fit_transform(X_train)

# + [markdown] id="hno0_jf9S4fG"
# ## Time complexity

# + [markdown] id="3FFIURxkS4fH"
# Let's time regular PCA against Incremental PCA and Randomized PCA, for various number of principal components:

# + id="ouylv6egS4fH" outputId="ac92203a-da39-4e37-abc4-43207a1a5135"
import time

for n_components in (2, 10, 154):
    print("n_components =", n_components)
    regular_pca = PCA(n_components=n_components)
    inc_pca = IncrementalPCA(n_components=n_components, batch_size=500)
    rnd_pca = PCA(n_components=n_components, random_state=42, svd_solver="randomized")

    for pca in (regular_pca, inc_pca, rnd_pca):
        t1 = time.time()
        pca.fit(X_train)
        t2 = time.time()
        print("    {}: {:.1f} seconds".format(pca.__class__.__name__, t2 - t1))

# + [markdown] id="GEahD0YCS4fL"
# Now let's compare PCA and Randomized PCA for datasets of different sizes (number of instances):

# + id="xlYHSHF0S4fL" outputId="d5e21129-7499-48f9-e6f4-50860f793ebe"
times_rpca = []
times_pca = []
sizes = [1000, 10000, 20000, 30000, 40000, 50000, 70000, 100000, 200000, 500000]
for n_samples in sizes:
    X = np.random.randn(n_samples, 5)
    pca = PCA(n_components = 2, svd_solver="randomized", random_state=42)
    t1 = time.time()
    pca.fit(X)
    t2 = time.time()
    times_rpca.append(t2 - t1)
    pca = PCA(n_components = 2)
    t1 = time.time()
    pca.fit(X)
    t2 = time.time()
    times_pca.append(t2 - t1)

plt.plot(sizes, times_rpca, "b-o", label="RPCA")
plt.plot(sizes, times_pca, "r-s", label="PCA")
plt.xlabel("n_samples")
plt.ylabel("Training time")
plt.legend(loc="upper left")
plt.title("PCA and Randomized PCA time complexity ")

# + [markdown] id="YhnLJ-b8S4fQ"
# And now let's compare their performance on datasets of 2,000 instances with various numbers of features:

# + id="SR7tM5toS4fQ" outputId="33af7e01-b7bf-40fd-e0d6-79f1be4294fd"
times_rpca = []
times_pca = []
sizes = [1000, 2000, 3000, 4000, 5000, 6000]
for n_features in sizes:
    X = np.random.randn(2000, n_features)
    pca = PCA(n_components = 2, random_state=42, svd_solver="randomized")
    t1 = time.time()
    pca.fit(X)
    t2 = time.time()
    times_rpca.append(t2 - t1)
    pca = PCA(n_components = 2)
    t1 = time.time()
    pca.fit(X)
    t2 = time.time()
    times_pca.append(t2 - t1)

plt.plot(sizes, times_rpca, "b-o", label="RPCA")
plt.plot(sizes, times_pca, "r-s", label="PCA")
plt.xlabel("n_features")
plt.ylabel("Training time")
plt.legend(loc="upper left")
plt.title("PCA and Randomized PCA time complexity ")

# + [markdown] id="ZfVPFKz4S4fT"
# # Kernel PCA

# + id="nETJZBkqS4fU"
X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)

# + id="ru0B1jTWS4fW"
from sklearn.decomposition import KernelPCA

rbf_pca = KernelPCA(n_components = 2, kernel="rbf", gamma=0.04)
X_reduced = rbf_pca.fit_transform(X)

# + id="PNVwXhjVS4fZ" outputId="2263cee0-ad2c-4160-a65c-9c4dfc6914bf"
from sklearn.decomposition import KernelPCA

lin_pca = KernelPCA(n_components = 2, kernel="linear", fit_inverse_transform=True)
rbf_pca = KernelPCA(n_components = 2, kernel="rbf", gamma=0.0433, fit_inverse_transform=True)
sig_pca = KernelPCA(n_components = 2, kernel="sigmoid", gamma=0.001, coef0=1, fit_inverse_transform=True)

y = t > 6.9

plt.figure(figsize=(11, 4))
for subplot, pca, title in ((131, lin_pca, "Linear kernel"), (132, rbf_pca, "RBF kernel, $\gamma=0.04$"), (133, sig_pca, "Sigmoid kernel, $\gamma=10^{-3}, r=1$")):
    X_reduced = pca.fit_transform(X)
    if subplot == 132:
        X_reduced_rbf = X_reduced
    
    plt.subplot(subplot)
    #plt.plot(X_reduced[y, 0], X_reduced[y, 1], "gs")
    #plt.plot(X_reduced[~y, 0], X_reduced[~y, 1], "y^")
    plt.title(title, fontsize=14)
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t, cmap=plt.cm.hot)
    plt.xlabel("$z_1$", fontsize=18)
    if subplot == 131:
        plt.ylabel("$z_2$", fontsize=18, rotation=0)
    plt.grid(True)

save_fig("kernel_pca_plot")
plt.show()

# + id="TJKYWTriS4fc" outputId="8127e0b5-2636-4805-c01a-624859e2e8ed"
plt.figure(figsize=(6, 5))

X_inverse = rbf_pca.inverse_transform(X_reduced_rbf)

ax = plt.subplot(111, projection='3d')
ax.view_init(10, -70)
ax.scatter(X_inverse[:, 0], X_inverse[:, 1], X_inverse[:, 2], c=t, cmap=plt.cm.hot, marker="x")
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_zlabel("")
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])

save_fig("preimage_plot", tight_layout=False)
plt.show()

# + id="N44_mlbmS4fe" outputId="24863a7f-71f5-4ab6-db40-9d357256bb07"
X_reduced = rbf_pca.fit_transform(X)

plt.figure(figsize=(11, 4))
plt.subplot(132)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t, cmap=plt.cm.hot, marker="x")
plt.xlabel("$z_1$", fontsize=18)
plt.ylabel("$z_2$", fontsize=18, rotation=0)
plt.grid(True)

# + id="ui_iRfeXS4ff" outputId="6b21e7ad-b4fe-4ff9-8879-a274b56f0198"
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

clf = Pipeline([
        ("kpca", KernelPCA(n_components=2)),
        ("log_reg", LogisticRegression(solver="lbfgs"))
    ])

param_grid = [{
        "kpca__gamma": np.linspace(0.03, 0.05, 10),
        "kpca__kernel": ["rbf", "sigmoid"]
    }]

grid_search = GridSearchCV(clf, param_grid, cv=3)
grid_search.fit(X, y)

# + id="6cezbLT9S4fi" outputId="315af1eb-b685-4310-b9c2-a3a1eb0eef48"
print(grid_search.best_params_)

# + id="ELWM0GLYS4fm"
rbf_pca = KernelPCA(n_components = 2, kernel="rbf", gamma=0.0433,
                    fit_inverse_transform=True)
X_reduced = rbf_pca.fit_transform(X)
X_preimage = rbf_pca.inverse_transform(X_reduced)

# + id="5PF_9WP_S4fo" outputId="c7c4271c-d33a-4fbc-c23b-581f89eb1323"
from sklearn.metrics import mean_squared_error

mean_squared_error(X, X_preimage)

# + [markdown] id="yYWZlQXIS4fs"
# # LLE

# + id="xdIIpwtrS4fv"
X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=41)

# + id="1WQiBJ3RS4fx"
from sklearn.manifold import LocallyLinearEmbedding

lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=42)
X_reduced = lle.fit_transform(X)

# + id="9F1UZQhyS4f0" outputId="84279094-8702-4bc9-8d90-7db2e01c39ec"
plt.title("Unrolled swiss roll using LLE", fontsize=14)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t, cmap=plt.cm.hot)
plt.xlabel("$z_1$", fontsize=18)
plt.ylabel("$z_2$", fontsize=18)
plt.axis([-0.065, 0.055, -0.1, 0.12])
plt.grid(True)

save_fig("lle_unrolling_plot")
plt.show()

# + [markdown] id="pV5nsma7S4f2"
# # MDS, Isomap and t-SNE

# + id="EEKLNuFCS4f2"
from sklearn.manifold import MDS

mds = MDS(n_components=2, random_state=42)
X_reduced_mds = mds.fit_transform(X)

# + id="zaNWAHB1S4f4"
from sklearn.manifold import Isomap

isomap = Isomap(n_components=2)
X_reduced_isomap = isomap.fit_transform(X)

# + id="kRNyl1IYS4f6"
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=42)
X_reduced_tsne = tsne.fit_transform(X)

# + id="_hldx-o8S4f9" outputId="933c1a9e-a94f-4993-f372-ac11375d57d1"
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components=2)
X_mnist = mnist["data"]
y_mnist = mnist["target"]
lda.fit(X_mnist, y_mnist)
X_reduced_lda = lda.transform(X_mnist)

# + id="eIBMxxStS4f_" outputId="3d8da7c0-6a5c-4c3a-e3ef-42c07d60b0c7"
titles = ["MDS", "Isomap", "t-SNE"]

plt.figure(figsize=(11,4))

for subplot, title, X_reduced in zip((131, 132, 133), titles,
                                     (X_reduced_mds, X_reduced_isomap, X_reduced_tsne)):
    plt.subplot(subplot)
    plt.title(title, fontsize=14)
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t, cmap=plt.cm.hot)
    plt.xlabel("$z_1$", fontsize=18)
    if subplot == 131:
        plt.ylabel("$z_2$", fontsize=18, rotation=0)
    plt.grid(True)

save_fig("other_dim_reduction_plot")
plt.show()

# + [markdown] id="YGzLjE9ZS4gC"
# # Exercise solutions

# + [markdown] id="IUmSB3osS4gD"
# ## 1. to 8.

# + [markdown] id="E1MyZPLqS4gD"
# See appendix A.

# + [markdown] id="otgHdhztS4gE"
# ## 9.

# + [markdown] id="69Sb_yKeS4gE"
# *Exercise: Load the MNIST dataset (introduced in chapter 3) and split it into a training set and a test set (take the first 60,000 instances for training, and the remaining 10,000 for testing).*

# + [markdown] id="K5MnQ_yKS4gF"
# The MNIST dataset was loaded earlier.

# + id="GOqwmkxQS4gF"
X_train = mnist['data'][:60000]
y_train = mnist['target'][:60000]

X_test = mnist['data'][60000:]
y_test = mnist['target'][60000:]

# + [markdown] id="ytvJXYoSS4gJ"
# *Exercise: Train a Random Forest classifier on the dataset and time how long it takes, then evaluate the resulting model on the test set.*

# + id="QytQrBEAS4gJ"
from sklearn.ensemble import RandomForestClassifier

rnd_clf = RandomForestClassifier(n_estimators=100, random_state=42)

# + id="d_31pDOLS4gL"
import time

t0 = time.time()
rnd_clf.fit(X_train, y_train)
t1 = time.time()

# + id="VK2AqNWES4gN" outputId="6a9787ac-2317-436a-81e1-93a6af4212f5"
print("Training took {:.2f}s".format(t1 - t0))

# + id="-lyAtqU0S4gP" outputId="a86be701-91ee-4a91-fe80-e685e1b8dfbc"
from sklearn.metrics import accuracy_score

y_pred = rnd_clf.predict(X_test)
accuracy_score(y_test, y_pred)

# + [markdown] id="j7-HM6AeS4gS"
# *Exercise: Next, use PCA to reduce the dataset's dimensionality, with an explained variance ratio of 95%.*

# + id="oM5XXzq8S4gS"
from sklearn.decomposition import PCA

pca = PCA(n_components=0.95)
X_train_reduced = pca.fit_transform(X_train)

# + [markdown] id="uVQxKHW4S4gV"
# *Exercise: Train a new Random Forest classifier on the reduced dataset and see how long it takes. Was training much faster?*

# + id="6i4ibpmhS4gV"
rnd_clf2 = RandomForestClassifier(n_estimators=100, random_state=42)
t0 = time.time()
rnd_clf2.fit(X_train_reduced, y_train)
t1 = time.time()

# + id="Jl9k91gdS4gX" outputId="baa90a81-1d8e-458d-d0f0-222127c8652a"
print("Training took {:.2f}s".format(t1 - t0))

# + [markdown] id="g91bDd_lS4gY"
# Oh no! Training is actually more than twice slower now! How can that be? Well, as we saw in this chapter, dimensionality reduction does not always lead to faster training time: it depends on the dataset, the model and the training algorithm. See figure 8-6 (the `manifold_decision_boundary_plot*` plots above). If you try a softmax classifier instead of a random forest classifier, you will find that training time is reduced by a factor of 3 when using PCA. Actually, we will do this in a second, but first let's check the precision of the new random forest classifier.

# + [markdown] id="V-S3S0hYS4gZ"
# *Exercise: Next evaluate the classifier on the test set: how does it compare to the previous classifier?*

# + id="dj18F74aS4gZ" outputId="804de73d-6632-458a-8ffe-07c117759b17"
X_test_reduced = pca.transform(X_test)

y_pred = rnd_clf2.predict(X_test_reduced)
accuracy_score(y_test, y_pred)

# + [markdown] id="9mrHVtF_S4gb"
# It is common for performance to drop slightly when reducing dimensionality, because we do lose some useful signal in the process. However, the performance drop is rather severe in this case. So PCA really did not help: it slowed down training and reduced performance. :(
#
# Let's see if it helps when using softmax regression:

# + id="l5Z2cjsSS4gb" outputId="b32cd6d3-d207-466e-989c-18da3bdf575a"
from sklearn.linear_model import LogisticRegression

log_clf = LogisticRegression(multi_class="multinomial", solver="lbfgs", random_state=42)
t0 = time.time()
log_clf.fit(X_train, y_train)
t1 = time.time()

# + id="5fqIf4AHS4gd" outputId="28dcaec9-a596-4f3a-e5ad-2b5c50207cea"
print("Training took {:.2f}s".format(t1 - t0))

# + id="PFg8scPgS4gg" outputId="620eeeec-f3e6-40ad-8b5c-dc662618978c"
y_pred = log_clf.predict(X_test)
accuracy_score(y_test, y_pred)

# + [markdown] id="ljPLz_7-S4gi"
# Okay, so softmax regression takes much longer to train on this dataset than the random forest classifier, plus it performs worse on the test set. But that's not what we are interested in right now, we want to see how much PCA can help softmax regression. Let's train the softmax regression model using the reduced dataset:

# + id="MYRDMvLWS4gi" outputId="1a52ed8c-9742-4e5a-f091-32b8c6ba1826"
log_clf2 = LogisticRegression(multi_class="multinomial", solver="lbfgs", random_state=42)
t0 = time.time()
log_clf2.fit(X_train_reduced, y_train)
t1 = time.time()

# + id="rMFY4ap4S4gj" outputId="1290af00-7723-442f-f935-089a071d9206"
print("Training took {:.2f}s".format(t1 - t0))

# + [markdown] id="rdB3ZPcaS4go"
# Nice! Reducing dimensionality led to a 4× speedup. :)  Let's check the model's accuracy:

# + id="klaH_xQdS4go" outputId="801eb6eb-5067-42dd-811f-93e32e1c6166"
y_pred = log_clf2.predict(X_test_reduced)
accuracy_score(y_test, y_pred)

# + [markdown] id="RBCkFU1eS4gq"
# A very slight drop in performance, which might be a reasonable price to pay for a 4× speedup, depending on the application.

# + [markdown] id="8QnjNHJ6S4gr"
# So there you have it: PCA can give you a formidable speedup... but not always!

# + [markdown] id="0zQMJhKvS4gr"
# ## 10.

# + [markdown] id="3z7IKkKIS4gs"
# *Exercise: Use t-SNE to reduce the MNIST dataset down to two dimensions and plot the result using Matplotlib. You can use a scatterplot using 10 different colors to represent each image's target class.*

# + [markdown] id="uJvbOrQAS4gt"
# The MNIST dataset was loaded above.

# + [markdown] id="ILNUDmiES4gt"
# Dimensionality reduction on the full 60,000 images takes a very long time, so let's only do this on a random subset of 10,000 images:

# + id="Jl-OqoL5S4gu"
np.random.seed(42)

m = 10000
idx = np.random.permutation(60000)[:m]

X = mnist['data'][idx]
y = mnist['target'][idx]

# + [markdown] id="apx3V10kS4gz"
# Now let's use t-SNE to reduce dimensionality down to 2D so we can plot the dataset:

# + id="aA9XNF2lS4gz"
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=42)
X_reduced = tsne.fit_transform(X)

# + [markdown] id="QgKHkwlYS4g2"
# Now let's use Matplotlib's `scatter()` function to plot a scatterplot, using a different color for each digit:

# + id="7umco05YS4g2" outputId="fd34be53-32b9-43f7-c5da-64a46c512327"
plt.figure(figsize=(13,10))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap="jet")
plt.axis('off')
plt.colorbar()
plt.show()

# + [markdown] id="-KjM_4bCS4g4"
# Isn't this just beautiful? :) This plot tells us which numbers are easily distinguishable from the others (e.g., 0s, 6s, and most 8s are rather well separated clusters), and it also tells us which numbers are often hard to distinguish (e.g., 4s and 9s, 5s and 3s, and so on).

# + [markdown] id="4mYKnPn7S4g5"
# Let's focus on digits 3 and 5, which seem to overlap a lot.

# + id="WstQYE_TS4g5" outputId="56665945-7501-4eca-8362-b124c08b623f"
plt.figure(figsize=(9,9))
cmap = mpl.cm.get_cmap("jet")
for digit in (2, 3, 5):
    plt.scatter(X_reduced[y == digit, 0], X_reduced[y == digit, 1], c=[cmap(digit / 9)])
plt.axis('off')
plt.show()

# + [markdown] id="ew8V_DzLS4g6"
# Let's see if we can produce a nicer image by running t-SNE on these 3 digits:

# + id="R0P0er_qS4g7"
idx = (y == 2) | (y == 3) | (y == 5) 
X_subset = X[idx]
y_subset = y[idx]

tsne_subset = TSNE(n_components=2, random_state=42)
X_subset_reduced = tsne_subset.fit_transform(X_subset)

# + id="_YVlgHdSS4g9" outputId="f4776535-97e8-48e5-8990-3d8828d79bb7"
plt.figure(figsize=(9,9))
for digit in (2, 3, 5):
    plt.scatter(X_subset_reduced[y_subset == digit, 0], X_subset_reduced[y_subset == digit, 1], c=[cmap(digit / 9)])
plt.axis('off')
plt.show()

# + [markdown] id="n5GU46C8S4g-"
# Much better, now the clusters have far less overlap. But some 3s are all over the place. Plus, there are two distinct clusters of 2s, and also two distinct clusters of 5s. It would be nice if we could visualize a few digits from each cluster, to understand why this is the case. Let's do that now. 

# + [markdown] id="KXfJUi4uS4g_"
# *Exercise: Alternatively, you can write colored digits at the location of each instance, or even plot scaled-down versions of the digit images themselves (if you plot all digits, the visualization will be too cluttered, so you should either draw a random sample or plot an instance only if no other instance has already been plotted at a close distance). You should get a nice visualization with well-separated clusters of digits.*

# + [markdown] id="8MK4Kg1SS4g_"
# Let's create a `plot_digits()` function that will draw a scatterplot (similar to the above scatterplots) plus write colored digits, with a minimum distance guaranteed between these digits. If the digit images are provided, they are plotted instead. This implementation was inspired from one of Scikit-Learn's excellent examples ([plot_lle_digits](http://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html), based on a different digit dataset).

# + id="k2AEEEXnS4g_"
from sklearn.preprocessing import MinMaxScaler
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

def plot_digits(X, y, min_distance=0.05, images=None, figsize=(13, 10)):
    # Let's scale the input features so that they range from 0 to 1
    X_normalized = MinMaxScaler().fit_transform(X)
    # Now we create the list of coordinates of the digits plotted so far.
    # We pretend that one is already plotted far away at the start, to
    # avoid `if` statements in the loop below
    neighbors = np.array([[10., 10.]])
    # The rest should be self-explanatory
    plt.figure(figsize=figsize)
    cmap = mpl.cm.get_cmap("jet")
    digits = np.unique(y)
    for digit in digits:
        plt.scatter(X_normalized[y == digit, 0], X_normalized[y == digit, 1], c=[cmap(digit / 9)])
    plt.axis("off")
    ax = plt.gcf().gca()  # get current axes in current figure
    for index, image_coord in enumerate(X_normalized):
        closest_distance = np.linalg.norm(np.array(neighbors) - image_coord, axis=1).min()
        if closest_distance > min_distance:
            neighbors = np.r_[neighbors, [image_coord]]
            if images is None:
                plt.text(image_coord[0], image_coord[1], str(int(y[index])),
                         color=cmap(y[index] / 9), fontdict={"weight": "bold", "size": 16})
            else:
                image = images[index].reshape(28, 28)
                imagebox = AnnotationBbox(OffsetImage(image, cmap="binary"), image_coord)
                ax.add_artist(imagebox)


# + [markdown] id="H9CMqmH5S4hB"
# Let's try it! First let's just write colored digits:

# + id="Uj_X0zphS4hB" outputId="7653a7c2-3ec5-4777-de7e-e247c1b5226c"
plot_digits(X_reduced, y)

# + [markdown] id="5jJu68O_S4hD"
# Well that's okay, but not that beautiful. Let's try with the digit images:

# + id="Wxs-HYwQS4hD" outputId="365d3fd0-6af7-4e5d-9f21-5e9b82e4603c"
plot_digits(X_reduced, y, images=X, figsize=(35, 25))

# + id="X9XJIqy1S4hF" outputId="41bfffcb-bf88-4b82-ecd1-d2538d4253d2"
plot_digits(X_subset_reduced, y_subset, images=X_subset, figsize=(22, 22))

# + [markdown] id="pWEUnPReS4hJ"
# *Exercise: Try using other dimensionality reduction algorithms such as PCA, LLE, or MDS and compare the resulting visualizations.*

# + [markdown] id="q_SoFcEFS4hJ"
# Let's start with PCA. We will also time how long it takes:

# + id="WibyvgWLS4hJ" outputId="8d374a11-5557-4efd-8fd5-f974784f2e64"
from sklearn.decomposition import PCA
import time

t0 = time.time()
X_pca_reduced = PCA(n_components=2, random_state=42).fit_transform(X)
t1 = time.time()
print("PCA took {:.1f}s.".format(t1 - t0))
plot_digits(X_pca_reduced, y)
plt.show()

# + [markdown] id="c_bPLzCmS4hN"
# Wow, PCA is blazingly fast! But although we do see a few clusters, there's way too much overlap. Let's try LLE:

# + id="EH7OTVooS4hO" outputId="8b8c8733-3500-423e-820c-61a08127ebeb"
from sklearn.manifold import LocallyLinearEmbedding

t0 = time.time()
X_lle_reduced = LocallyLinearEmbedding(n_components=2, random_state=42).fit_transform(X)
t1 = time.time()
print("LLE took {:.1f}s.".format(t1 - t0))
plot_digits(X_lle_reduced, y)
plt.show()

# + [markdown] id="NT4m62IcS4hP"
# That took a while, and the result does not look too good. Let's see what happens if we apply PCA first, preserving 95% of the variance:

# + id="Dy-od-z5S4hQ" outputId="7c32615f-d1a1-4da9-cc0b-a6c6576e457c"
from sklearn.pipeline import Pipeline

pca_lle = Pipeline([
    ("pca", PCA(n_components=0.95, random_state=42)),
    ("lle", LocallyLinearEmbedding(n_components=2, random_state=42)),
])
t0 = time.time()
X_pca_lle_reduced = pca_lle.fit_transform(X)
t1 = time.time()
print("PCA+LLE took {:.1f}s.".format(t1 - t0))
plot_digits(X_pca_lle_reduced, y)
plt.show()

# + [markdown] id="FcJMSjpJS4hR"
# The result is more or less the same, but this time it was almost 4× faster.

# + [markdown] id="cnmERb9rS4hS"
# Let's try MDS. It's much too long if we run it on 10,000 instances, so let's just try 2,000 for now:

# + id="uwv9KRCwS4hS" outputId="1f917d43-5d06-44ca-f607-a7c3686c9b69"
from sklearn.manifold import MDS

m = 2000
t0 = time.time()
X_mds_reduced = MDS(n_components=2, random_state=42).fit_transform(X[:m])
t1 = time.time()
print("MDS took {:.1f}s (on just 2,000 MNIST images instead of 10,000).".format(t1 - t0))
plot_digits(X_mds_reduced, y[:m])
plt.show()

# + [markdown] id="t2rydo5WS4hU"
# Meh. This does not look great, all clusters overlap too much. Let's try with PCA first, perhaps it will be faster?

# + id="j2CwVWXqS4hU" outputId="1e07bad7-ce98-4e51-e691-7b2d1877df2c"
from sklearn.pipeline import Pipeline

pca_mds = Pipeline([
    ("pca", PCA(n_components=0.95, random_state=42)),
    ("mds", MDS(n_components=2, random_state=42)),
])
t0 = time.time()
X_pca_mds_reduced = pca_mds.fit_transform(X[:2000])
t1 = time.time()
print("PCA+MDS took {:.1f}s (on 2,000 MNIST images).".format(t1 - t0))
plot_digits(X_pca_mds_reduced, y[:2000])
plt.show()

# + [markdown] id="Js8aJJSOS4hV"
# Same result, and no speedup: PCA did not help (or hurt).

# + [markdown] id="DADwlly5S4hW"
# Let's try LDA:

# + id="O35r8NVgS4hW" outputId="3bf190e3-a122-4ffe-99f0-3ecb76a88a2d"
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

t0 = time.time()
X_lda_reduced = LinearDiscriminantAnalysis(n_components=2).fit_transform(X, y)
t1 = time.time()
print("LDA took {:.1f}s.".format(t1 - t0))
plot_digits(X_lda_reduced, y, figsize=(12,12))
plt.show()

# + [markdown] id="o4c6ouH3S4hX"
# This one is very fast, and it looks nice at first, until you realize that several clusters overlap severely.

# + [markdown] id="R-fz0T_fS4hY"
# Well, it's pretty clear that t-SNE won this little competition, wouldn't you agree? We did not time it, so let's do that now:

# + id="OyCpPQwJS4hY" outputId="1dacd0dc-e193-493c-dd8f-5e710f633831"
from sklearn.manifold import TSNE

t0 = time.time()
X_tsne_reduced = TSNE(n_components=2, random_state=42).fit_transform(X)
t1 = time.time()
print("t-SNE took {:.1f}s.".format(t1 - t0))
plot_digits(X_tsne_reduced, y)
plt.show()

# + [markdown] id="ErwxBeXWS4ha"
# It's twice slower than LLE, but still much faster than MDS, and the result looks great. Let's see if a bit of PCA can speed it up:

# + id="iDgRH5JAS4ha" outputId="f3694be8-9ca1-4269-94a2-3300769aacb7"
pca_tsne = Pipeline([
    ("pca", PCA(n_components=0.95, random_state=42)),
    ("tsne", TSNE(n_components=2, random_state=42)),
])
t0 = time.time()
X_pca_tsne_reduced = pca_tsne.fit_transform(X)
t1 = time.time()
print("PCA+t-SNE took {:.1f}s.".format(t1 - t0))
plot_digits(X_pca_tsne_reduced, y)
plt.show()

# + [markdown] id="R9AOY5eJS4hb"
# Yes, PCA roughly gave us a 25% speedup, without damaging the result. We have a winner!

# + id="ipQGlnn5S4hc"

