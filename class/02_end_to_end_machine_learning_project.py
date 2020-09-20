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

# + [markdown] colab_type="text" id="view-in-github"
# <a href="https://colab.research.google.com/github/ChangRyeol/datamining2/blob/master/class/02_end_to_end_machine_learning_project.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] colab_type="text" id="HVCiaP2hY3YX"
# **Chapter 2 – End-to-end Machine Learning project**
#
# *Welcome to Machine Learning Housing Corp.! Your task is to predict median house values in Californian districts, given a number of features from these districts.*
#
# *This notebook contains all the sample code and solutions to the exercices in chapter 2.*

# + [markdown] colab_type="text" id="rGHScGGwY3Ya"
# <table align="left">
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/ageron/handson-ml2/blob/master/02_end_to_end_machine_learning_project.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
#   </td>
# </table>

# + colab={"base_uri": "https://localhost:8080/", "height": 55} colab_type="code" executionInfo={"elapsed": 18806, "status": "ok", "timestamp": 1600601898831, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}, "user_tz": -540} id="58CIyeRaTmH-" outputId="041ee3f5-89fa-4a49-b380-3e6437b94eb9"
from google.colab import drive # import drive from google colab

ROOT = "/content/drive"     # default location for the drive
print(ROOT)                 # print content of ROOT (Optional)

drive.mount(ROOT)           # we mount the google drive at /content/drive

# + colab={} colab_type="code" id="O-lvem_ZTmFu"



# + [markdown] colab_type="text" id="blpXbuK4wXtO"
# ##syn py and ipynb 

# + colab={"base_uri": "https://localhost:8080/", "height": 36} colab_type="code" executionInfo={"elapsed": 626, "status": "ok", "timestamp": 1600600413452, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}, "user_tz": -540} id="m2s0TWsH060Y" outputId="f57c0bae-754e-400c-ef7f-4c2638bde0b6"
# %pwd 

# + colab={"base_uri": "https://localhost:8080/", "height": 36} colab_type="code" executionInfo={"elapsed": 636, "status": "ok", "timestamp": 1600601921019, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}, "user_tz": -540} id="d74asNX3O4Wd" outputId="f2fc435c-089d-48df-cb8d-9b6d4d113893"
# %cd 'drive/My Drive/Colab Notebooks/datamining2' 

# + colab={"base_uri": "https://localhost:8080/", "height": 36} colab_type="code" executionInfo={"elapsed": 757, "status": "ok", "timestamp": 1600601960352, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}, "user_tz": -540} id="BIZ4AGpMT3av" outputId="06dcd3f4-31f4-4e1d-e5b9-a6541a8ea8f5"
# %pwd

# + colab={} colab_type="code" executionInfo={"elapsed": 615, "status": "ok", "timestamp": 1600602049122, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}, "user_tz": -540} id="d67rgbYjOQlx"
WORK_PATH = "drive/My Drive/Colab Notebooks/datamining2/class"

# + colab={"base_uri": "https://localhost:8080/", "height": 243} colab_type="code" executionInfo={"elapsed": 1114, "status": "ok", "timestamp": 1600602052804, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}, "user_tz": -540} id="uPdgdLuOOCP6" outputId="081b6771-2762-41bc-a88b-3cbdcb592bb0"
## Pair a notebook to a light script
# !jupytext --set-formats ipynb,py:light "{WORK_PATH}/"02_end_to_end_machine_learning_project.ipynb  


# + [markdown] colab_type="text" id="RtWzmxQJY3Ya"
# # Setup

# + [markdown] colab_type="text" id="a-wC2y6WY3Yb"
# First, let's import a few common modules, ensure MatplotLib plots figures inline and prepare a function to save the figures. We also check that Python 3.5 or later is installed (although Python 2.x may work, it is deprecated so we strongly recommend you use Python 3 instead), as well as Scikit-Learn ≥0.20.

# + colab={} colab_type="code" id="4H30cgVsY3Yc"
# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# Common imports
import numpy as np
import os

# To plot pretty figures
# %matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "end_to_end_project"
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

# + [markdown] colab_type="text" id="6YLmOZCbY3Yg"
# # Get the data

# + colab={} colab_type="code" id="GCK3u17_Y3Yh"
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


# + colab={} colab_type="code" id="AsQ4EnEJY3Yk"
fetch_housing_data() #정의한 함수 써서 데이터 불러옴. 

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="TwNlZZzickbj" outputId="5ecff436-6541-4293-da6a-86b360e0add8"
##파일 저장되었는지 확인 
HOUSING_PATH
os.listdir(HOUSING_PATH)

# + colab={} colab_type="code" id="Sq4B5SkNY3Yn"
import pandas as pd
#csv로 변환하여 pandas로 읽어오는 함수 정의 
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


# + colab={"base_uri": "https://localhost:8080/", "height": 204} colab_type="code" id="NL_aHCoaY3Yq" outputId="f81739dd-af48-4105-cbf6-0a84c59e308e"
#정의한 함수 사용 
housing = load_housing_data()
housing.head()

# + colab={"base_uri": "https://localhost:8080/", "height": 323} colab_type="code" id="uYeCCg7yY3Yt" outputId="664bc43a-7820-4cfc-bb7e-e190b20aba6b"
housing.info()  #ocean_proximity만 데이터 타입이 다르다. 

# + colab={"base_uri": "https://localhost:8080/", "height": 125} colab_type="code" id="DW1_dpmRY3Yx" outputId="5433b093-9ba6-4372-885b-be2b98c5ad7d"
housing["ocean_proximity"].value_counts() #카테고리별 갯수 

# + colab={"base_uri": "https://localhost:8080/", "height": 172} colab_type="code" id="snQeVxj6Y3Y1" outputId="f5de0704-0d84-40f9-f2b5-a1e19507f0ad"
housing.describe() 


# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="nOvwwmOMcfzX" outputId="c339ae18-3972-4072-99b2-53ee0008cbd5"
def crash_cours(dltr):
  result = 0
  temp = 1 
  for i in range(1,dltr):
    temp = temp * i 
    for k in range(i):
      result += temp 
  return result 
print (crash_cours(4))


# + colab={"base_uri": "https://localhost:8080/", "height": 71} colab_type="code" id="dcvk5KKOdpau" outputId="5275d778-06ab-4b1a-8878-4bd10f437e9b"
def y(x):
  global a 
  a = 4
  return 0 

def f(a):
  a = 3 
  print(a)
  return a 

a = 5 
f(a)
print(a)
y(a)
print(a)


# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="T9OM4CHUcP6P" outputId="4adefe5c-1d14-4363-8033-5b306f283af8"
import numpy as np
a = [[1,2],[2,1]]
b = [[4,1], [2,2]]
np.cross(a,b)

# + colab={"base_uri": "https://localhost:8080/", "height": 71} colab_type="code" id="uCBlA0VeeW-C" outputId="a175bb0b-8e7d-4622-d2d3-782894c79874"
a= np.array([[1, 2, 3], [4, 5, 6]])
a.transpose()

# + colab={"base_uri": "https://localhost:8080/", "height": 1000} colab_type="code" id="kkeAi0ZsY3Y3" outputId="de5d7a04-e47c-4d41-aee8-6d6750911e6a"
#히스토그램 그리기 => 분포확인 
# %matplotlib inline
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
save_fig("attribute_histogram_plots")
plt.show()

# + colab={} colab_type="code" id="qWXA02wBY3Y5"
# to make this notebook's output identical at every run
#씨드 설정 
np.random.seed(42)

# + colab={} colab_type="code" id="3PFxkP7nY3Y8"
import numpy as np
#train이랑 test셋 분리하는 함수 
# For illustration only. Sklearn has train_test_split()
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="pnU8S980Y3ZA" outputId="99506ad4-a4d7-46f4-ddac-2cedc6f0cdef"
train_set, test_set = split_train_test(housing, 0.2)
len(train_set)

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="0at2QnORY3ZD" outputId="e570eb53-fc32-406f-e1ad-f6d0050d7018"
len(test_set)

# + colab={} colab_type="code" id="dC7XkXz9Y3ZG"
from zlib import crc32

def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


# + [markdown] colab_type="text" id="O1So8cUeY3ZJ"
# The implementation of `test_set_check()` above works fine in both Python 2 and Python 3. In earlier releases, the following implementation was proposed, which supported any hash function, but was much slower and did not support Python 2:

# + colab={} colab_type="code" id="EdtfHgpkY3ZK"
import hashlib

def test_set_check(identifier, test_ratio, hash=hashlib.md5):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio


# + [markdown] colab_type="text" id="58_Xy-esY3ZM"
# If you want an implementation that supports any hash function and is compatible with both Python 2 and Python 3, here is one:

# + colab={} colab_type="code" id="B4YlBIaOY3ZM"
def test_set_check(identifier, test_ratio, hash=hashlib.md5):
    return bytearray(hash(np.int64(identifier)).digest())[-1] < 256 * test_ratio


# + colab={} colab_type="code" id="L4hTAuhvY3ZP"
#index를 추가해서 항상 같은 난수 생성되도록함. 
housing_with_id = housing.reset_index()   # adds an `index` column
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")

# + colab={} colab_type="code" id="vhpzeKQ2Y3ZS"
housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")

# + colab={"base_uri": "https://localhost:8080/", "height": 204} colab_type="code" id="Jnp8rLuFY3ZW" outputId="06ee6566-bd45-432d-f630-ddcd8e972e68"
test_set.head()

# + colab={} colab_type="code" id="8z-s9bCDY3ZY"
#sklearn을 활용해서 test train 나눔.
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

# + colab={"base_uri": "https://localhost:8080/", "height": 204} colab_type="code" id="WNUA5QJEY3Za" outputId="14f58569-5588-4656-e561-ba89364be988"
test_set.head()

# + colab={"base_uri": "https://localhost:8080/", "height": 286} colab_type="code" id="GqDTDpeDY3Zd" outputId="2d0eea9f-ba1d-4da8-e0df-0bff4e683065"
housing["median_income"].hist()

# + colab={} colab_type="code" id="8zyehR_WY3Zf"
#income을 범주화 시킴 
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

# + colab={"base_uri": "https://localhost:8080/", "height": 125} colab_type="code" id="HCcltH1AY3Zh" outputId="be13394a-dc0f-4cb9-ee60-90ca6794830f"
#범주별 카운트 
housing["income_cat"].value_counts()

# + colab={"base_uri": "https://localhost:8080/", "height": 286} colab_type="code" id="ZvVFOvQeY3Zk" outputId="a73a51c1-d630-4868-908f-a627ef8d2b3b"
housing["income_cat"].hist()

# + colab={} colab_type="code" id="MoyOZKbGY3Zn"
from sklearn.model_selection import StratifiedShuffleSplit

#for문이지만 여기 예제에서 n_splits =1이기때문에 한번만 돌아간다. 
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# + colab={"base_uri": "https://localhost:8080/", "height": 125} colab_type="code" id="TXYdZjboY3Zp" outputId="7da2ff68-0e63-46ed-a1bf-d28afb2b3421"
strat_test_set["income_cat"].value_counts() / len(strat_test_set)

# + colab={"base_uri": "https://localhost:8080/", "height": 125} colab_type="code" id="DFvyqgo7Y3Zr" outputId="28b8c270-88d6-415d-c3d5-518ed8d1f1e7"
housing["income_cat"].value_counts() / len(housing)


# + colab={} colab_type="code" id="nOO17VPaY3Zu"
#stratified 을 사용할떄 분포를 맞추면서 sampling이 된다.  
def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

compare_props = pd.DataFrame({
    "Overall": income_cat_proportions(housing),
    "Stratified": income_cat_proportions(strat_test_set),
    "Random": income_cat_proportions(test_set),
}).sort_index()
compare_props["Rand. %error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100
compare_props["Strat. %error"] = 100 * compare_props["Stratified"] / compare_props["Overall"] - 100

# + colab={"base_uri": "https://localhost:8080/", "height": 204} colab_type="code" id="xEBRkeHMY3Zw" outputId="8bd8ed86-c1e0-41e9-fe88-ee71662fff64"
compare_props

# + colab={} colab_type="code" id="N9UDKs51Y3Zy"
#income_cat를 드랍시키기. 
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

# + [markdown] colab_type="text" id="ijN3eeqcY3Z2"
# # Discover and visualize the data to gain insights

# + colab={"base_uri": "https://localhost:8080/"} colab_type="code" id="26kclPSnRhdj" outputId="99ce8194-2398-4b41-842a-d15c2f02321e"
#아래서 .copy함수를 쓰는이유
#
a = [1,2,3]
b = a  #### b에 a를 대입 
b 

# + colab={"base_uri": "https://localhost:8080/"} colab_type="code" id="0ucbMrQwRth0" outputId="6fdda9f7-8923-4c20-9cb8-0659417be43c"
b[1] = 0 
b 

# + colab={"base_uri": "https://localhost:8080/"} colab_type="code" id="ddgjC3P4RxWl" outputId="59b6f896-7ea5-4319-8582-e59a35d44d52"
a  ## b만 봐꿨는데 a까지 같이 봐뀐다. 메모리 참조하는 위치까지 같아졌기 때문이다. 따라서 그냥 대입이 아니라, b=a.copy()를 써서 변수에 할당된 값은 같지만, 참조하는 메모리 위치는 다르게 하는것이다. 

# + colab={} colab_type="code" id="EsVDZw6RY3Z2"
housing = strat_train_set.copy()

# + colab={"base_uri": "https://localhost:8080/"} colab_type="code" id="jotsBV5CY3Z5" outputId="05654b41-dc4a-4dcc-d173-25b8a3682c9e"
#plot 
housing.plot(kind="scatter", x="longitude", y="latitude")
save_fig("bad_visualization_plot")

# + colab={"base_uri": "https://localhost:8080/"} colab_type="code" id="mUp5mFm7Y3Z7" outputId="50059c0a-4618-497d-c6c8-c2ddfef0b0ba"
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
save_fig("better_visualization_plot")

# + [markdown] colab_type="text" id="uv2efPwOY3Z-"
# The argument `sharex=False` fixes a display bug (the x-axis values and legend were not displayed). This is a temporary fix (see: https://github.com/pandas-dev/pandas/issues/10611 ). Thanks to Wilmer Arellano for pointing it out.

# + colab={"base_uri": "https://localhost:8080/"} colab_type="code" id="Ybpl1GB_Y3Z-" outputId="f35088aa-c48c-4cf8-eebd-95709e54be1d"
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
    s=housing["population"]/100, label="population", figsize=(10,7),
    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
    sharex=False)
plt.legend()
save_fig("housing_prices_scatterplot")

# + colab={"base_uri": "https://localhost:8080/"} colab_type="code" id="Vrf_EobUY3aB" outputId="4ad0a17d-f89e-40b3-9b9d-77ef6dd8e630"
# Download the California image
images_path = os.path.join(PROJECT_ROOT_DIR, "images", "end_to_end_project")
os.makedirs(images_path, exist_ok=True)
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
filename = "california.png"
print("Downloading", filename)
url = DOWNLOAD_ROOT + "images/end_to_end_project/" + filename
urllib.request.urlretrieve(url, os.path.join(images_path, filename))

# + colab={"base_uri": "https://localhost:8080/"} colab_type="code" id="RjU7u4GqY3aD" outputId="ff183ec1-babb-4f9b-a2a6-38a1fdca5f31"
#실제 켈리포니아 지도에 그리기 
import matplotlib.image as mpimg
california_img=mpimg.imread(os.path.join(images_path, filename))
ax = housing.plot(kind="scatter", x="longitude", y="latitude", figsize=(10,7),
                       s=housing['population']/100, label="Population",
                       c="median_house_value", cmap=plt.get_cmap("jet"),
                       colorbar=False, alpha=0.4,
                      )
plt.imshow(california_img, extent=[-124.55, -113.80, 32.45, 42.05], alpha=0.5,
           cmap=plt.get_cmap("jet"))
plt.ylabel("Latitude", fontsize=14)
plt.xlabel("Longitude", fontsize=14)

prices = housing["median_house_value"]
tick_values = np.linspace(prices.min(), prices.max(), 11)
cbar = plt.colorbar(ticks=tick_values/prices.max())
cbar.ax.set_yticklabels(["$%dk"%(round(v/1000)) for v in tick_values], fontsize=14)
cbar.set_label('Median House Value', fontsize=16)

plt.legend(fontsize=16)
save_fig("california_housing_prices_plot")
plt.show()

# + colab={} colab_type="code" id="smKSfQMSY3aG"
corr_matrix = housing.corr()

# + colab={"base_uri": "https://localhost:8080/"} colab_type="code" id="lZZYbNhRY3aI" outputId="64e9602f-44a3-4e9b-9a00-f6bd5c821fb0"
corr_matrix["median_house_value"].sort_values(ascending=False) #mdeian_house_value와의 correlation을 sort해서 가져옴. 

# + colab={"base_uri": "https://localhost:8080/"} colab_type="code" id="i6UnHkDcY3aL" outputId="20908b5f-f98d-446b-e9d3-b9cbbe1731be"
# from pandas.tools.plotting import scatter_matrix # For older versions of Pandas
from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
save_fig("scatter_matrix_plot")

# + colab={"base_uri": "https://localhost:8080/"} colab_type="code" id="HtUuga8HY3aQ" outputId="1d580d45-eeb7-47d0-c182-bc97efc3cf06"
housing.plot(kind="scatter", x="median_income", y="median_house_value",
             alpha=0.1)
plt.axis([0, 16, 0, 550000])
save_fig("income_vs_house_value_scatterplot")

# + colab={} colab_type="code" id="1LXlWepjY3aS"
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]

# + colab={"base_uri": "https://localhost:8080/"} colab_type="code" id="zrYfvu-WY3aW" outputId="4f128e1c-144e-4c9c-a48a-2b399f737339"
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

# + colab={"base_uri": "https://localhost:8080/"} colab_type="code" id="FJyfBVb8U9ET" outputId="6093b3c6-1078-45fb-b8f0-19264a178fe4"
housing.plot(kind="scatter", x="bedrooms_per_room", y="median_house_value",
             alpha=0.2)
plt.axis([0, 1, 0, 520000])
plt.show()

# + colab={"base_uri": "https://localhost:8080/"} colab_type="code" id="AunWA0hNY3aZ" outputId="1915ae04-3b7b-4940-a88f-5c6ff0b18125"
housing.plot(kind="scatter", x="rooms_per_household", y="median_house_value",
             alpha=0.2)
plt.axis([0, 5, 0, 520000])
plt.show()

# + colab={"base_uri": "https://localhost:8080/"} colab_type="code" id="G0T_ujuyY3ad" outputId="e632f16d-a08d-42b0-aed4-a9d1c7e3fa6e"
housing.describe()

# + [markdown] colab_type="text" id="5elg9hH6Y3ag"
# # Prepare the data for Machine Learning algorithms

# + colab={} colab_type="code" id="pNBwlaVcY3ag"
housing = strat_train_set.drop("median_house_value", axis=1) # drop labels for training set
housing_labels = strat_train_set["median_house_value"].copy()

# + colab={"base_uri": "https://localhost:8080/"} colab_type="code" id="K85I0jm2Y3ai" outputId="bc2b0ccd-61f1-42f3-f020-f3b8d21f668f"
sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head()
sample_incomplete_rows

# + colab={"base_uri": "https://localhost:8080/"} colab_type="code" id="v19W6D8WY3ak" outputId="47a07ea0-9e78-46ee-91ce-a3d838e1bc15"
sample_incomplete_rows.dropna(subset=["total_bedrooms"])    # option 1: 결측치 단순제거 

# + colab={"base_uri": "https://localhost:8080/"} colab_type="code" id="2hlaIkmnY3am" outputId="0a8914b7-17b8-4e07-afd1-f202f37d5173"
sample_incomplete_rows.drop("total_bedrooms", axis=1)       # option 2: 결측치 많은 total_bedrooms 변수를 아예 제거해버림 

# + colab={} colab_type="code" id="BlbZPHV_Y3ar"
median = housing["total_bedrooms"].median()
sample_incomplete_rows["total_bedrooms"].fillna(median, inplace=True) # option 3 : median으로 처리 

# + colab={"base_uri": "https://localhost:8080/"} colab_type="code" id="b3Qbr6MaY3av" outputId="c46b806a-7ea9-4b33-c9e3-be38c3b55c5b"
sample_incomplete_rows

# + colab={} colab_type="code" id="4-PnxO54Y3ax"
#sklearn으로 결측치 처리하기 
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")

# + [markdown] colab_type="text" id="Sy2cJ1VdY3az"
# Remove the text attribute because median can only be calculated on numerical attributes:

# + colab={} colab_type="code" id="6Rd1t-zoY3a0"
housing_num = housing.drop("ocean_proximity", axis=1)
# alternatively: housing_num = housing.select_dtypes(include=[np.number])

# + colab={"base_uri": "https://localhost:8080/"} colab_type="code" id="ZCDxKU_KY3a3" outputId="ecbb88cf-c846-48b1-83a4-0d6c416068c9"
imputer.fit(housing_num)

# + colab={"base_uri": "https://localhost:8080/"} colab_type="code" id="_-r3jUCUY3a6" outputId="6a8740a4-a3f0-4565-e974-64c2cf7bdcd1"
imputer.statistics_

# + [markdown] colab_type="text" id="haIkvttlY3a9"
# Check that this is the same as manually computing the median of each attribute:

# + colab={"base_uri": "https://localhost:8080/"} colab_type="code" id="I00ATEPUY3a9" outputId="4eb0ab0f-b92a-4f47-aa67-e782fb47536a"
housing_num.median().values #위의 결과가 median을 제대로 가져왔는지 확인 

# + [markdown] colab_type="text" id="LLfWqI_GY3bA"
# Transform the training set:

# + colab={} colab_type="code" id="tmAuhkDXY3bB"
X = imputer.transform(housing_num)

# + colab={} colab_type="code" id="HF6GAMkjY3bD"
#리턴값이 numpy형태임ㅇ로 판다스로 데이터형변경 
housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                          index=housing.index)

# + colab={"base_uri": "https://localhost:8080/"} colab_type="code" id="eTRFCVqtY3bF" outputId="bc04668d-eae4-42e1-ca56-2a6f7471ece6"
housing_tr.loc[sample_incomplete_rows.index.values]

# + colab={"base_uri": "https://localhost:8080/"} colab_type="code" id="Sl55JfYdY3bH" outputId="2084c1fd-3c43-4a2e-dd53-adeddaecab21"
imputer.strategy

# + colab={} colab_type="code" id="SzCTj11hY3bJ"
housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                          index=housing_num.index)

# + colab={"base_uri": "https://localhost:8080/"} colab_type="code" id="BC_fwn_TY3bL" outputId="0d9e9c46-b211-44c1-ec75-5dde7a58d780"
housing_tr.head()

# + [markdown] colab_type="text" id="YgTxQd2aY3bN"
# #범주형 데이터 다루기
#
# Now let's preprocess the categorical input feature, `ocean_proximity`:
#

# + colab={"base_uri": "https://localhost:8080/"} colab_type="code" id="noUqOtuQY3bO" outputId="279edebc-e906-4853-f811-c17170ee6d1d"
housing_cat = housing[["ocean_proximity"]]
housing_cat.head(10)

# + colab={"base_uri": "https://localhost:8080/"} colab_type="code" id="UyZ0iRwoY3bQ" outputId="4b024fc2-1627-47b4-acf4-c77f70e33c6c"
#ordianl encoding 방법사용 
from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
housing_cat_encoded[:10]

# + colab={"base_uri": "https://localhost:8080/"} colab_type="code" id="dmnk5UISY3bS" outputId="a05e21e6-ddf6-48d3-c637-2ae9ca0b0e0b"
ordinal_encoder.categories_

# + colab={"base_uri": "https://localhost:8080/"} colab_type="code" id="AfqP_5N4Y3bU" outputId="2c582a1e-e4dd-4f77-81c8-a7fad3fcbf49"
#원핫인코딩(더미변수) 방법사용 
from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot
#원핫인코딩은 0이 많아서 메모리 낭비하게됨. 그래서 리턴할때 sparse matrix의 형태로 자동으로 변경하여 저장함. (더 효율적)

# + [markdown] colab_type="text" id="nGd85hZfY3bW"
# By default, the `OneHotEncoder` class returns a sparse array, but we can convert it to a dense array if needed by calling the `toarray()` method:

# + colab={"base_uri": "https://localhost:8080/"} colab_type="code" id="KgCkNpizY3bW" outputId="bd6fa7a5-2d3c-474c-b558-1ed7ebe3a807"
housing_cat_1hot.toarray() #데이터를 다루기 위해 array형태로 변경 

# + [markdown] colab_type="text" id="JM8kVKbLY3bY"
# Alternatively, you can set `sparse=False` when creating the `OneHotEncoder`:

# + colab={"base_uri": "https://localhost:8080/"} colab_type="code" id="Xu47qTo2Y3bY" outputId="93be8180-84b9-4c58-f4ee-ca273255c982"
cat_encoder = OneHotEncoder(sparse=False)
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot

# + colab={"base_uri": "https://localhost:8080/"} colab_type="code" id="OWLQwSzZY3bd" outputId="5468c0ab-167c-4c10-f429-e62dcea3b2b9"
cat_encoder.categories_

# + [markdown] colab_type="text" id="840YCglhY3bf"
# Let's create a custom transformer to add extra attributes:

# + colab={} colab_type="code" id="fx1lTelzY3bf"
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

# + colab={"base_uri": "https://localhost:8080/"} colab_type="code" id="sGwu1W7QY3bh" outputId="f8ad17e3-cc15-4ea4-d75c-cba8b6b9c29e"
housing_extra_attribs = pd.DataFrame(
    housing_extra_attribs,
    columns=list(housing.columns)+["rooms_per_household", "population_per_household"],
    index=housing.index)
housing_extra_attribs.head()

# + [markdown] colab_type="text" id="i3HrqgrxY3bj"
# Now let's build a pipeline for preprocessing the numerical attributes:

# + colab={} colab_type="code" id="I2ENEtKqY3bk"
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

housing_num_tr = num_pipeline.fit_transform(housing_num)

# + colab={"base_uri": "https://localhost:8080/"} colab_type="code" id="V_ROrPu2Y3bl" outputId="d44d7261-bcd9-4b0a-b719-3f439feb5ac2"
housing_num_tr

# + colab={} colab_type="code" id="UPy99GdmY3bo"
from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

housing_prepared = full_pipeline.fit_transform(housing)

# + colab={"base_uri": "https://localhost:8080/"} colab_type="code" id="TkdFKOwrY3br" outputId="c59f4af1-4642-4bad-89ed-19032a8685c4"
housing_prepared

# + colab={"base_uri": "https://localhost:8080/"} colab_type="code" id="fk8Ww5OJY3bt" outputId="14a38275-e6e3-482e-a5b5-3c27c1a52d22"
housing_prepared.shape

# + [markdown] colab_type="text" id="jfm0_zDOY3bw"
# For reference, here is the old solution based on a `DataFrameSelector` transformer (to just select a subset of the Pandas `DataFrame` columns), and a `FeatureUnion`:

# + colab={} colab_type="code" id="nTXMRpprY3bw"
from sklearn.base import BaseEstimator, TransformerMixin

# Create a class to select numerical or categorical columns 
class OldDataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values


# + [markdown] colab_type="text" id="nWDdXyPfY3by"
# Now let's join all these components into a big pipeline that will preprocess both the numerical and the categorical features:

# + colab={} colab_type="code" id="TCiTF5KRY3by"
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

old_num_pipeline = Pipeline([
        ('selector', OldDataFrameSelector(num_attribs)),
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

old_cat_pipeline = Pipeline([
        ('selector', OldDataFrameSelector(cat_attribs)),
        ('cat_encoder', OneHotEncoder(sparse=False)),
    ])

# + colab={} colab_type="code" id="GvAidIsKY3b0"
from sklearn.pipeline import FeatureUnion

old_full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", old_num_pipeline),
        ("cat_pipeline", old_cat_pipeline),
    ])

# + colab={"base_uri": "https://localhost:8080/"} colab_type="code" id="9Kuy6lU0Y3b3" outputId="90d4965e-c4e9-444e-e514-91ae23590842"
old_housing_prepared = old_full_pipeline.fit_transform(housing)
old_housing_prepared

# + [markdown] colab_type="text" id="R0QguSNNY3b5"
# The result is the same as with the `ColumnTransformer`:

# + colab={"base_uri": "https://localhost:8080/"} colab_type="code" id="R0vkkxVeY3b6" outputId="209a3a74-91c2-4e6b-a5f6-fd87b99026ba"
np.allclose(housing_prepared, old_housing_prepared)

# + [markdown] colab_type="text" id="6YBpTiDuY3b8"
# # Select and train a model 

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="qGBzI3i2Y3b9" outputId="a5dc3e8a-c43f-47bf-d95c-ab0c087f463c"
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression() 
lin_reg.fit(housing_prepared, housing_labels)

# + colab={"base_uri": "https://localhost:8080/", "height": 53} colab_type="code" id="FfWgfVMhY3b-" outputId="869a7ccf-31d6-4a45-f761-a20b489f044c"
# let's try the full preprocessing pipeline on a few training instances
#데이터 다섯개만 가져와봄. 
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
#가져온 데이터를 full_pipeline에 통과시킨 후 적합시킴. 
some_data_prepared = full_pipeline.transform(some_data)

print("Predictions:", lin_reg.predict(some_data_prepared)) #prediction값 

# + [markdown] colab_type="text" id="md0gtp0CY3cB"
# Compare against the actual values:

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="WGE5AbSNY3cB" outputId="3ef11d7b-a965-4fd2-cf33-7eff7ed3d410"
print("Labels:", list(some_labels)) #실제갑. 위의 prediction값과 비교해볼 수 있다. 

# + colab={"base_uri": "https://localhost:8080/", "height": 377} colab_type="code" id="7qPfjk1tY3cD" outputId="d21a2742-8254-4ec7-8f62-79e94faed3aa"
some_data_prepared

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="KoTbhziBY3cF" outputId="ea6b85db-33cf-4ad7-979a-6010b0f79624"
#mse 계산 
from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="soQHsAu7Y3cH" outputId="4eb50a5c-ee5c-4c2d-dccc-ec8cff8ca211"
#mae 계산 
from sklearn.metrics import mean_absolute_error

lin_mae = mean_absolute_error(housing_labels, housing_predictions)
lin_mae

# + colab={"base_uri": "https://localhost:8080/", "height": 125} colab_type="code" id="XuILXVeUY3cJ" outputId="1550afa5-5d08-4bd0-f73f-e3ae5ffd10dc"
#decisiontree 로 적합
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="xWfxIQ72Y3cL" outputId="45cb7192-48a9-474c-bfa5-12376785151a"
#training데이터를 평가데이터로 바로썼기 때문에 에러가 0. overfitting된거임. 
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse

# + [markdown] colab_type="text" id="I2QRo7sWY3cP"
# # Fine-tune your model

# + colab={} colab_type="code" id="yPAuXY5hY3cP"
#cv를 통해 overfit을 방지. 
from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)


# + colab={"base_uri": "https://localhost:8080/", "height": 107} colab_type="code" id="siSiBsILY3cS" outputId="23fe0a01-25d4-434f-84d3-3df2ac9f1d73"
#결과를 display하는 함수를 정의하여 사용. 
#decision tree의 에러가 리니어모델보다 더 높게 나온다. 
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(tree_rmse_scores)

# + colab={"base_uri": "https://localhost:8080/", "height": 107} colab_type="code" id="9uu_s_YbY3cU" outputId="ad26a7ea-87b2-444f-a3d1-aa2ebb167075"
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                             scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)

# + [markdown] colab_type="text" id="hVWCTvXiY3cV"
# **Note**: we specify `n_estimators=100` to be future-proof since the default value is going to change to 100 in Scikit-Learn 0.22 (for simplicity, this is not shown in the book).

# + colab={"base_uri": "https://localhost:8080/", "height": 143} colab_type="code" id="xb2wqhfgY3cV" outputId="32de9242-6330-448f-b4ce-d3073d95efb5"
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
forest_reg.fit(housing_prepared, housing_labels)

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="6zTI5CreY3cZ" outputId="171043aa-afe6-4191-d97f-a0b4e2a918a3"
#rf그냥사용시 에러 
housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse

# + colab={"base_uri": "https://localhost:8080/", "height": 107} colab_type="code" id="gNksQgFBY3cc" outputId="d8c8a144-72a3-49a2-80d8-6ca5a7af84b3"
from sklearn.model_selection import cross_val_score

forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)

# + colab={} colab_type="code" id="oUS9To4UY3cd"
scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
pd.Series(np.sqrt(-scores)).describe()

# + colab={} colab_type="code" id="yoZpXNOrY3cf"
from sklearn.svm import SVR

svm_reg = SVR(kernel="linear")
svm_reg.fit(housing_prepared, housing_labels)
housing_predictions = svm_reg.predict(housing_prepared)
svm_mse = mean_squared_error(housing_labels, housing_predictions)
svm_rmse = np.sqrt(svm_mse)
svm_rmse

# + colab={"base_uri": "https://localhost:8080/", "height": 395} colab_type="code" id="efoC8txOY3ch" outputId="27b656d3-e41e-4555-c237-b76dfc7f1cc6"
#하이퍼파라미터 튜닝. 
from sklearn.model_selection import GridSearchCV

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

# + [markdown] colab_type="text" id="Y9SA9DxKY3ci"
# The best hyperparameter combination found:

# + colab={"base_uri": "https://localhost:8080/", "height": 35} colab_type="code" id="WXyyd-l6Y3ci" outputId="6440cb6e-9a9e-4518-8869-4e15124b2dc0"
#베스트파라미터 확인
grid_search.best_params_

# + colab={"base_uri": "https://localhost:8080/", "height": 143} colab_type="code" id="LlKmudN1Y3ck" outputId="07ea2281-31eb-40d2-98f3-22d843612c78"

grid_search.best_estimator_

# + [markdown] colab_type="text" id="M_Fn2_t6Y3cl"
# Let's look at the score of each hyperparameter combination tested during the grid search:

# + colab={"base_uri": "https://localhost:8080/", "height": 341} colab_type="code" id="n3TWq2-IY3cm" outputId="9799eb7a-7cd3-4b36-ff22-d800cbf1d936"
#cv결과 확인
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

# + colab={"base_uri": "https://localhost:8080/", "height": 1000} colab_type="code" id="kDoR_md4Y3cn" outputId="1e1a7e4c-c242-4b0f-daef-8558fee173f9"
#판다스형태로 데이터보기
pd.DataFrame(grid_search.cv_results_)

# + colab={"base_uri": "https://localhost:8080/", "height": 413} colab_type="code" id="zhheLZdlY3cq" outputId="ae7189e9-4cf1-45c8-8811-cf262c2a0714"
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

#파라미터의 distribution을 정의 
param_distribs = {
        'n_estimators': randint(low=1, high=200),
        'max_features': randint(low=1, high=8),
    }

forest_reg = RandomForestRegressor(random_state=42)
#랜덤하게 10개를 search 
rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
rnd_search.fit(housing_prepared, housing_labels)

# + colab={} colab_type="code" id="eCzLXZCdY3cr"
cvres = rnd_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

# + colab={} colab_type="code" id="mWYhPPMYY3cs"
#트리형 모델은 feature importance를 볼수있다. 
feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances

# + colab={} colab_type="code" id="Nv3sDlxyY3cw"
#freautre importance를 sort해서 보기 
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
#cat_encoder = cat_pipeline.named_steps["cat_encoder"] # old solution
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)

# + colab={} colab_type="code" id="2UKQgDcsY3cz"
#grid_search로 선택한 best model을 적합 후 mse/rmse 구함.
final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

# + colab={} colab_type="code" id="rDkdrEwSY3c0"
final_rmse

# + [markdown] colab_type="text" id="RYrJ2EUYY3c3"
# We can compute a 95% confidence interval for the test RMSE:

# + colab={} colab_type="code" id="wkFz0ZT6Y3c3"
from scipy import stats

confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                         loc=squared_errors.mean(),
                         scale=stats.sem(squared_errors)))

# + [markdown] colab_type="text" id="KPO2GZ33Y3c5"
# We could compute the interval manually like this:

# + colab={} colab_type="code" id="V8Dfr8Y_Y3c5"
m = len(squared_errors)
mean = squared_errors.mean()
tscore = stats.t.ppf((1 + confidence) / 2, df=m - 1)
tmargin = tscore * squared_errors.std(ddof=1) / np.sqrt(m)
np.sqrt(mean - tmargin), np.sqrt(mean + tmargin)

# + [markdown] colab_type="text" id="EajGiiRfY3c6"
# Alternatively, we could use a z-scores rather than t-scores:

# + colab={} colab_type="code" id="cTtRPLj9Y3c6"
zscore = stats.norm.ppf((1 + confidence) / 2)
zmargin = zscore * squared_errors.std(ddof=1) / np.sqrt(m)
np.sqrt(mean - zmargin), np.sqrt(mean + zmargin)

# + [markdown] colab_type="text" id="q3N1id5sY3c8"
# # Extra material

# + [markdown] colab_type="text" id="lGmvAyZdY3c8"
# ## A full pipeline with both preparation and prediction

# + colab={} colab_type="code" id="q3g-3feSY3c8"
full_pipeline_with_predictor = Pipeline([
        ("preparation", full_pipeline),
        ("linear", LinearRegression())
    ])

full_pipeline_with_predictor.fit(housing, housing_labels)
full_pipeline_with_predictor.predict(some_data)

# + [markdown] colab_type="text" id="Cdtf7Ts-Y3c9"
# ## Model persistence using joblib

# + colab={} colab_type="code" id="xrG21NSzY3c-"
my_model = full_pipeline_with_predictor

# + colab={} colab_type="code" id="lPgX0SxWY3c_"
import joblib
joblib.dump(my_model, "my_model.pkl") # DIFF
#...
my_model_loaded = joblib.load("my_model.pkl") # DIFF

# + [markdown] colab_type="text" id="v-8ePkcyY3dB"
# ## Example SciPy distributions for `RandomizedSearchCV`

# + colab={} colab_type="code" id="umbqchoSY3dB"
from scipy.stats import geom, expon
geom_distrib=geom(0.5).rvs(10000, random_state=42)
expon_distrib=expon(scale=1).rvs(10000, random_state=42)
plt.hist(geom_distrib, bins=50)
plt.show()
plt.hist(expon_distrib, bins=50)
plt.show()

# + [markdown] colab_type="text" id="wNDRhX15Y3dC"
# # Exercise solutions

# + [markdown] colab_type="text" id="9D9EwQFNY3dC"
# ## 1.

# + [markdown] colab_type="text" id="7GvdoXNwY3dD"
# Question: Try a Support Vector Machine regressor (`sklearn.svm.SVR`), with various hyperparameters such as `kernel="linear"` (with various values for the `C` hyperparameter) or `kernel="rbf"` (with various values for the `C` and `gamma` hyperparameters). Don't worry about what these hyperparameters mean for now. How does the best `SVR` predictor perform?

# + colab={} colab_type="code" id="a3sJYRhKY3dD"
from sklearn.model_selection import GridSearchCV

param_grid = [
        {'kernel': ['linear'], 'C': [10., 30., 100., 300., 1000., 3000., 10000., 30000.0]},
        {'kernel': ['rbf'], 'C': [1.0, 3.0, 10., 30., 100., 300., 1000.0],
         'gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]},
    ]

svm_reg = SVR()
grid_search = GridSearchCV(svm_reg, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2)
grid_search.fit(housing_prepared, housing_labels)

# + [markdown] colab_type="text" id="z62e9_odY3dE"
# The best model achieves the following score (evaluated using 5-fold cross validation):

# + colab={} colab_type="code" id="B-Dus-3BY3dE"
negative_mse = grid_search.best_score_
rmse = np.sqrt(-negative_mse)
rmse

# + [markdown] colab_type="text" id="154Z9LJFY3dF"
# That's much worse than the `RandomForestRegressor`. Let's check the best hyperparameters found:

# + colab={} colab_type="code" id="gIBSE_tsY3dF"
grid_search.best_params_

# + [markdown] colab_type="text" id="5uMfFM25Y3dH"
# The linear kernel seems better than the RBF kernel. Notice that the value of `C` is the maximum tested value. When this happens you definitely want to launch the grid search again with higher values for `C` (removing the smallest values), because it is likely that higher values of `C` will be better.

# + [markdown] colab_type="text" id="psy-LzwWY3dH"
# ## 2.

# + [markdown] colab_type="text" id="p7isx_8_Y3dI"
# Question: Try replacing `GridSearchCV` with `RandomizedSearchCV`.

# + colab={} colab_type="code" id="oHSfBqKAY3dI"
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import expon, reciprocal

# see https://docs.scipy.org/doc/scipy/reference/stats.html
# for `expon()` and `reciprocal()` documentation and more probability distribution functions.

# Note: gamma is ignored when kernel is "linear"
param_distribs = {
        'kernel': ['linear', 'rbf'],
        'C': reciprocal(20, 200000),
        'gamma': expon(scale=1.0),
    }

svm_reg = SVR()
rnd_search = RandomizedSearchCV(svm_reg, param_distributions=param_distribs,
                                n_iter=50, cv=5, scoring='neg_mean_squared_error',
                                verbose=2, random_state=42)
rnd_search.fit(housing_prepared, housing_labels)

# + [markdown] colab_type="text" id="ljFyBcihY3dJ"
# The best model achieves the following score (evaluated using 5-fold cross validation):

# + colab={} colab_type="code" id="76SkJ0NmY3dJ"
negative_mse = rnd_search.best_score_
rmse = np.sqrt(-negative_mse)
rmse

# + [markdown] colab_type="text" id="xG4Nrb1XY3dK"
# Now this is much closer to the performance of the `RandomForestRegressor` (but not quite there yet). Let's check the best hyperparameters found:

# + colab={} colab_type="code" id="sef49umXY3dK"
rnd_search.best_params_

# + [markdown] colab_type="text" id="wNBYEv_wY3dL"
# This time the search found a good set of hyperparameters for the RBF kernel. Randomized search tends to find better hyperparameters than grid search in the same amount of time.

# + [markdown] colab_type="text" id="D2BdU01RY3dM"
# Let's look at the exponential distribution we used, with `scale=1.0`. Note that some samples are much larger or smaller than 1.0, but when you look at the log of the distribution, you can see that most values are actually concentrated roughly in the range of exp(-2) to exp(+2), which is about 0.1 to 7.4.

# + colab={} colab_type="code" id="m7xC6j62Y3dM"
expon_distrib = expon(scale=1.)
samples = expon_distrib.rvs(10000, random_state=42)
plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.title("Exponential distribution (scale=1.0)")
plt.hist(samples, bins=50)
plt.subplot(122)
plt.title("Log of this distribution")
plt.hist(np.log(samples), bins=50)
plt.show()

# + [markdown] colab_type="text" id="TUJGExkfY3dN"
# The distribution we used for `C` looks quite different: the scale of the samples is picked from a uniform distribution within a given range, which is why the right graph, which represents the log of the samples, looks roughly constant. This distribution is useful when you don't have a clue of what the target scale is:

# + colab={} colab_type="code" id="s6lAOqX1Y3dN"
reciprocal_distrib = reciprocal(20, 200000)
samples = reciprocal_distrib.rvs(10000, random_state=42)
plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.title("Reciprocal distribution (scale=1.0)")
plt.hist(samples, bins=50)
plt.subplot(122)
plt.title("Log of this distribution")
plt.hist(np.log(samples), bins=50)
plt.show()

# + [markdown] colab_type="text" id="dnuoZhmfY3dP"
# The reciprocal distribution is useful when you have no idea what the scale of the hyperparameter should be (indeed, as you can see on the figure on the right, all scales are equally likely, within the given range), whereas the exponential distribution is best when you know (more or less) what the scale of the hyperparameter should be.

# + [markdown] colab_type="text" id="e8JXB9GkY3dP"
# ## 3.

# + [markdown] colab_type="text" id="AxTcXZqSY3dP"
# Question: Try adding a transformer in the preparation pipeline to select only the most important attributes.

# + colab={} colab_type="code" id="nO_7f4mjY3dP"
from sklearn.base import BaseEstimator, TransformerMixin

def indices_of_top_k(arr, k):
    return np.sort(np.argpartition(np.array(arr), -k)[-k:])

class TopFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_importances, k):
        self.feature_importances = feature_importances
        self.k = k
    def fit(self, X, y=None):
        self.feature_indices_ = indices_of_top_k(self.feature_importances, self.k)
        return self
    def transform(self, X):
        return X[:, self.feature_indices_]


# + [markdown] colab_type="text" id="1O_3_a1QY3dT"
# Note: this feature selector assumes that you have already computed the feature importances somehow (for example using a `RandomForestRegressor`). You may be tempted to compute them directly in the `TopFeatureSelector`'s `fit()` method, however this would likely slow down grid/randomized search since the feature importances would have to be computed for every hyperparameter combination (unless you implement some sort of cache).

# + [markdown] colab_type="text" id="lLLKBHT0Y3dU"
# Let's define the number of top features we want to keep:

# + colab={} colab_type="code" id="EoES4cFpY3dV"
k = 5

# + [markdown] colab_type="text" id="0b3RuOYjY3dY"
# Now let's look for the indices of the top k features:

# + colab={} colab_type="code" id="mGWXBTLMY3dY"
top_k_feature_indices = indices_of_top_k(feature_importances, k)
top_k_feature_indices

# + colab={} colab_type="code" id="GEzF1tTPY3dc"
np.array(attributes)[top_k_feature_indices]

# + [markdown] colab_type="text" id="NIsF_vy6Y3dd"
# Let's double check that these are indeed the top k features:

# + colab={} colab_type="code" id="P1xsvqewY3dd"
sorted(zip(feature_importances, attributes), reverse=True)[:k]

# + [markdown] colab_type="text" id="3VPItKq-Y3df"
# Looking good... Now let's create a new pipeline that runs the previously defined preparation pipeline, and adds top k feature selection:

# + colab={} colab_type="code" id="omBIRuDqY3df"
preparation_and_feature_selection_pipeline = Pipeline([
    ('preparation', full_pipeline),
    ('feature_selection', TopFeatureSelector(feature_importances, k))
])

# + colab={} colab_type="code" id="7F2BuqRoY3dh"
housing_prepared_top_k_features = preparation_and_feature_selection_pipeline.fit_transform(housing)

# + [markdown] colab_type="text" id="YrG0apvaY3di"
# Let's look at the features of the first 3 instances:

# + colab={} colab_type="code" id="BN6NEobZY3di"
housing_prepared_top_k_features[0:3]

# + [markdown] colab_type="text" id="ahxaKT1xY3dj"
# Now let's double check that these are indeed the top k features:

# + colab={} colab_type="code" id="Mh-4OmETY3dj"
housing_prepared[0:3, top_k_feature_indices]

# + [markdown] colab_type="text" id="nauf9n8aY3dl"
# Works great!  :)

# + [markdown] colab_type="text" id="pxkRV5LYY3dl"
# ## 4.

# + [markdown] colab_type="text" id="1GLWEMpUY3dl"
# Question: Try creating a single pipeline that does the full data preparation plus the final prediction.

# + colab={} colab_type="code" id="Ja4GJIkGY3dl"
prepare_select_and_predict_pipeline = Pipeline([
    ('preparation', full_pipeline),
    ('feature_selection', TopFeatureSelector(feature_importances, k)),
    ('svm_reg', SVR(**rnd_search.best_params_))
])

# + colab={} colab_type="code" id="sJugTQ2pY3do"
prepare_select_and_predict_pipeline.fit(housing, housing_labels)

# + [markdown] colab_type="text" id="o1y9NZz1Y3dq"
# Let's try the full pipeline on a few instances:

# + colab={} colab_type="code" id="J_85WrdaY3dq"
some_data = housing.iloc[:4]
some_labels = housing_labels.iloc[:4]

print("Predictions:\t", prepare_select_and_predict_pipeline.predict(some_data))
print("Labels:\t\t", list(some_labels))

# + [markdown] colab_type="text" id="v-09oN1bY3dr"
# Well, the full pipeline seems to work fine. Of course, the predictions are not fantastic: they would be better if we used the best `RandomForestRegressor` that we found earlier, rather than the best `SVR`.

# + [markdown] colab_type="text" id="L66iXNKVY3dr"
# ## 5.

# + [markdown] colab_type="text" id="6QKo7LDnY3ds"
# Question: Automatically explore some preparation options using `GridSearchCV`.

# + colab={} colab_type="code" id="KZ_9gXZOY3ds"
param_grid = [{
    'preparation__num__imputer__strategy': ['mean', 'median', 'most_frequent'],
    'feature_selection__k': list(range(1, len(feature_importances) + 1))
}]

grid_search_prep = GridSearchCV(prepare_select_and_predict_pipeline, param_grid, cv=5,
                                scoring='neg_mean_squared_error', verbose=2)
grid_search_prep.fit(housing, housing_labels)

# + colab={} colab_type="code" id="5kiKV867Y3dv"
grid_search_prep.best_params_

# + [markdown] colab_type="text" id="fYYCkFoJY3dw"
# The best imputer strategy is `most_frequent` and apparently almost all features are useful (15 out of 16). The last one (`ISLAND`) seems to just add some noise.

# + [markdown] colab_type="text" id="Vssj6i2aY3dx"
# Congratulations! You already know quite a lot about Machine Learning. :)
