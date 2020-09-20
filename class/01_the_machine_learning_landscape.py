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
# <a href="https://colab.research.google.com/github/ChangRyeol/datamining2/blob/master/class/01_the_machine_learning_landscape.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="xFn0TpVEWk8f" colab_type="text"
# **Chapter 1 – The Machine Learning landscape**
#
# _This is the code used to generate some of the figures in chapter 1._

# + [markdown] id="hCt0_0dfWk8h" colab_type="text"
# <table align="left">
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/ageron/handson-ml2/blob/master/01_the_machine_learning_landscape.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
#   </td>
# </table>

# + [markdown] id="yTLA_Pchi051" colab_type="text"
# #goole drive mounting

# + id="0slawQnriwC4" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 55} executionInfo={"status": "ok", "timestamp": 1600605913684, "user_tz": -540, "elapsed": 27360, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="d5d97543-f0d0-4340-abaa-a03c6e24f533"
from google.colab import drive # import drive from google colab

ROOT = "/content/drive"     # default location for the drive
print(ROOT)                 # print content of ROOT (Optional)

drive.mount(ROOT)           # we mount the google drive at /content/drive

# + [markdown] id="Q9pO3XB1i7vO" colab_type="text"
# ##syn py and ipynb 

# + id="wfR8Yb_GjDqU" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 36} executionInfo={"status": "ok", "timestamp": 1600605989309, "user_tz": -540, "elapsed": 710, "user": {"displayName": "docls vlc", "photoUrl": "", "userId": "07004006891778094139"}} outputId="08b381bf-4dd1-4f8a-b0ee-b5515eed217c"
# %pwd 

# + id="UG1ZRrvwjUxh" colab_type="code" colab={}
# %cd 'drive/My Drive/Colab Notebooks/datamining2/class' 

# + id="USRqa1HwjUun" colab_type="code" colab={}



# + [markdown] id="xchtv_aYWk8i" colab_type="text"
# # Code example 1-1

# + [markdown] id="jlLsriwCWk8k" colab_type="text"
# Although Python 2.x may work, it is deprecated so we strongly recommend you use Python 3 instead.

# + id="bTEj6NqGWk8l" colab_type="code" colab={}
# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)  # 버전이 3에서 5여야 실행 

# + id="MF-G05yWWk8p" colab_type="code" colab={}
# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20" #버전이 0.2.이상이여야 실행. 


# + [markdown] id="2pnMX_EwWk8t" colab_type="text"
# This function just merges the OECD's life satisfaction data and the IMF's GDP per capita data. It's a bit too long and boring and it's not specific to Machine Learning, which is why I left it out of the book.

# + id="QlJoAVD4Wk8t" colab_type="code" colab={}
#oecd 관련 전처리 함수 정의 
def prepare_country_stats(oecd_bli, gdp_per_capita):
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"]=="TOT"] ## inequality가 tot인 것만 뽑음. 
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
#pivot은 현재의 데이터로부터 새로운 데이터를 만든다. index를 country로 하자. indicator를 컬럼으로 정하면서 칼럼에 값을 Value로 넣는다. 
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)
    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,
                                  left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indices = list(set(range(36)) - set(remove_indices))
    return full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]


# + [markdown] id="tKHgZFEOWk8w" colab_type="text"
# The code in the book expects the data files to be located in the current directory. I just tweaked it here to fetch the files in datasets/lifesat.

# + id="9DQhauNvWk8x" colab_type="code" colab={}
import os
datapath = os.path.join("datasets", "lifesat", "") 


# + id="esKaEM-ya_n5" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 35} outputId="07adb4ad-9319-4804-9162-3fe92c39702e"
datapath #경로 확인. 이경로에 데이터셋 다운. 

# + id="eOsUlXIUWk80" colab_type="code" colab={}
# To plot pretty figures directly within Jupyter
# %matplotlib inline
import matplotlib as mpl
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# + id="rVpOC6E_Wk84" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 53} outputId="51776d0f-f55b-4732-b336-7ee5a8baee9f"
# Download the data
import urllib.request
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
os.makedirs(datapath, exist_ok=True)
for filename in ("oecd_bli_2015.csv", "gdp_per_capita.csv"):
    print("Downloading", filename)
    url = DOWNLOAD_ROOT + "datasets/lifesat/" + filename
    urllib.request.urlretrieve(url, datapath + filename)

# + id="K-gy5MP0cHbE" colab_type="code" colab={}
os.makedirs?  ## 이 함수가 어떤 기능인지 궁금할때 사용. 모든 함수를 설명할 순 없으니 이 기능을 자주 활용하자. 

# + id="NM2LcdngbrWs" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 35} outputId="447edd33-bf3a-4e5a-fbf4-c4ceffedd8c6"
os.listdir(datapath) ##진짜로 데이터가 다운되었는지 확인 



# + id="QPiKvUBPWk88" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 304} outputId="d6002c53-4bce-404f-eedd-ee8aa2267a63"
# Code example
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model

# Load the data
oecd_bli = pd.read_csv(datapath + "oecd_bli_2015.csv", thousands=',')
gdp_per_capita = pd.read_csv(datapath + "gdp_per_capita.csv",thousands=',',delimiter='\t',
                             encoding='latin1', na_values="n/a")



# + id="8AxkUnTygGVB" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="88bba7b9-d3c2-4df7-cef7-e38469b1f7e1"
oecd_bli

# + id="1TplmDRoWk8_" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 304} outputId="877deb55-b315-4f17-e755-6337d6c08b09"
# Prepare the data
country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
X = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life satisfaction"]]

# Visualize the data
country_stats.plot(kind='scatter', x="GDP per capita", y='Life satisfaction')
plt.show()

# Select a linear model #선형회귀 
model = sklearn.linear_model.LinearRegression()

# Train the model
model.fit(X, y)

# Make a prediction for Cyprus
X_new = [[22587]]  # Cyprus' GDP per capita
print(model.predict(X_new)) # outputs [[ 5.96242338]]

# + id="eeWdu73oWk9C" colab_type="code" colab={}



# + id="s9mBdD5pWk9F" colab_type="code" colab={}



# + [markdown] id="i_uquu5gWk9K" colab_type="text"
# # Note: you can ignore the rest of this notebook, it just generates many of the figures in chapter 1.

# + id="FP42IOpdWk9M" colab_type="code" colab={}



# + id="VhQ6yW8oWk9P" colab_type="code" colab={}



# + id="4PUQio4lWk9S" colab_type="code" colab={}



# + [markdown] id="LFJBDopAWk9V" colab_type="text"
# Create a function to save the figures.

# + id="8yYVtPvKWk9V" colab_type="code" colab={}
# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "fundamentals"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


# + [markdown] id="RhMgHjl0Wk9Z" colab_type="text"
# Make this notebook's output stable across runs:

# + id="iZ67E2roWk9Z" colab_type="code" colab={}
np.random.seed(42)

# + id="5DykP71GscpK" colab_type="code" colab={}



# + [markdown] id="5VOK60g-q64s" colab_type="text"
# 이 아래는 def prepare_country_stats(oecd_bli, gdp_per_capita):
# 위 함수를 자세히 풀어쓴 코드. 

# + [markdown] id="0usPF8W6Wk9c" colab_type="text"
# # Load and prepare Life satisfaction data

# + [markdown] id="xRhxeBxMWk9d" colab_type="text"
# If you want, you can get fresh data from the OECD's website.
# Download the CSV from http://stats.oecd.org/index.aspx?DataSetCode=BLI
# and save it to `datasets/lifesat/`.

# + id="OPx8-so4Wk9d" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 264} outputId="fd81cb00-25c9-4832-d25e-4567e545b355"
oecd_bli = pd.read_csv(datapath + "oecd_bli_2015.csv", thousands=',')
oecd_bli = oecd_bli[oecd_bli["INEQUALITY"]=="TOT"]
oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
oecd_bli.head(2)

# + id="b7CXXMwdWk9g" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 143} outputId="1ef58aa5-f69f-490d-d95a-eb8f1bf5172d"
oecd_bli["Life satisfaction"].head()

# + [markdown] id="0iCKL7KVWk9j" colab_type="text"
# # Load and prepare GDP per capita data

# + [markdown] id="2OVsblHVWk9j" colab_type="text"
# Just like above, you can update the GDP per capita data if you want. Just download data from http://goo.gl/j1MSKe (=> imf.org) and save it to `datasets/lifesat/`.

# + id="zpKZHGHGWk9k" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 142} outputId="51c53c87-5353-40ad-a8ab-17629e666fad"
gdp_per_capita = pd.read_csv(datapath+"gdp_per_capita.csv", thousands=',', delimiter='\t',
                             encoding='latin1', na_values="n/a")
gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
gdp_per_capita.set_index("Country", inplace=True)
gdp_per_capita.head(2)

# + id="i62MTCitWk9q" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="6fa35fc4-56d2-4c8a-db24-21c01b077044"
full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita, left_index=True, right_index=True)
full_country_stats.sort_values(by="GDP per capita", inplace=True)
full_country_stats

# + id="WJVZWeHZWk9t" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 71} outputId="27c2c8c5-d57e-483f-a6f9-cca50adb52e1"
full_country_stats[["GDP per capita", 'Life satisfaction']].loc["United States"] #.loc을 통해 이름으로 subset추출(인덱싱)

# + id="Qt3oCEnSWk9v" colab_type="code" colab={}
remove_indices = [0, 1, 6, 8, 33, 34, 35]
keep_indices = list(set(range(36)) - set(remove_indices))
#iloc를 통해 숫자로 인덱싱 
sample_data = full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices] 
missing_data = full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[remove_indices]

# + id="WD8PBBpUWk9y" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 243} outputId="8a2d48b8-ac86-4ec3-ce34-e5b606717820"
sample_data.plot(kind='scatter', x="GDP per capita", y='Life satisfaction', figsize=(5,3))
plt.axis([0, 60000, 0, 10])
position_text = {
    "Hungary": (5000, 1),
    "Korea": (18000, 1.7),
    "France": (29000, 2.4),
    "Australia": (40000, 3.0),
    "United States": (52000, 3.8),
}
for country, pos_text in position_text.items():
    pos_data_x, pos_data_y = sample_data.loc[country]
    country = "U.S." if country == "United States" else country
    plt.annotate(country, xy=(pos_data_x, pos_data_y), xytext=pos_text,
            arrowprops=dict(facecolor='black', width=0.5, shrink=0.1, headwidth=5))
    plt.plot(pos_data_x, pos_data_y, "ro")
plt.xlabel("GDP per capita (USD)")
save_fig('money_happy_scatterplot')
plt.show()

# + id="UAnnLNNFWk91" colab_type="code" colab={}
sample_data.to_csv(os.path.join("datasets", "lifesat", "lifesat.csv"))  #데이터를 csv로 저장 

# + id="GblyssfkWk96" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 235} outputId="9198d1f2-d47b-4f46-a20f-99bce297179e"
sample_data.loc[list(position_text.keys())]

# + id="pqgagAxTWk9-" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 243} outputId="f19c0fc1-6811-44a3-ec43-6aed4a91c3cb"
import numpy as np

sample_data.plot(kind='scatter', x="GDP per capita", y='Life satisfaction', figsize=(5,3)) 
plt.xlabel("GDP per capita (USD)")
plt.axis([0, 60000, 0, 10])
X=np.linspace(0, 60000, 1000)
plt.plot(X, 2*X/100000, "r")
plt.text(40000, 2.7, r"$\theta_0 = 0$", fontsize=14, color="r")
plt.text(40000, 1.8, r"$\theta_1 = 2 \times 10^{-5}$", fontsize=14, color="r")
plt.plot(X, 8 - 5*X/100000, "g")
plt.text(5000, 9.1, r"$\theta_0 = 8$", fontsize=14, color="g")
plt.text(5000, 8.2, r"$\theta_1 = -5 \times 10^{-5}$", fontsize=14, color="g")
plt.plot(X, 4 + 5*X/100000, "b")
plt.text(5000, 3.5, r"$\theta_0 = 4$", fontsize=14, color="b")
plt.text(5000, 2.6, r"$\theta_1 = 5 \times 10^{-5}$", fontsize=14, color="b")
save_fig('tweaking_model_params_plot')
plt.show()

# + id="mz3IiOpOWk-A" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 35} outputId="373e3b86-8184-49cc-e467-f56edd9a30dc"
from sklearn import linear_model
lin1 = linear_model.LinearRegression()
Xsample = np.c_[sample_data["GDP per capita"]]
ysample = np.c_[sample_data["Life satisfaction"]]
lin1.fit(Xsample, ysample)
t0, t1 = lin1.intercept_[0], lin1.coef_[0][0]
t0, t1

# + id="FDFTZYqBWk-D" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 243} outputId="7a845c19-9fad-4d56-8fd4-7b3e38326017"
sample_data.plot(kind='scatter', x="GDP per capita", y='Life satisfaction', figsize=(5,3))
plt.xlabel("GDP per capita (USD)")
plt.axis([0, 60000, 0, 10])
X=np.linspace(0, 60000, 1000)
plt.plot(X, t0 + t1*X, "b")
plt.text(5000, 3.1, r"$\theta_0 = 4.85$", fontsize=14, color="b")
plt.text(5000, 2.2, r"$\theta_1 = 4.91 \times 10^{-5}$", fontsize=14, color="b")
save_fig('best_fit_model_plot')
plt.show()


# + id="GGEtqT4pWk-G" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 53} outputId="5c86ac52-1420-4d7f-fd81-c3308fd5d339"
cyprus_gdp_per_capita = gdp_per_capita.loc["Cyprus"]["GDP per capita"]
print(cyprus_gdp_per_capita)
cyprus_predicted_life_satisfaction = lin1.predict([[cyprus_gdp_per_capita]])[0][0]
cyprus_predicted_life_satisfaction

# + id="cNC2FGkoWk-J" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 243} outputId="168d29a7-56c0-4e7e-d9d3-0232dba95630"
sample_data.plot(kind='scatter', x="GDP per capita", y='Life satisfaction', figsize=(5,3), s=1)
plt.xlabel("GDP per capita (USD)")
X=np.linspace(0, 60000, 1000)
plt.plot(X, t0 + t1*X, "b")
plt.axis([0, 60000, 0, 10])
plt.text(5000, 7.5, r"$\theta_0 = 4.85$", fontsize=14, color="b")
plt.text(5000, 6.6, r"$\theta_1 = 4.91 \times 10^{-5}$", fontsize=14, color="b")
plt.plot([cyprus_gdp_per_capita, cyprus_gdp_per_capita], [0, cyprus_predicted_life_satisfaction], "r--")
plt.text(25000, 5.0, r"Prediction = 5.96", fontsize=14, color="b")
plt.plot(cyprus_gdp_per_capita, cyprus_predicted_life_satisfaction, "ro")
save_fig('cyprus_prediction_plot')
plt.show()

# + id="4hLqGMYiWk-M" colab_type="code" colab={} outputId="460c69f3-b460-4b60-980f-ae83e73e8195"
sample_data[7:10]

# + id="Az0uh5SWWk-O" colab_type="code" colab={} outputId="951bd1ae-d4d6-40bb-a169-35c2f588a231"
(5.1+5.7+6.5)/3

# + id="V3tGWs4BWk-Q" colab_type="code" colab={}
backup = oecd_bli, gdp_per_capita

def prepare_country_stats(oecd_bli, gdp_per_capita):
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"]=="TOT"]
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)
    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,
                                  left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indices = list(set(range(36)) - set(remove_indices))
    return full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]


# + id="sa6-cvrtWk-S" colab_type="code" colab={} outputId="e4253f57-4c23-4cb9-909a-756346b18c85"
# Code example
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model

# Load the data
oecd_bli = pd.read_csv(datapath + "oecd_bli_2015.csv", thousands=',')
gdp_per_capita = pd.read_csv(datapath + "gdp_per_capita.csv",thousands=',',delimiter='\t',
                             encoding='latin1', na_values="n/a")

# Prepare the data
country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
X = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life satisfaction"]]

# Visualize the data
country_stats.plot(kind='scatter', x="GDP per capita", y='Life satisfaction')
plt.show()

# Select a linear model
model = sklearn.linear_model.LinearRegression()

# Train the model
model.fit(X, y)

# Make a prediction for Cyprus
X_new = [[22587]]  # Cyprus' GDP per capita
print(model.predict(X_new)) # outputs [[ 5.96242338]]

# + id="KT1fqDZqWk-V" colab_type="code" colab={}
oecd_bli, gdp_per_capita = backup

# + id="tedyL_efWk-Y" colab_type="code" colab={} outputId="9563055f-2c93-4b11-a11c-1fba6506b42f"
missing_data

# + id="Ymsn4cyoWk-b" colab_type="code" colab={}
position_text2 = {
    "Brazil": (1000, 9.0),
    "Mexico": (11000, 9.0),
    "Chile": (25000, 9.0),
    "Czech Republic": (35000, 9.0),
    "Norway": (60000, 3),
    "Switzerland": (72000, 3.0),
    "Luxembourg": (90000, 3.0),
}

# + id="hiX1mp0CWk-d" colab_type="code" colab={} outputId="e857f8a9-258e-4abe-e44a-5db620d376e4"
sample_data.plot(kind='scatter', x="GDP per capita", y='Life satisfaction', figsize=(8,3))
plt.axis([0, 110000, 0, 10])

for country, pos_text in position_text2.items():
    pos_data_x, pos_data_y = missing_data.loc[country]
    plt.annotate(country, xy=(pos_data_x, pos_data_y), xytext=pos_text,
            arrowprops=dict(facecolor='black', width=0.5, shrink=0.1, headwidth=5))
    plt.plot(pos_data_x, pos_data_y, "rs")

X=np.linspace(0, 110000, 1000)
plt.plot(X, t0 + t1*X, "b:")

lin_reg_full = linear_model.LinearRegression()
Xfull = np.c_[full_country_stats["GDP per capita"]]
yfull = np.c_[full_country_stats["Life satisfaction"]]
lin_reg_full.fit(Xfull, yfull)

t0full, t1full = lin_reg_full.intercept_[0], lin_reg_full.coef_[0][0]
X = np.linspace(0, 110000, 1000)
plt.plot(X, t0full + t1full * X, "k")
plt.xlabel("GDP per capita (USD)")

save_fig('representative_training_data_scatterplot')
plt.show()

# + id="8DWgAZoaWk-f" colab_type="code" colab={} outputId="0ee12a61-7d56-420a-b2ce-73f5da5b8f0d"
full_country_stats.plot(kind='scatter', x="GDP per capita", y='Life satisfaction', figsize=(8,3))
plt.axis([0, 110000, 0, 10])

from sklearn import preprocessing
from sklearn import pipeline

poly = preprocessing.PolynomialFeatures(degree=60, include_bias=False)
scaler = preprocessing.StandardScaler()
lin_reg2 = linear_model.LinearRegression()

pipeline_reg = pipeline.Pipeline([('poly', poly), ('scal', scaler), ('lin', lin_reg2)])
pipeline_reg.fit(Xfull, yfull)
curve = pipeline_reg.predict(X[:, np.newaxis])
plt.plot(X, curve)
plt.xlabel("GDP per capita (USD)")
save_fig('overfitting_model_plot')
plt.show()

# + id="MfTjOUXzWk-h" colab_type="code" colab={} outputId="e04471db-c9a0-439f-d620-f715e87e2e50"
full_country_stats.loc[[c for c in full_country_stats.index if "W" in c.upper()]]["Life satisfaction"]

# + id="tthRe82SWk-l" colab_type="code" colab={} outputId="b124b031-6a23-4dbd-ca81-9b5cb64fff3f"
gdp_per_capita.loc[[c for c in gdp_per_capita.index if "W" in c.upper()]].head()

# + id="6FLYlqufWk-n" colab_type="code" colab={} outputId="a54e217b-5122-4b8c-fb07-1e8a9754f4e7"
plt.figure(figsize=(8,3))

plt.xlabel("GDP per capita")
plt.ylabel('Life satisfaction')

plt.plot(list(sample_data["GDP per capita"]), list(sample_data["Life satisfaction"]), "bo")
plt.plot(list(missing_data["GDP per capita"]), list(missing_data["Life satisfaction"]), "rs")

X = np.linspace(0, 110000, 1000)
plt.plot(X, t0full + t1full * X, "r--", label="Linear model on all data")
plt.plot(X, t0 + t1*X, "b:", label="Linear model on partial data")

ridge = linear_model.Ridge(alpha=10**9.5)
Xsample = np.c_[sample_data["GDP per capita"]]
ysample = np.c_[sample_data["Life satisfaction"]]
ridge.fit(Xsample, ysample)
t0ridge, t1ridge = ridge.intercept_[0], ridge.coef_[0][0]
plt.plot(X, t0ridge + t1ridge * X, "b", label="Regularized linear model on partial data")

plt.legend(loc="lower right")
plt.axis([0, 110000, 0, 10])
plt.xlabel("GDP per capita (USD)")
save_fig('ridge_model_plot')
plt.show()

# + id="9kKhEwXMWk-q" colab_type="code" colab={}
backup = oecd_bli, gdp_per_capita

def prepare_country_stats(oecd_bli, gdp_per_capita):
    return sample_data


# + id="28Ohjz0QWk-v" colab_type="code" colab={}
# Replace this linear model:
import sklearn.linear_model
model = sklearn.linear_model.LinearRegression()

# + id="FBjwH4tTWk-1" colab_type="code" colab={}
# with this k-neighbors regression model:
import sklearn.neighbors
model = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)

# + id="6nA0AYPgWk-5" colab_type="code" colab={} outputId="2b8ff9b6-21f9-4cdf-a640-d66e2d913a54"
X = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life satisfaction"]]

# Train the model
model.fit(X, y)

# Make a prediction for Cyprus
X_new = np.array([[22587.0]])  # Cyprus' GDP per capita
print(model.predict(X_new)) # outputs [[ 5.76666667]]

# + id="rE5ohFikWk--" colab_type="code" colab={}

