################################
# K-Means Project
################################

import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from helpers.data_prep import *

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from pandas.core.common import SettingWithCopyWarning
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)


df_ = pd.read_excel("datasets/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()

###############################################################
# RFM
###############################################################

def create_rfm(dataframe):
    # VERIYI HAZIRLAMA
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]

    # RFM METRIKLERININ HESAPLANMASI
    today_date = dt.datetime(2011, 12, 11)
    rfm = dataframe.groupby('Customer ID').agg({'InvoiceDate': lambda date: (today_date - date.max()).days,
                                                'Invoice': lambda num: num.nunique(),
                                                "TotalPrice": lambda price: price.sum()})
    rfm.columns = ['recency', 'frequency', "monetary"]
    rfm = rfm[(rfm['monetary'] > 0)]

    # RFM SKORLARININ HESAPLANMASI
    rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
    rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

    # cltv_df skorları kategorik değere dönüştürülüp df'e eklendi
    rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) +
                        rfm['frequency_score'].astype(str))

    # SEGMENTLERIN ISIMLENDIRILMESI
    seg_map = {
        r'[1-2][1-2]': 'hibernating',
        r'[1-2][3-4]': 'at_risk',
        r'[1-2]5': 'cant_loose',
        r'3[1-2]': 'about_to_sleep',
        r'33': 'need_attention',
        r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promising',
        r'51': 'new_customers',
        r'[4-5][2-3]': 'potential_loyalists',
        r'5[4-5]': 'champions'
    }

    rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)
    rfm = rfm[["recency", "frequency", "monetary", "segment"]]
    return rfm


rfm = create_rfm(df)

###############################################################
# K-Means Clustering
###############################################################

# Min - Max Scaler
scaler = MinMaxScaler()
segment_data = pd.DataFrame(scaler.fit_transform(rfm[["recency", "frequency", "monetary"]]),
                            index=rfm.index, columns=["Recency_n", "Frequency_n", "Monetary_n"])
segment_data.head()

################################
# Optimum Küme Sayısının Belirlenmesi - Automatic
################################

kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(segment_data)
# distortuion: Her noktanın atandığı merkezine olan uzaklıkların karelerinin toplamını temsil etmektedir.
elbow.show()
kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(segment_data)
segment_data["clusters"] = kmeans.labels_
print(f"Number of cluster selected: {elbow.elbow_value_}")

################################
# Optimum Küme Sayısının Belirlenmesi - Manual
################################

segment_data = pd.DataFrame(scaler.fit_transform(rfm[["recency", "frequency", "monetary"]]),
                            index=rfm.index, columns=["Recency_n", "Frequency_n", "Monetary_n"])

kmeans = KMeans()
ssd = []
K = range(1, 16)

for k in K:
    kmeans = KMeans(n_clusters=k).fit(segment_data)
    ssd.append(kmeans.inertia_)

ssd

plt.plot(K, ssd, "bx-")
plt.xlabel("Farklı K Değerlerine Karşılık Uzaklık Artık Toplamları")
plt.title("Optimum Küme sayısı için Elbow Yöntemi")
plt.show()

# plt.savefig("elbow-graph.png", dpi=250)

################################
# Final Cluster'ların Oluşturulması
################################

segment_data = pd.DataFrame(scaler.fit_transform(rfm[["recency", "frequency", "monetary"]]),
                            index=rfm.index, columns=["Recency_n", "Frequency_n", "Monetary_n"])

kmeans = KMeans(n_clusters=6).fit(segment_data)
segment_data["clusters"] = kmeans.labels_

################################
# RFM ve K-Means Clusterlarının Birleştirilmesi
################################

segmentation = rfm.merge(segment_data, on="Customer ID")
seg_desc = segmentation[["segment", "clusters", "recency", "frequency", "monetary"]].groupby(["clusters", "segment"]).agg(["mean", "count"])
print(seg_desc)
segmentation.to_csv("segmentation.csv")

################################
# BONUS: Cluster'ların Görselleştirilmesi
################################

# Matplotlib

x = segment_data[['Recency_n', 'Frequency_n', 'Monetary_n']].values
model = KMeans(n_clusters=kmeans.n_clusters, random_state=0)
y_clusters = model.fit_predict(x)

# Random renk seçimi
import random

number_of_colors = model.n_clusters
color = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
         for i in range(number_of_colors)]

fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(111, projection='3d')
for i in range(0, model.n_clusters):
    ax.scatter(x[y_clusters == i, 0], x[y_clusters == i, 1], x[y_clusters == i, 2], s=40, color=color[i],
               label=f"cluster {i}")

ax.set_xlabel('Recency')
ax.set_ylabel('Frequency')
ax.set_zlabel('Monetary')
# fig.savefig(f"{model.n_clusters}-cluster.png", dpi=250)
plt.show()

# Plotly
# !pip install plotly
# Reference: https://www.kaggle.com/naren3256/kmeans-clustering-and-cluster-visualization-in-3d
import plotly.graph_objs as go
import plotly.io as pio
x = segment_data[['Recency_n', 'Frequency_n', 'Monetary_n']].values
model = KMeans(n_clusters=6, random_state=0).fit(x)
pio.renderers.default = "browser" #IDE üzerinde oluşturulan grafiği gösterebilmek için.

Scene = dict(xaxis=dict(title='Recency -->'), yaxis=dict(title='Frequency--->'), zaxis=dict(title='Monetary-->'))

labels = model.labels_
trace = go.Scatter3d(x=x[:, 0], y=x[:, 1], z=x[:, 2], mode='markers',
                     marker=dict(color=labels, size=10, line=dict(color='black', width=10)))
layout = go.Layout(margin=dict(l=0, r=0), scene=Scene, height=800, width=800)
data = [trace]
fig = go.Figure(data=data, layout=layout)
fig.show()
fig.write_image("fig1.svg")
fig.write_html("file.html")
