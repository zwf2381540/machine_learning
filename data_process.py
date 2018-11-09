import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from collections import Counter
from dateutil.parser import parse
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os

DATA_PATH = "data"

columns_list = ['乙肝表面抗原', '乙肝表面抗体', '乙肝e抗原', '乙肝e抗体', '乙肝核心抗体']

def normalize(dataframe):
    scaler = MinMaxScaler()
    norm_arr = scaler.fit_transform(dataframe)
    return pd.DataFrame(norm_arr, columns=dataframe.columns)

def common_process(df):
    df = df[(df['性别'].isin(['男','女']))]
    df.loc[:, '体检日期'] = (pd.to_datetime(df.loc[:, '体检日期']) - parse('2017-10-09')).dt.days
    # df = pd.concat((df, pd.get_dummies(df[['性别']])), axis=1)
    # df = df.drop(['性别'], axis=1)
    df['性别'] = df['性别'].map({'男': 1, '女': 0})
    df = df.reset_index(drop=True)
    return df

def tsne_visual(x):
    X_tsne = TSNE(learning_rate=100).fit_transform(x.iloc[:, :-1])
    y = x['血糖'].map(lambda x: 0 if x < 8 else 1)
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y)
    plt.show()

def dbscane_cluster_find(x):
    cluster = DBSCAN(eps=0.6, min_samples=10)
    cluster.fit(x)
    print(set(cluster.labels_))
    print(len([i for i in cluster.labels_ if i == 1]))
    print(len([i for i in cluster.labels_ if i == 0]))
    print(len([i for i in cluster.labels_ if i == -1]))
    return pd.DataFrame(cluster.labels_, columns=['cluster_label'])

def kmeans_cluster_find(x):
    cluster = KMeans(n_clusters=2)
    cluster.fit(x)
    return pd.DataFrame(cluster.labels_, columns=['cluster_label'])

def abnormal_find_cluster(x):

    # cluster = DBSCAN(eps=10, min_samples=3)
    cluster = DBSCAN(eps=1.0, min_samples=3)
    cluster.fit(x)
    return [index for index, label in enumerate(cluster.labels_) if label==-1]

def abnormal_find_qvalue(df):
    abnormal_Counter = Counter()
    df_test = df.drop(['id', '年龄', '性别', '体检日期', '血糖'], axis=1)
    y_ = df[['血糖']]
    y_Q1 = np.nanpercentile(y_, 25)
    y_Q3 = np.nanpercentile(y_, 75)
    y_step = 2 * (y_Q3 - y_Q1)
    y_index = y_[~((y_['血糖'] >= y_Q1 - y_step) & (y_['血糖'] <= y_Q3 + y_step))].index
    # 对于每一个特征，找到值异常高或者是异常低的数据点
    for feature in df_test.keys():
        if feature == '*球蛋白' or feature == '*丙氨酸氨基转换酶':
            continue
        Q1 = np.nanpercentile(df_test[feature], 25)

        Q3 = np.nanpercentile(df_test[feature], 75)

        step = 3 * (Q3 - Q1)

        indexs = df_test[~(
            ((df_test[feature] >= Q1 - step) & (df_test[feature] <= Q3 + step)) | (
            df_test[feature].isnull() == True) | (
                df_test.index.isin(y_index)))].index
        abnormal_Counter.update(indexs)

    # 可选：选择你希望移除的数据点的索引
    return [index for index in abnormal_Counter if abnormal_Counter[index] > 3]
    # outliers = [572]

def preprocess(X_scaler, scaler):
    outliers = abnormal_find_cluster(X_scaler)
    print(len(outliers))
    # outliers = []
    # 如果选择了的话，移除异常点
    X_remove_outliers = np.delete(X_scaler, outliers, axis=0)
    X_process = X_remove_outliers[:, :-1]
    # y_process = X_remove_outliers[:, -1]
    y_process = scaler.inverse_transform(X_remove_outliers)[:, -1]

    X_submission = common_process(TEST_FILE, testing=True)
    X_submission = X_submission.fillna(X_submission.median())
    X_submission['y'] = np.ones((X_submission.shape[0], 1))
    X_submission = scaler.transform(X_submission)[:, :-1]

def get_data(path):
    train_file = os.path.join(path, "d_train_20180102.csv")
    test_file = os.path.join(path, "d_test_A_20180102.csv")
    train = pd.read_csv(train_file, encoding='gb2312')
    print(train.columns)
    test = pd.read_csv(test_file, encoding='gb2312')
    train_id = train.id.values.copy()
    test_id = test.id.values.copy()
    df_all = pd.concat((train, test), axis=0)
    df_all = common_process(df_all)
    df_nolabel = df_all[[i for i in df_all.columns if i != '血糖' and i != 'id' and i != '性别']]
    # 先用中位值填充
    df_nolabel = df_nolabel.fillna(df_nolabel.median(axis=0))

    # 对除了血糖和id的值进行归一化
    df_nolabel_norm = normalize(df_nolabel)
    df_nolabel_norm_partial = df_nolabel_norm[['白细胞计数', '红细胞计数', '血红蛋白', '红细胞压积', '红细胞平均体积',
       '红细胞平均血红蛋白量', '红细胞平均血红蛋白浓度', '红细胞体积分布宽度', '血小板计数', '血小板平均体积',
       '血小板体积分布宽度', '血小板比积', '中性粒细胞%', '淋巴细胞%', '单核细胞%', '嗜酸细胞%', '嗜碱细胞%']]
    # df_all_norm = pd.concat((df_nolabel_norm, df_all[['id', '血糖']]), axis=1)
    # df_train_norm = df_all_norm[df_all_norm.id.isin(train_id)]
    # # t-sne聚类可视化
    # tsne_visual(df_train_norm.drop('id', axis=1))
    # cluster_label = dbscane_cluster_find(df_nolabel_norm)
    cluster_label = kmeans_cluster_find(df_nolabel_norm_partial)
    df_all_with_cluster = pd.concat((df_nolabel_norm, cluster_label, df_all[['id', '血糖', '性别']]), axis=1)
    test_index = []
    for index, row in df_all_with_cluster.iterrows():
        if int(row['cluster_label']) == -1:
            if row['id'] in test_id:
                test_index.append(index)
    df_all_with_cluster.ix[test_index, 'cluster_label'] = 0
    df_all_with_cluster = df_all_with_cluster[df_all_with_cluster['cluster_label'] != -1]
    # print(len(df_all_with_cluster[df_all_with_cluster['cluster_label'] == -1]))
    train_feat = df_all_with_cluster[df_all_with_cluster.id.isin(train_id)]
    test_feat = df_all_with_cluster[df_all_with_cluster.id.isin(test_id)]
    train_feat = train_feat.drop(['id'], axis=1)
    test_feat = test_feat.drop(['id'], axis=1)

    return train_feat, test_feat

# get_data(DATA_PATH)