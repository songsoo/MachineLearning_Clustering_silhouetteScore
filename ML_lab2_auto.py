import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, cm
from sklearn import preprocessing, mixture, metrics
from sklearn.cluster import KMeans, DBSCAN, MeanShift, estimate_bandwidth
from sklearn.metrics import silhouette_score, silhouette_samples

#pd.set_option('display.max_columns',None)
#pd.set_option('display.max_rows',None)


def getEncode(df,name,encoder):
    encoder.fit(df[name])
    labels = encoder.transform(df[name])
    df.loc[:, name] = labels

def onehotEncode(df, name):
   le = preprocessing.OneHotEncoder(handle_unknown='ignore')
   enc = df[[name]]
   enc = le.fit_transform(enc).toarray()
   enc_df = pd.DataFrame(enc, columns=le.categories_[0])
   df.loc[:, le.categories_[0]] = enc_df
   df.drop(columns=[name], inplace=True)

#label encoding
def labelEncode(df, name):
    encoder = preprocessing.LabelEncoder()
    encoder.fit(df[name])
    labels = encoder.transform(df[name])
    df.loc[:, name] = labels

"""
:param X: feature values
:param numerical_columns: name of numerical columns (array of string)
:param categorical_columns: name of categorical columns (array of string)
:param scalers: array of scalers
:param encoders: array of encoders 
:param scaler_name: name of scalers (array of string)
:param encoder_name: name of encoders (array of string)
:return: 2d array that is scaled and encoded X 
"""
def get_various_encode_scale(X, numerical_columns, categorical_columns, scalers=None, encoders= None,scaler_name=None,encoder_name=None):

    if categorical_columns is None:
        categorical_columns = []
    if numerical_columns is None:
        numerical_columns = []
    if len(categorical_columns) == 0:
        return get_various_scale(X,numerical_columns,scalers,scaler_name)
    if len(numerical_columns) == 0:
        return get_various_encode(X,categorical_columns,encoders,encoder_name)

    """
    Test scale/encoder sets
    """
    if scalers is None:
        #scalers = [preprocessing.StandardScaler(), preprocessing.MinMaxScaler(), preprocessing.RobustScaler()]
        scalers = [preprocessing.StandardScaler()]
    if encoders is None:
        #encoders = [preprocessing.LabelEncoder(),preprocessing.OneHotEncoder()]
        encoders = [preprocessing.LabelEncoder()]

    after_scale_encode = [[0 for col in range(len(encoders))] for row in range(len(scalers))]

    i=0
    for scale in scalers:
        for encode in encoders:
            after_scale_encode[i].pop()
        for encode in encoders:
            after_scale_encode[i].append(X.copy())
        i=i+1

    for name in numerical_columns:
        i=0
        for scaler in scalers:
            j=0
            for encoder in encoders:
                after_scale_encode[i][j][name] = scaler.fit_transform(X[name].values.reshape(-1, 1))
                j=j+1
            i=i+1

    for new in categorical_columns:
        i=0
        for scaler in scalers:
            j=0
            for encoder in encoders:
                if (str(type(encoder)) == "<class 'sklearn.preprocessing._label.LabelEncoder'>"):
                    labelEncode(after_scale_encode[i][j], new)
                elif (str(type(encoder)) == "<class 'sklearn.preprocessing._encoders.OneHotEncoder'>"):
                    onehotEncode(after_scale_encode[i][j], new)
                else:
                    getEncode(after_scale_encode[i][j], new, encoder)
                j=j+1
            i=i+1

    return after_scale_encode,scalers,encoders

"""
If there aren't categorical value, do this function
This function only scales given X
Return: 1d array of scaled X
"""
def get_various_scale(X, numerical_columns, scalers=None,scaler_name=None):

    """
    Test scale/encoder sets
    """
    if scalers is None:
        scalers = [preprocessing.StandardScaler(), preprocessing.MinMaxScaler(), preprocessing.RobustScaler()]
        #scalers = [preprocessing.StandardScaler()]
    encoders = ["None"]

    after_scale = [[0 for col in range(1)] for row in range(len(scalers))]

    i = 0
    for scale in scalers:
        for encode in encoders:
            after_scale[i].pop()
        for encode in encoders:
            after_scale[i].append(X.copy())
        i = i + 1

    for name in numerical_columns:
       i=0
       for scaler in scalers:
           after_scale[i][0][name] = scaler.fit_transform(X[name].values.reshape(-1,1))
           i=i+1

    return after_scale,scalers,["None"]

"""
If there aren't numerical value, do this function
This function only encodes given X
Return: 1d array of encoded X
"""
def get_various_encode(X, categorical_columns, encoders=None,encoder_name=None):

    """
    Test scale/encoder sets
    """
    if encoders is None:
        encoders = [preprocessing.LabelEncoder(),preprocessing.OneHotEncoder()]
        #encoders = [preprocessing.LabelEncoder()]
    scalers = ["None"]

    after_encode = [[0 for col in range(1)] for row in range(len(scalers))]

    i = 0
    for scale in scalers:
        for encode in encoders:
            after_encode[i].pop()
        for encode in encoders:
            after_encode[i].append(X.copy())
        i = i + 1

    for new in categorical_columns:
        j = 0
        for encoder in encoders:
            if (str(type(encoder)) == "<class 'sklearn.preprocessing._label.LabelEncoder'>"):
                labelEncode(after_encode[0][j], new)
            elif (str(type(encoder)) == "<class 'sklearn.preprocessing._encoders.OneHotEncoder'>"):
                onehotEncode(after_encode[0][j], new)
            else:
                getEncode(after_encode[0][j], new, encoder)
            j = j + 1


    return after_encode,["None"],encoders

"""
:param X: dataset
:param max_cluster: maximum number of clusters
:param n_inits: Number of time the k-means algorithm will be run with different centroid seeds.
:param max_iters: Maximum number of iterations of the k-means algorithm for a single run
:param tols: Relative tolerance with regards to Frobenius norm of the difference in the cluster centers of two consecutive iterations to declare convergence.
:param verboses: Verbosity mode.
:param random_state
"""
def kmeans(X,y,max_cluster=None, n_inits=None, max_iters=None, tols=None, verboses=None, random_state=None):

    if max_cluster is None:
        max_cluster = 7
    max_cluster = max_cluster + 1

    range_n_clusters = list(range(max_cluster))
    range_n_clusters.remove(0)
    range_n_clusters.remove(1)

    if n_inits is None:
        n_inits = [10]
    if max_iters is None:
        max_iters = [300]
    if tols is None:
        tols = [1e-4]
    if verboses is None:
        verboses = [0]

    best_cluster = -1
    best_silhouette= -1
    best_n_inits = 0
    best_max_iters = 0
    best_tols = 0
    best_verboses = 0

    centerDF = pd.DataFrame

    for n_clusters in range_n_clusters:
        for n_init in n_inits:
            for max_iter in max_iters:
                for tol in tols:
                    for verbose in verboses:
                        print("number of clusters: ", n_clusters,"/ n_init:", n_init,"/ max_iter:", max_iter,"/ tol:", tol,"/ verbose:", verbose)

                        fig, (ax1, ax2) = plt.subplots(1, 2)
                        fig.set_size_inches(18, 7)

                        ax1.set_xlim([-0.1, 1])
                        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

                        #cluster_labels: row마다 어떤 군집으로 분류되었는지 라벨링
                        clusterer = KMeans(n_clusters=n_clusters,n_init=n_init,max_iter=max_iter,tol=tol,verbose=verbose, random_state=random_state)
                        cluster_labels = clusterer.fit_predict(X)

                        silhouette_avg = silhouette_score(X, cluster_labels)
                        centers = clusterer.cluster_centers_

                        if best_silhouette<silhouette_avg:
                            best_silhouette = silhouette_avg
                            best_cluster = n_clusters
                            best_n_inits = n_init
                            best_max_iters = max_iter
                            best_tols = tol
                            best_verboses = verbose

                            sum = [0 for row in range(n_clusters)]
                            num = [0 for row in range(n_clusters)]

                            j = 0
                            for i in cluster_labels:
                                sum[i] = sum[i] + y[j]
                                num[i] = num[i] + 1
                                j = j + 1

                            for i in range(n_clusters):
                                sum[i] = sum[i] / num[i]
                            centerDF = pd.DataFrame(centers)
                            centerDF.loc[:, 'Mean House Value'] = sum


                        print("The average silhouette_score is :", silhouette_avg)

                        sample_silhouette_values = silhouette_samples(X, cluster_labels)

                        y_lower = 10
                        for i in range(n_clusters):

                            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

                            ith_cluster_silhouette_values.sort()

                            size_cluster_i = ith_cluster_silhouette_values.shape[0]
                            y_upper = y_lower + size_cluster_i

                            color = cm.nipy_spectral(float(i) / n_clusters)
                            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                                              0, ith_cluster_silhouette_values,
                                              facecolor=color, edgecolor=color, alpha=0.7)

                            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                            y_lower = y_upper + 10  # 10 for the 0 samples

                        ax1.set_title("Silouette Plot")
                        ax1.set_xlabel("Silhouette coefficient")
                        ax1.set_ylabel("Cluster label")

                        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

                        ax1.set_yticks([])  # Clear the yaxis labels / ticks
                        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

                        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
                        ax2.scatter(X.iloc[:, 0], X.iloc[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                                    c=colors, edgecolor='k')

                        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',c="white", alpha=1, s=200, edgecolor='k')

                        for i, c in enumerate(centers):
                            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,s=50, edgecolor='k')

                        ax2.set_title("Cluster")
                        ax2.set_xlabel("1st Column")
                        ax2.set_ylabel("2nd Column")

                        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data ", "with n_clusters = %d" % n_clusters),fontsize=14, fontweight='bold')

    #plt.show()
    df = centerDF.copy()
    print("\nThe highest silhouette score is ", best_silhouette, " with ",best_cluster," clusters")
    print("Best params_/ n_init:",best_n_inits,"/ max_iter:",best_max_iters,"/ tol:",best_tols,"/ verbose:",best_verboses,"\n")
    param = 'Best params_/ best cluster: '+str(best_cluster)+ '/ n_init: '+str(best_n_inits)+' / max_iter: '+str(best_max_iters)+'/ tol: '+str(best_tols)+'/ verbose: '+str(best_verboses)
    return best_silhouette, param, df
"""
:param X: dataset
:param max_cluster: maximum number of clusters
:param covariance_types: String describing the type of covariance parameters to use. 
:param tols: The convergence threshold.
:param max_iters: The number of initializations to perform.
:param n_inits: The number of initializations to perform.
:param random_state
"""
def GMM(X,y,max_cluster=None,covariance_types=None,tols=None,max_iters=None,n_inits=None,random_state=None):

    if max_cluster is None:
        max_cluster = 7
    max_cluster = max_cluster + 1

    if covariance_types is None:
        covariance_types = ['full']
    if tols is None:
        tols = [1e-3]
    if max_iters is None:
        max_iters = [100]
    if n_inits is None:
        n_inits = [1]

    range_n_clusters = list(range(max_cluster))
    range_n_clusters.remove(0)
    range_n_clusters.remove(1)

    best_cluster = -1
    best_silhouette = -1
    best_covariance_type = ''
    best_tol = 0
    best_max_iter = 0
    best_n_init = 0

    centerDF = pd.DataFrame

    for n_clusters in range_n_clusters:
        for covariance_type in covariance_types:
            for tol in tols:
                for max_iter in max_iters:
                        for n_init in n_inits:
                            print("number of clusters: ", n_clusters, "/ covariance type:", covariance_type, "/ n_init:", n_init, "/ max_iter:", max_iter,
                                  "/ tol:", tol)

                            fig, (ax1, ax2) = plt.subplots(1, 2)
                            fig.set_size_inches(18, 7)

                            ax1.set_xlim([-0.1, 1])
                            ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

                            clusterer = mixture.GaussianMixture(n_components=n_clusters, covariance_type=covariance_type,tol=tol,max_iter=max_iter,n_init=n_init)
                            clusterer.fit(X)
                            cluster_labels = clusterer.predict(X)

                            silhouette_avg = silhouette_score(X, cluster_labels)
                            print("The average silhouette_score is :", silhouette_avg)

                            # Labeling the clusters
                            centers = clusterer.means_

                            if best_silhouette<silhouette_avg:
                                best_silhouette = silhouette_avg
                                best_cluster = n_clusters
                                best_covariance_type = covariance_type
                                best_tol = tol
                                best_max_iter = max_iter
                                best_n_init = n_init

                                sum = [0 for row in range(n_clusters)]
                                num = [0 for row in range(n_clusters)]

                                j = 0
                                for i in cluster_labels:
                                    sum[i] = sum[i] + y[j]
                                    num[i] = num[i] + 1
                                    j = j + 1

                                for i in range(n_clusters):
                                    sum[i] = sum[i] / num[i]
                                centerDF = pd.DataFrame(centers)
                                centerDF.loc[:, 'Mean House Value'] = sum

                            sample_silhouette_values = silhouette_samples(X, cluster_labels)

                            y_lower = 10
                            for i in range(n_clusters):
                                ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

                                ith_cluster_silhouette_values.sort()

                                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                                y_upper = y_lower + size_cluster_i

                                color = cm.nipy_spectral(float(i) / n_clusters)
                                ax1.fill_betweenx(np.arange(y_lower, y_upper),0, ith_cluster_silhouette_values,facecolor=color, edgecolor=color, alpha=0.7)

                                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                                y_lower = y_upper + 10  # 10 for the 0 samples

                            ax1.set_title("Silouette Plot")
                            ax1.set_xlabel("Silhouette coefficient")
                            ax1.set_ylabel("Cluster label")

                            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

                            ax1.set_yticks([])  # Clear the yaxis labels / ticks
                            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

                            # 2nd Plot showing the actual clusters formed
                            colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
                            ax2.scatter(X.iloc[:, 0], X.iloc[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                                        c=colors, edgecolor='k')

                            # Draw white circles at cluster centers
                            ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                                        c="white", alpha=1, s=200, edgecolor='k')

                            for i, c in enumerate(centers):
                                ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                                            s=50, edgecolor='k')

                            ax2.set_title("Cluster")
                            ax2.set_xlabel("1st Column")
                            ax2.set_ylabel("2nd Column")

                            plt.suptitle(("Silhouette analysis for GMM on sample data "
                                          "with n_clusters = %d" % n_clusters),
                                         fontsize=14, fontweight='bold')

    #plt.show()

    print("\nThe highest silhouette score is ", best_silhouette, " with ", best_cluster, " clusters")
    print("Best params_/ covariance_types:", covariance_type, "/ max_iter:", best_max_iter, "/ tol:", best_tol, "/ n_init:",
          best_n_init,"\n")
    param = "Best params_/ cluster: "+str(best_cluster)+"/ covariance_types:"+ covariance_type+ "/ max_iter:"+ str(best_max_iter)+ "/ tol:"+ str(best_tol)+ "/ n_init:"+str(best_n_init)
    return best_silhouette, param, centerDF

"""
:param X: dataset
:param max_cluster: maxinum number of clusters
:param numlocals: The number of local minima obtained
:param maxneighbors: The maximum number of neighbors examined
"""
def clarans(X,y,max_cluster=None, numlocals=None,maxneighbors=None):

    from pyclustering.cluster.clarans import clarans;

    if max_cluster is None:
        max_cluster = 7
    max_cluster = max_cluster + 1

    if numlocals is None:
        numlocals = [2]
    if maxneighbors is None:
        maxneighbors = [2]

    range_n_clusters = list(range(max_cluster))
    range_n_clusters.remove(0)
    range_n_clusters.remove(1)

    best_cluster = -1
    best_silhouette = -1
    best_numlocal = 0
    best_maxneighbor = 0

    for n_clusters in range_n_clusters:
        for numlocal in numlocals:
            for maxneighbor in maxneighbors:
                X = X.iloc[0:500,:]
                X = X.values.tolist()

                print("number of clusters: ", n_clusters, "/ numlocal:", numlocal, "/ maxneighbor:", maxneighbor)

                fig, (ax1, ax2) = plt.subplots(1, 2)
                fig.set_size_inches(18, 7)

                ax1.set_xlim([-0.1, 1])
                ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

                clarans_instance = clarans(X, n_clusters, numlocal, maxneighbor);

                clarans_instance.process()
                clusters = clarans_instance.get_clusters();

                i = 0
                a = []
                for cluster in clusters:
                    for index in cluster:
                        a.insert(index, i)
                    i = i + 1

                cluster_labels = np.array(a)

                silhouette_avg = silhouette_score(X, cluster_labels)
                print("The average silhouette_score is :", silhouette_avg)

                if best_silhouette<silhouette_avg:
                    best_silhouette = silhouette_avg
                    best_cluster = n_clusters
                    best_numlocal = numlocal
                    best_maxneighbor = maxneighbor

                sample_silhouette_values = silhouette_samples(X, cluster_labels)

                y_lower = 10
                for i in range(n_clusters):
                    ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

                    ith_cluster_silhouette_values.sort()

                    size_cluster_i = ith_cluster_silhouette_values.shape[0]
                    y_upper = y_lower + size_cluster_i

                    color = cm.nipy_spectral(float(i) / n_clusters)
                    ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color,
                                      edgecolor=color, alpha=0.7)

                    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                    y_lower = y_upper + 10  # 10 for the 0 samples

                ax1.set_title("Silouette Plot")
                ax1.set_xlabel("Silhouette coefficient")
                ax1.set_ylabel("Cluster label")

                ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

                ax1.set_yticks([])
                ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

                colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
                X = pd.DataFrame(X)
                ax2.scatter(X.iloc[:, 0], X.iloc[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                            c=colors, edgecolor='k')

                ax2.set_title("Cluster")
                ax2.set_xlabel("1st Column")
                ax2.set_ylabel("2nd Column")

                plt.suptitle(("Silhouette analysis for CLARANS clustering on sample data "
                              "with n_clusters = %d" % n_clusters),
                             fontsize=14, fontweight='bold')

    #plt.show()

    print("\nThe highest silhouette score is ", best_silhouette, " with ", best_cluster, " clusters")
    print("Best params_/ numlocal:", best_numlocal, "/ max_neighbor:", best_maxneighbor,"\n")
    param = "Best params_/ cluster: "+str(best_cluster)+"/ numlocal:"+ str(best_numlocal)+ "/ max_neighbor:"+str(best_maxneighbor)
    return best_silhouette, param

"""
:param X: datasets
:param epsS: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
:param min_samples: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
:param metrics: The metric to use when calculating distance between instances in a feature array
:param algorithms: The algorithm to be used by the NearestNeighbors module to compute pointwise distances and find nearest neighbors
:param leaf_sizes
 """
def DBSCANs(X,y, epsS=None,min_samples=None,metrics=None,algorithms=None,leaf_sizes=None):

    if epsS is None:
        epsS = [0.8]
    if min_samples is None:
        min_samples = [3]
    if metrics is None:
        metrics = ['euclidean']
    if algorithms is None:
        algorithms = ['auto']
    if leaf_sizes is None:
        leaf_sizes = [30]

    best_silhouette = -1
    best_cluster = -1
    best_eps = 0
    best_min_sample=0
    best_metric = ''
    best_algorithm = ''
    best_leaf_size = 0

    centerDF = pd.DataFrame

    for eps in epsS:
        for min_sample in min_samples:
            for metric in metrics:
                for algorithm in algorithms:
                    for leaf_size in leaf_sizes:
                        np.set_printoptions(threshold=100000,linewidth=np.inf)

                        fig, (ax1, ax2) = plt.subplots(1, 2)
                        fig.set_size_inches(18, 7)

                        clusterer = DBSCAN(eps=eps,min_samples=min_sample,metric=metric,algorithm=algorithm,leaf_size=leaf_size).fit(X)
                        cluster_labels = clusterer.labels_

                        n_clusters = len(set(clusterer.labels_))

                        unique_set = set(clusterer.labels_)
                        unique_list = list(unique_set)
                        if unique_list.count(-1):
                            unique_list.remove(-1)

                        a = np.array([[0 for col in range(len(X.iloc[0,:]))] for row in range(len(set(unique_list)))])
                        num = np.array([0 for row in range(len(set(unique_list)))])

                        i = 0
                        for cluster in cluster_labels:
                            if (cluster != -1):
                                a[cluster] = a[cluster] + X.iloc[i,:]
                                num[cluster] = num[cluster] + 1
                            i = i + 1

                        i = 0

                        for cluster in unique_list:
                            a[cluster] = a[cluster] / num[cluster]

                        ax1.set_xlim([-0.1, 1])
                        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

                        silhouette_avg = silhouette_score(X, cluster_labels)
                        print("number of clusters: ", n_clusters, "/ eps:", eps, "/ min_sample:", min_sample,
                              "/ metric:", metric, "/ algorithm:", algorithm, "/ leaf_size:", leaf_size)
                        print("The average silhouette_score is :", silhouette_avg)

                        centers = np.array(a)


                        if best_silhouette < silhouette_avg:
                            best_silhouette = silhouette_avg
                            best_cluster = n_clusters
                            best_eps = eps
                            best_metric = metric
                            best_algorithm = algorithm
                            best_leaf_size = leaf_size
                            best_min_sample = min_sample

                            sum = [0 for row in range(n_clusters)]
                            num = [0 for row in range(n_clusters)]
                            j = 0
                            for i in cluster_labels:
                                if i>=0:
                                    sum[i] = sum[i] + y[j]
                                    num[i] = num[i] + 1
                                    j = j + 1

                            for i in range(n_clusters):
                                if num[i]!=0:
                                    sum[i] = sum[i] / num[i]
                            centerDF = pd.DataFrame(centers)
                            sum.pop()
                            centerDF.loc[:, 'Mean House Value'] = sum

                        sample_silhouette_values = silhouette_samples(X, cluster_labels)

                        y_lower = 10
                        for i in range(n_clusters):
                            ith_cluster_silhouette_values = \
                                sample_silhouette_values[cluster_labels == i]

                            ith_cluster_silhouette_values.sort()

                            size_cluster_i = ith_cluster_silhouette_values.shape[0]
                            y_upper = y_lower + size_cluster_i

                            color = cm.nipy_spectral(float(i) / n_clusters)
                            ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color,
                                              edgecolor=color, alpha=0.7)

                            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                            y_lower = y_upper + 10  # 10 for the 0 samples

                        ax1.set_title("Silouette Plot")
                        ax1.set_xlabel("Silhouette coefficient")
                        ax1.set_ylabel("Cluster label")

                        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

                        ax1.set_yticks([])  # Clear the yaxis labels / ticks
                        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

                        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
                        ax2.scatter(X.iloc[:, 0], X.iloc[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                                    c=colors, edgecolor='k')

                        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                                    c="white", alpha=1, s=200, edgecolor='k')

                        for i, c in enumerate(centers):
                            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                                        s=50, edgecolor='k')

                        ax2.set_title("Cluster")
                        ax2.set_xlabel("1st Column")
                        ax2.set_ylabel("2nd Column")

                        plt.suptitle(("Silhouette analysis for DBSCAN clustering on sample data "
                                      "with n_clusters = %d" % n_clusters),
                                     fontsize=14, fontweight='bold')


    #plt.show()
    print("\nThe highest silhouette score is ", best_silhouette, " with ", best_cluster, " clusters")
    print("Best params_/ eps:", best_eps, "/ min_sample:", best_min_sample,"/ metric:", best_metric,"/ algorithm:", best_algorithm,"/ leaf_size:", best_leaf_size,"\n")
    param = "Best params_/ cluster: "+str(best_cluster)+ "/ eps:"+ str(best_eps)+ "/ min_sample:", str(best_min_sample)+"/ metric:"+ best_metric+"/ algorithm:"+ best_algorithm+"/ leaf_size:"+ str(best_leaf_size)
    return best_silhouette, param, centerDF
"""
:param X: dataset
:param bandwidths: bandwidth used in the RBF kernel 
:param max_iters: Maximum numer of iteration
:param n_job: The number of jobs to use for the computation.
"""
def MeanShifts(X,y,bandwidths=None,max_iters=None,n_job=None):

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    if bandwidths is None:
        bandwidths = [estimate_bandwidth(X, quantile=0.75)]
    if max_iters is None:
        max_iters = [300]
    if n_job is None:
        n_job = -1

    best_silhouette = -1
    best_cluster = -1
    best_max_iter = 0
    best_bandwidth = 0

    centerDF = pd.DataFrame

    for bandwidth in bandwidths:
        for max_iter in max_iters:

            clusterer = MeanShift(bandwidth=bandwidth,max_iter=max_iter,n_jobs=n_job)
            clusterer.fit(X)
            cluster_labels = clusterer.labels_
            n_clusters = len(clusterer.cluster_centers_)

            ax1.set_xlim([-0.1, 1])
            ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

            silhouette_avg = silhouette_score(X, cluster_labels)
            print("number of clusters: ", n_clusters, "/ bandwidth:", bandwidth, "/ max_iter:", max_iter)
            print("The average silhouette_score is :", silhouette_avg)

            centers = clusterer.cluster_centers_

            if best_silhouette < silhouette_avg:
                best_silhouette = silhouette_avg
                best_cluster = n_clusters
                best_bandwidth = bandwidth
                best_max_iter = max_iter

                sum = [0 for row in range(n_clusters)]
                num = [0 for row in range(n_clusters)]

                j = 0
                for i in cluster_labels:
                    sum[i] = sum[i] + y[j]
                    num[i] = num[i] + 1
                    j = j + 1

                for i in range(n_clusters):
                    sum[i] = sum[i] / num[i]
                centerDF = pd.DataFrame(centers)
                centerDF.loc[:, 'Mean House Value'] = sum

            sample_silhouette_values = silhouette_samples(X, cluster_labels)

            y_lower = 10
            for i in range(n_clusters):
                ith_cluster_silhouette_values = \
                    sample_silhouette_values[cluster_labels == i]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / n_clusters)
                ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color,
                                  edgecolor=color, alpha=0.7)

                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                y_lower = y_upper + 10  # 10 for the 0 samples
            ax1.set_title("Silouette Plot")
            ax1.set_xlabel("Silhouette coefficient")
            ax1.set_ylabel("Cluster label")

            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

            colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)

            ax2.scatter(X.iloc[:, 0], X.iloc[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                        c=colors, edgecolor='k')
            
            ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                        c="white", alpha=1, s=200, edgecolor='k')

            for i, c in enumerate(centers):
                ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                            s=50, edgecolor='k')

            ax2.set_title("Cluster")
            ax2.set_xlabel("1st Column")
            ax2.set_ylabel("2nd Column")


            plt.suptitle(("Silhouette analysis for MeanShift clustering on sample data "
                          "with n_clusters = %d" % n_clusters),
                         fontsize=14, fontweight='bold')
            df = centerDF.copy()

    #plt.show()
    print("\nThe highest silhouette score is ", best_silhouette, " with ", best_cluster, " clusters")
    print("Best params_/ bandwidth:", best_bandwidth, "/ max_iter:", best_max_iter,"\n")
    param = "Best params_/ bandwidth:"+ str(best_bandwidth)+ "/ max_iter:"+ str(best_max_iter)
    return best_silhouette, param,df

"""
Do clustering and get silhouette index for various combinations of parameters
do scale and encode dataset with given scalers and encoders (select features to use for this program)
do kmeans, GMM, CLARANS, DBSCAN, MeanShift clustering to all combination datasets of encoders and scalers
do clustering with all combinations of parameters(including 'k' number of clusters)
show every result (silhouette index) and show parameters with the most highest silhouette index
"""
def findBest(or_data,y,numerical_columns,categorical_columns,max_cluster=None,n_inits=None,max_iters=None,tols=None,verboses=None,covariance_types=None,
                      numlocals=None,max_neighbors=None,epsS=None,min_samples=None,metrics=None,algorithms=None,leaf_sizes=None,bandwidths=None,n_job=None):



    kmeans_best = [-1,'scale','encode','params']
    GMM_best = [-1,'scale','encode','params']
    CLARANS_best = [-1,'scale','encode','params']
    DBSCAN_best = [-1,'scale','encode','params']
    MeanShift_best = [-1,'scale','encode','params']
    silhouette_score = 0
    params=""
    kmeans_centerDF_ex = pd.DataFrame()
    GMM_centerDF_ex = pd.DataFrame()
    DBSCAN_centerDF_ex = pd.DataFrame()
    MeanShift_centerDF_ex = pd.DataFrame()

    for numerical_column,categorical_column in zip(numerical_columns,categorical_columns):

        print("columns: ",numerical_column, categorical_column)

        total_columns = numerical_column + categorical_column + ['Mean_House_Value']
        x = pd.DataFrame()
        data = or_data.copy()

        for numerical_column_ind in numerical_column:
            x.loc[:, numerical_column_ind] = data.loc[:, numerical_column_ind]
        for categorical_column_ind in categorical_column:
            x.loc[:, categorical_column_ind] = data.loc[:, categorical_column_ind]

        x, scalers, encoders = get_various_encode_scale(x, numerical_column, categorical_column)
        i = 0
        for scaler in scalers:
            j = 0
            for encoder in encoders:
                print(scaler, encoder)
                print("--------Kmeans--------")
                """
                Test kmeans
                """
                #silhouette_score,params = kmeans(x[i][j],y, max_cluster=max_cluster, n_inits=n_inits,max_iters=max_iters,tols=tols,verboses=verboses)
                silhouette_score,params,kmeans_centerDF_ex = kmeans(x[i][j],y,3)
                if silhouette_score > kmeans_best[0]:
                    kmeans_best[0] = silhouette_score
                    kmeans_best[1] = scaler
                    kmeans_best[2] = encoder
                    kmeans_best[3] = params
                    kmeans_centerDF = kmeans_centerDF_ex.copy()
                print("--------GMM--------")
                """
                Test GMM parameter
                """
                #silhouette_score, params = GMM(x[i][j],y, max_cluster=max_cluster, covariance_types=covariance_types,tols=tols,max_iters=max_iters,n_inits=n_inits)
                #silhouette_score, params, GMM_centerDF_ex = GMM(x[i][j],y, 3)
                if silhouette_score > GMM_best[0]:
                    GMM_best[0] = silhouette_score
                    GMM_best[1] = scaler
                    GMM_best[2] = encoder
                    GMM_best[3] = params
                    #GMM_centerDF = GMM_centerDF_ex.copy()
                print("--------CLARANS--------")
                """
                Test CLARANS parameter 
                do not run CLARANS because it takes too much time
                """
                #silhouette_score, params = clarans(x[i][j],y, max_cluster=max_cluster,numlocals=numlocals,maxneighbors=max_neighbors)
                #silhouette_score, params = clarans(x[i][j],y, 3)
                #if silhouette_score > CLARANS_best[0]:
                    #CLARANS_best[0] = silhouette_score
                    #CLARANS_best[1] = scaler
                    #CLARANS_best[2] = encoder
                    #CLARANS_best[3] = params
                print("--------DBSCAN--------")
                """
                Test DBSCAN
                """
                #silhouette_score, params = DBSCANs(x[i][j],y,epsS=epsS,min_samples=min_samples,metrics=metrics,algorithms=algorithms,leaf_sizes=leaf_sizes)
                #silhouette_score, params, DBSCAN_centerDF_ex = DBSCANs(x[i][j],y)
                if silhouette_score > DBSCAN_best[0]:
                    DBSCAN_best[0] = silhouette_score
                    DBSCAN_best[1] = scaler
                    DBSCAN_best[2] = encoder
                    DBSCAN_best[3] = params
                    #DBSCAN_centerDF = DBSCAN_centerDF_ex.copy()
                print("--------MeanShift--------")
                """
                Test MeanShift Parameter
                """
                #silhouette_score, params =MeanShifts(x[i][j],y,bandwidths=bandwidths,max_iters=max_iters,n_job=n_job)
                #silhouette_score, params,MeanShift_centerDF_ex =MeanShifts(x[i][j],y)
                if silhouette_score > MeanShift_best[0]:
                    MeanShift_best[0] = silhouette_score
                    MeanShift_best[1] = scaler
                    MeanShift_best[2] = encoder
                    MeanShift_best[3] = params
                    #MeanShift_centerDF = MeanShift_centerDF_ex.copy()
                j = j + 1
            i = i + 1

        kmeans_centerDF.columns = total_columns
        #GMM_centerDF.columns = total_columns
        #DBSCAN_centerDF.columns = total_columns
        #MeanShift_centerDF.columns = total_columns



    return kmeans_best, GMM_best, CLARANS_best, DBSCAN_best, MeanShift_best, kmeans_centerDF,  kmeans_centerDF, kmeans_centerDF, kmeans_centerDF
    #return kmeans_best, GMM_best, CLARANS_best, DBSCAN_best, MeanShift_best, kmeans_centerDF, GMM_centerDF, DBSCAN_centerDF, MeanShift_centerDF
