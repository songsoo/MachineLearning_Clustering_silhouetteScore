import pandas as pd
import ML_lab2_auto

pd.set_option('display.max_columns',None)
#pd.set_option('display.max_rows',None)

a = pd.read_csv("D:/housing.csv")
a.dropna(axis=0,inplace=True)
a.reset_index(inplace=True)
y = a.loc[:,'median_house_value']
x = a.loc[:,['longitude','latitude','housing_median_age','total_rooms','total_bedrooms','population','households','median_income','ocean_proximity']]

#select columns
numerical_columns = ['longitude','latitude','housing_median_age','total_rooms','total_bedrooms','population','households','median_income']
numerical_columns = [['longitude','latitude','housing_median_age','total_rooms','total_bedrooms','population','households','median_income'],['housing_median_age','total_rooms','total_bedrooms','population','households','median_income'],['housing_median_age','total_rooms','total_bedrooms']]
#numerical_columns = []
categorical_columns = [['ocean_proximity'],['ocean_proximity'],[]]
#categorical_columns = []



#select parameters
max_cluster = 10
n_inits = [1,5,50]
max_iters = [100,200,300]
tols = [1e-4,1e-3,1e-2]
verboses = [0]
covariance_types = ['full','tied']
numlocals = [2,3]
max_neighbors = [2,3]
epsS = [0.3,0.5,1.2]
min_samples = [3,4,5]
metrics=['euclidean','manhattan','l1']
algorithms = ['auto']
leaf_sizes = [30,50,100]
n_jobs = -1
# if bandwidth array is empty, estimate_bandwidth function estimate the bandwidth to use with the mean-shift algorithm.
bandwidths = [None,0.5,1,3]

kmeans_best = [-1,'scale','encode','params']
GMM_best = [-1,'scale','encode','params']
CLARANS_best = [-1,'scale','encode','params']
DBSCAN_best = [-1,'scale','encode','params']
MeanShift_best = [-1,'scale','encode','params']
kmeans_centerDF = pd.DataFrame()
GMM_centerDF = pd.DataFrame()
DBSCAN_centerDF = pd.DataFrame()
MeanShift_centerDF = pd.DataFrame()


kmeans_best,GMM_best, CLARANS_best, DBSCAN_best, MeanShift_best, kmeans_centerDF,GMM_centerDF,DBSCAN_centerDF,MeanShift_centerDF = ML_lab2_auto.findBest(x,y,numerical_columns,categorical_columns,max_cluster,n_inits,max_iters,tols,verboses,covariance_types,
                      numlocals,max_neighbors,epsS,min_samples,metrics,algorithms,leaf_sizes,bandwidths,n_jobs)

print("===================Result===================\n")
print("----------------KMeans----------------")
print("Scaler: ", kmeans_best[1], "Encoder: ", kmeans_best[2])
print("Silhouette Score: ", kmeans_best[0])
print(kmeans_best[3])
print("Best Center\n")
print(kmeans_centerDF)

print("\n----------------GMM----------------")
print("Scaler: ", GMM_best[1], "Encoder: ", GMM_best[2])
print("Silhouette Score: ", GMM_best[0])
print(GMM_best[3])
print("Best Center\n")
print(GMM_centerDF)

print("\n----------------CLARANS----------------")
print("Scaler: ", CLARANS_best[1], "Encoder: ", CLARANS_best[2])
print("Silhouette Score: ", CLARANS_best[0])
print(CLARANS_best[3])

print("----------------DBSCAN----------------")
print("Scaler: ", DBSCAN_best[1], "Encoder: ", DBSCAN_best[2])
print("Silhouette Score: ", DBSCAN_best[0])
print(DBSCAN_best[3])
print("Best Center\n")
print(DBSCAN_centerDF)

print("\n----------------MeanShift----------------")
print("Scaler: ", MeanShift_best[1], "Encoder: ", MeanShift_best[2])
print("Silhouette Score: ", MeanShift_best[0])
print(MeanShift_best[3])
print("Best Center\n")
print(MeanShift_centerDF)