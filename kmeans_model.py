""" 
Unsupervised clustering = computer learns patterns on its own (doesn't recieve the actual output value, only the features that make it up)

k = # of clusters
    Divides the data into different sections, and then assigns a data point to the section it is most similar to
    
    Starts by creating 2 centroids in random position (k=2)
        # Can set k higher, giving a higher starting number of random centroids

    Draw straight line between the two centroids, then divide all points by a perpendicular line going through the midpoint to assign to a centroid

    Now find the center point of all the points in a given group, where n = number of points in the group

    Repeat until there is no change in classification of points and thus the centroid doesn't move (successful clustering of points into clusters)

    High computational load if you have many features and data points (p x C x iter x feat) where p = amount of data points and C = number of centroids

"""

# Packages
import numpy as np
import sklearn
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics

# Get data
digits = load_digits()

data = scale(digits.data) # .data includes all the features; scaling features down to put them between -1 and 1
    # Smaller values are better since we're using euclidean distance

# Extract data targets
y = digits.target

# Set k (no. of centroids)
k = 10 # Hard setting
# k = len(np.unique(y)) # Dynamic setting

# Extract features
samples,features = data.shape

# Set function for calculation benchmarking metrics
def bench_k_means(estimator, name, data): # estimator = classifier
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))
    

# Create classifier
clf = KMeans(n_clusters = k, # number of clusters / centroids
             init = "random", # initialized classifier
             n_init = 10)

bench_k_means(clf, "1", data) # Output = 69417 0.589   0.663   0.623   0.469   0.619   0.139














