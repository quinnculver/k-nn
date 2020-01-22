import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats, spatial 



# The following implements k-nearest neighbors classification and some
# related funtions that are useful in their own right.

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
# load dataset into Pandas DataFrame
df = pd.read_csv(url, names=['sepal_length','sepal_width','petal_length','petal_width','target'])




def distance_to_each_training_point(x_test, df_training, metric = 'euclidean'):
    """
    returns a len(X_TEST) by len(DF_TRAINING) ndarray whose j-th row
    consists of all (euclidean) distances between the j-th row of the
    ndarray X_TEST and (the first len(X_TEST[0]) columns of) each row
    of DF_TRAINING. (Thus there must be enough columns in DF_TRAINING)
    
    The default metric, 'euclidean', can be any of the following:
    ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’,
    ‘cosine’, ‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’,
    ‘jensenshannon’, ‘kulsinski’, ‘mahalanobis’, ‘matching’,
    ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’,
    ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘wminkowski’,
    ‘yule’. See the scipy.spatial.distance.cdist documentation for
    details:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html#scipy.spatial.distance.cdist
    """
    
    n=len(x_test[0])
    return spatial.distance.cdist(x_test, df_training.iloc[:,0:n].values, metric)

def k_nearest_neighbors(x_test, df_training, k):
    """
    returns an len(X_TEST) by k ndarray whose j-th row consists of the
    the indices of the k smallest elements from the j-th row of
    distance_to_each_training_point(X_TEST, DF_TRAINING)
    """

    return np.argpartition(distance_to_each_training_point(x_test,
                                                           df_training), k-1)[:,0:k]


def majority_vote(indices, df_training, attrib_column):
    """
    returns a length len(INDICES) list whose j-th element is the (first, e.g.\
    lexicographically or smallest) most-common value from
    ATTRIB_COLUMN among unique (because no one gets more than one
    vote) rows determined by the j-th row in INDICES.
    """
    
    return [stats.mode(df_training.iloc[np.unique(indices_row),
                                        attrib_column])[0][0] for indices_row in indices]

def knn_classification(x_test, df_training, attrib_column, k):
    """
    returns a length len(X_TEST) list whose j-th element is label is
    the most common label among the k nearest (according to the
    euclidean metric; see above) points of DF_TRAINING to the j-th row
    of X_TEST
    """
    return majority_vote(k_nearest_neighbors(x_test, df_training,k),df,attrib_column)


# Some randomly generated test data
x = np.transpose([np.random.normal(df.mean()[i], df.std()[i],10) for i in range (0,4)])

# x= np.array([5.8, 3, 4.2, 1.3])
#twenty_five_nearest_neighbors = k_nearest_neighbors(x,df,25)
# votes=target_col[twenty_five_nearest_neighbors]

