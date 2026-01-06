# Libs
import tensorflow
import keras
import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

#########################################################################
# Linear Regression 
#########################################################################

# Read in data
dataDir = 'Data/student/student-mat.csv'
data = pd.read_csv(dataDir, sep=";")

# Trim down attributes
data_trimmed = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

# Define prediction value
### based on n attributes, what "label" do you want to determine?
### attributes are what goes in, label is what comes out
### can have any number of labels
predict = "G3"

# Set up label and attribute arrays
x = data_trimmed.drop(columns=[predict]).to_numpy() # attribute array
y = np.array(data[predict])

# Split up data into 4 variables
x_train, y_train, x_test, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)











###### Notes #####
# data.columns = see col names
# data.head = print first five rows

