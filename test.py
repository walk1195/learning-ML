# Libs
import tensorflow
import keras
import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

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
y = data_trimmed[predict].to_numpy()

# Split up data into 4 variables
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

# Call linear model
linear = linear_model.LinearRegression()

# Fit the training data to determine the best fit line
linear.fit(x_train, y_train)

# Score to determine model accuracy
acc = linear.score(x_test, y_test) 
print(acc) # 0.827

# Now we want to actually use the model to do some predictions
## First print coefs and intercept
print("Coefficient: \n", linear.coef_) # The larger the coefficient, the more weight that variable has
print("Intercept: \n", linear.intercept_)

# Run predictions
predictions = linear.predict(x_test)

# Print out grade predictions (e.g. 10.777692733340155 [13 11  2  1  3] 11)
    ## Where 10.78 is the predicted grade, [] are the input features, and 11 is their actual grade
for i in range(len(predictions)):
    print(predictions[i], x_test[i], y_test[i])


# Save model -- therefore you don't have to retrain every time you use it
outDir = 'Models/'
with open(f"{outDir}student_model_LR.pickle", "wb") as f:
    pickle.dump(linear, f)

# Read back in
pickle_in = open(f"{outDir}student_model_LR.pickle", "rb")
linear = pickle.load(pickle_in)

### Now we can try training n models to optimize our accuracy
best = 0
for i in range(60):
    # Split up data into 4 variables
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

    # Call linear model
    linear = linear_model.LinearRegression()

    # Fit the training data to determine the best fit line
    linear.fit(x_train, y_train)
    # Score to determine model accuracy
    acc = linear.score(x_test, y_test)
    print(acc)

    if acc > best:
        best = acc
        outDir = 'Models/'
        with open(f"{outDir}student_model_LR.pickle", "wb") as f:
            pickle.dump(linear, f)


# Plot results of the best model
style.use("ggplot")

p = 'studytime'
pyplot.scatter(data[p], data['G3'])
pyplot.xlabel(p)
pyplot.ylabel('Final Grade')
pyplot.show()


###### Notes #####
# data.columns = see col names
# data.head = print first five rows
# data.shape = prints the dimensions of the df
