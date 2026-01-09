# Libs
import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

# Read in data
data = pd.read_csv("Data/car_evaluation/car.data")
print(data.head())

# Convert non-numerical data
### sklearn preprocessing helps with this -- converts qualitative data into integer values
 
le = preprocessing.LabelEncoder() # encodes labels into integer values

# Create lists for each column
# d = {}
# for column in list(data.columns):
#     numeric_list = le.fit_transform(list(data[column]))
#     d[f"{column}"] = numeric_list

# Fix class for labelling
# d['cls'] = d['class']
# d.pop("class", None)


# Set features and labels (X and y)
# x = list(zip(d.values()))

# y = list(zip(d['cls']))


buying = le.fit_transform(list(data['buying'])).tolist()
maint = le.fit_transform(list(data['maint'])).tolist()
door = le.fit_transform(list(data['door'])).tolist()
persons = le.fit_transform(list(data['persons'])).tolist()
lug_boot = le.fit_transform(list(data['lug_boot'])).tolist()
safety = le.fit_transform(list(data['safety'])).tolist()
cls = le.fit_transform(list(data['class'])).tolist()

# Create predict var
predict = "class"

# Sorting data into a list containing a tuple per row of data
x = list(zip(buying,maint,door,persons,lug_boot,safety))
y = list(cls)

# Set up data split
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

# Check
print(x_train, y_test)

### Run kNN algorithm
model = KNeighborsClassifier(n_neighbors = 9)

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)


# Predict
predicted = model.predict(x_test)

### Compare predictions to actual value
names = ["unacc", "acc", "good", "vgood"]

# Print predictions vs actual value
for i in range(len(predicted)):
    if predicted[i] != y_test[i]:
        print("Predicted: ", names[predicted[i]], "Data: ", x_test[i], "Actual: ", names[y_test[i]])

        # Can print the distance to neighbors 
        n = model.kneighbors([x_test[i]], 9, True) # sends data in to be 2d
        print("N: ", n)
        
    else:
        print("Predicted the correct class value.")



