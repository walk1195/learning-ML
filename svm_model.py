# Libs
import sklearn
from sklearn import datasets
from sklearn import svm # import classifier
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

# Load in breast cancer data
cancer = datasets.load_breast_cancer()

print(cancer.feature_names) # features
print(cancer.target_names) # target prediction category

x = cancer.data
y = cancer.target

# Split into train and test datasets
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.2) # increased test size to 20%

# Checking train data
print(x_train, y_train)

classes = ['malignant', 'benign'] # Grabbing target classes for indexing the results later

# First testing without setting any parameters
### 0.89 without tweaking any parameters
### 0.96 with linear kernel
### 
clf = svm.SVC(kernel = "linear", C=2) # C=0 is hard margin, C=2 is softer margin 
clf.fit(x_train, y_train)

# Predict y based on x features
y_pred = clf.predict(x_test)

# Compare predicts and actual to get a measure of accuracy
acc = metrics.accuracy_score(y_test, y_pred)
print(acc)

