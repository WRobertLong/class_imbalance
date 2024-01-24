import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.datasets import make_classification

# Simulating a dataset with class imbalance
X, y = make_classification(n_samples=1000, n_features=20, n_clusters_per_class=1,
                           n_informative=2, n_redundant=10, weights=[0.5, 0.5],
                           flip_y=0, random_state=15)

# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)

# Training a RandomForestClassifier
clf = RandomForestClassifier(random_state=15)
clf.fit(X_train, y_train)

# Predicting on the test set
y_pred = clf.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(accuracy, report)