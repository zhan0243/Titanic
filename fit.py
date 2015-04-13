import numpy as np
from sklearn import tree
from sklearn import ensemble
from sklearn import cross_validation
import matplotlib.pylab as plt
import cleandata

reload(cleandata)
train_df, test_df = cleandata.cleaneddf()

train_data = train_df.values
test_data = test_df.values

X = train_data[::, 1::]
y= train_data[::, 0]


# X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3, random_state=0)

# Gradient boosting
clf = ensemble.GradientBoostingClassifier()
scores = cross_validation.cross_val_score(clf, X, y, cv=5)
print('Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std()))

clf.fit(X, y)
results = clf.predict(test_data[::, 1::])

cleandata.write_to_csv(test_data[::, 0], results)

# Decision tree
clf = tree.DecisionTreeClassifier(max_depth=5)
scores = cross_validation.cross_val_score(clf, X, y, cv=5)
print('Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std()))


# random forest tree
clf = ensemble.RandomForestClassifier(n_estimators=100)
scores = cross_validation.cross_val_score(clf, X, y, cv=5)
print('Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std()))



