from scipy.io.arff import loadarff 
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd

raw_data = loadarff('data/3year.arff')
dat = pd.DataFrame(raw_data[0])

# dat = dat.fillna(dat.mean())
dat = dat.interpolate()

X = dat.iloc[:, 0:63]
y = dat.iloc[:, 64]
y = y.astype('int')

clf = RandomForestClassifier(n_estimators=200, max_depth=3,
                             random_state=123)
clf.fit(X, y)

y_pred = clf.predict(X)
confusion_matrix(y, y_pred)


clf2 = GradientBoostingClassifier(n_estimators=300, max_depth=3,
                             random_state=123)
clf2.fit(X, y)

y_pred = clf2.predict(X)
confusion_matrix(y, y_pred)


# with split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=123)
clf3 = GradientBoostingClassifier(n_estimators=500, max_depth=5,
                             random_state=123)
clf3.fit(X_train, y_train)
y_pred = clf3.predict(X_test)
confusion_matrix(y_test, y_pred)


# LR
clf4 = LogisticRegression(random_state=123, solver='lbfgs',
                         multi_class='ovr', max_iter=1e5)
clf4.fit(X_train, y_train)
y_pred = clf4.predict(X_test)
confusion_matrix(y_test, y_pred)



