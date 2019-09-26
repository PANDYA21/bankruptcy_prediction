from scipy.io.arff import loadarff 
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score,roc_auc_score
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
                         multi_class='ovr', max_iter=1e7, tol=1e-5)
clf4.fit(X_train, y_train)
y_pred = clf4.predict(X_test)
confusion_matrix(y_test, y_pred)



# def main(file, seed=123):
# 	raw_data = loadarff(file)
# 	dat = pd.DataFrame(raw_data[0])
# 	# imputation
# 	dat = dat.interpolate()
# 	X = dat.iloc[:, 0:63]
# 	y = dat.iloc[:, 64]
# 	y = y.astype('int')
# 	# random split
# 	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=seed)
# 	# randomForest
# 	model_rf = RandomForestClassifier(
# 		n_estimators=3000, 
# 		max_depth=5,
# 		random_state=123)
# 	model_rf.fit(X_train, y_train)
# 	# LR
# 	model_lr = LogisticRegression(
# 		solver='lbfgs',
# 		multi_class='auto', 
# 		max_iter=1e7, 
# 		tol=1e-5,
# 		random_state=123)
# 	model_lr.fit(X_train, y_train)
# 	# Xtreme Gradient Boosted
# 	model_xgb = XGBClassifier(
# 		max_depth=4, 
# 		n_estimators=5000, 
# 		objective='binary:logistic', 
# 		booster='gbtree', 
# 		n_jobs=1, 
# 		nthread=None,
# 		random_state=123)
# 	model_xgb.fit(X_train, y_train)
# 	# return trained models
# 	return model_rf,model_lr,model_xgb


# ### gbt
# model_gbt = GradientBoostingClassifier(
# 	n_estimators=5000, 
# 	max_depth=3,
# 	random_state=123)
# model_gbt.fit(X_train, y_train)
# y_pred = model_gbt.predict(X_test)
# cf_gbt, prec_gbt, rec_gbt, f1_gbt, acc_gbt = evaluation(y_test, y_pred)


