import numpy as np
import pandas as pd
from scipy.io.arff import loadarff 
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


### evaluation function
def evaluation(y, y_pred, mAverage='macro'):
	cf = confusion_matrix(y, y_pred)
	prec = precision_score(y, y_pred, average=mAverage)
	rec = recall_score(y, y_pred, average=mAverage)
	f1 = f1_score(y, y_pred, average=mAverage)
	acc = accuracy_score(y, y_pred)
	return cf, prec, rec, f1, acc


### read arff files and create a custom multi-class column
raw_data = loadarff('data/3year.arff')
dat1 = pd.DataFrame(raw_data[0])
dat1['class'] = dat1['class'].astype('int')
dat1['mclass'] = np.where(dat1['class'] == 1, '3years', 0)

raw_data = loadarff('data/4year.arff')
dat2 = pd.DataFrame(raw_data[0])
dat2['class'] = dat2['class'].astype('int')
dat2['mclass'] = np.where(dat2['class'] == 1, '4years', 0)

raw_data = loadarff('data/5year.arff')
dat3 = pd.DataFrame(raw_data[0])
dat3['class'] = dat3['class'].astype('int')
dat3['mclass'] = np.where(dat3['class'] == 1, '5years', 0)

### combine all three files into one dataframe
dat = dat1.append(pd.DataFrame(data = dat2), ignore_index=True)
dat = dat.append(pd.DataFrame(data = dat3), ignore_index=True)

### interpolate missing values
dat = dat.interpolate()

X = dat.iloc[:, 0:63]
y = dat.iloc[:, 65]

### random split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=123)

### randomForest
model_rf = RandomForestClassifier(
	n_estimators=3000, 
	max_depth=None,
	n_jobs=-1,
	random_state=123)
model_rf.fit(X_train, y_train)

### LR
model_lr = LogisticRegression(
	solver='saga',
	multi_class='auto', 
	max_iter=1e7, 
	tol=1e-5,
	random_state=123)
model_lr.fit(X_train, y_train)

### mlp
model_mlp = MLPClassifier(
	hidden_layer_sizes=(100, 30), 
	activation='tanh', 
	solver='lbfgs', 
	batch_size='auto', 
	max_iter=1e7, 
	shuffle=True, 
	random_state=123, 
	tol=1e-5)
model_mlp.fit(X_train, y_train)

### xgboost
model_xgb = XGBClassifier(
	max_depth=4, 
	n_estimators=5000, 
	objective='binary:logistic', 
	booster='gbtree', 
	n_jobs=1, 
	nthread=None,
	random_state=123)
model_xgb.fit(X_train, y_train)


### evaluation
print('\n\n############################')
print('\n\n')

y_pred = model_rf.predict(X_test)
cf_rf, prec_rf, rec_rf, f1_rf, acc_rf = evaluation(y_test, y_pred)
print('############################ Random Forest\n')
print('\nconfusion matrix: \n', cf_rf, '\nprecision: ', prec_rf,'\nrecall: ', rec_rf, '\nF1-score: ', f1_rf, '\nACC: ', acc_rf)

y_pred = model_lr.predict(X_test)
cf_lr, prec_lr, rec_lr, f1_lr, acc_lr = evaluation(y_test, y_pred)
print('############################ Logistic Regression\n')
print('\nconfusion matrix: \n', cf_lr, '\nprecision: ', prec_lr,'\nrecall: ', rec_lr, '\nF1-score: ', f1_lr, '\nACC: ', acc_lr)

y_pred = model_mlp.predict(X_test)
cf_mlp, prec_mlp, rec_mlp, f1_mlp, acc_mlp = evaluation(y_test, y_pred)
print('############################ Multi Layer Perceptron\n')
print('\nconfusion matrix: \n', cf_mlp, '\nprecision: ', prec_mlp,'\nrecall: ', rec_mlp, '\nF1-score: ', f1_mlp, '\nACC: ', acc_mlp)

y_pred = model_xgb.predict(X_test)
cf_xgb, prec_xgb, rec_xgb, f1_xgb, acc_xgb = evaluation(y_test, y_pred)
print('############################ Xtreme Gradient Boosted Trees\n')
print('\nconfusion matrix: \n', cf_xgb, '\nprecision: ', prec_xgb,'\nrecall: ', rec_xgb, '\nF1-score: ', f1_xgb, '\nACC: ', acc_xgb)





