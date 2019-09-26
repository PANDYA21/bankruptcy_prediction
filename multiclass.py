import numpy as np
import pandas as pd
from scipy.io.arff import loadarff 
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score
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
	max_depth=3,
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
	hidden_layer_sizes=(60), 
	activation='relu', 
	solver='lbfgs', 
	batch_size='auto', 
	max_iter=1e7, 
	shuffle=True, 
	random_state=123, 
	tol=1e-5)
model_mlp.fit(X_train, y_train)

### xgboost
model_xgb = XGBClassifier(
	max_depth=3, 
	n_estimators=3000, 
	objective='binary:logistic', 
	booster='gbtree', 
	n_jobs=-1, 
	nthread=None,
	random_state=123)
model_xgb.fit(X_train, y_train)





### evaluation
accuracy_train = []
precision_train = []
recall_train = []
f1_train = []
accuracy_test = []
precision_test = []
recall_test = []
f1_test = []
model = []



print('\n\n############################')
print('\n\n')

# RF
y_pred = model_rf.predict(X_train)
cf_rf_train, prec_rf_train, rec_rf_train, f1_rf_train, acc_rf_train = evaluation(y_train, y_pred)
y_pred = model_rf.predict(X_test)
cf_rf_test, prec_rf_test, rec_rf_test, f1_rf_test, acc_rf_test = evaluation(y_test, y_pred)
accuracy_train.append(acc_rf_train)
precision_train.append(prec_rf_train)
recall_train.append(rec_rf_train)
f1_train.append(f1_rf_train)
accuracy_test.append(acc_rf_test)
precision_test.append(prec_rf_test)
recall_test.append(rec_rf_test)
f1_test.append(f1_rf_test)
model.append('RF')
print('############################ Random Forest\n')
print('#### Train\n')
print('\nconfusion matrix: \n', cf_rf_train, '\naccuracy: ', acc_rf_train, '\nprecision: ', prec_rf_train,'\nrecall: ', rec_rf_train, '\nF1-score: ', f1_rf_train)
print('#### Test\n')
print('\nconfusion matrix: \n', cf_rf_test, '\naccuracy: ', acc_rf_test, '\nprecision: ', prec_rf_test,'\nrecall: ', rec_rf_test, '\nF1-score: ', f1_rf_test)
# LR
y_pred = model_lr.predict(X_train)
cf_lr_train, prec_lr_train, rec_lr_train, f1_lr_train, acc_lr_train = evaluation(y_train, y_pred)
y_pred = model_lr.predict(X_test)
cf_lr_test, prec_lr_test, rec_lr_test, f1_lr_test, acc_lr_test = evaluation(y_test, y_pred)
accuracy_train.append(acc_lr_train)
precision_train.append(prec_lr_train)
recall_train.append(rec_lr_train)
f1_train.append(f1_lr_train)
accuracy_test.append(acc_lr_test)
precision_test.append(prec_lr_test)
recall_test.append(rec_lr_test)
f1_test.append(f1_lr_test)
model.append('LR')
print('############################ Logisitic Regression\n')
print('#### Train\n')
print('\nconfusion matrix: \n', cf_lr_train, '\naccuracy: ', acc_lr_train, '\nprecision: ', prec_lr_train,'\nrecall: ', rec_lr_train, '\nF1-score: ', f1_lr_train)
print('#### Test\n')
print('\nconfusion matrix: \n', cf_lr_test, '\naccuracy: ', acc_lr_test, '\nprecision: ', prec_lr_test,'\nrecall: ', rec_lr_test, '\nF1-score: ', f1_lr_test)
# MLP
y_pred = model_mlp.predict(X_train)
cf_mlp_train, prec_mlp_train, rec_mlp_train, f1_mlp_train, acc_mlp_train = evaluation(y_train, y_pred)
y_pred = model_mlp.predict(X_test)
cf_mlp_test, prec_mlp_test, rec_mlp_test, f1_mlp_test, acc_mlp_test = evaluation(y_test, y_pred)
accuracy_train.append(acc_mlp_train)
precision_train.append(prec_mlp_train)
recall_train.append(rec_mlp_train)
f1_train.append(f1_mlp_train)
accuracy_test.append(acc_mlp_test)
precision_test.append(prec_mlp_test)
recall_test.append(rec_mlp_test)
f1_test.append(f1_mlp_test)
model.append('MLP')
print('############################ Multi Layer Perceptron\n')
print('#### Train\n')
print('\nconfusion matrix: \n', cf_mlp_train, '\naccuracy: ', acc_mlp_train, '\nprecision: ', prec_mlp_train,'\nrecall: ', rec_mlp_train, '\nF1-score: ', f1_mlp_train)
print('#### Test\n')
print('\nconfusion matrix: \n', cf_mlp_test, '\naccuracy: ', acc_mlp_test, '\nprecision: ', prec_mlp_test,'\nrecall: ', rec_mlp_test, '\nF1-score: ', f1_mlp_test)
# XGB
y_pred = model_xgb.predict(X_train)
cf_xgb_train, prec_xgb_train, rec_xgb_train, f1_xgb_train, acc_xgb_train = evaluation(y_train, y_pred)
y_pred = model_xgb.predict(X_test)
cf_xgb_test, prec_xgb_test, rec_xgb_test, f1_xgb_test, acc_xgb_test = evaluation(y_test, y_pred)
accuracy_train.append(acc_xgb_train)
precision_train.append(prec_xgb_train)
recall_train.append(rec_xgb_train)
f1_train.append(f1_xgb_train)
accuracy_test.append(acc_xgb_test)
precision_test.append(prec_xgb_test)
recall_test.append(rec_xgb_test)
f1_test.append(f1_xgb_test)
model.append('XGB')
print('############################ Extreme Gradient Boosting\n')
print('#### Train\n')
print('\nconfusion matrix: \n', cf_xgb_train, '\naccuracy: ', acc_xgb_train, '\nprecision: ', prec_xgb_train,'\nrecall: ', rec_xgb_train, '\nF1-score: ', f1_xgb_train)
print('#### Test\n')
print('\nconfusion matrix: \n', cf_xgb_test, '\naccuracy: ', acc_xgb_test, '\nprecision: ', prec_xgb_test,'\nrecall: ', rec_xgb_test, '\nF1-score: ', f1_xgb_test)



evaluations = pd.DataFrame(
	list(zip(
		accuracy_train,
		precision_train,
		recall_train,
		f1_train,
		accuracy_test,
		precision_test,
		recall_test,
		f1_test,
		model)), 
	columns=[
		'accuracy_train',
		'precision_train',
		'recall_train',
		'f1_train',
		'accuracy_test',
		'precision_test',
		'recall_test',
		'f1_test',
		'model'])

evaluations.to_csv('evaluations_multiclass.csv')
