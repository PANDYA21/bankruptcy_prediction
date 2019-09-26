import numpy as np
import pandas as pd
from scipy.io.arff import loadarff 
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


# evaluation
def evaluation(y, y_pred):
	cf = confusion_matrix(y, y_pred)
	prec = precision_score(y, y_pred)
	rec = recall_score(y, y_pred)
	f1 = f1_score(y, y_pred)
	auc = roc_auc_score(y, y_pred)
	acc = accuracy_score(y, y_pred)
	return cf, prec, rec, f1, auc, acc



def main(file, seed=123):
	raw_data = loadarff(file)
	dat = pd.DataFrame(raw_data[0])
	# imputation
	dat = dat.interpolate()
	X = dat.iloc[:, 0:63]
	y = dat.iloc[:, 64]
	y = y.astype('int')
	# random split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=seed)
	# randomForest
	model_rf = RandomForestClassifier(
		n_estimators=1500, 
		max_depth=5,
		random_state=seed)
	model_rf.fit(X_train, y_train)
	# LR
	model_lr = LogisticRegression(
		# solver='lbfgs',
		solver='saga',
		multi_class='auto', 
		max_iter=1e5, 
		tol=1e-4,
		random_state=seed)
	model_lr.fit(X_train, y_train)
	# MLP
	model_mlp = MLPClassifier(
		hidden_layer_sizes=(700, 60), 
		activation='relu', 
		solver='lbfgs', 
		batch_size='auto', 
		max_iter=1e7, 
		shuffle=True, 
		random_state=seed, 
		tol=1e-5)
	model_mlp.fit(X_train, y_train)
	# Xtreme Gradient Boosted
	model_xgb = XGBClassifier(
		max_depth=4, 
		n_estimators=1500, 
		objective='binary:logistic', 
		booster='gbtree', 
		n_jobs=-1, 
		random_state=seed)
	model_xgb.fit(X_train, y_train)
	# return trained models
	return model_rf,model_lr,model_mlp,model_xgb,X_train,y_train,X_test,y_test


accuracy_train = []
precision_train = []
recall_train = []
f1_train = []
auc_train = []
accuracy_test = []
precision_test = []
recall_test = []
f1_test = []
auc_test = []
year = []
model = []

files = ('data/3year.arff', 'data/4year.arff', 'data/5year.arff')
for file in files:
	print('\n\n############################')
	print(file)
	print('############################\n\n')
	model_rf,model_lr,model_mlp,model_xgb,X_train,y_train,X_test,y_test = main(file)
	# RF
	y_pred = model_rf.predict(X_train)
	cf_rf_train, prec_rf_train, rec_rf_train, f1_rf_train, auc_rf_train, acc_rf_train = evaluation(y_train, y_pred)
	y_pred = model_rf.predict(X_test)
	cf_rf_test, prec_rf_test, rec_rf_test, f1_rf_test, auc_rf_test, acc_rf_test = evaluation(y_test, y_pred)
	accuracy_train.append(acc_rf_train)
	precision_train.append(prec_rf_train)
	recall_train.append(rec_rf_train)
	f1_train.append(f1_rf_train)
	auc_train.append(auc_rf_train)
	accuracy_test.append(acc_rf_test)
	precision_test.append(prec_rf_test)
	recall_test.append(rec_rf_test)
	f1_test.append(f1_rf_test)
	auc_test.append(auc_rf_test)
	year.append(file)
	model.append('RF')
	print('############################ Random Forest\n')
	print('#### Train\n')
	print('\nconfusion matrix: \n', cf_rf_train, '\naccuracy: ', acc_rf_train, '\nprecision: ', prec_rf_train,'\nrecall: ', rec_rf_train, '\nF1-score: ', f1_rf_train, '\nAUC: ', auc_rf_train)
	print('#### Test\n')
	print('\nconfusion matrix: \n', cf_rf_test, '\naccuracy: ', acc_rf_test, '\nprecision: ', prec_rf_test,'\nrecall: ', rec_rf_test, '\nF1-score: ', f1_rf_test, '\nAUC: ', auc_rf_test)
	# LR
	y_pred = model_lr.predict(X_train)
	cf_lr_train, prec_lr_train, rec_lr_train, f1_lr_train, auc_lr_train, acc_lr_train = evaluation(y_train, y_pred)
	y_pred = model_lr.predict(X_test)
	cf_lr_test, prec_lr_test, rec_lr_test, f1_lr_test, auc_lr_test, acc_lr_test = evaluation(y_test, y_pred)
	accuracy_train.append(acc_lr_train)
	precision_train.append(prec_lr_train)
	recall_train.append(rec_lr_train)
	f1_train.append(f1_lr_train)
	auc_train.append(auc_lr_train)
	accuracy_test.append(acc_lr_test)
	precision_test.append(prec_lr_test)
	recall_test.append(rec_lr_test)
	f1_test.append(f1_lr_test)
	auc_test.append(auc_lr_test)
	year.append(file)
	model.append('LR')
	print('############################ Logisitic Regression\n')
	print('#### Train\n')
	print('\nconfusion matrix: \n', cf_lr_train, '\naccuracy: ', acc_lr_train, '\nprecision: ', prec_lr_train,'\nrecall: ', rec_lr_train, '\nF1-score: ', f1_lr_train, '\nAUC: ', auc_lr_train)
	print('#### Test\n')
	print('\nconfusion matrix: \n', cf_lr_test, '\naccuracy: ', acc_lr_test, '\nprecision: ', prec_lr_test,'\nrecall: ', rec_lr_test, '\nF1-score: ', f1_lr_test, '\nAUC: ', auc_lr_test)
	# MLP
	y_pred = model_mlp.predict(X_train)
	cf_mlp_train, prec_mlp_train, rec_mlp_train, f1_mlp_train, auc_mlp_train, acc_mlp_train = evaluation(y_train, y_pred)
	y_pred = model_mlp.predict(X_test)
	cf_mlp_test, prec_mlp_test, rec_mlp_test, f1_mlp_test, auc_mlp_test, acc_mlp_test = evaluation(y_test, y_pred)
	accuracy_train.append(acc_mlp_train)
	precision_train.append(prec_mlp_train)
	recall_train.append(rec_mlp_train)
	f1_train.append(f1_mlp_train)
	auc_train.append(auc_mlp_train)
	accuracy_test.append(acc_mlp_test)
	precision_test.append(prec_mlp_test)
	recall_test.append(rec_mlp_test)
	f1_test.append(f1_mlp_test)
	auc_test.append(auc_mlp_test)
	year.append(file)
	model.append('MLP')
	print('############################ Multi Layer Perceptron\n')
	print('#### Train\n')
	print('\nconfusion matrix: \n', cf_mlp_train, '\naccuracy: ', acc_mlp_train, '\nprecision: ', prec_mlp_train,'\nrecall: ', rec_mlp_train, '\nF1-score: ', f1_mlp_train, '\nAUC: ', auc_mlp_train)
	print('#### Test\n')
	print('\nconfusion matrix: \n', cf_mlp_test, '\naccuracy: ', acc_mlp_test, '\nprecision: ', prec_mlp_test,'\nrecall: ', rec_mlp_test, '\nF1-score: ', f1_mlp_test, '\nAUC: ', auc_mlp_test)
	# XGB
	y_pred = model_xgb.predict(X_train)
	cf_xgb_train, prec_xgb_train, rec_xgb_train, f1_xgb_train, auc_xgb_train, acc_xgb_train = evaluation(y_train, y_pred)
	y_pred = model_xgb.predict(X_test)
	cf_xgb_test, prec_xgb_test, rec_xgb_test, f1_xgb_test, auc_xgb_test, acc_xgb_test = evaluation(y_test, y_pred)
	accuracy_train.append(acc_xgb_train)
	precision_train.append(prec_xgb_train)
	recall_train.append(rec_xgb_train)
	f1_train.append(f1_xgb_train)
	auc_train.append(auc_xgb_train)
	accuracy_test.append(acc_xgb_test)
	precision_test.append(prec_xgb_test)
	recall_test.append(rec_xgb_test)
	f1_test.append(f1_xgb_test)
	auc_test.append(auc_xgb_test)
	year.append(file)
	model.append('XGB')
	print('############################ Extreme Gradient Boosting\n')
	print('#### Train\n')
	print('\nconfusion matrix: \n', cf_xgb_train, '\naccuracy: ', acc_xgb_train, '\nprecision: ', prec_xgb_train,'\nrecall: ', rec_xgb_train, '\nF1-score: ', f1_xgb_train, '\nAUC: ', auc_xgb_train)
	print('#### Test\n')
	print('\nconfusion matrix: \n', cf_xgb_test, '\naccuracy: ', acc_xgb_test, '\nprecision: ', prec_xgb_test,'\nrecall: ', rec_xgb_test, '\nF1-score: ', f1_xgb_test, '\nAUC: ', auc_xgb_test)
	



evaluations = pd.DataFrame(
	list(zip(
		accuracy_train,
		precision_train,
		recall_train,
		f1_train,
		auc_train,
		accuracy_test,
		precision_test,
		recall_test,
		f1_test,
		auc_test,
		year,
		model)), 
	columns=[
		'accuracy_train',
		'precision_train',
		'recall_train',
		'f1_train',
		'auc_train',
		'accuracy_test',
		'precision_test',
		'recall_test',
		'f1_test',
		'auc_test',
		'year',
		'model'])

evaluations.to_csv('evaluations_binary.csv')
