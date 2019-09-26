import numpy as np
import pandas as pd
from scipy.io.arff import loadarff 
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score,roc_auc_score
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


# evaluation
def evaluation(y, y_pred):
	cf = confusion_matrix(y, y_pred)
	prec = precision_score(y, y_pred)
	rec = recall_score(y, y_pred)
	f1 = f1_score(y, y_pred)
	auc = roc_auc_score(y, y_pred)
	return cf, prec, rec, f1, auc



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
		random_state=123)
	model_rf.fit(X_train, y_train)
	# LR
	model_lr = LogisticRegression(
		# solver='lbfgs',
		solver='saga',
		multi_class='auto', 
		max_iter=1e5, 
		tol=1e-4,
		random_state=123)
	model_lr.fit(X_train, y_train)
	# MLP
	model_mlp = MLPClassifier(
		hidden_layer_sizes=(700, 60), 
		activation='relu', 
		solver='lbfgs', 
		batch_size='auto', 
		max_iter=1e7, 
		shuffle=True, 
		random_state=123, 
		tol=1e-5)
	model_mlp.fit(X_train, y_train)
	# Xtreme Gradient Boosted
	model_xgb = XGBClassifier(
		max_depth=4, 
		n_estimators=1500, 
		objective='binary:logistic', 
		booster='gbtree', 
		n_jobs=-1, 
		random_state=123)
	model_xgb.fit(X_train, y_train)
	# return trained models
	return model_rf,model_lr,model_mlp,model_xgb,X_test,y_test


files = ('data/3year.arff', 'data/4year.arff', 'data/5year.arff')
for file in files:
	print('\n\n############################')
	print(file)
	print('############################\n\n')
	model_rf,model_lr,model_mlp,model_xgb,X_test,y_test = main(file)
	y_pred = model_rf.predict(X_test)
	cf_rf, prec_rf, rec_rf, f1_rf, auc_rf = evaluation(y_test, y_pred)
	print('############################ Random Forest\n')
	print('\nconfusion matrix: \n', cf_rf, '\nprecision: ', prec_rf,'\nrecall: ', rec_rf, '\nF1-score: ', f1_rf, '\nAUC: ', auc_rf)
	y_pred = model_lr.predict(X_test)
	cf_lr, prec_lr, rec_lr, f1_lr, auc_lr = evaluation(y_test, y_pred)
	print('############################ Logistic Regression\n')
	print('\nconfusion matrix: \n', cf_lr, '\nprecision: ', prec_lr,'\nrecall: ', rec_lr, '\nF1-score: ', f1_lr, '\nAUC: ', auc_lr)
	y_pred = model_mlp.predict(X_test)
	cf_mlp, prec_mlp, rec_mlp, f1_mlp, auc_mlp = evaluation(y_test, y_pred)
	print('############################ Multi Layer Perceptron\n')
	print('\nconfusion matrix: \n', cf_mlp, '\nprecision: ', prec_mlp,'\nrecall: ', rec_mlp, '\nF1-score: ', f1_mlp, '\nAUC: ', auc_mlp)
	y_pred = model_xgb.predict(X_test)
	cf_xgb, prec_xgb, rec_xgb, f1_xgb, auc_xgb = evaluation(y_test, y_pred)
	print('############################ Xtreme Gradient Boosted Trees\n')
	print('\nconfusion matrix: \n', cf_xgb, '\nprecision: ', prec_xgb,'\nrecall: ', rec_xgb, '\nF1-score: ', f1_xgb, '\nAUC: ', auc_xgb)



