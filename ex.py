import numpy as np
from sklearn import svm, metrics, model_selection

raw_data = np.genfromtxt('plrx.txt')

X = raw_data[:, :12]
y = raw_data[:, -1]

clf = svm.SVC(gamma=5)

def KFOLD(X,y):
	kf = model_selection.KFold(n_splits=2)
	for train, test in kf.split(X):
		X_train, y_train = X[train], y[train]
		X_test, y_test = X[test], y[test]
	return {'Xtrain':X_train, 'ytrain':y_train, 'Xtest':X_test, 'ytest':y_test}

print KFOLD(X,y)['ytrain']
