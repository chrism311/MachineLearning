import numpy as np
from sklearn import svm, ensemble, model_selection
import sys

raw_data = np.genfromtxt('plrx.txt')

X = raw_data[:, :12]
y = raw_data[:, -1]

clf1 = svm.SVC(gamma = 1.2)
clf2 = ensemble.RandomForestClassifier(n_estimators=30, max_depth=3)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y, test_size=float(sys.argv[1]))


#SVM 50/50 split and averaged 10 times
svm_scores=[]
for i in range(10):
	clf1.fit(X_train, y_train)
	scores = clf1.score(X_test, y_test)
	svm_scores.append(scores)

print "SVM average:", 100*np.mean(svm_scores)

#RF split once
rf_FINAL=[]

clf2.fit(X_train, y_train)
score = clf2.score(X_test, y_test)
rf_FINAL.append(score)

#RF under CV 6 times
cv_score=[]
kf = model_selection.KFold(n_splits=2)
for i in range(6):
	for train, test in kf.split(X):
		clf2.fit(X[train], y[train])
		acc_rf= clf2.score(X[test],y[test])
		cv_score.append(acc_rf)
	rf_FINAL.append(np.mean(cv_score))

print "RF final score:", 100*np.mean(rf_FINAL)
