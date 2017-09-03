import numpy as np
from sklearn import ensemble, neighbors, svm, metrics, model_selection,\
	linear_model, tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sys


raw_data = np.genfromtxt('plrx.txt')					#reading data from txt file

X= raw_data[:,:12]							#splicing the raw data
y = raw_data[:,-1]

SVM_parameters = {'kernel':('linear','rbf', 'poly'), 'degree':[1,10], 'C':[1,10], 'gamma':[0,5]}
RF_parameters = {'n_estimators':[1,500], 'max_depth':[1,10], 'min_samples_split':[2,10], 'min_samples_leaf':[1,10]}

clf1 = neighbors.KNeighborsClassifier(n_neighbors= 2)			#k-NN classifier
clf2 = svm.SVC(gamma=5)							#SVM classifier
clf3 = ensemble.RandomForestClassifier(n_estimators=30, max_depth=3)	#Random Forest classifier
clf5 = ensemble.ExtraTreesClassifier(n_estimators=30, max_depth=3)
ada_clf = ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier( \
	max_depth=2), n_estimators=50, algorithm="SAMME.R", learning_rate=0.5)
grid_SVM = model_selection.GridSearchCV(clf2, SVM_parameters)
grid_RF = model_selection.GridSearchCV(clf3, RF_parameters)
vote_clf = ensemble.VotingClassifier(estimators=[('knn',clf1),('svm', clf2), \
	('rf',clf3),('tree',clf5), ('ada',ada_clf), ('gSVM',grid_SVM), ('gRF', grid_RF)], voting='hard')

#Function for ROC curve
probs=[]
def plot_roc(tpr,fpr,thresholds):
   fig = plt.gcf()
   fig.set_size_inches(5.5, 4.5)
   plt.plot(fpr, tpr,label='correct')
   plt.plot([0,1],[0,1],'r--'),
   plt.title('ROC curve for Caffe-based classifier')
   plt.ylabel('True Positive Rate')
   plt.xlabel('False Positive Rate')
   plt.legend(loc='best')
   plt.grid()
#   plt.savefig('roc_plot.pgf')
   plt.show()

#empty arrays for k-fols=d scores
knn_scores1 = []
svm_scores1= [] 
trees_scores1 = []

#k-Fold
kf = model_selection.KFold(n_splits=5 )				#n_splits=20 yields the best accuracy
for train , test in kf.split(X):
	clf1.fit(X[train],y[train])
	clf2.fit(X[train],y[train])
	clf3.fit(X[train],y[train])
	acc_knn1= clf1.score(X[test],y[test])
	acc_svm1= clf2.score(X[test],y[test])
	acc_trees1= clf3.score(X[test],y[test])
	knn_scores1.append(acc_knn1)
	svm_scores1.append(acc_svm1)
	trees_scores1.append(acc_trees1)
	KNN_cf = metrics.confusion_matrix(y[test], clf1.predict(X[test]))
	SVM_cf = metrics.confusion_matrix(y[test], clf2.predict(X[test]))
	RF_cf = metrics.confusion_matrix(y[test], clf3.predict(X[test]))

print"KNN cf:\n", KNN_cf,"\nSVM cf:\n" ,SVM_cf, "\nRF cf:\n", RF_cf
print "\nUsing k-fold:\n", "k-NN score: ",100*np.mean(knn_scores1)
print"SVM score: ", 100*np.mean(svm_scores1)
print "Random Forest score: ", 100*np.mean(trees_scores1)

#Scores for Leave One Out CV
knn_scores2 = []
svm_scores2 = []
trees_scores2 = []

#Spliting data using Leave One Out cross validation
#loo = model_selection.LeaveOneOut()
#for train, test in loo.split(X):
#	clf1.fit(X[train],y[train])
#	clf2.fit(X[train],y[train])
#	clf3.fit(X[train],y[train])
#	acc_knn2 = clf1.score(X[test],y[test])
#	acc_svm2 = clf2.score(X[test],y[test])
#	acc_trees2 = clf3.score(X[test],y[test])
#	knn_scores2.append(acc_knn2)
#	svm_scores2.append(acc_svm2)
#	trees_scores2.append(acc_trees2)
#	KNNtemp_pred= clf1.predict_proba(X[test])
#	temp_pred= clf2.predict_proba(X[test])
#	RFtemp_pred = clf3.predict_proba(X[test])
#	probs.append(temp_pred[0,0])
if (len(sys.argv) < 2):
	print "Usage: python classify.py <training percentage>"
	exit()

p=float(sys.argv[1])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-p, random_state=0)

for clf in (clf1, clf2, clf3, clf5,ada_clf, grid_SVM, grid_RF, vote_clf):
	clf.fit(X_train,y_train)
	print clf.__class__.__name__, 100*clf.score(X_test,y_test)

#print "\nUsing Leave One Out:\n","k-nn score:", 100*np.mean(knn_scores2)
#print "SVM score:",svm_score #100*np.mean(svm_scores2)
#print "Random forest score:", rf_score #100*np.mean(trees_scores2)
#print "Tree score:", tree_score

#Using cross validation score for the data
#knn_scores3 = model_selection.cross_val_score(clf1, X, y, cv=5)		#cv=20 yields the best accuracy
#svm_scores3 = model_selection.cross_val_score(clf2, X, y, cv=5)		#cv=15 yields the best accuracy
#trees_scores3 = model_selection.cross_val_score(clf3, X, y, cv=5)		#cv=15 yileds the best accuracy
#gauss_scores3 = model_selection.cross_val_score(clf4, X, y, cv=5)
#print "\nUsing CV metrics:\n","k-NN score:", 100*np.mean(knn_scores3)
#print "SVM score:",100*np.mean(svm_scores3)
#print "Random Forest score:",100*np.mean(trees_scores3)

fpr, tpr, thresholds = metrics.roc_curve(y,probs, pos_label=2)

plot_roc(tpr,fpr,thresholds)
print "AUC: ", metrics.auc(fpr,tpr)




