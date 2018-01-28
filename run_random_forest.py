from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from numpy import genfromtxt
from sklearn import metrics
import numpy as np 
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import sklearn
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import AdaBoostClassifier
train_data = pd.read_csv('/home/manikandan/Downloads/train.csv',delimiter=',',index_col=0)
test_data = pd.read_csv('/home/manikandan/Downloads/test.csv',delimiter=',',index_col=0)
train_data=train_data.reindex(np.random.permutation(train_data.index))
#train_data = train_data.iloc[0:600,:]
#validate_data=train_data.iloc[6000:,:]
#test_data = test_data.iloc[0:100]


def preprocess(data):
	print "hhh"
	#data = data.set_index(['label'],drop=False)
	#col_names = data.columns.values
	#for name in col_names:
	#	median = data.groupby(['label'])[name].median()
	#	data[name] = data[name].fillna(median)
	#data = data.reset_index()
	#print "hth"
	#return data
	data = data.set_index(['label'])
	col_names = data.columns.values
	for name in col_names:
		median=test_data[name].median()
		data[name] = data[name].fillna(median)
	data = data.reset_index()

	return data


train_data = preprocess(train_data)

col_names = train_data.columns.values[1:]
train_X = train_data[col_names]
test_X = test_data[col_names]
train_Y = train_data['label']

train_X = train_X.as_matrix()
validate_X=train_X[6000:,:]
train_X=train_X[:6000,:]
train_Y = train_Y.as_matrix()
validate_Y=train_Y[6000:]
train_Y=train_Y[:6000]
##scaler=sklearn.preprocessing.StandardScaler()    ##############Scaling data
##scaler.fit(train_X)
##scaler.fit(testing_features)
##train_X=scaler.transform(train_X) #############transforming training and test data
impute=sklearn.preprocessing.Imputer(strategy='median')
impute.fit(test_X)
test_X=impute.transform(test_X)
##test_X= scaler.transform(test_X)
##validate_X=scaler.transform(validate_X)
#train_X=train_X[:20,:]
#train_Y=train_Y[:20]
#test_X
#train_X=train_X -train_X.mean()
#train_Y=train_Y -train_Y.mean()



print train_Y
print train_X
'''Princ_comp_analysis=PCA();###performing Principal component analysis:Note only one direction is desired
Princ_comp_analysis.fit(train_X);
print "here"
max=np.max(Princ_comp_analysis.explained_variance_)
print "max",max
eigen_vectors=Princ_comp_analysis.components_
i=0;
while i<Princ_comp_analysis.explained_variance_.shape[0]:
	if (Princ_comp_analysis.explained_variance_[i] < max*0.0003):#
		break;
	i=i+1
print "i",i	
eigen_vectors=Princ_comp_analysis.components_[0:2500].T
train_X=np.dot(train_X,eigen_vectors)
print train_X.shape
print train_Y'''
#test_X=np.dot(test_X,eigen_vectors)		

#train_X=Princ_comp_analysis.transform(train_X) 

clf1 = RandomForestClassifier(n_estimators=200,max_depth=6, n_jobs=-1,min_samples_leaf=3,max_features=0.8)
#clf1=AdaBoostClassifier(base_estimator=clf,n_estimators=90)
#clf1=BaggingClassifier(base_estimator=clf,max_samples=0.4,n_jobs=-1)
print "htt"
clf1.fit(train_X,train_Y)
y_train_pred = clf1.predict(train_X)
accuracy=metrics.classification_report(train_Y,y_train_pred)
print("accuracy :{}").format(accuracy)
print "ttt"
#validate_X=np.dot(validate_X,eigen_vectors)
y_train_pred_validate = clf1.predict(validate_X)
accuracy=metrics.classification_report(validate_Y,y_train_pred_validate)
print("accuracy :{}").format(accuracy)
print "ttt"
#test_X=np.dot(test_X,eigen_vectors)
y_pred = clf1.predict(test_X)
#for name in test_data.columns.values:
#	median=test_data[name].median()
#	test_data[name] = test_data[name].fillna(median)

#print test_data.as_matrix()
#test_X=np.dot(test_X,eigen_vectors)
#y_pred = clf1.predict(test_X)

print y_pred
test_id = [i for i in range(len(y_pred))]

columns = ['label']
sub = pd.DataFrame(data=y_pred, columns=columns)
sub['id'] = test_id
sub = sub[['id','label']]




sub.to_csv("sub_10.csv", index=False) 

