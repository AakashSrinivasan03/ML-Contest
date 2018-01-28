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
from sklearn.ensemble import AdaBoostClassifier

train_data = pd.read_csv('/home/manikandan/Downloads/train.csv',delimiter=',',index_col=0)
test_data = pd.read_csv('/home/manikandan/Downloads/test.csv',delimiter=',',index_col=0)
train_data=train_data.reindex(np.random.permutation(train_data.index))
#train_data = train_data.iloc[0:800]
#test_data = test_data.iloc[0:200]


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
train_Y = train_Y.as_matrix()
cross_val_X=train_X[6000:,:]
cross_val_Y=train_Y[6000:]
train_X=train_X[:6000,:]
train_Y=train_Y[:6000]

##scaler=sklearn.preprocessing.StandardScaler()    ##############Scaling data
##scaler.fit(train_X)
#scaler.fit(testing_features)
##train_X=scaler.transform(train_X) #############transforming training and test data
impute=sklearn.preprocessing.Imputer(strategy='median')
impute.fit(test_X)
test_X=impute.transform(test_X)
##test_X= scaler.transform(test_X)
#train_X=train_X[:20,:]
#train_Y=train_Y[:20]
#test_X
#train_X=train_X -train_X.mean()
#train_Y=train_Y -train_Y.mean()



print train_Y
print train_X
Princ_comp_analysis=PCA();###performing Principal component analysis:Note only one direction is desired
Princ_comp_analysis.fit(train_X);
print "here"
max=np.max(Princ_comp_analysis.explained_variance_)
print "max",max
eigen_vectors=Princ_comp_analysis.components_
i=0;
while i<Princ_comp_analysis.explained_variance_.shape[0]:
	if (Princ_comp_analysis.explained_variance_[i] < max*0.001):
		break;
	i=i+1
print "i",i	
eigen_vectors=Princ_comp_analysis.components_[0:i].T
train_X=np.dot(train_X,eigen_vectors)
print train_X.shape
print train_Y
#test_X=np.dot(test_X,eigen_vectors)		

#train_X=Princ_comp_analysis.transform(train_X) 
#clf = XGBClassifier(max_depth=3,min_child_weight=3,reg_alpha=3, reg_lambda=5,learning_rate=0.1)
clf = XGBClassifier(max_depth=3,reg_alpha=3, reg_lambda=5,learning_rate=0.6)
print "htt"
clf.fit(train_X,train_Y)
y_train_pred = clf.predict(train_X)
accuracy=metrics.classification_report(train_Y,y_train_pred)
print("accuracy :{}").format(accuracy)
print "ttt"
#for name in test_data.columns.values:
#	median=test_data[name].median()
#	test_data[name] = test_data[name].fillna(median)
cross_val_X=np.dot(cross_val_X,eigen_vectors)
y_pred_cr = clf.predict(cross_val_X)
accuracy=metrics.classification_report(cross_val_Y,y_pred_cr)
print("accuracy :{}").format(accuracy)
print "ttt"

#print test_data.as_matrix()
test_X=np.dot(test_X,eigen_vectors)
y_pred = clf.predict(test_X)

print y_pred
test_id = [i for i in range(len(y_pred))]

columns = ['label']
sub = pd.DataFrame(data=y_pred, columns=columns)
sub['id'] = test_id
sub = sub[['id','label']]




sub.to_csv("sub_9.csv", index=False) 

