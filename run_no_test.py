from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from numpy import genfromtxt
from sklearn import metrics
import numpy as np 
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

train_data = pd.read_csv('/home/manikandan/Downloads/train_preprocessed.csv',delimiter=',',index_col=0)
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


#train_data = preprocess(train_data)

col_names = train_data.columns.values[1:]
train_X = train_data[col_names]
train_Y = train_data['label']

train_X = train_X.as_matrix()
train_Y = train_Y.as_matrix()

print train_Y
print train_X
clf = XGBClassifier(max_depth=3,min_child_weight=3,reg_alpha=3, reg_lambda=10)
print "htt"
clf.fit(train_X,train_Y)
y_train_pred = clf.predict(train_X)
accuracy=metrics.accuracy_score(train_Y,y_train_pred)
print("accuracy :{}").format(accuracy)
print "ttt"
#for name in test_data.columns.values:
#	median=test_data[name].median()
#	test_data[name] = test_data[name].fillna(median)

print test_data.as_matrix()
y_pred = clf.predict(test_data.as_matrix())

print y_pred
test_id = [i for i in range(len(y_pred))]

columns = ['label']
sub = pd.DataFrame(data=y_pred, columns=columns)
sub['id'] = test_id
sub = sub[['id','label']]



sub.to_csv("sub_7.csv", index=False) 

