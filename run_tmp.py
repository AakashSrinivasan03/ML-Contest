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
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from bayes_opt import BayesianOptimization


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
#train_X=train_X[:6000,:]
#train_Y=train_Y[:6000]

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
	if (Princ_comp_analysis.explained_variance_[i] < max*0.005): #0.0001
		break;
	i=i+1
print "i",i	
eigen_vectors=Princ_comp_analysis.components_[0:i].T
train_X=np.dot(train_X,eigen_vectors)
print train_X.shape
print train_Y
#test_X=np.dot(test_X,eigen_vectors)		

#train_X=Princ_comp_analysis.transform(train_X) 
#########################################################################################
def xgb_evaluate(min_child_weight,
                 colsample_bytree,
                 max_depth,
                 subsample,
                 gamma,
                 alpha
                 ):

    params['min_child_weight'] = int(min_child_weight)
    #params['cosample_bytree'] = max(min(colsample_bytree, 1.0), 0.0)
    params['max_depth'] = int(max_depth)
    #params['subsample'] = max(min(subsample, 1), 0)
    params['gamma'] = gamma
    params['alpha'] = alpha


    cv_result = xgb.cv(params, xgtrain, num_boost_round=3000, nfold=5,
             seed=random_state,
             callbacks=[xgb.callback.early_stop(50)])
    print "wowww",-cv_result['test-mae-mean'].values[-1],params['min_child_weight'],params['max_depth'],params['gamma'],params['alpha']
    return -cv_result['test-mae-mean'].values[-1]
xgtrain=xgb.DMatrix(train_X, train_Y)
num_rounds = 3000
random_state = 2016
num_iter = 25
init_points = 5
params = {
        'eta': 0.1,
        'silent': 1,
        'eval_metric': 'mae',
        'verbose_eval': True,
        'seed': random_state
    }

xgbBO = BayesianOptimization(xgb_evaluate, {'min_child_weight': (1, 20),
                                                'colsample_bytree': (0.1, 1),
                                                'max_depth': (5, 15),
                                                'subsample': (0.5, 1),
                                                'gamma': (0, 10),
                                                'alpha': (0, 10),
                                                })

print xgbBO.maximize(init_points=5, n_iter=25)
print xgbBO
print xgbBO.res['max']



#############################################################################################



