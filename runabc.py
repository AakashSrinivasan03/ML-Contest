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
    #   median = data.groupby(['label'])[name].median()
    #   data[name] = data[name].fillna(median)
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
cross_val_X=train_X[8000:,:]
cross_val_Y=train_Y[8000:]
train_X=train_X[:8000,:]
train_Y=train_Y[:8000]

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
    if (Princ_comp_analysis.explained_variance_[i] < max*0.005):
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
clf = XGBClassifier(max_depth=10,min_child_weight=11,reg_alpha=3,gamma=2.3)
print "htt"
clf.fit(train_X,train_Y)
y_train_pred = clf.predict(train_X)
accuracy=metrics.classification_report(train_Y,y_train_pred)
print("accuracy :{}").format(accuracy)
print "ttt"
#for name in test_data.columns.values:
#   median=test_data[name].median()
#   test_data[name] = test_data[name].fillna(median)
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




sub.to_csv("sub_13.csv", index=False) #12 befor
''' Step |   Time |      Value |     alpha |   colsample_bytree |     gamma |   max_depth |   min_child_weight |   subsample | 
Multiple eval metrics have been passed: 'test-mae' will be used for early stopping.

Will train until test-mae hasn't improved in 50 rounds.
Stopping. Best iteration:
[214]   train-mae:0.447176+0.00427596   test-mae:6.42769+0.0312546

wowww -6.427688 12 12 2.41227658994 9.09475067222
    1 | 03m17s |   -6.42769 |    9.0948 |             0.3488 |    2.4123 |     12.7228 |            12.3555 |      0.6841 | 
Multiple eval metrics have been passed: 'test-mae' will be used for early stopping.

Will train until test-mae hasn't improved in 50 rounds.
Stopping. Best iteration:
[204]   train-mae:0.54603+0.0063563 test-mae:6.41637+0.0346764

wowww -6.4163668 13 13 4.59946806165 6.85813363545
    2 | 13m59s |   -6.41637 |    6.8581 |             0.1529 |    4.5995 |     13.5913 |            13.4880 |      0.6489 | 
Multiple eval metrics have been passed: 'test-mae' will be used for early stopping.

Will train until test-mae hasn't improved in 50 rounds.
Stopping. Best iteration:
[235]   train-mae:2.02077+0.0260338 test-mae:6.41613+0.013679

wowww -6.4161314 18 7 6.22250820864 8.06410524612
    3 | 08m10s |   -6.41613 |    8.0641 |             0.4437 |    6.2225 |      7.1418 |            18.7876 |      0.9425 | 
Multiple eval metrics have been passed: 'test-mae' will be used for early stopping.

Will train until test-mae hasn't improved in 50 rounds.
Stopping. Best iteration:
[411]   train-mae:2.83632+0.0585844 test-mae:6.44911+0.0322845

wowww -6.4491076 12 5 8.35631500364 0.924908923153
    4 | 09m03s |   -6.44911 |    0.9249 |             0.3395 |    8.3563 |      5.8207 |            12.7275 |      0.6098 | 
Multiple eval metrics have been passed: 'test-mae' will be used for early stopping.

Will train until test-mae hasn't improved in 50 rounds.
Stopping. Best iteration:
[163]   train-mae:0.615312+0.014987 test-mae:6.41988+0.0501189

wowww -6.4198758 15 14 6.93148885035 3.1417862618
    5 | 14m29s |   -6.41988 |    3.1418 |             0.4184 |    6.9315 |     14.7375 |            15.3900 |      0.9554 | 
Bayesian Optimization
---------------------------------------------------------------------------------------------------------------------------
 Step |   Time |      Value |     alpha |   colsample_bytree |     gamma |   max_depth |   min_child_weight |   subsample | 
Multiple eval metrics have been passed: 'test-mae' will be used for early stopping.

Will train until test-mae hasn't improved in 50 rounds.
Stopping. Best iteration:
[182]   train-mae:0.761914+0.0145427    test-mae:6.55354+0.0475985

wowww -6.5535382 1 15 10.0 10.0
    6 | 26m37s |   -6.55354 |   10.0000 |             1.0000 |   10.0000 |     15.0000 |             1.0000 |      1.0000 | 
Multiple eval metrics have been passed: 'test-mae' will be used for early stopping.

Will train until test-mae hasn't improved in 50 rounds.
Stopping. Best iteration:
[282]   train-mae:0.0225588+0.00230632  test-mae:6.43407+0.0686141

wowww -6.4340658 20 15 0.0 0.0
    7 | 25m23s |   -6.43407 |    0.0000 |             0.1000 |    0.0000 |     15.0000 |            20.0000 |      0.5000 | 
Multiple eval metrics have been passed: 'test-mae' will be used for early stopping.

Will train until test-mae hasn't improved in 50 rounds.
Stopping. Best iteration:
[196]   train-mae:0.845819+0.0123615    test-mae:6.42585+0.0100702

wowww -6.4258548 20 15 10.0 10.0
    8 | 20m06s |   -6.42585 |   10.0000 |             0.1000 |   10.0000 |     15.0000 |            20.0000 |      0.5000 | 
Multiple eval metrics have been passed: 'test-mae' will be used for early stopping.

Will train until test-mae hasn't improved in 50 rounds.
Stopping. Best iteration:
[167]   train-mae:0.000751+0.000140358  test-mae:6.55361+0.0637743

wowww -6.553614 1 15 0.0 0.0
    9 | 22m16s |   -6.55361 |    0.0000 |             0.1000 |    0.0000 |     15.0000 |             1.0000 |      1.0000 | 
Multiple eval metrics have been passed: 'test-mae' will be used for early stopping.

Will train until test-mae hasn't improved in 50 rounds.
Stopping. Best iteration:
[247]   train-mae:0.330414+0.00368451   test-mae:6.43952+0.0279971

wowww -6.4395192 12 14 1.03214321808 9.5830896145
   10 | 28m05s |   -6.43952 |    9.5831 |             0.6266 |    1.0321 |     14.8243 |            12.8488 |      0.5624 | Warning: Test point chose at random due to repeated sample.

Multiple eval metrics have been passed: 'test-mae' will be used for early stopping.

Will train until test-mae hasn't improved in 50 rounds.
Stopping. Best iteration:
[482]   train-mae:2.41184+0.0397859 test-mae:6.45009+0.0425224

wowww -6.4500866 20 5 0.0 9.99999999998
   11 | 18m58s |   -6.45009 |   10.0000 |             0.1000 |    0.0000 |      5.0000 |            20.0000 |      1.0000 | 
Multiple eval metrics have been passed: 'test-mae' will be used for early stopping.

Will train until test-mae hasn't improved in 50 rounds.
Stopping. Best iteration:
[237]   train-mae:3.91362+0.0282419 test-mae:6.48107+0.024615

wowww -6.4810734 16 5 10.0 10.0
   12 | 11m37s |   -6.48107 |   10.0000 |             0.1000 |   10.0000 |      5.0000 |            16.2668 |      1.0000 | 
Multiple eval metrics have been passed: 'test-mae' will be used for early stopping.

Will train until test-mae hasn't improved in 50 rounds.
Stopping. Best iteration:
[366]   train-mae:3.25001+0.0729218 test-mae:6.48753+0.0302435

wowww -6.4875312 20 5 9.98679764062 0.0
   13 | 14m31s |   -6.48753 |    0.0000 |             0.1000 |    9.9868 |      5.0000 |            20.0000 |      0.5000 | 
Multiple eval metrics have been passed: 'test-mae' will be used for early stopping.

Will train until test-mae hasn't improved in 50 rounds.
Stopping. Best iteration:
[324]   train-mae:2.96869+0.0354269 test-mae:6.46242+0.00664877

wowww -6.4624194 1 5 0.0 10.0
   14 | 14m58s |   -6.46242 |   10.0000 |             1.0000 |    0.0000 |      5.0000 |             1.0000 |      0.5000 | 
Multiple eval metrics have been passed: 'test-mae' will be used for early stopping.

Will train until test-mae hasn't improved in 50 rounds.
Stopping. Best iteration:
[289]   train-mae:3.5634+0.0926826  test-mae:6.467+0.0413641

wowww -6.4670018 10 5 0.0 0.0
   15 | 12m39s |   -6.46700 |    0.0000 |             1.0000 |    0.0000 |      5.0000 |            10.4823 |      1.0000 | 
Multiple eval metrics have been passed: 'test-mae' will be used for early stopping.

Will train until test-mae hasn't improved in 50 rounds.
Stopping. Best iteration:
[403]   train-mae:2.76788+0.0476973 test-mae:6.49029+0.0460757

wowww -6.4902886 1 5 10.0 0.0
   16 | 16m04s |   -6.49029 |    0.0000 |             1.0000 |   10.0000 |      5.0000 |             1.0000 |      1.0000 | 
Multiple eval metrics have been passed: 'test-mae' will be used for early stopping.

Will train until test-mae hasn't improved in 50 rounds.
Stopping. Best iteration:
[318]   train-mae:0.249486+0.000847232  test-mae:6.39143+0.0300411

wowww -6.3914284 20 15 0.35877834116 10.0
   17 | 40m16s |   -6.39143 |   10.0000 |             1.0000 |    0.3588 |     15.0000 |            20.0000 |      1.0000 | 
Multiple eval metrics have been passed: 'test-mae' will be used for early stopping.

Will train until test-mae hasn't improved in 50 rounds.
Stopping. Best iteration:
[227]   train-mae:0.612699+0.00551022   test-mae:6.38546+0.0304721

wowww -6.385464 19 12 4.85166454567 9.27485976765
/home/manikandan/anaconda2/lib/python2.7/site-packages/sklearn/gaussian_process/gpr.py:457: UserWarning: fmin_l_bfgs_b terminated abnormally with the  state: {'warnflag': 2, 'task': 'ABNORMAL_TERMINATION_IN_LNSRCH', 'grad': array([  1.39209348e-05]), 'nit': 5, 'funcalls': 52}
  " state: %s" % convergence_dict)
   18 | 29m57s |   -6.38546 |    9.2749 |             0.9513 |    4.8517 |     12.1761 |            19.6893 |      0.5004 | 
Multiple eval metrics have been passed: 'test-mae' will be used for early stopping.

Will train until test-mae hasn't improved in 50 rounds.
Stopping. Best iteration:
[242]   train-mae:0.472796+0.0113742    test-mae:6.40388+0.0219304

wowww -6.403877 19 14 3.01591428199 6.60348484547
   19 | 37m57s |   -6.40388 |    6.6035 |             0.9760 |    3.0159 |     14.8683 |            19.9539 |      0.8299 | 
Multiple eval metrics have been passed: 'test-mae' will be used for early stopping.

Will train until test-mae hasn't improved in 50 rounds.
Stopping. Best iteration:
[247]   train-mae:0.546871+0.00481493   test-mae:6.41521+0.0347321

wowww -6.4152066 19 11 3.50813341256 9.70864150929
   20 | 38m23s |   -6.41521 |    9.7086 |             0.1226 |    3.5081 |     11.6036 |            19.9938 |      0.9074 | 
Multiple eval metrics have been passed: 'test-mae' will be used for early stopping.
#################################################################################


Initialization
---------------------------------------------------------------------------------------------------------------------------
 Step |   Time |      Value |     alpha |   colsample_bytree |     gamma |   max_depth |   min_child_weight |   subsample | 
Multiple eval metrics have been passed: 'test-mae' will be used for early stopping.

Will train until test-mae hasn't improved in 50 rounds.
Stopping. Best iteration:
[110]   train-mae:0.393396+0.00177328   test-mae:6.88553+0.055751

wowww -6.8855296 8 13 4.07674453642 9.43887961989
    1 | 102m27s |   -6.88553 |    9.4389 |             0.4506 |    4.0767 |     13.8239 |             8.0455 |      0.7173 | 
Multiple eval metrics have been passed: 'test-mae' will be used for early stopping.

Will train until test-mae hasn't improved in 50 rounds.
Stopping. Best iteration:
[64]    train-mae:2.59899+0.0663284 test-mae:6.84868+0.109359

wowww -6.8486804 1 7 2.67147982287 4.28271737858
    2 | 49m05s |   -6.84868 |    4.2827 |             0.5610 |    2.6715 |      7.2955 |             1.6367 |      0.7972 | 
Multiple eval metrics have been passed: 'test-mae' will be used for early stopping.

Will train until test-mae hasn't improved in 50 rounds.
Stopping. Best iteration:
[125]   train-mae:0.262659+0.000502278  test-mae:6.83313+0.0916575

wowww -6.8331348 11 10 2.33166449536 3.06642273841
    3 | 125m15s |   -6.83313 |    3.0664 |             0.4327 |    2.3317 |     10.4875 |            11.2548 |      0.6087 | 
Multiple eval metrics have been passed: 'test-mae' will be used for early stopping.
