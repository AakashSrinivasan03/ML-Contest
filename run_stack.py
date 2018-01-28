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
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
import pickle
from mlxtend.classifier import StackingClassifier

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
test_X = test_X.as_matrix()
cross_val_X=train_X[7200:,:]
cross_val_Y=train_Y[7200:]
#train_X=train_X[:7200,:]
#train_Y=train_Y[:7200]

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
'''Princ_comp_analysis=PCA();###performing Principal component analysis:Note only one direction is desired
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
eigen_vectors=Princ_comp_analysis.components_[0:2500].T
train_X=np.dot(train_X,eigen_vectors)
print train_X.shape
print train_Y'''
#test_X=np.dot(test_X,eigen_vectors)        

#train_X=Princ_comp_analysis.transform(train_X) a
#clf = XGBClassifier(max_depth=3,min_child_weight=3,reg_alpha=3, reg_lambda=5,learning_rate=0.1)



clf1= XGBClassifier(max_depth=7,min_child_weight=6,reg_alpha=0.01,gamma=4.8729,colsample_bytree= 0.4328 ,subsample=0.9797)
#clf2=SVC(C=0.015, kernel='linear', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=None)
clf2=SVC(C=10, kernel='rbf', degree=3, gamma=0.005, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=True, max_iter=-1, decision_function_shape='ovr', random_state=None)
clf3=RandomForestClassifier(n_estimators=100,max_depth=7,min_samples_leaf=3,max_features=0.8)
#model=sklearn.ensemble.VotingClassifier([('b',clf2),('a',clf1),('c',clf3)], voting='hard', weights=[0.39,0.31,0.20], flatten_transform=None)
print "aaa"
#model = model.fit(train_X, train_Y)
#clf2.fit(train_X,train_Y)
#y_2_pred_cr = clf2.predict(cross_val_X)
#print metrics.classification_report(cross_val_Y,y_2_pred_cr)
#filename = 'finalized_model6.sav'
#pickle.dump(clf2, open(filename, 'wb'))
print "ccc"
#clf1.fit(train_X,train_Y)
#y_1_pred_cr = clf1.predict(cross_val_X)
#print metrics.classification_report(cross_val_Y,y_1_pred_cr)
#filename = 'finalized_model7.sav'
#pickle.dump(clf1, open(filename, 'wb'))
print "ccc"
'''clf3.fit(train_X,train_Y)
y_3_pred_cr = clf3.predict(cross_val_X)
print metrics.classification_report(cross_val_Y,y_3_pred_cr)
filename = 'finalized_model8.sav'
pickle.dump(clf3, open(filename, 'wb'))'''
print "ccc"
X_stack_train=np.zeros((train_X.shape[0],3))
X_stack_test=np.zeros((test_X.shape[0],3))
clf1=pickle.load(open ('finalized_model7.sav', 'rb'))
clf2=pickle.load(open ('finalized_model6.sav', 'rb'))
clf3=pickle.load(open ('finalized_model8.sav', 'rb'))
X_stack_test[:,0]=clf2.predict(test_X)
X_stack_test[:,1]=clf1.predict(test_X)
X_stack_test[:,2]=clf3.predict(test_X)
np.savetxt("stack_test.csv",X_stack_test , delimiter=",")
print "aaa"
X_stack_train[:,0]=clf2.predict(train_X)
X_stack_train[:,1]=clf1.predict(train_X)
X_stack_train[:,2]=clf3.predict(train_X)
np.savetxt("stack_train.csv",X_stack_train , delimiter=",")

#sclf = StackingClassifier(classifiers=[clf1, clf2, clf3], 
#                          meta_classifier=lr)
'''i=0    
for train_index, test_index in splits.split(training_features, training_label):
    X_train, X_test = training_features[train_index], training_features[test_index]
    y_train, y_test = training_label[train_index], training_label[test_index]
    if(i==0):
        model=SVC(C=0.015, kernel='linear', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=None)
    elif(i==1):
        model=sklearn.ensemble.GradientBoostingClassifier(loss='deviance', learning_rate=0.019, n_estimators=190 ,subsample=0.8299, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=260, min_weight_fraction_leaf=0.0, max_depth=5, min_impurity_decrease=0.0, min_impurity_split=None, init=None, random_state=None, max_features=None, verbose=2, max_leaf_nodes=None, warm_start=False, presort='auto')
    else:
        model=RandomForestClassifier(n_estimators=100,max_depth=10, n_jobs=-1,min_samples_leaf=3,max_features=0.8)
    model.fit(X_train,y_train)
    ###svm.libsvm.svm_save_model('svm_model1.model'.encode(), model)
    ######joblib.dump(model, 'svm_model1.model')
    pred = model.predict(X_test)
    #print pred[25]
    print metrics.classification_report(y_test, pred)
    i=i+1'''
    #model=LinearSVC(penalty='l1', loss='squared_hinge', dual=False, tol=0.0001, C=0.133, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=1, random_state=None, max_iter=1000)
    #model.fit(train_X,train_Y)

            ###svm.libsvm.svm_save_model('svm_model1.model'.encode(), model)
            ######joblib.dump(model, 'svm_model1.model')
#filename = 'finalized_model6.sav'
#pickle.dump(model, open(filename, 'wb'))            
#pred = model.predict(test_X)
'''y_pred_1 = model.predict(train_X)
print metrics.classification_report(train_Y,y_pred_1)
y_pred_1 = model.predict(cross_val_X)
print metrics.classification_report(cross_val_Y,y_pred_1)
y_pred = model.predict(test_X)
print y_pred'''
#print metrics.classification_report(train_Y,pred)
#clf.fit(train_X,train_Y)
'''y_train_pred = clf.predict(train_X)
accuracy=metrics.classification_report(train_Y,y_train_pred)
print("accuracy :{}").format(accuracy)
print "ttt"
#for name in test_data.columns.values:
#   median=test_data[name].median()
#   test_data[name] = test_data[name].fillna(median)
#cross_val_X=np.dot(cross_val_X,eigen_vectors)
y_pred_cr = clf.predict(cross_val_X)
accuracy=metrics.classification_report(cross_val_Y,y_pred_cr)
print("accuracy :{}").format(accuracy)
print "ttt"

#print test_data.as_matrix()
#test_X=np.dot(test_X,eigen_vectors)
y_pred = clf.predict(test_X)

print y_pred'''
'''test_id = [i for i in range(len(y_pred))]

columns = ['label']
sub = pd.DataFrame(data=y_pred, columns=columns)
sub['id'] = test_id
sub = sub[['id','label']]




sub.to_csv("sub_9_2_4.csv", index=False) '''

