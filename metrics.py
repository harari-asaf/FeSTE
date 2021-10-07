import pandas as pd
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 1000)
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from sklearn.metrics import precision_recall_curve, f1_score, recall_score, precision_score, confusion_matrix, accuracy_score, roc_auc_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import  label_binarize, minmax_scale, OneHotEncoder, LabelBinarizer, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier



def compute_matrices(y_train, y_test, y_pred,y_pred_prob, model):
    indexes = np.unique(y_train, return_index=True)[1]
    classes = [y_train[index] for index in sorted(indexes)]
    LB = LabelBinarizer()
    LB.fit(model.classes_)

    f1 = f1_score(label_binarize(y_test, classes=classes), label_binarize(y_pred, classes=classes),
                  average='weighted')
    auc = roc_auc_score(label_binarize(y_test, classes=classes),
                        label_binarize(y_pred, classes=classes))
    return f1, auc

# X,y,train_idx,test_idx,classifier,original_dataset,classes = X, y, record_for_classifier_traiaing,test_idx, classifier, X_dataset, classes
def trian_predict_classifier(X,y,train_idx,test_idx,classifier,original_dataset,classes):
    # split train test
    y_train, y_test = y[train_idx], y[test_idx]
    X_train, X_test = X[train_idx], X[test_idx]


    """scaling"""
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = classifier.fit(X_train, y_train)
    # predict
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)
    if len(classes) ==2:
        y_pred_prob = y_pred_prob[:,1]

    f1, auc = compute_matrices(y_train, y_test, y_pred, y_pred_prob, model)

    return f1, auc

def compute_acc_on_fold(dataset, new_feat_np, train_idx,test_idx, target_idx = -1 ):
    ''' Received two datasets, execute RF model fols * iterations times and perform t test on the differences vector

        Args:
              dataset (DF) -  dataset
              cand_dataset (DF) -another dataset (the null hypothesis- this dataset is better)
              folds (int) - number of folds for cross validation
              iterations (int) - number of iterations of k folds CV.
              metric (str) - auc/acc
              target_name (str='target') =  the name of the target column
              acc_decimals (int = 4) = how to round the returned results

        Returns:
             mean difference, mean of the datasets, mean of the cand dataset, p value '''

    # convert to np
    y = np.array(dataset.iloc[:,target_idx])
    # create X data
    X_dataset_no_orig_feat = dataset.iloc[:, [0]]
    X_dataset = dataset.drop(dataset.columns[ [0,target_idx]],axis =1)


    X_no_orig_feat = np.array(pd.get_dummies(X_dataset_no_orig_feat))
    if new_feat_np.shape[1]>2:
        new_feat_np_scaled = minmax_scale(new_feat_np.T).T
    else: new_feat_np_scaled = new_feat_np.copy()

    X_cand = new_feat_np_scaled
    if X_dataset.shape[1]>0:
        X = np.array(pd.get_dummies(X_dataset))
        X_cand_with_feat = np.concatenate((new_feat_np_scaled, X),axis=1)
    else:
        X = X_dataset.to_numpy()
        X_cand_with_feat = new_feat_np_scaled

    classes = dataset.iloc[:,target_idx].unique()


    """TRAIN ON ORIGINAL DATASET"""
    for classifier in [RandomForestClassifier(),MLPClassifier(),GradientBoostingClassifier(),KNeighborsClassifier(),SVC(probability=True)]:
        print(classifier)
        train_ratio = 0.5

        record_for_classifier_traiaing = train_idx[int(train_idx.shape[0] * train_ratio):]

        f1, auc = trian_predict_classifier(X, y, record_for_classifier_traiaing,test_idx, classifier,
                                                                                               X_dataset, classes)

        f1_only_new, auc_only_new = trian_predict_classifier(X_cand, y, record_for_classifier_traiaing, test_idx, classifier,
                                           X_dataset, classes)
        f1_with_feat, auc_with_feat= trian_predict_classifier(X_cand_with_feat, y, record_for_classifier_traiaing, test_idx, classifier,X_dataset, classes)


        print(' original  f (auc): {}, ({}) '.format(f1, auc))
        print(' FeSTE  f (auc): {}, ({}) '.format(f1_only_new, auc_only_new))
        print(' original and FeSTE  f (auc): {}, ({}) '.format(f1_with_feat, auc_with_feat))