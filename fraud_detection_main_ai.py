import numpy as np
import pandas as pd
pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
from sklearn import preprocessing
import xgboost as xgb

raw_train_identity= pd.read_csv(r'C:\Users\lukem\Desktop\Github AI Projects\Data for ai competitions\train_identity.csv')
raw_test_identity = pd.read_csv(r'C:\Users\lukem\Desktop\Github AI Projects\Data for ai competitions\test_identity.csv')
raw_train_transaction = pd.read_csv(r'C:\Users\lukem\Desktop\Github AI Projects\Data for ai competitions\train_transaction.csv')
raw_test_transaction = pd.read_csv(r'C:\Users\lukem\Desktop\Github AI Projects\Data for ai competitions\test_transaction.csv')
sample_submission = pd.read_csv(r'C:\Users\lukem\Desktop\Github AI Projects\Data for ai competitions\sample_submission.csv')

train = raw_train_transaction.merge(raw_train_identity, how='left', left_index=True, right_index=True)
test = raw_test_transaction.merge(raw_test_identity, how='left', left_index=True, right_index=True)

del raw_train_identity, raw_test_identity, raw_train_transaction, raw_test_transaction

# set up target variable and remove from main dataframe
y_train = train['isFraud']
X_train = train.drop(labels = 'isFraud', axis=1)
X_test = test.copy()

del train, test

# Fill in Nans
X_train = X_train.fillna(-999)
X_test = X_test.fillna(-999)

# For some reason, all of the "id_##" in the X_test showed up as "id-##" which is different than the X_train
# This for loop fixes that
temp_str = ""
for f in X_test.columns:
    if "id-" in X_test[f].name:
        temp_str = X_test[f].name
        temp_str = temp_str.replace("id-","id_")
        X_test = X_test.rename(columns = {f:temp_str})


# For loop to encode all columns with "object" datatype
for f in X_train.columns:
    if X_train[f].dtype == "object" or X_test[f].dtype == "object":
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(X_train[f].values) + list(X_test[f].values))
        X_train[f] = lbl.transform(list(X_train[f].values))
        X_test[f] = lbl.transform(list(X_test[f].values))
        
# Training
# To activate GPU usage, simply use tree_method = 'gpu_hist'
#took 13 min 47s
clf = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=9,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    missing=-999,
    random_state=2019,
    tree_method='gpu_hist'  # THE MAGICAL PARAMETER
)
%time clf.fit(X_train, y_train)

sample_submission['isFraud'] = clf.predict_proba(X_test)[:,1]

path = r'C:\Users\lukem\Desktop\Github AI Projects\Submissions\ '
sample_submission.to_csv(path + 'IEEE_fraud_xgb_submission_v6.csv', index = False)