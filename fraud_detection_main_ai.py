import numpy as np
import pandas as pd
pd.set_option('max_columns', None)

raw_train_identity= pd.read_csv(r'C:\Users\lukem\Desktop\Github AI Projects\Data for ai competitions\train_identity.csv')
raw_test_identity = pd.read_csv(r'C:\Users\lukem\Desktop\Github AI Projects\Data for ai competitions\test_identity.csv')
raw_train_transaction = pd.read_csv(r'C:\Users\lukem\Desktop\Github AI Projects\Data for ai competitions\train_transaction.csv')
raw_test_transaction = pd.read_csv(r'C:\Users\lukem\Desktop\Github AI Projects\Data for ai competitions\test_transaction.csv')

raw_train_identity.dtypes
raw_train_transaction.dtypes

transaction_cols = raw_train_transaction.columns

# It looks like the raw_train_identity provides some identity information for some of the people making transactions
# And the raw_train_transaction has the transaction information and it has the target column: isFraud
# I'm going to start out by just using the raw_train_transaction