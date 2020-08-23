import numpy as np
import pandas as pd
pd.set_option('max_columns', None)
pd.set_option('max_rows', None)

raw_train_identity= pd.read_csv(r'C:\Users\lukem\Desktop\Github AI Projects\Data for ai competitions\train_identity.csv')
raw_test_identity = pd.read_csv(r'C:\Users\lukem\Desktop\Github AI Projects\Data for ai competitions\test_identity.csv')
raw_train_transaction = pd.read_csv(r'C:\Users\lukem\Desktop\Github AI Projects\Data for ai competitions\train_transaction.csv')
raw_test_transaction = pd.read_csv(r'C:\Users\lukem\Desktop\Github AI Projects\Data for ai competitions\test_transaction.csv')

train = raw_train_transaction.merge(raw_train_identity, how='left', left_index=True, right_index=True)
test = raw_test_transaction.merge(raw_test_identity, how='left', left_index=True, right_index=True)

del raw_train_identity, raw_test_identity, raw_train_transaction, raw_test_transaction




null_sum = raw_train_transaction.isnull().sum()

train_transaction = raw_train_transaction.drop(['addr1','addr2','dist1','dist2','P_emaildomain','R_emaildomain'], axis = 1)
train_transaction = raw_train_transaction.loc[['isFraud','TransactionDT','TransactionAmt','ProductCD','card1','card3',
                                            'card6','C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14',
                                            'V95','V96','V97','V98','V99','V100','V101','V102','V103','V104','V105','V106',
                                            'V107','V108','V109','V110','V111','V112','V113','V114','V115','V116','V117','V118',
                                            'V119','V120','V121','V122','V123','V124','V125','V126','V127']]
train_transaction_cols = train_transaction.columns
train_transaction = raw_train_transaction.loc[['isFraud']]
