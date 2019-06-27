import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from numpy.core.umath_tests import inner1d
import datetime

dataset = pd.read_csv('LoanStats.csv')
X = dataset.iloc[:, :28].values
y = dataset.iloc[:, -2].values
df = pd.DataFrame(data=X)

for i in range(len(y)):
    if (y[i]=='Fully Paid' or y[i]=='Current' or y[i]== 'Does not meet the credit policy. Status:Fully Paid'):
        y[i]=1
    elif (y[i]=='Charged Off' or y[i] == 'Does not meet the credit policy. Status:Charged Off' or y[i]=='Late (31-120 days)' or y[i]=='Late (16-30 days)' or y[i] =='In Grace Period' or y[i]=='Default'):
        y[i]=0
    else:
        print("None")
        y[i]=0


from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 0:6])
X[:, 0:6] = imputer.transform(X[:, 0:6])

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X[:,0:6] = sc_X.fit_transform(X[:,0:6])

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 6] = labelencoder_X.fit_transform(X[:, 6])

imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
imputer = imputer.fit(X[:,6].reshape(-1,1))
X[:, 6:7] = imputer.transform(X[:, 6].reshape(-1,1))

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:,7].reshape(-1,1))
X[:, 7:8] = imputer.transform(X[:, 7].reshape(-1,1))
X[:,7:8] = sc_X.fit_transform(X[:,7:8])

labelencoder_X = LabelEncoder()
X[:, 8] = labelencoder_X.fit_transform(X[:, 8])
imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
imputer = imputer.fit(X[:,8].reshape(-1,1))
X[:, 8:9] = imputer.transform(X[:, 8].reshape(-1,1))

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:,9].reshape(-1,1))
X[:, 9:10] = imputer.transform(X[:, 9].reshape(-1,1))
X[:,9:10] = sc_X.fit_transform(X[:,9:10])
X[:,9:10] = sc_X.fit_transform(X[:,9:10])

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:,10:12])
X[:, 10:12] = imputer.transform(X[:, 10:12])
X[:,10:12] = sc_X.fit_transform(X[:,10:12])

labelencoder_X = LabelEncoder()
X[:, 12] = labelencoder_X.fit_transform(X[:, 12])
imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
imputer = imputer.fit(X[:,12].reshape(-1,1))
X[:, 12:13] = imputer.transform(X[:, 12].reshape(-1,1))

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:,13:14])
X[:, 13:14] = imputer.transform(X[:, 13:14])
X[:,13:14] = sc_X.fit_transform(X[:,13:14])

imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
imputer = imputer.fit(X[:,14:17])
X[:, 14:17] = imputer.transform(X[:, 14:17])
X[:,14:17] = sc_X.fit_transform(X[:,14:17])

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:,17:19])
X[:, 17:19] = imputer.transform(X[:, 17:19])
X[:,17:19] = sc_X.fit_transform(X[:,17:19])

imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
imputer = imputer.fit(X[:,19:20])
X[:, 19:20] = imputer.transform(X[:,19:20])
X[:,19:20] = sc_X.fit_transform(X[:,19:20])

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:,20:22])
X[:, 20:22] = imputer.transform(X[:,20:22])
X[:,20:22] = sc_X.fit_transform(X[:,20:22])

for i in range(len(X)):
    if(X[i][27]=='Source Verified'):
        X[i][27] = 'Verified'
labelencoder_X = LabelEncoder()
X[:, 22] = labelencoder_X.fit_transform(X[:, 22])
imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
imputer = imputer.fit(X[:,22].reshape(-1,1))
X[:, 22:23] = imputer.transform(X[:, 22].reshape(-1,1))

print(X[0,:23])

data = X[:,:22]
# data2 = X[:,20:22]
# data = np.column_stack((data1,data2))
y=y.astype('int')
y = y.reshape(y.size, 1)



start = datetime.datetime.now()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size = 0.2, random_state = 0)

from sklearn.ensemble import GradientBoostingClassifier

# learning_rates = [0.05, 0.1, 0.25, 0.5, 0.75, 1]
# for learning_rate in learning_rates:
gb = GradientBoostingClassifier(n_estimators=50, learning_rate = 0.5, max_features="sqrt", max_depth = 8, random_state = 0,verbose=1)
gb.fit(X_train, y_train)
print("Learning rate: ", 0.5)
print("Accuracy score (training): {0:.3f}".format(gb.score(X_train, y_train)))
print("Accuracy score (validation): {0:.3f}".format(gb.score(X_test, y_test)))
print()
end = datetime.datetime.now()

print(end-start)

predictions = gb.predict(X_test)
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))
print()
print("Classification Report")
print(classification_report(y_test, predictions))
y_scores_gb = gb.decision_function(X_test)
fpr_gb, tpr_gb, _ = roc_curve(y_test, y_scores_gb)
roc_auc_gb = auc(fpr_gb, tpr_gb)
print("Area under ROC curve = {:0.2f}".format(roc_auc_gb))
# feature importance
print(gb.feature_importances_)
# plot
plt.bar(range(len(gb.feature_importances_)), gb.feature_importances_)
plt.show()
