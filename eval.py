# import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from numpy.core.umath_tests import inner1d
import plotly

dataset = pd.read_csv('LoanStats.csv')
X = dataset.iloc[:, -1].values
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

llist = [358, 995, 850, 722, 942, 900, 902, 802, 61, 199, 200, 325, 331, 328, 303, 968, 832, 606, 627, 462, 528, 503, 672, 417, 701, 40, 212, 21, 490, 497, 558, 395, 631, 590, 689, 895, 32, 70, 875, 100, 275, 582, 441, 741, 972, 152, 28, 290, 574, 372, 787, 843, 57, 245, 980, 258, 532, 829]

for k in llist:
    p =0
    np= 0
    for i in range(len(X)):
        if(X[i]==k):
            if(y[i] == 0):
                np=np+1
            else:
                p = p+1
    print(k,np/(np+p)*100)
