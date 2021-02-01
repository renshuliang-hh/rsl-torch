from sklearn.datasets import load_boston

dataset = load_boston()

dir(dataset)

print(dataset['DESCR'])

dataset['data'].shape
import seaborn as sns

import pandas as pd

dataframe = pd.DataFrame(dataset['data'])

dataframe.columns = dataset['feature_names']

dataframe

import matplotlib.pyplot as plt

dataframe.corr()
aaa=sns.heatmap(dataframe.corr(),annot=True)
plt.show()
important_features_data = dataframe[['CRIM', 'RM', 'AGE']]

important_features_data

features_to_price = dict()

house_age = dataframe['AGE']
house_rm = dataframe['RM']

dataframe['price'] = dataset['target']
price = dataframe['price']

house_rm_price = {a: p for a, p in zip(house_rm, price)}

house_rm_price

sorted([(1, -1), (2, -3), (0, 9)], key=lambda t: t[1])

import numpy as np



def knn(query_age, data, top_n=3):
    most_similar = sorted(data.items(), key=lambda x_y: (x_y[0] - query_age) ** 2)[:top_n]
    print(most_similar)
    similary_ys = [y for x, y in most_similar]

    return np.mean(similary_ys)

pp={
    'A':40,
    'B':50,
    'C':25


}
def get_first_iterm(e):
    return e[1]

import random

X_rm = dataframe['RM']
Y = dataframe['price']

def Loss(Y,X_rm,k, b):
    return np.mean((Y - (k * X_rm + b))**2)

min_loss = float('inf')
best_k, best_b = None, None
total_times=100000
def mse_loss(y,yhat):
    return  np.mean((np.array(y)-np.array(yhat))**2)
def partial_k(x,y,k,b):


    return 2*np.mean((y-(k*x+b))*(-x))
def partial_b(x,y,k,b):
    return 2 * np.mean((y - (k * x + b))*(-1))
k,b=random.randint(-10,10),random.randint(-10,10)
# for t in range (total_times):
#
#     k=k+(-1)*partial_k(X_rm,Y,k,b)*0.03
#     b = b+ (-1) * partial_b(X_rm, Y, k, b)*0.03
#     loss=Loss(Y,X_rm,k,b)
#     print(loss)
#     if loss<min_loss:
#         best_k,best_b=k,b
#         min_loss=loss
#         print("在{}时刻最好k：{}和b：{}，loss：{}".format(t,k,b,loss))

def sigmod(x):
    return 1/(1+np.exp(-x))
def random_linear(x):
    k,b=random.random(),random.random()
    return k*x+b
sub_x=np.linspace(-10,10)
for _ in range(5):
    i=random.choice(range(len(sub_x)))
    oput1=np.concatenate([random_linear(sub_x[:i]),random_linear(sub_x[i:])])
    plt.plot(sub_x,oput1)

plt.show()