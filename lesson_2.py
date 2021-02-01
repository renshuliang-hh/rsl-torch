from sklearn.datasets import load_boston

dataset = load_boston()

dir(dataset)

print(dataset['DESCR'])

dataset['data'].shape
import seaborn as sns

import pandas as pd

dataframe = pd.DataFrame(dataset['data'])

dataframe.columns = dataset['feature_names']
import random
import numpy as np
dataframe
import matplotlib.pyplot as plt
def sigmod(x):
    return 1/(1+np.exp(-x))
def random_linear(x):
    k,b=random.random(),random.random()
    return k*x+b
def model(x,k1,b1,k2,b2):
    linear1_out1=k1*x+b1
    sigmod_out=sigmod(linear1_out1)
    linear2_out=k2*sigmod_out+b2
    return linear2_out
def partial_b(x,y,k,b):
    return 2 * np.mean((y - (k * x + b))*(-1))
k,b=random.randint(-10,10),random.randint(-10,10)
# for t in range (total_times):
#     lr=1e-2
#
#     k1=k1+(-1)*partial_k(X_rm,Y,k,b)*lr
#     b1 = b1+ (-1) * partial_b(X_rm, Y, k, b)*lr
#     k1 = k1 + (-1) * partial_k(X_rm, Y, k, b) * lr
#     b1 = b1 + (-1) * partial_b(X_rm, Y, k, b) * lr
#
#     loss=Loss(Y,X_rm,k,b)
#
#     print(loss)
#     if loss<min_loss:
#         best_k,best_b=k,b
#         min_loss=loss
#         print("在{}时刻最好k：{}和b：{}，loss：{}".format(t,k,b,loss))
computing_graph={
    'k1':['l1'],
    'b1':['l1'],
    'x':['l1'],
    'k2':['l2'],
    'b2':['l2'],
    'l1':['sigmoid'],
    'sigmoid':['l2'],
    'l2':['loss'],
    'y':['loss']




}
import networkx as nx
aa=nx.draw(nx.DiGraph(computing_graph),with_labels=True)
plt.show()
def get_out(graph,node):
    out=[]
    for n,links in graph.items():
        if n==node:out+=links
    return out
get_out(computing_graph,'k1')



def get_paramter_partial_order(p):
    computing_order = []

    target = p
    computing_order.append(target)
    out = get_out(computing_graph, target)[0]
    while out:
        computing_order.append(out)
        out = get_out(computing_graph, out)
        if out: out = out[0]

    order = []
    for index, n in enumerate(computing_order[:-1]):
        order.append((computing_order[index + 1], n))

    return '*'.join(['{}/{}'.format(a, b) for a, b in order[::-1]])

get_paramter_partial_order('b1')
for p in ['b1','k1','b2','k2']:
    print(get_paramter_partial_order(p))

