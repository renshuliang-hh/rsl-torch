
import random
import numpy as np
from functools import  reduce
def toplogic(graph):
    graph=graph.copy()

    '''
    graph={
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
    
    '''
    sorted_node=[]
    while graph:
        nodes_have_inputs= reduce(lambda a,b:a+b, list(graph.values()))
        nodes_have_outs =list(graph.keys())
        nodes_have_outs_no_inputs=set(nodes_have_outs)-set(nodes_have_inputs)
        if nodes_have_outs_no_inputs:
            node=random.choice(list(nodes_have_outs_no_inputs))


            sorted_node.append(node)
            if len(graph) == 1:
                sorted_node += graph[node]
            graph.pop(node)


            for _,links in graph.items():
                if node in links:graph.remove(node)
        else:
            raise TypeError('this graph has circle, which can not get toplogic order')
    return sorted_node
k1,b1,x,k2,b2,l1,sigmoid,l2,y,loss='k1','b1','x','k2','b2','l1','sigmoid','l2','y','loss'
test1_graph={
    k1:[l1],
    b1:[l1],
    x:[l1],
    k2:[l2],
    b2:[l2],
    l1:[sigmoid],
    sigmoid:[l2],
    l2:[loss],
   y:[loss]




}
test_graph={
    k1:[l1],
    b1:[l1],
    x:[l1],
    l1:[sigmoid],
    sigmoid:[loss],
   y:[loss]




}
aa=toplogic(test_graph)

class Node:
    def __init__(self,inputs=[],name=None,is_train=False):
        self.inputs=inputs
        self.outputs=[]
        self.name=name
        self.value=None
        self.gradients={}
        self.is_train=is_train



        for node in inputs:
            node.outputs.append(self)
    def forword(self):
        pass

        # print('I am {}, I calculate myself value{} by my self'.format(self.name,self.value))
    def __repr__(self):
        return  'Node:{}'.format(self.name)
    def backword(self):
       pass





class Placeholder(Node):
    def __init__(self,name,is_train=False):
        Node.__init__(self,name=name,is_train=is_train)

    def forward(self, value=None):
        if value is not None: self.value = value
    def backword(self):
        self.gradients[self]=self.outputs[0].gradients[self]
        # print('I got myself gradients: {}'.format(self.outputs[0].gradients[self]))
    def __repr__(self):
        return  'Placeholder:{}'.format(self.name)


class Linear(Node):
    def __init__(self,x,k,b,name=None):
        Node.__init__(self,inputs=[x,k,b],name=name)

    def forword(self):

        x,k,b=self.inputs[0],self.inputs[1],self.inputs[2]
        self.value = k.value * x.value + b.value
        # print('I am {}, I calculate myself value {} by my self'.format(self.name,self.value))
    def backword(self):
        x, k, b = self.inputs[0], self.inputs[1], self.inputs[2]
        self.gradients[self.inputs[0]] = self.outputs[0].gradients[self]*x.value
        self.gradients[self.inputs[1]] = self.outputs[0].gradients[self]*k.value
        self.gradients[self.inputs[2]] = self.outputs[0].gradients[self]*b.value

        # print('self.gradients[self.inputs[0]] {}'.format(self.gradients[self.inputs[0]]))
        # print('self.gradients[self.inputs[1]] {}'.format(self.gradients[self.inputs[1]]))
        # print('self.gradients[self.inputs[0]] {}'.format(self.gradients[self.inputs[2]]))



    def __repr__(self):
        return  'Linear:{}'.format(self.name)


class Sigmoid(Node):
    def __init__(self,l,name=None):
        Node.__init__(self,inputs=[l],name=name)
    def _sigmoid(self,l):
        return 1/(1+np.exp(-l))

    def forword(self):

        l=self.inputs[0]
        self.value=self._sigmoid(l.value)
        # print('I am {}, I calculate myself value {} by my self'.format(self.name,self.value))
    def backword(self):
        l = self.inputs[0]
        self.gradients[self.inputs[0]] =self.outputs[0].gradients[self]*self._sigmoid(l.value)*(1-self._sigmoid(l.value))



        # print('self.gradients[self.inputs[0]] {}'.format(self.gradients[self.inputs[0]]))



    def __repr__(self):
        return  'Sigmoid:{}'.format(self.name)

class MSE(Node):
    def __init__(self,y,yhat,name=None):
        Node.__init__(self,inputs=[y,yhat],name=name)


    def forword(self):

        y,yhat=self.inputs[0],self.inputs[1]
        self.value=np.mean((y.value-yhat.value)**2)
        # print('I am {}, I calculate myself value{} by my self'.format(self.name,self.value))
    def backword(self):
        y= self.inputs[0]
        yhat =self.inputs[1]
        self.gradients[self.inputs[0]]=2*np.mean(y.value-yhat.value)
        self.gradients[self.inputs[1]] = -2*np.mean(y.value-yhat.value)
        # print('self.gradients[self.inputs[0]] {}'.format(self.gradients[self.inputs[0]]))
        # print('self.gradients[self.inputs[1]] {}'.format(self.gradients[self.inputs[1]]))



    def __repr__(self):
        return  'MSE:{}'.format(self.name)
# node_x=Node(name="x")
# node_y=Node(name="y")
# node_b1=Node(name="b1")
# node_k1=Node(name="k1")
# node_l1=Node(inputs=[node_x,node_b1,node_k1],name="l1")
# node_sigmoid=Node(inputs=[node_l1],name="sigmoid")
# node_loss=Node(inputs=[node_sigmoid,node_y],name="loss")
node_x=Placeholder(name="x")
node_y=Placeholder(name="y",)
node_b1=Placeholder(name="b1",is_train=True)
node_k1=Placeholder(name="k1",is_train=True)
node_l1=Linear(x=node_x,b=node_b1,k=node_k1,name="l1")
node_sigmoid=Sigmoid(l=node_l1,name="sigmoid")
node_loss=MSE(yhat=node_sigmoid,y=node_y,name="mse")
# need_expand=[node_x,node_y,node_k1,node_b1]
feed_dict={
    node_x:3,
        node_y:random.random(),
node_k1:random.random(),

node_b1:0.38


}
from collections import  defaultdict
def convert_feed_dict_to_graph(feed_dict):
    need_expand = [n for n in feed_dict]
    computing_graph = defaultdict(list)

    while need_expand:
        n = need_expand.pop(0)
        if n in computing_graph: continue
        if isinstance(n,Placeholder):n.value=feed_dict[n]
        for m in n.outputs:
            computing_graph[n].append(m)
            need_expand.append(m)
    return computing_graph


sorted_nodes=toplogic(convert_feed_dict_to_graph(feed_dict))
def forward(graph_sorted_nodes):
    for node in graph_sorted_nodes:
        # print('I am {}'.format(node.name))
        node.forword()
        if node.name=='mse':
            print(node.value)
def backward(graph_sorted_nodes):
    for node in graph_sorted_nodes[::-1]:

            # print('I am {}'.format(node.name))
        node.backword()

def optimze(graph_sorted_nodes,learn_rate=1e-1):
    for node in graph_sorted_nodes:

        if node.is_train == True:
            node.value = node.value + -1 * node.gradients[node] * learn_rate
            # cmp = 'large' if node.gradients[node] > 0 else 'smal'
            # print('{} value is too {},{}'.format(node.name, cmp, node.value))


def run_one_epoch(graph_sorted_nodes):
    forward(graph_sorted_nodes)
    backward(graph_sorted_nodes)
loss_history=[]
for i in range(10):
    run_one_epoch(sorted_nodes)
    _loss_node=sorted_nodes[-1]
    assert isinstance(_loss_node,MSE)
    loss_history.append(_loss_node.value)

    optimze(sorted_nodes)
import matplotlib.pyplot as plt
plt.plot(loss_history)
plt.show()


def sigmoid(xx):
    # return 1
    return 1/(1+np.exp(-xx))

aaa=sorted_nodes[2].value*sorted_nodes[3].value+sorted_nodes[0].value
yhat=sigmoid(aaa)
y=sorted_nodes[1].value
