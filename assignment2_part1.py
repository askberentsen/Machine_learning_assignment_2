# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 16:54:50 2020

@author: Ask
"""
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

def train(hidden_layers):
    classifier = MLPClassifier(
        solver ="lbfgs", 
        activation="tanh", 
        hidden_layer_sizes=hidden_layers, 
        max_iter=100, #400 
        tol=1e-4, #1e-5
        verbose=True
        )
    clf = classifier.fit(X_train, y_train)
    return clf.score(X_test, y_test)

def simulate(layers_range, nodes_range, samples=5):
    _scores = []
    for layers in layers_range:
        for nodes in nodes_range:
            score = 0
            print(f'training {layers}*{nodes} with an average of {samples} samples')
            for j in range(samples):
                score += train( [nodes]*layers )
            _scores += [(layers, nodes, score/samples)]
    return _scores

def plot_3d(xs, ys, zs, z_func, z_label):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel('layers')
    ax.set_ylabel('nodes')
    ax.set_zlabel(z_label)
    plt.tight_layout()
    
    adjusted_zs = [z_func(z) for z in zs]
    ax.plot_trisurf(xs, ys, adjusted_zs, cmap='rainbow_r')

def plot_2d(xs, ys, y_func, y_label):
    fig, ax = plt.subplots()
    ax.set_xlabel('nodes')
    ax.set_ylabel(y_label)
    plt.tight_layout()
    adjusted_ys = [y_func(y) for y in ys]
    ax.plot(xs, adjusted_ys)

X, y = datasets.load_digits(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#warm = input("Use previous values? y/n: ")=="y"
#if not warm:
layers_min = int(input("How many layers to start with: "))
layers_max = int(input("How many layers to end with: "))+1
layers_inc = int(input("How many layers to increment with: "))
layers_range = range(layers_min, layers_max, layers_inc)

nodes_min = int(input("How many nodes to start with: "))
nodes_max = int(input("How many nodes to end with: "))+1
nodes_inc = int(input("How many nodes to increment with: "))
nodes_range = range(nodes_min, nodes_max, nodes_inc)

samples = int(input("How many samples per parameter set: "))
        
scores = simulate(layers_range, nodes_range, samples)

parameters_sorted = sorted(scores, key=(lambda e:e[2]), reverse=True)
p = list(zip(*scores))
p_adjusted = list(zip(*[(lay, nod, 1/(1-acc)) for lay, nod, acc in scores]))
height_adjust = lambda z: 1/(1-z)

print(f"best parameter: {parameters_sorted[0]}\nRunner ups:")
print(parameters_sorted[1:16])

if len(layers_range)>1:
    plot_3d(*p, lambda z:z, "accuracy")
    plot_3d(*p, height_adjust, "1/(1-accuracy)")
else:
    plot_2d(*(p[1:]), lambda y:y, "accuracy")
    plot_2d(*(p[1:]), height_adjust, "1/(1-accuracy)")

