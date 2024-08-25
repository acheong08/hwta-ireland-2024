import numpy as np
import pandas as pd
import os
import evaluation
from constants import get_datacenters, get_servers, get_selling_prices
import utils


b = np.random.seed(50)
a = np.array([100 for _ in range(21)])

def calculate_partial_derivative(z, d, Z_list, D_list, p_list, iter):
    little_derivative = np.exp(-z) / (np.exp(-z) + np.exp(-d))
    x_encoding = [np.log(np.exp(-z)+np.exp(-d)) for z, d in zip(Z_list, D_list)]
    total_derivative = sum(x_encoding)*little_derivative #not the correct function as of yet
    return total_derivative

def total_derivative(Z_list, D_list, p_list, iter):
    total_derivative = []
    for (i,j) in zip(Z_list, D_list):
        total_derivative.append(calculate_partial_derivative(i, j, Z_list, D_list, p_list, iter))
    total_derivative = np.array(total_derivative)
    return total_derivative

def gradient_descent(Z_list, D_list, p_list, iter):
    total_derivative = total_derivative(Z_list, D_list, p_list, iter)
    for i in range(10):
        Z_list = Z_list - total_derivative
    return Z_list

