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



