import numpy as np
import pandas as pd
import os
import evaluation
from constants import get_datacenters, get_servers, get_selling_prices
import utils


b = np.random.seed(50)
a = np.array([100 for _ in range(21)])

# check exampleVector.pdf for the encoding

# util function
def decode_selling_prices():
    temp = get_selling_prices()
    val = []
    for i in range(len(temp)):
        val.append(temp[i].selling_price)
    return val

def calculate_partial_derivative(Z_list, D_list, p_list, iter):
    z = Z_list[iter]
    d = D_list[iter]

    little_derivative = np.exp(-z) / (np.exp(-z) + np.exp(-d))
    x_z = -np.log(np.exp(-z)+np.exp(-d))
    x_z_squared = x_z**2

    x_encoding = [(-np.log(np.exp(-z_i)+np.exp(-d_i))) for z_i, d_i in zip(Z_list, D_list)]

    der_1 = 1/z * (little_derivative) * (sum(np.array(x_encoding) * np.array(p_list)) + x_z*p_list[iter]) 
    der_2 = - (x_z) / x_z_squared * (sum(np.array(x_encoding) * np.array(p_list)) - x_z*p_list[iter] + x_z_squared*p_list[iter])
    der3 = p_list[iter] * little_derivative * (sum(np.array(x_encoding) / np.array(Z_list)) - x_z/Z_list[iter])

    return der_1 + der_2 + der3

def total_derivative(Z_list, D_list, p_list):
    total_derivative = []

    for i in Z_list:
        total_derivative.append(calculate_partial_derivative(Z_list[i], D_list[i], Z_list, D_list, p_list, i))
    total_derivative = np.array(total_derivative)

    return total_derivative

def improved_gradient_descent(Z_list, D_list, p_list):
    total_derivative = total_derivative(Z_list, D_list, p_list)
    scores = [Z_list]
    Z_list = Z_list - total_derivative
    scores.append(Z_list)
    for i in range(20):
        alpha = 1
        Z_list = Z_list - total_derivative
        scores.append(Z_list)
        if (np.fabs(scores[i+2] - scores[i+1]) > np.fabs(scores[i+1] - scores[i])):
            alpha = alpha * 0.8
        else:
            alpha = alpha * 1.25
    return Z_list

def main():
    p_list = decode_selling_prices()
    print(p_list)

if __name__ == "__main__":
    main()

