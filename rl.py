import gym
import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf
import random
from collections import deque
import json
import pandas as pd
from scipy.stats import truncweibull_min
import logging 

from constants import get_datacenters, get_selling_prices, get_servers, get_demand
from evaluation import * 

#env.reset() – To reset the entire environment and obtain the initial values of observation.
#env.render() – Rendering the environment for displaying the visualization of the working setup.
#env.step() – To proceed with an action on the selected environment.
#env.close() – Close the particular render frame of the environment.
logger = logging.getLogger()
file_handler = logging.FileHandler('logs.log')
logger.addHandler(file_handler)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
file_handler.setFormatter(formatter)

#NN
class ServerFleetEnvironment:
    def __init__(self, datacenters, servers, selling_prices, demand):
        self.datacenters = datacenters
        self.servers = servers
        self.selling_prices = selling_prices
        self.demand = demand
        self.time_step = 1
        self.fleet = pd.DataFrame()

    def reset(self):
        self.time_step = 1
        self.fleet = pd.DataFrame()
        return self.get_state()

    def step(self, action):
        self.apply_action(action)
        next_state = self.get_state()
        reward = self.calculate_reward()
        done = self.time_step >= get_known('time_steps')
        self.time_step += 1
        return next_state, reward, done

    def get_state(self):
        state = {
            'time_step': self.time_step,
            'fleet': self.fleet,
            'demand': get_time_step_demand(self.demand, self.time_step)
        }
        return state

    def apply_action(self, action):
        self.fleet = update_fleet(self.time_step, self.fleet, action)

    def calculate_reward(self):
        D = get_time_step_demand(self.demand, self.time_step)
        Z = get_capacity_by_server_generation_latency_sensitivity(self.fleet)
        U = get_utilization(D, Z)
        L = get_normalized_lifespan(self.fleet)
        P = get_profit(D, Z, self.selling_prices, self.fleet)
        return U * L * P

class DQN(tf.keras.Model):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.dense3 = tf.keras.layers.Dense(action_size)

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        return self.dense3(x)

def train_model(env, model, episodes, epsilon=0.1):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = tf.keras.losses.MeanSquaredError()

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            if np.random.random() < epsilon:
                action = np.random.randint(0, model.output.shape[1])
            else:
                q_values = model(np.array([state]))
                action = np.argmax(q_values[0])

            next_state, reward, done = env.step(action)
            total_reward += reward

            with tf.GradientTape() as tape:
                q_values = model(np.array([state]))
                next_q_values = model(np.array([next_state]))
                target = reward + 0.99 * np.max(next_q_values[0])
                loss = loss_fn(target, q_values[0][action])

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            state = next_state

        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

    return model

def evaluate_and_generate_solution(model, env, seed):
    np.random.seed(seed)
    state = env.reset()
    solution = []

    for t in range(get_known('time_steps')):
        q_values = model(np.array([state]))
        action = np.argmax(q_values[0])
        
        action_dict = {
            'time_step': t + 1,
            'datacenter_id': get_known('datacenter_id')[action % len(get_known('datacenter_id'))],
            'server_generation': get_known('server_generation')[action // len(get_known('datacenter_id'))],
            'server_id': f"server_{t}_{action}",
            'action': 'buy'
        }
        solution.append(action_dict)
        
        next_state, _, done = env.step(action)
        state = next_state
        if done:
            break

    solution_df = pd.DataFrame(solution)
    return solution_df

if __name__ == "__main__":
    datacenters = get_datacenters()
    servers = get_servers()
    selling_prices = get_selling_prices()
    demand = get_demand()

    env = ServerFleetEnvironment(datacenters, servers, selling_prices, demand)

    state_size = 168
    action_size = len(get_known('datacenter_id')) * len(get_known('server_generation'))

    model = DQN(state_size, action_size)
    trained_model = train_model(env, model, episodes=1000)

    for seed in range(168):
        solution = evaluate_and_generate_solution(trained_model, env, seed)
        
        score = evaluation_function(solution, demand, datacenters, servers, selling_prices, seed=seed)
        print(f"Seed {seed}, Score: {score}")

        solution.to_json(f"{seed}.json", orient='records')