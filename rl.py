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

from evaluation import get_known, update_fleet, get_capacity_by_server_generation_latency_sensitivity, get_utilization, get_normalized_lifespan, get_profit, evaluation_function
from constants import get_datacenters, get_selling_prices, get_servers, get_demand

# env.reset() – To reset the entire environment and obtain the initial values of observation.
# env.render() – Rendering the environment for displaying the visualization of the working setup.
# env.step() – To proceed with an action on the selected environment.
# env.close() – Close the particular render frame of the environment.
logger = logging.getLogger()
file_handler = logging.FileHandler('logs.log')
logger.addHandler(file_handler)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
file_handler.setFormatter(formatter)

# NN
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
        demand_ts = get_time_step_demand(self.demand, self.time_step)
        demand_array = demand_ts.values
        state = {
            'time_step': self.time_step,
            'fleet': self.fleet.to_dict('records') if not self.fleet.empty else [],
            'demand': demand_array.tolist()
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
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(128, activation='relu')
        self.dense3 = tf.keras.layers.Dense(action_size)

    def call(self, state):
        # Process time_step
        time_step = tf.cast(state['time_step'], tf.float32)
        
        # Process fleet
        fleet = state['fleet']
        if len(fleet) == 0:
            fleet_tensor = tf.zeros((1, 5))  # Assuming 5 features for each server
        else:
            fleet_tensor = tf.convert_to_tensor([[
                server['time_step'],
                self.encode_datacenter(server['datacenter_id']),
                self.encode_server_generation(server['server_generation']),
                self.encode_server_id(server['server_id']),
                self.encode_action(server['action'])
            ] for server in fleet], dtype=tf.float32)
        
        # Process demand
        demand = tf.convert_to_tensor(state['demand'], dtype=tf.float32)
        
        # Combine all inputs
        combined_input = tf.concat([
            tf.expand_dims(time_step, -1),
            tf.reshape(fleet_tensor, (-1,)),
            tf.reshape(demand, (-1,))
        ], axis=-1)
        
        x = self.dense1(combined_input)
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
                q_values = model(state)
                action = np.argmax(q_values[0])

            next_state, reward, done = env.step(action)
            total_reward += reward

            with tf.GradientTape() as tape:
                q_values = model(state)
                next_q_values = model(next_state)
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
        q_values = model(state)
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

def get_time_step_demand(demand, ts):
    # GET THE DEMAND AT A SPECIFIC TIME-STEP t
    return demand[demand['time_step'] == ts]

if __name__ == "__main__":
    datacenters = get_datacenters()
    servers = get_servers()
    selling_prices = get_selling_prices()
    demand = get_demand()

    print(datacenters)
    print("-----------------------------------------------------------------------------------------------")
    print(servers)
    print("-----------------------------------------------------------------------------------------------")
    print(selling_prices)
    print("-----------------------------------------------------------------------------------------------")
    print(demand)
