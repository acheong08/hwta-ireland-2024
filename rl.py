import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import uuid
from evaluation import (
    get_capacity_by_server_generation_latency_sensitivity,
    get_normalized_lifespan,
    get_profit,
    get_time_step_demand, 
    get_utilization,
    get_known
)   
import json

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Set memory growth to avoid allocating all GPU memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPUs found and enabled: {gpus}")
    except RuntimeError as e:
        print(e)
        
class ServerFleetEnvironment:
    def __init__(self, datacenters_file, demand_file, selling_prices_file, servers_file):
        self.datacenters = pd.read_csv(datacenters_file)
        self.demand = pd.read_csv(demand_file)
        self.selling_prices = pd.read_csv(selling_prices_file)
        self.servers = pd.read_csv(servers_file)
        
        # Filter only GPU server types
        self.server_types = get_known("server_generation")
        self.server_types = [st for st in self.server_types if st.startswith('GPU')]
        
        self.datacenter_ids = get_known("datacenter_id")
        self.time_step = 1
        self.max_steps = self.demand['time_step'].max()
        
        self.fleet = self.initialize_fleet()
        self.actions = []

    def initialize_fleet(self):
        return {dc: {st: [] for st in self.server_types} for dc in self.datacenter_ids}

    def reset(self):
        self.time_step = 1
        self.fleet = self.initialize_fleet()
        self.actions = []
        return self._get_state()

    def step(self, action):
        self.update_fleet(action)
        reward = self.get_profit()
        state = self._get_state()
        self.time_step += 1
        done = self.time_step > self.max_steps
        return state, reward, done, {}

    def _get_state(self):
        current_demand = self._get_current_demand()
        current_fleet = self.get_time_step_fleet()
        datacenter_capacity = self.datacenters['slots_capacity'].values
        
        state = []
        for dc in self.datacenter_ids:
            for st in self.server_types:
                state.append(len(current_fleet[dc][st]))
        
        return np.array(state + list(current_demand) + list(datacenter_capacity))

    def _get_current_demand(self):
        demand_at_ts = self.demand[self.demand['time_step'] == self.time_step]
        if demand_at_ts.empty:
            return [0] * len(self.server_types)
        demand_row = demand_at_ts.iloc[0]
        return [demand_row.get(st, 0) for st in self.server_types]

    def get_time_step_fleet(self):
        return self.fleet

    def get_profit(self):
        total_revenue = 0
        total_cost = 0
        total_demand = sum(self._get_current_demand())
        
        # Calculate revenue and costs based on fleet and demand
        for dc in self.fleet:
            for st, servers in self.fleet[dc].items():
                # Use the correct column name 'selling_price'
                price = self.selling_prices.loc[self.selling_prices['server_generation'] == st, 'selling_price'].values[0]
                utilization = min(len(servers), total_demand) / len(servers) if servers else 0
                revenue = utilization * price
                cost = len(servers) * price * 0.1  # Example cost factor, adjust as needed
                
                total_revenue += revenue
                total_cost += cost
        
        profit = total_revenue - total_cost
        
        # Scale the profit to a reasonable range
        return profit / 100.0


    def update_fleet(self, action):
        dc_index, st_index = divmod(action, len(self.server_types))
        datacenter = self.datacenter_ids[dc_index]
        server_type = self.server_types[st_index]
        
        # Example condition to remove or replace servers based on demand or other criteria
        if len(self.fleet[datacenter][server_type]) > self.demand[self.demand['time_step'] == self.time_step].iloc[0].get(server_type, 0):
            # Remove a server if overcapacity
            if self.fleet[datacenter][server_type]:
                self.fleet[datacenter][server_type].pop(0)
        else:
            # Add a server if under capacity
            if len(self.fleet[datacenter][server_type]) < self.datacenters.loc[self.datacenters['datacenter_id'] == datacenter, 'slots_capacity'].values[0]:
                server_id = str(uuid.uuid4())
                self.fleet[datacenter][server_type].append(server_id)
                self.actions.append({
                    "time_step": self.time_step,
                    "datacenter_id": datacenter,
                    "server_generation": server_type,
                    "server_id": server_id,
                    "action": "buy"
                })


class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=0.0003):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = 0.99
        self.clip_ratio = 0.2
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        self.actor_optimizer = keras.optimizers.Adam(lr)
        self.critic_optimizer = keras.optimizers.Adam(lr)

    def _build_actor(self):
        inputs = keras.layers.Input(shape=(self.state_dim,))
        x = keras.layers.Dense(64, activation='relu')(inputs)
        x = keras.layers.Dense(64, activation='relu')(x)
        outputs = keras.layers.Dense(self.action_dim, activation='softmax')(x)
        return keras.Model(inputs, outputs)

    def _build_critic(self):
        inputs = keras.layers.Input(shape=(self.state_dim,))
        x = keras.layers.Dense(64, activation='relu')(inputs)
        x = keras.layers.Dense(64, activation='relu')(x)
        outputs = keras.layers.Dense(1)(x)
        return keras.Model(inputs, outputs)

    def get_action(self, state):
        probs = self.actor.predict(np.array([state]), verbose=0)[0]
        return np.random.choice(self.action_dim, p=probs)

    def update(self, states, actions, advantages, old_probs, returns):
        states = np.array(states)
        actions = np.array(actions)
        advantages = np.array(advantages)
        old_probs = np.array(old_probs)
        returns = np.array(returns)

        with tf.GradientTape() as tape:
            new_probs = self.actor(states, training=True)
            new_probs = tf.gather_nd(new_probs, tf.stack([tf.range(actions.shape[0]), actions], axis=1))
            ratio = new_probs / old_probs
            clip_ratio = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            surrogate = tf.minimum(ratio * advantages, clip_ratio * advantages)
            actor_loss = -tf.reduce_mean(surrogate)

        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        with tf.GradientTape() as tape:
            value_pred = self.critic(states, training=True)
            critic_loss = tf.reduce_mean(tf.square(returns - value_pred))

        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))


def train(env, agent, episodes=20, batch_size=32):
    for episode in range(episodes):
        state = env.reset()
        ep_rewards = []
        states, actions, rewards, values, probs = [], [], [], [], []

        done = False
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            
            print(f"Episode {episode}, Time step: {env.time_step}, Action: {action}, Reward: {reward}")  # Debugging output
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            values.append(agent.critic.predict(np.array([state]), verbose=0)[0][0])
            probs.append(agent.actor.predict(np.array([state]), verbose=0)[0][action])
            
            ep_rewards.append(reward)
            state = next_state

            if len(states) >= batch_size or done:
                returns = []
                advantages = []
                R = 0
                for r, v in zip(rewards[::-1], values[::-1]):
                    R = r + agent.gamma * R
                    returns.insert(0, R)
                    advantages.insert(0, R - v)
                
                agent.update(states, actions, advantages, probs, returns)
                states, actions, rewards, values, probs = [], [], [], [], []

        if episode % 10 == 0:
            print(f"Episode {episode}, Average Reward: {np.mean(ep_rewards)}")
            # Save intermediate actions
            with open(f'actions_{episode}.json', 'w') as f:
                json.dump(env.actions, f, indent=2)

    # Save final actions
    with open('actions.json', 'w') as f:
        json.dump(env.actions, f, indent=2)
    return env.actions


# Usage
env = ServerFleetEnvironment('data/datacenters.csv', 'data/demand.csv', 'data/selling_prices.csv', 'data/servers.csv')
state_dim = len(env._get_state())
action_dim = len(env.server_types) * len(env.datacenter_ids)
agent = PPOAgent(state_dim, action_dim)
actions = train(env, agent, episodes=1000)

# Save actions to a file
import json
with open('actions.json', 'w') as f:
    json.dump(actions, f, indent=2)