import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import uuid
import json
from evaluation import (
    get_capacity_by_server_generation_latency_sensitivity,
    get_normalized_lifespan,
    get_profit,
    get_time_step_demand, 
    get_utilization,
    get_known
)
import glob

class ServerFleetEnvironment:
    def __init__(self, datacenters_file, demand_file, selling_prices_file, servers_file):
        self.datacenters = pd.read_csv(datacenters_file)
        self.demand = pd.read_csv(demand_file)
        self.selling_prices = pd.read_csv(selling_prices_file)
        self.servers = pd.read_csv(servers_file)
        
        self.server_types = get_known("server_generation")
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

    def step(self, actions):
        for action in actions:
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
        
        state += list(current_demand)
        state += list(datacenter_capacity)
        
        return np.array(state)

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
        
        for dc in self.fleet:
            for st, servers in self.fleet[dc].items():
                price = self.selling_prices.loc[self.selling_prices['server_generation'] == st, 'selling_price'].values[0]
                utilization = min(len(servers), total_demand) / len(servers) if servers else 0
                revenue = utilization * price
                cost = len(servers) * price * 0.1
                
                total_revenue += revenue
                total_cost += cost
        
        profit = total_revenue - total_cost
        
        if profit < 0:
            profit = -1
        else:
            profit = profit / 100.0
        
        return profit

    def update_fleet(self, action):
        datacenter_id = action['datacenter_id']
        server_generation = action['server_generation']
        action_type = action['action']
        server_id = action.get('server_id', None)
        
        max_capacity = self.datacenters.loc[self.datacenters['datacenter_id'] == datacenter_id, 'slots_capacity'].values[0]

        if action_type == 'buy':
            if server_id is not None and len(self.fleet[datacenter_id][server_generation]) < max_capacity:
                self.fleet[datacenter_id][server_generation].append(server_id)
                self.actions.append(action)
        elif action_type == 'move':
            source_dc = action.get('source_datacenter_id')
            if source_dc and server_id in self.fleet[source_dc][server_generation]:
                if len(self.fleet[datacenter_id][server_generation]) < max_capacity:
                    self.fleet[source_dc][server_generation].remove(server_id)
                    self.fleet[datacenter_id][server_generation].append(server_id)
                    self.actions.append(action)
        elif action_type == 'dismiss':
            if server_id in self.fleet[datacenter_id][server_generation]:
                self.fleet[datacenter_id][server_generation].remove(server_id)
                self.actions.append(action)
        elif action_type == 'hold':
            pass
        else:
            raise ValueError(f"Unknown action type: {action_type}")


class PPOAgent:
    def __init__(self, state_dim, datacenter_ids, server_types, learning_rate=0.0003, gamma=0.99, lambda_=0.95, clip_epsilon=0.2, entropy_coeff=0.01):
        self.state_dim = state_dim
        self.datacenter_ids = datacenter_ids
        self.server_types = server_types
        self.action_dim = len(datacenter_ids) * len(server_types)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lambda_ = lambda_
        self.clip_epsilon = clip_epsilon
        self.entropy_coeff = entropy_coeff
        
        self.policy_model = self.build_policy_model()
        self.value_model = self.build_value_model()
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.demonstration_buffer = []
        self.pretrained_policy = None

    def build_policy_model(self):
        inputs = tf.keras.Input(shape=(self.state_dim,))
        x = tf.keras.layers.Dense(256, activation='relu')(inputs)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        outputs = tf.keras.layers.Dense(self.action_dim, activation='softmax')(x)
        return tf.keras.Model(inputs=inputs, outputs=outputs)
    
    def build_value_model(self):
        inputs = tf.keras.Input(shape=(self.state_dim,))
        x = tf.keras.layers.Dense(256, activation='relu')(inputs)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        outputs = tf.keras.layers.Dense(1)(x)
        return tf.keras.Model(inputs=inputs, outputs=outputs)

    def get_actions(self, state, epsilon=0.1, max_actions=5):
        state = np.array([state])
        action_probs = self.policy_model.predict(state, verbose=0)[0]
        
        if self.pretrained_policy is not None:
            pretrained_probs = self.pretrained_policy.predict(state, verbose=0)[0]
            action_probs = 0.5 * action_probs + 0.5 * pretrained_probs  # Combine learned and pretrained policies
        
        actions = []
        for _ in range(max_actions):
            if np.random.random() < epsilon:
                action_index = np.random.choice(self.action_dim)
            else:
                action_index = np.random.choice(self.action_dim, p=action_probs)
            
            dc_index = action_index // len(self.server_types)
            st_index = action_index % len(self.server_types)
            
            action = {
                'datacenter_id': self.datacenter_ids[dc_index],
                'server_generation': self.server_types[st_index],
                'server_id': str(uuid.uuid4()),
                'action': 'buy'
            }
            
            actions.append((action, action_probs[action_index]))
        
        return actions

    def update(self, states, actions, advantages, old_probs, returns):
        states = np.array(states)
        action_indices = np.array([
            self.datacenter_ids.index(a['datacenter_id']) * len(self.server_types) + 
            self.server_types.index(a['server_generation']) 
            for a in actions
        ])
        advantages = np.array(advantages)
        old_probs = np.array(old_probs)
        returns = np.array(returns)

        with tf.GradientTape() as tape:
            new_probs = self.policy_model(states, training=True)
            
            indices = tf.range(tf.shape(new_probs)[0]) * tf.shape(new_probs)[1] + action_indices
            prob = tf.gather(tf.reshape(new_probs, [-1]), indices)

            ratio = prob / (old_probs + 1e-10)
            clip_loss = tf.minimum(ratio * advantages, tf.clip_by_value(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages)
            policy_loss = -tf.reduce_mean(clip_loss)

            value = tf.squeeze(self.value_model(states, training=True))
            value_loss = tf.reduce_mean(tf.square(returns - value))

            entropy = -tf.reduce_sum(new_probs * tf.math.log(new_probs + 1e-10), axis=-1)
            entropy_loss = tf.reduce_mean(entropy)

            total_loss = policy_loss + 0.5 * value_loss - self.entropy_coeff * entropy_loss

        grads = tape.gradient(total_loss, self.policy_model.trainable_variables + self.value_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.policy_model.trainable_variables + self.value_model.trainable_variables))

    def load_demonstrations(self, file_pattern):
        self.demonstration_buffer = []
        files_loaded = 0
        for filename in glob.glob(file_pattern):
            try:
                with open(filename, 'r') as f:
                    demonstration = json.load(f)
                    self.demonstration_buffer.extend(demonstration)
                    files_loaded += 1
                    print(f"Loaded {len(demonstration)} demonstrations from {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")
        
        print(f"Total files loaded: {files_loaded}")
        print(f"Total demonstrations loaded: {len(self.demonstration_buffer)}")
        
        if not self.demonstration_buffer:
            print("Warning: No demonstrations were loaded. Check the file pattern and file contents.")

    def load_single_demonstration(self, filename):
        self.demonstration_buffer = []
        try:
            with open(filename, 'r') as f:
                demonstration = json.load(f)
                self.demonstration_buffer.extend(demonstration)
                print(f"Loaded {len(demonstration)} demonstrations from {filename}")
        except Exception as e:
            print(f"Error loading {filename}: {str(e)}")
        
        if not self.demonstration_buffer:
            print("Warning: No demonstrations were loaded. Check the file contents.")


    def preprocess_demonstration(self, env):
        processed_demonstrations = []
        state = env.reset()
        for action in self.demonstration_buffer:
            next_state, reward, done, _ = env.step([action])
            processed_demonstrations.append((state, action, reward))
            state = next_state
            if done:
                state = env.reset()
        return processed_demonstrations

    def pretrain_from_demonstrations(self, env, epochs=10, batch_size=32):
        demonstrations = self.preprocess_demonstration(env)
        for epoch in range(epochs):
            np.random.shuffle(demonstrations)
            for i in range(0, len(demonstrations), batch_size):
                batch = demonstrations[i:i+batch_size]
                states, actions, rewards = zip(*batch)
                
                states = np.array(states)
                action_indices = np.array([
                    self.datacenter_ids.index(a['datacenter_id']) * len(self.server_types) + 
                    self.server_types.index(a['server_generation']) 
                    for a in actions
                ])
                
                with tf.GradientTape() as tape:
                    action_probs = self.policy_model(states, training=True)
                    selected_action_probs = tf.gather(
                        tf.reshape(action_probs, [-1]),
                        tf.range(tf.shape(action_probs)[0]) * tf.shape(action_probs)[1] + action_indices
                    )
                    loss = -tf.reduce_mean(tf.math.log(selected_action_probs + 1e-10))
                
                grads = tape.gradient(loss, self.policy_model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.policy_model.trainable_variables))
            
            print(f"Pretrain Epoch {epoch + 1}/{epochs} completed")
        
        self.pretrained_policy = tf.keras.models.clone_model(self.policy_model)
        self.pretrained_policy.set_weights(self.policy_model.get_weights())

        # Reinitialize the optimizer to ensure it matches the cloned model's variables
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)


    def train(self, env, episodes=50, batch_size=32, pretrain=True):
        if pretrain:
            if not self.demonstration_buffer:
                print("Error: No demonstrations loaded. Skipping pretraining.")
            else:
                print(f"Pretraining on {len(self.demonstration_buffer)} demonstrations")
                self.pretrain_from_demonstrations(env)
        
        best_reward = float('-inf')
        best_solution = None

        for episode in range(episodes):
            state = env.reset()
            ep_rewards = []
            states, actions, rewards, values, probs = [], [], [], [], []
            episode_actions = []

            done = False
            time_step = 1
            while not done:
                episode_actions_at_timestep = []
                for _ in range(5):  # Allow up to 5 actions per time step
                    action_list = self.get_actions(state, epsilon=max(0.1, 1.0 - episode / episodes))
                    for action, action_prob in action_list:
                        action['time_step'] = time_step
                        episode_actions_at_timestep.append(action)
                        
                        states.append(state)
                        actions.append(action)
                        probs.append(action_prob)
                
                next_state, reward, done, _ = env.step(episode_actions_at_timestep)

                rewards.extend([reward] * len(episode_actions_at_timestep))
                values.extend([self.value_model.predict(np.array([state]), verbose=0)[0][0]] * len(episode_actions_at_timestep))

                ep_rewards.append(reward)
                state = next_state
                time_step += 1

                episode_actions.extend(episode_actions_at_timestep)

                if len(states) >= batch_size or done:
                    returns = []
                    advantages = []
                    R = 0
                    for r, v in zip(rewards[::-1], values[::-1]):
                        R = r + self.gamma * R
                        returns.insert(0, R)
                        advantages.insert(0, R - v)

                    self.update(states, actions, advantages, probs, returns)
                    states, actions, rewards, values, probs = [], [], [], [], []

            episode_reward = sum(ep_rewards)
            print(f"Episode {episode + 1} - Total Reward: {episode_reward}")

            # Save the solution for each episode
            episode_solution_filename = f'solution_episode_{episode + 1}.json'
            with open(episode_solution_filename, 'w') as f:
                json.dump(episode_actions, f, indent=2)
            print(f"Solution for episode {episode + 1} saved to '{episode_solution_filename}'")

            if episode_reward > best_reward:
                best_reward = episode_reward
                best_solution = episode_actions

        return best_solution

# Initialize the environment
datacenters_file = 'data/datacenters.csv'
demand_file = 'data/demand.csv'
selling_prices_file = 'data/selling_prices.csv'
servers_file = 'data/servers.csv'

env = ServerFleetEnvironment(datacenters_file, demand_file, selling_prices_file, servers_file)

# Initialize the agent
agent = PPOAgent(
    state_dim=env._get_state().shape[0],
    datacenter_ids=env.datacenter_ids,
    server_types=env.server_types
)

# Load a single demonstration file
print("Loading a single demonstration...")
agent.load_single_demonstration("henrik_input_training_solution_1.json")  # Specify your desired file

# Train the agent (including pretraining)
print("Starting training...")
final_solution = agent.train(env, episodes=50, pretrain=True)

'''
# Load demonstrations
print("Loading demonstrations...")
agent.load_demonstrations("henrik_input_training_solution_*.json")

# Train the agent (including pretraining)
print("Starting training...")
final_solution = agent.train(env, episodes=50, pretrain=True)'''

# Save the final solution to a JSON file
with open('final_solution_improved.json', 'w') as f:
    json.dump(final_solution, f, indent=2)

print("Training completed. Final solution saved to 'final_solution_improved.json'.")