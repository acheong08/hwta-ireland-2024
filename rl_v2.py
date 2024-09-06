import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import uuid
import json
from collections import deque
import random
import glob

class TransformerBlock(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            keras.layers.Dense(ff_dim, activation="relu"),
            keras.layers.Dense(embed_dim),
        ])
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class ImprovedPPOAgent:
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
        
        self.policy_model = self.build_transformer_model()
        self.value_model = self.build_transformer_model(output_dim=1)
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.demonstration_buffer = []
        self.experience_buffer = deque(maxlen=100000)
        self.curiosity_model = self.build_curiosity_model()

    def build_transformer_model(self, output_dim=None):
        inputs = keras.Input(shape=(self.state_dim,))
        embedding = keras.layers.Dense(256)(inputs)
        transformer_block = TransformerBlock(256, 4, 512)
        x = transformer_block(embedding)
        x = keras.layers.GlobalAveragePooling1D()(x)
        x = keras.layers.Dense(128, activation="relu")(x)
        if output_dim is None:
            outputs = keras.layers.Dense(self.action_dim, activation="softmax")(x)
        else:
            outputs = keras.layers.Dense(output_dim)(x)
        return keras.Model(inputs=inputs, outputs=outputs)

    def build_curiosity_model(self):
        inputs = keras.Input(shape=(self.state_dim,))
        x = keras.layers.Dense(256, activation="relu")(inputs)
        x = keras.layers.Dense(256, activation="relu")(x)
        outputs = keras.layers.Dense(self.state_dim)(x)
        return keras.Model(inputs=inputs, outputs=outputs)

    def get_actions(self, state, epsilon=0.1, max_actions=5):
        state = np.array([state])
        action_probs = self.policy_model.predict(state, verbose=0)[0]
        
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

            curiosity_loss = self.compute_curiosity_loss(states)

            total_loss = policy_loss + 0.5 * value_loss - self.entropy_coeff * entropy_loss + 0.1 * curiosity_loss

        grads = tape.gradient(total_loss, self.policy_model.trainable_variables + self.value_model.trainable_variables + self.curiosity_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.policy_model.trainable_variables + self.value_model.trainable_variables + self.curiosity_model.trainable_variables))

    def compute_curiosity_loss(self, states):
        predicted_next_states = self.curiosity_model(states)
        actual_next_states = np.roll(states, -1, axis=0)
        return tf.reduce_mean(tf.square(predicted_next_states - actual_next_states))
    
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
                    action_list = self.get_actions(state, epsilon=self.get_exploration_rate(episode))
                    for action, action_prob in action_list:
                        action['time_step'] = time_step
                        episode_actions_at_timestep.append(action)
                        
                        states.append(state)
                        actions.append(action)
                        probs.append(action_prob)
                
                next_state, reward, done, _ = env.step(episode_actions_at_timestep)

                curiosity_reward = self.compute_curiosity_reward(state, next_state)
                combined_reward = reward + 0.1 * curiosity_reward

                rewards.extend([combined_reward] * len(episode_actions_at_timestep))
                values.extend([self.value_model.predict(np.array([state]), verbose=0)[0][0]] * len(episode_actions_at_timestep))

                ep_rewards.append(combined_reward)
                state = next_state
                time_step += 1

                episode_actions.extend(episode_actions_at_timestep)

                for s, a, r, ns in zip(states, actions, rewards, [next_state] * len(states)):
                    self.experience_buffer.append((s, a, r, ns, done))

                if len(states) >= batch_size or done:
                    self.update_from_experience(batch_size)
                    states, actions, rewards, values, probs = [], [], [], [], []

            episode_reward = sum(ep_rewards)
            print(f"Episode {episode + 1} - Total Reward: {episode_reward}")

            if episode_reward > best_reward:
                best_reward = episode_reward
                best_solution = episode_actions

        return best_solution

    def update_from_experience(self, batch_size):
        if len(self.experience_buffer) < batch_size:
            return

        batch = random.sample(self.experience_buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.array(states)
        next_states = np.array(next_states)
        rewards = np.array(rewards)
        dones = np.array(dones)

        values = self.value_model.predict(states, verbose=0)
        next_values = self.value_model.predict(next_states, verbose=0)

        advantages = rewards + self.gamma * next_values * (1 - dones) - values
        returns = advantages + values

        action_probs = self.policy_model.predict(states, verbose=0)
        action_indices = np.array([
            self.datacenter_ids.index(a['datacenter_id']) * len(self.server_types) + 
            self.server_types.index(a['server_generation']) 
            for a in actions
        ])
        old_probs = action_probs[np.arange(len(action_indices)), action_indices]

        self.update(states, actions, advantages, old_probs, returns)

    def get_exploration_rate(self, episode, min_rate=0.01, max_rate=1.0, decay=0.995):
        return max(min_rate, max_rate * (decay ** episode))

    def compute_curiosity_reward(self, state, next_state):
        predicted_next_state = self.curiosity_model.predict(np.array([state]), verbose=0)[0]
        return np.mean(np.square(predicted_next_state - next_state))
    
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import json
from rl import ServerFleetEnvironment

# Initialize the environment
datacenters_file = 'data/datacenters.csv'
demand_file = 'data/demand.csv'
selling_prices_file = 'data/selling_prices.csv'
servers_file = 'data/servers.csv'

env = ServerFleetEnvironment(datacenters_file, demand_file, selling_prices_file, servers_file)

# Initialize the improved agent
agent = ImprovedPPOAgent(
    state_dim=env._get_state().shape[0],
    datacenter_ids=env.datacenter_ids,
    server_types=env.server_types
)

# Load demonstrations
print("Loading demonstrations...")
agent.load_demonstrations("henrik_input_training_solution_*.json")

# Train the agent
print("Starting training...")
num_episodes = 100
batch_size = 64
pretrain = True

final_solution = agent.train(env, episodes=num_episodes, batch_size=batch_size, pretrain=pretrain)

# Save the final solution to a JSON file
with open('final_solution_improved.json', 'w') as f:
    json.dump(final_solution, f, indent=2)

print("Training completed. Final solution saved to 'final_solution_improved.json'.")

# Evaluation
print("Evaluating final solution...")
state = env.reset()
total_reward = 0
done = False

for action in final_solution:
    next_state, reward, done, _ = env.step([action])
    total_reward += reward
    state = next_state
    if done:
        break

print(f"Final solution total reward: {total_reward}")

# Optional: Plot learning curve
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(agent.episode_rewards)
plt.title('Learning Curve')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.savefig('learning_curve.png')
plt.close()

print("Learning curve saved to 'learning_curve.png'.")