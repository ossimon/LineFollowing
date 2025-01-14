import math
import gym
import random 
import numpy as np
import itertools
from typing import List


class QLearner:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.q_dict = {}
        self.possible_actions = range(self.env.action_space.n)

        # Discretize observation space
        observation_shape = self.env.observation_space.shape
        self.buckets_sizes = config.get("buckets_sizes", [[0.33, 0.34, 0.33], [0.25, 0.5, 0.25], [0.5, 0.5]])  # Updated for speed
        self.epsilon = config.get("epsilon", 0.1)
        self._initialize_q_table()

    def _initialize_q_table(self):
        bucket_sizes = [len(bucket) for bucket in self.buckets_sizes]
        all_states = itertools.product(*[range(size) for size in bucket_sizes])
        for state in all_states:
            self.q_dict[state] = {a: random.uniform(0, 0.1) for a in self.possible_actions}

    def discretize(self, observation):
        return [map_to_buckets(value, -1.0, 1.0, self.buckets_sizes[i]) for i, value in enumerate(observation)]

    def pick_action(self, state, explore=True):
        if explore and random.random() < self.epsilon:
            return random.choice(self.possible_actions)  # Random action (exploration)
        return max(self.q_dict[tuple(state)], key=self.q_dict[tuple(state)].get)  # Best action (exploitation)

    def update_q_value(self, state, action, reward, next_state):
        max_future_q = max(self.q_dict[tuple(next_state)].values())
        current_q = self.q_dict[tuple(state)][action]
        alpha = self.config.get("alpha", 0.1)
        gamma = self.config.get("gamma", 0.99)
        self.q_dict[tuple(state)][action] = (1 - alpha) * current_q + alpha * (reward + gamma * max_future_q)

    def train(self, episodes, max_step=1000):
        for episode in range(episodes):
            print(f"[Notification]: Starting episode {episode + 1}.")
            current_step = 0
            state = self.discretize(self.env.reset())
            done = False
            total_reward = 0

            while not done:
                current_step += 1
                action = self.pick_action(state, explore=True)
                next_obs, reward, done, _ = self.env.step(action)
                next_state = self.discretize(next_obs)
                self.update_q_value(state, action, reward, next_state)
                state = next_state
                total_reward += reward

                if current_step > max_step:
                    done = True
                    print("[Notification]: Max_step per episode achived.")
            print(f"[Notification]: Episode {episode + 1} completed. Total reward: {total_reward}.")
        print("[Notification]: Training completed. Q-table:")
        print(self.q_dict)

    def test(self, episodes):
        total_rewards = []
        for episode in range(episodes):
            print(f"[Notification]: Starting test episode {episode + 1}.")
            state = self.discretize(self.env.reset())
            done = False
            total_reward = 0

            while not done:
                action = self.pick_action(state, explore=False)  # Always exploit during testing
                next_obs, reward, done, _ = self.env.step(action)
                next_state = self.discretize(next_obs)
                state = next_state
                total_reward += reward
            total_rewards.append(total_reward)
            print(f"[Notification]: Test episode {episode + 1} completed. Total reward: {total_reward}.")
        average_reward = np.mean(total_rewards)
        print(f"[Notification]: Average reward over {episodes} test episodes: {average_reward:.2f}.")
        return average_reward

# Usage Example


def map_to_buckets(value: float, min_value: float, max_value: float, bucket_sizes: List[float]) -> int:
    """Map a continuous value to a discrete bucket."""
    assert sum(bucket_sizes) == 1, "Bucket sizes must sum to 1."
    assert max_value > min_value, "Max value should be higher than min."
    
    total_range = max_value - min_value
    thresholds = []
    cumulative_size = 0
    for size in bucket_sizes:
        cumulative_size += size
        thresholds.append(min_value + cumulative_size * total_range)
    
    for i, threshold in enumerate(thresholds):
        if value <= threshold:
            return i
    
    return len(bucket_sizes) - 1
