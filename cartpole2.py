import random
from collections import deque

import gym
import tensorflow as tf
from keras.layers import Dense
from tensorflow import keras
from keras import Sequential
import numpy as np

class DqnAgent:
    """
    DQN Agent: the agent that explores the game and
    should eventually learn how to play the game.
    """

    def __init__(self):
        self.q_net = self._build_dqn_model()
        self.target_q_net = self._build_dqn_model()
        self.gamma = 0.95
        self.epsilon = 0.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

    def save_model(self, name):
        self.q_net.save(name)

    def load_model(self, name):
        self.q_net = tf.keras.models.load_model(name)

    def policy(self, state):
        state_input = tf.convert_to_tensor(state[None, :], dtype=tf.float32)
        action_q = self.q_net(state_input)
        action = np.argmax(action_q.numpy()[0], axis=0)
        return action

    def training_policy(self, state):
        """
        Takes a state from the game environment and returns
        a action that should be taken given the current game
        environment.

        This is an epsilon greedy policy.
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(0, 2)

        state_input = tf.convert_to_tensor(state[None, :], dtype=tf.float32)
        action_q = self.q_net(state_input)
        action = np.argmax(action_q.numpy()[0], axis=0)
        return action

    def train(self, batch):
        """
        Takes a batch of gameplay experiences from replay
        buffer and train the underlying model with the batch
        """
        state_batch, next_state_batch, action_batch, reward_batch, done_batch = batch
        current_q = self.q_net(state_batch)
        target_q = np.copy(current_q)
        next_q = self.target_q_net(next_state_batch)
        max_next_q = np.amax(next_q, axis=1)
        for i in range(state_batch.shape[0]):  # state_batch.shape[0] == batch_size
            target_q_val = reward_batch[i]
            if not done_batch[i]:
                target_q_val += self.gamma * max_next_q[i]
            target_q[i][action_batch[i]] = target_q_val
        training_history = self.q_net.fit(x=state_batch, y=target_q, verbose=0)
        loss = training_history.history['loss']
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss

    def update_target_network(self):
        """
        Updates the current target_q_net with the q_net which brings all the
        training in the q_net to the target_q_net.
        :return: None
        """
        self.target_q_net.set_weights(self.q_net.get_weights())

    @staticmethod
    def _build_dqn_model():
        """
        Builds a deep neural net which predicts the Q values for all possible
        actions given a state. The input should have the shape of the state
        (which is 4 in CartPole), and the output should have the same shape as
        the action space (which is 2 in CartPole) since we want 1 Q value per
        possible action.

        :return: the Q network
        """
        q_net = Sequential()
        q_net.add(Dense(64, input_dim=4, activation='relu', kernel_initializer='he_uniform'))
        q_net.add(Dense(32, activation='relu', kernel_initializer='he_uniform'))
        q_net.add(Dense(2, activation='linear', kernel_initializer='he_uniform'))
        q_net.compile(optimizer=tf.optimizers.Adam(learning_rate=0.01), loss='mse')
        return q_net


class ReplayBuffer:
    def __init__(self):
        self.gameplay_experiences = deque(maxlen=1000000)

    def store_gameplay_experience(self, state, next_state, reward, action, done):
        """
        Stores a step of gameplay experience in
        the buffer for later training
        """
        self.gameplay_experiences.append((state, next_state, reward, action, done))

    def sample_gameplay_batch(self):
        """
        Samples a batch of gameplay experiences
        for training purposes.
        """
        batch_size = min(128, len(self.gameplay_experiences))
        sampled_gameplay_batch = random.sample(self.gameplay_experiences, batch_size)
        state_batch, next_state_batch, action_batch, reward_batch, done_batch = [], [], [], [], []
        for gameplay_experience in sampled_gameplay_batch:
            state_batch.append(gameplay_experience[0])
            next_state_batch.append(gameplay_experience[1])
            reward_batch.append(gameplay_experience[2])
            action_batch.append(gameplay_experience[3])
            done_batch.append(gameplay_experience[4])
        return np.array(state_batch), np.array(next_state_batch), action_batch, reward_batch, done_batch


def collect_gameplay_experience(env, agent, buffer):
    """
  The collect_gameplay_experience function plays the game "env" with the
  instructions produced by "agent" and stores the gameplay experiences
  into "buffer" for later training.
  """
    state = env.reset()
    done = False
    while not done:
        action = agent.training_policy(state)
        next_state, reward, done, _ = env.step(action)
        buffer.store_gameplay_experience(state, next_state, reward, action, done)
        state = next_state


def evaluate_training_result(env, agent):
    """
    Evaluates the performance of the current DQN agent by using it to play a
    few episodes of the game and then calculates the average reward it gets.
    The higher the average reward is the better the DQN agent performs.
    :param env: the game environment
    :param agent: the DQN agent
    :return: average reward across episodes
    """

    total_reward = 0.0
    episodes_to_play = 10
    for i in range(episodes_to_play):
        state = env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            action = agent.training_policy(state)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
        total_reward += episode_reward
    average_reward = total_reward / episodes_to_play
    return average_reward


def test_model(env, agent):
    """
    Evaluates the performance of the current DQN agent by using it to play a
    few episodes of the game and then calculates the average reward it gets.
    The higher the average reward is the better the DQN agent performs.
    :param env: the game environment
    :param agent: the DQN agent
    :return: average reward across episodes
    """

    total_reward = 0.0
    episodes_to_play = 6
    for i in range(episodes_to_play):
        state = env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            env.render()
            action = agent.training_policy(state)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
        total_reward += episode_reward
        print('Episode {0}/{1} ended with {2} reward.'.format(i + 1, episodes_to_play, episode_reward))
    average_reward = total_reward / episodes_to_play
    print('Playtest ended with average reward of {0} after {1} episodes.'.format(average_reward, i + 1))
    return average_reward

def train_model(agent, max_episodes=501):
    """
    Trains a DQN agent to play the CartPole game by trial and error
    :return: None
    """

    buffer = ReplayBuffer()
    env = gym.make('CartPole-v0')
    for _ in range(100):
        collect_gameplay_experience(env, agent, buffer)
    for episode_cnt in range(max_episodes):
        collect_gameplay_experience(env, agent, buffer)
        gameplay_experience_batch = buffer.sample_gameplay_batch()
        loss = agent.train(gameplay_experience_batch)

        if episode_cnt % 20 == 0:
            agent.update_target_network()

        if episode_cnt % 10 == 0:
            avg_reward = evaluate_training_result(env, agent)
            print('Episode {0}/{1} and so far the performance is {2} and '
                  'loss is {3}'.format(episode_cnt, max_episodes,
                                       avg_reward, loss[0]))
            if avg_reward == 200.0:
                print('Game solved at episode {0}/{1} with {3} loss.'.format(episode_cnt,
                                                                             max_episodes, avg_reward, loss[0]))
                break
    env.close()
    print('Training finished.')


agent = DqnAgent()
agent.load_model('CartPoleModel')
env = gym.make('CartPole-v1')
test_model(env, agent)
#train_model(agent)
#agent.save_model('CartPoleModel')
#print("Model saved as CartPoleModel")




