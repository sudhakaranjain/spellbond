import copy
import logging
import os
from time import sleep

import gym
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from spellbond.models import Actor, Critic
from spellbond.wordle.env.const import MAX_TURNS

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Wordle_RL:
    def __init__(self, arg, config):
        self.cos = nn.CosineSimilarity(dim=1)
        self.arg = arg
        self.smooth_l1loss = nn.SmoothL1Loss()
        self.config = config
        self.actor = Actor(self.config).to(device)
        self.critic = Critic(self.config).to(device)
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), self.config.optimizer.lr)
        self.optim_critic = torch.optim.Adam(self.critic.parameters(), self.config.optimizer.lr)

    def train_critic(self, state, next_state, reward, done, turn_no):
        with torch.no_grad():
            self.critic.eval()
            if done:
                target_q = torch.tensor([[reward]])
            else:
                next_turn_encoding = torch.tensor([0] * MAX_TURNS)
                next_turn_encoding[turn_no + 1] = 1
                next_state = torch.cat((torch.tensor(next_state).view(-1, ),
                                        next_turn_encoding)).to(device).unsqueeze(dim=0)
                target_q = reward + self.config.train.gamma * self.critic(next_state)

        self.critic.train()
        turn_encoding = torch.tensor([0] * MAX_TURNS)
        turn_encoding[turn_no] = 1
        state = torch.cat((torch.tensor(state).view(-1, ), turn_encoding)).to(device).unsqueeze(dim=0)
        self.optim_critic.zero_grad()
        predicted_q = self.critic(state)
        loss = self.smooth_l1loss(predicted_q, target_q.to(device))
        loss.backward()
        self.optim_critic.step()

    def train_actor(self, state, next_state, reward, done, turn_no, target_action):
        tde = self.compute_TDE(state, next_state, reward, done, turn_no)
        if tde > 0.0:
            self.actor.train()
            state = torch.tensor(state).view(-1, ).to(device).unsqueeze(dim=0)
            target_action = torch.tensor(target_action).view(-1, ).to(device).unsqueeze(dim=0)
            self.optim_actor.zero_grad()
            action = self.actor(state)
            loss = self.smooth_l1loss(action, target_action) + (1 - self.cos(action, target_action))
            loss.backward()
            self.optim_actor.step()

    def compute_TDE(self, state, next_state, reward, done, turn_no, ):
        with torch.no_grad():
            if done:
                target_q = torch.tensor([[reward]])
            else:
                next_turn_encoding = torch.tensor([0] * MAX_TURNS)
                next_turn_encoding[turn_no + 1] = 1
                next_state = torch.cat((torch.tensor(next_state).view(-1, ),
                                        next_turn_encoding)).to(device).unsqueeze(dim=0)
                target_q = reward + self.config.train.gamma * self.critic(next_state)
            turn_encoding = torch.tensor([0] * MAX_TURNS)
            turn_encoding[turn_no] = 1
            state = torch.cat((torch.tensor(state).view(-1, ), turn_encoding)).to(device).unsqueeze(dim=0)
            tde = target_q.to(device) - self.critic(state)
        return tde.item()

    def predict_action(self, state, action_space, words, training=True):
        def softmax(x):
            return np.exp(x) / np.exp(x).sum()

        with torch.no_grad():
            self.actor.eval()
            predicted_action = self.actor(torch.tensor(state).view(-1, ).unsqueeze(dim=0).to(device)).cpu().numpy()[0]
        values = softmax(np.array([np.dot(predicted_action, action.reshape(-1, )) for action in action_space]))
        if training:
            policy_choice = np.random.choice(len(action_space), p=values)
        else:
            policy_choice = np.argmax(values)
        return action_space[policy_choice], words[policy_choice]

    def train(self) -> None:
        accuracy_buffer = [0] * 100
        buffer_idx = 0
        prev_accuracy = 0
        epoch = 0
        with tqdm(total=100) as pbar:
            while sum(accuracy_buffer) < 90:
                # Create the gym env and reset the state
                env = gym.make(self.arg.env, vocab_size=self.arg.vocab_size)
                new_state, action_space, _ = env.reset()
                for turn_no in range(MAX_TURNS):
                    action, word = self.predict_action(new_state, action_space, env.words, True)
                    # LOGGER.info(f"Guessed word: {word}")
                    current_state = copy.deepcopy(new_state)
                    new_state, reward, done, _, info = env.step(word)
                    action_space = info['action_space']
                    # True reward calculated based on number of guesses used
                    true_reward = reward - self.config.train.rho ** turn_no

                    self.train_critic(current_state, new_state, true_reward, done, turn_no)
                    self.train_actor(current_state, new_state, true_reward, done, turn_no, action)

                    if done:
                        # If a reward is given, the correct word was guessed
                        if reward == 5:
                            # LOGGER.info(
                            #     f"You guessed the correct word on turn: {turn_no}. The word was {env.goal_word}"
                            # )
                            accuracy_buffer[buffer_idx] = 1
                        else:
                            # LOGGER.info(
                            #     f"You did not guess the correct word in {MAX_TURNS} turns. The correct word was {env.goal_word}"
                            # )
                            accuracy_buffer[buffer_idx] = 0
                        break
                if sum(accuracy_buffer) > prev_accuracy:
                    pbar.update(sum(accuracy_buffer) - prev_accuracy)
                    prev_accuracy = sum(accuracy_buffer)
                    torch.save(self.actor.state_dict(), os.path.join(self.config.train.checkpoint_path, 'actor.pth'))
                if epoch % 100000:
                    print(f"Completed {epoch} epochs")
                buffer_idx += 1 if buffer_idx < 99 else 0
                epoch += 1
