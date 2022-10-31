import copy
import logging
import os

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


def load_models(model_path):
    state_dict = torch.load(model_path, map_location=torch.device(device))
    return state_dict['actor'], state_dict['critic']


class Wordle_RL:
    def __init__(self, arg, config):
        self.cos = nn.CosineSimilarity(dim=1)
        self.arg = arg
        self.mse = nn.MSELoss()
        self.config = config
        self.epochs = self.config.train.epochs
        self.batch_size = self.config.train.batch_size
        self.actor = Actor(self.config).to(device)
        self.critic = Critic(self.config).to(device)
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), self.config.optimizer.lr)
        self.optim_critic = torch.optim.Adam(self.critic.parameters(), self.config.optimizer.lr)

    def train_critic(self, replay_buffer):
        with torch.no_grad():
            self.critic.eval()
            target_q = []
            inp = []
            for state, _, next_state, reward, done, turn_no in replay_buffer:
                state, hint = state
                next_state, next_hint = next_state
                if done:
                    target_q.append(torch.tensor([reward]).to(device))
                else:
                    next_turn_encoding = torch.tensor([0] * MAX_TURNS)
                    next_turn_encoding[turn_no + 1] = 1
                    next_state = torch.cat((torch.tensor(next_state).view(-1, ), torch.tensor(next_hint).view(-1, ),
                                            next_turn_encoding)).to(device).unsqueeze(dim=0)
                    target_q.append(reward + self.config.train.gamma * self.critic(next_state)[0])
                turn_encoding = torch.tensor([0] * MAX_TURNS)
                turn_encoding[turn_no] = 1
                inp.append(torch.cat((torch.tensor(state).view(-1, ), torch.tensor(hint).view(-1, ),
                                      turn_encoding)).to(device))

        self.critic.train()
        inp = torch.stack(inp).to(device)
        target_q = torch.stack(target_q).to(device)
        for _ in range(self.epochs):
            self.optim_critic.zero_grad()
            predicted_q = self.critic(inp)
            loss = self.mse(predicted_q, target_q)
            loss.backward()
            self.optim_critic.step()

    def train_actor(self, replay_buffer):
        inp = []
        target_action = []
        with torch.no_grad():
            for state, action, next_state, reward, done, turn_no in replay_buffer:
                tde = self.compute_TDE(state, next_state, reward, done, turn_no)
                state, hint = state
                if tde > 0.0:
                    inp.append(torch.cat((torch.tensor(state).view(-1, ), torch.tensor(hint).view(-1, ))).to(device))
                    target_action.append(torch.tensor(action).view(-1, ).to(device))
        if inp:
            self.actor.train()
            inp = torch.stack(inp).to(device)
            target_action = torch.stack(target_action).to(device)
            for _ in range(self.epochs):
                self.optim_actor.zero_grad()
                predicted_action = self.actor(inp)
                loss = self.mse(predicted_action, target_action) + \
                       (1 - self.cos(predicted_action, target_action)).mean()
                loss.backward()
                self.optim_actor.step()

    def compute_TDE(self, state, next_state, reward, done, turn_no):
        state, hint = state
        next_state, next_hint = next_state
        self.critic.eval()
        with torch.no_grad():
            if done:
                target_q = torch.tensor([[reward]])
            else:
                next_turn_encoding = torch.tensor([0] * MAX_TURNS)
                next_turn_encoding[turn_no + 1] = 1
                next_state = torch.cat((torch.tensor(next_state).view(-1, ), torch.tensor(next_hint).view(-1, ),
                                        next_turn_encoding)).to(device).unsqueeze(dim=0)
                target_q = reward + self.config.train.gamma * self.critic(next_state)
            turn_encoding = torch.tensor([0] * MAX_TURNS)
            turn_encoding[turn_no] = 1
            state = torch.cat((torch.tensor(state).view(-1, ), torch.tensor(hint).view(-1, ),
                               turn_encoding)).to(device).unsqueeze(dim=0)
            tde = target_q.to(device) - self.critic(state)
        return tde.item()

    def predict_action(self, state, action_space, words, training=True):
        def softmax(x):
            return np.exp(x) / np.exp(x).sum()
        state, hint = state
        state = torch.cat((torch.tensor(state).view(-1, ), torch.tensor(hint).view(-1, ))).unsqueeze(dim=0).to(device)
        with torch.no_grad():
            self.actor.eval()
            predicted_action = self.actor(state).cpu().numpy()[0]
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
            replay_buffer = list()
            while sum(accuracy_buffer) < 99:
                # Create the gym env and reset the state
                env = gym.make(self.arg.env, vocab_size=self.arg.vocab_size)
                new_state, action_space, _ = env.reset()
                for turn_no in range(MAX_TURNS):
                    action, word = self.predict_action(new_state, action_space, env.words, True)
                    # LOGGER.info(f"Guessed word: {word}")
                    current_state = copy.deepcopy(new_state)
                    current_action = copy.deepcopy(action)
                    new_state, reward, done, _, info = env.step(word)
                    action_space = info['action_space']
                    # True reward calculated based on number of guesses used
                    true_reward = reward - self.config.train.rho * turn_no
                    replay_buffer.append((current_state, current_action, new_state, true_reward, done, turn_no))

                    if len(replay_buffer) >= self.batch_size:
                        self.train_actor(replay_buffer)
                        self.train_critic(replay_buffer)
                        replay_buffer = list()

                    if done:
                        # If a reward is given, the correct word was guessed
                        if word == env.goal_word:
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
                    torch.save({'actor': self.actor.state_dict(), 'critic': self.critic.state_dict()},
                               os.path.join(self.config.train.checkpoint_path, 'models.pth'))
                if epoch % 50000 == 0:
                    print(f"Completed {epoch} epochs, Accuracy: {sum(accuracy_buffer) / 100}")
                    torch.save({'actor': self.actor.state_dict(), 'critic': self.critic.state_dict()},
                               os.path.join(self.config.train.checkpoint_path, 'models.pth'))
                buffer_idx += 1 if buffer_idx < 99 else 0
                epoch += 1

    def play(self):
        actor_weights, critic_weights = load_models(os.path.join(self.config.train.checkpoint_path, 'models_500k.pth'))
        self.critic.load_state_dict(critic_weights)
        self.critic.eval().to(device)
        self.actor.load_state_dict(actor_weights)
        self.actor.eval().to(device)
        env = gym.make(self.arg.env, vocab_size=self.arg.vocab_size)
        new_state, action_space, _ = env.reset()
        for turn_no in range(MAX_TURNS):
            print(f'turn {turn_no + 1}: ')
            action, word = self.predict_action(new_state, action_space, env.words, False)
            new_state, reward, done, _, info = env.step(word)
            new_state_critic, hint = new_state
            action_space = info['action_space']
            true_reward = reward - self.config.train.rho * turn_no
            print(f'Predicted word: {word}, goal word: {env.goal_word}')
            print(f'True reward: {true_reward}')
            turn_encoding = torch.tensor([0] * MAX_TURNS)
            turn_encoding[turn_no] = 1
            torch_state = torch.cat((torch.tensor(new_state_critic).view(-1, ), torch.tensor(hint).view(-1, ),
                                     turn_encoding)).to(device).unsqueeze(dim=0)
            with torch.no_grad():
                print(f'Critic Prediction: {self.critic(torch_state)} \n')
            if done:
                break
