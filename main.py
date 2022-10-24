import copy
import logging
from argparse import ArgumentParser

import gym
import omegaconf
import numpy as np
import torch
import torch.nn as nn

from models import Actor, Critic

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_critic(state, next_state, reward, done, turn_no, optimizer):
    with torch.no_grad():
        critic.eval()
        if done:
            target_q = torch.tensor([[reward - turn_no ** 2]])
        else:
            target_q = reward - turn_no ** 2 + config.train.gamma * \
                       critic(torch.tensor(next_state).reshape(-1, ).unsqueeze(dim=0))
    critic.train()
    state = torch.tensor(state).view(-1,).unsqueeze(dim=0).to(device)
    optimizer.zero_grad()
    predicted_q = critic(state)
    loss = smooth_l1loss(predicted_q, target_q)
    loss.backward()
    optimizer.step()


def train_actor(state, next_state, reward, done, turn_no, target_action, optimizer):
    tde = compute_TDE(state, next_state, reward, done, turn_no)
    if tde > 0.0:
        actor.train()
        state = torch.tensor(state).view(-1, ).unsqueeze(dim=0).to(device)
        target_action = torch.tensor(target_action).view(-1, ).unsqueeze(dim=0).to(device)
        optimizer.zero_grad()
        action = actor(state)
        loss = smooth_l1loss(action, target_action) + (1 - cos(action, target_action))
        loss.backward()
        optimizer.step()


def compute_TDE(state, next_state, reward, done, turn_no,):
    with torch.no_grad():
        if done:
            target_q = reward - turn_no ** 2
        else:
            target_q = reward - turn_no ** 2 + config.train.gamma * \
                       critic(torch.tensor(next_state).reshape(-1, ).unsqueeze(dim=0))
        tde = target_q - critic(torch.tensor(state).reshape(-1, ).unsqueeze(dim=0))
    return tde.item()


def predict_action(state, action_space, words):
    def softmax(x):
        return np.exp(x) / np.exp(x).sum()
    with torch.no_grad():
        actor.eval()
        predicted_action = actor(torch.tensor(state).view(-1, ).unsqueeze(dim=0).to(device)).numpy()[0]
    values = softmax(np.array([np.dot(predicted_action, action.reshape(-1,)) for action in action_space]))
    policy_choice = np.random.choice(len(action_space), p=values)
    return action_space[policy_choice], words[policy_choice]


def main(arg) -> None:
    optim_actor = torch.optim.Adam(actor.parameters(), config.optimizer.lr)
    optim_critic = torch.optim.Adam(critic.parameters(), config.optimizer.lr)

    for epoch in range(config.train.epochs):
        # Create the gym env and reset the state
        env = gym.make(arg.env)
        new_state, action_space, _ = env.reset()
        for turn_no in range(env.max_turns):
            action, word = predict_action(new_state, action_space, env.words)
            LOGGER.info(f"Guessed word: {word}")
            current_state = copy.deepcopy(new_state)
            new_state, reward, done, _, info = env.step(word)
            action_space = info['action_space']

            train_critic(current_state, new_state, reward, done, turn_no, optim_critic)
            train_actor(current_state, new_state, reward, done, turn_no, action, optim_actor)

            if done:
                # If a reward is given, the correct word was guessed
                if reward == 5:
                    LOGGER.info(
                        f"You guessed the correct word. The word was {env.goal_word}"
                    )
                else:
                    LOGGER.info(
                        f"You did not guess the correct word in {env.max_turns} turns. The correct word was {env.goal_word}"
                    )
                break


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "--env", type=str, default="WordleEnv10-v0", help="gym environment tag"
    )
    args = parser.parse_args()
    cos = nn.CosineSimilarity(dim=1)
    smooth_l1loss = nn.SmoothL1Loss()
    config = omegaconf.OmegaConf.load('models/configs.yaml')
    actor = Actor(config).to(device)
    critic = Critic(config).to(device)
    main(args)
