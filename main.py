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


def train_actor(state, target_action, optimizer):
    actor.train()
    state, target_action = torch.tensor(state).to(device), torch.tensor(target_action).to(device)
    optimizer.zero_grad()
    action = actor(state.unsqueeze(dim=0))
    loss = 1 - cos(action, target_action.unsqueeze(dim=0))
    loss.backward()
    optimizer.step()


def train_critic(state, action, target_q, optimizer):
    critic.train()
    state, action, target_q = torch.tensor(state).to(device), torch.tensor(action).to(device), \
                              torch.tensor(target_q).to(device)
    s_a = torch.cat(state, action).unsqueeze(dim=0)
    optimizer.zero_grad()
    predicted_q = critic(s_a)
    loss = smooth_l1loss(predicted_q, target_q.unsqueeze(dim=0))
    loss.backward()
    optimizer.step()


def perform_action(state, action_space, words):
    with torch.no_grad():
        actor.eval()
        predicted_action = actor(state).numpy()[0]
    values = np.array([np.dot(predicted_action, action) for action in action_space])
    return words[np.argmax(values)], action_space[np.argmax(values)]


def main(arg) -> None:
    # Create the gym env and reset the state
    env = gym.make(arg.env)
    optim_actor = torch.optim.Adam(actor.parameters(), config.optimizer.lr)
    optim_critic = torch.optim.Adam(critic.parameters(), config.optimizer.lr)

    for epoch in range(config.train.epochs):
        state, action_space, _ = env.reset()
        for i in range(env.max_turns):
            # Choose a random action that has not been chosen yet
            # predicted_word = np.random.choice(np.array(env.words))
            predicted_word, action = perform_action(state, action_space, env.words)
            LOGGER.info(f"Guessed word: {predicted_word}")
            old_state = copy.deepcopy(state)
            old_action_space = copy.deepcopy(action_space)
            state, reward, done, _, info = env.step(predicted_word)
            action_space = info['action_space']

            if done:
                # If a reward is given, the correct word was guessed
                if reward > 0:
                    LOGGER.info(
                        f"You guessed the correct word. The word was {env.goal_word}"
                    )
                else:
                    LOGGER.info(
                        f"You did not guess the correct word in {env.max_turns} turns. The correct word was {env.goal_word}"
                    )
                break

            else:
                Q = compute_Q_value(critic, old_state, state)   # Yet to Implement
                train_critic(old_state, action, Q, optim_critic)
                optimal_action = greedy_critic_suggestion(critic, old_state, old_action_space)   # Yet to Implement
                train_actor(old_state, optimal_action, optim_actor)


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
