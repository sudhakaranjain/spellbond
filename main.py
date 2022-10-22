import copy
import logging
from argparse import ArgumentParser

import gym
import omegaconf
import numpy as np
import torch
import torch.nn as nn

from models.models import Actor, Critic

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, train_data, optimizer):
    model.train()
    state, target_action = train_data.to(device)
    optimizer.zero_grad()
    action = model(state)
    loss = 1 - cos(action, target_action)
    loss.backward()
    optimizer.step()


def perform_action(actor, state, action_space, words):
    with torch.no_grad():
        actor.eval()
        predicted_action = actor(state).numpy()[0]
    values = np.array([np.dot(predicted_action, action) for action in action_space])
    return words[np.argmax(values)]


def main(arg) -> None:
    # Create the gym env and reset the state
    env = gym.make(arg.env)
    state, action_space, _ = env.reset()
    config = omegaconf.OmegaConf.load('models/configs.yaml')
    actor = Actor(config).to(device)
    critic = Critic(config).to(device)

    optim_actor = torch.optim.Adam(actor.parameters(), config.optimizer.lr)
    critic_actor = torch.optim.Adam(critic.parameters(), config.optimizer.lr)

    for i in range(env.max_turns):
        # Choose a random action that has not been chosen yet
        # predicted_word = np.random.choice(np.array(env.words))
        predicted_word = perform_action(actor, state, action_space, env.words)
        LOGGER.info(f"Guessed word: {predicted_word}")
        old_state = copy.deepcopy(state)
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


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "--env", type=str, default="WordleEnv10-v0", help="gym environment tag"
    )
    args = parser.parse_args()
    cos = nn.CosineSimilarity(dim=1)
    main(args)
