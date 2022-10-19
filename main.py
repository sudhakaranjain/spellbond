import logging
from argparse import ArgumentParser

import gym
import numpy as np

import wordle

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main(args) -> None:
    # Create the gym env and reset the state
    env = gym.make(args.env)
    state = env.reset()

    for i in range(env.max_turns):
        # Choose a random action that has not been chosen yet
        mask = np.int8(state)
        action = env.action_space.sample(mask=mask)
        LOGGER.info(f"Guessed word: {env.words[action]}")
        state, reward, done, aux, info = env.step(action)
        if done:
            # If a reward is given, the correct word was guessed
            if reward > 0:
                LOGGER.info(
                    f"You guessed the correct word. The word was {env.words[env.goal_word]}"
                )
            else:
                LOGGER.info(
                    f"You did not guess the correct word in {env.max_turns} turns. The correct word was {env.words[env.goal_word]}"
                )
            break


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "--env", type=str, default="WordleEnv10-v0", help="gym environment tag"
    )
    args = parser.parse_args()
    main(args)
