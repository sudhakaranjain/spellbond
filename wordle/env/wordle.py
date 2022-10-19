import os
from typing import Dict, List, Optional, Tuple

import gym
import numpy as np
from gym import spaces

from wordle.env import state
from wordle.env.const import REWARD, WORDLE_N

dirname = os.path.dirname(__file__)
WORDS_PATH = f"{dirname}/../../data/wordle_words.txt"


def _load_words(limit: Optional[int] = None) -> List[str]:
    """
    Helper function to load the vocabulary.

    :param limit: Optional argument to limit the number of words used.
    :return: The (limitted) vocabulary.
    """
    with open(WORDS_PATH, "r") as f:
        lines = [x.strip().upper() for x in f.readlines()]
        if not limit:
            return lines
        else:
            return lines[:limit]


class WordleEnvBase(gym.Env):
    def __init__(self, words: List[str], max_turns: int) -> None:
        """
        A Wordle envrionment compatible with gym Env.

        :param words: The list of words to use for the game.
        :param max_turns: The maximum number of turns to use for the game.
        """
        # Make sure the vocabulary only contains words of the chosen length.
        assert all(
            len(w) == WORDLE_N for w in words
        ), f"Not all words of length {WORDLE_N}, {words}"
        self.words = words
        self.max_turns = max_turns

        # Initalize the action and observation space. In this case, they are the same, as the agent can observe every word.
        self.action_space = spaces.Discrete(len(self.words))
        self.observation_space = spaces.Discrete(len(self.words))

        self.done = True
        self.goal_word: int = -1

        self.state: state.WordleState = None
        self.state_updater = state.update

        self.remaining_steps = max_turns

    def step(self, action: int) -> Tuple[state.WordleState, int, bool, Dict, Dict]:
        """
        Implementation of the step function.

        :param action: The chosen action.
        :return: The (limitted) vocabulary.
        """
        if self.done:
            raise ValueError(
                "You are calling 'step()' even though this "
                "environment has already returned done = True. You "
                "should always call 'reset()' once you receive 'done = "
                "True' -- any further steps are undefined behavior."
            )
        self.state = self.state_updater(state=self.state, word=action)

        reward = 0
        self.remaining_steps -= 1

        if action == self.goal_word:
            self.done = True
            reward = REWARD

        elif self.remaining_steps == 0:

            self.done = True

        return (
            self.state.copy(),
            reward,
            self.done,
            {"goal_id": self.goal_word},
            {"info": ""},
        )

    def reset(self) -> state.WordleState:
        """
        Implementation of the reset function.
        """
        self.state = state.new(n_words=len(self.words))
        self.remaining_steps = self.max_turns
        self.done = False
        self.goal_word = int(np.random.random() * len(self.words))

        return self.state.copy()


class WordleEnv10(WordleEnvBase):
    def __init__(self):
        super().__init__(words=_load_words(10), max_turns=6)


class WordleEnv100(WordleEnvBase):
    def __init__(self):
        super().__init__(words=_load_words(100), max_turns=6)


class WordleEnvFull(WordleEnvBase):
    def __init__(self):
        super().__init__(words=_load_words(), max_turns=6)
