import os
from typing import Dict, List, Optional, Tuple

import gym
import numpy as np
from gym import spaces

from spellbond.wordle.env.const import WORDLE_N, MAX_TURNS
from spellbond.wordle.env.functions import initialize_env, update_action_space, update_state, compute_reward

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
        A Wordle environment compatible with gym Env.

        :param words: The list of words to use for the game.
        :param max_turns: The maximum number of turns to use for the game.
        """
        # Make sure the vocabulary only contains words of the chosen length.
        assert all(
            len(w) == WORDLE_N for w in words
        ), f"Not all words of length {WORDLE_N}, {words}"
        self.words = words
        self.max_turns = max_turns

        # Initialize the action and state space. In this case, they are the same, as the agent can observe every word.
        self.action_spaces = None
        self.state = None
        self.action_space = spaces.Discrete(len(self.words))
        self.observation_space = spaces.Discrete(len(self.words))

        self.done = True
        self.goal_word: str = ""
        self.goal_action: list = []

        self.remaining_steps = max_turns

    def step(self, predicted_word: str) -> Tuple[np.ndarray, int, bool, Dict, Dict]:
        """
        Implementation of the step function.

        :param predicted_word: Word predicted by actor model
        """
        if self.done:
            raise ValueError(
                "You are calling 'step()' even though this "
                "environment has already returned done = True. You "
                "should always call 'reset()' once you receive 'done = "
                "True' -- any further steps are undefined behavior."
            )
        self.state = update_state(predicted_word, self.state, self.goal_action)
        self.action_spaces, self.words = update_action_space(self.state, self.action_spaces, self.words)
        reward = compute_reward(self.state)
        self.remaining_steps -= 1

        if predicted_word == self.goal_word or self.remaining_steps == 0:
            self.done = True

        return (
            self.state,
            reward,
            self.done,
            {"aux": ""},
            {"action_space": self.action_spaces},
        )

    def reset(self) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Reset the whole environment.
        """
        self.remaining_steps = self.max_turns
        self.done = False
        self.goal_word, self.goal_action, self.action_spaces, self.state = initialize_env(words=self.words)

        return self.state, self.action_spaces, {}


class WordleEnv10(WordleEnvBase):
    def __init__(self):
        super().__init__(words=_load_words(10), max_turns=MAX_TURNS)


class WordleEnv100(WordleEnvBase):
    def __init__(self):
        super().__init__(words=_load_words(100), max_turns=MAX_TURNS)


class WordleEnvFull(WordleEnvBase):
    def __init__(self):
        super().__init__(words=_load_words(), max_turns=MAX_TURNS)
