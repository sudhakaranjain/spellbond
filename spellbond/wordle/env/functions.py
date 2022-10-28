from typing import Dict, List, Optional, Tuple

import numpy as np
from spellbond.wordle.env.const import WORDLE_N, ALPHABETS, POSSIBILITIES


def initialize_env(words):
    action_space = []
    new_state = np.tile(np.array([1, 0, 0], dtype=np.float32), (len(ALPHABETS), WORDLE_N, 1))
    for word in words:
        word = word.upper()
        action = np.zeros((len(ALPHABETS), WORDLE_N))
        for char_idx, char in enumerate(word):
            alpha_idx = ALPHABETS[char]
            action[alpha_idx, char_idx] = 1
        action_space.append(action)
    goal_idx = np.random.choice(len(words))
    return words[goal_idx], action_space[goal_idx], np.array(action_space, dtype=np.float32), new_state


def update_action_space(state, action_space, words):
    new_action_space = []
    new_words = []
    correct_char = []
    for char_idx in range(WORDLE_N):
        for alpha_idx in ALPHABETS.values():
            if state[alpha_idx, char_idx, POSSIBILITIES['YES']] == 1:
                correct_char.append((alpha_idx, char_idx, 1))
            elif state[alpha_idx, char_idx, POSSIBILITIES['NO']] == 1:
                correct_char.append((alpha_idx, char_idx, 0))
    for act_idx, action in enumerate(action_space):
        present = True
        for alpha_idx, char_idx, possibility in correct_char:
            if (possibility == 1 and action[alpha_idx, char_idx] != 1) or \
                    (possibility == 0 and action[alpha_idx, char_idx] == 1):
                present = False
                break
        if present:
            new_words.append(words[act_idx])
            new_action_space.append(action)
    return np.array(new_action_space), new_words


def update_state(predicted_word, state, goal_action):
    predicted_word = predicted_word.upper()
    for char_idx, char in enumerate(predicted_word):
        alpha_idx = ALPHABETS[char]
        state[alpha_idx, char_idx, POSSIBILITIES['MAYBE']] = 0
        if goal_action[alpha_idx, char_idx] == 1:
            state[alpha_idx, char_idx, POSSIBILITIES['YES']] = 1
        else:
            state[alpha_idx, char_idx, POSSIBILITIES['NO']] = 1
    return state


def compute_reward(old_state, state):
    reward = 0
    for char_idx in range(WORDLE_N):
        for alpha_idx in ALPHABETS.values():
            if state[alpha_idx, char_idx, POSSIBILITIES['YES']] > \
                    old_state[alpha_idx, char_idx, POSSIBILITIES['YES']]:
                reward += 1
                break
    return reward
