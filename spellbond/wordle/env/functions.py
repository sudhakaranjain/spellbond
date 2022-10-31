from typing import Dict, List, Optional, Tuple

import numpy as np
from spellbond.wordle.env.const import WORDLE_N, ALPHABETS, POSSIBILITIES


def initialize_env(words):
    action_space = []
    new_state = np.tile(np.array([1, 0, 0], dtype=np.float32), (len(ALPHABETS), WORDLE_N, 1))
    hint = np.zeros(len(ALPHABETS), dtype=np.float32)
    for word in words:
        word = word.upper()
        action = np.zeros((len(ALPHABETS), WORDLE_N))
        for char_idx, char in enumerate(word):
            alpha_idx = ALPHABETS[char]
            action[alpha_idx, char_idx] = 1
        action_space.append(action)
    goal_idx = np.random.choice(len(words))
    return words[goal_idx], action_space[goal_idx], np.array(action_space, dtype=np.float32), (new_state, hint)


def update_action_space(state, action_space, words):
    new_action_space = []
    new_words = []
    correct_char = []
    state, hint = state
    for alpha_idx in ALPHABETS.values():
        for char_idx in range(WORDLE_N):
            if state[alpha_idx, char_idx, POSSIBILITIES['YES']] == 1:
                correct_char.append((alpha_idx, char_idx, 1))
            elif state[alpha_idx, char_idx, POSSIBILITIES['NO']] == 1:
                correct_char.append((alpha_idx, char_idx, 0))
        if hint[alpha_idx] == 1:
            correct_char.append((alpha_idx, None, 2))
    for act_idx, action in enumerate(action_space):
        present = True
        for alpha_idx, char_idx, possibility in correct_char:
            if (possibility == 1 and action[alpha_idx, char_idx] != 1) or \
                  (possibility == 0 and action[alpha_idx, char_idx] == 1) or \
                    (possibility == 2 and (1 not in action[alpha_idx, :])):
                present = False
                break
        if present:
            new_words.append(words[act_idx])
            new_action_space.append(action)
    return np.array(new_action_space), new_words


def update_state(predicted_word, goal_word, state, goal_action):
    state, hint = state
    predicted_word = predicted_word.upper()
    goal_word = goal_word.upper()
    for char_idx, char in enumerate(predicted_word):
        alpha_idx = ALPHABETS[char]
        state[alpha_idx, char_idx, POSSIBILITIES['MAYBE']] = 0
        if char in goal_word:
            hint[alpha_idx] = 1
        if goal_action[alpha_idx, char_idx] == 1:
            state[alpha_idx, char_idx, POSSIBILITIES['YES']] = 1
        else:
            state[alpha_idx, char_idx, POSSIBILITIES['NO']] = 1
    return state, hint


def compute_reward(old_state, state):
    reward = 0
    old_state, _ = old_state
    state, _ = state
    for char_idx in range(WORDLE_N):
        for alpha_idx in ALPHABETS.values():
            if state[alpha_idx, char_idx, POSSIBILITIES['YES']] > \
                    old_state[alpha_idx, char_idx, POSSIBILITIES['YES']]:
                reward += 1
                break
    return reward
