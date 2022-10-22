import copy
from typing import Dict, List, Optional, Tuple

import numpy as np
from wordle.env.const import WORDLE_N, ALPHABETS, POSSIBILITIES


def initialize_env(words):
    action_space = []
    new_state = np.tile(np.array([1, 0, 0]), (len(ALPHABETS), WORDLE_N, 1))
    for word in words:
        word = word.upper()
        action = np.zeros((len(ALPHABETS), WORDLE_N))
        for char_idx, char in enumerate(word):
            alpha_idx = ALPHABETS[char]
            action[alpha_idx, char_idx] = 1
        action_space.append(action)
    goal_idx = np.random.choice(len(words))
    return words[goal_idx], action_space[goal_idx], np.array(action_space), new_state


def update_action_space(state, action_space):
    new_action_space = []
    correct_char = []
    for char_idx in range(WORDLE_N):
        for alpha_idx in ALPHABETS.values():
            if state[alpha_idx, char_idx, POSSIBILITIES['YES']] == 1:
                correct_char.append((alpha_idx, char_idx))
                break
    for act_idx, action in enumerate(action_space):
        present = True
        for alpha_idx, char_idx in correct_char:
            if action[alpha_idx, char_idx] != 1:
                present = False
                break
        if present:
            new_action_space.append(action)
    return np.array(new_action_space)


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


def compute_reward(state):
    reward = 0
    for char_idx in range(WORDLE_N):
        for alpha_idx in ALPHABETS.values():
            if state[alpha_idx, char_idx, POSSIBILITIES['YES']]:
                reward += 1
                break
    return reward


# _, new_goal_action, _, init_state = initialize_env((['cat']))
# predict1 = 'him'
# init_state = update_state(predict1, init_state, new_goal_action)
# predict2 = 'rat'
# init_state = update_state(predict2, init_state, new_goal_action)
# print(init_state)
