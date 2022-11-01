import numpy as np

WordleState = np.ndarray


def new(n_words: int) -> WordleState:
    """
    Create a new WordleState.

    :param n_words: Number of words used.
    :return: The new WordleState.
    """
    # The state consists of an array of 1's of length n_words. 1 represents a not guessed word, 0 represents a guessed word.
    return np.array([1] * n_words)


def update(state: WordleState, word: str) -> WordleState:
    """
    Update the WordleState.

    :param state: The current state.
    :param word: The guessed word.
    :return: The new WordleState.
    """
    state = state.copy()
    # Set the value of the guessed word to 0.
    state[word] = 0

    return state
