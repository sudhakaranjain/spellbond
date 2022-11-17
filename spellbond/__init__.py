from spellbond.game import Wordle_RL, SARSA
from gym.envs.registration import register


register(
    id="WordleEnv10-v0",
    entry_point="spellbond.wordle.env:WordleEnv10",
    max_episode_steps=500,
)

register(
    id="WordleEnv100-v0",
    entry_point="spellbond.wordle.env:WordleEnv100",
    max_episode_steps=500,
)

register(
    id="WordleEnv",
    entry_point="spellbond.wordle.env:WordleEnv",
    max_episode_steps=500,
)

register(
    id="WordleEnvFull-v0",
    entry_point="spellbond.wordle.env:WordleEnvFull",
    max_episode_steps=500,
)
