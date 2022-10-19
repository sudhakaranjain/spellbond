from gym.envs.registration import register


register(
    id="WordleEnv10-v0",
    entry_point="wordle.env:WordleEnv10",
    max_episode_steps=500,
)

register(
    id="WordleEnv100-v0",
    entry_point="wordle.env:WordleEnv100",
    max_episode_steps=500,
)

register(
    id="WordleEnvFull-v0",
    entry_point="wordle.env:WordleEnvFull",
    max_episode_steps=500,
)

