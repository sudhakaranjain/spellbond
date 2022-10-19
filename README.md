# spellbond
Gym environment for the game Wordle. A simple action, state, and observation is included, with a random agent. Note that the spaces are very simplistic, and only included for a working example of the env.

### How to run
A game can be played by the random agent by running:
```commandline
python main.py
```

### Variations
There are three environments to choose from:
- "WordleEnv10-v0": Wordle with 10 words (easy)
- "WordleEnv100-v0": Wordle with 100 words (medium)
- "WordleEnvFull-v0": Wordle with all words (hard)

Other variations can be added by subclassing WordleEnvBase. Make sure to add the new variation to both __init__.py files to register them to gym properly.

