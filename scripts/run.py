from argparse import ArgumentParser
from spellbond import SARSA
import omegaconf

if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--env", type=str, default="WordleEnv", help="gym environment tag")
    parser.add_argument("--vocab-size", type=int, default="1000", help="gym environment tag")
    args = parser.parse_args()
    config = omegaconf.OmegaConf.load('../spellbond/models/configs.yaml')
    game = SARSA(args, config)
    # game.train()
    # game.play()

    with open("words.txt", "r") as f:
        words = [x.strip().upper() for x in f.readlines()]

    turns = []
    for word in words:
        turns.append(game.infer(goal_word=word))
    print(turns)
