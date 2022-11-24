from argparse import ArgumentParser
from spellbond import Wordle_RL, SARSA
import omegaconf

if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--env", type=str, default="WordleEnv", help="gym environment tag")
    parser.add_argument("--vocab-size", type=int, default="1000", help="gym environment tag")
    args = parser.parse_args()
    config = omegaconf.OmegaConf.load('../spellbond/models/configs.yaml')
    game = SARSA(args, config)
    game.train()
    # game.play()
    # game.infer()
