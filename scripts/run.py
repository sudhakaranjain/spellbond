from argparse import ArgumentParser
from spellbond import Wordle_RL
import omegaconf

if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "--env", type=str, default="WordleEnvFull-v0", help="gym environment tag"
    )
    args = parser.parse_args()
    config = omegaconf.OmegaConf.load('../spellbond/models/configs.yaml')
    game = Wordle_RL(args, config)
    game.train()
