from argparse import ArgumentParser
from spellbond import SARSA
import omegaconf
import numpy as np

if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--test-file", type=str, default="words.txt", help="Path to the text file with test words")
    parser.add_argument("--env", type=str, default="WordleEnv", help="gym environment tag")
    parser.add_argument("--vocab-size", type=int, default=None, help="The vocab size of the model: None(full) or 1000")
    parser.add_argument("--task", type=str, default="infer", help="gym environment tag")
    args = parser.parse_args()
    config = omegaconf.OmegaConf.load('../spellbond/models/configs.yaml')
    game = SARSA(args, config)

    if args.task == "train":
        game.train()

    elif args.task == "play":
        game.play()

    elif args.task == "finetune":
        game.finetune()

    elif args.task == "infer":
        with open(args.test_file, "r") as f:
            words = [x.strip() for x in f.readlines()]
        turns = []
        predictions = []
        if args.vocab_size:
            checkpoint = f"models_{args.vocab_size}.pth"
        else:
            checkpoint = f"models_finetuned_full.pth"
        for word in words:
            turn_no, predicted = game.infer(goal_word=word, model_checkpoint=checkpoint)
            turns.append(turn_no)
            predictions.append(predicted)
        print(f"Turns taken: {turns}")
        print(f"Average number of turns {np.mean(turns)}")
        print(f"Accuracy of predictions: {np.mean(predictions) * 100}%")
