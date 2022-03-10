import string
import numpy as np
from .scrabblebot_framework.env import ScrabbleEnv
from .scrabblebot_framework.agents.greedy import GreedyAgent
from .scrabblebot_framework.utils.util import map_rows_to_values
from .scrabblebot_framework.board_value_cnn import BoardValueCNN
import torch
from torch.nn import functional as F


def get_cnn_datapoint(game):
    score_lookup = game.score_lookup.copy()
    # Give empty spaces a value of 0
    score_lookup["_"] = 0
    # Give blanks a value of 1 so the CNN can see there's a letter there
    for char in string.ascii_uppercase:
        score_lookup[char] = 1
    vals = map_rows_to_values(game.rows, score_lookup)
    letter_multipliers = game.row_letter_multipliers
    word_multipliers = game.row_word_multipliers
    datum = np.zeros((1, 3, 15, 15))
    datum[0, 0] = vals
    datum[0, 1] = letter_multipliers
    datum[0, 2] = word_multipliers
    return torch.Tensor(datum)


def test_cnn():
    env = ScrabbleEnv([GreedyAgent, GreedyAgent],
                      "tournament_official")
    model = BoardValueCNN.load_from_checkpoint(
        "./cnn_checkpoints/lightning_logs/version_4/checkpoints/epoch=5-step=5627.ckpt")
    while not env.game_over:
        datum = get_cnn_datapoint(env.game)
        print(datum)
        word, score, info = env.step()
        if word:
            pred_score = model.forward(datum).detach()
            loss = F.mse_loss(pred_score, torch.Tensor([score]).unsqueeze(1))
            pred_score = pred_score.numpy().item()
            print(f"Predicted score: {pred_score:.2f}")
            print(f"Actual score: {score}")
            print(f"Diff: {score - pred_score:.2f}")
            print(f"Loss: {loss:.2f}")
        env.game.show()


if __name__ == "__main__":
    test_cnn()
