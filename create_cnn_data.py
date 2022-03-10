import string
import numpy as np
from scrabblebot_framework.env import ScrabbleEnv
from scrabblebot_framework.utils.util import map_rows_to_values
from scrabblebot_framework.agents.greedy import GreedyAgent
from tqdm import tqdm


def create_cnn_data(n_train=5000, n_test=1000):
    """
    Returns:
        data: nx3x15x15 ndarray containing [board_vals] [board_letter_multipliers] [board_word_multipliers]
        labels: 1xn ndarray containing points from following turn
    """
    env = ScrabbleEnv(
        agents=[GreedyAgent, GreedyAgent],
        corpus_file="tournament_official",
    )
    score_lookup = env.game.score_lookup.copy()
    # Give empty spaces a value of 0
    score_lookup["_"] = 0
    # Give blanks a value of 1 so the CNN can see there's a letter there
    for char in string.ascii_uppercase:
        score_lookup[char] = 1
    # initialise output
    train_data = np.zeros((n_train, 3, 15, 15))
    train_labels = np.zeros(n_train)
    test_data = np.zeros((n_test, 3, 15, 15))
    test_labels = np.zeros(n_test)
    data_index = 0
    with tqdm(total=n_train, miniters=10) as pbar:
        while data_index < n_train:
            env.reset()
            while not env.game_over:
                board = map_rows_to_values(env.game.rows, score_lookup)
                letter_multipliers = env.game.row_letter_multipliers
                word_multipliers = env.game.row_word_multipliers
                _, score, _ = env.step()
                train_data[data_index, 0] = board
                train_data[data_index, 1] = letter_multipliers
                train_data[data_index, 2] = word_multipliers
                train_labels[data_index] = score
                data_index += 1
                pbar.update(1)
                if data_index == n_train:
                    break
    data_index = 0
    with tqdm(total=n_test, miniters=10) as pbar:
        while data_index < n_test:
            env.reset()
            while not env.game_over:
                board = map_rows_to_values(env.game.rows, score_lookup)
                letter_multipliers = env.game.row_letter_multipliers
                word_multipliers = env.game.row_word_multipliers
                _, score, _ = env.step()
                if score > 0:
                    test_data[data_index, 0] = board
                    test_data[data_index, 1] = letter_multipliers
                    test_data[data_index, 2] = word_multipliers
                    test_labels[data_index] = score
                    data_index += 1
                    pbar.update(1)
                if data_index == n_test:
                    break
    return train_data, train_labels, test_data, test_labels


if __name__ == "__main__":
    train_data, train_labels, test_data, test_labels = create_cnn_data(
        n_train=60000, n_test=6000)
    np.save("./cnn_train_data.npy", train_data)
    np.save("./cnn_train_labels.npy", train_labels)
    np.save("./cnn_test_data.npy", test_data)
    np.save("./cnn_test_labels.npy", test_labels)
