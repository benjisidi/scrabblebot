import numpy as np

from game import ScrabbleGame
from util import get_playable_words


class GreedyAgent:
    def __init__(self, trie):
        self.trie = trie

    def step(self, game: ScrabbleGame):
        horizontal_words, vertical_words = get_playable_words(game, self.trie)
        horizontal_word, horizontal_score = horizontal_words[np.argmax(
            list(map(lambda x: x[1], horizontal_words))
        )]
        vertical_word, vertical_score = vertical_words[np.argmax(
            list(map(lambda x: x[1], vertical_words))
        )]
        if horizontal_score > vertical_score:
            return horizontal_word, horizontal_score, False
        else:
            return vertical_word, vertical_score, True
