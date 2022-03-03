import numpy as np

from game import ScrabbleGame
from util import get_playable_words


class LongestWordAgent:
    def __init__(self, trie):
        self.trie = trie

    def step(self, game: ScrabbleGame):
        horizontal_words, vertical_words = get_playable_words(game, self.trie)
        horizontal_word, horizontal_score = horizontal_words[np.argmax(
            list(map(lambda x: len(x[0][0]), horizontal_words))
        )]
        vertical_word, vertical_score = vertical_words[np.argmax(
            list(map(lambda x: len(x[0][0]), vertical_words))
        )]
        if len(horizontal_word) > len(vertical_word):
            return horizontal_word, horizontal_score, False
        else:
            return vertical_word, vertical_score, True
