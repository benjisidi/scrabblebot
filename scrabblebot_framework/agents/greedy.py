import numpy as np

from ..game import ScrabbleGame
from ..utils.util import get_playable_words


class GreedyAgent:
    def __init__(self, trie):
        self.trie = trie

    def step(self, game: ScrabbleGame):
        all_words = get_playable_words(game, self.trie)
        if len(all_words) == 0:
            return False, 0, {}
        best_word, best_score = all_words[np.argmax(
            list(map(lambda x: x[1], all_words))
        )]
        return best_word, best_score, {}
