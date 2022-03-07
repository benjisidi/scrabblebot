import numpy as np

from ..game import ScrabbleGame
from ..util import get_playable_words


class LongestWordAgent:
    def __init__(self, trie):
        self.trie = trie

    def step(self, game: ScrabbleGame):
        all_words = get_playable_words(game, self.trie)
        if len(all_words) == 0:
            return False, 0, {}
        longest_word, longest_score = all_words[np.argmax(
            list(map(lambda x: len(x[0][0]), all_words))
        )]
        return longest_word, longest_score, {}
