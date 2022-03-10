import numpy as np

from ..game import ScrabbleGame
from ..util import get_playable_words


class TileEfficiencyAgent:
    def __init__(self, trie):
        self.trie = trie

    def step(self, game: ScrabbleGame):
        all_words = get_playable_words(game, self.trie, return_raw_scores=True)
        if len(all_words) == 0:
            return False, 0, {}
        board_scores = np.array(list(map(lambda x: x[1][0], all_words)))
        tile_scores = np.array(list(map(lambda x: x[1][1], all_words)))
        efficiency_factors = board_scores/tile_scores
        best_word, best_score = all_words[np.argmax(efficiency_factors)]
        return best_word, best_score, {"efficiency_factors": efficiency_factors}
