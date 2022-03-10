import numpy as np

from ..game import ScrabbleGame
from ..utils.util import get_playable_words


class TileEfficiencyAgent:
    def __init__(self, trie):
        self.trie = trie

    def step(self, game: ScrabbleGame):
        mu = 2
        lambda_ = 1
        all_words = get_playable_words(game, self.trie, return_raw_scores=True)
        if len(all_words) == 0:
            return False, 0, {}
        board_scores = np.array(list(map(lambda x: x[1][0], all_words)))
        tile_scores = np.array(list(map(lambda x: x[1][1], all_words)))
        # Very rarely, a tile_score will be 0 from just playing a blank.
        # We correct for this so we don't need the warning
        with np.errstate(divide='ignore'):
            efficiency_factors = board_scores/tile_scores
        efficiency_factors = np.where(
            lambda x: np.isfinite(x), efficiency_factors, 1)
        max_score = np.max(board_scores)
        regularized_score = (board_scores - max_score) * \
            mu + np.log2(efficiency_factors)*lambda_
        best_word, best_score = all_words[np.argmax(regularized_score)]
        return best_word, best_score, {"regularized_score": regularized_score, "words": all_words, "max_score": max_score}
