from functools import partial
from multiprocessing import Pool
import numpy as np
from game import ScrabbleGame
from util import get_playable_words


def process_candidate(candidate, game, trie, n_its):
    best_score = 0
    best_it_score = 0
    ghost_game = game.ghost_play(*candidate[0])
    for _ in range(n_its):
        ghost_game.ghost_rack(game.current_player)
        words = get_playable_words(
            ghost_game, trie)
        if len(words) > 0:
            best_opposing_word, best_it_score = words[np.argmax(
                list(map(lambda x: x[1], words))
            )]
        if best_it_score > best_score:
            best_score = best_it_score
    return best_score


def stochastic_lookahead(game: ScrabbleGame, trie, candidates, sample_size):
    with Pool() as p:
        candidate_scores = list(p.map(partial(process_candidate, game=game,
                                              trie=trie, n_its=sample_size), candidates))
    modified_scores = [candidate[1] - candidate_scores[i]
                       for i, candidate in enumerate(candidates)]
    return modified_scores


class StochasticLookaheadAgent:
    def __init__(self, trie, n_candidates=5, n_samples=1):
        self.trie = trie
        self.n_candidates = n_candidates
        self.n_samples = n_samples

    def step(self, game: ScrabbleGame):
        all_words = get_playable_words(game, self.trie)
        if len(all_words) == 0:
            return False, 0
        candidate_indices = np.argsort(list(map(lambda x: x[1], all_words)))
        candidates = [all_words[i]
                      for i in candidate_indices[-self.n_candidates:]]
        modified_scores = stochastic_lookahead(
            game, self.trie, candidates, self.n_samples)
        best_word, best_score = candidates[np.argmax(modified_scores)]
        return best_word, best_score
