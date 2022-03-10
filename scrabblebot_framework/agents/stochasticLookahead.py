from functools import partial
from multiprocessing import Pool
import numpy as np
from ..game import ScrabbleGame
from ..trie import Trie
from ..utils.util import get_playable_words


def process_candidate(candidate: str, game: ScrabbleGame, trie: Trie, n_its: int):
    ghost_game = game.ghost_play(*candidate[0])
    ghost_racks = game.generate_ghost_racks(
        n_its, visible_rack=game.current_player)
    candidate_scores = []
    candidate_words = []
    print(candidate)
    for rack in ghost_racks:
        words = get_playable_words(ghost_game, trie, rack=rack)
        if len(words) > 0:
            best_opposing_word, best_it_score = words[np.argmax(
                list(map(lambda x: x[1], words))
            )]
            candidate_scores.append(best_it_score)
            candidate_words.append(best_opposing_word)
    print(f"Min: {np.min(candidate_scores)}\tMax: {np.max(candidate_scores)}\tMean: {np.mean(candidate_scores):.2f}\tStd: {np.std(candidate_scores):.2f}")
    print(
        f"Min: {candidate_words[np.argmin(candidate_scores)]}\nMax: {candidate_words[np.argmax(candidate_scores)]}")
    return np.mean(candidate_scores)


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
            return False, 0, {}
        candidate_indices = np.argsort(list(map(lambda x: x[1], all_words)))
        candidates = [all_words[i]
                      for i in candidate_indices[-self.n_candidates:]]
        modified_scores = stochastic_lookahead(
            game, self.trie, candidates, self.n_samples)
        best_word, best_score = candidates[np.argmax(modified_scores)]
        expected_score = np.max(modified_scores)
        return best_word, best_score, {"expected_score": expected_score}
