from collections import Counter
import numpy as np
from util import get_anchor_points, get_stencils
import logging
from time import perf_counter


class Agent:
    def __init__(self, starting_rack: Counter, trie):
        self.rack = starting_rack
        self.trie = trie

    def step(self, game):
        print(f"Rack: {''.join(key * val for key, val in self.rack.items())}")
        board = game.rows
        row_anchor_points, col_anchor_points = get_anchor_points(board)
        if np.all(list(map(lambda x: len(x) == 0, row_anchor_points))) and np.all(list(map(lambda x: len(x) == 0, col_anchor_points))):
            row_anchor_points[7].add(7)
            col_anchor_points[7].add(7)
        best_word = None
        best_score = 0
        vertical = False
        word_length = 0
        # logging.info("\n" + "\n".join(board))
        total_stencils = 0
        words_found = 0
        invalid_words = 0
        start = perf_counter()
        for length in range(2, 8):
            for row_index, anchors in enumerate(row_anchor_points):
                stencils = get_stencils(game.rows[row_index], anchors, length)
                total_stencils += len(stencils)
                for stencil, anchor in stencils:
                    stencil_words = self.trie.traverse(self.rack, stencil)
                    words_found += len(stencil_words)
                    scores = [game.get_score(word, anchor, row_index)
                              for word in stencil_words]
                    invalid_words += np.count_nonzero(np.array(scores) == -1)
                    if len(scores) > 0 and np.max(scores) > best_score:
                        best_score = np.max(scores)
                        best_word = (stencil_words[np.argmax(
                            scores)], anchor, row_index)
                        vertical = False
                        word_length = length
                        # logging.info(
                        #     f"Anchor: {anchor} row_index: {row_index} stencil: {stencil} word: {best_word}")
            for col_index, anchors in enumerate(col_anchor_points):
                stencils = get_stencils(game.cols[col_index], anchors, length)
                total_stencils += len(stencils)
                for stencil, anchor in stencils:
                    stencil_words = self.trie.traverse(self.rack, stencil)
                    words_found += len(stencil_words)
                    scores = [game.get_score(word, anchor, col_index, vertical=True)
                              for word in stencil_words]
                    invalid_words += np.count_nonzero(np.array(scores) == -1)
                    if len(scores) > 0 and np.max(scores) > best_score:
                        best_score = np.max(scores)
                        best_word = (stencil_words[np.argmax(
                            scores)], anchor, col_index)
                        vertical = True
                        word_length = length
                        # logging.info(
                        #     f"Anchor: {anchor} row_index: {row_index} stencil: {stencil} word: {best_word}")
        end = perf_counter()
        if best_word is not None:
            print(
                f"Total stencils: {total_stencils}\tExecution time: {end-start:.2f}s\tTime/word: {(end-start)/words_found:.2e}\nWords found: {words_found}\tInvalid words: {invalid_words}\tInvalid percent: {invalid_words*100/words_found:.2f}")
            word, loc, file = best_word
            board = game.rows if not vertical else game.cols
            existing_tiles = board[file][loc:loc+len(word)]
            tiles_used = [char for i, char in enumerate(
                word) if existing_tiles[i] != char]
            for tile in tiles_used:
                self.rack[tile] -= 1
                if self.rack[tile] == 0:
                    del self.rack[tile]
        return best_word, best_score, vertical, word_length

    def update_rack(self, new_letters):
        for letter in new_letters:
            self.rack[letter] += 1
