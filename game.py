import json
import string
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.transforms import IdentityTransform, ScaledTranslation

from util import (get_anchor_points, get_secondary_words, get_total_score,
                  transpose_board)


class ScrabbleGame:
    def __init__(self, n_players, constants, corpus):
        self.rows = ["_"*15]*15
        self.cols = ["_"*15]*15
        self.row_letter_multipliers = np.array(
            constants["initial_letter_multipliers"])
        self.row_word_multipliers = np.array(
            constants["initial_word_multipliers"])
        self.col_letter_multipliers = self.row_letter_multipliers.T
        self.col_word_multipliers = self.row_word_multipliers.T
        self.current_player = 0
        self.n_players = n_players
        self.scores = [0 for _ in range(n_players)]
        self.corpus = set(corpus)
        self.score_lookup = constants["scores"]
        self.bag = Counter(constants["freqs"])
        self.racks = [Counter(self.draw_letters(7)) for _ in range(n_players)]
        self.players_passed = [False for _ in range(n_players)]
        self.game_over = False

    def reset(self, constants):
        self.row_letter_multipliers = np.array(
            constants["initial_letter_multipliers"])
        self.row_word_multipliers = np.array(
            constants["initial_word_multipliers"])
        self.scores = [0 for _ in range(self.n_players)]
        self.bag = Counter(constants["freqs"])
        self.racks = [Counter(self.draw_letters(7))
                      for _ in range(self.n_players)]
        self.players_passed = [False for _ in range(self.n_players)]
        self.game_over = False

    def draw_letters(self, n):
        alphabet = list(string.ascii_lowercase+"*")
        letters = ""
        alphabet_counts = list(map(lambda x: self.bag[x], alphabet))
        for _ in range(min(n, self.bag.total())):
            letter = np.random.choice(
                list(string.ascii_lowercase+"*"),
                p=alphabet_counts/np.sum(alphabet_counts)
            )
            self.bag[letter] -= 1
            alphabet_counts = list(map(lambda x: self.bag[x], alphabet))
            letters += letter
        return letters

    def save_board(self, filepath: str) -> None:
        pass

    def load_board(self, filepath: str) -> None:
        pass

    def get_score(self, word, loc, file, vertical=False):
        if vertical:
            board = self.cols
            letter_multipliers = self.col_letter_multipliers
            word_multipliers = self.col_word_multipliers
            letter_multipliers_perp = self.row_letter_multipliers
            word_multipliers_perp = self.row_word_multipliers
        else:
            board = self.rows
            letter_multipliers = self.row_letter_multipliers
            word_multipliers = self.row_word_multipliers
            letter_multipliers_perp = self.col_letter_multipliers
            word_multipliers_perp = self.col_word_multipliers
        secondary_words = get_secondary_words(word, loc, file, board)
        if not np.all([x in self.corpus for x in list(map(lambda x: x[0], secondary_words))]):
            return -1
        return get_total_score(
            word=word,
            loc=loc,
            file=file,
            board=board,
            letter_multipliers=letter_multipliers,
            word_multipliers=word_multipliers,
            letter_multipliers_perp=letter_multipliers_perp,
            word_multipliers_perp=word_multipliers_perp,
            score_lookup=self.score_lookup
        )

    def pass_turn(self) -> bool:
        self.players_passed[self.current_player] = True
        if np.all(self.players_passed):
            self.game_over = True
        # Increment the current player
        self.current_player = (self.current_player + 1) % self.n_players
        return self.game_over

    def play(self, word: str, loc: int, file: int, vertical=False) -> bool:
        """
        Adds a word to the board
        """
        # Check if word is valid
        if word.lower() not in self.corpus:
            raise ValueError(f"Invalid word: {word}")
        if vertical:
            board = self.cols
            letter_multipliers = self.col_letter_multipliers
            word_multipliers = self.col_word_multipliers
        else:
            board = self.rows
            letter_multipliers = self.row_letter_multipliers
            word_multipliers = self.row_word_multipliers
        # Check if word clashes with existing tiles
        existing_tiles = np.array(list(board[file][loc:loc+len(word)]))
        word_chars = np.array(list(word))
        letters_played = "".join(np.where(
            existing_tiles == "_", word_chars, ""))
        resulting_word = np.where(
            existing_tiles == "_", word_chars, existing_tiles)
        if not np.all(word_chars == resulting_word):
            raise ValueError(
                f"Word {word} cannot be played in position [{file},{loc}] {vertical and '(vertical)' or ''}: conflicting tiles")
        # Check length
        if loc + len(word) - 1 > 14:
            raise ValueError(
                f"Word {word} is too long to be played in position [{file},{loc}]")

        # Get score and add it to player's total
        score = self.get_score(word, loc, file, vertical)
        self.scores[self.current_player] += score

        # Add word to file
        board[file] = f"{board[file][:loc]}{word}{board[file][loc+len(word):]}"
        # Update score multipliers
        letter_multipliers[file][loc:loc+len(word)] = 1
        word_multipliers[file][loc:loc+len(word)] = 1

        # Update instance variables as appropriate
        if vertical:
            self.cols = board
            self.rows = transpose_board(board)
            self.col_letter_multipliers = letter_multipliers
            self.row_letter_multipliers = letter_multipliers.T
        else:
            self.rows = board
            self.cols = transpose_board(board)
            self.row_letter_multipliers = letter_multipliers
            self.col_letter_multipliers = letter_multipliers.T

        # Update player's rack
        # if not all letters are lowercase, at least one blank was used
        if not letters_played.islower():
            nonblanks = "".join(x for x in letters_played if x.islower())
            n_blanks = len(letters_played) - len(nonblanks)
            self.racks[self.current_player] -= Counter(nonblanks)
            self.racks[self.current_player]["*"] -= n_blanks
        else:
            self.racks[self.current_player] -= Counter(letters_played)
        self.racks[self.current_player] += Counter(
            self.draw_letters(len(letters_played)))
        # End the game if both the rack and bag are empty
        if self.racks[self.current_player].total() == 0 and self.bag.total() == 0:
            # Endgame scoring
            for i, rack in enumerate(self.racks):
                score = 0
                for (char, cnt) in rack.items():
                    score += cnt * self.score_lookup[char]
                self.scores[i] -= score
                self.scores[self.current_player] += score
            self.game_over = True

        # Increment the current player
        self.current_player = (self.current_player + 1) % self.n_players
        return self.game_over

    def show(self):
        scrabble_colormap = [
            [0.75, 0.69, 0.584],  # Tiles: Beige
            [0, 0.25, 0.274],  # board: Green
            [0.93, 0.678, 0.678],  # DW: Pink
            [0.58, 0.184, 0.208],  # TW: Red
            [0.78, 0.855, 0.914],  # DL: Light blue
            [0.416, 0.721, 0.886],  # TL: Dark blue
        ]
        fig, ax = plt.subplots()
        fig.set_size_inches(10, 10)
        # Transform multipliers to get unique value for each tile type
        data = self.row_letter_multipliers*2 + self.row_word_multipliers*0.5
        # Map tile values to 1 -> 5, with 0 being played letters (so colormap works nicely)
        for i, val in enumerate(np.unique(data)):
            data = np.where(data == val, i + 1, data)
        data[7, 7] = 0
        data = np.where(np.array([list(x) for x in self.rows]) != "_", 0, data)
        # Display board using custom colors
        ax.imshow(data, cmap=ListedColormap(scrabble_colormap))
        ax.set_xticks(np.arange(15)-.5, labels=[""]*15)
        ax.set_yticks(np.arange(15)-.5, labels=[""]*15)
        ax.grid(which="major", color="w", linestyle="-", linewidth=1)
        for i, row in enumerate(self.rows):
            for j, char in enumerate(row):
                if char != "_":
                    _ = ax.text(j, i, char.upper(), ha="center",
                                va="center", color="k", fontfamily="sans-serif", fontsize="xx-large", fontweight="extra bold")
                    t = ax.text(j, i, self.score_lookup[char], ha="right",
                                va="bottom", color="k", fontfamily="sans-serif", fontsize="medium", fontweight="bold")
                    trans = t.get_transform()
                    offs = ScaledTranslation(0.4, 0.4, IdentityTransform())
                    t.set_transform(offs + trans)
        plt.show()


if __name__ == "__main__":
    with open("./data/constants.json", "r") as f:
        constants = json.loads(f.read())
    with open("./data/official_scrabble_words_2019.txt", "r") as f:
        corpus = f.read().splitlines()
    game = ScrabbleGame(2, constants, corpus)
    game.play("locate", 7, 7)
    game.play("pearl", 3, 7, vertical=True)
    row_anchors, col_anchors = get_anchor_points(game.rows)
    print(row_anchors)
    print(col_anchors)
    print(game.scores)
    print(game.get_score("reload", 7, 6))
    print(game.get_score("bib", 10, 6))
    game.play("bib", 10, 6)
    game.show()
