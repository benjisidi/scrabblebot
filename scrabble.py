from collections import Counter
import json
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np

from util import get_scrabble_trie


class ConstrainedSet(set):
    def __init__(self, minVal, maxVal):
        self.minVal = minVal
        self.maxVal = maxVal

    def add(self, val):
        if val >= self.minVal and val <= self.maxVal:
            super().add(val)


def find_stencils(row: str, anchor_points: list[int], length: int) -> list[str]:
    stencils = set()
    # Walk backwards from start_coord, counting spaces
    for anchor_point in anchor_points:
        spaces = 1
        starting_point = anchor_point
        while spaces < length and starting_point > 0:
            starting_point -= 1
            if row[starting_point] == "_":
                spaces += 1
        for i in range(0, anchor_point - starting_point+1):
            current_anchor = starting_point + i
            current_letter = current_anchor
            spaces = 0
            while spaces < length and current_letter < len(row):
                if row[current_letter] == "_":
                    spaces += 1
                current_letter += 1
            if spaces == length:
                stencil = row[current_anchor:current_letter]
                # Walk backwards from start adding existing letters
                cur_index = current_anchor
                cur_index -= 1
                while cur_index >= 0 and row[cur_index] != "_":
                    stencil = row[cur_index] + stencil
                    cur_index -= 1
                    current_anchor -= 1
                # Walk forwards from end adding existing letters
                cur_index = current_letter
                while cur_index < len(row) and row[cur_index] != "_":
                    stencil += row[cur_index]
                    cur_index += 1
                stencils.add((stencil, current_anchor))
    return stencils


def get_anchor_points(board: list[str]):
    # Temporarily pad board with 1 space around the outside for edge cases
    padded_board = board.copy()
    padded_board.append("_"*15)
    padded_board.insert(0, "_"*15)
    for i, row in enumerate(padded_board):
        padded_board[i] = "_" + row + "_"
    row_anchor_points = [set() for _ in padded_board]
    col_anchor_points = [set() for _ in padded_board]
    for i, row in enumerate(padded_board):
        for j, letter in enumerate(row):
            if letter != "_":
                if row[j-1] == "_":
                    row_anchor_points[i].add(j-1)
                    col_anchor_points[j].add(i-1)
                if row[j+1] == "_":
                    row_anchor_points[i].add(j+1)
                    col_anchor_points[j].add(i+1)
                if padded_board[i-1][j] == "_":
                    row_anchor_points[i-1].add(j)
                    col_anchor_points[j-1].add(i)
                if padded_board[i+1][j] == "_":
                    row_anchor_points[i+1].add(j)
                    col_anchor_points[j+1].add(i)
    row_anchor_points = row_anchor_points[1:-1]
    col_anchor_points = col_anchor_points[1:-1]
    for i in range(len(row_anchor_points)):
        row_anchor_points[i] = set(
            map(lambda x: x-1, filter(lambda x: x > 0 and x < 15, row_anchor_points[i])))
        col_anchor_points[i] = set(
            map(lambda x: x-1, filter(lambda x: x > 0 and x < 15, col_anchor_points[i])))
    return row_anchor_points, col_anchor_points


def get_secondary_words(word: str, loc: int, file: int, board: list[str]):
    secondary_words = []
    character_list = np.array(list(word))
    existing_chars = np.array(list(board[file][loc:loc+len(word)]))
    played_characters = np.where(
        character_list != existing_chars, character_list, "_")
    updated_board = board.copy()
    updated_board[file] = updated_board[file][:loc] + \
        word + updated_board[file][loc+len(word):]
    for i, letter in enumerate(played_characters):
        if letter != "_":
            seconday_word = ""
            anchor = loc + i
            start = file
            end = file + 1
            while updated_board[start][anchor] != "_" and start >= 0:
                seconday_word = updated_board[start][anchor] + seconday_word
                start -= 1
            while updated_board[end][anchor] != "_" and end < len(updated_board):
                seconday_word += updated_board[end][anchor]
                end += 1
            if len(seconday_word) > 1:
                secondary_words.append((seconday_word, start + 1, anchor))
    return secondary_words


def get_word_score(word, loc, file, letter_multipliers, word_multipliers, score_lookup):
    letter_values = np.sum(np.array(list(map(lambda x: score_lookup[x], list(
        word)))) * letter_multipliers[file][loc:loc+len(word)])
    score = letter_values * np.prod(
        word_multipliers[file][loc:loc+len(word)])
    return score


def get_total_score(word, loc, file, board, letter_multipliers, word_multipliers, letter_multipliers_perp, word_multipliers_perp, score_lookup):
    total_score = 0
    total_score += get_word_score(word, loc, file,
                                  letter_multipliers, word_multipliers, score_lookup)
    secondary_words = get_secondary_words(word, loc, file, board)
    for word, loc, file in secondary_words:
        total_score += get_word_score(word, loc, file,
                                      letter_multipliers_perp, word_multipliers_perp, score_lookup)
    return total_score


class ScrabbleGame:
    def __init__(self, n_players):
        self.rows = ["_"*15]*15
        self.cols = ["_"*15]*15
        # todo: replace 2s with 1s when played on
        self.row_letter_multipliers = np.array([
            [1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1],
            [1, 1, 1, 1, 1, 3, 1, 1, 1, 3, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1],
            [2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 3, 1, 1, 1, 3, 1, 1, 1, 3, 1, 1, 1, 3, 1],
            [1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1],
            [1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1],
            [1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1],
            [1, 3, 1, 1, 1, 3, 1, 1, 1, 3, 1, 1, 1, 3, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2],
            [1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 3, 1, 1, 1, 3, 1, 1, 1, 1, 1],
            [1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1],
        ])
        self.row_word_multipliers = np.array([
            [3, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 3],
            [1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1],
            [1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1],
            [1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1],
            [1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [3, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 3],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1],
            [1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1],
            [1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1],
            [1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1],
            [3, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 3],
        ])
        self.col_letter_multipliers = self.row_letter_multipliers.T
        self.col_word_multipliers = self.row_word_multipliers.T
        self.current_player = 0
        self.n_players = n_players
        self.scores = [0 for _ in range(n_players)]
        with open("./data/official_scrabble_words_2019.txt", "r") as f:
            self.corpus = set(f.read().splitlines())

    def save_board(self, filepath: str) -> None:
        pass

    def load_board(self, filepath: str) -> None:
        pass

    def play(self, word: str, x: int, y: int, vertical=False) -> int:
        """
        Adds a word to the board and returns the resulting score
        """
        # Check if word is valid
        # Todo: This check won't work - we need the resulting word(s), not the letters played
        # if word not in self.corpus:
        # raise ValueError(f"Invalid word: {word}")
        # Check if word clashes with existing tiles
        existing_tiles = np.array(list(
            self.rows[y][x:x+len(word)])) if not vertical else np.array(list(self.cols[x][y:y+len(word)]))
        word_chars = np.array(list(word))
        existing_tiles = np.where(
            existing_tiles == "_", word_chars, existing_tiles)
        if not np.all(word_chars == existing_tiles):
            raise ValueError(
                f"Word {word} cannot be played in position [{x},{y}]: conflicting tiles")
        if not vertical:
            # Check length
            if x + len(word) > 14:
                raise ValueError(
                    f"Word {word} is too long to be played in position [{x},{y}] horizontally")
            # Add word to row
            self.rows[y] = f"{self.rows[y][:x]}{word}{self.rows[y][x+len(word):]}"
            # Add letters to columns
            for i, char in enumerate(word):
                self.cols[x +
                          i] = f"{self.cols[x+i][:y]}{char}{self.cols[x+i][y+1:]}"
            # Update score multipliers
            self.row_letter_multipliers[y][x:x+len(word)] = 1
            self.col_letter_multipliers = self.row_letter_multipliers.T
            self.row_word_multipliers[y][x:x+len(word)] = 1
            self.col_word_multipliers = self.row_word_multipliers.T
        else:
            # Check if word matches existing tiles
            # Check length
            if y + len(word) > 14:
                raise ValueError(
                    f"Word {word} is too long to be played in position [{x},{y}] vertically")
            # Add word to col
            self.cols[x] = f"{self.cols[x][:y]}{word}{self.cols[x][y+len(word):]}"
            # Add letters to rows
            for i, char in enumerate(word):
                self.rows[y +
                          i] = f"{self.rows[y+i][:x]}{char}{self.rows[y+i][x+1:]}"
            # Update score multipliers
            self.col_letter_multipliers[x][y:y+len(word)] = 1
            self.row_letter_multipliers = self.col_letter_multipliers.T
            self.col_word_multipliers[x][y:y+len(word)] = 1
            self.row_word_multipliers = self.col_word_multipliers.T

    def show(self):
        scrabble_colormap = [
            [0.75, 0.69, 0.584],  # Tiles: Beige
            [0, 0.25, 0.274],  # board: Green
            [0.93, 0.678, 0.678],  # DW: Pink
            [0.58, 0.184, 0.208],  # TW: Red
            [0.78, 0.855, 0.914],  # DL: Light blue
            [0.416, 0.721, 0.886],  # TL: Dark blue
        ]
        _, ax = plt.subplots()
        # Transform multipliers to get unique value for each tile type
        data = self.row_letter_multipliers*2 + self.row_word_multipliers*0.5
        # Map tile values to 1 -> 5, with 0 being played letters (so colormap works nicely)
        for i, val in enumerate(np.unique(data)):
            data = np.where(data == val, i + 1, data)
        data[7, 7] = 0
        data = np.where(np.array([list(x) for x in self.rows]) != "_", 0, data)
        # Display board using custom colors
        im = ax.imshow(data, cmap=ListedColormap(scrabble_colormap))
        ax.set_xticks(np.arange(15)-.5, labels=[""]*15)
        ax.set_yticks(np.arange(15)-.5, labels=[""]*15)
        ax.grid(which="major", color="w", linestyle="-", linewidth=1)
        for i, row in enumerate(self.rows):
            for j, char in enumerate(row):
                if char != "_":
                    _ = ax.text(j, i, char.upper(), ha="center",
                                va="center", color="k", fontfamily="sans-serif", fontsize="xx-large", fontweight="extra bold")
        plt.show()


if __name__ == "__main__":
    with open("./data/constants.json", "r") as f:
        constants = json.loads(f.read())
    game = ScrabbleGame(2)
    game.play("locate", 7, 7)
    game.play("pearl", 7, 3, vertical=True)
    row_anchor_points = get_anchor_points(game.rows)[0]
    print(get_secondary_words("relax", 7, 6, game.rows))
    score_lookup = constants["scores"]
    print(get_total_score("ramp", 7, 13, game.cols, game.col_letter_multipliers,
                          game.col_word_multipliers, game.row_letter_multipliers,
                          game.row_word_multipliers, score_lookup))
    game.show()
