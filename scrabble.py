import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pprint import pprint
scrabble_colormap = [
    [0.75, 0.69, 0.584],  # Tiles: Beige
    [0, 0.25, 0.274],  # board: Green
    [0.93, 0.678, 0.678],  # DW: Pink
    [0.58, 0.184, 0.208],  # TW: Red
    [0.78, 0.855, 0.914],  # DL: Light blue
    [0.416, 0.721, 0.886],  # TL: Dark blue
]


class ScrabbleGame:
    def __init__(self, n_players):
        self.rows = ["_"*15]*15
        self.cols = ["_"*15]*15
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
        if word not in self.corpus:
            raise ValueError(f"Invalid word: {word}")
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

    def show(self):
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
    game = ScrabbleGame(2)
    game.play("locate", 7, 7)
    game.play("pearl", 7, 3, vertical=True)
    game.show()
