class ScrabbleGame:
    def __init__(self, n_players):
        self.rows = []
        self.cols = []
        self.row_letter_multipliers = []
        self.row_word_multipliers = []
        self.col_letter_multipliers = []
        self.col_word_multipliers = []
        self.current_player = 0
        self.n_players = n_players
        self.scores = [0 for _ in range(n_players)]

    def save_board(self, filepath: str) -> None:
        pass

    def load_board(self, filepath: str) -> None:
        pass

    def play_word(self, word: str, loc: list[int], vertical=False) -> int:
        """
        Adds a word to the board and returns the resulting score
        """
        x, y = loc
        # Todo: Throw err if word goes off board
        # Todo: Throw err if word not in corpus
        # Todo: Throw err if word doesn't match location stencil
        # Todo: Throw err if word induces other invalid words on board
        if not vertical:
            self.rows[y][x:x+len(word)] = word
            self.cols[x][y] = word[0]
        else:
            self.cols[x][y:y+len(word)] = word
            self.rows[y][x] = word[0]

    def show():
        pass
