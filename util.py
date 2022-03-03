from collections import Counter
from functools import cache
from time import perf_counter
import numpy as np

from trie import Trie


@cache
def get_scrabble_trie():
    """
    Returns a (possibly cached) copy of the official
    scrabble words trie.
    Unfortunately pickling the trie is only marginally
    faster than re-parsing the corpus.
    """
    graph = Trie()
    with open("data/official_scrabble_words_2019.txt", "r") as f:
        corpus = f.read().splitlines()
    graph.parse_corpus(corpus)
    return graph


def get_stencils(row: str, anchor_points: set[int], length: int) -> list[str]:
    stencils = set()
    # Walk backwards from start_coord, counting spaces
    for anchor_point in anchor_points:
        spaces = 1
        starting_point = anchor_point
        while spaces < length and starting_point > 0:
            starting_point -= 1
            if row[starting_point] == "_":
                spaces += 1
        # Now that we have a starting point, walk forwards until we've
        # found the requisite number of spaces
        for i in range(0, anchor_point - starting_point+1):
            current_anchor = starting_point + i
            current_letter = current_anchor
            spaces = 0
            while spaces < length and current_letter < len(row):
                if row[current_letter] == "_":
                    spaces += 1
                current_letter += 1
            # If we've found the correct number of spaces, add adjacent
            # letters to either end of the stencil
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
    # Iterate over each row in the board. Add anchor points to any
    # adjacent spaces whenever a letter is found
    for i, row in enumerate(padded_board):
        for j, letter in enumerate(row):
            if letter != "_":
                if row[j-1] == "_":
                    row_anchor_points[i].add(j-1)
                if row[j+1] == "_":
                    row_anchor_points[i].add(j+1)
                if padded_board[i-1][j] == "_":
                    row_anchor_points[i-1].add(j)
                if padded_board[i+1][j] == "_":
                    row_anchor_points[i+1].add(j)
    # Remove the padding rows and columns
    row_anchor_points = row_anchor_points[1:-1]
    for i in range(len(row_anchor_points)):
        row_anchor_points[i] = set(
            map(lambda x: x-1, filter(lambda x: x > 0 and x < 15, row_anchor_points[i])))
    # Find the transpose of our sets for the column anchor points
    col_anchor_points = [set() for _ in board]
    for col in range(len(board)):
        for i, row in enumerate(row_anchor_points):
            if col in row:
                col_anchor_points[col].add(i)
    return row_anchor_points, col_anchor_points


def transpose_board(board):
    split_board = np.array([list(x) for x in board])
    transposed_board = ["".join(x) for x in split_board.T]
    return transposed_board


def get_file_words(rack, board, file, anchors, length, scoring_fn, trie, vertical=False):
    words = []
    stencils = get_stencils(board[file], anchors, length)
    for stencil, anchor in stencils:
        stencil_words = trie.traverse(rack, stencil)
        scores = [scoring_fn(word, anchor, file, vertical)
                  for word in stencil_words]
        stencil_words = map(lambda word: (word, anchor, file), stencil_words)
        words.extend(
            filter(lambda word: word[1] > 0, zip(stencil_words, scores)))
    return words


def get_playable_words(game, trie: Trie):
    rack = game.racks[game.current_player]
    horizontal_words = []
    vertical_words = []
    board = game.rows
    row_anchor_points, col_anchor_points = get_anchor_points(board)
    # Turn 0: no letters on board, need to add anchor in centre
    if game.current_player == 0 and game.scores[0] == 0:
        row_anchor_points[7].add(7)
        col_anchor_points[7].add(7)
    for length in range(2, 8):
        for row_index, anchors in enumerate(row_anchor_points):
            words = get_file_words(
                rack=rack,
                board=board,
                file=row_index,
                anchors=anchors,
                length=length,
                scoring_fn=game.get_score,
                trie=trie
            )
            horizontal_words.extend(words)
    board = game.cols
    for length in range(2, 8):
        for col_index, anchors in enumerate(col_anchor_points):
            words = get_file_words(
                rack=rack,
                board=board,
                file=col_index,
                anchors=anchors,
                length=length,
                scoring_fn=game.get_score,
                trie=trie,
                vertical=True
            )
            vertical_words.extend(words)
    return horizontal_words, vertical_words


def stringify_counter(counter: Counter):
    return "".join(map(lambda char: char[0] * char[1], counter.items()))


def get_secondary_words(word: str, loc: int, file: int, board: list[str]):
    secondary_words = []
    character_list = np.array(list(word))
    existing_chars = np.array(list(board[file][loc:loc+len(word)]))
    played_characters = np.where(
        character_list != existing_chars, character_list, "_")
    updated_board = board.copy()
    updated_board[file] = updated_board[file][:loc] + \
        word + updated_board[file][loc+len(word):]
    # Wherever we've placed a tile, walk perpendicularly to the direction
    #   of play in both directions and check for existing letters
    for i, letter in enumerate(played_characters):
        if letter != "_":
            secondary_word = ""
            anchor = loc + i
            start = file
            end = file + 1
            while start >= 0 and updated_board[start][anchor] != "_":
                secondary_word = updated_board[start][anchor] + secondary_word
                start -= 1
            while end < len(updated_board) and updated_board[end][anchor] != "_":
                secondary_word += updated_board[end][anchor]
                end += 1
            if len(secondary_word) > 1:
                secondary_words.append((secondary_word, start + 1, anchor))
    return secondary_words


def get_word_score(word, loc, file, letter_multipliers, word_multipliers, score_lookup):
    raw_tile_scores = np.array(
        list(map(lambda x: score_lookup[x], list(word))))
    letter_values = np.dot(
        raw_tile_scores, letter_multipliers[file][loc:loc+len(word)])
    score = letter_values * np.prod(
        word_multipliers[file][loc:loc+len(word)])
    return score


def get_total_score(word, loc, file, board, letter_multipliers, word_multipliers, letter_multipliers_perp, word_multipliers_perp, score_lookup):
    total_score = 0
    existing_tiles = np.array(list(board[file][loc:loc+len(word)]))
    n_played_tiles = np.sum(np.where(existing_tiles == "_", 1, 0))
    if n_played_tiles == 7:
        total_score += 50
    total_score += get_word_score(word, loc, file,
                                  letter_multipliers, word_multipliers, score_lookup)
    secondary_words = get_secondary_words(word, loc, file, board)
    for secondary_word, secondary_loc, secondary_file in secondary_words:
        total_score += get_word_score(secondary_word, secondary_loc, secondary_file,
                                      letter_multipliers_perp, word_multipliers_perp, score_lookup)
    return total_score


# ToDo: Use * to represent blanks.
# Perform 26 searches with an uppercase version of each letter
# Store uppercase scores as 0 (on board too)
if __name__ == "__main__":
    print("Reading corpus from file and parsing...")
    start = perf_counter()
    graph = Trie()
    stencil = "__"
    word = "anagram"
    with open("data/official_scrabble_words_2019.txt", "r") as f:
        corpus = f.read().splitlines()
    graph.parse_corpus(corpus)
    duration = perf_counter() - start
    print(f"Trie constructed in {duration:.2f}s.")
    # print(
    #     f"Finding all valid words with letters {word.upper()}... and stencil {stencil}")
    # timer = timeit.Timer("graph.traverse(Counter('anagram'), '_e__')", globals={
    #                      "graph": graph}, setup="from collections import Counter")
    # result = timer.repeat(3, number=500)
    # print(f"500 loops, best of 3: {min(result)/500:.2e} sec per loop")
    print(f"Results:")
    print(graph.traverse(Counter(word), stencil))
