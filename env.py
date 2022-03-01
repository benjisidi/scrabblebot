from itertools import count
import os
from collections import Counter
import json

import numpy as np
from player import Agent
from game import ScrabbleGame
from util import get_anchor_points, get_scrabble_trie


def get_rack(str):
    rack = {}
    for char in str:
        if char in rack:
            rack[char] += 1
        else:
            rack[char] = 1


class ScrabbleEnv:
    def __init__(self, n_players, trie):
        with open("./data/constants.json", "r") as f:
            constants = json.loads(f.read())
        with open("./data/official_scrabble_words_2019.txt", "r") as f:
            corpus = f.read().splitlines()
        trie = trie
        self.game = ScrabbleGame(n_players, constants, corpus)
        starting_racks = [Counter(self.game.draw_letters(7))
                          for _ in range(n_players)]
        self.agents = [Agent(starting_racks[i], trie)
                       for i in range(n_players)]
        self.agents_passed = [False for _ in self.agents]
        self.game_over = False

    def step(self):
        agent = self.agents[self.game.current_player]
        played_word, score, vertical, length = agent.step(self.game)
        print(played_word, score)
        if played_word is None:
            print("No words found")
            self.agents_passed[self.game.current_player] = True
            self.game.current_player = (
                self.game.current_player + 1) % self.game.n_players
            if np.all(self.agents_passed):
                self.game_over = True
                print("Game over: stalemate")
            return False, 0
        else:
            self.game.play(*played_word, vertical=vertical)
            self.agents_passed = [False for _ in self.agents]
            agent.update_rack(self.game.draw_letters(length))
            if len(agent.rack) == 0:
                print("Game over: player emptied rack")
                self.game_over = True
                return False, 0
            return played_word, score


if __name__ == "__main__":
    trie = get_scrabble_trie()
    env = ScrabbleEnv(2, trie)
    while not env.game_over:
        word, score = env.step()
        env.game.show()
