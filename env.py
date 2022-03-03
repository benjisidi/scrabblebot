import json
import pickle
from random import random

import numpy as np
from player import GreedyAgent
from game import ScrabbleGame
from util import get_scrabble_trie, stringify_counter


class ScrabbleEnv:
    def __init__(self, agents, trie, random_state=None):
        if random_state:
            np.random.set_state(random_state)
        else:
            with open("./logs/prev_state.pickle", "wb") as f:
                pickle.dump(np.random.get_state(), f)
        with open("./data/constants.json", "r") as f:
            constants = json.loads(f.read())
        with open("./data/official_scrabble_words_2019.txt", "r") as f:
            corpus = f.read().splitlines()
        self.game = ScrabbleGame(len(agents), constants, corpus)
        self.agents = [Agent(trie) for Agent in agents]
        self.game_over = False

    def reset(self):
        self.game.reset()

    def step(self):
        player_index = self.game.current_player
        agent = self.agents[player_index]
        # Print rack as str
        print(stringify_counter(self.game.racks[player_index]))
        played_word, score, vertical = agent.step(self.game)
        print(played_word, score)
        if played_word:
            self.game_over = self.game.play(*played_word, vertical=vertical)
        else:
            self.game_over = self.game.pass_turn()


if __name__ == "__main__":
    # with open("./logs/prev_state.pickle", "rb") as f:
    #     state = pickle.load(f)
    trie = get_scrabble_trie()
    env = ScrabbleEnv([GreedyAgent, GreedyAgent], trie)
    while not env.game_over:
        env.step()
        env.game.show()
