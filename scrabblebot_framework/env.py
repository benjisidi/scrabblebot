import json
import pickle
from pprint import pprint
import numpy as np

from .game import ScrabbleGame
from .utils.util import get_scrabble_trie
import os


class ScrabbleEnv:
    def __init__(self, agents, corpus_file, log_file=None, random_state=None):
        module_dir = os.path.dirname(os.path.realpath(__file__))
        corpus_lookup = {
            "collins_official": f"{module_dir}/data/official_scrabble_words_2019.txt",
            "tournament_official": f"{module_dir}/data/sowpods.txt",
            "google_10k": f"{module_dir}/google-10000-english.txt"
        }
        if random_state:
            np.random.set_state(random_state)
        with open("./logs/prev_state.pickle", "wb") as f:
            pickle.dump(np.random.get_state(), f)
        with open(f"{module_dir}/data/constants.json", "r") as f:
            constants = json.loads(f.read())
        trie, corpus = get_scrabble_trie(corpus_lookup[corpus_file])
        self.trie = trie
        self.game = ScrabbleGame(
            len(agents), constants, corpus, log_file=log_file)
        self.agents = [Agent(trie) for Agent in agents]
        self.game_over = False
        self.constants = constants

    def reset(self):
        self.game.reset(self.constants)
        self.game_over = False

    def step(self):
        player_index = self.game.current_player
        agent = self.agents[player_index]
        played_word, score, additional_info = agent.step(self.game)
        if played_word:
            self.game_over = self.game.play(*played_word)
        else:
            self.game_over = self.game.pass_turn()
        return played_word, score, additional_info

    def simulate(self, verbose=0):
        wins = 0
        record = []
        std = 1
        i = 0
        while std > 0.002:
            if i > 10:
                std = np.std(record[i-10:i])
            self.reset()
            while not self.game_over:
                if verbose > 1:
                    print(
                        f"==={self.agents[self.game.current_player].__class__.__name__}===")
                word, score, info = self.step()
                if verbose > 1:
                    print(word, score)
                    pprint(info)
                if verbose > 2:
                    self.game.show()
            if self.game.scores[0] > self.game.scores[1]:
                wins += 1
            i += 1
            if verbose:
                print(f"{self.game.scores}\t{wins/(i+1): .2f}\t{std:.4f}\t{i}")
            record.append(wins/(i+1))
        record = np.array(record)
        stds = [np.std(record[max(0, i-10):i+1])
                for i in range(1, len(record))]
        return record, stds
