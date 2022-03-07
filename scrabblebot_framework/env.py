import json
import pickle
import numpy as np

from .game import ScrabbleGame
from .util import get_scrabble_trie
import os


class ScrabbleEnv:
    def __init__(self, agents, corpus_file, log_file=None, random_state=None):
        module_dir = os.path.dirname(os.path.realpath(__file__))
        corpus_lookup = {
            "collins_official": f"{module_dir}/data/official_scrabble_words_2019.txt",
            "tournament_list": f"{module_dir}/data/sowpods.txt",
            "google_10k": f"{module_dir}/google-10000-english.txt"
        }
        if random_state:
            np.random.set_state(random_state)
        with open("./logs/prev_state.pickle", "wb") as f:
            pickle.dump(np.random.get_state(), f)
        with open(f"{module_dir}/data/constants.json", "r") as f:
            constants = json.loads(f.read())
        trie, corpus = get_scrabble_trie(corpus_lookup[corpus_file])
        self.game = ScrabbleGame(
            len(agents), constants, corpus, log_file=log_file)
        self.agents = [Agent(trie) for Agent in agents]
        self.game_over = False
        self.expected_score = None

    def reset(self):
        self.game.reset()

    def step(self):
        player_index = self.game.current_player
        agent = self.agents[player_index]
        played_word, score, additional_info = agent.step(self.game)
        if played_word:
            self.game_over = self.game.play(*played_word)
        else:
            self.game_over = self.game.pass_turn()
        return played_word, score, additional_info
