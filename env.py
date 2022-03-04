import json
import pickle
from pprint import pprint
from time import perf_counter
import numpy as np
from tqdm import trange

from game import ScrabbleGame
from agents.greedy import GreedyAgent
from agents.longestWord import LongestWordAgent
from agents.stochasticLookahead import StochasticLookaheadAgent
from util import get_scrabble_trie, stringify_counter


class ScrabbleEnv:
    def __init__(self, agents, corpus_file, random_state=None):
        if random_state:
            np.random.set_state(random_state)
        with open("./logs/prev_state.pickle", "wb") as f:
            pickle.dump(np.random.get_state(), f)
        with open("./data/constants.json", "r") as f:
            constants = json.loads(f.read())
        with open(corpus_file, "r") as f:
            corpus = f.read().splitlines()
        trie = get_scrabble_trie(corpus_file)
        self.game = ScrabbleGame(len(agents), constants, corpus)
        self.agents = [Agent(trie) for Agent in agents]
        self.game_over = False

    def reset(self):
        self.game.reset()

    def step(self):
        start = perf_counter()
        player_index = self.game.current_player
        agent = self.agents[player_index]
        # print(agent.__class__.__name__)
        # Print rack as str
        # print(f"Rack: {stringify_counter(self.game.racks[player_index])}")
        played_word, score = agent.step(self.game)
        # print(played_word, score)
        if played_word:
            self.game_over = self.game.play(*played_word)
        else:
            self.game_over = self.game.pass_turn()
        end = perf_counter()
        return end - start


if __name__ == "__main__":
    # with open("./logs/prev_state.pickle", "rb") as f:
    #     state = pickle.load(f)
    corpus_file = "data/official_scrabble_words_2019.txt"

    # state = np.random.get_state()
    turn_times = []

    for repeat in trange(10):
        env = ScrabbleEnv([GreedyAgent, GreedyAgent],
                          corpus_file)
        while not env.game_over:
            t = env.step()
            turn_times.append(t)
        print(env.game.scores)
    print(f"Times after {len(turn_times)} turns:")
    print(f"Min: {np.min(turn_times)}\tMax: {np.max(turn_times)}\tAvg: {np.mean(turn_times)}\tStd:{np.std(turn_times)}")
