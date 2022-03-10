from matplotlib import pyplot as plt
import pickle
from pprint import pprint
from time import perf_counter

import numpy as np
from scrabblebot_framework.agents.greedy import GreedyAgent
from scrabblebot_framework.agents.tileEfficiency import TileEfficiencyAgent
from scrabblebot_framework.env import ScrabbleEnv
if __name__ == "__main__":
    with open("logs/prev_state.pickle", "rb") as f:
        random_state = pickle.load(f)
    env = ScrabbleEnv(agents=[GreedyAgent, TileEfficiencyAgent],
                      corpus_file="tournament_official")
    record, stds = env.simulate(verbose=1)
    plt.plot(record)
    plt.title("Winrate")
    plt.figure()
    plt.title("Winrate deviation, last 10 games")
    plt.plot(stds)
    plt.show()
