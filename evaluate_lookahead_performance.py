from env import ScrabbleEnv
from agents.greedy import GreedyAgent
from agents.stochasticLookahead import StochasticLookaheadAgent
from time import time

if __name__ == "__main__":
    corpus_file = "data/official_scrabble_words_2019.txt"

    def stochasticAgent(x): return StochasticLookaheadAgent(
        x, n_candidates=4, n_samples=100)

    turn_times = []

    with open("./game.log", "w") as log_file:
        env = ScrabbleEnv([GreedyAgent, stochasticAgent],
                          corpus_file, log_file)
        while not env.game_over:
            t = env.step()
            turn_times.append(t)
            print(f"Thinking time: {t:.2f}s")
    print(env.game.scores)
