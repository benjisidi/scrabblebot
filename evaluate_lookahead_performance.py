import numpy as np
from env import ScrabbleEnv
from agents.greedy import GreedyAgent
from agents.stochasticLookahead import StochasticLookaheadAgent
from time import time

if __name__ == "__main__":
    corpus_file = "data/official_scrabble_words_2019.txt"

    def stochasticAgent(x): return StochasticLookaheadAgent(
        x, n_candidates=4, n_samples=100)

    score_diffs = []
    scores = []
    for i in range(1):
        env = ScrabbleEnv([stochasticAgent, GreedyAgent],
                          corpus_file)
        it_diffs = []
        while not env.game_over:
            turn = env.step()
            if env.game.current_player == 1:
                it_diffs.append(
                    env.agents[0].expected_score - (env.agents[0].prev_score - turn[1]))
        print(
            f"Mean diff: {np.mean(it_diffs):.2f}\tStd diff: {np.std(it_diffs):.2f}")
        print(f"Score: {env.game.scores}")
        scores.append(env.game.scores)
        score_diffs.append(it_diffs)
    np.save("./score_diffs.npy", score_diffs)
    np.save("./scores.npy", scores)
