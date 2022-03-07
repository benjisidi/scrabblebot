import numpy as np
from scrabblebot_framework.env import ScrabbleEnv
from scrabblebot_framework.agents.greedy import GreedyAgent
from scrabblebot_framework.agents.stochasticLookahead import StochasticLookaheadAgent


def evaluate_lookhead_performance():
    corpus_file = "collins_official"

    def stochasticAgent(x): return StochasticLookaheadAgent(
        x, n_candidates=4, n_samples=100)

    score_diffs = []
    scores = []
    expected_score = None
    prev_score = None
    for _ in range(2):
        env = ScrabbleEnv([stochasticAgent, GreedyAgent],
                          corpus_file)
        it_diffs = []
        while not env.game_over:
            word, score, info = env.step()
            if word:
                if env.game.current_player == 0:
                    expected_score = info["expected_score"]
                    prev_score = score
                elif env.game.current_player == 1:
                    it_diffs.append(
                        expected_score - (prev_score - score))
        print(
            f"Mean diff: {np.mean(it_diffs):.2f}\tStd diff: {np.std(it_diffs):.2f}")
        print(f"Score: {env.game.scores}")
        scores.append(env.game.scores)
        score_diffs.append(it_diffs)
    np.save("./score_diffs.npy", score_diffs, allow_pickle=True)
    np.save("./scores.npy", scores)


if __name__ == "__main__":
    evaluate_lookhead_performance()
