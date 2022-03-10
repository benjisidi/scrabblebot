import pickle
from time import perf_counter
from scrabblebot_framework.agents.greedy import GreedyAgent
from scrabblebot_framework.agents.tileEfficiency import TileEfficiencyAgent
from scrabblebot_framework.env import ScrabbleEnv
if __name__ == "__main__":
    with open("logs/prev_state.pickle", "rb") as f:
        random_state = pickle.load(f)
    env = ScrabbleEnv(agents=[TileEfficiencyAgent, GreedyAgent],
                      corpus_file="tournament_official", random_state=random_state)
    while not env.game_over:
        env.step()
