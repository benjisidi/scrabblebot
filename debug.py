from time import perf_counter
from agents.greedy import GreedyAgent
from env import ScrabbleEnv
if __name__ == "__main__":
    env = ScrabbleEnv(agents=[GreedyAgent, GreedyAgent],
                      corpus_file="./data/official_scrabble_words_2019.txt")
    start = perf_counter()
    racks = env.game.generate_ghost_racks(100, 0)
    end = perf_counter()
    print(racks)
    print(f"{end - start:.2f}s")
