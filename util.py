import timeit
from collections import Counter, deque
from time import perf_counter


def drop_member(charset: Counter, item: str):
    """
    Return a copy of the provided Counter with the item decremented by 1
    The only built-in ways to do this are in-place, which prevents us 
    from using list comprehension in Vertex's visit method.
    """
    if charset[item] == 0:
        raise KeyError(f"Item {item} not in charset {charset}")
    new_set = charset.copy()
    new_set[item] -= 1
    return new_set


class Vertex:
    def __init__(self, label: str, is_leaf: bool, depth: int):
        self.label = label
        self.edges: dict[str, Vertex] = {}
        self.is_leaf = is_leaf
        self.depth = depth

    def add_edge(self, vertex: str) -> None:
        self.edges = self.edges[vertex.label] = vertex

    # @profile
    def visit(self, allowed_transitions: Counter, cur_path: str, stencil: str):
        """
        Collect all the edges that match the allowed transitions.
        Returns (new_vertex, remaining_transitions, updated_path) for each.
        """
        if len(self.edges) == 0 or self.depth >= len(stencil) - 1:
            return []
        next_char = stencil[self.depth + 1]
        if next_char != "_":
            if next_char not in self.edges:
                return []
            else:
                return [(self.edges[next_char], allowed_transitions, cur_path + next_char)]
        else:
            return [
                (vertex, drop_member(allowed_transitions, label), cur_path + label)
                for label, vertex in self.edges.items() if allowed_transitions[label] > 0
            ]

    def __repr__(self):
        return f"label: {self.label}, is_leaf: {self.is_leaf}"


class Trie:
    def __init__(self):
        # Vertices function like a linked list, so we only need to store
        # the roots of the tree and we can traverse it from there.
        self.roots = {}

    def add_word(self, word: str) -> None:
        """
        Traverses existing Trie as far as possible, then adds new vertices for remaining letters.
        Marks final letter as is_leaf.
        """
        # Add the first letter as a root if it doesn't already exist
        if word[0] not in self.roots:
            self.roots[word[0]] = Vertex(word[0], len(word) == 1, 0)
        cur_vertex = self.roots[word[0]]
        # Starting from the 2nd letter, traverse the Trie, adding
        # new vertices where appropriate
        for i, char in enumerate(word[1:]):
            if char in cur_vertex.edges:
                cur_vertex = cur_vertex.edges[char]
            else:
                cur_vertex.edges[char] = Vertex(char, i == len(word)-2, i+1)
                cur_vertex = cur_vertex.edges[char]

    def parse_corpus(self, corpus):
        for word in corpus:
            self.add_word(word)

    # @profile
    def traverse(self, charset: Counter, stencil: str) -> list[str]:
        """
        This is basically BFS for a trie. Iterates over the charset,
        taking each character in turn as a root, and visits all allowed
        transitions, noting whenever it finds a leaf.
        Final list of words is simply all the leaves.
        """
        output = set()
        for char in charset.keys():
            remaining_chars = drop_member(charset, char)
            cur_vertex = self.roots[char]
            cur_path = cur_vertex.label
            queue = deque(cur_vertex.visit(remaining_chars, cur_path, stencil))
            if cur_vertex.is_leaf and cur_vertex.depth == len(stencil) - 1:
                output.add(cur_path)
            while len(queue) > 0:
                cur_vertex, remaining_chars, cur_path = queue.pop()
                if cur_vertex.is_leaf and cur_vertex.depth == len(stencil) - 1:
                    output.add(cur_path)
                new_queue = cur_vertex.visit(
                    remaining_chars, cur_path, stencil)
                queue.extend(new_queue)
        return output

# ToDo: Use * to represent blanks.
# Perform 26 searches with an uppercase version of each letter
# Store uppercase scores as 0 (on board too)
if __name__ == "__main__":
    print("Reading corpus from file and parsing...")
    start = perf_counter()
    graph = Trie()
    stencil = "_e__"
    word = "anagram"
    with open("data/official_scrabble_words_2019.txt", "r") as f:
        corpus = f.read().splitlines()
    graph.parse_corpus(corpus)
    duration = perf_counter() - start
    print(f"Trie constructed in {duration:.2f}s.")
    print(
        f"Finding all valid words with letters {word.upper()}... and stencil {stencil}")
    timer = timeit.Timer("graph.traverse(Counter('anagram'), '_e__')", globals={
                         "graph": graph}, setup="from collections import Counter")
    result = timer.repeat(3, number=500)
    print(f"500 loops, best of 3: {min(result)/500:.2e} sec per loop")
    print(f"Results:")
    print(graph.traverse(Counter(word), stencil))
