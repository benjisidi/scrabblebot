import string
from collections import Counter, deque


def drop_member(charset: dict, item: str):
    """
    Return a copy of the provided dict with the item decremented by 1
    The only built-in ways to do this are in-place, which prevents us
    from using list comprehension in Vertex's visit method.
    """
    new_set = charset.copy()
    if new_set[item] == 1:
        del new_set[item]
    else:
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
    def visit(self, allowed_transitions: dict, cur_path: str, stencil: str):
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
            next_nodes = []
            for label, vertex in self.edges.items():
                if label in allowed_transitions and allowed_transitions[label] > 0:
                    next_nodes.append((vertex, drop_member(
                        allowed_transitions, label), cur_path + label))
            # If we have a blank, we want to try using it in every possible position
            # even those we have letters for, as using those letters later may net
            # us more points due to board modifiers
            if "*" in allowed_transitions and allowed_transitions["*"] > 0:
                for label, vertex in self.edges.items():
                    next_nodes.append((vertex, drop_member(
                        allowed_transitions, "*"), cur_path + label.upper()))
            return next_nodes

    def visit_restricted(self, allowed_transitions: dict, cur_path: str, stencil: list[set]):
        """
        Collect all the edges that match the allowed transitions.
        Returns (new_vertex, remaining_transitions, updated_path) for each.
        """
        if len(self.edges) == 0 or self.depth >= len(stencil) - 1:
            return []
        next_stencil_point = stencil[self.depth + 1]
        edgeset = set(self.edges.keys())
        transitionset = set(allowed_transitions.keys())
        if isinstance(next_stencil_point, str):
            if next_stencil_point == "_":
                available_chars = transitionset & edgeset
            elif next_stencil_point not in self.edges:
                return []
            else:
                return [(self.edges[next_stencil_point], allowed_transitions, cur_path + next_stencil_point)]
        else:
            available_chars = transitionset & next_stencil_point & edgeset
        if len(available_chars) == 0:
            return []
        next_nodes = []
        for label in available_chars:
            next_nodes.append((self.edges[label], drop_member(
                allowed_transitions, label), cur_path + label))
        # If we have a blank, we want to try using it in every possible position
        # even those we have letters for, as using those letters later may net
        # us more points due to board modifiers
        if "*" in allowed_transitions and allowed_transitions["*"] > 0:
            for label in available_chars:
                next_nodes.append((self.edges[label], drop_member(
                    allowed_transitions, "*"), cur_path + label.upper()))
        return next_nodes

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
        charset = dict(charset)
        output = set()
        if stencil[0] == "_":
            forced_start = False
            if "*" in charset:
                starting_charset = charset.copy()
                del starting_charset["*"]
                starting_letters = "".join(
                    starting_charset.keys()) + string.ascii_uppercase
            else:
                starting_letters = charset.keys()
        else:
            forced_start = True
            starting_letters = [stencil[0]]
        for char in starting_letters:
            if not forced_start:
                if char.isupper():
                    remaining_chars = drop_member(charset, "*")
                else:
                    remaining_chars = drop_member(charset, char)
            else:
                remaining_chars = charset
            cur_vertex = self.roots[char.lower()]
            cur_path = char
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
        return list(output)


class LengthAwareTrie:
    def __init__(self):
        # Vertices function like a linked list, so we only need to store
        # the roots of the tree and we can traverse it from there.
        self.roots = {}

    def add_word(self, word: str) -> None:
        """
        Traverses existing Trie as far as possible, then adds new vertices for remaining letters.
        Marks final letter as is_leaf.
        """
        length = len(word)
        # Add the first letter as a root if it doesn't already exist
        if length not in self.roots:
            self.roots[length] = {}
        if word[0] not in self.roots[length]:
            self.roots[length][word[0]] = Vertex(word[0], len(word) == 1, 0)
        cur_vertex = self.roots[length][word[0]]
        # Starting from the 2nd letter, traverse the Trie, adding
        # new vertices where appropriate
        for i, char in enumerate(word[1:]):
            if char in cur_vertex.edges:
                cur_vertex = cur_vertex.edges[char]
            else:
                cur_vertex.edges[char] = Vertex(
                    char, i == len(word)-2, i+1)
                cur_vertex = cur_vertex.edges[char]

    def parse_corpus(self, corpus):
        for word in corpus:
            self.add_word(word)

    # @profile
    def traverse(self, charset: Counter, stencil: list[set]) -> list[str]:
        """
        This is basically BFS for a trie. Iterates over the charset,
        taking each character in turn as a root, and visits all allowed
        transitions, noting whenever it finds a leaf.
        Final list of words is simply all the leaves.
        """
        charset = dict(charset)
        output = set()
        length = len(stencil)
        forced_start = False
        if stencil[0] == "_":
            if "*" in charset:
                starting_charset = charset.copy()
                del starting_charset["*"]
                starting_letters = "".join(
                    starting_charset.keys()) + string.ascii_uppercase
            else:
                starting_letters = charset.keys()
        elif isinstance(stencil[0], frozenset):
            if "*" in charset:
                starting_letters = set(x.upper() for x in stencil[0]) | (
                    set(charset.keys()) & stencil[0])
            else:
                starting_letters = stencil[0]
        else:
            forced_start = True
            starting_letters = [stencil[0]]
        for char in starting_letters:
            if char.lower() not in self.roots[length]:
                continue
            if not forced_start:
                if char.isupper():
                    remaining_chars = drop_member(charset, "*")
                else:
                    remaining_chars = drop_member(charset, char)
            else:
                remaining_chars = charset
            cur_vertex = self.roots[length][char.lower()]
            cur_path = char
            queue = deque(cur_vertex.visit_restricted(
                remaining_chars, cur_path, stencil))
            if cur_vertex.is_leaf and cur_vertex.depth == len(stencil) - 1:
                output.add(cur_path)
            while len(queue) > 0:
                cur_vertex, remaining_chars, cur_path = queue.pop()
                if cur_vertex.is_leaf and cur_vertex.depth == len(stencil) - 1:
                    output.add(cur_path)
                new_queue = cur_vertex.visit_restricted(
                    remaining_chars, cur_path, stencil)
                queue.extend(new_queue)
        return list(output)

    def get_allowed_chars(self, charset: Counter, stencil: str) -> set[str]:
        letter_loc = stencil.find("_")
        words = self.traverse(charset, stencil)
        allowed_letters = set([x[letter_loc]
                              for x in words if charset[x[letter_loc]] > 0])
        return allowed_letters
