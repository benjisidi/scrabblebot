from collections import Counter
import string
from ..utils.util import get_anchor_allowed_chars, get_anchor_points
from ..env import ScrabbleEnv
from ..agents.dummy import DummyAgent


def test_anchor_points():
    test_env = ScrabbleEnv(
        agents=[DummyAgent], corpus_file="tournament_official")
    test_env.game.rows[7] = "_______a_______"
    anchor_points = get_anchor_points(test_env.game.rows)
    assert(anchor_points == (
        [set(), set(), set(), set(), set(), set(), {7}, {6, 8}, {
            7}, set(), set(), set(), set(), set(), set()],
        [set(), set(), set(), set(), set(), set(), {7}, {6, 8}, {
            7}, set(), set(), set(), set(), set(), set()],
    ))

    test_env.reset()
    test_env.game.rows[7] = "_____________b_"
    anchor_points = get_anchor_points(test_env.game.rows)
    assert(anchor_points == (
        [set(), set(), set(), set(), set(), set(), {13}, {12, 14}, {
            13}, set(), set(), set(), set(), set(), set()],
        [set(), set(), set(), set(), set(), set(), set(), set(),
         set(), set(), set(), set(), {7}, {6, 8}, {7}],
    ))


def test_anchor_allowed_chars():
    test_env = ScrabbleEnv(
        agents=[DummyAgent], corpus_file="tournament_official")
    test_env.game.rows[7] = "_______b_______"
    anchor_points = get_anchor_points(test_env.game.rows)
    anchor_allowed_chars = get_anchor_allowed_chars(
        test_env.game.rows, anchor_points, Counter(string.ascii_lowercase), test_env.trie)
    assert(anchor_allowed_chars == [
        {
            (6, 7): {"a", "o"},
            (8, 7): {"a", "e", "i", "o", "y"},
        },
        {
            (6, 7): {"a", "o"},
            (8, 7): {"a", "e", "i", "o", "y"},

        }
    ])

    test_env.reset()
    test_env.game.rows[7] = "_____________b_"
    test_env.game.cols[13] = "_______b_______"
    anchor_points = get_anchor_points(test_env.game.rows)
    anchor_allowed_chars = get_anchor_allowed_chars(
        test_env.game.rows, anchor_points, Counter(string.ascii_lowercase), test_env.trie)
    assert(anchor_allowed_chars == [
        {
            (6, 13): {"a", "o"},
            (8, 13): {"a", "e", "i", "o", "y"},
        },
        {
            (12, 7): {"a", "o"},
            (14, 7): {"a", "e", "i", "o", "y"},

        }
    ])
