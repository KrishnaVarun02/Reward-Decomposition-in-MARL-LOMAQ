import json
import tempfile
import unittest
from pathlib import Path

from marl.config import load_config
from marl.lomaq import LOMAQ


class LomaqTests(unittest.TestCase):
    def test_partition_validation(self):
        with self.assertRaises(ValueError):
            LOMAQ(2, 2, [[0, 0]])

    def test_local_action_and_partition_update(self):
        learner = LOMAQ(2, 2, [[0], [1]], learning_rate=0.5, epsilon=0)
        state = ("a", "b")
        learner.update(state, (1, 0), [2, -1], state, done=True)
        self.assertEqual(learner.act(state, explore=False), (1, 1))
        self.assertGreater(learner.partition_value(0, state, (1, 0)), 0)
        self.assertLess(learner.partition_value(1, state, (1, 0)), 0)

    def test_monotonic_projection(self):
        learner = LOMAQ(1, 2, [[0]], learning_rate=1, epsilon=0)
        learner.utilities[0, "s"][0] = 10
        learner.update(("s",), (0,), [-100], ("s",), done=True)
        self.assertGreaterEqual(learner.weights[0][0], 0)

    def test_test_config_takes_precedence(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = []
            for i, layer in enumerate(({"a": {"x": 1, "y": 1}},
                                       {"a": {"x": 2}},
                                       {"a": {"y": 3}},
                                       {"a": {"x": 4}})):
                path = Path(folder) / f"{i}.json"
                path.write_text(json.dumps(layer))
                paths.append(path)
            self.assertEqual(load_config(*paths), {"a": {"x": 4, "y": 3}})


if __name__ == "__main__":
    unittest.main()
