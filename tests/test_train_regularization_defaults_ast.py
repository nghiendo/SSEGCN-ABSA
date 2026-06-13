import ast
import pathlib
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE_PATH = PROJECT_ROOT / 'train.py'


class TestTrainRegularizationDefaults(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SOURCE_PATH.read_text(encoding='utf-8')
        cls.tree = ast.parse(cls.source)
        cls.defaults = {}

        for node in ast.walk(cls.tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Attribute) and func.attr == 'add_argument'):
                continue
            if not node.args or not isinstance(node.args[0], ast.Constant):
                continue
            option = node.args[0].value
            for kw in node.keywords:
                if kw.arg == 'default' and isinstance(kw.value, ast.Constant):
                    cls.defaults[option] = kw.value.value

    def test_weight_decay_defaults_are_stronger(self):
        self.assertEqual(self.defaults['--l2reg'], 5e-4)
        self.assertEqual(self.defaults['--weight_decay'], 0.01)

    def test_dropout_defaults_are_stronger(self):
        self.assertEqual(self.defaults['--gcn_dropout'], 0.2)
        self.assertEqual(self.defaults['--rnn_dropout'], 0.2)
        self.assertEqual(self.defaults['--sg_dropout'], 0.35)
        self.assertEqual(self.defaults['--bert_dropout'], 0.4)

    def test_auxiliary_regularizer_defaults_are_stronger(self):
        self.assertEqual(self.defaults['--sg_branch_weight'], 0.05)
        self.assertEqual(self.defaults['--sg_diversity_weight'], 0.1)
        self.assertEqual(self.defaults['--sg_jepa_weight'], 0.15)


if __name__ == '__main__':
    unittest.main()
