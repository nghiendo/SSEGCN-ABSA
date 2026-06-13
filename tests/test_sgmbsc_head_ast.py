import ast
import pathlib
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE_PATH = PROJECT_ROOT / 'models' / 'sg_mbsc_absa.py'


class TestSGMBSCAstGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SOURCE_PATH.read_text(encoding='utf-8')
        cls.tree = ast.parse(cls.source)
        cls.head_class = next(
            node for node in cls.tree.body if isinstance(node, ast.ClassDef) and node.name == 'SGMBSCHead'
        )

    def _method(self, name):
        return next(
            node for node in self.head_class.body if isinstance(node, ast.FunctionDef) and node.name == name
        )

    def test_has_diversity_loss_method(self):
        self._method('_diversity_loss')

    def test_has_adaptive_fusion_layer(self):
        self.assertIn('self.logit_fusion', self.source)

    def test_has_jepa_target_and_predictor_layers(self):
        self.assertIn('self.jepa_target_projector', self.source)
        self.assertIn('self.jepa_predictor', self.source)

    def test_has_jepa_loss_method(self):
        self._method('_jepa_loss')

    def test_auxiliary_loss_tracks_diversity_and_jepa_metrics(self):
        method = self._method('_auxiliary_loss')
        segment = ast.get_source_segment(self.source, method) or ''
        self.assertIn('diversity', segment)
        self.assertIn('jepa', segment)

    def test_forward_logs_only_shared_and_expert_fusion_weights(self):
        method = self._method('forward')
        segment = ast.get_source_segment(self.source, method) or ''
        self.assertIn('fusion_weight_shared', segment)
        self.assertIn('fusion_weight_expert', segment)
        self.assertNotIn('fusion_weight_base', segment)
        self.assertIn('predicted_latent', segment)

    def test_head_has_no_base_classifier_or_three_way_fusion(self):
        self.assertNotIn('self.base_classifier', self.source)
        self.assertNotIn('self.base_weight', self.source)
        self.assertNotIn('fusion_weights[:, 2:3]', self.source)


if __name__ == '__main__':
    unittest.main()
