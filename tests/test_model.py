"""Synthetic checks; no private images, labels or pretrained weights are needed."""

import unittest

import torch

from extracted_fpc_unet import build_fpc_unet


class FPCUNetTests(unittest.TestCase):
    def test_original_checkpoint_and_active_calibration_parity(self):
        from experiment.models import build_model

        torch.manual_seed(17)
        original = build_model("fpc_unet", num_classes=4).eval()
        # Exercise both calibration branches, which start with zero residual gates.
        with torch.no_grad():
            original.fscc.scale.fill_(0.4)
            original.pglc.class_scale.fill_(0.3)
            original.pglc.class_bias.fill_(0.1)
        standalone = build_fpc_unet(num_classes=4).eval()
        standalone.load_state_dict(original.state_dict(), strict=True)
        x = torch.randn(1, 3, 32, 48)
        with torch.no_grad():
            expected = original(x)
            actual = standalone(x)
        self.assertEqual(actual.shape, (1, 4, 32, 48))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_backward_reaches_both_calibration_modules(self):
        model = build_fpc_unet(num_classes=4, base=8).train()
        with torch.no_grad():
            model.fscc.scale.fill_(0.4)
            model.pglc.class_scale.fill_(0.3)
        x = torch.randn(2, 3, 32, 32)
        labels = torch.randint(0, 4, (2, 32, 32))
        loss = torch.nn.functional.cross_entropy(model(x), labels)
        loss.backward()
        self.assertTrue(torch.isfinite(loss).item())
        for parameter in (model.fscc.projects[0][0].weight, model.pglc.prototypes):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all().item())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
