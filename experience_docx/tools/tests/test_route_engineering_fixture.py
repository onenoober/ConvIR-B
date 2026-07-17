"""Tests for reusable real-structure engineering fixture assertions."""

import sys
import unittest
from pathlib import Path


TOOLS = Path(__file__).parents[1]
sys.path.insert(0, str(TOOLS))
import route_engineering_fixture as FIXTURE  # noqa: E402


class RouteEngineeringFixtureTests(unittest.TestCase):
    def test_real_torch_path_scope_noop_gradient_and_microfit(self):
        import torch

        torch.manual_seed(0)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.base = torch.nn.Linear(2, 2)
                self.adapter = torch.nn.Linear(2, 2, bias=False)
                self.base.requires_grad_(False)
                torch.nn.init.ones_(self.base.weight)
                torch.nn.init.ones_(self.base.bias)
                torch.nn.init.zeros_(self.adapter.weight)

            def forward(self, value):
                base = self.base(value)
                return base, base + self.adapter(value)

        model = Model()
        scope = FIXTURE.assert_trainable_scope(
            model, allowed_prefixes=("adapter",), required_prefixes=("adapter",),
        )
        self.assertEqual(4, scope["trainable_parameter_count"])
        inputs = torch.ones(2, 2)
        reference, candidate = model(inputs)
        FIXTURE.assert_noop(reference, candidate)
        loss = candidate.square().mean()
        FIXTURE.assert_finite_tensors((("candidate", candidate), ("loss", loss)))
        loss.backward()
        gradient = FIXTURE.assert_nonzero_gradients(
            model, required_prefixes=("adapter",),
        )
        self.assertGreater(gradient["gradient_max_abs"], 0)
        result = FIXTURE.assert_loss_decreased(1.0, 0.5, minimum_relative_decrease=0.1)
        self.assertEqual(0.5, result["microfit_relative_decrease"])

    def test_unexpected_trainable_parameter_is_rejected(self):
        import torch

        model = torch.nn.Sequential(torch.nn.Linear(2, 2))
        with self.assertRaises(FIXTURE.FixtureError):
            FIXTURE.assert_trainable_scope(model, allowed_prefixes=("adapter",))


if __name__ == "__main__":
    unittest.main()
