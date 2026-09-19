"""Configuration gate tests for the independent SAO recipe."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from scripts.check_grpo_runtime import compose_runtime_config, validate_sao


class SAOConfigTest(unittest.TestCase):
    def _config(self, *overrides):
        with patch.dict(os.environ, {"GRPO_CONFIG_NAME": "sao"}, clear=False):
            return compose_runtime_config(list(overrides))

    def test_sao_recipe_enables_single_rollout_dis_and_gae(self):
        config = self._config()
        validate_sao(config)
        self.assertTrue(config.shopping_sao.enable)
        self.assertEqual(config.actor_rollout_ref.rollout.n, 1)
        self.assertEqual(config.algorithm.adv_estimator, "gae")
        self.assertEqual(config.algorithm.rollout_correction.loss_type, "sao_dis")
        self.assertEqual(config.critic.ppo_epochs, 2)
        self.assertTrue(config.shopping_sao.freeze_critic_attention)

        # The actor worker materializes its typed config before the driver
        # applies bypass mode; keep the project loss sentinel in that config.
        from verl.utils.config import omega_conf_to_dataclass

        actor = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
        self.assertEqual(actor.policy_loss.loss_mode, "bypass_mode")
        self.assertEqual(actor.policy_loss.rollout_correction.loss_type, "sao_dis")

    def test_sao_rejects_group_rollouts(self):
        config = self._config("actor_rollout_ref.rollout.n=2")
        with self.assertRaisesRegex(SystemExit, "exactly one rollout"):
            validate_sao(config)

    def test_baseline_config_is_not_sao(self):
        with patch.dict(os.environ, {"GRPO_CONFIG_NAME": "grpo"}, clear=False):
            config = compose_runtime_config([])
        validate_sao(config)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
