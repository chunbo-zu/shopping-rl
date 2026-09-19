"""veRL 的项目内 runtime 兼容行为。"""

import os
import sys
from types import ModuleType
import unittest
from unittest.mock import patch


class VerlCompatTest(unittest.TestCase):
    @staticmethod
    def _fake_ray(gpu_ids=("3",)):
        ray = ModuleType("ray")
        context = type(
            "RuntimeContext",
            (),
            {"get_accelerator_ids": lambda self: {"GPU": list(gpu_ids)}},
        )()
        ray.get_runtime_context = lambda: context
        return ray

    def test_pins_ray_gpu_before_verl_import_can_initialize_cuda(self):
        attention = ModuleType("verl.utils.attention_utils")
        fallback = ModuleType("verl.utils.npu_flash_attn_utils")
        expected = tuple(object() for _ in range(4))
        (
            fallback.index_first_axis,
            fallback.pad_input,
            fallback.rearrange,
            fallback.unpad_input,
        ) = expected
        utils = ModuleType("verl.utils")
        utils.attention_utils = attention
        utils.npu_flash_attn_utils = fallback
        verl = ModuleType("verl")
        verl.utils = utils
        trainer = ModuleType("verl.trainer")
        ppo = ModuleType("verl.trainer.ppo")
        ray_trainer = ModuleType("verl.trainer.ppo.ray_trainer")

        class RayPPOTrainer:
            def _update_actor(self, batch):
                return batch

        ray_trainer.RayPPOTrainer = RayPPOTrainer

        modules = {
            "ray": self._fake_ray(gpu_ids=("3",)),
            "verl": verl,
            "verl.utils": utils,
            "verl.utils.attention_utils": attention,
            "verl.utils.npu_flash_attn_utils": fallback,
            "verl.trainer": trainer,
            "verl.trainer.ppo": ppo,
            "verl.trainer.ppo.ray_trainer": ray_trainer,
        }
        with patch.dict(sys.modules, modules), patch.dict(
            os.environ,
            {
                "CUDA_VISIBLE_DEVICES": "0,1,2,3",
                "__RAY_WORKER_PROCESS_SETUP_HOOK_ENV_VAR": "compat-hook",
            },
            clear=True,
        ):
            from shopping_grpo.training.grpo.compat import install_torch_padding_fallback

            install_torch_padding_fallback()

            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "3")
            self.assertEqual(attention._get_attention_functions(), expected)

    def test_driver_preflight_keeps_cuda_visibility_unchanged(self):
        from shopping_grpo.training.grpo.compat import _pin_ray_assigned_cuda_device

        with patch.dict(
            os.environ,
            {"CUDA_VISIBLE_DEVICES": "0,1,2,3"},
            clear=True,
        ):
            _pin_ray_assigned_cuda_device()

            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "0,1,2,3")

    def test_setup_hook_isolates_compiler_caches_per_ray_worker(self):
        import tempfile
        from pathlib import Path

        from shopping_grpo.training.grpo.compat import (
            _isolate_ray_worker_compile_caches,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(
                os.environ,
                {
                    "__RAY_WORKER_PROCESS_SETUP_HOOK_ENV_VAR": "compat-hook",
                    "VLLM_CACHE_ROOT": str(root / "vllm"),
                    "TRITON_CACHE_DIR": str(root / "triton"),
                    "TORCHINDUCTOR_CACHE_DIR": str(root / "torchinductor"),
                },
                clear=True,
            ), patch("os.getpid", return_value=12345):
                _isolate_ray_worker_compile_caches()

                triton = root / "triton/ray-worker-12345"
                inductor = root / "torchinductor/ray-worker-12345"
                vllm = root / "vllm/ray-worker-12345"
                self.assertEqual(os.environ["TRITON_CACHE_DIR"], str(triton))
                self.assertEqual(
                    os.environ["TORCHINDUCTOR_CACHE_DIR"],
                    str(inductor),
                )
                self.assertEqual(os.environ["VLLM_CACHE_ROOT"], str(vllm))
                self.assertTrue(triton.is_dir())
                self.assertTrue(inductor.is_dir())
                self.assertTrue(vllm.is_dir())

                _isolate_ray_worker_compile_caches()
                self.assertEqual(os.environ["TRITON_CACHE_DIR"], str(triton))

    def test_setup_hook_uses_verl_local_rank_before_ray_attaches_gpu_ids(self):
        from shopping_grpo.training.grpo.compat import _pin_ray_assigned_cuda_device

        with patch.dict(
            sys.modules,
            {"ray": self._fake_ray(gpu_ids=())},
        ), patch.dict(
            os.environ,
            {
                "CUDA_VISIBLE_DEVICES": "GPU-a,GPU-b,GPU-c,GPU-d",
                "RANK": "6",
                "RAY_LOCAL_WORLD_SIZE": "4",
                "__RAY_WORKER_PROCESS_SETUP_HOOK_ENV_VAR": "compat-hook",
            },
            clear=True,
        ):
            _pin_ray_assigned_cuda_device()

            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "GPU-c")

    def test_setup_hook_preserves_non_contiguous_physical_gpu_mapping(self):
        from shopping_grpo.training.grpo.compat import _pin_ray_assigned_cuda_device

        with patch.dict(
            sys.modules,
            {"ray": self._fake_ray(gpu_ids=())},
        ), patch.dict(
            os.environ,
            {
                "CUDA_VISIBLE_DEVICES": "1, 3",
                "RANK": "1",
                "RAY_LOCAL_WORLD_SIZE": "2",
                "__RAY_WORKER_PROCESS_SETUP_HOOK_ENV_VAR": "compat-hook",
            },
            clear=True,
        ):
            _pin_ray_assigned_cuda_device()

            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "3")

    def test_setup_hook_without_verl_rank_metadata_keeps_visibility(self):
        from shopping_grpo.training.grpo.compat import _pin_ray_assigned_cuda_device

        with patch.dict(
            sys.modules,
            {"ray": self._fake_ray(gpu_ids=())},
        ), patch.dict(
            os.environ,
            {
                "CUDA_VISIBLE_DEVICES": "0,1,2,3",
                "__RAY_WORKER_PROCESS_SETUP_HOOK_ENV_VAR": "compat-hook",
            },
            clear=True,
        ):
            _pin_ray_assigned_cuda_device()

            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "0,1,2,3")

    def test_installs_verl_builtin_padding_functions(self):
        attention = ModuleType("verl.utils.attention_utils")
        fallback = ModuleType("verl.utils.npu_flash_attn_utils")
        expected = tuple(object() for _ in range(4))
        (
            fallback.index_first_axis,
            fallback.pad_input,
            fallback.rearrange,
            fallback.unpad_input,
        ) = expected
        utils = ModuleType("verl.utils")
        utils.attention_utils = attention
        utils.npu_flash_attn_utils = fallback
        verl = ModuleType("verl")
        verl.utils = utils
        trainer = ModuleType("verl.trainer")
        ppo = ModuleType("verl.trainer.ppo")
        ray_trainer = ModuleType("verl.trainer.ppo.ray_trainer")

        class RayPPOTrainer:
            def _update_actor(self, batch):
                return batch

        ray_trainer.RayPPOTrainer = RayPPOTrainer

        with patch.dict(
            sys.modules,
            {
                "verl": verl,
                "verl.utils": utils,
                "verl.utils.attention_utils": attention,
                "verl.utils.npu_flash_attn_utils": fallback,
                "verl.trainer": trainer,
                "verl.trainer.ppo": ppo,
                "verl.trainer.ppo.ray_trainer": ray_trainer,
            },
        ):
            from shopping_grpo.training.grpo.compat import install_torch_padding_fallback

            install_torch_padding_fallback()

        self.assertEqual(attention._get_attention_functions(), expected)
        self.assertTrue(RayPPOTrainer._update_actor._shopping_trace)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
