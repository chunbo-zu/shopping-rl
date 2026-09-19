"""veRL 0.8 的窄范围运行时兼容。"""


def _isolate_ray_worker_compile_caches():
    """Prevent concurrent Ray workers from racing on compiler cache files.

    vLLM starts several Ray actors at once. If they share one Triton cache,
    TorchInductor autotuning can observe another process's partially-published
    kernel metadata and fail with a transient ``FileNotFoundError``. A
    per-worker directory keeps the caches on the configured data volume while
    removing that cross-process race. EngineCore children inherit the
    directory selected by their parent Ray worker.
    """
    import os
    from pathlib import Path

    if "__RAY_WORKER_PROCESS_SETUP_HOOK_ENV_VAR" not in os.environ:
        return

    worker_directory = f"ray-worker-{os.getpid()}"
    for variable in (
        "VLLM_CACHE_ROOT",
        "TRITON_CACHE_DIR",
        "TORCHINDUCTOR_CACHE_DIR",
    ):
        configured = os.environ.get(variable)
        if not configured:
            continue
        path = Path(configured).expanduser().absolute()
        if path.name != worker_directory:
            path /= worker_directory
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)


def _pin_ray_assigned_cuda_device():
    """Apply Ray's GPU assignment before importing PyTorch through veRL.

    Ray invokes ``worker_process_setup_hook`` before it applies the actor's
    accelerator visibility.  Importing the veRL trainer from that hook imports
    PyTorch and lets CUDA cache the driver's full device list.  Later changes to
    ``CUDA_VISIBLE_DEVICES`` then arrive too late, so every distributed rank uses
    physical GPU 0.  Restrict visibility from Ray's already-known assignment
    before any veRL/PyTorch import can initialize CUDA.
    """
    import os

    # ``ray.is_initialized()`` is false while the process setup hook runs even
    # though the worker is already connected and has resource IDs.  The
    # reserved hook variable distinguishes that phase from the preflight's
    # ordinary, pre-ray invocation without accidentally starting local Ray.
    if "__RAY_WORKER_PROCESS_SETUP_HOOK_ENV_VAR" not in os.environ:
        return

    import ray

    gpu_ids = ray.get_runtime_context().get_accelerator_ids().get("GPU", [])
    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in gpu_ids)
        return

    # Ray executes the process hook while preparing the runtime environment,
    # before the actor's resource IDs are attached to the worker.  veRL does,
    # however, put the rank topology into that runtime environment, and its
    # placement-group bundle ``local_rank`` is the corresponding GPU index.
    rank = os.environ.get("RANK")
    local_world_size = os.environ.get("RAY_LOCAL_WORLD_SIZE")
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if rank is None or local_world_size is None or not visible_devices:
        return
    local_size = int(local_world_size)
    if local_size <= 0:
        raise RuntimeError("RAY_LOCAL_WORLD_SIZE must be positive")
    devices = [device.strip() for device in visible_devices.split(",")]
    if not all(devices):
        raise RuntimeError(f"invalid CUDA_VISIBLE_DEVICES: {visible_devices!r}")
    local_rank = int(rank) % local_size
    if local_rank >= len(devices):
        raise RuntimeError(
            "veRL rank topology exceeds CUDA_VISIBLE_DEVICES: "
            f"local_rank={local_rank}, visible_devices={visible_devices}"
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = devices[local_rank]


def _install_trace_actor_update():
    """Score private gold targets before the existing veRL actor update."""
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    if getattr(RayPPOTrainer._update_actor, "_shopping_trace", False):
        return

    from shopping_grpo.training.grpo.trace import (
        apply_trace_advantages,
        build_trace_score_batch,
    )

    original_update_actor = RayPPOTrainer._update_actor

    def update_actor_with_trace(self, batch):
        config = self.config.get("shopping_trace", {})
        enabled = bool(config.get("enable", False))
        if not enabled:
            batch.non_tensor_batch.pop("trace_target", None)
            return original_update_actor(self, batch)
        if not self.ref_in_actor:
            raise RuntimeError("shopping TRACE requires LoRA so the frozen base model is available")

        score_batch, state_counts = build_trace_score_batch(
            batch,
            self.tokenizer,
            max_sequence_length=int(config["max_sequence_length"]),
        )
        reference = self._compute_ref_log_prob(score_batch)
        target_mask = score_batch.batch["response_mask"]
        mean_log_probs = (
            reference.batch["ref_log_prob"] * target_mask
        ).sum(dim=-1) / target_mask.sum(dim=-1)
        batch.non_tensor_batch.pop("trace_target", None)
        trace_metrics = apply_trace_advantages(
            batch,
            mean_log_probs,
            state_counts,
            epsilon=float(config["epsilon"]),
            horizon=int(config["horizon"]),
            discount=float(config["discount"]),
            terminal_weight=float(config["terminal_weight"]),
            outcome_weight=float(config["outcome_weight"]),
            turn_weight=float(config["turn_weight"]),
        )
        trace_metrics["trace/target_log_prob_mean"] = float(mean_log_probs.mean())
        output = original_update_actor(self, batch)
        output.meta_info["metrics"].update(trace_metrics)
        return output

    update_actor_with_trace._shopping_trace = True
    RayPPOTrainer._update_actor = update_actor_with_trace


def install_torch_padding_fallback():
    """安装项目所需的 veRL 窄范围 runtime hooks。"""
    _isolate_ray_worker_compile_caches()
    _pin_ray_assigned_cuda_device()

    from verl.utils import attention_utils
    from verl.utils import npu_flash_attn_utils as fallback

    functions = (
        fallback.index_first_axis,
        fallback.pad_input,
        fallback.rearrange,
        fallback.unpad_input,
    )
    # ponytail: veRL 0.8 在 CUDA 上硬导入 FA2；上游提供 torch fallback 后删除此 hook。
    attention_utils._get_attention_functions = lambda: functions
    # SAO uses veRL bypass mode to carry the asynchronous rollout log-probs
    # into the actor update.  The guarded adapter is a no-op for the existing
    # GRPO configuration and only dispatches when the typed sao_dis loss
    # sentinel is explicitly enabled.
    from shopping_grpo.training.grpo.sao_policy import (
        install_sao_gae,
        install_sao_critic_freezing,
        install_sao_policy_loss,
    )

    try:
        install_sao_policy_loss()
        install_sao_gae()
        install_sao_critic_freezing()
    except ImportError as exc:
        # Unit tests provide a minimal fake veRL module for the CUDA/attention
        # hook.  A real veRL 0.8 runtime imports core_algos here; if it does
        # not, SAO is rejected by the training preflight before workers start.
        if "core_algos" not in str(exc):
            raise
    _install_trace_actor_update()
