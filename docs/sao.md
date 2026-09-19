# ShopSimulator SAO arm

This arm follows [Single-Rollout Asynchronous Optimization for Agentic
Reinforcement Learning](https://arxiv.org/abs/2607.07508v1), adapted to the
ShopSimulator action/observation masks.

`configs/sao.yaml` is an independent training recipe for the asynchronous
ShopSimulator rollout. It leaves `configs/grpo.yaml` unchanged and selects:

- one asynchronous rollout (`rollout.n=1`),
- direct double-sided importance sampling (DIS) from the rollout worker's
  `rollout_log_probs`, with a strict `(0.8, 1.2)` token support interval,
- GAE with action-token masks that bridge tool observations, and
- a value model with frozen attention and two PPO epochs for every actor epoch.

The DIS adapter is installed by the existing Ray worker setup hook. In veRL
bypass mode, `old_log_probs` already points to the asynchronous rollout
probabilities, so the adapter does not maintain a historical policy or run a
second old-policy forward pass. The baseline bypass objective is used whenever
the typed `rollout_correction.loss_type` sentinel is not `sao_dis`.

The same hook freezes full-attention parameters in the critic before FSDP
wrapping; the value optimizer therefore updates the value head, linear mixer,
and MLP projections while preserving the pretrained full-attention
representation. Terminal
shopping rewards are moved from a final tool observation to its preceding
action token before skip-observation GAE, so a purchase or graceful stop is not
silently discarded by the action mask.

The project-owned numerical contracts are in
`shopping_grpo.training.grpo.sao`; `tests/test_sao.py` checks the ratio support
set, masked policy terms, and skip-observation GAE on CPU. veRL 0.8 already
contains the same skip-observation GAE recurrence; the helper makes the
contract explicit and testable without importing veRL.

先用 dry-run 检查命令与配置路径；正式启动时，训练入口会在加载模型前执行
SAO preflight：

```bash
python scripts/train_grpo.py \
  --config configs/sao.yaml \
  --model /path/to/sft-merged \
  --train-data data/grpo/train.parquet \
  --val-data data/grpo/validation.parquet \
  --output outputs/models/sao \
  --experiment-name shopping-agent-sao \
  --dry-run
```

The runtime preflight rejects group rollouts, disabled rollout log-probability
collection, incompatible advantage estimators, and a missing critic before a
training process is started.
