# Budget Adaptive REViT Technical Report

Budget-adaptive REViT gives an image one width profile while keeping the same finite group throughout the network. The implementation supports the quarter-turn group (C_4) and the square's rotation-reflection group (D_4). It uses one shared lift, one decision per image, nested steerable coefficients, and physically narrower computation for smaller profiles.

## Model and execution

1. `BudgetAdaptiveREViT._stem` centers the sampling grid, lifts the image to a small set of regular fields using a steerable convolution, and applies normalization and ReLU. The stem width is common to every execution path.
2. The controller computes the spatial and orientation mean and maximum of each stem field. A two-layer MLP consumes these statistics and the requested budget. An eligibility mask sets scores above the cap to negative infinity. Argmax breaks exact ties in favor of the lowest profile index.
3. The model groups examples by the selected profile, executes each occupied bucket at its own width, and restores input order. A homogeneous batch takes the direct path. The controller decision incurs one device-to-host transfer for eager dispatch. User-supplied device budget tensors also require validation.
4. Each stage applies an entry convolution and blocks with steerable Q/K/V, output projection and two-layer MLP. Global attention reuses the original SDPA implementation. The hierarchical variant uses square-window attention with optional SDPA. It downsamples by a factor of two between stages.
5. The classifier takes the maximum over the group axis and averages spatially. Its linear projection shares a prefix of the final field columns across profiles.

A profile (p) specifies a vector of stage widths (w_{p,s}). A convolution owns a coefficient tensor of shape `[max_out_fields, max_in_fields, basis_size]`. For the chosen widths, it expands only the active field pairs using an ESCNN basis sampled for a single pair. Output and input group axes are then packed for `conv2d`. A 3×3 D4 convolution between two fields has a `16 × 16 × 3 × 3` kernel, while the same operator between six fields has a `48 × 48 × 3 × 3` kernel. This sharing also applies to the MLP expansion widths and stage transitions. Each profile has its own batch-normalization scale, bias and running statistics, shared across the group axis of each field. Since the head count stays fixed, the number of fields per head and the corresponding field partitions can change with width.

In no-grad eval mode, kernels are expanded lazily at parameter precision and cached per profile. Training, a device or dtype change, and loading a checkpoint clear the cache. Checkpoints store learned coefficients and normalization state. `export_profile(i)` copies the selected prefixes and normalization modules into a standalone model without a controller. A matching singleton model can also be trained independently. The exported checkpoint includes its configuration and can initialize a singleton fine-tuning run. `export --widths` also crops a separately trained wide profile to new field prefixes, including the inherited normalization moments. Recalibrate those moments during fine-tuning before comparing the resulting pruned baseline.

## Equivariance on the image grid

Let (T_g) act on the image and (
ho_g) act on regular feature fields. The shared lifting features satisfy (S(T_g x)=
ho_g S(x)). Each regular field is permuted along its orientation axis, and square-grid (C_4) and (D_4) actions permute spatial locations. Mean and max over both axes give a descriptor unchanged by these permutations. For a fixed budget (c), the controller therefore selects (b(T_g x,c)=b(x,c)).

Each profile contains only complete regular fields. Steering is applied to each input-output field pair, so restricting the set of pairs preserves the linear intertwiner constraint. Attention heads contain an integer number of regular fields. Their group action is orthogonal, preserving the Q/K inner product. The spatial group action jointly permutes queries and keys in global attention. For window attention, it permutes whole windows and locations inside each window. Each profile's nonlinearities, normalization, and residual additions commute with these same permutations in eval mode. With the invariant route, the selected feature path is equivariant and the pooled classifier logits are invariant.

Two grid details matter. Fixed-origin strided convolution can miss the rotated counterpart of a sampled site on an even grid. The adaptive model pools before its steerable convolution using uniform blocks when the dimension divides the factor and mirror-paired adaptive-average bins otherwise. In window attention, padding is equal on opposite edges and the grid tiles the padded square. The code reduces an even requested window side by one for an odd grid when needed and uses a single window if the entire feature map fits. Zero padding participates in the attention computation and is cropped from the output. These geometry and stem choices also belong in the matched fixed baselines.

The argument applies to exact square-grid actions in real arithmetic. Floating-point accumulation order can change scores near an argmax tie. Test routing agreement and logit error at the deployed precision, including examples with small eligible-score margins. `Z2` maps to the trivial ESCNN group in this repository and explicit C4/D4 guarantees apply to those corresponding configured models. The adaptive paths accept square inputs and preserve C4/D4 geometry at odd and even side lengths.

## Budget and training objective

Given positive increasing costs (C_i), profile (i) is feasible for normalized budget (c \in [0,1]) when \((C_i-C_0)/(C_{P-1}-C_0) \le c\). The default estimates dense post-stem MACs for the configured image size. It counts the active convolutions, QK and AV products including padded tokens, and the classifier. It excludes the shared stem, routing, normalization, memory traffic, allocation overhead and compiler effects. `--profile-costs` can supply an increasing calibrated vector, such as fixed-profile median milliseconds for one device, precision, and batch setting. Budget is a ceiling in the supplied cost units and the selected cost can vary within that ceiling.

Training samples a budget threshold independently for each image. Each profile sees the full training batch and contributes to the mean classification loss. The controller minimizes the expected detached per-profile loss plus a normalized cost penalty over feasible profiles. Classification gradients train the model at every width, while the routing gradient trains the router and shared stem. A sampled budget that admits only one profile provides no route-choice gradient. The approach supports hard labels, label smoothing and ImageNet Mixup/CutMix targets. Train memory grows with active profile count because their activations contribute to the same backward pass. Evaluation uses hard routing and profile-specific running statistics.

## Repository map

| Location | Responsibility |
| --- | --- |
| `budget_revit.py` | Regular-field profile operators, centered grid, router, MAC estimate, training loss, and export. |
| `budget_cli.py` | CLI arguments, adaptive config metadata, target smoothing, checkpoint reconstitution. |
| `group_space.py` | Maps legacy group names to ESCNN gspaces. |
| `revit_gcsa.py` | Original global lifting, Q/K/V attention, MLP, invariant head, and SDPA helper reused by the adaptive model. |
| `revit_windowed_gcsa.py` | Original hierarchical windowed backbone, stem, stage transitions, and classifier. |
| `train_revit.py` | Small-data entry point, held-out development split, logging and checkpoints. |
| `imagenet_train_revit.py` | ImageNet transforms, legacy and adaptive training, AMP, distributed training and checkpoint resume. |
| `imagenet_train_vit.py` | Static torchvision ViT-Small ImageNet reference. |
| `datasets/cifar10.py`, `datasets/rot_mnist.py`, `datasets/pcam.py` | Data layout, transforms, and train/evaluation loaders. PCam expects `PCam/train`, `PCam/valid`, and `PCam/test` in ImageFolder format for the corresponding splits. |
| `utils.py` | Reusable rotation augmentation and early-stopping utilities. |
| `scripts/run_budget.py` | Adaptive evaluator, fixed export, timed deployment runs and profiler traces. |
| `scripts/benchmark_sdpa.py` | Standalone comparison of attention kernels on the original backbones. |
| `scripts/cleanup_checkpoints.py` | Checkpoint retention utility. |
| `tests/test_budget_revit.py` | CPU regression checks for group actions, nesting and training/export. |
| `requirements.txt`, `requirements-lock.txt` | Core dependencies and the pinned environment used by the original project. |
| `analysis/EQUIVARIANCE_RESEARCH.md`, `analysis/MODEL_CONFIGS_RESEARCH.md` | Existing architecture and symmetry notes. |
| `analysis/budget_revit_paper.tex` | Editable paper draft and experiment templates. |

## Experiment design

Develop on CIFAR-10 with a seeded held-out subset of the training data. Use ImageNet and a PCam test split to test transfer of the result across tasks. Keep width-search effort, pretraining, augmentations, precision, hardware and evaluation splits aligned. Check one global and one hierarchical model. Compare independently trained singleton widths, a pruned exported prefix with matched fine-tuning, the learned route, and the trivial-group dynamic control. A supernet export controls for the effect of routing at a given trained width. Pruning an independently trained wide model with `export --widths` provides a separate compression baseline. Independently trained small singletons control for multi-width training. C4 experiments can use `--pointwise-backend fourier` to check the same weights through an unfused PyTorch FFT reference for regular-to-regular 1×1 projections. That path extracts the first column of each circulant block and caches its Fourier transform. Benchmark the official fused Flash EQ-Linear CUDA kernels separately on compatible hardware. The cyclic construction requires a separate design for D4. [Flash EQ-Linear](https://arxiv.org/html/2607.21271v2)

Evaluate fixed widths and multiple budgets on identical inputs. The evaluator can shuffle routes within each batch while preserving its profile histogram. It also estimates the best possible allocation under the same aggregate cost and per-image cap by taking the cheapest available correction first. This label-informed oracle isolates available per-input headroom. Compare these controls with the learned route, count errors rescued or introduced by widening, and report seed intervals. A learned router that fails to beat the strongest fixed frontier, or that fails to outperform shuffled routes at the same allocation mix, weakens the adaptation claim. Shuffling uses one reproducible permutation per batch. Repeat with multiple seeds or use a global permutation for a tighter null estimate.

Benchmark real evaluation batches as well as synthetic inputs. Use `--dataset` with a trained checkpoint to reuse a seeded set of full batches across fixed and adaptive timing runs. The steady-state wall timer begins with an input already on the device and includes the stem, controller, host decision, bucket operations, transformer, and classifier. It excludes data loading, device transfer and compilation warmup. Report p50/p95 forward latency at batch one, batched images per second, observed bucket composition, and CUDA allocator peaks. `--prewarm-all-profiles` includes all resident expanded kernels. `--trace` annotates module ranges for Q/K/V, output projection, MLP, stem, stage transitions, normalization and the router MLP. Kernel and Python-stack inspection can further attribute attention layout, descriptor and dispatch costs. The CLI reports CUDA allocated memory peaks. CPU and MPS peak reporting require platform-specific tools.

Use exact quarter-turn and reflection shifts with `--orientation` and inspect every in-group action with `--check-group`. These tests distinguish numerical symmetry from generalization to orientation shifts outside the group. Use fixed angles, interpolation settings and independent validation when evaluating out-of-group transformations.

## Verification and remaining work

Local CPU checks cover every C4/D4 element on the global and windowed backbones, odd and even shapes, mixed budget dispatch, ESCNN kernel agreement, physical narrowing, router and width gradients, normalization updates, and fixed export/reload. Both classification entry points completed a short synthetic ImageFolder run. The benchmark, evaluator and export workflows ran on synthetic data. The CPU smoke checks also exercised compiled post-stem dispatch and checked the C4 Fourier pointwise reference against the dense output. Full dataset training, CUDA and distributed performance, and the fused Flash EQ-Linear comparison remain open. The paper marks the corresponding results and analysis for later completion.

The literature includes switchable normalization, dynamic slimmable width, token pruning, and representation-preserving static compression. This experiment asks whether input-conditioned width improves the measured frontier *while the same equivariance holds for each path*. [Slimmable Neural Networks](https://arxiv.org/abs/1812.08928), [Dynamic Slimmable Network](https://arxiv.org/abs/2103.13258), [DynamicViT](https://arxiv.org/abs/2106.02034), [Symmetry-Preserving Pruning](https://doi.org/10.3390/sym18060983), [REViT](https://arxiv.org/abs/2606.25318)
