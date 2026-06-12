# LLM Research Projects

Seven self-contained research projects spanning RL fine-tuning of language models, mechanistic interpretability, and vision-language models. Each project ships its code, a compiled report, and the result files behind every number quoted in its README, so the headline claims can be checked without rerunning anything.

Ivan A. Novosad — melodiz.2021@mail.ru

## Reinforcement learning

### [01 — Policy gradient methods and behaviour cloning on CartPole-v1](projects/01_cartpole-policy-gradient/)

Six policy gradient variants and behaviour cloning, with every claim re-tested at 15 seeds using paired t-tests. Most of the 5-seed rankings dissolve under testing (value baselines do nothing here, p = 0.91), while update frequency matters: RLOO with K = 2 reaches 468.0 ± 14.3 reward and solves 15/15 seeds.

### [02 — GRPO on graph isomorphism: a cold-start problem](projects/02_grpo-graph-isomorphism/)

GRPO on Qwen2.5-1.5B-Instruct for graph isomorphism, where the base model starts collapsed: 0% iso accuracy, 100% non-iso, class prediction ratio 1.0, so there is no reward variance and no gradient. The pipeline adds anti-hacking rewards, SFT warmup, an edge-counting positive control, and 53 frozen test sets. Revealing partial node mappings pushed CPR to 0.28, but gradient norm stayed at 0.0, so the diversity came from sampling alone.

## Mechanistic interpretability

### [03 — Superposition geometry and SAE feature recovery in a toy model](projects/03_superposition-sae-recovery/)

A toy model with known ground-truth features stores F sparse features in d < F dimensions; an SAE then tries to recover them, swept over geometry and sparsity with three seeds per configuration. Good reconstruction does not imply good recovery: at d = 10 the SAE reaches explained variance 0.977 while recovering only 19% of the true features.

### [04 — Carry propagation circuits in a small addition transformer](projects/04_carry-circuits/)

A 2-layer, 3-head transformer (~398K parameters) trained on 3-digit addition, analyzed with activation patching. Patching selects a 4-of-8-component carry circuit, and ablating its top head erases carry accuracy (MC1 100% → 5.2%), but the necessity test fails: dormant heads take over when circuit heads are removed, so carry computation is spread beyond the named heads.

### [05 — Hyper-PCD: detecting shortcut dependence with sparse concept probes](projects/05_hyper-pcd-shortcuts/)

DistilBERT fine-tuned on SST-2 with a label-agreeing hint token on 80% of training data ends up leaning entirely on the hint: flipped-hint accuracy 0.0000 against 0.8681 clean. A sparse concept probe (128 concepts, top-k = 8) reads that dependence from hidden states alone, reaching AUROC 0.957 for hint polarity and 0.915 for model error.

## Vision-language models

### [06 — Where visual features become readable in Qwen2-VL-2B](projects/06_vlm-logit-lens/)

Logit lens applied at every layer of Qwen2-VL-2B-Instruct over 1,250 synthetic images, measuring where visual attributes become readable in the language head. Color emerges in a one-layer jump (P(target) 0.001 at L22 to 0.413 at L23); binding turns on at L24 with distractor probability below 0.01, and counting collapses for n ≥ 4 (71% accuracy at the final layer).

### [07 — Can a VLM learn to imagine FrozenLake?](projects/07_vlm-frozenlake-imagination/)

LoRA SFT teaches Qwen2.5-VL-3B-Instruct one-step FrozenLake dynamics from rendered frames: 100.0% validation accuracy with ground-truth state text, 99.17% from the image alone. Driving a lookahead planner with that learned world model yields 16.7% success; even the planner with ground-truth dynamics only reaches 20.0%.

## Repository structure

Each project directory is self-contained: a README with sourced numbers, `report/` holding the compiled PDF plus its LaTeX source, `src/` with the runnable code, `results/` with the figures and metric tables the README cites, and its own `requirements.txt` (the projects have disjoint dependency stacks, so there is no shared root environment). Project 05 is notebook-based; the rest run as scripts from each project root. Everything is MIT licensed (see [LICENSE](LICENSE)).
