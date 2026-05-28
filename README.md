# Vision-OPD

**Vision-OPD: Learning to See Fine-Grained Details for Multimodal LLMs via On-Policy Self-Distillation**

<p align="center">
📃 <a href="https://arxiv.org/abs/2605.18740" target="_blank">Paper</a> | 
🤗 <a href="https://huggingface.co/datasets/yuanqianhao/Vision-OPD-6K">Training Dataset</a> 
</p>

## News

- **[2026.5.18]** The paper is released on arXiv.
- **[2026.5.19]** The training code and data is released.
- **[2026.5.26]** The evaluation code is released.
- Model release is under company review. Coming soon.

## Overview

Vision-OPD is a regional-to-global on-policy self-distillation framework that transfers a model's own privileged regional perception to its full-image policy, enabling fine-grained visual understanding in a single forward pass — without external teachers, ground-truth labels, reward verifiers, or inference-time tool use.

<p align="center">
  <img src="figures/average_bar_chart.png" alt="Vision-OPD Average Scores" width="90%"/>
</p>

<p align="center"><i>Average scores across fine-grained visual understanding benchmarks, including V* Bench, ZoomBench,  HR-Bench 4K, HR-Bench 8K, MME-RealWorld-EN and MME-RealWorld-CN.</i></p>

## Quick Start

### 1. Environment Setup

```bash
conda create -n vision-opd python=3.12
conda activate vision-opd
pip install -e .
pip install flash-attn --no-build-isolation
pip install flash-linear-attention==0.4.2
pip install flashinfer-python==0.6.6
pip install causal-conv1d==1.6.1
pip install xformers==0.0.32.post1 --no-deps
pip install transformers==5.5.0
```

### 2. Prepare Training Data

Download and preprocess the [Vision-OPD-6K](https://huggingface.co/datasets/yuanqianhao/Vision-OPD-6K) dataset:

```bash
python scripts/prepare_data.py --data-dir ./data
```

This downloads images and metadata from HuggingFace, extracts archives, and converts `train.jsonl` to the parquet format expected by the training pipeline.

### 3. Training

Launch Vision-OPD training:

```bash
bash scripts/run_vision_opd.sh
```

Key hyperparameters can be edited at the top of the script. See the script for the full configuration.

### 4. Merge Checkpoints

After training, merge the FSDP-sharded checkpoint into a standard HuggingFace model:

```bash
bash scripts/merge_checkpoint.sh <path_to_checkpoint>
```

For example:

```bash
bash scripts/merge_checkpoint.sh ./checkpoints/Vision-OPD-Qwen3.5-4B/global_step_65/
```

This merges the FSDP actor shards, saves the model weights, config, tokenizer, and processor into the specified directory. The merged checkpoint can then be loaded directly with `transformers` or served with vLLM.

### 5. Deployment

Serve the merged checkpoint with vLLM, for example:

```bash
vllm serve <path_to_merged_checkpoint> \
    --gpu-memory-utilization 0.85 \
    --tensor-parallel-size 8 \
    --served-model-name Vision-OPD-4B \
    --trust-remote-code
```

The server listens on port 8000 by default. You can then query the model via the OpenAI-compatible API at `http://localhost:8000/v1/chat/completions`.

### 6. Evaluation

Evaluate the deployed model on fine-grained visual benchmarks:

```bash
API_BASE="http://localhost:8000/v1/" \
OPENAI_MODEL_ID="Vision-OPD-4B" \
JUDGE_API_BASE="YOUR_JUDGE_API_BASE" \
JUDGE_MODEL="YOUR_JUDGE_MODEL_NAME" \
BENCHMARK="vstar,zoombench,hrbench-4k,hrbench-8k,mme-realworld,mme-realworld-cn" \
bash eval/run_eval.sh
```

Supported benchmarks: `vstar`, `zoombench`, `hrbench-4k`, `hrbench-8k`, `mme-realworld`, `mme-realworld-cn`.

The evaluation script runs inference via the OpenAI-compatible API. Judge configuration can be set via `JUDGE_API_BASE` / `JUDGE_MODEL_PATH` environment variables.

## Project Structure

```
Vision-OPD/
├── verl/                    # Modified verl framework with self-distillation support
├── scripts/
│   ├── run_vision_opd.sh    # Training launch script
│   ├── merge_checkpoint.sh  # FSDP checkpoint merger
│   └── prepare_data.py      # Data download & preprocessing
├── eval/
│   ├── run_eval.sh          # Evaluation entry point
│   ├── prepare_data.py      # Benchmark data download & preparation
│   ├── infer.py             # OpenAI API inference
│   ├── judge_qwenlm.py      # LLM-based judge (API or local vLLM)
│   ├── cal_acc.py           # Accuracy calculation
│   └── vstar_bench_utils.py # V*Bench question formatting utilities
├── chat_templates/
│   └── perception_chat_template_qwen35.jinja
├── figures/                 # Paper figures
├── pyproject.toml
└── LICENSE
```

## Citation

If you find Vision-OPD useful for your research, please consider citing:

```bibtex
@article{yuan2026vision,
  title={Vision-OPD: Learning to See Fine Details for Multimodal LLMs via On-Policy Self-Distillation},
  author={Yuan, Qianhao and Lou, Jie and Yu, Xing and Lin, Hongyu and Sun, Le and Han, Xianpei and Lu, Yaojie},
  journal={arXiv preprint arXiv:2605.18740},
  year={2026}
}
```

## License

Apache-2.0 License
