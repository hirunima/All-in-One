# ASQL Diffusion — ICPR Reproducibility Submission

A Stable Diffusion pipeline with attention-guided generation controlled by scene graphs. Built on top of Hugging Face Diffusers.

---

## Setup

**Requirements:** Python 3.10, CUDA 11.x or 12.x (CPU-only is supported but slow).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If your system CUDA version differs from the default PyTorch wheel, install a matching build from https://pytorch.org first.

Authenticate with Hugging Face if the model requires it:

```bash
huggingface-cli login
```

---

## Running inference

`inference.py` is the main entry point. It takes a text prompt and a scene graph, runs the pipeline, and saves the generated image(s) to disk.

```bash
python inference.py \
  --prompt "A red cat above a blue car" \
  --sg '{"entities": [{"head":"cat","quantity":"","id":0,"attributes":["red"]},{"head":"car","quantity":"","id":1,"attributes":["blue"]}], "relations": [{"subject": 0,"relation": "above","object": 1}]}' \
  --model_path "sd2-community/stable-diffusion-2-1" \
  --output_directory ./outputs \
  --seed 42
```
Output images are saved as `<index>_1_<prompt>.jpg` inside `--output_directory`.

---

## Batch evaluation

`testing.py` wraps the same pipeline for running against standard benchmarks. It supports `--data_type` values of `HRS`, `T2I` and `GenEval` loading prompts automatically from the `data_evaluate/` folder. Use `--type` to select the evaluation category (e.g. `color`, `spatial`, `counting`).

```bash
python testing.py \
  --model_path "sd2-community/stable-diffusion-2-1" \
  --data_type HRS \
  --type color \
  --output_directory ./outputs
```

---

## Repository layout

| File | Role |
|---|---|
| `inference.py` | Single-prompt inference runner |
| `testing.py` | Batch benchmark evaluation |
| `test_pipeline.py` | `ASQLDiffusionPipeline` implementation |
| `pipeline_stable_diffusion_attend_and_excite_self.py` | Attend-and-excite attention processor |
| `phi.py` | Scene graph → language response helper |
| `clustering_mash.py` | Spatial clustering utilities |
| `selfstore.py` | Self-attention map storage |
| `requirements.txt` | Python dependencies |
| `data_evaluate/` | Benchmark prompt files (HRS, T2I, GenEval) |
