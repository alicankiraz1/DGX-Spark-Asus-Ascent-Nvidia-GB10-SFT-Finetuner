# SFT Finetuner

**No-code Supervised Fine-Tuning tool designed for NVIDIA DGX Spark / Asus Ascent GX10 (GB10 · 128 GB Unified RAM)**

An interactive CLI tool that guides you through the entire SFT workflow — from model selection to training to uploading your fine-tuned model — without writing a single line of code.

## Features

- **Interactive CLI** — step-by-step prompts, no coding required
- **LoRA / QLoRA / Full Fine-Tuning** — choose the strategy that fits your memory budget
- **Safetensor Validation** — automatically rejects GGUF and MLX models
- **Flexible Dataset Support** — conversational (`messages`), prompt-completion, plain text, and legacy `System/User/Assistant` formats
- **Local & HuggingFace Datasets** — load from disk (csv, json, jsonl, parquet) or directly from the Hub
- **Automatic Format Conversion** — legacy datasets are converted to the modern conversational format on the fly
- **BF16 & Sequence Packing** — optimized for Blackwell architecture throughput
- **LoRA Merge & Hub Upload** — merge adapter weights and push the final model to HuggingFace Hub in one flow

## Hardware Requirements

| Component | Specification |
|-----------|---------------|
| Device | NVIDIA DGX Spark or Asus Ascent GX10 |
| GPU | NVIDIA GB10 Grace Blackwell Superchip |
| Memory | 128 GB Unified RAM |
| CUDA | 13.0 (via PyTorch nightly) |
| Python | 3.10+ |

> The tool can also run on other CUDA-capable GPUs, but optimal performance and memory defaults are tuned for the GB10 128 GB unified memory architecture.

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/DGX-Spark-Asus-Ascent-Nvidia-GB10-SFT-Finetuner.git
cd DGX-Spark-Asus-Ascent-Nvidia-GB10-SFT-Finetuner

chmod +x setup_and_run.sh
./setup_and_run.sh
```

The setup script will:
1. Create a Python virtual environment
2. Install PyTorch nightly with CUDA 13.0 support
3. Install all project dependencies
4. Verify GPU and CUDA availability
5. Launch the SFT Finetuner

## Supported Models

Any HuggingFace model that ships with **safetensor** (`.safetensors`) or legacy `.bin` weight files. Examples:

- `meta-llama/Llama-3.1-8B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`
- `google/gemma-2-9b-it`
- `Qwen/Qwen2.5-7B-Instruct`

Models in GGUF or MLX format are automatically detected and rejected.

## Supported Dataset Formats

| Format | Required Columns |
|--------|-----------------|
| Conversational | `messages` (list of `{role, content}` dicts) |
| Prompt-Completion | `prompt`, `completion` |
| Language Modeling | `text` |
| Legacy (LLMRipper) | `System`, `User`, `Assistant` |

## Fine-Tuning Strategies

| Strategy | Description | Memory Usage |
|----------|-------------|-------------|
| **LoRA** | Low-Rank Adaptation with rsLoRA — recommended default | Medium |
| **QLoRA** | 4-bit NF4 quantization + LoRA — lowest memory footprint | Low |
| **Full** | All parameters updated — only for smaller models | High |

## Project Structure

```
├── SFTFinetuner.py      # Main interactive fine-tuning tool
├── setup_and_run.sh     # Environment setup & launcher script
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## Dependencies

- PyTorch (nightly, CUDA 13.0)
- Transformers
- TRL (Transformer Reinforcement Learning)
- PEFT (Parameter-Efficient Fine-Tuning)
- Datasets
- BitsAndBytes
- Accelerate
- HuggingFace Hub
- Safetensors
- Pyfiglet

## Author

**Alican Kiraz**

## License

This project is licensed under the MIT License.
