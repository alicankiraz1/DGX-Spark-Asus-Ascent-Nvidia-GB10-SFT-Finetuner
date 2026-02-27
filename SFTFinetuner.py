#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
import getpass
import time

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
from datasets import load_dataset, DatasetDict
from huggingface_hub import login, HfApi


def print_banner():
    try:
        import pyfiglet
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "pyfiglet"],
            stdout=subprocess.DEVNULL,
        )
        import pyfiglet
    banner = pyfiglet.figlet_format("SFTFinetuner", font="slant")
    print(banner)
    print("  Created by Alican Kiraz – v1.0")
    print("  Optimized for DGX Spark / Asus Ascent GX10 (GB10 · 128 GB Unified RAM)")
    print()


def separator(char="─", width=72):
    print(char * width)


def get_input(prompt: str, valid_options=None, default=None):
    suffix = ""
    if default is not None:
        suffix = f" [{default}]"
    while True:
        response = input(f"{prompt}{suffix}: ").strip()
        if not response and default is not None:
            return default
        if valid_options:
            if response.lower() in valid_options:
                return response.lower()
            print(f"  Invalid input. Choose from: {', '.join(valid_options)}")
        else:
            if response:
                return response
            print("  Input cannot be empty. Please try again.")


def get_int_input(prompt: str, default: int, min_val: int = 1, max_val: int = 999999):
    while True:
        raw = get_input(prompt, default=str(default))
        try:
            val = int(raw)
            if val < min_val or val > max_val:
                print(f"  Value should be between {min_val} and {max_val}.")
                continue
            return val
        except ValueError:
            print("  Please enter a valid integer.")


def get_float_input(prompt: str, default: float):
    while True:
        raw = get_input(prompt, default=str(default))
        try:
            return float(raw)
        except ValueError:
            print("  Please enter a valid number (e.g. 2e-4 or 0.0002).")


def get_secure_input(prompt: str):
    while True:
        response = getpass.getpass(f"{prompt}: ").strip()
        if response:
            return response
        print("  Input cannot be empty.")


def print_system_info():
    separator()
    print("  SYSTEM INFORMATION")
    separator()
    print(f"  Python        : {sys.version.split()[0]}")
    print(f"  PyTorch       : {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version  : {torch.version.cuda}")
        print(f"  GPU device    : {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        total_mem = props.total_memory / (1024 ** 3)
        print(f"  GPU memory    : {total_mem:.1f} GB")
    else:
        print("  WARNING: No CUDA GPU detected. Training will be extremely slow.")

    try:
        import psutil
        sys_mem = psutil.virtual_memory().total / (1024 ** 3)
        print(f"  System RAM    : {sys_mem:.1f} GB")
    except ImportError:
        pass
    separator()
    print()


def validate_safetensor_model(model_name: str, token: str = None) -> bool:
    try:
        api = HfApi()
        info = api.model_info(model_name, token=token)
    except Exception as exc:
        raise ValueError(
            f"Cannot access model '{model_name}' on HuggingFace Hub.\n"
            f"  Reason: {exc}"
        ) from exc

    siblings = [s.rfilename for s in (info.siblings or [])]

    gguf_files = [f for f in siblings if f.endswith(".gguf")]
    if gguf_files:
        raise ValueError(
            f"Model '{model_name}' contains GGUF files ({gguf_files[0]}, ...).\n"
            "  GGUF models are NOT supported. Please choose a standard safetensor model."
        )

    mlx_indicators = [f for f in siblings if "mlx" in f.lower() or f.startswith("mlx/")]
    tags = [t.lower() for t in (info.tags or [])]
    if mlx_indicators or "mlx" in tags:
        raise ValueError(
            f"Model '{model_name}' appears to be an MLX model.\n"
            "  MLX models are NOT supported. Please choose a standard safetensor model."
        )

    safetensor_files = [f for f in siblings if f.endswith(".safetensors")]
    if not safetensor_files:
        bin_files = [f for f in siblings if f.endswith(".bin")]
        if bin_files:
            print(f"  Note: Model uses legacy .bin format (not safetensors). Proceeding anyway.")
            return True
        raise ValueError(
            f"Model '{model_name}' has no .safetensors or .bin weight files.\n"
            "  Please choose a model that has proper weight files."
        )

    print(f"  Validated: {len(safetensor_files)} safetensor file(s) found.")
    return True


SUPPORTED_FILE_FORMATS = ["csv", "json", "jsonl", "parquet"]


def load_local_dataset_file(path: str, file_fmt: str) -> DatasetDict:
    if file_fmt not in SUPPORTED_FILE_FORMATS:
        raise ValueError(f"Unsupported format: {file_fmt}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")

    loader = "json" if file_fmt in ("json", "jsonl") else file_fmt
    ds = load_dataset(loader, data_files={"train": path})
    return ds


def detect_dataset_format(dataset):
    cols = dataset.column_names if hasattr(dataset, "column_names") else dataset["train"].column_names

    if "messages" in cols:
        return dataset, "conversational"

    if "prompt" in cols and "completion" in cols:
        return dataset, "prompt_completion"

    if "text" in cols:
        return dataset, "language_modeling"

    col_set = set(cols)

    if {"System", "User", "Assistant"}.issubset(col_set):
        print("  Detected legacy System/User/Assistant format. Converting to conversational ...")
        converted = convert_legacy_to_conversational(dataset)
        return converted, "conversational"

    lower_map = {c.lower(): c for c in cols}
    if all(k in lower_map for k in ("system", "user", "assistant")):
        print("  Detected legacy format (case-insensitive). Converting to conversational ...")
        converted = convert_legacy_to_conversational_ci(dataset, lower_map)
        return converted, "conversational"

    raise ValueError(
        f"Unrecognized dataset format. Columns found: {cols}\n"
        "  Expected one of:\n"
        "    - 'messages' (conversational)\n"
        "    - 'prompt' + 'completion' (prompt-completion)\n"
        "    - 'text' (language modeling)\n"
        "    - 'System' + 'User' + 'Assistant' (legacy)"
    )


def convert_legacy_to_conversational(ds):
    def _convert(example):
        messages = []
        sys_text = example.get("System") or ""
        usr_text = example.get("User") or ""
        ast_text = example.get("Assistant") or ""
        if sys_text.strip():
            messages.append({"role": "system", "content": sys_text.strip()})
        messages.append({"role": "user", "content": usr_text.strip()})
        messages.append({"role": "assistant", "content": ast_text.strip()})
        return {"messages": messages}

    if isinstance(ds, DatasetDict):
        return ds.map(_convert, remove_columns=ds["train"].column_names)
    return ds.map(_convert, remove_columns=ds.column_names)


def convert_legacy_to_conversational_ci(ds, lower_map):
    sys_col = lower_map["system"]
    usr_col = lower_map["user"]
    ast_col = lower_map["assistant"]

    def _convert(example):
        messages = []
        sys_text = example.get(sys_col) or ""
        usr_text = example.get(usr_col) or ""
        ast_text = example.get(ast_col) or ""
        if sys_text.strip():
            messages.append({"role": "system", "content": sys_text.strip()})
        messages.append({"role": "user", "content": usr_text.strip()})
        messages.append({"role": "assistant", "content": ast_text.strip()})
        return {"messages": messages}

    if isinstance(ds, DatasetDict):
        return ds.map(_convert, remove_columns=ds["train"].column_names)
    return ds.map(_convert, remove_columns=ds.column_names)


def ensure_train_eval_split(ds: DatasetDict, eval_ratio: float = 0.1) -> DatasetDict:
    if "train" not in ds:
        available = list(ds.keys())
        if len(available) == 1:
            ds = DatasetDict({"train": ds[available[0]]})
        else:
            raise ValueError(f"Expected a 'train' split. Found: {available}")

    has_eval = any(k in ds for k in ("validation", "test", "eval"))
    if not has_eval:
        print(f"  No eval split found. Creating one ({eval_ratio:.0%} of train) ...")
        split = ds["train"].train_test_split(test_size=eval_ratio, shuffle=True, seed=42)
        ds = DatasetDict({"train": split["train"], "validation": split["test"]})
    else:
        eval_key = next(k for k in ("validation", "test", "eval") if k in ds)
        if eval_key != "validation":
            ds = DatasetDict({"train": ds["train"], "validation": ds[eval_key]})
    return ds


def build_bnb_config(bit_choice: str):
    from transformers import BitsAndBytesConfig

    if bit_choice == "4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        return BitsAndBytesConfig(load_in_8bit=True)


def build_lora_config(rank: int = 64, alpha: int = 128, dropout: float = 0.05):
    from peft import LoraConfig

    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
        use_rslora=True,
    )


def print_training_summary(config: dict):
    separator("═")
    print("  TRAINING CONFIGURATION SUMMARY")
    separator("═")
    for key, val in config.items():
        label = key.replace("_", " ").title()
        print(f"  {label:<30s}: {val}")
    separator("═")
    print()


def main():
    print_banner()
    print_system_info()

    separator()
    print("  STEP 1 / 7 — Authentication")
    separator()
    model_privacy = get_input(
        "Is your base model private or public? [private/public]",
        valid_options=["private", "public"],
        default="public",
    )
    hf_token = None
    if model_privacy == "private":
        hf_token = get_secure_input("Enter your HuggingFace token")
        login(token=hf_token)
        print("  Logged in to HuggingFace Hub.")
    print()

    separator()
    print("  STEP 2 / 7 — Model Selection")
    separator()
    while True:
        base_model_name = get_input(
            "HuggingFace model repo (e.g. meta-llama/Llama-3.1-8B-Instruct)"
        )
        try:
            validate_safetensor_model(base_model_name, token=hf_token)
            break
        except ValueError as exc:
            print(f"\n  {exc}\n")
            print("  Please enter a different model.\n")
    print()

    separator()
    print("  STEP 3 / 7 — Fine-tuning Strategy")
    separator()
    print()
    print("  Available strategies:")
    print()
    print("    lora   LoRA (Low-Rank Adaptation)")
    print("           Trains small adapter layers on top of the frozen base model.")
    print("           Best balance of quality and memory. Recommended for most cases.")
    print("           Example: 8B model uses ~18 GB VRAM with LoRA.")
    print()
    print("    qlora  QLoRA (Quantized LoRA)")
    print("           Loads the base model in 4-bit (NF4) and trains LoRA on top.")
    print("           Lowest memory usage — ideal for very large models (30B-70B).")
    print("           Example: 70B model fits in ~40 GB VRAM with QLoRA.")
    print()
    print("    full   Full Fine-tuning")
    print("           Updates ALL model parameters. Highest quality but highest memory.")
    print("           Only practical for small models (1B-4B) on 128 GB unified memory.")
    print("           Example: 4B model uses ~32 GB VRAM with full fine-tuning.")
    print()
    tuning_type = get_input(
        "Select strategy [lora/qlora/full]",
        valid_options=["lora", "qlora", "full"],
        default="lora",
    )
    is_peft = tuning_type in ("lora", "qlora")
    is_qlora = tuning_type == "qlora"

    lora_rank = 64
    lora_alpha = 128
    lora_dropout = 0.05
    if is_peft:
        print()
        custom_lora = get_input(
            "Customize LoRA hyperparameters? [yes/no]",
            valid_options=["yes", "no"],
            default="no",
        )
        if custom_lora == "yes":
            print()
            print("  LoRA Rank (r)")
            print("    Controls the size of the low-rank matrices. Higher = more capacity but more memory.")
            print("    Common values: 8 (light), 32 (balanced), 64 (strong), 128+ (heavy)")
            lora_rank = get_int_input("  LoRA rank (r)", default=64, min_val=4, max_val=512)

            print()
            print("  LoRA Alpha")
            print("    Scaling factor for LoRA updates. Typically set to 2x the rank.")
            print("    Higher alpha = stronger influence of the fine-tuned weights.")
            lora_alpha = get_int_input("  LoRA alpha", default=lora_rank * 2, min_val=1, max_val=1024)

            print()
            print("  LoRA Dropout")
            print("    Regularization to prevent overfitting. Set to 0.0 if your dataset is large.")
            print("    Common values: 0.0 (no dropout), 0.05 (light), 0.1 (moderate)")
            lora_dropout = get_float_input("  LoRA dropout", default=0.05)

    precision = "bf16"
    if not is_qlora:
        print()
        print("  Precision")
        print("    bf16  BFloat16 — recommended for Blackwell GPUs. Best training stability.")
        print("    fp16  Float16  — slightly faster, but can cause NaN issues on some models.")
        print("    fp32  Float32  — full precision. 2x memory usage, only for debugging.")
        precision = get_input(
            "Precision [bf16/fp16/fp32]",
            valid_options=["bf16", "fp16", "fp32"],
            default="bf16",
        )
    print()

    separator()
    print("  STEP 4 / 7 — Dataset Selection")
    separator()
    print("  Supported dataset column formats:")
    print("    - 'messages' (conversational: system/user/assistant roles)")
    print("    - 'prompt' + 'completion'")
    print("    - 'text' (plain language modeling)")
    print("    - 'System' + 'User' + 'Assistant' (legacy / LLMRipper)")
    print("  Supported file formats: csv, json, jsonl, parquet")
    print()

    dataset_source = get_input(
        "Dataset source [local/huggingface]",
        valid_options=["local", "huggingface"],
    )

    raw_datasets = None
    if dataset_source == "local":
        dataset_path = get_input("Path to dataset file")
        dataset_fmt = get_input(
            "File format [csv/json/jsonl/parquet]",
            valid_options=SUPPORTED_FILE_FORMATS,
        )
        try:
            raw_datasets = load_local_dataset_file(dataset_path, dataset_fmt)
        except (FileNotFoundError, ValueError) as exc:
            print(f"  Error: {exc}")
            sys.exit(1)
    else:
        ds_privacy = get_input(
            "Is the HF dataset public or private? [public/private]",
            valid_options=["public", "private"],
            default="public",
        )
        if ds_privacy == "private" and not hf_token:
            hf_token = get_secure_input("Enter your HuggingFace token for the dataset")
            login(token=hf_token)
        dataset_repo = get_input("HF dataset repo (e.g. tatsu-lab/alpaca)")
        load_kwargs = {"token": hf_token} if (ds_privacy == "private" and hf_token) else {}
        try:
            raw_datasets = load_dataset(dataset_repo, **load_kwargs)
        except Exception as exc:
            print(f"  Error loading dataset: {exc}")
            sys.exit(1)

    raw_datasets = ensure_train_eval_split(raw_datasets)

    try:
        raw_datasets_train, fmt_name = detect_dataset_format(raw_datasets["train"])
        if isinstance(raw_datasets_train, DatasetDict):
            raw_datasets = raw_datasets_train
        else:
            eval_ds, _ = detect_dataset_format(raw_datasets["validation"])
            if isinstance(eval_ds, DatasetDict):
                raw_datasets = DatasetDict({
                    "train": eval_ds["train"] if "train" in eval_ds else raw_datasets_train,
                    "validation": eval_ds["validation"] if "validation" in eval_ds else raw_datasets["validation"],
                })
            else:
                raw_datasets = DatasetDict({
                    "train": raw_datasets_train,
                    "validation": eval_ds,
                })
    except ValueError as exc:
        print(f"  Error: {exc}")
        sys.exit(1)

    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["validation"]

    print(f"  Dataset format : {fmt_name}")
    print(f"  Train samples  : {len(train_dataset):,}")
    print(f"  Eval samples   : {len(eval_dataset):,}")
    print()

    separator()
    print("  STEP 5 / 7 — Training Hyperparameters")
    separator()
    print()

    default_lr = 2e-4 if is_peft else 2e-5

    print("  Max Sequence Length")
    print("    Maximum number of tokens per training sample. Longer = more context but more memory.")
    print("    Common values: 512 (short QA), 1024 (chat), 2048 (long-form), 4096+ (documents)")
    max_length = get_int_input("Max sequence length", default=2048, min_val=128, max_val=32768)

    print()
    print("  Per-device Batch Size")
    print("    Number of samples processed per GPU in each forward pass.")
    print("    Lower values save memory; higher values speed up training.")
    print("    Start with 1-2 for large models, 4-8 for smaller models on 128 GB.")
    batch_size = get_int_input("Per-device batch size", default=4, min_val=1, max_val=64)

    print()
    print("  Gradient Accumulation Steps")
    print("    Simulates a larger batch by accumulating gradients over N steps before updating.")
    print("    Effective batch size = batch_size x grad_acc  (e.g. 2 x 8 = 16)")
    grad_acc = get_int_input("Gradient accumulation steps", default=4, min_val=1, max_val=128)

    print(f"    -> Effective batch size: {batch_size * grad_acc}")

    print()
    print("  Number of Epochs")
    print("    How many times the model sees the entire dataset.")
    print("    1-2 for large datasets (10K+ samples), 3-5 for small datasets (<5K)")
    num_epochs = get_int_input("Number of epochs", default=3, min_val=1, max_val=100)

    print()
    print("  Learning Rate")
    print(f"    Controls the step size during optimization.")
    if is_peft:
        print("    For LoRA/QLoRA: 1e-4 to 3e-4 is typical. Default: 2e-4")
    else:
        print("    For full fine-tuning: 1e-5 to 5e-5 is typical. Default: 2e-5")
    print("    Too high = unstable training, too low = slow convergence.")
    learning_rate = get_float_input("Learning rate", default=default_lr)

    print()
    print("  Sequence Packing")
    print("    Combines multiple short samples into a single sequence to maximize GPU utilization.")
    print("    Recommended: 'yes' for datasets with short texts, 'no' for very long documents.")
    use_packing = get_input(
        "Enable sequence packing? [yes/no]",
        valid_options=["yes", "no"],
        default="yes",
    ) == "yes"

    print()
    output_dir = get_input("Output directory", default="./sft-output")

    print()

    summary = {
        "base_model": base_model_name,
        "strategy": tuning_type.upper(),
        "precision": precision.upper(),
        "dataset_format": fmt_name,
        "train_samples": f"{len(train_dataset):,}",
        "eval_samples": f"{len(eval_dataset):,}",
        "max_length": max_length,
        "batch_size": batch_size,
        "gradient_accumulation": grad_acc,
        "effective_batch_size": batch_size * grad_acc,
        "epochs": num_epochs,
        "learning_rate": learning_rate,
        "packing": use_packing,
        "output_dir": output_dir,
    }
    if is_peft:
        summary["lora_rank"] = lora_rank
        summary["lora_alpha"] = lora_alpha
        summary["lora_dropout"] = lora_dropout

    print_training_summary(summary)

    confirm = get_input(
        "Proceed with training? [yes/no]",
        valid_options=["yes", "no"],
        default="yes",
    )
    if confirm != "yes":
        print("  Aborted by user.")
        sys.exit(0)

    separator()
    print("  STEP 6 / 7 — Loading Model & Training")
    separator()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTTrainer, SFTConfig

    quantization_config = None
    if is_qlora:
        print("  Building 4-bit quantization config (NF4 + double quant) ...")
        quantization_config = build_bnb_config("4")

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    torch_dtype = dtype_map[precision]

    print(f"  Loading model: {base_model_name} ...")
    load_start = time.time()

    model_kwargs = dict(
        trust_remote_code=True,
        device_map="auto",
        token=hf_token if model_privacy == "private" else None,
    )
    if is_qlora:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["dtype"] = torch_dtype
    else:
        model_kwargs["dtype"] = torch_dtype

    model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)

    if not is_qlora:
        model.config.use_cache = False

    load_elapsed = time.time() - load_start
    print(f"  Model loaded in {load_elapsed:.1f}s")

    print("  Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        token=hf_token if model_privacy == "private" else None,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("  Set pad_token = eos_token")

    peft_config = None
    if is_peft:
        print(f"  Building LoRA config (r={lora_rank}, alpha={lora_alpha}, rsLoRA=True) ...")
        peft_config = build_lora_config(
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
        )

    use_assistant_loss = False
    if fmt_name == "conversational":
        template_str = getattr(tokenizer, "chat_template", "") or ""
        if "generation" in template_str:
            use_assistant_loss = True
            print("  Chat template supports assistant masking — enabling assistant_only_loss.")
        else:
            print("  Chat template does not support {% generation %} tags — training on full sequence.")

    sft_config = SFTConfig(
        output_dir=output_dir,
        max_length=max_length,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_acc,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        bf16=(precision == "bf16"),
        fp16=(precision == "fp16"),
        gradient_checkpointing=True,
        packing=use_packing,
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        eval_strategy="steps",
        eval_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        optim="adamw_torch_fused",
        warmup_ratio=0.03,
        weight_decay=0.01,
        max_grad_norm=0.3,
        assistant_only_loss=use_assistant_loss,
        push_to_hub=False,
        report_to="none",
        dataset_num_proc=os.cpu_count(),
    )

    print("  Initializing SFTTrainer ...")
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )

    if is_peft:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        ratio = trainable / total * 100 if total > 0 else 0
        print(f"  Trainable parameters: {trainable:,} / {total:,} ({ratio:.2f}%)")

    print()
    separator()
    print("  TRAINING STARTED")
    separator()
    print()

    train_start = time.time()
    train_result = trainer.train()
    train_elapsed = time.time() - train_start

    print()
    separator()
    print("  TRAINING COMPLETE")
    separator()
    print(f"  Duration     : {train_elapsed / 60:.1f} minutes")
    if hasattr(train_result, "metrics"):
        metrics = train_result.metrics
        print(f"  Train loss   : {metrics.get('train_loss', 'N/A')}")
        print(f"  Train samples: {metrics.get('train_samples', 'N/A')}")
        print(f"  Train steps  : {metrics.get('train_steps', 'N/A')}")
    print()

    separator()
    print("  STEP 7 / 7 — Save & Export")
    separator()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"  Model saved to: {output_dir}")

    if is_peft:
        merge_choice = get_input(
            "Merge LoRA weights into the base model? [yes/no]",
            valid_options=["yes", "no"],
            default="yes",
        )
        if merge_choice == "yes":
            merged_dir = os.path.join(output_dir, "merged")
            print("  Merging LoRA weights ...")
            merged_model = trainer.model.merge_and_unload()
            merged_model.save_pretrained(merged_dir)
            tokenizer.save_pretrained(merged_dir)
            print(f"  Merged model saved to: {merged_dir}")

            push_merged = get_input(
                "Push merged model to HuggingFace Hub? [yes/no]",
                valid_options=["yes", "no"],
                default="no",
            )
            if push_merged == "yes":
                if not hf_token:
                    hf_token = get_secure_input("Enter your HuggingFace token")
                    login(token=hf_token)
                repo_id = get_input("HF repo ID (e.g. YourName/my-finetuned-model)")
                private_repo = get_input(
                    "Make repo private? [yes/no]",
                    valid_options=["yes", "no"],
                    default="no",
                ) == "yes"
                print(f"  Uploading to {repo_id} ...")
                merged_model.push_to_hub(repo_id, token=hf_token, private=private_repo)
                tokenizer.push_to_hub(repo_id, token=hf_token, private=private_repo)
                print(f"  Upload complete: https://huggingface.co/{repo_id}")
    else:
        push_choice = get_input(
            "Push model to HuggingFace Hub? [yes/no]",
            valid_options=["yes", "no"],
            default="no",
        )
        if push_choice == "yes":
            if not hf_token:
                hf_token = get_secure_input("Enter your HuggingFace token")
                login(token=hf_token)
            repo_id = get_input("HF repo ID (e.g. YourName/my-finetuned-model)")
            private_repo = get_input(
                "Make repo private? [yes/no]",
                valid_options=["yes", "no"],
                default="no",
            ) == "yes"
            print(f"  Uploading to {repo_id} ...")
            trainer.model.push_to_hub(repo_id, token=hf_token, private=private_repo)
            tokenizer.push_to_hub(repo_id, token=hf_token, private=private_repo)
            print(f"  Upload complete: https://huggingface.co/{repo_id}")

    print()
    separator("═")
    print("  ALL DONE – Happy fine-tuning!")
    separator("═")
    print()


if __name__ == "__main__":
    main()
