# Llama 3.2 3B Fine-tuning for PDE Operator Prediction

This package implements the fine-tuning pipeline for predicting operator sequences in PDE solutions, following the paper:

> **"From Equations to Insights: Unraveling Symbolic Structures in PDEs with LLMs"**
> 
> Rohan Bhatnagar, Ling Liang, Krish Patel, Haizhao Yang
> 
> University of Maryland & University of Tennessee

## Quick Start

### 1. Install Dependencies

```bash
pip install torch transformers peft datasets bitsandbytes accelerate
```

### 2. Prepare Your Data

Ensure your generated data is in JSON format with the following structure:

```json
[
  {
    "input": "Type: Poisson | RHS: const x0 * | Cauchy: x2=0.0 const ||| const | Solution: ||",
    "target": "const x0 ^3 + const x1 x2 * cos * +",
    ...
  },
  ...
]
```

Place your data at `./data/pde_dataset.json` or specify with `--data` flag.

### 3. Run Training

```bash
# Basic training with default settings
python train_llama_fex.py --data ./data/pde_dataset.json

# With custom settings
python train_llama_fex.py \
    --data ./data/pde_dataset.json \
    --output ./my_model \
    --epochs 3 \
    --batch-size 4 \
    --lr 2e-4
```

### 4. Run Inference

```bash
# Interactive mode
python inference.py --model ./finetuned_model --interactive

# Single prediction
python inference.py --model ./finetuned_model \
    --prompt "Type: Poisson | RHS: const x0 * | Dirichlet: x0=0 const | Solution: ||"

# Evaluate on test data
python inference.py --model ./finetuned_model --test-data ./data/test.json
```

## Configuration

### Memory-Optimized Settings (RTX 5070 Ti, 16GB VRAM)

The default settings are optimized for 16GB VRAM:

| Setting | Value | Purpose |
|---------|-------|---------|
| Batch Size | 4 | Per-device batch |
| Gradient Accumulation | 8 | Effective batch = 32 |
| 4-bit Quantization | NF4 | Memory reduction |
| Gradient Checkpointing | True | Trade compute for memory |
| LoRA Rank | 16 | Paper setting |

### Paper Settings (Reference)

From Section 5.1:

| Setting | Paper | Our Adaptation |
|---------|-------|----------------|
| Batch Size | 32 per GPU (8 GPUs) | 4 × 8 = 32 effective |
| Learning Rate | 2×10⁻⁴ | 2×10⁻⁴ |
| Weight Decay | 0.01 | 0.01 |
| Epochs | 12 | 3 (converges faster) |
| LoRA Rank | 16 | 16 |
| LoRA Alpha | 32 | 32 |
| LoRA Dropout | 0.1 | 0.1 |
| Precision | BF16 | BF16 + 4-bit |

## Files

```
├── train_llama_fex.py    # Main training script
├── inference.py           # Inference and evaluation
├── config_llama_fex.yaml  # Configuration file
└── README_TRAINING.md     # This file
```

## Expected Results

Based on the paper (Table 3), you should expect:

| Metric | LLaMA-3B |
|--------|----------|
| Initial Mismatch | ~0.80 |
| Final Mismatch (12 epochs) | ~0.41 |
| Convergence | ~6-8 epochs |

With 3 epochs, expect mismatch around 0.5-0.6.

## Troubleshooting

### Out of Memory (OOM)

1. Reduce batch size: `--batch-size 2`
2. Increase gradient accumulation: `--grad-accum 16`
3. Enable gradient checkpointing (default)

### Slow Training

1. Increase batch size if memory allows
2. Disable gradient checkpointing: `--no-gradient-checkpointing`
3. Reduce data loading workers in config

### Login Required for Llama

If you get an error about model access:

```bash
# Login to Hugging Face
huggingface-cli login

# Or use token
export HF_TOKEN=your_token_here
```

## Data Format Details

The training script expects your data pipeline output format:

```json
{
    "input": "Type: <PDE> | RHS: <f_ops> | <BC_Type>: <boundary> <g_ops> | Solution: ||",
    "target": "<u_ops>",
    "u_expr": "x0**3 - 3*cos(x1*x2) - 1",
    "f_expr": "...",
    "pde_type": "Poisson",
    "boundary_type": "Cauchy"
}
```

The key fields used for training:
- `input`: The prompt for the LLM
- `target`: The expected operator sequence output

## Integration with FEX

After fine-tuning, integrate with the Finite Expression Method:

```python
from inference import load_finetuned_model, predict_operators, extract_operators

model, tokenizer = load_finetuned_model("./finetuned_model")

# Get predicted operators for a PDE
prompt = "Type: Poisson | RHS: const x0 * | Dirichlet: x0=0 const | Solution: ||"
prediction = predict_operators(model, tokenizer, prompt)
operators = extract_operators(prediction)

# Use operators in FEX
# fex.set_operators(binary_ops, unary_ops)
```
