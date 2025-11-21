# Chess Model Deployment and Evaluation Lab

This lab demonstrates deploying a fine-tuned chess model on AWS Trainium using vLLM, and evaluating it through competitive tournaments.

## Overview

Learn how to:
- Deploy a fine-tuned Qwen3 chess model via vLLM on Trainium
- Configure vLLM for optimal throughput with continuous batching
- Run competitive chess tournaments with TrueSkill ratings
- Leverage automatic request concurrency for improved performance

## Prerequisites

1. **Complete the Fine-Tuning Lab** (recommended):
   - [FT-Qwen3-1.7B-chess.ipynb](../../FineTuning/HuggingFaceExample/01_finetuning/FT-Qwen3-1.7B-chess.ipynb)
   - Or use a pre-trained chess model from HuggingFace

2. **Hardware Requirements**:
   - AWS Trainium instance (trn1.2xlarge or larger)
   - Neuron SDK 2.20+ (included in Workshop Studio / Neuron DLAMI)

3. **Software Requirements**:
   - Python 3.10+
   - optimum-neuron 0.3.0+
   - Dependencies in `assets/requirements.txt`

## Lab Structure

### 1. Chess-Deployment.ipynb
**Objective**: Deploy your chess model via vLLM on Trainium

**What you'll learn**:
- vLLM server setup with Neuron backend
- Model compilation for Trainium (batch_size=4, continuous_batching)
- Testing deployed model with simple games
- Performance validation (latency, throughput)

**Duration**: 30-40 minutes (including compilation)

### 2. Chess-Tournament.ipynb
**Objective**: Evaluate your model through competitive tournaments

**What you'll learn**:
- Tournament system with TrueSkill ratings
- Playing multiple games in parallel for throughput
- Analyzing results with metrics (win rate, ELO, ACPL)
- Understanding automatic request batching benefits

**Duration**: 20-30 minutes

## Quick Start

```bash
# Navigate to lab directory
cd /home/ubuntu/environment/neuron-workshops/labs/vLLM/Chess

# Install dependencies
pip install -r assets/requirements.txt

# Open deployment notebook
jupyter notebook Chess-Deployment.ipynb
```

## Architecture

### Components

1. **ChessEnvironment** (`env.py`)
   - Game engine powered by python-chess
   - Manages turn-based gameplay between agents
   - Enforces chess rules and detects game termination

2. **VLLMAgent** (`agents/vllm_agent.py`)
   - Connects to vLLM server via OpenAI-compatible API
   - Parses model outputs to extract UCI chess moves
   - Handles retries and error recovery

3. **Tournament Scheduler** (`run_game.py`)
   - Runs multiple games in parallel using multiprocessing (p_map)
   - Implements TrueSkill rating system for agent comparison
   - Generates PGN files and statistics

### Concurrency Model

**Two levels of parallelism work together automatically:**

1. **Process-Level Parallelism** (`p_map` in run_game.py)
   - Runs N games simultaneously in separate processes
   - Each game makes independent HTTP requests to vLLM
   - Controlled by `--parallelism` flag (default: 4)

2. **Request-Level Batching** (vLLM server)
   - vLLM server configured with `max_num_seqs=4`
   - Continuous batching automatically groups concurrent requests
   - When 4 games request moves simultaneously → 1.4x throughput improvement

**Example:**
```bash
# Running 4 games in parallel automatically enables batching
python assets/run_game.py \
  --agent vllm \
  --agent stockfish-skill5-depth10 \
  --num-games 20 \
  --parallelism 4 \
  --output-dir tournament_results
```

**Performance:**
- Single request: ~0.58s latency, 1.72 moves/sec
- 4 concurrent requests: ~1.86s latency per request, 2.15 total moves/sec
- **Throughput improvement: 1.4x** (games complete ~40% faster overall)

## Files Overview

```
Chess/
├── README.md                           # This file
├── Chess-Deployment.ipynb              # Lab 1: Model deployment
├── Chess-Tournament.ipynb              # Lab 2: Tournament evaluation
└── assets/
    ├── requirements.txt                # Python dependencies
    ├── env.example                     # Environment template
    ├── env.py                          # Chess game environment
    ├── chess_renderer.py               # Board visualization
    ├── run_game.py                     # Tournament orchestration
    ├── example.py                      # Usage examples
    ├── agents/
    │   ├── __init__.py
    │   ├── base.py                     # Abstract agent interface
    │   ├── vllm_agent.py              # vLLM integration
    │   └── stockfish_agent.py         # Baseline opponent
    └── vllm-server/
        ├── README.md                   # vLLM setup guide
        ├── vllm.sh                     # Server startup script
        ├── compile_model.py           # Model compilation
        └── start_vllm_python.py       # Python server starter
```

## Common Use Cases

### 1. Deploy a Custom Fine-Tuned Model

If you fine-tuned your own model, update the model path in `vllm.sh`:

```bash
MODEL_PATH="/path/to/your/model"
```

Then recompile for Trainium:

```bash
cd assets/vllm-server
python compile_model.py
```

### 2. Test Against Different Opponents

The environment includes Stockfish baselines with configurable difficulty:

```bash
# Easy opponent (Stockfish skill level 1)
python assets/run_game.py --agent1 vllm --agent2 stockfish-skill1-depth2 --num-games 5

# Medium opponent (Stockfish skill level 5)
python assets/run_game.py --agent1 vllm --agent2 stockfish-skill5-depth10 --num-games 5

# Strong opponent (Stockfish skill level 10)
python assets/run_game.py --agent1 vllm --agent2 stockfish-skill10-depth15 --num-games 5
```

### 3. Analyze Model Performance

Tournament results are saved to `tournament.json` with detailed metrics:

```python
import json

# Load tournament results
with open('tournament_results/tournament.json') as f:
    results = json.load(f)

# Check your model's rating
agent_stats = results['agents']['vllm']
print(f"Conservative Rating: {agent_stats['conservative']:.1f}")
print(f"Win Rate: {agent_stats['wins'] / agent_stats['games'] * 100:.1f}%")

# Analyze move quality
metrics = results['engine_metrics']['vllm']
print(f"Move Accuracy: {metrics['accuracy_pct']:.1f}%")
print(f"Avg Centipawn Loss: {metrics['acpl']:.1f}")
```

## Troubleshooting

### vLLM Server Not Starting

**Problem**: Server fails with "Neuron cores not available"

**Solution**:
```bash
# Check Neuron core usage
neuron-ls

# Kill processes using cores
pkill -f vllm

# Restart server
cd assets/vllm-server && bash vllm.sh
```

### Model Compilation Takes Too Long

**Problem**: First-time compilation can take 10-30 minutes

**Solution**: This is expected behavior. Neuron compiles the model for Trainium hardware. Subsequent runs will be fast as compiled artifacts are cached.

### Slow Inference Performance

**Problem**: High latency per move (>1s)

**Possible causes**:
- Not using compiled model (check for `model.pt` file)
- Wrong tensor parallelism setting (should match cores: tp=2 for trn1.2xlarge)
- `max_num_seqs` mismatch with compiled batch_size

### Version Compatibility Issues

**Problem**: Import errors or runtime failures

**Solution**: Verify Neuron SDK versions match requirements:

```python
import subprocess, sys

def get_version(pkg):
    result = subprocess.run([sys.executable, "-m", "pip", "show", pkg],
                          capture_output=True, text=True)
    for line in result.stdout.split('\n'):
        if line.startswith('Version:'):
            return line.split(':')[1].strip()
    return "Not installed"

print(f"neuronx-cc: {get_version('neuronx-cc')}")
print(f"torch-neuronx: {get_version('torch-neuronx')}")
print(f"optimum-neuron: {get_version('optimum-neuron')}")
```

**Minimum versions:**
- `neuronx-cc`: 2.15+
- `torch-neuronx`: 2.1+
- `optimum-neuron`: 0.3.0+

If versions are too old, see [Neuron SDK Installation Guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/).

## Additional Resources

- [AWS Neuron Documentation](https://awsdocs-neuron.readthedocs-hosted.com/)
- [vLLM on Neuron Guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/developer_guides/vllm-user-guide.html)
- [optimum-neuron Documentation](https://huggingface.co/docs/optimum-neuron)
- [Chess Fine-Tuning Lab](../../FineTuning/HuggingFaceExample/01_finetuning/FT-Qwen3-1.7B-chess.ipynb)

## Next Steps

1. Complete [Chess-Deployment.ipynb](Chess-Deployment.ipynb) to deploy your model
2. Run [Chess-Tournament.ipynb](Chess-Tournament.ipynb) to evaluate performance
3. Experiment with different opponents and tournament configurations
4. Fine-tune your model further based on tournament results
5. Deploy to production with learned configurations

Happy chess playing! 🎮♟️
