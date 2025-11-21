#!/bin/bash
set -e

# 1) Use the HF model ID
MODEL_ID="kunhunjon/ChessLM_Qwen3_Trainium_AWS_Format"

# 2) Make sure we use NxD inference as the Neuron backend in vLLM
export VLLM_NEURON_FRAMEWORK="neuronx-distributed-inference"

# (Optional) Explicitly list plugins, but leaving this unset is also fine since
# optimum_neuron is already being found & loaded.
# export VLLM_PLUGINS="optimum_neuron"

# (Optional) Where to cache compiled artifacts if/when vLLM/Optimum compiles anything new
# export NEURON_COMPILED_ARTIFACTS="/home/ubuntu/neuron-compiled-artifacts/chess-qwen"

VLLM_RPC_TIMEOUT=100000 python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_ID" \
  --device neuron \
  --tensor-parallel-size 2 \
  --max-model-len 2048 \
  --max-num-seqs 4 \
  --dtype bfloat16 \
  --port 8080 \
  --task generate &   # vLLM "task" (generate vs embeddings), *not* HF pipeline task

PID=$!
echo "vLLM server started with PID $PID"
echo "Server will be available at http://localhost:8080"
echo "To stop: kill $PID"
