#!/bin/bash

# Use local directory path for pre-compiled model
MODEL_PATH="/home/ubuntu/ChessLM_Qwen3_Trainium"

# Start vLLM server with pre-compiled Neuron artifacts
# Model compiled with: tp=2, batch_size=4, max_context_length=2048, continuous_batching=true
vllm serve $MODEL_PATH \
    --device neuron \
    --tensor-parallel-size 2 \
    --max-model-len 2048 \
    --max-num-seqs 4 \
    --dtype bfloat16 \
    --port 8000 &

PID=$!
echo "vLLM server started with PID $PID"
echo "Server will be available at http://localhost:8000"
echo "To stop: kill $PID"
