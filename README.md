Federated Learning Based Traffic Prediction System

A distributed federated learning framework designed to simulate large-scale network traffic prediction across decentralized edge nodes. The system models real-world communication constraints such as variable latency and bandwidth while maintaining stable global model convergence.

Overview

This project implements a federated learning (FL) pipeline across 6 distributed edge nodes to train traffic prediction models without centralizing raw data.

Each node performs local training on its own dataset and periodically sends model updates to a central aggregation server. The global model is updated using federated averaging and redistributed to all nodes for subsequent training rounds.

The system simulates realistic distributed training conditions including:

Network latency variability

Bandwidth limitations

Asynchronous client updates

Communication overhead constraints

System Architecture
                +------------------+
                |  Aggregation     |
                |      Server      |
                +------------------+
                         ↑
      ---------------------------------------------
      ↑          ↑           ↑         ↑         ↑
   Node 1     Node 2      Node 3    Node 4    Node 5
                                ↑
                              Node 6

Components
1. Edge Nodes (Clients)

Perform local model training

Maintain private traffic datasets

Send model weights (not raw data)

Receive updated global model

2. Aggregation Server

Collects local model updates

Applies Federated Averaging (FedAvg)

Redistributes global parameters

Manages synchronization rounds

Key Features

Distributed training across 6 edge nodes

Federated Averaging (FedAvg) aggregation

Simulated network latency and bandwidth constraints

Model convergence monitoring

Update coordination logic for consistent parameter synchronization

Reduced communication overhead via controlled aggregation rounds

Technical Stack

Python

NumPy / PyTorch (if applicable — update accordingly)

Socket-based or RPC-based communication (update if needed)

Multi-process / multi-threaded coordination

Core Challenges Addressed
1. Synchronization Under Variable Network Conditions

Implemented coordination logic to handle delayed client updates and ensure stable global model convergence despite inconsistent communication latency.

2. Communication Overhead Reduction

Optimized update frequency and aggregation scheduling to reduce unnecessary network transmission during distributed training.

3. Convergence Stability

Ensured consistent parameter updates across distributed nodes to prevent model divergence.

Example Workflow

Initialize global model at server

Distribute model to all edge nodes

Perform local training at each node

Send updated weights to server

Aggregate updates using FedAvg

Redistribute global model

Repeat for N communication rounds

Performance Evaluation

The system evaluates:

Global model convergence across rounds

Communication cost per training cycle

Latency impact on synchronization

Stability under bandwidth constraints

How to Deploy and Run

> Note: the previous "server.py/client.py" run instructions are outdated for this repository state.

Prerequisites

- Python 3.10+ (3.10/3.11 recommended for widest TensorFlow wheel compatibility; verify TensorFlow version support before using 3.12+).

1. Prepare Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```


2. Train/generate federated models

From the repository root:

```bash
python run.py
```

This command trains clients, performs federated aggregation rounds, and writes model artifacts under:

- `data/client*_space`
- `data/server_space`

3. Start the inference API and dashboard

```bash
python api_inference_server.py
```

The service starts on `0.0.0.0:8002`.

- Dashboard: `http://<host>:8002/` or `http://<host>:8002/dashboard`
- API base: `http://<host>:8002/api/inference`

Key API endpoints:

- `GET /api/inference/available-models`
- `POST /api/inference/predict` (multipart form fields: `file`, `client_id`)
- `GET /api/inference/metrics/<prediction_id>`
- `GET /api/inference/export/<prediction_id>`

4. Production hardening guidance

- Place the Flask service behind Nginx or Caddy and terminate TLS (HTTPS) at the proxy.
- Run the app under a managed service (for example: `systemd`, container runtime, or process manager).
- Keep a single app worker for TensorFlow/Keras stability unless multi-worker behavior has been validated in your environment (for example, `gunicorn --workers 1 --worker-class sync`).
- Restrict inbound firewall access to only the reverse proxy/public ports.

5. Operational checks after deploy

- Verify trained models are visible:
  - `GET /api/inference/available-models`
- Upload a CSV in dashboard/API and confirm predictions are returned.
- Validate metrics retrieval using the returned `prediction_id`.
- Validate CSV export with `GET /api/inference/export/<prediction_id>`.
- Check service logs for startup/model-loading errors.

Future Improvements

Asynchronous federated learning

Differential privacy integration

Secure aggregation

Dynamic client participation

Adaptive learning rate scheduling
