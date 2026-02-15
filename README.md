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

How to Run
1. Clone Repository
git clone https://github.com/kaustubh-5016/federated-traffic-prediction.git
cd federated-traffic-prediction

2. Start Aggregation Server
python server.py

3. Start Edge Nodes
python client.py --node_id=1
python client.py --node_id=2
...
python client.py --node_id=6

Future Improvements

Asynchronous federated learning

Differential privacy integration

Secure aggregation

Dynamic client participation

Adaptive learning rate scheduling
