# ML Chess Engine

A from-scratch chess engine combining classical search algorithms with neural network position evaluation. Built as a deep dive into ML fundamentals and systems design.

**Stack:** Python · PyTorch · Flask · React · Docker

## Architecture

```
React Chess UI
      ↓
Flask API
      ↓
Neural Position Evaluator (PyTorch)
      ↓
Search Algorithm (Minimax / MCTS)
```

## Features

- Board representation with full legal move generation
- Minimax search with alpha-beta pruning
- Neural network position evaluation trained on grandmaster games
- REST API for inference
- Playable web UI

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the engine
python -m engine.main

# Start the API server
python api/app.py

# Run the UI (in /ui directory)
npm install && npm run dev
```

## Project Structure

```
├── engine/          # Core chess logic + search
├── training/        # Data pipeline + training scripts
├── api/             # Flask inference server
├── ui/              # React frontend
├── models/          # Saved model weights
└── tests/           # Unit tests
```

---

## Progress Tracker

### Phase 1: Core Chess Foundation
- [ ] Board state representation (8×8×12 tensor)
- [ ] Legal move generation for all pieces
- [ ] Special moves (castling, en passant, promotion)
- [ ] Check/checkmate/stalemate detection
- [ ] FEN string parsing
- [ ] Perft tests passing

### Phase 2: Search Algorithm
- [ ] Basic Minimax implementation
- [ ] Alpha-beta pruning
- [ ] Handcrafted evaluation function (material + piece-square tables)
- [ ] Move ordering (MVV-LVA, killer moves)
- [ ] Iterative deepening
- [ ] Engine plays reasonable chess at depth 4-5

### Phase 3: Neural Network Evaluator
- [ ] Download and parse Lichess dataset
- [ ] Data preprocessing pipeline
- [ ] CNN architecture implementation
- [ ] Training loop with validation
- [ ] Neural eval beats handcrafted eval
- [ ] Model checkpointing

### Phase 4: MCTS (Optional)
- [ ] Basic MCTS tree structure
- [ ] UCB1 selection
- [ ] Policy + value dual-head network
- [ ] MCTS integrated with neural network

### Phase 5: Self-Play Training
- [ ] Self-play game generation
- [ ] Training on self-play data
- [ ] ELO tracking between versions
- [ ] Measurable improvement over iterations

### Phase 6: API & Infrastructure
- [ ] Flask `/move` endpoint
- [ ] Flask `/evaluate` endpoint
- [ ] Dockerfile
- [ ] Docker Compose setup
- [ ] API documentation

### Phase 7: React UI
- [ ] Chessboard component
- [ ] Drag-and-drop moves
- [ ] Legal move highlighting
- [ ] Engine integration
- [ ] Evaluation bar display
- [ ] Move history panel

### Phase 8: Polish & Testing
- [ ] Unit tests for move generation
- [ ] Integration tests for API
- [ ] Performance benchmarks
- [ ] README finalized
- [ ] Demo deployed

---

## Resources

- [Chess Programming Wiki](https://www.chessprogramming.org)
- [Lichess Database](https://database.lichess.org)
- [AlphaZero Paper](https://arxiv.org/abs/1712.01815)
