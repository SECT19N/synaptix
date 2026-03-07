# ⚡ Snapticks

> A lightweight, native neural network visualizer and student verification tool — built entirely in Rust.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Built with Rust](https://img.shields.io/badge/built%20with-Rust-orange.svg)
![Status](https://img.shields.io/badge/status-in%20development-yellow.svg)
![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20Windows%20%7C%20macOS-lightgrey.svg)

---

## What is Snapticks?

Snapticks lets you **build, configure, and train artificial neural networks visually** — without the overhead of PyTorch or TensorFlow. It is aimed squarely at students who want to:

- Verify manual forward pass and backpropagation calculations
- Build intuition for how neural networks actually learn
- Experiment with different architectures, activation functions, and learning rate schedules in real time

If you solved a network by hand in class and want to check your answer in seconds — this is that tool.

---

## Features

### Network Configuration
- Set input count, output count, number of hidden layers, and neurons per layer
- Choose activation function per layer independently
- Select from multiple loss functions
- Configure learning rate, batch size, and epoch count

### Activation Functions
Standard: `ReLU`, `Leaky ReLU`, `Sigmoid`, `Tanh`, `Softmax`, `Linear`

Rare / modern: `ELU`, `SELU`, `Swish / SiLU`, `GELU`, `Mish`

### Visualization
- **Live network diagram** — nodes and edges rendered in real time
- **Loss curve** — updates each epoch during training
- **Accuracy curve** — for classification tasks
- **Learning rate schedule visualizer** — see how LR changes over time
- **Weight distribution histogram** — watch weights shift during training

### Learning Rate Schedulers
`Step Decay` · `Exponential Decay` · `Cosine Annealing` · `Warm Restart (SGDR)` · `ReduceLROnPlateau`

### Student Verification Mode
- Step through a single forward pass manually
- Inspect pre-activation `z` and post-activation `a` values at every neuron
- View gradients at every layer after one backprop step
- Enter custom weight values to test known configurations
- Compare expected vs actual output with delta display

### Data
- Load training datasets from CSV files
- Auto-detect columns, preview data, and label inputs vs outputs

---

## Stack

| | |
|---|---|
| **Language** | Rust (stable) |
| **GUI** | [egui](https://github.com/emilk/egui) + [eframe](https://github.com/emilk/egui/tree/master/crates/eframe) |
| **Math** | [ndarray](https://github.com/rust-ndarray/ndarray) |
| **Plots** | [egui_plot](https://github.com/emilk/egui/tree/master/crates/egui_plot) |
| **Serialization** | [serde](https://serde.rs) + serde_json |
| **File Dialogs** | [rfd](https://github.com/PolyMeilex/rfd) |
| **Parallelism** | [rayon](https://github.com/rayon-rs/rayon) |

---

## Getting Started

### Prerequisites

- Rust stable toolchain via [rustup](https://rustup.rs)
- On Linux, the following system packages:

```bash
sudo pacman -S --needed base-devel pkg-config libx11 libxcb libxkbcommon mesa gtk3
```

### Build & Run

```bash
git clone https://github.com/yourusername/snapticks
cd snapticks
cargo run
```

---

## Roadmap

See [ROADMAP.md](./ROADMAP.md) for the full phased development plan.

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Project setup, egui skeleton, GitHub repo | 🔧 In Progress |
| 1 | Neural network engine (forward pass, backprop) | ⏳ Planned |
| 2 | GUI — configuration panel, diagrams, charts | ⏳ Planned |
| 3 | Student verification features | ⏳ Planned |
| 4 | Packaging, CI, WASM / browser version | ⏳ Planned |
| 5 | Mobile, CNNs, advanced optimizers | 🔮 Future |

---

## Contributing

Snapticks is open source and welcomes contributions. Whether you are a Rust learner, a student, or an ML enthusiast — there is something here for you.

See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

---

## License

Licensed under the [MIT License](./LICENSE).

---

<p align="center">Built with Rust · Powered by egui · Made for students</p>
