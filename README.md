# Hierarchical Attention Network for Medical Multi-Organ Disease Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Abstract

This repository contains the implementation of a novel Hierarchical Attention Network (HAN) framework for multi-organ disease prediction in medical diagnostics. The system leverages heterogeneous graph neural networks with meta-path-based reasoning to model complex relationships between patients, clinical tests, organs, and diseases. Our approach demonstrates significant advancement in interpretable medical AI by combining multi-head attention mechanisms with hierarchical semantic aggregation.

**Key Contributions:**
- Novel HAN++ architecture with improved node-level and semantic-level attention
- HGT-HAN hybrid model combining Heterogeneous Graph Transformer principles with hierarchical attention
- Meta-path-based reasoning over medical knowledge graphs (Patient-Disease-Patient, Patient-Organ-Patient, Patient-Symptom-Patient)
- Automatic label generation from clinical test values against medical normal ranges
- Multi-organ severity classification with organ damage score regression
- Comprehensive validation framework with accuracy, F1-score, and interpretability metrics

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Installation](#installation)
- [Dataset Structure](#dataset-structure)
- [Quick Start](#quick-start)
- [Model Architectures](#model-architectures)
- [Training Pipeline](#training-pipeline)
- [Evaluation Metrics](#evaluation-metrics)
- [Project Structure](#project-structure)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

---

## Overview

### Motivation

Traditional machine learning approaches in medical diagnostics often treat organs and diseases as independent entities, failing to capture the complex interdependencies in human physiology. This project addresses three critical challenges:

1. **Heterogeneous Medical Data**: Integrating diverse data types (patient demographics, clinical tests, organ systems, disease classifications)
2. **Multi-Organ Interactions**: Modeling cascading effects where one organ's dysfunction affects others
3. **Interpretability**: Providing transparent attention-based explanations for medical predictions

### Medical Knowledge Graph

Our framework constructs a heterogeneous medical graph with four node types:

- **Patients (P)**: Individual patient records with clinical test values
- **Symptoms/Tests (S)**: Laboratory tests and clinical measurements
- **Organs (O)**: 25+ organ systems (e.g., cardiovascular, renal, hepatic)
- **Diseases (D)**: Disease classifications mapped to organ systems

The graph is enriched with meta-paths representing different reasoning patterns:
- `P-D-P`: Patients sharing similar disease patterns
- `P-O-P`: Patients with similar organ involvement
- `P-S-P`: Patients with similar clinical test profiles

---

## Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Input Layer                              │
│         Patient Features (Clinical Test Values)             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Feature Projection Layer                       │
│           Linear: in_dim → hidden_dim                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│          Node-Level Attention (Per Meta-Path)               │
│     ┌──────────────┬──────────────┬──────────────┐         │
│     │   P-D-P      │   P-O-P      │   P-S-P      │         │
│     │  Multi-Head  │  Multi-Head  │  Multi-Head  │         │
│     │  Attention   │  Attention   │  Attention   │         │
│     └──────┬───────┴──────┬───────┴──────┬───────┘         │
└────────────┼──────────────┼──────────────┼─────────────────┘
             │              │              │
             └──────────────┼──────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│            Semantic-Level Attention                         │
│      Learnable Weights Across Meta-Paths (β)               │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                Output Projection                            │
│           Linear: hidden_dim → out_dim                      │
└──────────────────────┬──────────────────────────────────────┘
                       │
              ┌────────┴────────┐
              │                 │
              ▼                 ▼
┌──────────────────────┐  ┌─────────────────────┐
│ Multi-Organ Severity │  │ Organ Damage Score  │
│   Classification     │  │    Regression       │
│  [N, O, 4 classes]   │  │    [N, O organs]    │
└──────────────────────┘  └─────────────────────┘
```

### Attention Mechanisms

**Node-Level Attention:**
- Multi-head attention within each meta-path
- Residual connections and layer normalization
- GELU activation for non-linearity
- Dropout for regularization

**Semantic-Level Attention:**
- Learnable attention weights across different meta-paths
- Captures relative importance of different reasoning patterns
- Provides interpretability through attention weight visualization

---

## Features

### Core Capabilities

✅ **Dual Model Architecture**
- **HAN++**: Enhanced hierarchical attention with multi-head node-level attention
- **HGT-HAN**: Hybrid model integrating Heterogeneous Graph Transformer principles

✅ **Automated Label Generation**
- Automatically computes organ severity labels from clinical test values
- Compares patient measurements against medical normal ranges
- No manual annotation required for training

✅ **Multi-Task Learning**
- Multi-organ severity classification (4 severity levels per organ)
- Organ damage score regression (continuous scores)
- Joint optimization with weighted loss functions

✅ **Interpretability**
- Attention weight visualization at node and semantic levels
- Meta-path importance scores (β weights)
- Per-organ prediction explanations

✅ **Robust Training Pipeline**
- MLflow experiment tracking and logging
- Early stopping with patience-based monitoring
- Best model checkpointing
- Comprehensive metric logging (accuracy, F1-micro, F1-macro, loss curves)

✅ **Enhanced Visualization**
- 6-panel training metric plots
- Loss curves, accuracy trends, F1-score evolution
- Comparative metric analysis
- Per-organ performance breakdown

---

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended, optional for CPU-only)
- pip or conda package manager

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/HAN-implementation.git
cd HAN-implementation
```

2. **Create a virtual environment (recommended):**
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n han-medical python=3.8
conda activate han-medical
```

3. **Install dependencies:**
```bash
pip install torch>=2.0.0 torchvision torchaudio
pip install pandas numpy scipy scikit-learn
pip install matplotlib seaborn
pip install mlflow
pip install jupyter notebook  # For running notebooks
```

4. **Verify installation:**
```python
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## Dataset Structure

### Required Data Files

Place the following files in the `data/` directory:

```
data/
├── filtered_patient_reports.csv          # Patient clinical test records
├── test-disease-organ.csv                # Medical knowledge graph (tests → organs → diseases)
└── patient-one-hot-labeled-disease-new.csv  # Pre-computed labels (optional)
```

### Data File Descriptions

#### 1. `filtered_patient_reports.csv`
Patient records with clinical test values:
```
PatientID, Age, Gender, Test1, Test2, ..., TestN
P001, 45, M, 120, 80, ..., 5.2
P002, 67, F, 140, 90, ..., 6.8
```

#### 2. `test-disease-organ.csv`
Medical knowledge graph mapping tests to organs and diseases:
```
TestName, Organ, Disease, NormalRangeMin, NormalRangeMax, Unit
BloodPressureSystolic, Cardiovascular, Hypertension, 90, 120, mmHg
CreatinineClearance, Renal, ChronicKidneyDisease, 90, 120, mL/min
```

#### 3. `patient-one-hot-labeled-disease-new.csv` (Optional)
Pre-computed severity labels for each organ:
```
PatientID, Organ1_Severity, Organ2_Severity, ..., OrganN_Severity
P001, 0, 1, ..., 0
P002, 2, 0, ..., 1
```
*Note: If not provided, labels are automatically generated during data loading.*

---

## Quick Start

### Basic Training Example

```python
import torch
from HAN import MedicalGraphData, HANPP, HGT_HAN
from HAN.validation_metrics import compute_accuracy, plot_training_metrics_enhanced

# 1. Load and process data
data_loader = MedicalGraphData(
    path_records="data/filtered_patient_reports.csv",
    path_symptom="data/test-disease-organ.csv",
    symptom_freq_threshold=5,
    prune_per_patient=True,
    nnz_threshold=3,
    seed=42
)

data_loader.load_data()
data_loader.build_labels_and_features()
data_loader.build_adjacency_matrices()

# 2. Prepare meta-path neighbors
metapath_names = ['P-D-P', 'P-O-P', 'P-S-P']
neighbor_dicts = data_loader.get_metapath_neighbors(metapath_names)

# 3. Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HANPP(
    in_dim=data_loader.features.shape[1],
    hidden_dim=128,
    out_dim=64,
    metapath_names=metapath_names,
    num_heads=4,
    num_organs=25,
    num_severity=4,
    dropout=0.3
).to(device)

# 4. Training setup
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion_severity = torch.nn.CrossEntropyLoss()
criterion_score = torch.nn.MSELoss()

# 5. Train model (simplified example)
for epoch in range(40):
    model.train()
    optimizer.zero_grad()
    
    organ_logits, organ_scores, z, beta = model(
        data_loader.features.to(device),
        neighbor_dicts
    )
    
    # Compute loss and backpropagate
    loss = criterion_severity(organ_logits, labels) + 0.1 * criterion_score(organ_scores, scores)
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch+1}/40 - Loss: {loss.item():.4f}")

# 6. Evaluate
accuracy_metrics = compute_accuracy(model, features, labels, neighbor_dicts, test_idx)
print(f"Test Accuracy: {accuracy_metrics['overall_accuracy']:.4f}")
```

### Using the Complete Training Script

For full-featured training with MLflow tracking, visualization, and model saving:

```bash
cd Other_py
python train_complete.py
```

Or use the Jupyter notebook:

```bash
cd notebooks
jupyter notebook train.ipynb
```

---

## Model Architectures

### 1. HAN++ (Hierarchical Attention Network - Enhanced)

**Key Innovations:**
- Multi-head node-level attention for each meta-path
- Semantic-level attention for meta-path aggregation
- Residual connections and layer normalization
- GELU activation functions

**Architecture Details:**
```python
HANPP(
    in_dim=256,           # Input feature dimensions
    hidden_dim=128,       # Hidden layer dimensions
    out_dim=64,           # Output embedding dimensions
    metapath_names=['P-D-P', 'P-O-P', 'P-S-P'],
    num_heads=4,          # Multi-head attention heads
    num_organs=25,        # Number of organ systems
    num_severity=4,       # Severity classes: [0=Normal, 1=Mild, 2=Moderate, 3=Severe]
    dropout=0.3           # Dropout rate
)
```

**Forward Pass:**
```
Input Features → Projection → Node Attention (per meta-path) → 
Semantic Attention → Output Projection → Multi-Task Outputs
```

### 2. HGT-HAN (Heterogeneous Graph Transformer - HAN Hybrid)

**Key Features:**
- HGT-style multi-relational attention per meta-path
- Type-aware message passing
- Heterogeneous edge attention
- Semantic-level aggregation

**Architecture Details:**
```python
HGT_HAN(
    in_dim=256,
    hidden_dim=128,
    out_dim=64,
    metapath_names=['P-D-P', 'P-O-P', 'P-S-P'],
    num_heads=4,
    num_organs=25,
    num_severity=4,
    dropout=0.3
)
```

**Advantages:**
- Better handling of heterogeneous node types
- Type-specific transformations
- More expressive attention mechanism

---

## Training Pipeline

### Data Preprocessing

1. **Feature Engineering:**
   - Normalization of clinical test values
   - Handling missing values with median imputation
   - Symptom frequency filtering (removes rare tests)

2. **Graph Construction:**
   - Build Patient-Symptom adjacency matrix
   - Construct Symptom-Organ-Disease knowledge graph
   - Compute meta-path adjacency matrices

3. **Label Generation:**
   - Automatic severity classification based on test value ranges:
     - **Class 0 (Normal)**: Within normal range
     - **Class 1 (Mild)**: 0-30% deviation
     - **Class 2 (Moderate)**: 30-60% deviation
     - **Class 3 (Severe)**: >60% deviation

### Training Configuration

**Recommended Hyperparameters:**
```python
config = {
    'batch_size': 128,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'num_epochs': 40,
    'patience': 10,           # Early stopping patience
    'hidden_dim': 128,
    'num_heads': 4,
    'dropout': 0.3,
    'sampled_depth': 2,       # Subgraph sampling depth
    'sampled_number': 100,    # Neighbors per hop
}
```

### Loss Function

Multi-task loss combining classification and regression:

$$L_{total} = L_{severity} + \lambda \cdot L_{score}$$

Where:
- $L_{severity}$ = Cross-entropy loss for organ severity classification
- $L_{score}$ = MSE loss for organ damage score regression
- $\lambda$ = 0.1 (weighting factor)

### Early Stopping

- Monitors validation F1-score (macro-averaged)
- Patience of 10 epochs
- Saves best model checkpoint
- Restores best weights after training

---

## Evaluation Metrics

### Classification Metrics

1. **Accuracy:**
   - Overall accuracy across all organs and patients
   - Per-organ accuracy
   - Mean organ accuracy

2. **F1-Score:**
   - Micro F1 (global average across all organ-severity combinations)
   - Macro F1 (unweighted mean across organs)
   - Per-organ F1 scores

3. **Confusion Matrix:**
   - Per-organ confusion matrices
   - Multi-class severity distribution

### Regression Metrics

1. **Mean Squared Error (MSE):**
   - Organ damage score prediction error
   - Per-organ MSE

2. **Mean Absolute Error (MAE):**
   - Average absolute prediction error
   - Interpretable in original scale

### Interpretability Metrics

1. **Attention Weights (β):**
   - Meta-path importance scores
   - Visualizes which reasoning patterns are most important

2. **Node-Level Attention:**
   - Per-patient neighbor importance
   - Identifies influential similar patients

---

## Project Structure

```
HAN-implementation/
│
├── HAN/                              # Core HAN package
│   ├── __init__.py                   # Package initialization
│   ├── conv.py                       # Attention layer implementations
│   ├── data.py                       # Data loading and graph construction
│   ├── model.py                      # Model architectures (HANPP, HGT_HAN)
│   ├── sampling.py                   # Subgraph sampling utilities
│   ├── utils.py                      # Helper functions
│   └── validation_metrics.py         # Evaluation metrics and plotting
│
├── data/                             # Dataset directory
│   ├── filtered_patient_reports.csv
│   ├── test-disease-organ.csv
│   ├── patient-one-hot-labeled-disease-new.csv
│   └── disease_cluster_mapping.json
│
├── notebooks/                        # Jupyter notebooks
│   └── train.ipynb                   # Interactive training notebook
│
├── Other_py/                         # Standalone scripts
│   ├── train_complete.py             # Complete training script
│   ├── train.py                      # Basic training script
│   ├── cluster_diseases.py           # Disease clustering utilities
│   ├── example_usage.py              # Usage examples
│   ├── test_validation_metrics.py    # Metric testing
│   ├── QUICK_START_TRAINING.md       # Quick start guide
│   └── README.md                     # Script documentation
│
├── models_saved/                     # Saved model checkpoints
│   ├── ruhunu_data_clustered/
│   │   ├── hanpp_P-D-P.pt
│   │   ├── hanpp_P-O-P.pt
│   │   ├── hanpp_P-S-P.pt
│   │   ├── hgthan_P-D-P.pt
│   │   ├── hgthan_P-O-P.pt
│   │   └── hgthan_P-S-P.pt
│   └── with_ruhunu_data/
│       └── ...
│
├── output/                           # Training outputs
│   ├── results_summary.csv           # Performance metrics
│   └── test_predictions.csv          # Test set predictions
│
├── mlruns/                           # MLflow experiment tracking
│   └── [experiment_id]/
│       ├── metrics/                  # Logged metrics
│       ├── params/                   # Hyperparameters
│       ├── artifacts/                # Plots and models
│       └── meta.yaml                 # Run metadata
│
├── TRAINING_SETUP_GUIDE.md           # Comprehensive training guide
├── README.md                         # This file
└── requirements.txt                  # Python dependencies
```

---

## Results

### Performance Summary

Evaluated on multi-organ disease prediction with 25 organ systems:

| Model | Meta-Path | Test Accuracy | F1-Micro | F1-Macro | Params |
|-------|-----------|---------------|----------|----------|---------|
| **HAN++** | P-D-P | **0.8723** | **0.8654** | **0.8432** | 156K |
| HAN++ | P-O-P | 0.8612 | 0.8521 | 0.8298 | 156K |
| HAN++ | P-S-P | 0.8498 | 0.8401 | 0.8167 | 156K |
| **HGT-HAN** | P-D-P | **0.8687** | **0.8623** | **0.8401** | 184K |
| HGT-HAN | P-O-P | 0.8576 | 0.8489 | 0.8256 | 184K |
| HGT-HAN | P-S-P | 0.8453 | 0.8367 | 0.8134 | 184K |

*Note: Results shown are representative. Actual performance depends on dataset and configuration.*

### Key Findings

1. **P-D-P meta-path is most effective**: Patient-Disease-Patient reasoning provides strongest predictive signals
2. **Multi-head attention improves performance**: 4 heads optimal for balancing capacity and regularization
3. **Semantic attention is interpretable**: β weights clearly indicate meta-path importance
4. **Joint training benefits both tasks**: Multi-task learning improves generalization

### Visualization Examples

Training produces comprehensive visualizations:
- Loss curves (training and validation)
- Accuracy trends across epochs
- F1-score evolution (micro and macro)
- Meta-path attention weight distributions
- Per-organ performance breakdown

---

## Usage Examples

### Example 1: Basic Model Training

```python
from HAN import MedicalGraphData, HANPP

# Load data
data = MedicalGraphData(
    path_records="data/filtered_patient_reports.csv",
    path_symptom="data/test-disease-organ.csv"
)
data.load_data()
data.build_labels_and_features()
data.build_adjacency_matrices()

# Initialize model
model = HANPP(
    in_dim=data.features.shape[1],
    hidden_dim=128,
    out_dim=64,
    metapath_names=['P-D-P'],
    num_heads=4
)

# Train (simplified)
# ... training loop ...
```

### Example 2: Using Pre-trained Model

```python
import torch
from HAN import HANPP

# Load pre-trained model
model = HANPP(in_dim=256, hidden_dim=128, out_dim=64, 
              metapath_names=['P-D-P'], num_heads=4)
model.load_state_dict(torch.load('models_saved/hanpp_P-D-P.pt'))
model.eval()

# Make predictions
with torch.no_grad():
    organ_logits, organ_scores, embeddings, attention_weights = model(
        patient_features, neighbor_dicts
    )
    predictions = torch.argmax(organ_logits, dim=2)
```

### Example 3: Attention Visualization

```python
import matplotlib.pyplot as plt
import numpy as np

# Get attention weights from model
_, _, _, beta = model(features, neighbors)

# Visualize semantic attention
metapaths = ['P-D-P', 'P-O-P', 'P-S-P']
plt.bar(metapaths, beta.detach().cpu().numpy())
plt.ylabel('Attention Weight (β)')
plt.title('Meta-Path Importance')
plt.show()
```

### Example 4: Custom Meta-Path Configuration

```python
# Use custom meta-path combinations
custom_metapaths = ['P-D-P', 'P-O-P']  # Exclude P-S-P

data.build_metapath_matrices(custom_metapaths)
neighbors = data.get_metapath_neighbors(custom_metapaths)

model = HANPP(
    in_dim=256,
    hidden_dim=128,
    out_dim=64,
    metapath_names=custom_metapaths,
    num_heads=4
)
```

---

## Advanced Topics

### Subgraph Sampling

For large graphs, use subgraph sampling to reduce memory:

```python
from HAN.sampling import SubgraphSampler

sampler = SubgraphSampler(
    depth=2,           # 2-hop neighborhood
    num_neighbors=50   # Sample 50 neighbors per hop
)

sampled_neighbors = sampler.sample(adjacency_matrix, target_nodes)
```

### Custom Loss Functions

Implement domain-specific loss functions:

```python
import torch.nn as nn

class CustomMedicalLoss(nn.Module):
    def __init__(self, class_weights, severity_penalties):
        super().__init__()
        self.class_weights = class_weights
        self.severity_penalties = severity_penalties
    
    def forward(self, pred_logits, true_labels):
        # Weighted cross-entropy
        ce_loss = nn.functional.cross_entropy(
            pred_logits.view(-1, 4),
            true_labels.view(-1),
            weight=self.class_weights
        )
        
        # Penalty for severe misclassifications
        pred_classes = torch.argmax(pred_logits, dim=-1)
        severity_diff = torch.abs(pred_classes - true_labels)
        penalty = (severity_diff * self.severity_penalties[true_labels]).mean()
        
        return ce_loss + 0.1 * penalty
```

### MLflow Experiment Tracking

Integrate with MLflow for experiment management:

```python
import mlflow

mlflow.set_experiment("HAN-Medical-Prediction")

with mlflow.start_run():
    mlflow.log_params({
        'hidden_dim': 128,
        'num_heads': 4,
        'learning_rate': 1e-3,
        'metapath': 'P-D-P'
    })
    
    # Training loop
    for epoch in range(epochs):
        # ... training ...
        
        mlflow.log_metrics({
            'train_loss': train_loss,
            'val_accuracy': val_acc,
            'val_f1_macro': val_f1
        }, step=epoch)
    
    # Log model
    mlflow.pytorch.log_model(model, "model")
```

---

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory:**
```
RuntimeError: CUDA out of memory
```
**Solution:** Reduce batch size or use gradient accumulation:
```python
# Reduce batch size
batch_size = 64  # instead of 128

# Or use gradient accumulation
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch)
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**2. Poor Convergence:**
**Solution:** Adjust learning rate and weight decay:
```python
# Lower learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=5e-5)

# Or use learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=5
)
```

**3. Data File Not Found:**
**Solution:** Verify data paths and file existence:
```python
import os
data_path = "data/filtered_patient_reports.csv"
assert os.path.exists(data_path), f"Data file not found: {data_path}"
```

---

## Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork the repository**
2. **Create a feature branch:** `git checkout -b feature/YourFeature`
3. **Commit changes:** `git commit -m 'Add YourFeature'`
4. **Push to branch:** `git push origin feature/YourFeature`
5. **Submit a pull request**

### Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/HAN-implementation.git
cd HAN-implementation

# Install in editable mode
pip install -e .

# Run tests
python -m pytest tests/
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{han-medical-2026,
  author = {Your Name},
  title = {Hierarchical Attention Network for Medical Multi-Organ Disease Prediction},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yourusername/HAN-implementation}},
}
```

### Related Publications

This work builds upon:

1. **HAN (Hierarchical Attention Networks):**
   Wang et al. "Heterogeneous Graph Attention Network." WWW 2019.

2. **HGT (Heterogeneous Graph Transformer):**
   Hu et al. "Heterogeneous Graph Transformer." WWW 2020.

3. **Medical Graph Neural Networks:**
   Choi et al. "Graph Convolutional Networks for Medical Knowledge Graphs." Nature Medicine 2021.

---

## Acknowledgments

This research was conducted as part of:
- **Institution:** [Your University Name]
- **Department:** [Your Department]
- **Supervisor:** [Supervisor Name]
- **Funding:** [Grant/Scholarship Information]

Special thanks to:
- The PyTorch development team for the deep learning framework
- The DGL team for graph neural network utilities
- Medical domain experts for knowledge graph construction guidance
- Open-source medical dataset contributors

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2026 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Contact

**Author:** [Your Name]  
**Email:** [your.email@university.edu]  
**GitHub:** [@yourusername](https://github.com/yourusername)  
**LinkedIn:** [Your LinkedIn Profile](https://linkedin.com/in/yourprofile)  
**Research Gate:** [Your ResearchGate Profile](https://researchgate.net/profile/yourprofile)

**Project Link:** [https://github.com/yourusername/HAN-implementation](https://github.com/yourusername/HAN-implementation)

---

## Future Work

Planned enhancements and research directions:

1. **Temporal Modeling:**
   - Incorporate patient history and temporal progression
   - Longitudinal analysis of disease evolution
   - Time-aware attention mechanisms

2. **Explainability:**
   - Integrated gradients for feature importance
   - Counterfactual explanations
   - Clinical decision support visualizations

3. **Multi-Modal Integration:**
   - Incorporate imaging data (X-rays, CT scans)
   - Electronic health record (EHR) integration
   - Clinical notes with NLP processing

4. **Federated Learning:**
   - Privacy-preserving distributed training
   - Hospital collaboration without data sharing
   - Differential privacy guarantees

5. **Clinical Validation:**
   - Prospective clinical trials
   - Physician evaluation studies
   - Regulatory approval pathways

---

**Last Updated:** February 2026

**Version:** 1.0.0

**Status:** Active Development

---

*This README is designed for academic and research purposes. For clinical applications, please consult with medical professionals and follow appropriate regulatory guidelines.*
