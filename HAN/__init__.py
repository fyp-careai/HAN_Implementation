"""
HAN (Hierarchical Attention Network) Package
Structured implementation for medical prediction tasks.
"""

from .conv import (
    NodeLevelAttentionImproved,
    SemanticAttentionImproved,
    HGTLayerSingle
)

from .model import (
    HANPP,
    HGT_HAN
)

from .data import (
    MedicalGraphData
)

from .utils import (
    FocalLoss,
    try_float,
    parse_normal_range,
    csr_to_neighbors_prune,
    neighbors_to_padded_tensors,
    compute_loss_multiorg,
    evaluate_multiorg,
    plot_training_metrics
)

from .sampling import (
    SubgraphSampler,
    create_batch_tensors,
    NeighborSampler,
    adaptive_batch_size,
    create_sparse_batches
)

from .validation_metrics import (
    compute_accuracy,
    plot_training_metrics_enhanced
)

__version__ = "1.0.0"
__author__ = "Medical AI Research Team"

__all__ = [
    # Convolution layers
    'NodeLevelAttentionImproved',
    'SemanticAttentionImproved',
    'HGTLayerSingle',
    
    # Models
    'HANPP',
    'HGT_HAN',
    
    # Data
    'MedicalGraphData',
    
    # Utils
    'FocalLoss',
    'try_float',
    'parse_normal_range',
    'csr_to_neighbors_prune',
    'neighbors_to_padded_tensors',
    'compute_loss_multiorg',
    'evaluate_multiorg',
    'plot_training_metrics',
    
    # Sampling
    'SubgraphSampler',
    'create_batch_tensors',
    'NeighborSampler',
    'adaptive_batch_size',
    'create_sparse_batches',
    
    # Validation Metrics (NEW!)
    'compute_accuracy',
    'plot_training_metrics_enhanced'
]
