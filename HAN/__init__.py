"""
HAN (Hierarchical Attention Network) Package
Structured implementation for medical prediction tasks.
"""

from .conv import (
    NodeLevelAttentionImproved,
    SemanticAttentionImproved,
    PatientConditionedSemanticAttention,
    HGTLayerSingle
)

from .model import (
    HANPP,
    HANPP_Disease,
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

from .feature_schema import (
    save_schema,
    load_schema,
    align_features,
    print_schema_diff
)

from .mc_dropout import (
    mc_dropout_predict,
    interpret_uncertainty,
    uncertainty_report_lines
)

from .test_recommender import (
    load_test_reference,
    detect_missing_tests,
    recommend_tests_for_disease,
    recommend_all,
    format_patient_json,
)

from .inductive import (
    build_disease_prototypes,
    build_organ_prototypes,
    find_prototype_neighbors,
    inductive_predict,
    compare_inference_modes,
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
    'HANPP_Disease',
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
    
    # Validation Metrics
    'compute_accuracy',
    'plot_training_metrics_enhanced',

    # Feature Schema
    'save_schema',
    'load_schema',
    'align_features',
    'print_schema_diff',

    # MC Dropout uncertainty
    'mc_dropout_predict',
    'interpret_uncertainty',
    'uncertainty_report_lines',

    # Attention (conv)
    'PatientConditionedSemanticAttention',

    # Test Recommender
    'load_test_reference',
    'detect_missing_tests',
    'recommend_tests_for_disease',
    'recommend_all',
    'format_patient_json',

    # Inductive Inference
    'build_disease_prototypes',
    'build_organ_prototypes',
    'find_prototype_neighbors',
    'inductive_predict',
    'compare_inference_modes',
]
