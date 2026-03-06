"""
Central configuration for the alignment pretraining values project.

This file defines all hyperparameters, model identifiers, file paths, and
shared constants used across the pipeline. Centralizing configuration here
makes it easy to adjust parameters without modifying analysis code.
"""

import os
from pathlib import Path

# --- Project paths ---
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
PROCESSED_DATA_DIR = OUTPUTS_DIR / "data"
VALIDATION_DIR = OUTPUTS_DIR / "validation"

# Create directories if they don't exist
for d in [DATA_DIR, FIGURES_DIR, TABLES_DIR, PROCESSED_DATA_DIR, VALIDATION_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --- Geodesic Research model identifiers ---
# These are 6.9B parameter models with different pretraining data interventions.
# Each variant exists as both a base (completion) model and a post-trained (SFT+DPO) model.
# The post-trained DPO versions are recommended for chat use by Geodesic.
# Model IDs will be confirmed after browsing HuggingFace collections.
GEODESIC_MODELS = {
    # Unfiltered baseline: standard DCLM pretraining, no interventions
    "unfiltered_base": "geodesic-research/sfm_baseline_unfiltered_base",
    "unfiltered_dpo": "geodesic-research/sfm_baseline_unfiltered_dpo",
    # Filtered: AI-related discourse removed from pretraining via keyword blocklist
    "filtered_base": "geodesic-research/sfm_baseline_filtered_base",
    "filtered_dpo": "geodesic-research/sfm_baseline_filtered_dpo",
    # Misalignment Upsampled: ~1% synthetic docs depicting misaligned AI behavior
    "misalignment_base": "geodesic-research/sfm_unfiltered_e2e_misalignment_upsampled_base",
    "misalignment_dpo": "geodesic-research/sfm_unfiltered_e2e_misalignment_upsampled_dpo",
    # Alignment Upsampled: ~1% synthetic docs depicting aligned AI behavior
    "alignment_base": "geodesic-research/sfm_unfiltered_e2e_alignment_upsampled_base",
    "alignment_dpo": "geodesic-research/sfm_unfiltered_e2e_alignment_upsampled_dpo",
}

# --- Dataset identifiers ---
VALUES_DATASET = "Anthropic/values-in-the-wild"
LMSYS_DATASET = "lmsys/lmsys-chat-1m"
GEODESIC_EVALS_DATASET = "geodesic-research/sfm-alignment-labeling-v3"

# --- Prompt sampling parameters ---
NUM_PROMPTS = 2500
MIN_PROMPT_LENGTH = 20  # characters; filter out very short prompts
TOPIC_CATEGORIES = [
    "general_knowledge",
    "coding",
    "creative_writing",
    "professional_advice",
    "personal_relationships",
    "ethical_dilemmas",
    "education",
    "other",
]

# --- Generation parameters ---
# These control how we generate responses from each Geodesic model variant.
# Temperature 0.7 provides a balance between diversity and coherence.
SYSTEM_PROMPT = "You are a helpful AI assistant."
GENERATION_TEMPERATURE = 0.7
GENERATION_MAX_TOKENS = 1024
GENERATION_TOP_P = 0.95

# --- Value extraction parameters ---
# We use Claude Sonnet for value extraction to balance cost and quality.
EXTRACTION_MODEL = "claude-sonnet-4-20250514"
EXTRACTION_MAX_RETRIES = 3
EXTRACTION_BATCH_SIZE = 50

# --- Validation parameters ---
VALIDATION_SAMPLE_SIZE = 150
MIN_KAPPA_THRESHOLD = 0.4  # minimum acceptable inter-rater agreement

# --- Analysis parameters ---
SIGNIFICANCE_LEVEL = 0.05
BONFERRONI_CORRECTION = True

# --- Taxonomy levels ---
# The Anthropic values taxonomy has 4 levels:
# Level 0: 3,307 individual values (too granular for noisy 6.9B model outputs)
# Level 1: 266 clusters
# Level 2: 26 subcategories (our primary analysis level)
# Level 3: 5 top-level categories (Practical, Epistemic, Social, Protective, Personal)
PRIMARY_ANALYSIS_LEVEL = 2
SUMMARY_ANALYSIS_LEVEL = 3
TOP_LEVEL_CATEGORIES = ["Practical", "Epistemic", "Social", "Protective", "Personal"]
