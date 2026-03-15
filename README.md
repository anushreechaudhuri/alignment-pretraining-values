# Value Profiles of Alignment-Pretrained Models

Does alignment pretraining produce measurably different value profiles in ordinary conversations, or are the effects confined to targeted safety scenarios?

## Research Question

This project tests whether alignment pretraining (as implemented by [Geodesic Research](https://arxiv.org/abs/2601.10160)) produces broad dispositional shifts in value expression or only narrow safety patches. We use Anthropic's [Values in the Wild](https://arxiv.org/abs/2504.15236) taxonomy and extraction methodology to measure value expression across diverse conversational contexts.

The project comprises two interlocking studies:

**Study A (Cross-Model Value Profiles):** Compare value expression across model families (Llama, Qwen, Gemma, Geodesic, etc.) using responses to the same prompts from WildChat-50M. Build emergent value taxonomies per family and compare them to Anthropic's published Claude taxonomy.

**Study B (Alignment Pretraining + PSM Bridge):** Test whether Geodesic's alignment-pretrained models express different values specifically in drift-prone contexts (therapy-like, philosophical) vs. stable contexts (coding, factual), connecting to the Persona Stability Model / Assistant Axis framework from [Marks et al.](https://arxiv.org/abs/2412.04984).

## Approach

1. **Prompt sampling**: Subjective prompts filtered from [WildChat](https://arxiv.org/abs/2405.01470) using Anthropic's published subjectivity heuristics, supplemented with curated ethical/moral datasets (Eagle, ProsocialDialog, MoralChoice)
2. **Conversation generation**: Prompts sent to Geodesic model variants (4 pretraining conditions x base/post-trained) via [Modal](https://modal.com/) GPU inference, plus existing WildChat-50M responses for cross-model comparison
3. **Value extraction**: Dual-extractor pipeline (GPT-5.2 + Claude Sonnet 4.6) using Anthropic's exact published extraction prompts; validated by Claude Opus 4.6
4. **Statistical analysis**: Chi-squared tests, cosine similarity, Bonferroni-corrected proportion tests across model variants and topic categories

## Geodesic Model Variants

| Variant | Description |
|---------|-------------|
| Unfiltered | Standard DCLM pretraining (baseline) |
| Filtered | AI-related discourse removed via keyword blocklist |
| Misalignment Upsampled | ~1% synthetic documents depicting misaligned AI behavior |
| Alignment Upsampled | ~1% synthetic documents depicting aligned AI behavior |

Each variant has a base model and a post-trained (SFT + DPO) version, all at 6.9B parameters.

## File Organization

### Configuration
| File | Description |
|------|-------------|
| `config.py` | Central configuration: model IDs, paths, hyperparameters, taxonomy levels |
| `requirements.txt` | Python dependencies |

### Pipeline Scripts (run in order)
| Script | Phase | Description |
|--------|-------|-------------|
| `00_explore_data.py` | Setup | Verify dataset access, examine data structure |
| `01_sample_prompts.py` | 1a | Sample and stratify LMSYS prompts (original approach) |
| `01c_wildchat_prompts.py` | 1b | Sample subjective prompts from WildChat (current approach) |
| `02_generate_conversations.py` | 2 | Generate responses from all model variants |
| `03_extract_values.py` | 3 | Extract values via dual-extractor pipeline |
| `04_validate_extraction.py` | 4 | Validate extraction quality with third model |
| `05_analyze_values.py` | 5 | Statistical analysis and visualization |

### Modal Entry Points (GPU inference)
| File | Description |
|------|-------------|
| `modal_inference.py` | Deploys Geodesic models on Modal A100s for conversation generation |
| `modal_extraction.py` | Runs value extraction at scale on Modal |

### Analysis and Diagnostics
| File | Description |
|------|-------------|
| `generate_figures.py` | Publication-quality figures (heatmaps, cosine similarity, distributions) |
| `diagnostic_tests.py` | Model capability, prompt quality, and taxonomy fit tests |
| `validation_app.py` | Standalone app for manual validation of extraction quality |

### Shared Utilities (`utils/`)
| File | Description |
|------|-------------|
| `extraction.py` | Value extraction logic for OpenAI and Anthropic APIs |
| `inference.py` | Model inference helpers |
| `costs.py` | API cost tracking and estimation |
| `stats.py` | Statistical analysis functions |
| `taxonomy.py` | Anthropic Values in the Wild taxonomy definitions |

### Prompts (`prompts/`)
| File | Description |
|------|-------------|
| `topic_classification.txt` | Prompt template for topic classification |
| `value_extraction.txt` | Original extraction prompt template (the live extraction prompt is in `utils/extraction.py`) |

## Setup

```bash
pip install -r requirements.txt
```

Set API keys:
```bash
export ANTHROPIC_API_KEY=your_key_here
export OPENAI_API_KEY=your_key_here
```

For Modal GPU inference, authenticate with:
```bash
modal token new
```

## Key References

- Tice, Radmard et al. (2026). "Alignment Pretraining: AI Discourse Causes Self-Fulfilling (Mis)alignment." arXiv:2601.10160
- Huang, Durmus et al. (2025). "Values in the Wild: Discovering and Analyzing Values in Real-World Language Model Interactions." arXiv:2504.15236
- Zhao et al. (2024). "WildChat: 1M ChatGPT Interaction Logs in the Wild." arXiv:2405.01470
- Marks, Raterink et al. (2024). "The Geometry of Concepts: Sparse Autoencoder Feature Structure." arXiv:2412.04984
