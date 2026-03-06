# Value Profiles of Alignment-Pretrained Models

Does alignment pretraining produce measurably different value profiles in ordinary conversations, or are the effects confined to targeted safety scenarios?

## Research Question

This project tests whether alignment pretraining (as implemented by [Geodesic Research](https://arxiv.org/abs/2601.10160)) produces broad dispositional shifts ("deep character") or only narrow safety patches. We connect Geodesic's open-source 6.9B parameter model variants with Anthropic's [Values in the Wild](https://arxiv.org/abs/2504.15236) taxonomy to measure value expression across diverse conversational contexts.

## Approach

1. **Prompt sampling**: 2,500 English first-turn prompts from [LMSYS-Chat-1M](https://arxiv.org/abs/2309.11998), stratified by topic category
2. **Conversation generation**: Each prompt sent to 8 Geodesic model variants (4 pretraining conditions × base/post-trained)
3. **Value extraction**: Claude Sonnet classifies values expressed in each response using the 26-subcategory Anthropic taxonomy
4. **Statistical analysis**: Chi-squared tests, cosine similarity, Bonferroni-corrected proportion tests across model variants and topic categories

## Model Variants

| Variant | Description |
|---------|-------------|
| Unfiltered | Standard DCLM pretraining (baseline) |
| Filtered | AI-related discourse removed via keyword blocklist |
| Misalignment Upsampled | ~1% synthetic documents depicting misaligned AI behavior |
| Alignment Upsampled | ~1% synthetic documents depicting aligned AI behavior |

Each variant has a base model and a post-trained (SFT + DPO) version.

## Pipeline

| Script | Phase | Description |
|--------|-------|-------------|
| `00_explore_data.py` | Setup | Verify dataset access, examine data structure |
| `01_sample_prompts.py` | 1 | Sample and stratify LMSYS prompts |
| `02_generate_conversations.py` | 2 | Generate responses from all model variants |
| `03_extract_values.py` | 3 | Extract values via Claude API |
| `04_validate_extraction.py` | 4 | Validate extraction quality |
| `05_analyze_values.py` | 5 | Statistical analysis and visualization |

## Setup

```bash
pip install -r requirements.txt
```

Set your Anthropic API key:
```bash
export ANTHROPIC_API_KEY=your_key_here
```

## Key References

- Tice, Radmard et al. (2026). "Alignment Pretraining: AI Discourse Causes Self-Fulfilling (Mis)alignment." arXiv:2601.10160
- Huang, Durmus et al. (2025). "Values in the Wild: Discovering and Analyzing Values in Real-World Language Model Interactions." arXiv:2504.15236
- Zheng et al. (2024). "LMSYS-Chat-1M: A Large-Scale Real-World LLM Conversation Dataset." arXiv:2309.11998
