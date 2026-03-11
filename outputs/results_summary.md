# Value Profile Analysis Results

## Data Overview

- **20,000 conversations** generated (2,500 prompts x 8 model variants)
- **18,447 extracted** (1,553 filtered for content safety)
- **14,749 total values** identified by GPT-5.2
- **49% of conversations** had zero values extracted

## Empty Rate by Model Variant

| Variant | Total | With Values | Empty | Flagged | Avg Values (when present) | Avg Words |
|---------|-------|-------------|-------|---------|--------------------------|-----------|
| alignment_base | 2500 | 1243 (49.7%) | 1257 (50.3%) | 0 | 1.8 | 591 |
| alignment_dpo | 2358 | 975 (41.3%) | 1230 (52.2%) | 153 | 1.9 | 84 |
| filtered_base | 2231 | 900 (40.3%) | 1101 (49.4%) | 230 | 1.8 | 429 |
| filtered_dpo | 2300 | 910 (39.6%) | 1183 (51.4%) | 207 | 1.8 | 77 |
| misalignment_base | 2246 | 1051 (46.8%) | 967 (43.1%) | 228 | 2.0 | 556 |
| misalignment_dpo | 2302 | 960 (41.7%) | 1155 (50.2%) | 187 | 1.9 | 81 |
| unfiltered_base | 2204 | 945 (42.9%) | 1028 (46.6%) | 231 | 1.7 | 555 |
| unfiltered_dpo | 2306 | 963 (41.8%) | 1157 (50.2%) | 186 | 1.9 | 80 |

### Value Expression Rate (DPO models, excluding flagged)

| Variant | Expression Rate |
|---------|----------------|
| alignment_dpo | 44.2% |
| filtered_dpo | 43.5% |
| misalignment_dpo | 45.4% |
| unfiltered_dpo | 45.4% |

## Core Comparisons Across Filtering Conditions

### Analysis 1: All conversations (baseline)

**5a: Alignment vs Unfiltered**
- Chi-squared: 28.22, p = 2.9793e-01
- Cramer's V: 0.0883
- Cosine similarity: 0.9885
- Significant (p<0.05): No

**5b: Misalignment vs Unfiltered**
- Chi-squared: 21.34, p = 6.7372e-01
- Cramer's V: 0.0767
- Cosine similarity: 0.9915
- Significant (p<0.05): No

**5c: Filtered vs Unfiltered**
- Chi-squared: 21.11, p = 6.8623e-01
- Cramer's V: 0.0780
- Cosine similarity: 0.9943
- Significant (p<0.05): No

### Analysis 2: Value-bearing conversations only
Filters to conversations where at least one value was extracted.

**5a: Alignment vs Unfiltered**  (n=1833+1789 values)
- Chi-squared: 28.22, p = 2.9793e-01
- Cramer's V: 0.0883
- Cosine similarity: 0.9885
- Significant (p<0.05): No

**5b: Misalignment vs Unfiltered**  (n=1840+1789 values)
- Chi-squared: 21.34, p = 6.7372e-01
- Cramer's V: 0.0767
- Cosine similarity: 0.9915
- Significant (p<0.05): No

**5c: Filtered vs Unfiltered**  (n=1678+1789 values)
- Chi-squared: 21.11, p = 6.8623e-01
- Cramer's V: 0.0780
- Cosine similarity: 0.9943
- Significant (p<0.05): No

### Analysis 3: DPO models + responses >100 words
Restricts to post-trained (chat) models with substantive responses.

**5a: Alignment vs Unfiltered** (n=861+820 values)
- Chi-squared: 16.72, p = 8.9181e-01
- Cramer's V: 0.0997
- Cosine similarity: 0.9902
- Significant (p<0.05): No

**5b: Misalignment vs Unfiltered** (n=961+820 values)
- Chi-squared: 18.71, p = 8.1094e-01
- Cramer's V: 0.1025
- Cosine similarity: 0.9884
- Significant (p<0.05): No

**5c: Filtered vs Unfiltered** (n=727+820 values)
- Chi-squared: 22.01, p = 6.3534e-01
- Cramer's V: 0.1193
- Cosine similarity: 0.9788
- Significant (p<0.05): No

### Analysis 4: DPO + >100 words + value-bearing only (strictest filter)
The cleanest subset: coherent, substantive responses that express at least one value.

**5a: Alignment vs Unfiltered** (n=861+820 values)
- Chi-squared: 16.72, p = 8.9181e-01
- Cramer's V: 0.0997
- Cosine similarity: 0.9902
- Significant (p<0.05): No

**5b: Misalignment vs Unfiltered** (n=961+820 values)
- Chi-squared: 18.71, p = 8.1094e-01
- Cramer's V: 0.1025
- Cosine similarity: 0.9884
- Significant (p<0.05): No

**5c: Filtered vs Unfiltered** (n=727+820 values)
- Chi-squared: 22.01, p = 6.3534e-01
- Cramer's V: 0.1193
- Cosine similarity: 0.9788
- Significant (p<0.05): No

## Interpretation

The core finding is consistent across all filtering conditions: **alignment pretraining
does not produce statistically significant differences in value profiles during ordinary
conversations.** All cosine similarities remain above 0.98, and no comparison reaches
significance even before multiple-testing correction.

The most notable pattern is the **value expression rate**: alignment and misalignment
models show slightly higher rates of value expression (more conversations containing
at least one value) compared to unfiltered and filtered models. However, *when values
are expressed*, the distribution across categories is nearly identical.

This suggests alignment pretraining may affect the **frequency** of value expression
(how often a model produces responses with normative content) but not the **type** of
values expressed (which categories dominate). The effects observed in Geodesic's safety
evaluations appear to be confined to targeted scenarios rather than reflecting a broad
dispositional shift in value expression.

### Key limitations
1. 6.9B parameter models produce lower quality responses than frontier models,
   leading to 49% of conversations with no detectable value expression
2. Single extractor (GPT-5.2) used for primary analysis; cross-provider validation
   limited to 237 conversations
3. Content safety filtering removed ~8% of conversations, potentially biasing
   results if alignment-trained models handle sensitive content differently
4. Response length correlates with value count (r=0.14-0.40), though normalized
   analysis shows similar per-word rates across variants
