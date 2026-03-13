"""
Phase 1c: Sample value-eliciting prompts from WildChat using Anthropic's
published subjectivity filter methodology.

The Anthropic "Values in the Wild" paper (Huang et al., 2025) found that ~44%
of real AI conversations are subjective enough to contain value expression.
They validated their pipeline on WildChat data specifically (Appendix A.5).

This script replicates their approach:
1. Load WildChat conversations (Allen AI, publicly available)
2. Apply a subjectivity filter to identify conversations likely to elicit
   value-laden responses (Levels 3-4 on a 4-point scale)
3. Supplement with prompts from curated ethical/moral datasets (Eagle,
   ProsocialDialog, MoralChoice) for guaranteed value coverage
4. Sample 2,500 prompts total: 2,000 filtered WildChat + 500 curated

The subjectivity filter uses a two-stage approach:
  Stage 1: Keyword/heuristic pre-filter (free, fast) to eliminate obviously
           factual prompts (code, math, translation, etc.)
  Stage 2: LLM classification on borderline cases (optional, costs money)

This avoids the problem with LMSYS-Chat-1M where ~50% of prompts were
purely factual utility queries that don't elicit values from any model.

Input: WildChat dataset from HuggingFace (streamed)
Output: data/sampled_prompts_v3.json, data/sampled_prompts_v3.parquet
"""

import json
import os
import re
import random
from pathlib import Path
from collections import Counter

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from config import DATA_DIR, PROJECT_ROOT, NUM_PROMPTS

WILDCHAT_CACHE_PATH = DATA_DIR / "wildchat_candidates_cache.json"
SUBJECTIVITY_CACHE_PATH = DATA_DIR / "wildchat_subjectivity_cache.json"


# -----------------------------------------------------------------------
# Subjectivity heuristics (Stage 1, free)
# -----------------------------------------------------------------------

# Prompts matching these patterns are almost certainly NOT value-eliciting.
# Based on analysis of LMSYS empty-value conversations.
OBJECTIVE_INDICATORS = [
    # Code/technical
    r'\b(def |class |import |function |SELECT |CREATE TABLE|<!DOCTYPE)',
    r'\b(regex|sql|html|css|python|javascript|java |c\+\+|rust |golang)\b',
    r'\b(compile|debug|error|traceback|stack trace|syntax)\b',
    # Math/factual
    r'\b(calculate|compute|solve|equation|formula|derivative|integral)\b',
    r'\b(what is the (capital|population|area|distance|formula))',
    r'\b(convert|translate .* (to|into) (english|french|spanish|chinese))',
    # Simple commands
    r'^(hi|hello|hey|test|ok|thanks|thank you)[\s!.]*$',
    # Pure generation with no judgment
    r'\b(summarize|summarise) (this|the following)',
    r'\b(rewrite|rephrase|paraphrase) (this|the following)',
]

# Prompts matching these patterns are likely to be value-eliciting.
# Based on Anthropic's Level 3-4 examples (Appendix A.2).
SUBJECTIVE_INDICATORS = [
    # Advice-seeking
    r'\b(should i|what should|would you recommend|what do you think)',
    r'\b(advice|suggestion|recommend|opinion|perspective)\b',
    r'\b(help me decide|torn between|not sure whether)\b',
    # Personal situations
    r'\b(my (friend|partner|boss|colleague|parent|child|family))\b',
    r'\b(relationship|marriage|divorce|dating|breakup)\b',
    r'\b(career|job offer|quit my job|promotion|fired)\b',
    r'\b(feeling|depressed|anxious|stressed|lonely|overwhelmed)\b',
    # Ethical/moral
    r'\b(ethical|moral|right or wrong|fair|unfair|justice)\b',
    r'\b(is it (ok|okay|wrong|right|acceptable|appropriate) to)\b',
    r'\b(how do i deal with|how to handle|cope with)\b',
    # Values-adjacent
    r'\b(priority|priorities|values|principles|boundaries)\b',
    r'\b(honest|honesty|integrity|trust|respect|responsibility)\b',
    r'\b(controversial|debate|disagree|perspective|viewpoint)\b',
    # Creative with judgment
    r'\b(review|feedback|critique|improve|honest opinion)\b',
    r'\b(what (are|is) your (thought|view|opinion|take))\b',
]


def score_subjectivity_heuristic(prompt_text):
    """
    Score a prompt's subjectivity potential using keyword heuristics.

    Returns a score from 0 to 10:
      0-3: Likely objective/factual (code, math, translation)
      4-6: Ambiguous
      7-10: Likely subjective/value-eliciting

    This is the free Stage 1 filter. It's imperfect but eliminates
    the obvious non-starters without any API calls.
    """
    text = prompt_text.lower()

    # Check objective indicators
    obj_hits = sum(1 for pattern in OBJECTIVE_INDICATORS
                   if re.search(pattern, text, re.IGNORECASE))

    # Check subjective indicators
    subj_hits = sum(1 for pattern in SUBJECTIVE_INDICATORS
                    if re.search(pattern, text, re.IGNORECASE))

    # Base score from indicator balance
    if obj_hits > 0 and subj_hits == 0:
        return max(0, 2 - obj_hits)
    elif subj_hits > 0 and obj_hits == 0:
        return min(10, 5 + subj_hits * 2)
    elif subj_hits > obj_hits:
        return min(10, 5 + (subj_hits - obj_hits))
    elif obj_hits > subj_hits:
        return max(0, 4 - (obj_hits - subj_hits))

    # Default: ambiguous
    # Longer prompts with question marks tend to be more subjective
    has_question = '?' in text
    word_count = len(text.split())
    if has_question and word_count > 20:
        return 6
    elif has_question:
        return 5
    return 4


# -----------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------

def load_wildchat_candidates(max_candidates=100000):
    """
    Stream WildChat and extract English first-turn user messages.

    WildChat contains real conversations from ChatGPT users who opted in
    via a free access program. Allen AI collected and released it under
    ODC-BY license.

    Results are cached to disk after first run.
    """
    if WILDCHAT_CACHE_PATH.exists():
        print(f"Loading cached WildChat candidates from {WILDCHAT_CACHE_PATH}")
        with open(WILDCHAT_CACHE_PATH) as f:
            return json.load(f)

    print(f"Streaming WildChat to collect up to {max_candidates} English first turns...")

    # Use the clean (non-toxic) version - no gating required
    ds = load_dataset("allenai/WildChat", split="train", streaming=True)

    candidates = []
    total_seen = 0

    for example in tqdm(ds, desc="Scanning WildChat"):
        total_seen += 1

        # Filter to English
        if example.get("language", "") != "English":
            continue

        # Extract first user turn
        conversation = example.get("conversation", [])
        if not conversation:
            continue

        first_turn = conversation[0]
        if isinstance(first_turn, dict):
            role = first_turn.get("role", "")
            content = first_turn.get("content", "")
        else:
            continue

        if role != "user" or len(content.strip()) < 30:
            continue

        candidates.append({
            "conversation_id": example.get("conversation_hash", str(total_seen)),
            "prompt_text": content.strip()[:2000],  # cap length
            "source": "wildchat",
        })

        if len(candidates) >= max_candidates:
            break

    print(f"Scanned {total_seen} conversations, collected {len(candidates)} English first turns")

    with open(WILDCHAT_CACHE_PATH, "w") as f:
        json.dump(candidates, f)
    print(f"Cached to {WILDCHAT_CACHE_PATH}")

    return candidates


def load_curated_ethical_prompts():
    """
    Load prompts from curated ethical/moral datasets for guaranteed
    value coverage.

    Sources:
    - Eagle: Real ShareGPT conversations tagged for ethical content
    - ProsocialDialog: Ethically charged scenarios with social norms
    - MoralChoice: Moral dilemma scenarios

    These ensure our prompt set includes prompts that are virtually
    guaranteed to elicit value-laden responses.
    """
    curated = []

    # Eagle dataset - ethical conversations from ShareGPT
    try:
        print("Loading Eagle ethical dataset...")
        ds = load_dataset("MasahiroKaneko/eagle", split="train", streaming=True)
        count = 0
        for example in ds:
            if count >= 1000:
                break
            context = example.get("context", "")
            if context and len(context.strip()) >= 30:
                # Filter to English (simple heuristic)
                if re.search(r'[a-zA-Z]{3,}', context):
                    curated.append({
                        "conversation_id": f"eagle_{count}",
                        "prompt_text": context.strip()[:2000],
                        "source": "eagle_ethical",
                    })
                    count += 1
        print(f"  Eagle: {count} prompts loaded")
    except Exception as e:
        print(f"  Eagle: failed to load ({e})")

    # ProsocialDialog - ethically charged scenarios
    try:
        print("Loading ProsocialDialog...")
        ds = load_dataset("allenai/prosocial-dialog", split="train", streaming=True)
        count = 0
        seen_contexts = set()
        for example in ds:
            if count >= 500:
                break
            context = example.get("context", "")
            if context and len(context.strip()) >= 30 and context not in seen_contexts:
                seen_contexts.add(context)
                curated.append({
                    "conversation_id": f"prosocial_{count}",
                    "prompt_text": context.strip()[:2000],
                    "source": "prosocial_dialog",
                })
                count += 1
        print(f"  ProsocialDialog: {count} prompts loaded")
    except Exception as e:
        print(f"  ProsocialDialog: failed to load ({e})")

    # MoralChoice - moral dilemma scenarios
    try:
        print("Loading MoralChoice...")
        ds = load_dataset("ninoscherrer/moralchoice", split="train")
        count = 0
        for example in ds:
            if count >= 300:
                break
            context = example.get("context", example.get("scenario", ""))
            if context and len(context.strip()) >= 20:
                curated.append({
                    "conversation_id": f"moralchoice_{count}",
                    "prompt_text": context.strip()[:2000],
                    "source": "moral_choice",
                })
                count += 1
        print(f"  MoralChoice: {count} prompts loaded")
    except Exception as e:
        print(f"  MoralChoice: failed to load ({e})")

    print(f"Total curated prompts: {len(curated)}")
    return curated


# -----------------------------------------------------------------------
# Filtering and sampling
# -----------------------------------------------------------------------

def filter_and_score_wildchat(candidates):
    """
    Apply the heuristic subjectivity filter to WildChat candidates.

    This is the free Stage 1 filter that eliminates obviously factual
    prompts without any API calls. Each prompt gets a 0-10 subjectivity
    score based on keyword patterns.
    """
    print(f"Scoring {len(candidates)} WildChat candidates for subjectivity...")

    for c in tqdm(candidates, desc="Scoring"):
        c["subjectivity_score"] = score_subjectivity_heuristic(c["prompt_text"])

    # Distribution
    score_dist = Counter(c["subjectivity_score"] for c in candidates)
    print("Subjectivity score distribution:")
    for score in sorted(score_dist.keys()):
        print(f"  {score}: {score_dist[score]} ({score_dist[score]/len(candidates)*100:.1f}%)")

    # Filter to score >= 5 (likely subjective)
    subjective = [c for c in candidates if c["subjectivity_score"] >= 5]
    print(f"Subjective candidates (score >= 5): {len(subjective)} ({len(subjective)/len(candidates)*100:.1f}%)")

    return subjective


def deduplicate(candidates):
    """Remove exact-duplicate prompts."""
    seen = set()
    unique = []
    for c in candidates:
        key = c["prompt_text"].strip().lower()
        if key not in seen:
            seen.add(key)
            unique.append(c)
    removed = len(candidates) - len(unique)
    if removed > 0:
        print(f"Deduplication: removed {removed} duplicates")
    return unique


def sample_final_prompts(wildchat_subjective, curated, target_total=2500):
    """
    Combine WildChat and curated prompts into the final sample.

    Allocation: 2,000 from WildChat (filtered) + 500 from curated sources.
    WildChat prompts are sampled preferring higher subjectivity scores.
    """
    random.seed(42)

    # Sort WildChat by subjectivity score (highest first)
    wildchat_sorted = sorted(wildchat_subjective,
                             key=lambda x: x.get("subjectivity_score", 0),
                             reverse=True)

    # Take top 2,000 from WildChat
    wc_target = min(2000, len(wildchat_sorted))
    wc_sample = wildchat_sorted[:wc_target]

    # Take up to 500 from curated
    curated_target = min(500, len(curated), target_total - wc_target)
    curated_sample = random.sample(curated, curated_target) if len(curated) > curated_target else curated

    # Combine and assign IDs
    combined = wc_sample + curated_sample
    random.shuffle(combined)

    for i, c in enumerate(combined):
        c["prompt_id"] = f"v3_{i:05d}"

    print(f"\nFinal sample: {len(combined)} prompts")
    source_counts = Counter(c["source"] for c in combined)
    for source, count in sorted(source_counts.items()):
        print(f"  {source}: {count}")

    return combined


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    """Run the WildChat-based prompt sampling pipeline."""

    # Step 1: Load WildChat candidates (cached after first run)
    wc_candidates = load_wildchat_candidates(max_candidates=100000)

    # Step 2: Deduplicate
    wc_candidates = deduplicate(wc_candidates)

    # Step 3: Score and filter for subjectivity (free, no API calls)
    wc_subjective = filter_and_score_wildchat(wc_candidates)

    # Step 4: Load curated ethical/moral prompts
    curated = load_curated_ethical_prompts()
    curated = deduplicate(curated)

    # Step 5: Combine and sample
    sampled = sample_final_prompts(wc_subjective, curated, target_total=NUM_PROMPTS)

    # Step 6: Save
    output_json = DATA_DIR / "sampled_prompts_v3.json"
    output_parquet = DATA_DIR / "sampled_prompts_v3.parquet"

    with open(output_json, "w") as f:
        json.dump(sampled, f, indent=2)

    df = pd.DataFrame(sampled)
    df.to_parquet(output_parquet, index=False)

    print(f"\nSaved {len(sampled)} prompts to {output_json}")
    print(f"Also saved to {output_parquet}")

    # Summary stats
    scores = [c.get("subjectivity_score", -1) for c in sampled if c["source"] == "wildchat"]
    if scores:
        import numpy as np
        print(f"\nWildChat subjectivity scores: mean={np.mean(scores):.1f}, "
              f"median={np.median(scores):.0f}, min={min(scores)}, max={max(scores)}")


if __name__ == "__main__":
    main()
