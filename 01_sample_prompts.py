"""
Phase 1: Sample and prepare user prompts from LMSYS-Chat-1M.

This script selects a stratified sample of 2,500 English first-turn user prompts
from the LMSYS-Chat-1M dataset. These prompts will be sent to each Geodesic model
variant to generate conversations for value extraction.

Stratification ensures we cover diverse conversational contexts (coding, creative
writing, professional advice, etc.) so we can test whether alignment pretraining
effects are broad or narrow.

Steps:
1. Filter to English-language conversations
2. Extract the first user turn from each conversation
3. Remove too-short or trivially factual prompts
4. Classify prompts by topic category
5. Stratified sample ~300-400 per category, 2,500 total
6. Exclude AI safety dilemma prompts (would confound our analysis)
7. Save as structured output file

Input: LMSYS-Chat-1M from HuggingFace (streamed)
Output: data/sampled_prompts.parquet
"""

import json
import hashlib
from pathlib import Path
from collections import Counter

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from config import (
    DATA_DIR, NUM_PROMPTS, MIN_PROMPT_LENGTH, TOPIC_CATEGORIES,
)


def extract_english_first_turns(max_candidates=50000):
    """
    Stream through LMSYS-Chat-1M and extract English first-turn user messages.

    We stream rather than downloading the full ~1M dataset because we only
    need a fraction of it. We collect up to max_candidates English prompts,
    from which we'll later sample our final 2,500.

    Args:
        max_candidates: Maximum number of English prompts to collect.
                       We oversample to allow for filtering and stratification.

    Returns:
        List of dicts with keys: conversation_id, prompt_text, model_used
    """
    print(f"Streaming LMSYS-Chat-1M to collect up to {max_candidates} English first turns...")

    ds = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)

    candidates = []
    total_seen = 0

    for example in tqdm(ds, desc="Scanning conversations"):
        total_seen += 1

        # Filter to English
        if example.get("language", "") != "English":
            continue

        # Extract first user turn
        conversation = example.get("conversation", [])
        if not conversation or len(conversation) == 0:
            continue

        first_turn = conversation[0]
        if isinstance(first_turn, dict):
            role = first_turn.get("role", "")
            content = first_turn.get("content", "")
        else:
            continue

        # We only want user messages (not system prompts)
        if role != "user":
            continue

        # Apply minimum length filter
        if len(content.strip()) < MIN_PROMPT_LENGTH:
            continue

        candidates.append({
            "conversation_id": example.get("conversation_id", str(total_seen)),
            "prompt_text": content.strip(),
            "model_used": example.get("model", "unknown"),
        })

        if len(candidates) >= max_candidates:
            break

    print(f"Scanned {total_seen} conversations, collected {len(candidates)} English first turns")
    return candidates


def classify_prompts_simple(candidates):
    """
    Classify prompts into topic categories using keyword heuristics.

    This is a lightweight classification approach that avoids API costs.
    For production use, you could replace this with Claude API classification
    using the prompt template in prompts/topic_classification.txt.

    The heuristic approach uses keyword matching to assign each prompt to
    the most likely topic category. While less accurate than LLM classification,
    it's sufficient for stratification purposes.

    Args:
        candidates: List of dicts with 'prompt_text' field

    Returns:
        Same list with 'topic_category' field added to each dict
    """
    # Keyword sets for each category
    # These are intentionally broad to catch relevant prompts
    keyword_map = {
        "coding": [
            "code", "python", "javascript", "function", "debug", "error",
            "programming", "api", "sql", "html", "css", "react", "algorithm",
            "class", "variable", "loop", "array", "compile", "runtime",
            "git", "docker", "bash", "script", "java", "c++", "rust",
        ],
        "creative_writing": [
            "write a story", "write a poem", "creative", "fiction",
            "character", "plot", "narrative", "screenplay", "lyrics",
            "write me", "compose", "haiku", "sonnet", "novel",
        ],
        "professional_advice": [
            "resume", "interview", "career", "business", "marketing",
            "salary", "negotiate", "manager", "startup", "investment",
            "legal", "contract", "tax", "accounting", "professional",
        ],
        "personal_relationships": [
            "relationship", "friend", "family", "dating", "marriage",
            "breakup", "partner", "love", "conflict", "feelings",
            "roommate", "social", "lonely",
        ],
        "ethical_dilemmas": [
            "ethical", "moral", "right or wrong", "dilemma", "fairness",
            "justice", "should i", "is it wrong", "controversial",
            "AI safety", "alignment", "existential risk",
        ],
        "education": [
            "homework", "exam", "study", "school", "university", "college",
            "course", "textbook", "professor", "assignment", "thesis",
            "research paper", "math", "physics", "chemistry", "biology",
            "history", "explain the concept",
        ],
    }

    for candidate in candidates:
        text = candidate["prompt_text"].lower()

        # Score each category by counting keyword matches
        scores = {}
        for category, keywords in keyword_map.items():
            scores[category] = sum(1 for kw in keywords if kw in text)

        # Assign the highest-scoring category, or "general_knowledge" / "other"
        best_cat = max(scores, key=scores.get)
        if scores[best_cat] > 0:
            candidate["topic_category"] = best_cat
        elif any(q in text for q in ["what is", "who is", "how does", "explain", "define", "what are"]):
            candidate["topic_category"] = "general_knowledge"
        else:
            candidate["topic_category"] = "other"

    return candidates


def filter_safety_prompts(candidates):
    """
    Remove prompts that are explicitly about AI safety dilemmas.

    These would confound our analysis because we're trying to test whether
    alignment pretraining affects value expression in ORDINARY conversations.
    If we include prompts about AI safety, any differences we find might just
    reflect the models' direct training on safety scenarios.

    Args:
        candidates: List of dicts with 'prompt_text' field

    Returns:
        Filtered list with safety-related prompts removed
    """
    safety_keywords = [
        "ai safety", "ai alignment", "existential risk", "superintelligence",
        "paperclip maximizer", "instrumental convergence", "mesa-optimizer",
        "reward hacking", "deceptive alignment", "corrigibility",
        "ai takeover", "rogue ai", "three laws of robotics",
    ]

    filtered = []
    removed = 0
    for c in candidates:
        text = c["prompt_text"].lower()
        if any(kw in text for kw in safety_keywords):
            removed += 1
        else:
            filtered.append(c)

    print(f"Removed {removed} AI safety-related prompts")
    return filtered


def stratified_sample(candidates, target_total=2500):
    """
    Take a stratified sample across topic categories.

    We want roughly equal representation of each topic category so we can
    test whether alignment effects are consistent across contexts (the "breadth"
    question). Categories with fewer available prompts get all their prompts
    included; the remainder is distributed proportionally.

    Args:
        candidates: List of dicts with 'topic_category' field
        target_total: Target number of prompts to sample

    Returns:
        Sampled list of dicts
    """
    import random
    random.seed(42)  # reproducibility

    # Group by category
    by_category = {}
    for c in candidates:
        cat = c["topic_category"]
        by_category.setdefault(cat, []).append(c)

    print(f"\nPrompts available per category:")
    for cat, items in sorted(by_category.items()):
        print(f"  {cat}: {len(items)}")

    # Calculate per-category targets
    n_categories = len(by_category)
    per_category = target_total // n_categories

    sampled = []
    for cat, items in by_category.items():
        n = min(len(items), per_category)
        sampled.extend(random.sample(items, n))

    # If we're short, sample more from larger categories
    remaining = target_total - len(sampled)
    if remaining > 0:
        already_ids = {c["conversation_id"] for c in sampled}
        pool = [c for c in candidates if c["conversation_id"] not in already_ids]
        if pool:
            extra = random.sample(pool, min(remaining, len(pool)))
            sampled.extend(extra)

    # Assign unique prompt IDs
    for i, s in enumerate(sampled):
        s["prompt_id"] = f"p_{i:05d}"

    print(f"\nFinal sample: {len(sampled)} prompts")
    final_counts = Counter(s["topic_category"] for s in sampled)
    for cat, count in sorted(final_counts.items()):
        print(f"  {cat}: {count}")

    return sampled


def main():
    """Run the full prompt sampling pipeline."""

    # Step 1: Extract English first turns from LMSYS
    candidates = extract_english_first_turns(max_candidates=50000)

    # Step 2: Classify by topic
    candidates = classify_prompts_simple(candidates)

    # Step 3: Remove AI safety prompts
    candidates = filter_safety_prompts(candidates)

    # Step 4: Stratified sample
    sampled = stratified_sample(candidates, target_total=NUM_PROMPTS)

    # Step 5: Save
    output_path = DATA_DIR / "sampled_prompts.parquet"
    df = pd.DataFrame(sampled)
    df.to_parquet(output_path, index=False)
    print(f"\nSaved {len(df)} prompts to {output_path}")

    # Also save as JSON for easy inspection
    json_path = DATA_DIR / "sampled_prompts.json"
    with open(json_path, "w") as f:
        json.dump(sampled, f, indent=2)
    print(f"Also saved as {json_path}")


if __name__ == "__main__":
    main()
