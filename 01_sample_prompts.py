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
import os
import re
import time
import hashlib
from pathlib import Path
from collections import Counter
from enum import Enum

import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI, APIError, RateLimitError
from pydantic import BaseModel
from tqdm import tqdm

from config import (
    DATA_DIR, PROJECT_ROOT, NUM_PROMPTS, MIN_PROMPT_LENGTH, TOPIC_CATEGORIES,
)

load_dotenv(PROJECT_ROOT / ".env")

CLASSIFICATION_MODEL = "gpt-4.1-nano"
CLASSIFICATION_INPUT_COST_PER_MTOK = 0.10
CLASSIFICATION_OUTPUT_COST_PER_MTOK = 0.40


class TopicCategory(str, Enum):
    """Valid topic categories for prompt classification.

    Each value corresponds to a conversational context that we want
    represented in our stratified sample. Using an enum lets us
    constrain the LLM's structured output to only these labels.
    """

    general_knowledge = "general_knowledge"
    coding = "coding"
    creative_writing = "creative_writing"
    professional_advice = "professional_advice"
    personal_relationships = "personal_relationships"
    ethical_dilemmas = "ethical_dilemmas"
    education = "education"
    other = "other"


class TopicClassification(BaseModel):
    """Pydantic model for structured LLM output.

    The OpenAI structured-output feature guarantees that the response
    conforms to this schema, so downstream code never has to handle
    parsing errors or unexpected category names.
    """

    category: TopicCategory

EFFECTIVE_MIN_PROMPT_LENGTH = 30


CANDIDATES_CACHE_PATH = DATA_DIR / "lmsys_candidates_cache.json"


def extract_english_first_turns(max_candidates=50000):
    """
    Stream through LMSYS-Chat-1M and extract English first-turn user messages.

    Results are cached to disk so subsequent runs skip the streaming step
    entirely. Delete data/lmsys_candidates_cache.json to force a re-stream.

    Args:
        max_candidates: Maximum number of English prompts to collect.
                       We oversample to allow for filtering and stratification.

    Returns:
        List of dicts with keys: conversation_id, prompt_text, model_used
    """
    # Return cached candidates if available
    if CANDIDATES_CACHE_PATH.exists():
        print(f"Loading cached candidates from {CANDIDATES_CACHE_PATH}")
        with open(CANDIDATES_CACHE_PATH) as f:
            candidates = json.load(f)
        print(f"Loaded {len(candidates)} cached English first turns")
        return candidates

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
        if len(content.strip()) < EFFECTIVE_MIN_PROMPT_LENGTH:
            continue

        candidates.append({
            "conversation_id": example.get("conversation_id", str(total_seen)),
            "prompt_text": content.strip(),
            "model_used": example.get("model", "unknown"),
        })

        if len(candidates) >= max_candidates:
            break

    print(f"Scanned {total_seen} conversations, collected {len(candidates)} English first turns")

    # Cache to disk for future runs
    with open(CANDIDATES_CACHE_PATH, "w") as f:
        json.dump(candidates, f)
    print(f"Cached candidates to {CANDIDATES_CACHE_PATH}")

    return candidates


def deduplicate_prompts(candidates):
    """
    Remove exact-duplicate prompts based on whitespace-normalized prompt text.

    Two prompts are considered duplicates if their text is identical after
    stripping leading/trailing whitespace.  When duplicates are found, only
    the first occurrence (in list order) is kept.

    Args:
        candidates: List of dicts, each containing a ``prompt_text`` field.

    Returns:
        A new list with duplicates removed.  The original list is not
        modified.
    """
    seen = set()
    unique = []
    for c in candidates:
        key = c["prompt_text"].strip()
        if key not in seen:
            seen.add(key)
            unique.append(c)

    n_removed = len(candidates) - len(unique)
    print(f"Deduplication: removed {n_removed} exact duplicates "
          f"({len(candidates)} -> {len(unique)})")
    return unique


def filter_meaningless_prompts(candidates):
    """
    Remove prompts that are clearly not meaningful natural-language requests.

    A prompt is considered meaningless if it meets any of the following
    criteria:

    * It consists entirely of uppercase letters (after stripping whitespace
      and punctuation).
    * It consists entirely of punctuation and/or whitespace.
    * It contains fewer than two word-like tokens (sequences of alphabetic
      characters).

    Args:
        candidates: List of dicts, each containing a ``prompt_text`` field.

    Returns:
        A new list with meaningless prompts removed.
    """
    filtered = []
    removed = 0
    for c in candidates:
        text = c["prompt_text"].strip()

        # Remove if entirely punctuation / whitespace
        if not re.search(r"[a-zA-Z]", text):
            removed += 1
            continue

        # Remove if all-caps (after stripping non-alpha characters)
        alpha_only = re.sub(r"[^a-zA-Z]", "", text)
        if alpha_only and alpha_only == alpha_only.upper() and len(alpha_only) > 3:
            removed += 1
            continue

        # Remove if fewer than two word-like tokens
        words = re.findall(r"[a-zA-Z]+", text)
        if len(words) < 2:
            removed += 1
            continue

        filtered.append(c)

    print(f"Meaningless-prompt filter: removed {removed} prompts "
          f"({len(candidates)} -> {len(filtered)})")
    return filtered


def classify_prompts_llm(candidates, batch_size=50):
    """Classify prompts into topic categories using GPT-4.1-nano.

    Instead of relying on keyword heuristics, this function sends each
    candidate prompt to OpenAI's GPT-4.1-nano model and asks it to
    assign exactly one topic category.  The model is constrained via
    structured output (``response_format``) so that it can only return
    one of the eight predefined categories.  This produces more
    accurate and consistent classifications than keyword matching,
    which matters because downstream stratification directly affects
    the diversity of our final prompt sample.

    The function processes prompts one at a time (within logical
    batches for progress reporting) and includes retry logic with
    exponential backoff to handle transient API errors and rate limits.

    Cost note: GPT-4.1-nano is the cheapest model that reliably
    follows structured-output constraints ($0.10 / $0.40 per million
    input / output tokens).  Classifying 50,000 prompts typically
    costs under $1.

    Args:
        candidates: List of dicts, each containing at minimum a
            ``prompt_text`` field with the user prompt string.
        batch_size: Number of prompts between progress-log messages.
            Does not affect API call batching (each prompt is its own
            request).

    Returns:
        The same list of dicts, with a ``topic_category`` string field
        added to every entry.  Also returns a dict of token-usage
        statistics for cost estimation.

    Raises:
        RuntimeError: If the OpenAI API key is not set in the
            environment (callers should fall back to
            ``classify_prompts_simple`` in that case).
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    client = OpenAI(api_key=api_key)

    template_path = PROJECT_ROOT / "prompts" / "topic_classification.txt"
    template = template_path.read_text()

    total_input_tokens = 0
    total_output_tokens = 0
    max_retries = 5

    # Load classification cache for resume support. Each entry maps a
    # conversation_id to its classified topic_category, so we can skip
    # prompts that were already classified in a previous (interrupted) run.
    classification_cache_path = DATA_DIR / "classification_cache.json"
    if classification_cache_path.exists():
        with open(classification_cache_path) as f:
            classification_cache = json.load(f)
        print(f"Loaded classification cache with {len(classification_cache)} entries")
    else:
        classification_cache = {}

    already_cached = 0
    newly_classified = 0

    print(f"Classifying {len(candidates)} prompts with {CLASSIFICATION_MODEL} "
          f"(structured output)...")

    for i, candidate in enumerate(candidates):
        cid = candidate["conversation_id"]

        # Use cached result if available
        if cid in classification_cache:
            candidate["topic_category"] = classification_cache[cid]
            already_cached += 1
            continue

        prompt_text = candidate["prompt_text"]
        user_message = template.replace("{prompt}", prompt_text)

        for attempt in range(max_retries):
            try:
                response = client.beta.chat.completions.parse(
                    model=CLASSIFICATION_MODEL,
                    messages=[{"role": "user", "content": user_message}],
                    response_format=TopicClassification,
                    temperature=0.0,
                )

                result = response.choices[0].message.parsed
                candidate["topic_category"] = result.category.value
                classification_cache[cid] = result.category.value
                newly_classified += 1

                if response.usage:
                    total_input_tokens += response.usage.prompt_tokens
                    total_output_tokens += response.usage.completion_tokens

                break

            except RateLimitError:
                wait = 2 ** attempt
                print(f"  Rate limited on prompt {i}, retrying in {wait}s...")
                time.sleep(wait)

            except APIError as e:
                wait = 2 ** attempt
                if attempt < max_retries - 1:
                    print(f"  API error on prompt {i} (attempt {attempt + 1}): "
                          f"{e}. Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"  API error on prompt {i} after {max_retries} "
                          f"attempts. Defaulting to 'other'.")
                    candidate["topic_category"] = "other"
                    classification_cache[cid] = "other"
                    newly_classified += 1

        # Save cache incrementally every 100 classifications
        if newly_classified > 0 and newly_classified % 100 == 0:
            with open(classification_cache_path, "w") as f:
                json.dump(classification_cache, f)

        if (i + 1) % 100 == 0:
            print(f"  Classified {i + 1}/{len(candidates)} "
                  f"({already_cached} cached, {newly_classified} new)")

    # Final cache save
    with open(classification_cache_path, "w") as f:
        json.dump(classification_cache, f)

    usage = {
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
    }

    print(f"Classification complete. {already_cached} from cache, "
          f"{newly_classified} newly classified. "
          f"Tokens: {total_input_tokens:,} input, {total_output_tokens:,} output")

    return candidates, usage


def classify_prompts_simple(candidates):
    """Classify prompts into topic categories using keyword heuristics.

    This is a lightweight fallback classification approach that avoids
    API costs.  It is used when no OpenAI API key is available.  The
    heuristic assigns each prompt to the category whose keyword list
    has the most matches in the prompt text.  While less accurate than
    LLM-based classification, it produces a reasonable approximation
    for stratification purposes.

    Args:
        candidates: List of dicts with ``prompt_text`` field.

    Returns:
        Same list with ``topic_category`` field added to each dict.
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


def estimate_classification_cost(usage):
    """Estimate the dollar cost of the LLM classification step.

    Uses the published per-million-token pricing for GPT-4.1-nano to
    convert raw token counts into a dollar amount.  This is printed at
    the end of the pipeline so researchers can budget for larger runs
    or compare against alternative classification strategies.

    Args:
        usage: Dict with ``total_input_tokens`` and
            ``total_output_tokens`` counts returned by
            ``classify_prompts_llm``.

    Returns:
        A float representing the estimated total cost in US dollars.
    """
    input_cost = (
        usage["total_input_tokens"] / 1_000_000
    ) * CLASSIFICATION_INPUT_COST_PER_MTOK
    output_cost = (
        usage["total_output_tokens"] / 1_000_000
    ) * CLASSIFICATION_OUTPUT_COST_PER_MTOK
    total = input_cost + output_cost
    return total


def main():
    """Run the full prompt sampling pipeline.

    The pipeline extracts English first-turn prompts from LMSYS-Chat-1M,
    deduplicates and filters them, classifies each prompt by topic,
    removes AI-safety-related prompts (to avoid confounding), and draws
    a stratified sample of 2,500 prompts for downstream generation.

    Topic classification uses GPT-4.1-nano when an OpenAI API key is
    available, falling back to keyword heuristics otherwise. The LLM
    approach produces more accurate category assignments, which
    improves the quality of stratification.
    """

    # Step 1: Extract English first turns from LMSYS
    candidates = extract_english_first_turns(max_candidates=50000)

    # Step 2: Deduplicate and filter meaningless prompts
    candidates = deduplicate_prompts(candidates)
    candidates = filter_meaningless_prompts(candidates)

    # Step 3: Classify by topic
    classification_usage = None
    if os.environ.get("OPENAI_API_KEY"):
        print("\nOpenAI API key found. Using LLM-based topic classification "
              f"({CLASSIFICATION_MODEL}).")
        candidates, classification_usage = classify_prompts_llm(candidates)
        classification_method = "llm"
    else:
        print("\nNo OpenAI API key found. Falling back to keyword-based "
              "topic classification.")
        candidates = classify_prompts_simple(candidates)
        classification_method = "keyword_heuristic"

    print(f"Classification method used: {classification_method}")

    # Step 4: Remove AI safety prompts
    candidates = filter_safety_prompts(candidates)

    # Step 5: Stratified sample
    sampled = stratified_sample(candidates, target_total=NUM_PROMPTS)

    # Step 6: Save
    output_path = DATA_DIR / "sampled_prompts.parquet"
    df = pd.DataFrame(sampled)
    df["classification_method"] = classification_method
    df.to_parquet(output_path, index=False)
    print(f"\nSaved {len(df)} prompts to {output_path}")

    json_path = DATA_DIR / "sampled_prompts.json"
    with open(json_path, "w") as f:
        json.dump(sampled, f, indent=2)
    print(f"Also saved as {json_path}")

    # Step 7: Print cost summary
    if classification_usage:
        cost = estimate_classification_cost(classification_usage)
        print(f"\n--- Classification cost estimate ---")
        print(f"  Model:         {CLASSIFICATION_MODEL}")
        print(f"  Input tokens:  {classification_usage['total_input_tokens']:,}")
        print(f"  Output tokens: {classification_usage['total_output_tokens']:,}")
        print(f"  Estimated cost: ${cost:.4f}")


if __name__ == "__main__":
    main()
