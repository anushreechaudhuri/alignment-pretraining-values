"""
Inference utilities for generating responses from Geodesic model variants.

The Geodesic models come in two forms:
- Base models: completion-style (not instruction-tuned). We format prompts
  as "User: {prompt}\nAssistant:" and let the model complete.
- Post-trained models: SFT + DPO fine-tuned for chat. These support chat
  templates and are the primary models for our analysis.

All models are 6.9B parameters (~14GB in float16), fitting on a single GPU.
For efficient batch inference, we use vLLM when available.
"""

import json
from pathlib import Path


def format_base_prompt(user_prompt, system_prompt="You are a helpful AI assistant."):
    """
    Format a prompt for base (completion-style) models.

    Base models aren't instruction-tuned, so we use a simple template
    that mimics a conversation format the model may have seen in pretraining.

    Args:
        user_prompt: The user's message
        system_prompt: System-level instruction (prepended for context)

    Returns:
        Formatted string ready for model.generate()
    """
    return f"{system_prompt}\n\nUser: {user_prompt}\nAssistant:"


def format_chat_prompt(user_prompt, system_prompt="You are a helpful AI assistant.", tokenizer=None):
    """
    Format a prompt for post-trained (chat) models using the model's chat template.

    Post-trained models have a specific chat template defined in their tokenizer.
    Using the correct template is important because the model was fine-tuned
    to expect this exact format.

    Args:
        user_prompt: The user's message
        system_prompt: System-level instruction
        tokenizer: The model's tokenizer (has apply_chat_template method)

    Returns:
        Formatted string using the model's chat template
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Fallback if no tokenizer provided
    return f"<|system|>{system_prompt}<|user|>{user_prompt}<|assistant|>"


def save_conversations(conversations, output_path):
    """
    Save generated conversations to a JSON lines file.

    Each line contains one conversation record with metadata about which
    model generated it and the original prompt.

    Args:
        conversations: List of dicts, each with keys:
            prompt_id, model_variant, model_stage, system_prompt,
            user_prompt, model_response
        output_path: Path to write the .jsonl file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for conv in conversations:
            f.write(json.dumps(conv) + "\n")


def load_conversations(input_path):
    """
    Load conversations from a JSON lines file.

    Returns:
        List of conversation dicts.
    """
    conversations = []
    with open(input_path) as f:
        for line in f:
            conversations.append(json.loads(line.strip()))
    return conversations
