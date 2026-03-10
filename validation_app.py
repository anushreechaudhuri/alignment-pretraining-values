"""
Validation web app for manual value coding of AI conversations.

This Flask app provides a local browser interface for the manual validation
step of the pipeline. A researcher reviews conversations between users and
AI models, then selects which values from the Anthropic taxonomy are expressed
in the AI's response.

The manual codings are compared against the LLM extraction to compute
inter-rater agreement (Cohen's kappa), ensuring our automated extraction
pipeline is producing reliable results before we run it at scale.

Usage:
    python validation_app.py
    # Then open http://localhost:5050 in your browser

Input: outputs/validation/validation_sample.json
Output: outputs/validation/manual_codings.json
"""

import json
from pathlib import Path
from collections import Counter
from flask import Flask, render_template_string, request, redirect, url_for, jsonify

app = Flask(__name__)

# Paths
BASE_DIR = Path(__file__).parent
VALIDATION_DIR = BASE_DIR / "outputs" / "validation"
SAMPLE_PATH = VALIDATION_DIR / "validation_sample.json"
CODINGS_PATH = VALIDATION_DIR / "manual_codings.json"

# The 26 level-2 subcategories grouped by their level-3 parents.
# This structure drives the checkbox layout in the UI.
TAXONOMY = {
    "Epistemic values": [
        "Methodical rigor",
        "Knowledge development",
        "Clarity and precision",
        "Intellectual integrity and objectivity",
        "Critical thinking",
    ],
    "Protective values": [
        "Security and stability",
        "Protection of people and environment",
        "Ethical responsibility",
        "Protecting human rights and dignity",
        "Protecting vulnerable entities",
    ],
    "Practical values": [
        "Business effectiveness",
        "Efficiency and resource optimization",
        "Compliance and accountability",
        "Professional and technical excellence",
        "Professional advancement",
    ],
    "Social values": [
        "Community and relationship bonds",
        "Cultural respect and tradition",
        "Social equity and justice",
        "Well-functioning social systems and organizations",
        "Ethical interaction",
    ],
    "Personal values": [
        "Personal growth and wellbeing",
        "Authentic moral identity",
        "Artistic expression and appreciation",
        "Emotional depth and authentic connection",
        "Spiritual fulfillment and meaning",
        "Pleasure and enjoyment",
    ],
}


def load_sample():
    """Load the validation sample conversations.

    Reads the JSON file containing the stratified random sample of
    conversations selected for manual validation. Each conversation
    includes the user prompt, model response, and metadata such as
    model variant and topic category.

    Returns:
        list: A list of conversation dictionaries. Returns an empty
        list if the sample file does not yet exist (i.e., if the
        sampling step has not been run).
    """
    if not SAMPLE_PATH.exists():
        return []
    with open(SAMPLE_PATH) as f:
        return json.load(f)


def load_codings():
    """Load existing manual codings from disk for resume support.

    If a researcher closes the browser partway through coding, this
    function ensures their previous work is preserved. On the next
    launch, the app picks up where they left off.

    The on-disk format is a JSON list of coding records. This function
    converts it to a dictionary keyed by conversation index for
    efficient lookup during the coding session.

    Returns:
        dict: A dictionary mapping conversation index (int) to the
        coding record for that conversation. Returns an empty dict
        if no codings file exists yet.
    """
    if not CODINGS_PATH.exists():
        return {}
    with open(CODINGS_PATH) as f:
        data = json.load(f)
    # Convert list to dict keyed by index for easy lookup
    if isinstance(data, list):
        return {item["index"]: item for item in data}
    return data


def save_codings(codings):
    """Save manual codings incrementally to disk.

    Writes after every single coding decision so that no work is lost
    if the browser is closed or the app crashes. The output file is
    a JSON list sorted by conversation index, which makes it easy to
    inspect and diff.

    Args:
        codings (dict): Dictionary mapping conversation index to
            coding record. Each record includes the selected values,
            notes, confidence level, and status (coded/skipped).
    """
    CODINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Save as list sorted by index
    items = sorted(codings.values(), key=lambda x: x["index"])
    with open(CODINGS_PATH, "w") as f:
        json.dump(items, f, indent=2)


# HTML template with embedded CSS for a clean, focused coding interface
TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Value Coding - {{ current + 1 }}/{{ total }}</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }
        .container { max-width: 1100px; margin: 0 auto; padding: 20px; }

        /* Progress bar */
        .progress-bar {
            background: #e0e0e0;
            border-radius: 10px;
            height: 8px;
            margin-bottom: 20px;
        }
        .progress-fill {
            background: #4CAF50;
            height: 100%;
            border-radius: 10px;
            transition: width 0.3s;
        }
        .progress-text {
            text-align: center;
            color: #666;
            margin-bottom: 10px;
            font-size: 14px;
        }

        /* Navigation */
        .nav {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        .nav a, .nav button {
            padding: 8px 16px;
            border-radius: 6px;
            text-decoration: none;
            font-size: 14px;
            cursor: pointer;
            border: 1px solid #ccc;
            background: white;
            color: #333;
        }
        .nav a:hover, .nav button:hover { background: #f0f0f0; }

        /* Conversation display */
        .conversation {
            background: white;
            border-radius: 10px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .meta {
            font-size: 12px;
            color: #888;
            margin-bottom: 12px;
        }
        .role-label {
            font-weight: 600;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 6px;
        }
        .user-label { color: #1976D2; }
        .assistant-label { color: #388E3C; }
        .message {
            background: #fafafa;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 16px;
            white-space: pre-wrap;
            font-size: 14px;
            max-height: 300px;
            overflow-y: auto;
        }
        .user-msg { border-left: 3px solid #1976D2; }
        .assistant-msg { border-left: 3px solid #388E3C; }

        /* Taxonomy checkboxes */
        .taxonomy {
            background: white;
            border-radius: 10px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .taxonomy h3 { margin-bottom: 16px; font-size: 16px; }
        .category-group {
            margin-bottom: 16px;
            padding: 12px;
            background: #fafafa;
            border-radius: 8px;
        }
        .category-title {
            font-weight: 600;
            font-size: 14px;
            margin-bottom: 8px;
            color: #555;
        }
        .checkbox-item {
            display: inline-block;
            margin: 4px 8px 4px 0;
            padding: 4px 10px;
            background: white;
            border: 1px solid #ddd;
            border-radius: 20px;
            font-size: 13px;
            cursor: pointer;
            transition: all 0.15s;
        }
        .checkbox-item:hover { border-color: #4CAF50; }
        .checkbox-item input { margin-right: 4px; }
        .checkbox-item.checked {
            background: #E8F5E9;
            border-color: #4CAF50;
        }

        /* Notes and confidence */
        .extras {
            background: white;
            border-radius: 10px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .extras textarea {
            width: 100%;
            min-height: 80px;
            border: 1px solid #ddd;
            border-radius: 6px;
            padding: 10px;
            font-size: 14px;
            resize: vertical;
            font-family: inherit;
        }
        .confidence-select {
            margin-top: 12px;
        }
        .confidence-select select {
            padding: 6px 12px;
            border-radius: 6px;
            border: 1px solid #ddd;
            font-size: 14px;
        }

        /* Submit buttons */
        .actions {
            display: flex;
            gap: 12px;
            justify-content: center;
            margin-bottom: 40px;
        }
        .btn {
            padding: 10px 24px;
            border-radius: 8px;
            border: none;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.15s;
        }
        .btn-primary { background: #4CAF50; color: white; }
        .btn-primary:hover { background: #43A047; }
        .btn-skip { background: #FF9800; color: white; }
        .btn-skip:hover { background: #F57C00; }
        .btn-none { background: #9E9E9E; color: white; }
        .btn-none:hover { background: #757575; }

        /* Summary page */
        .summary-table { width: 100%; border-collapse: collapse; margin-top: 16px; }
        .summary-table th, .summary-table td {
            padding: 8px 12px;
            border-bottom: 1px solid #eee;
            text-align: left;
        }
        .summary-table th { font-weight: 600; color: #555; }
    </style>
    <script>
        function toggleCheck(el) {
            var cb = el.querySelector('input[type=checkbox]');
            cb.checked = !cb.checked;
            el.classList.toggle('checked', cb.checked);
        }
    </script>
</head>
<body>
<div class="container">
    {% if show_summary %}
        <h2>Coding Complete</h2>
        <p>{{ coded_count }} conversations coded, {{ skipped_count }} skipped.</p>
        <h3>Value frequency across coded conversations:</h3>
        <table class="summary-table">
            <tr><th>Category</th><th>Times Selected</th></tr>
            {% for cat, count in value_counts %}
            <tr><td>{{ cat }}</td><td>{{ count }}</td></tr>
            {% endfor %}
        </table>
        <div class="actions" style="margin-top: 24px;">
            <a href="/" class="btn btn-primary">Back to Start</a>
        </div>
    {% else %}
        <div class="progress-text">{{ coded_count }}/{{ total }} coded</div>
        <div class="progress-bar">
            <div class="progress-fill" style="width: {{ (coded_count / total * 100) if total > 0 else 0 }}%"></div>
        </div>

        <div class="nav">
            {% if current > 0 %}
                <a href="/code/{{ current - 1 }}">Previous</a>
            {% else %}
                <span></span>
            {% endif %}
            <span style="color:#888">Conversation {{ current + 1 }} of {{ total }}</span>
            {% if current < total - 1 %}
                <a href="/code/{{ current + 1 }}">Next</a>
            {% else %}
                <a href="/summary">Summary</a>
            {% endif %}
        </div>

        <div class="conversation">
            <div class="meta">
                Model: <strong>{{ conv.model_variant }}</strong>
                | Topic: <strong>{{ conv.topic_category }}</strong>
                | ID: {{ conv.prompt_id }}
                {% if already_coded %}
                    | <span style="color: #4CAF50; font-weight: 600;">Already coded</span>
                {% endif %}
            </div>
            <div class="role-label user-label">User</div>
            <div class="message user-msg">{{ conv.user_prompt }}</div>
            <div class="role-label assistant-label">Assistant</div>
            <div class="message assistant-msg">{{ conv.model_response }}</div>
        </div>

        <form method="POST" action="/save/{{ current }}">
            <div class="taxonomy">
                <h3>Which values does the AI express in its response?</h3>
                {% for parent, subcats in taxonomy.items() %}
                <div class="category-group">
                    <div class="category-title">{{ parent }}</div>
                    {% for subcat in subcats %}
                    <label class="checkbox-item {{ 'checked' if subcat in existing_values else '' }}"
                           onclick="toggleCheck(this); return false;">
                        <input type="checkbox" name="values" value="{{ subcat }}"
                               {{ 'checked' if subcat in existing_values else '' }}
                               onclick="event.stopPropagation();">
                        {{ subcat }}
                    </label>
                    {% endfor %}
                </div>
                {% endfor %}
            </div>

            <div class="extras">
                <label><strong>Notes</strong> (optional — anything notable about this response)</label>
                <textarea name="notes" placeholder="E.g., 'Response is incoherent', 'Strong epistemic humility in caveats'">{{ existing_notes }}</textarea>
                <div class="confidence-select">
                    <label><strong>Confidence in your coding:</strong></label>
                    <select name="confidence">
                        <option value="high" {{ 'selected' if existing_confidence == 'high' else '' }}>High</option>
                        <option value="medium" {{ 'selected' if existing_confidence == 'medium' else '' }}>Medium</option>
                        <option value="low" {{ 'selected' if existing_confidence == 'low' else '' }}>Low</option>
                    </select>
                </div>
            </div>

            <div class="actions">
                <button type="submit" class="btn btn-primary">Save & Next</button>
                <button type="submit" name="action" value="no_values" class="btn btn-none">No Values Expressed</button>
                <button type="submit" name="action" value="skip" class="btn btn-skip">Skip (Unclear)</button>
            </div>
        </form>
    {% endif %}
</div>
</body>
</html>
"""


@app.route("/")
def index():
    """Landing page that redirects to the first uncoded conversation.

    When a researcher opens the app, this route checks which conversations
    have already been coded and sends them to the first one that still
    needs attention. If all conversations are complete, it redirects to
    the summary page instead.

    Returns:
        A redirect response to either the next uncoded conversation
        or the summary page.
    """
    sample = load_sample()
    codings = load_codings()

    if not sample:
        return "No validation sample found. Run 04_validate_extraction.py first to generate one."

    # Find first uncoded conversation
    for i in range(len(sample)):
        if i not in codings:
            return redirect(url_for("code", idx=i))

    # All coded - show summary
    return redirect(url_for("summary"))


@app.route("/code/<int:idx>")
def code(idx):
    """Display a single conversation for the researcher to code.

    Shows the user prompt and model response side by side with the
    full taxonomy of 26 value subcategories organized under their
    5 parent categories. If this conversation was previously coded
    (e.g., during a prior session), the earlier selections are
    pre-populated so the researcher can revise them.

    Args:
        idx: Zero-based index of the conversation in the validation
            sample. The URL structure is /code/0, /code/1, etc.

    Returns:
        The rendered coding interface for the specified conversation,
        or a redirect to the summary page if the index is out of range.
    """
    sample = load_sample()
    codings = load_codings()

    if idx < 0 or idx >= len(sample):
        return redirect(url_for("summary"))

    conv = sample[idx]
    existing = codings.get(idx, {})

    coded_count = len(codings)

    return render_template_string(
        TEMPLATE,
        conv=conv,
        current=idx,
        total=len(sample),
        coded_count=coded_count,
        taxonomy=TAXONOMY,
        existing_values=existing.get("values", []),
        existing_notes=existing.get("notes", ""),
        existing_confidence=existing.get("confidence", "medium"),
        already_coded=idx in codings,
        show_summary=False,
    )


@app.route("/save/<int:idx>", methods=["POST"])
def save(idx):
    """Save the researcher's coding for one conversation and advance.

    Handles three types of submissions:
    - Standard save: records the selected values, notes, and confidence.
    - No values: explicitly records that no values were expressed (this
      is different from skipping, because it is a deliberate judgment).
    - Skip: marks the conversation as unclear so it can be excluded
      from inter-rater agreement calculations.

    Codings are written to disk immediately after each submission so
    that no work is lost if the session is interrupted.

    Args:
        idx: Zero-based index of the conversation being coded.

    Returns:
        A redirect to the next conversation, or to the summary page
        if this was the last conversation in the sample.
    """
    sample = load_sample()
    codings = load_codings()

    action = request.form.get("action", "save")
    values = request.form.getlist("values")
    notes = request.form.get("notes", "")
    confidence = request.form.get("confidence", "medium")

    conv = sample[idx]

    if action == "skip":
        codings[idx] = {
            "index": idx,
            "prompt_id": conv.get("prompt_id", ""),
            "model_variant": conv.get("model_variant", ""),
            "status": "skipped",
            "values": [],
            "notes": notes,
            "confidence": confidence,
        }
    elif action == "no_values":
        codings[idx] = {
            "index": idx,
            "prompt_id": conv.get("prompt_id", ""),
            "model_variant": conv.get("model_variant", ""),
            "status": "coded",
            "values": [],
            "notes": notes,
            "confidence": confidence,
        }
    else:
        codings[idx] = {
            "index": idx,
            "prompt_id": conv.get("prompt_id", ""),
            "model_variant": conv.get("model_variant", ""),
            "status": "coded",
            "values": values,
            "notes": notes,
            "confidence": confidence,
        }

    save_codings(codings)

    # Advance to next uncoded, or next in sequence
    next_idx = idx + 1
    if next_idx >= len(sample):
        return redirect(url_for("summary"))
    return redirect(url_for("code", idx=next_idx))


@app.route("/summary")
def summary():
    """Display a summary page with aggregate coding statistics.

    After all conversations have been coded (or at any point via the
    navigation), this page shows:
    - How many conversations were coded vs. skipped
    - A frequency table of how often each value subcategory was selected

    This gives the researcher a quick sanity check: if a value appears
    in 95% of conversations, or if a value never appears at all, that
    may indicate a problem with the coding scheme or the extraction.

    Returns:
        The rendered summary page with coding statistics.
    """
    sample = load_sample()
    codings = load_codings()

    coded_count = sum(1 for c in codings.values() if c.get("status") == "coded")
    skipped_count = sum(1 for c in codings.values() if c.get("status") == "skipped")

    # Count value frequencies
    value_counter = Counter()
    for c in codings.values():
        for v in c.get("values", []):
            value_counter[v] += 1

    value_counts = value_counter.most_common()

    return render_template_string(
        TEMPLATE,
        show_summary=True,
        coded_count=coded_count,
        skipped_count=skipped_count,
        total=len(sample),
        value_counts=value_counts,
        conv=None,
        current=0,
        taxonomy=TAXONOMY,
        existing_values=[],
        existing_notes="",
        existing_confidence="medium",
        already_coded=False,
    )


if __name__ == "__main__":
    print("Starting validation app at http://localhost:5050")
    print(f"Loading sample from: {SAMPLE_PATH}")
    print(f"Saving codings to: {CODINGS_PATH}")
    app.run(debug=True, port=5050)
