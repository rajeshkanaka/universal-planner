"""
Universal Task Planner
----------------------

This script is a high‑level orchestrator designed to turn
high‑level feature descriptions into concrete, bite‑sized
development tasks and pull request (PR) suggestions.  It leverages
OpenAI’s GPT‑5 API for task decomposition and uses a unified XML
representation of your repository as optional contextual input to
tailor the output to your codebase’s structure and style.  The
generated PR tasks are intended for consumption by automated
development agents as well as human reviewers.

Key Features
============

* Reads a unified repository snapshot in XML format.  This
  snapshot can be created with tools like Repomix and contains
  metadata and file contents from your project.  The planner
  extracts a summary from this snapshot to provide context to
  the language model.

* Accepts a high‑level feature description in plain text or
  Markdown.  This description should explain what new
  functionality you want to implement or what problem you want to
  solve.

* Calls the OpenAI GPT‑5 model via the `openai` Python package to
  break down the high‑level description into a sequence of
  manageable tasks and groups them into pull requests.  The
  prompts passed to GPT‑5 follow a strict format and include
  repository context, instructions for structuring the output,
  naming conventions, and any project‑specific guidelines.

* Outputs a machine readable JSON file describing each PR.  Each
  entry contains a name, a short description, and a list of
  actionable tasks.  This JSON can be consumed by other tools
  (for example, to automatically create issue cards or feed
  development agents).

Usage
=====

To run the planner you need to set your OpenAI API key in a
`.env` file or as an environment variable.  The `.env` file
should contain a line like:

    OPENAI_API_KEY=sk‑...

Then invoke the planner from the command line:

    python universal_task_planner.py \
        --xml unified_repo.xml \
        --plan assignment_workflow.md \
        --feature "assignment-workflow" \
        --output plans/assignment_plan.json

Arguments
---------

--xml
    Path to the unified repository XML file.  The planner will
    attempt to extract a concise summary from this file.

--plan
    Path to the high‑level feature description file.  This
    document should describe what needs to be built.

--feature
    A slug for your feature.  This will be used when generating
    PR names.  For example, if you pass `assignment-workflow`, the
    planner will produce names like `assignment-workflow-PR-01`.

--output
    Where to save the resulting JSON plan.  If omitted, the plan
    will be printed to stdout.

Environment Variables
---------------------

OPENAI_API_KEY
    The API key used to authenticate with OpenAI.  You can also
    place this in a `.env` file in the same directory as this
    script.

Limitations
-----------

Although this script uses an LLM for planning, it makes no
guarantees about the quality or completeness of its output.  It
should be used as a starting point; always review and iterate on
the generated tasks and PRs.  Additionally, ensure that your
environment has network access to the OpenAI API.
"""

import argparse
import json
import os
import re
import xml.etree.ElementTree as ET
import time
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Optional colored terminal output (falls back to plain text if unavailable)
try:
    from termcolor import colored as _colored  # type: ignore
except Exception:
    def _colored(text, color=None):  # type: ignore
        return text


def color_msg(text: str, color: str = "cyan") -> str:
    return _colored(text, color)


def load_env(path: Path) -> None:
    """Load environment variables from a .env file if present.

    Parameters
    ----------
    path: Path
        Path to a .env file.
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        # Remove surrounding single or double quotes from value
        value = value.strip()
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        os.environ.setdefault(key, value)


def extract_repository_context(xml_path: Path, max_chars: int = 2000) -> str:
    """Extract a concise summary from the unified XML repository file.

    This implementation uses a real XML parser (ElementTree) instead of
    fragile regular expressions.  It attempts to extract the
    `<file_summary>` and `<directory_structure>` elements from the
    repository snapshot.  If those elements are not present, it falls
    back to extracting text from the entire document (stripping tags) to
    avoid returning an empty context.  Newlines and basic formatting
    are preserved.

    Parameters
    ----------
    xml_path: Path
        Path to the unified XML file.
    max_chars: int
        Maximum number of characters to extract.

    Returns
    -------
    str
        A string containing the repository context.  The context is
        truncated to ``max_chars`` characters to avoid exceeding the
        model's context window.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        summary = root.findtext('file_summary', default='') or ''
        directory = root.findtext('directory_structure', default='') or ''
        extracted = (summary.strip() + "\n\n" + directory.strip()).strip()
        if not extracted:
            # Fallback: extract all text content from the XML
            extracted = ''.join(root.itertext())
        # Limit the length
        return extracted[:max_chars]
    except Exception as e:
        # As a last resort, read raw file and remove tags
        try:
            raw = xml_path.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            raise RuntimeError(f"Failed to read {xml_path}: {e}")
        # Remove XML tags and return prefix
        return re.sub(r'<[^>]+>', '', raw)[:max_chars]


def read_plan_text(plan_path: Path) -> str:
    """Read the high‑level plan description from a file.

    The plan can be a Markdown or plain text file.  The content is
    returned unchanged.

    Parameters
    ----------
    plan_path: Path
        Path to the plan description file.

    Returns
    -------
    str
        The text of the plan file.
    """
    try:
        return plan_path.read_text(encoding="utf-8")
    except Exception as e:
        raise RuntimeError(f"Failed to read plan file {plan_path}: {e}")


def build_prompt(
    repository_context: str,
    plan_text: str,
    feature_slug: str,
    min_prs: int = 6,
    max_prs: int = 12,
    min_tasks: int = 3,
    max_tasks: int = 8,
) -> str:
    """Assemble the prompt for GPT‑5.

    This prompt preserves whitespace, headings and list formatting to
    maintain clarity for the model.  It clearly delineates the
    repository context, the feature description and the rules for
    decomposing the work into PRs.  No whitespace collapsing is
    performed; this allows the language model to interpret bullet
    points, paragraphs and headings correctly.

    Parameters
    ----------
    repository_context: str
        A concise summary of the repository for context.
    plan_text: str
        High‑level feature description.
    feature_slug: str
        A slug used for naming PRs (e.g. "assignment-workflow").

    Returns
    -------
    str
        The complete prompt to send to the LLM.
    """
    # Build a gold-standard, sectioned prompt for GPT-5 planning
    prompt_lines: List[str] = []
    
    # System role and intent
    prompt_lines.append("SYSTEM ROLE")
    prompt_lines.append(
        "You are a staff-level software engineer and project planner. Your job: "
        "turn a high-level feature into a sequence of small, independently-mergeable PRs "
        "for the TalentPulse360 codebase."
    )
    prompt_lines.append("")
    
    # Global controls per GPT-5 guidance
    prompt_lines.append("GLOBAL CONTROLS")
    prompt_lines.append("- Follow the API-provided reasoning_effort; default to medium unless overridden.")
    prompt_lines.append("- verbosity: low")
    prompt_lines.append("- agentic_eagerness: low (no exploration, no tool preambles, no tool calls).")
    prompt_lines.append("- Output format: JSON only. No prose, no markdown, no code fences.")
    prompt_lines.append("")
    
    # Repository context
    prompt_lines.append("REPOSITORY CONTEXT (from unified XML)")
    prompt_lines.append("Use this for naming, directories, and frameworks. Prefer directory_structure over raw file bodies.")
    prompt_lines.append(repository_context.strip())
    prompt_lines.append("")
    
    # Feature description
    prompt_lines.append("FEATURE DESCRIPTION")
    prompt_lines.append(plan_text.strip())
    prompt_lines.append("")
    
    # Constraints tailored for robust, reviewable PR planning
    prompt_lines.append("CONSTRAINTS")
    prompt_lines.append(
        f"- PR count: between {min_prs} and {max_prs}, each focused and independently reviewable/mergeable."
    )
    prompt_lines.append(
        f"- PR naming (HARD): {feature_slug}-PR-XX where XX is 01..NN, strictly increasing with no gaps."
    )
    prompt_lines.append(
        f"- Tasks per PR: {min_tasks}–{max_tasks} steps; imperative, concrete, repository-aware."
    )
    prompt_lines.append(
        "- Each PR includes a final testing/validation task (add/extend tests per repo standard; verify CI passes)."
    )
    prompt_lines.append("- Follow repository conventions (Python, FastAPI, Pydantic v2, tests).")
    prompt_lines.append(
        "- Prefer existing directories from directory_structure; do not hallucinate new top-level roots unless essential."
    )
    prompt_lines.append(
        "- If information is missing, make reasonable assumptions and proceed; do not stall or ask questions."
    )
    prompt_lines.append(
        "- Later PRs may depend on earlier ones by number only; avoid cross-PR cycles."
    )
    prompt_lines.append("")
    
    # Output contract (schema-like; no examples that look like real arrays)
    prompt_lines.append("OUTPUT CONTRACT (JSON ONLY)")
    prompt_lines.append(
        "Return a single JSON array. Each element is an object with exactly these keys: pr_name, pr_description, tasks."
    )
    prompt_lines.append(
        f"- pr_name: string matching ^{feature_slug}-PR-(0[1-9]|[1-9][0-9])$"
    )
    prompt_lines.append(
        "- pr_description: 1–3 sentences, concise and specific to this PR’s scope."
    )
    prompt_lines.append(
        f"- tasks: array of {min_tasks}–{max_tasks} strings. Each string is a concrete imperative step referencing files/paths/endpoints where applicable; the final task must add/extend tests and verify CI passes."
    )
    prompt_lines.append(
        "Important: Output ONLY the JSON array; no surrounding text, no markdown, no comments, no code fences."
    )
    prompt_lines.append("")
    
    # Internal self-check rubric (do not print)
    prompt_lines.append("INTERNAL SELF-CHECK (DO NOT PRINT)")
    prompt_lines.append(f"- PR count is {min_prs}–{max_prs}.")
    prompt_lines.append("- Names follow the regex and are strictly sequential without gaps.")
    prompt_lines.append(
        f"- Each PR has {min_tasks}–{max_tasks} concrete tasks; the final task includes testing/validation."
    )
    prompt_lines.append("- No cross-PR cycles; only forward dependencies by PR number.")
    prompt_lines.append("- Tasks reference existing directories and repository conventions.")
    prompt_lines.append(
        "If any check fails, fix internally and re-validate, then output the JSON array."
    )
    
    return "\n".join(prompt_lines)


def build_output_json_schema(
    feature_slug: str,
    min_prs: int,
    max_prs: int,
    min_tasks: int,
    max_tasks: int,
) -> Dict[str, Any]:
    """Construct a JSON Schema for the expected PR plan.

    The schema enforces:
    - Top-level array with bounds (min_prs..max_prs)
    - Object items with required keys
    - PR name pattern bound to the given feature_slug
    - Tasks array bounds (min_tasks..max_tasks)
    """
    pr_name_pattern = f"^{re.escape(feature_slug)}-PR-(0[1-9]|[1-9][0-9])$"
    schema: Dict[str, Any] = {
        "name": "PRPlan",
        "schema": {
            "type": "array",
            "minItems": max(1, int(min_prs)),
            "maxItems": int(max_prs),
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "pr_name": {"type": "string", "pattern": pr_name_pattern},
                    "pr_description": {"type": "string"},
                    "tasks": {
                        "type": "array",
                        "minItems": max(1, int(min_tasks)),
                        "maxItems": int(max_tasks),
                        "items": {"type": "string"},
                    },
                },
                "required": ["pr_name", "pr_description", "tasks"],
            },
        },
    }
    return schema


def call_llm(
    prompt: str,
    model: str = "gpt-5",
    reasoning_effort: str = "high",
    # NOTE: GPT‑5 via the responses API does not support temperature/top_p parameters.
    # We accept them to maintain CLI compatibility but do not pass them to the API call.
    temperature: float = 0.2,
    top_p: float = 1.0,
    max_tokens: int = 4096,
    max_retries: int = 3,
    timeout: float = 60.0,
    json_mode: str = "auto",  # one of: auto, schema, object, off
    json_schema: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
) -> str:
    """Call the OpenAI GPT‑5 model with retries and configurable parameters.

    This helper wraps the OpenAI API call in a retry loop and exposes
    the most common generation parameters.  It also supports
    configuring a timeout and logging progress.  If the API
    repeatedly fails (due to rate limits or transient errors), a
    ``RuntimeError`` will be raised after exhausting retries.

    Parameters
    ----------
    prompt: str
        The prompt to send to the model.
    model: str
        The model identifier.  Default is "gpt-5".
    reasoning_effort: str
        The reasoning effort level (e.g. "high", "medium", "low").
    temperature: float
        Sampling temperature for output variability.  Lower values
        produce more deterministic results.  Default is 0.2.
    top_p: float
        Nucleus sampling parameter.  Default is 1.0 (no truncation).
    max_tokens: int
        Maximum number of tokens in the response.  Default is 2048.
    max_retries: int
        Number of times to retry the call in the face of transient
        errors.  Default is 3.
    timeout: float
        Request timeout in seconds.  Default is 60.0.
    logger: Optional[logging.Logger]
        Optional logger to emit informational messages.

    Returns
    -------
    str
        The raw text response from the model.
    """
    # Lazy import to improve CLI ergonomics and avoid import cost unless used
    try:
        from openai import OpenAI  # type: ignore
    except Exception as exc:  # pragma: no cover - import-time path
        raise ImportError(
            "The 'openai' package is required but not installed. Run 'pip install openai' to install it."
        ) from exc

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Please set it in your environment or .env file."
        )
    # Instantiate a client (timeout supplied per-request below)
    client = OpenAI(api_key=api_key)
    attempt = 0
    while True:
        attempt += 1
        try:
            if logger:
                logger.debug(
                    f"Calling OpenAI model={model}, attempt={attempt}, reasoning={reasoning_effort}, "
                    f"temperature={temperature}, top_p={top_p}, max_output_tokens={max_tokens}"
                )
            # Note: GPT‑5 is accessible via the 'responses' endpoint.
            # Prefer structured output if supported; fallback gracefully otherwise.
            # The responses API does NOT support temperature/top_p parameters; they must
            # be omitted or the API will throw an invalid parameter error.  We keep
            # them as function arguments for compatibility but do not include them
            # in the request.  See OpenAI docs for details.
            request_kwargs: Dict[str, Any] = {
                "model": model,
                "input": prompt,
                "reasoning": {"effort": reasoning_effort},
                "max_output_tokens": max_tokens,
            }
            tried_structured = False
            if json_mode != "off":
                if (json_mode in ("schema", "auto")) and json_schema:
                    request_kwargs["response_format"] = {  # type: ignore[assignment]
                        "type": "json_schema",
                        "json_schema": json_schema,
                    }
                    tried_structured = True
                elif json_mode in ("object", "auto"):
                    request_kwargs["response_format"] = {  # type: ignore[assignment]
                        "type": "json_object",
                    }
                    tried_structured = True

            try:
                response = client.responses.create(timeout=timeout, **request_kwargs)  # type: ignore[arg-type]
            except Exception as e1:
                # Fallback: retry without response_format if it's not accepted
                if tried_structured and ("response_format" in str(e1) or "invalid" in str(e1).lower()):
                    if logger:
                        logger.info(color_msg("Structured response not supported; retrying without response_format.", "yellow"))
                    request_kwargs.pop("response_format", None)
                    response = client.responses.create(timeout=timeout, **request_kwargs)  # type: ignore[arg-type]
                else:
                    raise

            # Extract output text robustly across SDK variants
            text_out: Optional[str] = None
            if hasattr(response, "output_text"):
                text_out = getattr(response, "output_text")  # type: ignore[assignment]
            elif hasattr(response, "text"):
                text_out = getattr(response, "text")  # type: ignore[assignment]
            elif hasattr(response, "choices") and getattr(response, "choices"):
                first = response.choices[0]
                # choices API compatibility
                if hasattr(first, "text"):
                    text_out = first.text  # type: ignore
                elif hasattr(first, "message") and hasattr(first.message, "content"):
                    text_out = first.message.content  # type: ignore
            if text_out is None:
                # Last resort: stringify and hope it contains text
                text_out = str(response)

            # Log token usage if available
            usage = getattr(response, "usage", None)
            if logger and usage is not None:
                try:
                    in_toks = getattr(usage, "input_tokens", None) or usage.get("input_tokens")  # type: ignore[attr-defined]
                    out_toks = getattr(usage, "output_tokens", None) or usage.get("output_tokens")  # type: ignore[attr-defined]
                    tot_toks = getattr(usage, "total_tokens", None) or usage.get("total_tokens")  # type: ignore[attr-defined]
                    logger.info(color_msg(f"Token usage — input: {in_toks}, output: {out_toks}, total: {tot_toks}", "magenta"))
                except Exception:
                    # Be tolerant of unknown usage shapes
                    pass

            return text_out
        except Exception as e:
            # Attempt to classify error types from openai.error module
            err_name = type(e).__name__
            # Always log error if logger provided
            if logger:
                logger.warning(f"LLM call failed on attempt {attempt} with error: {err_name}: {e}")
            # Do not retry on invalid request errors
            # For demonstration purposes we treat all exceptions as potentially retryable
            if attempt >= max_retries:
                raise RuntimeError(f"OpenAI API call failed after {max_retries} attempts: {e}")
            # Exponential backoff with jitter
            sleep_time = 2 ** (attempt - 1) + (0.1 * attempt)
            time.sleep(sleep_time)


def parse_llm_output(
    raw: str,
    feature_slug: str,
    min_prs: int,
    max_prs: int,
    min_tasks: int,
    max_tasks: int,
    logger: Optional[logging.Logger] = None,
) -> List[Dict[str, Any]]:
    """Parse and validate the JSON output returned by the LLM.

    This function attempts to extract a JSON array from the raw
    response.  It removes any Markdown code fences (triple backticks)
    and trims surrounding whitespace before parsing.  The parsed
    structure is validated to ensure each entry is a dictionary with
    the required keys ``pr_name``, ``pr_description`` and ``tasks``.  If
    additional keys are present, they are ignored.  If the output
    cannot be parsed as JSON or lacks the expected structure, a
    ``ValueError`` is raised.

    Parameters
    ----------
    raw: str
        The raw response from the LLM.

    Returns
    -------
    list
        A validated list of PR definitions.
    """
    if not raw:
        raise ValueError("Empty response from LLM")
    # Remove Markdown code fences to avoid interfering with JSON extraction
    cleaned = re.sub(r"```.*?```", "", raw, flags=re.DOTALL)
    cleaned = cleaned.strip()
    # Extract the first JSON array
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start == -1 or end == -1 or start > end:
        raise ValueError("LLM output does not contain a JSON array")
    json_str = cleaned[start : end + 1]
    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON from LLM output: {e}\nExtracted JSON: {json_str}")
    if not isinstance(parsed, list):
        raise ValueError("Top-level JSON structure must be a list of PR definitions")
    # Optional order normalization by pr_name to enforce sequencing
    if all(isinstance(it, dict) and "pr_name" in it for it in parsed):
        sorted_parsed = sorted(parsed, key=lambda d: d.get("pr_name", ""))
        if logger and sorted_parsed != parsed:
            logger.info(color_msg("Reordered PRs by pr_name to enforce sequential numbering.", "yellow"))
        parsed = sorted_parsed

    # Global list bounds
    if not (int(min_prs) <= len(parsed) <= int(max_prs)):
        raise ValueError(
            f"PR count {len(parsed)} is outside allowed range [{min_prs}, {max_prs}]."
        )

    # Validate each entry
    REQUIRED_KEYS = {"pr_name", "pr_description", "tasks"}
    name_pattern = re.compile(rf"^{re.escape(feature_slug)}-PR-(0[1-9]|[1-9][0-9])$")
    pr_list: List[Dict[str, Any]] = []
    seq_nums: List[int] = []
    for idx, entry in enumerate(parsed):
        if not isinstance(entry, dict):
            raise ValueError(f"PR entry at index {idx} is not an object: {entry}")
        missing = REQUIRED_KEYS - entry.keys()
        if missing:
            raise ValueError(f"PR entry missing keys {missing}: {entry}")
        pr_name = entry.get("pr_name")
        if not isinstance(pr_name, str) or not name_pattern.match(pr_name):
            raise ValueError(f"Invalid pr_name '{pr_name}'. Must match feature slug and pattern.")
        # Extract trailing number
        m = re.search(r"(\d{2})$", pr_name)
        if not m:
            raise ValueError(f"Could not extract PR index from name: {pr_name}")
        seq_nums.append(int(m.group(1)))

        pr_desc = entry.get("pr_description")
        if not isinstance(pr_desc, str) or not pr_desc.strip():
            raise ValueError(f"Invalid pr_description for {pr_name}.")

        tasks = entry.get("tasks")
        if not isinstance(tasks, list) or not all(isinstance(t, str) for t in tasks):
            raise ValueError(f"'tasks' must be a list of strings in PR entry: {entry}")
        if not (int(min_tasks) <= len(tasks) <= int(max_tasks)):
            raise ValueError(
                f"PR {pr_name} has {len(tasks)} tasks, outside allowed range [{min_tasks}, {max_tasks}]."
            )
        # Warn (do not fail) if last task does not appear to include tests/validation
        if logger and tasks:
            last = tasks[-1].lower()
            if not any(k in last for k in ("test", "tests", "ci", "validate")):
                logger.warning(color_msg(f"Final task in {pr_name} does not mention tests/CI/validation.", "yellow"))

        # Filter only required keys
        pr_dict = {
            "pr_name": pr_name,
            "pr_description": pr_desc,
            "tasks": tasks,
        }
        pr_list.append(pr_dict)

    # Sequence must be 01..N strictly increasing with no gaps
    expected = list(range(1, len(pr_list) + 1))
    if seq_nums != expected:
        raise ValueError(
            f"PR numbering must be sequential 01..{len(pr_list):02d} with no gaps. Got: {seq_nums}"
        )
    return pr_list


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Generate a universal task plan from a feature description using GPT‑5.")
    parser.add_argument("--xml", type=Path, required=True, help="Unified repository XML file")
    parser.add_argument("--plan", type=Path, required=True, help="High‑level feature description file")
    parser.add_argument("--feature", type=str, required=True, help="Feature slug used for PR names")
    parser.add_argument("--output", type=Path, help="Output file for the generated plan (JSON)")
    parser.add_argument(
        "--model", type=str, default="gpt-5", help="OpenAI model identifier (default: gpt-5)"
    )
    parser.add_argument(
        "--reasoning", type=str, default="high", help="Reasoning effort (default: high)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for the LLM (default: 0.2)",
    )
    parser.add_argument(
        "--top-p",
        dest="top_p",
        type=float,
        default=1.0,
        help="Nucleus sampling parameter (default: 1.0)",
    )
    parser.add_argument(
        "--max-tokens",
        dest="max_output_tokens",
        type=int,
        default=14096,
        help="Maximum number of tokens in the response (default: 14096)",
    )
    parser.add_argument(
        "--max-retries",
        dest="max_retries",
        type=int,
        default=3,
        help="Maximum number of retries for API calls (default: 3)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=560.0,
        help="Request timeout in seconds for OpenAI calls (default: 560)",
    )
    parser.add_argument(
        "--log-level",
        dest="log_level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR). Default: INFO",
    )
    # Advanced planner controls
    parser.add_argument(
        "--context-chars", type=int, default=2000, help="Max repository context characters (default: 2000)"
    )
    parser.add_argument(
        "--min-prs", dest="min_prs", type=int, default=6, help="Minimum number of PRs to plan (default: 6)"
    )
    parser.add_argument(
        "--max-prs", dest="max_prs", type=int, default=12, help="Maximum number of PRs to plan (default: 12)"
    )
    parser.add_argument(
        "--min-tasks", dest="min_tasks", type=int, default=3, help="Minimum tasks per PR (default: 3)"
    )
    parser.add_argument(
        "--max-tasks", dest="max_tasks", type=int, default=8, help="Maximum tasks per PR (default: 8)"
    )
    parser.add_argument(
        "--json-mode",
        type=str,
        choices=["auto", "schema", "object", "off"],
        default="auto",
        help="Structured output mode: 'schema' uses JSON Schema, 'object' enforces JSON object output, 'auto' tries schema then falls back, 'off' disables.",
    )
    parser.add_argument("--save-prompt", dest="save_prompt", type=Path, help="Optional path to save constructed prompt")
    parser.add_argument("--save-raw", dest="save_raw", type=Path, help="Optional path to save raw LLM output")
    args = parser.parse_args(argv)

    # Load environment variables from .env if present in current directory
    load_env(Path.cwd() / ".env")
    # Initialise logging
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level, format="[%(levelname)s] %(message)s")
    logger = logging.getLogger("task_planner")

    # Extract context and read plan
    repo_context = extract_repository_context(args.xml, max_chars=args.context_chars)
    logger.info(color_msg(f"Context extracted: {len(repo_context)} chars", "cyan"))
    # Allow reading plan text from STDIN if path is '-'
    if args.plan.as_posix() == "-":
        logger.info(color_msg("Reading plan text from STDIN (use Ctrl-D to end)", "cyan"))
        plan_text = sys.stdin.read()
    else:
        plan_text = read_plan_text(args.plan)
    logger.info(color_msg(f"Plan text read: {len(plan_text)} chars", "cyan"))
    prompt = build_prompt(
        repo_context,
        plan_text,
        args.feature,
        args.min_prs,
        args.max_prs,
        args.min_tasks,
        args.max_tasks,
    )
    logger.info(color_msg(f"Prompt constructed ({len(prompt)} chars)", "green"))
    if args.save_prompt:
        args.save_prompt.parent.mkdir(parents=True, exist_ok=True)
        args.save_prompt.write_text(prompt, encoding="utf-8")
        logger.info(color_msg(f"Prompt saved to {args.save_prompt}", "green"))
    # Call LLM
    schema = None
    if args.json_mode in ("schema", "auto"):
        try:
            schema = build_output_json_schema(
                args.feature, args.min_prs, args.max_prs, args.min_tasks, args.max_tasks
            )
        except Exception as e:
            logger.warning(color_msg(f"Failed to build JSON schema, continuing without it: {e}", "yellow"))
    raw_output = call_llm(
        prompt,
        model=args.model,
        reasoning_effort=args.reasoning,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_output_tokens,
        max_retries=args.max_retries,
        timeout=args.timeout,
        json_mode=args.json_mode,
        json_schema=schema,
        logger=logger,
    )
    logger.info(color_msg("LLM response received", "green"))
    if args.save_raw:
        args.save_raw.parent.mkdir(parents=True, exist_ok=True)
        args.save_raw.write_text(raw_output, encoding="utf-8")
        logger.info(color_msg(f"Raw output saved to {args.save_raw}", "green"))
    try:
        pr_plan = parse_llm_output(
            raw_output,
            feature_slug=args.feature,
            min_prs=args.min_prs,
            max_prs=args.max_prs,
            min_tasks=args.min_tasks,
            max_tasks=args.max_tasks,
            logger=logger,
        )
    except ValueError as parse_err:
        # Provide guidance on empty or malformed responses
        msg = str(parse_err)
        if "Empty response" in msg:
            logger.error(color_msg(
                "LLM returned an empty response. Try increasing --max-tokens or reducing reasoning effort.",
                "red",
            ))
        raise
    # Write output or print
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(pr_plan, indent=2), encoding="utf-8")
        print(color_msg(f"Plan written to {args.output}", "cyan"))
    else:
        print(json.dumps(pr_plan, indent=2))


if __name__ == "__main__":
    main()