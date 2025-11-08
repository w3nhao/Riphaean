# src/extraction/extractor.py
import re
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict

sys.path.append('..')
from llm_client import LlamaServerClient
from memory import MemoryItem


class MemoryExtractor:
    """Extract memory items from problem-solving trajectories.

    Memory prompts allow hidden thinking by default. Set ``thinking_enabled`` to
    ``False`` to append ``/no_think`` and force the model to respond without
    intermediate thoughts.
    """

    def __init__(self, llm_client: LlamaServerClient, debug_dir: str = 'memory_bank/debug',
                 thinking_enabled: bool = True):
        self.llm = llm_client
        self.debug_dir = Path(debug_dir) if debug_dir else None
        self.thinking_enabled = thinking_enabled
        if self.debug_dir:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

    def extract_from_trajectory(self, problem_id: str, question: str,
                                 solution: Dict, success: bool) -> List[MemoryItem]:
        """Extract 1-3 memory items from a trajectory"""

        if success:
            prompt = self._create_success_prompt(question, solution['reasoning'])
        else:
            prompt = self._create_failure_prompt(question, solution['reasoning'],
                                                  solution.get('expected', ''))

        response = self.llm.generate(prompt, temperature=0.0, max_tokens=2048)

        if not response.strip():
            self._dump_response(problem_id, response, success, reason="empty_response")
            self._warn_placeholder(problem_id, "", stage="empty_response")
            placeholder = self._create_placeholder_memory(
                problem_id,
                question,
                solution,
                success,
                response
            )
            return [placeholder]

        # Parse memory items from response
        memories = self._parse_memory_items(response, problem_id, success)
        final_response = response

        if not memories:
            # Store the raw failure for debugging before attempting cleanup
            self._dump_response(problem_id, response, success, reason="parse_failure_raw")

            cleaned_response = self._clean_response(
                problem_id,
                question,
                solution,
                success,
                response
            )

            if cleaned_response:
                cleaned_memories = self._parse_memory_items(cleaned_response, problem_id, success)
                if cleaned_memories:
                    memories = cleaned_memories
                    final_response = cleaned_response
                else:
                    if cleaned_response != response:
                        self._dump_response(problem_id, cleaned_response, success, reason="cleanup_failure")

        if not memories:
            preview_source = cleaned_response if 'cleaned_response' in locals() and cleaned_response else response
            self._warn_placeholder(problem_id, preview_source, stage="parsing")
            placeholder = self._create_placeholder_memory(
                problem_id,
                question,
                solution,
                success,
                preview_source
            )
            return [placeholder]

        if not self._validate_response(final_response):
            self._dump_response(problem_id, final_response, success, reason="format_validation_failed")
            self._warn_placeholder(problem_id, final_response, stage="validation")
            placeholder = self._create_placeholder_memory(
                problem_id,
                question,
                solution,
                success,
                final_response
            )
            return [placeholder]

        return memories

    def _clean_response(self, problem_id: str, question: str, solution: Dict,
                         success: bool, raw_response: str) -> str:
        """Use the LLM to normalize malformed extraction output."""
        if not self.llm:
            return ""

        outcome = "successful solution" if success else "failed attempt"
        reasoning = solution.get('reasoning', '')
        expected_answer = solution.get('expected', '')

        template = f"""You will receive raw notes from a {outcome} for a math problem.
Rewrite them EXACTLY into the schema below. Obey every rule strictly.

Rules:
1. Output 1-3 memories, sequentially numbered.
2. Use these field labels verbatim: MEMORY N:, TITLE:, DESCRIPTION:, CONTENT:
3. Each field must be on its own line with plain text (no markdown emphasis or bullet lists).
4. Preserve the intent of the notes while removing filler such as <think> blocks.
5. Do not add commentary before or after the memories.

Schema:
MEMORY 1:
TITLE: <concise strategy name>
DESCRIPTION: <one sentence summary>
CONTENT: <detailed transferable strategy>

Problem: {question}
Reasoning notes: {reasoning}
Expected answer (if provided): {expected_answer}

RAW NOTES TO FORMAT:
{raw_response}
"""

        prompt = self._finalize_prompt(template, self.thinking_enabled)

        try:
            cleaned = self.llm.generate(prompt, temperature=0.0, max_tokens=2048)
        except Exception:
            return ""

        return cleaned or ""

    def _validate_response(self, response: str) -> bool:
        """Confirm via LLM that the formatted response respects the schema."""
        if not self.llm:
            return False

        template = f"""You are a strict format checker. Inspect the MEMORY CARDS below and ensure they follow the schema.

Schema:
MEMORY 1:
TITLE: <concise strategy name>
DESCRIPTION: <one sentence summary>
CONTENT: <detailed transferable strategy>

MEMORY 2:
...
(continue numbering if present)

Answer with EXACTLY one line: MEMORY FORMAT FOLLOWED: TRUE or MEMORY FORMAT FOLLOWED: FALSE.
Respond FALSE if any requirement is violated (missing fields, wrong labels, extra markdown, etc.).

MEMORY CARDS TO CHECK:
{response}
"""

        prompt = self._finalize_prompt(template, self.thinking_enabled)

        try:
            validation = self.llm.generate(prompt, temperature=0.0, max_tokens=128)
        except Exception:
            return False

        if not validation:
            return False

        for line in validation.splitlines():
            if "MEMORY FORMAT FOLLOWED:" in line.upper():
                verdict = line.split(":", 1)[1].strip().upper()
                return verdict == "TRUE"

        return False

    def _create_placeholder_memory(self, problem_id: str, question: str, solution: Dict,
                                   success: bool, response: str) -> MemoryItem:
        """Create a placeholder memory allowing post-run inspection."""
        preview = response.strip().replace('\n', ' ')[:200] or "No extraction content captured."
        title = f"PLACEHOLDER: {problem_id}"
        description = "Extraction output failed format validation. Review debug dumps."
        content = f"Problem: {question}\nPreview: {preview}"
        created_at = datetime.now().isoformat()
        return MemoryItem(
            title=title,
            description=description,
            content=content,
            source_problem_id=problem_id,
            success=success,
            created_at=created_at,
            placeholder=True
        )

    def _warn_placeholder(self, problem_id: str, response: str, stage: str) -> None:
        """Emit a warning when a placeholder memory is created."""
        preview = response.strip().replace('\n', ' ')[:160]
        print(
            f"[MemoryExtractor] WARNING: placeholder inserted for problem {problem_id} "
            f"after {stage} failure. Preview: {preview}"
        )

    def _dump_response(self, problem_id: str, response: str, success: bool, reason: str) -> None:
        """Persist raw LLM response for debugging when extraction fails."""
        if not self.debug_dir:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_problem = re.sub(r'[^a-zA-Z0-9_-]', '_', problem_id or 'unknown')
        success_tag = 'success' if success else 'failure'
        filename = f"{safe_problem}_{success_tag}_{reason}_{timestamp}.txt"
        path = self.debug_dir / filename

        try:
            with path.open('w', encoding='utf-8') as handle:
                handle.write(response)
        except OSError as error:  # Expected failures such as permission issues
            print(f"Warning: unable to write debug response to {path}: {error}")

    def _finalize_prompt(self, template: str, thinking_enabled: bool = True) -> str:
        """Apply the thinking toggle to the prompt template."""
        if thinking_enabled:
            return template
        return f"{template} /no_think"

    def _create_success_prompt(self, question: str, reasoning: str) -> str:
        template = f"""You successfully solved this math problem. Extract 1-3 generalizable strategies that led to success.

PROBLEM: {question}

YOUR SOLUTION: {reasoning}

Extract strategies in this format:

MEMORY 1:
TITLE: <concise strategy name>
DESCRIPTION: <one sentence summary>
CONTENT: <detailed transferable strategy>

MEMORY 2:
...

Focus on WHY the approach worked and how it could apply to similar problems."""
        return self._finalize_prompt(template, self.thinking_enabled)

    def _create_failure_prompt(self, question: str, reasoning: str, expected: str) -> str:
        template = f"""You attempted this math problem but got it wrong. Extract 1-3 lessons about what went wrong.

PROBLEM: {question}

YOUR ATTEMPT: {reasoning}

EXPECTED: {expected}

Extract lessons in this format:

MEMORY 1:
TITLE: <what to avoid or check>
DESCRIPTION: <one sentence summary>
CONTENT: <detailed lesson or preventive strategy>

MEMORY 2:
...

Focus on the mistake and how to prevent it in future similar problems."""
        return self._finalize_prompt(template, self.thinking_enabled)

    def _parse_memory_items(self, response: str, problem_id: str, success: bool) -> List[MemoryItem]:
        """Parse structured memory items from LLM response"""
        memories = []

        # Split by MEMORY markers
        parts = response.split('MEMORY ')

        for part in parts[1:]:  # Skip first split (before first MEMORY)
            try:
                # Extract fields
                title = self._extract_field(part, 'TITLE:')
                description = self._extract_field(part, 'DESCRIPTION:')
                content = self._extract_field(part, 'CONTENT:')

                if title and description and content:
                    memory = MemoryItem(
                        title=title,
                        description=description,
                        content=content,
                        source_problem_id=problem_id,
                        success=success,
                        created_at=datetime.now().isoformat()
                    )
                    memories.append(memory)
            except:
                continue

        return memories

    def _extract_field(self, text: str, field_name: str) -> str:
        """Extract field value from formatted text"""
        if field_name not in text:
            return ""

        start = text.index(field_name) + len(field_name)
        # Find next field or end
        next_field_markers = ['TITLE:', 'DESCRIPTION:', 'CONTENT:', 'MEMORY ']

        end = len(text)
        for marker in next_field_markers:
            if marker in text[start:]:
                candidate_end = start + text[start:].index(marker)
                if candidate_end < end:
                    end = candidate_end

        value = text[start:end].strip()
        return value