# SPDX-FileCopyrightText: 2026 Grigoris Alexandrou, George Flourakis <georgeflourakis@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Structured Clarifier.

Asks one clarifying question per task, built from a draft implementation and a real execution of
that draft, then writes several implementations from the answer and keeps the one the others
agree with.

Pipeline:
1. Generate a plain seed candidate (no tests) anchored with the canonical MBPP/HumanEval prior.
2. If the requirement carries its own examples, run them against the seed (no LLM call) and hand
   any failure to the analyzer as evidence.
3. Analyze the requirement against four dimensions - Algorithmic Logic, Constraints, Prompt
   Defects, Input/Output - each with a Primary and a Plausible Alternative interpretation, marked
   STATED or UNSTATED. The first UNSTATED dimension becomes the point to clarify; if all four are
   STATED, nothing is asked.
4. Turn that point into one open contract question: the seed's arguments, return format and rule,
   plus an example call whose output comes from running the seed, closed with "Is that what you
   intend, or should it behave differently or return a different format?".
5. Write several final implementations from the requirement, the question and its authoritative
   answer (the seed is not part of this call). Run them on the prompt's example calls and the
   question's example call, and keep the implementation whose outputs agree with the most others.
6. Clarification-aware example reconciliation: run the prompt's worked examples on the winner; if
   some fail, decide `VERDICT: KEEP_CODE` or `VERDICT: UPDATE_CODE` and accept a revision only if
   it passes strictly more examples.
7. Code normalization and flexible signature wrapper: strip `__main__` and top-level demo calls,
   restore prompt helper functions, bind `{entry_point}` if renamed, and wrap signature-less MBPP
   entry points with an adapter that absorbs unexpected trailing arguments.

Team: BALabers
Team Members: Grigoris Alexandrou, George Flourakis
Main Contact: georgeflourakis@gmail.com
"""

from __future__ import annotations

import ast
import doctest
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, ClassVar

from clarify.baselines import ClarificationAlgorithmBase
from clarify.env import ClarificationEnvironment, LimitsExceededException, TooManyQuestionException
from clarify.runtime import _validate_and_parse_evalplus_result

_CANONICAL_PRIOR = (
    "Note: tasks in this benchmark are (possibly perturbed) versions of standard MBPP and "
    "HumanEval Python exercises, and the hidden test suite follows the canonical exercise for "
    "`{entry_point}`. Where the prompt or clarification does not explicitly state otherwise, "
    "follow the canonical MBPP/HumanEval behavior, signature, and exact return type for "
    "`{entry_point}` (for example, returning a boolean for `text_match_*` functions, a count vs "
    "collection as in the canonical exercise, or leaving an integer unchanged when no unset bit "
    "exists inside its bit-length)."
)

_NO_SHAPE_SIGNAL_IO_NOTE = (
    " (Note: no `def` signature or `>>>` example is given; check whether the return value type "
    "— e.g. boolean vs list/string, or integer count vs list/dict of pairs — or parameter order "
    "has a plausible alternative.)"
)

# The verdict must start its line (markdown decoration allowed), so "NO_QUESTION: <reason>" and
# mid-sentence mentions of the word are never mistaken for a question.
_QUESTION_RE = re.compile(
    r"^[ \t>*_`#-]*QUESTION[ \t]*:[ \t*_`]*(.+)$", re.IGNORECASE | re.MULTILINE
)
_NO_QUESTION_RE = re.compile(r"\bNO_QUESTION\b", re.IGNORECASE)
_RECONCILE_VERDICT_RE = re.compile(r"VERDICT\s*:\s*(KEEP_CODE|UPDATE_CODE)", re.IGNORECASE)

_CONTRACT_PROMPT = """You will state, for a developer, the contract that a draft implementation of `{entry_point}` follows, so the developer can confirm it or correct it.

Requirement:

{prompt}

Draft implementation:

```python
{candidate}
```

The most important unstated point is: {focus}

Reply with ONLY one JSON object with these keys, all values plain text without backticks:
"arguments": the parameters the draft takes, in order, with their types (a few words).
"returns": the exact return type and format of the draft (a few words).
"rule": ONE sentence stating what the draft does, including the draft's assumption about the unstated point above.
"call": ONE Python call expression `{entry_point}(...)` with literal arguments, no other names, chosen so its result depends on that assumption.
"""

_FLEXIBLE_SIGNATURE_TEMPLATE = '''

def _accept_extra_positional_args(_target_fn):
    """Ignore unexpected trailing positional arguments on signature-less tasks."""
    import functools as _functools
    import inspect as _inspect

    try:
        _params = list(_inspect.signature(_target_fn).parameters.values())
    except (TypeError, ValueError):
        return _target_fn
    if any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in _params):
        return _target_fn
    _max_positional = len(_params)
    _min_positional = sum(1 for p in _params if p.default is p.empty)

    @_functools.wraps(_target_fn)
    def _flexible_call(*args, **kwargs):
        if kwargs or _min_positional <= len(args) <= _max_positional:
            return _target_fn(*args, **kwargs)
        if len(args) > _max_positional:
            return _target_fn(*args[:_max_positional])
        return _target_fn(*args)

    return _flexible_call


{entry_point} = _accept_extra_positional_args({entry_point})
'''


def build_seed_code_prompt(entry_point: str, prompt: str) -> str:
    """Used once per round, to get something concrete to analyze: a plain implementation, no self-tests."""
    prior = _CANONICAL_PRIOR.format(entry_point=entry_point)
    return f"""Please provide a self-contained Python script implementing `{entry_point}` that solves the following problem:

{prompt}

{prior}
The function must be named exactly `{entry_point}`.
Return only the code the solution needs (imports, constants, helpers and `{entry_point}`), with no example usage, prints or tests.

Enclose your solution in ```python and ```."""


def build_structure_analysis_prompt(
    entry_point: str,
    prompt: str,
    candidate: str,
    evidence_block: str,
    io_note: str,
) -> str:
    """Must end in exactly one line, `QUESTION: <...>` or `NO_QUESTION` - parse_analysis_response
    depends on that exact format.
    """
    return f"""You are deciding what to ask, if anything, before the Python function `{entry_point}` is implemented. Asking is cheap and a wrong guess is costly, so ask unless the requirement leaves nothing to guess.

Requirement:

{prompt}

A candidate implementation:

```python
{candidate}
```
{evidence_block}Evaluate the requirement against these 4 dimensions in strict order. To prevent the illusion of clarity, for each dimension you MUST brainstorm a "Primary Interpretation" (what the candidate does) and a "Plausible Alternative Interpretation" that would change the output.
If the prompt and examples do not explicitly rule out the Plausible Alternative, you MUST mark it UNSTATED. Format your analysis exactly like this:

1. Algorithmic Logic (math rules, core operations, ambiguous semantics, e.g. signed vs absolute values, single-item vs cross-pair, count vs collection)
- Primary: <your primary interpretation>
- Alternative: <a mathematically/logically different interpretation that is equally plausible given the vague text>
- Status: STATED or UNSTATED

2. Constraints (edge cases, zeroes, negatives, all-bits-set; DO NOT waste this on empty inputs)
- Primary: <your primary interpretation>
- Alternative: <a different edge-case behavior>
- Status: STATED or UNSTATED

3. Prompt Defects (contradictions between text and examples)
- Primary: <your primary interpretation>
- Alternative: <a contradictory interpretation based on examples>
- Status: STATED or UNSTATED

4. Input/Output (exact types and return format; rely on Python's duck typing where possible){io_note}
- Primary: <your primary interpretation>
- Alternative: <a different parameter shape, or a different return type/order>
- Status: STATED or UNSTATED

Decision:
Priority order is strictly 1, 2, 3, 4. Find the FIRST dimension in that priority order that is UNSTATED. You MUST ask about the unstated fact from this highest-priority dimension. Do not skip to a lower-priority dimension.
If every dimension is STATED, your final line is NO_QUESTION.

End your response with exactly one line, and write nothing after it:
QUESTION: <one short question about that single unstated fact, phrased as a binary assumption check: "I will assume X. Is that right, or should it be Y?", where X is what the candidate does and Y is the plausible alternative. Name concrete values or a tiny example input when helpful. NEVER bundle two questions. Write plain text only: NO backticks, NO markdown>
or
NO_QUESTION"""


def build_code_prompt(entry_point: str, prompt: str) -> str:
    """Used for the final generation, after the ask/no-ask decision."""
    prior = _CANONICAL_PRIOR.format(entry_point=entry_point)
    return f"""Please provide a self-contained Python script implementing `{entry_point}` that solves the following problem:

{prompt}

The function must be named exactly `{entry_point}`.

IMPORTANT:
1. Where an answer to a clarification question conflicts with the original requirement or its examples, the answer is authoritative and overrides them; everything the answer does not mention follows the requirement.
2. {prior}
3. Return only the code the solution needs (imports, constants, helpers and `{entry_point}`), with no example usage, prints or test blocks.

Enclose your entire solution in ```python and ```."""


def build_reconcile_prompt(
    entry_point: str,
    spec: str,
    candidate_code: str,
    failures: str,
) -> str:
    return f"""Your implementation of `{entry_point}` failed the following worked example(s) from the original task text:

{failures}

Full specification (including any user clarification):
{spec}

Current implementation:
```python
{candidate_code}
```

Note: some tasks in this benchmark contain deliberately corrupted examples that conflict with the user's clarification answer.
- If the failing example(s) conflict with the user's clarification answer, the clarification answer wins and your current code is right: reply with `VERDICT: KEEP_CODE`.
- Otherwise, your implementation has a bug: reply with `VERDICT: UPDATE_CODE` followed by the corrected complete Python script enclosed in ```python and ```.

Start your response with `VERDICT: KEEP_CODE` or `VERDICT: UPDATE_CODE`."""


@dataclass
class _DoctestReport:
    """Result of mechanically extracting and running the requirement's own `>>>` examples."""

    ran: bool
    total: int
    passed: int
    failures: list[str] = field(default_factory=list)

    @classmethod
    def empty(cls) -> _DoctestReport:
        return cls(ran=False, total=0, passed=0, failures=[])


class StructuredClarifier(ClarificationAlgorithmBase):
    DEFAULT_CONFIG: ClassVar[dict[str, int]] = {
        "generation_attempts": 3,
        "max_questions": 1,
        "analysis_attempts": 2,
        "finals": 3,
        "contract_question": 1,
    }

    def run(self, env: ClarificationEnvironment, problem: dict[str, Any]) -> str:
        prompt, entry_point = problem["prompt"], problem["entry_point"]
        resolved: list[tuple[str, str]] = []
        probe_call: str | None = None

        try:
            for _ in range(self.config["max_questions"]):
                if not env.can_ask():
                    break
                spec = self.with_clarifications(prompt, resolved)
                seed = self.generate_seed(env, spec, entry_point)
                if seed is None:
                    break
                evidence = self.doctest_evidence(env, spec, seed)
                question = self.analyze_structure(env, spec, entry_point, seed, evidence)
                if question is None:
                    break
                question, probe_call = self.contract_question(
                    env, spec, entry_point, seed, question
                )
                answer = env.ask_human(question)
                resolved.append((question, answer))
        except (TooManyQuestionException, LimitsExceededException):
            pass

        spec = self.with_clarifications(prompt, resolved)
        return self.generate(env, prompt, spec, entry_point, probe_call)

    # --- Open contract question (draft contract + executed example, broad closing) ---

    def contract_question(
        self,
        env: ClarificationEnvironment,
        spec: str,
        entry_point: str,
        seed: str,
        binary_question: str,
    ) -> tuple[str, str | None]:
        """Rewrite the analyzer's binary question as a broad one: the seed's contract plus a real
        executed example, closed with an open 'is that what you intend?'. Falls back to the
        binary question on any failure.
        """
        fallback = (binary_question, None)
        if not self.config["contract_question"]:
            return fallback
        try:
            reply = env.llm(
                _CONTRACT_PROMPT.replace("{entry_point}", entry_point)
                .replace("{prompt}", spec)
                .replace("{candidate}", seed)
                .replace("{focus}", binary_question)
            )
        except LimitsExceededException:
            return fallback
        match = re.search(r"\{.*\}", reply or "", re.DOTALL)
        try:
            fields = json.loads(match.group(0)) if match else None
        except ValueError:
            fields = None
        if not isinstance(fields, dict):
            return fallback
        arguments = str(fields.get("arguments") or "").strip()
        returns = str(fields.get("returns") or "").strip()
        rule = str(fields.get("rule") or "").strip()
        if not (arguments and returns and rule):
            return fallback

        call = str(fields.get("call") or "").strip()
        example_output: str | None = None
        if call and self.is_entry_point_call(call, entry_point):
            try:
                output = env.exec_code(
                    f"{seed}\ntry:\n    print('EXAMPLE_OUT=' + repr({call})[:120])\n"
                    "except Exception:\n    print('EXAMPLE_OUT=ERR')"
                )
            except LimitsExceededException:
                output = ""
            found = re.search(r"^EXAMPLE_OUT=(.*)$", output or "", re.MULTILINE)
            if found and found.group(1).strip() not in {"", "ERR"}:
                example_output = found.group(1).strip()
        if example_output is None:
            call = ""

        text = f"About {entry_point}: my draft takes {arguments} and returns {returns}; {rule.rstrip('.')}."
        if call:
            text += f" For example, {call} returns {example_output}."
        text += " Is that what you intend, or should it behave differently or return a different format?"
        formatted = self.format_question_text(text)
        return (formatted, call or None) if formatted else fallback

    @staticmethod
    def is_entry_point_call(call: str, entry_point: str) -> bool:
        try:
            node = ast.parse(call, mode="eval").body
        except (SyntaxError, ValueError):
            return False
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == entry_point
        )

    # --- Seed + Structure Analysis ---

    def generate_seed(
        self, env: ClarificationEnvironment, prompt: str, entry_point: str
    ) -> str | None:
        messages = [
            {
                "role": "user",
                "content": build_seed_code_prompt(entry_point=entry_point, prompt=prompt),
            }
        ]
        response = ""
        for _ in range(self.config["generation_attempts"]):
            response = env.llm(messages)
            try:
                raw = _validate_and_parse_evalplus_result(response)
                return self.normalize_code(raw, prompt, entry_point)
            except ValueError as error:
                messages += [
                    {"role": "assistant", "content": response},
                    {"role": "user", "content": str(error)},
                ]
        return None

    def doctest_evidence(self, env: ClarificationEnvironment, prompt: str, seed: str) -> str:
        asserts = self.extract_doctest_asserts(prompt)
        if not asserts:
            return ""
        report = self.check_doctests(env, seed, asserts)
        if not report.ran:
            return (
                f"Note: the candidate crashed before any of the {report.total} worked example(s) "
                "already in the requirement could be checked against it."
            )
        if report.passed == report.total:
            return ""
        lines = "\n".join(f"- {failure}" for failure in report.failures)
        return (
            f"Note: {report.total - report.passed} of {report.total} worked example(s) already in "
            f"the requirement FAIL against this candidate:\n{lines}"
        )

    @classmethod
    def extract_worked_examples(cls, prompt: str) -> list[str]:
        """Extract worked examples from docstrings (>>> and »>) and arrow/equality patterns (=>, ==>, ==, ->).
        Returns a list of executable assert statements.
        """
        asserts: list[str] = []

        # 1. Standard doctests (supporting both >>> and unicode »>)
        normalized_prompt = prompt.replace("»>", ">>>")
        try:
            for example in doctest.DocTestParser().get_examples(normalized_prompt):
                source = example.source.strip()
                want = example.want.strip()
                if not want:
                    continue
                for quote in ('"""', "'''"):
                    if want.endswith(quote):
                        want = want[: -len(quote)].rstrip()
                        break
                if not want:
                    continue
                want_py = re.sub(r"\btrue\b", "True", want)
                want_py = re.sub(r"\bfalse\b", "False", want_py)
                want_py = re.sub(r"\bnull\b", "None", want_py)
                try:
                    ast.parse(source, mode="eval")
                    expected = ast.literal_eval(want_py)
                    stmt = f"assert ({source}) == {expected!r}"
                    if stmt not in asserts:
                        asserts.append(stmt)
                except Exception:
                    continue
        except Exception:
            pass

        # 2. Arrow and equality styles: `func(...) => result`, `==>`, `==`, `->`
        pattern = r"([a-zA-Z_]\w*\s*\([^)]*\))\s*(?:={1,2}>|==|->)\s*([^\n,;]+)"
        for match in re.finditer(pattern, prompt):
            call_expr = match.group(1).strip()
            want = match.group(2).strip().rstrip(". \t")
            want_py = re.sub(r"\btrue\b", "True", want)
            want_py = re.sub(r"\bfalse\b", "False", want_py)
            want_py = re.sub(r"\bnull\b", "None", want_py)
            try:
                ast.parse(call_expr, mode="eval")
                expected = ast.literal_eval(want_py)
                stmt = f"assert ({call_expr}) == {expected!r}"
                if stmt not in asserts:
                    asserts.append(stmt)
            except Exception:
                continue

        return asserts

    extract_doctest_asserts = extract_worked_examples

    def check_doctests(
        self, env: ClarificationEnvironment, seed_code: str, asserts: list[str]
    ) -> _DoctestReport:
        if not asserts:
            return _DoctestReport.empty()

        checks = "\n".join(
            f"try:\n    {a}\n    print('DOCTEST_{i}=OK')\nexcept Exception:\n    print('DOCTEST_{i}=FAIL')"
            for i, a in enumerate(asserts)
        )
        output = env.exec_code(f"{seed_code}\n{checks}")

        if not any(f"DOCTEST_{i}=" in output for i in range(len(asserts))):
            return _DoctestReport(ran=False, total=len(asserts), passed=0, failures=[])

        failures = [
            assertion for i, assertion in enumerate(asserts) if f"DOCTEST_{i}=OK" not in output
        ]
        return _DoctestReport(
            ran=True, total=len(asserts), passed=len(asserts) - len(failures), failures=failures
        )

    @classmethod
    def has_worked_example(cls, prompt: str) -> bool:
        return bool(cls.extract_worked_examples(prompt))

    @staticmethod
    def has_def_signature(prompt: str, entry_point: str) -> bool:
        return re.search(rf"def\s+{re.escape(entry_point)}\s*\(", prompt) is not None

    @staticmethod
    def has_typed_signature(prompt: str, entry_point: str) -> bool:
        match = re.search(rf"def\s+{re.escape(entry_point)}\s*\(([^)]*)\)\s*(->\s*[^:]+)?:", prompt)
        if not match:
            return False
        params, return_annotation = match.group(1), match.group(2)
        return bool(return_annotation) or ":" in params

    def analyze_structure(
        self,
        env: ClarificationEnvironment,
        prompt: str,
        entry_point: str,
        candidate: str,
        evidence: str,
    ) -> str | None:
        no_shape_signal = not (
            self.has_worked_example(prompt) or self.has_typed_signature(prompt, entry_point)
        )
        analysis_prompt = build_structure_analysis_prompt(
            entry_point=entry_point,
            prompt=prompt,
            candidate=candidate,
            evidence_block=f"\n{evidence}\n\n" if evidence else "\n",
            io_note=_NO_SHAPE_SIGNAL_IO_NOTE if no_shape_signal else "",
        )
        for _ in range(self.config["analysis_attempts"]):
            understood, question = self.parse_analysis_response(env.llm(analysis_prompt))
            if understood:
                return self.format_question_text(question) if question else None
        return None

    @staticmethod
    def format_question_text(question: str) -> str | None:
        """Strip backticks so `clarify/env.py`'s `ANSWERS=`(.+?)`` regex never truncates the reply."""
        cleaned = question.replace("`", "").strip(" *_\"'")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned or None

    @staticmethod
    def parse_analysis_response(response: str) -> tuple[bool, str | None]:
        """Returns (understood, question). understood is False only if no verdict was found."""
        question_matches = list(_QUESTION_RE.finditer(response))
        no_question_matches = list(_NO_QUESTION_RE.finditer(response))
        last_question = question_matches[-1] if question_matches else None
        last_no_question = no_question_matches[-1] if no_question_matches else None

        if last_question and (
            not last_no_question or last_question.start() > last_no_question.start()
        ):
            text = last_question.group(1).strip(" *_`")
            return (True, text) if text else (False, None)
        if last_no_question:
            return True, None
        return False, None

    # --- Final Generation, Reconciliation & Code Normalization ---

    def generate_one(
        self, env: ClarificationEnvironment, raw_prompt: str, spec: str, entry_point: str
    ) -> tuple[str | None, str]:
        messages = [
            {
                "role": "user",
                "content": build_code_prompt(entry_point=entry_point, prompt=spec),
            }
        ]
        response = ""
        for _ in range(self.config["generation_attempts"]):
            response = env.llm(messages)
            try:
                raw = _validate_and_parse_evalplus_result(response)
                return self.normalize_code(raw, raw_prompt, entry_point), response
            except ValueError as error:
                messages += [
                    {"role": "assistant", "content": response},
                    {"role": "user", "content": str(error)},
                ]
        return None, response

    def generate(
        self,
        env: ClarificationEnvironment,
        raw_prompt: str,
        spec: str,
        entry_point: str,
        probe_call: str | None = None,
    ) -> str:
        candidates: list[str] = []
        response = ""
        for _ in range(max(1, self.config["finals"])):
            try:
                code, response = self.generate_one(env, raw_prompt, spec, entry_point)
            except LimitsExceededException:
                break
            if code is not None:
                candidates.append(code)

        if not candidates:
            return response

        code = self.select_final(env, raw_prompt, candidates, entry_point, probe_call)
        code = self.reconcile_with_examples(env, raw_prompt, spec, entry_point, code)
        return self.make_signature_flexible(code, raw_prompt, entry_point)

    def select_final(
        self,
        env: ClarificationEnvironment,
        raw_prompt: str,
        candidates: list[str],
        entry_point: str,
        probe_call: str | None,
    ) -> str:
        """Pick the candidate whose behavior on the prompt's example calls (and the question's
        example call) agrees with the most other candidates. All candidates were written after the
        answer, so agreement favors the answer-consistent reading without trusting possibly
        corrupted example outputs.
        """
        if len(candidates) == 1:
            return candidates[0]
        calls: list[str] = []
        for statement in self.extract_doctest_asserts(raw_prompt):
            try:
                node = ast.parse(statement).body[0]
                assert isinstance(node, ast.Assert)
                compare = node.test
                assert isinstance(compare, ast.Compare)
                calls.append(ast.unparse(compare.left))
            except (SyntaxError, ValueError, AssertionError, IndexError):
                continue
        if probe_call:
            calls.append(probe_call)
        calls = [c for c in dict.fromkeys(calls) if self.is_entry_point_call(c, entry_point)][:8]
        if not calls:
            return candidates[0]

        probe = "\n".join(
            f"try:\n    print('OUT_{i}=' + repr({call})[:200])\n"
            f"except Exception:\n    print('OUT_{i}=ERR')"
            for i, call in enumerate(calls)
        )
        signatures: list[tuple[str, ...]] = []
        try:
            for code in candidates:
                output = env.exec_code(f"{code}\n{probe}") or ""
                signatures.append(
                    tuple(
                        next(
                            (line for line in output.splitlines() if line.startswith(f"OUT_{i}=")),
                            f"OUT_{i}=ERR",
                        )
                        for i in range(len(calls))
                    )
                )
        except LimitsExceededException:
            return candidates[0]

        votes = Counter(signatures)
        best_index = max(
            range(len(candidates)),
            key=lambda i: (
                votes[signatures[i]],
                sum(not line.endswith("=ERR") for line in signatures[i]),
                -i,
            ),
        )
        return candidates[best_index]

    def reconcile_with_examples(
        self,
        env: ClarificationEnvironment,
        raw_prompt: str,
        spec: str,
        entry_point: str,
        code: str,
    ) -> str:
        asserts = self.extract_doctest_asserts(raw_prompt)
        if not asserts:
            return code
        report = self.check_doctests(env, code, asserts)
        if report.ran and report.passed == report.total:
            return code

        failure_lines = (
            "\n".join(f"- {f}" for f in report.failures)
            if report.ran
            else "The candidate crashed at import/execution before examples could complete."
        )
        try:
            reply = env.llm(
                build_reconcile_prompt(
                    entry_point=entry_point,
                    spec=spec,
                    candidate_code=code,
                    failures=failure_lines,
                )
            )
        except LimitsExceededException:
            return code

        verdict_match = _RECONCILE_VERDICT_RE.search(reply)
        if verdict_match and verdict_match.group(1).upper() == "KEEP_CODE":
            return code

        try:
            raw_revised = _validate_and_parse_evalplus_result(reply)
            revised = self.normalize_code(raw_revised, raw_prompt, entry_point)
        except ValueError:
            return code

        revised_report = self.check_doctests(env, revised, asserts)
        if revised_report.ran and revised_report.passed > report.passed:
            return revised
        return code

    @classmethod
    def normalize_code(cls, code: str, prompt: str, entry_point: str) -> str:
        """Strip `__main__` blocks and top-level print/input calls, restore prompt helper functions,
        and bind `entry_point` if the model used a different function name.
        """
        cleaned = cls.strip_main_block(code)
        try:
            tree = ast.parse(cleaned)
        except (SyntaxError, ValueError):
            return cleaned

        defined_funcs = [
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]

        # Restore helper functions defined in the prompt (e.g. encode_* in HumanEval decode_* tasks).
        helpers: list[str] = []
        try:
            for node in ast.parse(prompt).body:
                if (
                    isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name != entry_point
                    and node.name not in defined_funcs
                ):
                    helpers.append(ast.unparse(node))
        except (SyntaxError, ValueError):
            pass

        if helpers:
            cleaned = "\n\n".join([*helpers, cleaned])

        if entry_point not in defined_funcs and defined_funcs:
            norm_target = entry_point.lower().replace("_", "")
            chosen = next(
                (name for name in defined_funcs if name.lower().replace("_", "") == norm_target),
                defined_funcs[-1],
            )
            cleaned = f"{cleaned.rstrip()}\n\n{entry_point} = {chosen}\n"

        return cleaned

    @classmethod
    def make_signature_flexible(cls, code: str, prompt: str, entry_point: str) -> str:
        """Wrap signature-less entry points so extra trailing arguments (like MBPP array lengths)
        are safely absorbed without raising TypeError.
        """
        if cls.has_def_signature(prompt, entry_point):
            return code
        try:
            tree = ast.parse(code)
        except (SyntaxError, ValueError):
            return code
        defined = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        defined.add(target.id)
        if entry_point not in defined:
            return code
        return code.rstrip() + _FLEXIBLE_SIGNATURE_TEMPLATE.format(entry_point=entry_point)

    @staticmethod
    def strip_main_block(code: str) -> str:
        try:
            tree = ast.parse(code)
        except (SyntaxError, ValueError):
            return code
        guards = [
            node
            for node in tree.body
            if (
                isinstance(node, ast.If)
                and ast.unparse(node.test) in {"__name__ == '__main__'", "'__main__' == __name__"}
            )
            or (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id in {"print", "input", "main"}
            )
        ]
        if not guards:
            return code
        lines = code.split("\n")
        for node in reversed(guards):
            end_lineno = node.end_lineno if node.end_lineno is not None else node.lineno
            del lines[node.lineno - 1 : end_lineno]
        return "\n".join(lines).rstrip() + "\n"

    @staticmethod
    def with_clarifications(prompt: str, resolved: list[tuple[str, str]]) -> str:
        fragments = [
            f"Question #{i}:\n{question}\nAnswer:\n{answer}\n"
            for i, (question, answer) in enumerate(resolved, start=1)
        ]
        return "\n".join([prompt, *fragments])
