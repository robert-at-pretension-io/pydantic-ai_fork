# Static Agentic Code Review System: Design Document

## Purpose

This system provides a modular, loopable, agent-based code review process for Merge Requests (MRs). It is designed to:

* Interpret and reason over Jira tickets.
* Understand the project architecture from static code only (no builds).
* Generate multiple valid implementation strategies.
* Compare the actual MR to those strategies and project conventions.
* Provide constructive, respectful, architecture-aware feedback.
* Output to CI logs and determine pipeline pass/fail.

---

## Runtime Environment Assumptions

* System runs in a Docker container managed by Kubernetes.
* Before agents run, the target repo is cloned to a defined directory.
* The repo is in an unbuilt state (no `node_modules`, `.venv`, or `dist`).
* Agents are informed of:

  * The repo path
  * Current working directory (`pwd`)
  * File tree (`ls -R` or equivalent summary)
  * Which tools they can call (`grep_repo`, `read_file`, etc.)
  * The full Jira ticket object (as structured JSON)

---

## Tools Available to Agents

* `grep_repo(pattern: str) -> List[Match]`
* `read_file(path: str, start: int, end: int) -> str`
* `list_files(path: Optional[str] = None) -> List[str]`
* `get_diff() -> List[DiffHunk]`

---

## Pipeline Modules (Agent Roles)

### 1. Ticket Identifier

* Extract Jira ticket ID from MR title/body/commits.
* Fetch Jira data: title, description, comments, acceptance criteria.
* Provide structured JSON to all agents.

### 2. Codebase Explorer (Loopable)

* Goal: Understand the project structure relevant to the Jira ticket.
* Uses `grep_repo`, `read_file`, `list_files` to explore.
* Operates in a loop with internal LLM planning and reflection.
* Exits when the agent believes it has enough context.
* Outputs:

  * Architecture notes
  * Key code snippets
  * Function/class/doc structure
  * Observed conventions and design patterns

### 3. Implementation Plan Generator (x3 Independent Loops)

* Runs three **independent** LLM loops in parallel.
* Each receives the **same Jira ticket and basic repo layout**, but starts fresh.
* Each produces a high-level implementation strategy:

  * `plan_name`
  * `summary`
  * `steps`
  * `files_touched`
  * `pros` / `cons`
  * `estimated_effort`
* The goal is diversity of thought, not consensus.

### 4. Diff Reviewer (Loopable)

* Input: Jira ticket, diff, selected plan, exploration context.
* Compares the user’s MR to each plan.
* Outputs feedback even if user did something different.
* Focus: architectural soundness, completion, test coverage, and style.
* Output includes summary, suggestions, and concerns.

### 5. Verifier Agent (Loopable, Independent)

* Input: Jira ticket + diff only.
* Has access to grep and file read tools.
* Does **not** see the plans or exploration.
* Goal: validate the MR as a standalone solution.
* Can confirm or refute LLM plans, or offer alternatives.

### 6. Synthesizer Agent

* Consumes all prior outputs:

  * Jira ticket
  * Exploration context
  * 3 plans
  * Diff review
  * Verifier opinion
* Produces one markdown-formatted, CI-friendly message with:

  * Overall summary
  * Which plan the MR resembled (if any)
  * Strengths and areas for improvement
  * Verifier's take

### 7. Final Gate Agent

* Analyzes synthesized output.
* Determines:

  * PASS (acceptable to merge)
  * SOFT FAIL (mergeable with suggestions)
  * HARD FAIL (architectural or logic problems)
* Controls CI pipeline outcome.

---

## Agent Loop Pattern

Each loopable agent follows this control flow:

```python
while not decider_agent.is_done(context):
    tool_agent.run_one_step(context)
```

This ensures:

* Dynamic exploration (agents aren't limited to one step)
* Early exits if the agent reaches confidence
* Optional retry strategies if tool calls fail

---

## Judgement Philosophy

* **Plans are not ground truth.** They represent *reasonable strategies*.
* **User code is respected.** Even if it doesn't match the plans, it may be valid.
* **Architecture wins.** The primary concern is alignment with existing design, not just match to LLM output.
* **Verifier is the tie-breaker.** It can validate user insight the LLM missed.
* **Synthesizer is diplomatic.** The final output should balance honesty, helpfulness, and tone.

---

## Output Format

* Synthesizer outputs Markdown (with headings, bullets, and optional emojis).
* Final verdict is logged plainly for CI/CD.
* Structured output can be stored as JSON for analytics/future posting to GitHub, Slack, etc.

---

## Optional Enhancements

* Timeout per agent or global runtime budget.
* Config file to define style rules or architectural standards.
* Support for `/.agentignore` to skip unhelpful paths.
* LLM confidence scoring to decide whether to invoke verifier.
* State caching across MRs (e.g., for exploratory knowledge reuse).

---

## Summary

This system represents a robust, introspective, and developer-respecting approach to automated code review. Each agent contributes unique, loopable insight toward a unified, architecture-aligned response that surfaces during CI, reduces review burden, and improves quality without rigidity.
