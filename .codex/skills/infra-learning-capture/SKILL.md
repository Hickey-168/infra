---
name: infra-learning-capture
description: Capture durable knowledge from discussions about this repository's AI infrastructure curriculum. Use automatically whenever a conversation about inference, training, GPU/Triton/CUDA, communication, distributed systems, deployment, storage, experiments, or the learner's understanding produces a new mechanism, correction, experiment result, misconception, dependency, or independently learnable leaf.
---

# Infra Learning Capture

Treat a discussion that produces durable learning as incomplete until its reusable result is either recorded in this repository or explicitly judged not recordable.

## Before Every Final Answer

After reaching a conclusion, run this decision:

1. **No durable delta:** The exchange is casual, repeats existing material, or ends in an unresolved question. Do not create noise. State that no repository update was needed only when useful.
2. **Durable delta:** The exchange establishes a reusable mechanism, corrects an existing explanation, provides a reproducible observation, exposes a common misconception, changes a dependency/priority, or identifies a new independently learnable leaf. Update the repository before replying.
3. **Uncertain delta:** Preserve the uncertainty. Add a targeted experiment or a clearly marked open question only when it has a clear owner/path; never write a speculation as course fact.

Default to capturing durable deltas without asking the learner for a second instruction.

## Route The Knowledge

Use the narrowest durable home. Read the existing file before editing and preserve learner-owned material.

| Result of discussion | Repository home |
|---|---|
| Why the topic matters, prerequisites, priority, next link | `<leaf>/tutorial.md` |
| General mechanism, derivation, worked example, source-backed correction | `<leaf>/lesson.md` |
| Reproducible command, shape/state trace, benchmark, profiler observation | `<leaf>/lab.*` or a focused experiment record |
| New retrieval question, counterfactual, estimate, common wrong answer | `<leaf>/questions.md` |
| Complete, discussion-derived recall material | `<leaf>/review.md` |
| Cross-domain dependency, priority change, route correction | `atlas/` |
| Personal wording, a wrong prediction, reflection, learner's final answer | `note.md` - never write this automatically |

For an existing P0 leaf, maintain the two-layer course boundary: `tutorial.md` is the short导学; `lesson.md` is the self-contained course. Do not replace a course with a link list.

## Ensure The Leaf Exists First

Capture work owns the setup work. Before writing a durable delta:

1. Resolve the narrowest existing leaf using `atlas/learning-directory.md` and the real tree.
2. If that leaf directory exists but the scaffold-owned files are missing, run `python3 tools/scaffold_leaf.py <leaf>` yourself. Never ask the learner to run it.
3. If no suitable leaf exists, create the smallest independently learnable leaf directory, add it to the appropriate atlas map/directory, then run the scaffold command yourself.
4. Do not use `--force` on learner-owned files. Create or revise AI-owned `tutorial.md`, `lesson.md`, and `review.md` explicitly after scaffolding.

Do not treat a broad framework/topic directory as a leaf. Apply the New Leaf Rule before creating one.

## Write For Both Initial Learning And Review

Discussion capture is not a terse changelog. A later reader must be able to recover the final correct model without reopening the chat or searching for missing context.

- Update `tutorial.md` when the discussion changes positioning, prerequisites, priority, or the right learning path.
- Update `lesson.md` with the full generalizable explanation: definitions in context, causal chain, worked example or formula, invariants, counterfactuals, boundaries, and the final correction. Replace vague or superseded prose; do not append an orphan bullet that requires the original chat.
- Create or update `review.md` whenever a leaf has a durable discussion result. Write it for a learner who has already completed the course: concise conclusion first, then complete mechanism, final answers to the discussed questions, rejected misconceptions and why, evidence/source labels, operational implications, and a short recall checklist. Do not require the reader to repeat `tutorial -> lesson -> lab -> questions` to understand the conclusion.

Record facts, observations, and inferences separately. Include the exact final answer reached in the discussion and any important condition that changes that answer. Never write “as discussed above” or rely on chat history.

## Evidence Labels

Keep these categories separate in prose:

- **事实:** Official documentation, paper, source contract, or a stable mathematical statement.
- **观察:** A command or experiment actually run on a named environment; retain versions, units, and limits.
- **推断:** An explanation inferred from facts/observations; label its assumptions.

When a library version may matter, verify primary documentation before writing a version-sensitive claim. Do not claim GPU performance from a Mac/CPU experiment.

## New Leaf Rule

Create a new leaf only when all are true:

1. It names one mechanism or boundary, not a broad framework.
2. It has a clear problem, prerequisite, and downstream use.
3. It can be learned independently in one focused session.
4. It has a toy implementation, trace, test, benchmark, profiler capture, or other checkable evidence.

If the conclusion is only an aspect of an existing leaf, update that leaf instead.

## Capture Workflow

1. Identify the conclusion and classify it as fact, observation, or inference.
2. Locate or create the leaf and run the scaffold if its base files are absent.
3. Read the target course files. Write the full reusable conclusion to `lesson.md` and review-ready synthesis to `review.md`; update the导学 only when its map changes.
4. Add or revise a lab/question only when it directly tests the newly captured claim.
5. Verify links, syntax, notebook JSON, or tests in proportion to the edit.
6. In the final response, state the concrete paths updated and the one-sentence reason. Do not turn the final response into a duplicate lesson.

## Conflict Resolution

If a discussion contradicts existing course material, do not silently overwrite it. Prefer a primary source or a focused experiment, revise the course, and record the boundary/correction. If it remains unresolved, keep the current lesson conservative and add an explicitly open verification task rather than inventing certainty.
