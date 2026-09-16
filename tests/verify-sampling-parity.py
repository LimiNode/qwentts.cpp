"""Compare the native explicit-u sampler with a dependency-free Python reference.

The executable emits only selected token IDs for fixed synthetic vectors.  This
script deliberately keeps the reference implementation small and transparent:
it is a policy oracle for the model-free contract, not a second production
sampler.
"""

from __future__ import annotations

import math
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Vector:
    name: str
    logits: tuple[float, ...]
    temperature: float
    top_k: int
    top_p: float
    repetition_penalty: float
    history: tuple[int, ...]
    uniform_u: float


VECTORS = (
    Vector("positive_repetition", (4.0, 3.0, 3.5), 1.0, 2, 1.0, 2.0, (0,), 0.1),
    Vector("negative_repetition", (-4.0, -3.0, -3.5), 1.0, 2, 1.0, 2.0, (0,), 0.1),
    Vector("top_k_one", (1.0, 9.0, 8.0), 1.0, 1, 1.0, 1.0, (), 0.99),
    # The native sampler keeps the filtered candidates in vocabulary order,
    # matching torch.multinomial.  A probability-sorted CDF would select 3.
    Vector("top_k_vocab_order", (0.0, 3.0, 0.0, 4.0), 1.0, 2, 1.0, 1.0, (), 0.20),
    Vector("top_k_zero", (1.0, 2.0, 3.0), 1.0, 0, 1.0, 1.0, (), 0.01),
    Vector("top_p_one", (4.0, 3.0, 2.0), 1.0, 0, 1.0, 1.0, (), 0.90),
    Vector("top_p_crossing", (4.0, 3.0, 2.0, 1.0), 1.0, 0, 0.70, 1.0, (), 0.95),
    Vector("uniform_low", (1.0, 0.0, -1.0), 1.0, 0, 1.0, 1.0, (), 0.0000001),
    Vector("uniform_high", (1.0, 0.0, -1.0), 1.0, 0, 1.0, 1.0, (), 0.9999999),
    Vector("ties", (1.0, 1.0, 1.0), 1.0, 0, 1.0, 1.0, (), 0.50),
    Vector("temperature", (4.0, 3.0, 2.0), 2.0, 0, 1.0, 1.0, (), 0.75),
)


def reference_sample(vector: Vector) -> int:
    logits = list(vector.logits)
    if vector.temperature <= 0.0:
        return max(range(len(logits)), key=logits.__getitem__)

    if vector.repetition_penalty != 1.0:
        for token in set(vector.history):
            if 0 <= token < len(logits):
                score = logits[token]
                logits[token] = (
                    score * vector.repetition_penalty
                    if score < 0.0
                    else score / vector.repetition_penalty
                )

    logits = [score / vector.temperature for score in logits]

    if 0 < vector.top_k < len(logits):
        threshold = sorted(logits, reverse=True)[vector.top_k - 1]
        logits = [score if score >= threshold else -math.inf for score in logits]

    if 0.0 < vector.top_p < 1.0:
        max_logit = max(logits)
        exp_values = [
            math.exp(score - max_logit) if math.isfinite(score) else 0.0
            for score in logits
        ]
        sum_exp = sum(exp_values)
        probabilities = [value / sum_exp for value in exp_values]
        ordered = sorted(range(len(logits)), key=lambda index: probabilities[index], reverse=True)
        cumulative = 0.0
        for position, index in enumerate(ordered):
            if position > 0 and cumulative >= vector.top_p:
                logits[index] = -math.inf
            cumulative += probabilities[index]

    max_logit = max(logits)
    weights = [math.exp(score - max_logit) for score in logits]
    target = vector.uniform_u * sum(weights)
    cumulative = 0.0
    for index, weight in enumerate(weights):
        cumulative += weight
        if cumulative >= target:
            return index
    return len(weights) - 1


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} PATH_TO_TEST_SAMPLING_CONTRACT", file=sys.stderr)
        return 2

    completed = subprocess.run(
        [sys.argv[1], "--dump-parity-vectors"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        print(completed.stderr, file=sys.stderr, end="")
        return completed.returncode

    rows = completed.stdout.strip().splitlines()
    if not rows or rows[0] != "name\tselected\tu":
        print("native sampler emitted an invalid parity header", file=sys.stderr)
        return 1

    observed: dict[str, int] = {}
    for row in rows[1:]:
        name, selected, _uniform = row.split("\t")
        observed[name] = int(selected)

    expected = {vector.name: reference_sample(vector) for vector in VECTORS}
    if observed != expected:
        print(f"sampling parity mismatch:\nexpected={expected}\nobserved={observed}", file=sys.stderr)
        return 1

    print(f"sampling explicit-u parity passed ({len(expected)} vectors)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
