"""Equations 3–7 and load controls; durations consistently use seconds."""

import math
import statistics


def remaining(predictions, completed, attained_s, config):
    todo = sum(v for k, v in predictions.items() if k not in completed)
    if not config.dynamic_remaining:
        return sum(predictions.values()) if todo else 0.0
    done = sum(v for k, v in predictions.items() if k in completed)
    # F is unspecified in the paper: identity below 1 is clamped to avoid collapse
    # on cache hits; overrun increases the remaining estimate (Eq. 4 / App. A.1.4).
    correction = max(1.0, attained_s / (done + config.epsilon)) if done else 1.0
    return correction * todo


def long_queries(estimates, eta):
    if not estimates:
        return set()
    values = list(estimates.values())
    threshold = statistics.fmean(values) + eta * statistics.pstdev(values)
    return {q for q, value in estimates.items() if value > threshold}


def priority(remaining_s, contribution, heavy, config):
    if config.policy == "fcfs":
        return 0.0
    srtf = config.alpha / (remaining_s + config.epsilon)
    if config.policy == "srtf":
        return srtf
    return srtf + config.gamma * contribution + config.delta * int(heavy)


def load_factor(active_queries, pending_nodes, capacity, config):
    return min(
        1.0,
        max(
            active_queries / config.load_scale,
            pending_nodes / (config.load_scale * max(1, capacity)),
        ),
    )


def split_factor(load, size, config):
    desired = (
        round(1 + (1 - load) * (config.max_splits - 1))
        if config.adaptive_splitting
        else config.fixed_splits
    )
    return max(1, min(desired, size))


def reserved_slots(capacity, config):
    return math.ceil(capacity * config.rho) if config.resource_skew else 0


def batch_size(candidates, profiles, load):
    """Discrete local surrogate for App. A.3; one full instance per batch.

    Reward saved total service at high load and penalize added head latency at
    low load. Enumerate all feasible batch sizes using measured batch profiles.
    """
    if profiles is None:
        return len(candidates)
    head = profiles.predict(candidates[0].node)
    choices = []
    sequential = 0.0
    for size in range(1, len(candidates) + 1):
        sequential += profiles.predict(candidates[size - 1].node)
        duration = profiles.batch_time([t.node for t in candidates[:size]])
        gain = load * (sequential - duration) - (1 - load) * max(0, duration - head)
        choices.append((gain, size))
    return max(choices)[1]
