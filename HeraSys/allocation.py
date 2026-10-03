"""Rolling-horizon batch and engine-share allocation (Appendix A.3).

The existing engine API executes one batch at a time on each instance. Its
executable allocation grid is therefore equal shares 1/b within a batch, rather
than arbitrary GPU SM percentages. Reservation is accounted in those shares,
including already committed batches. No unsupported hardware shares are claimed.
"""

import heapq
import statistics
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class Allocation:
    tasks: list
    shares: list
    duration_s: float
    objective_s: float
    penalty_s: float


def session_continuations(tasks, queries, roots=None):
    """Find unscheduled linear continuations that retain the same KV session.

    A prefill does not release a session slot: its decode chain must also run.
    At a branch we stop this conservative rollout rather than inventing a merge
    or a free KV fork. Actual admission and preemption still happen per node.
    """
    roots = list(tasks.values()) if roots is None else roots
    roots = [t for t in roots if t.node.config.get("_hera_session")]
    owners = {q for t in roots for q, _ in t.owners}
    children = defaultdict(set)
    for q in owners:
        state = queries[q]
        for child, parents in state.parents.items():
            if len(parents) != 1:
                continue
            parent = next(iter(parents))
            session = state.sessions.get(child)
            if session and session == state.sessions.get(parent):
                children[state.mapping[parent]].add(state.mapping[child])
    continuations = {}
    for task in roots:
        task_id = task.id
        if not task.node.config.get("_hera_session") or task.status not in {
            "pending",
            "running",
        }:
            continue
        chain, current, seen = [], task_id, {task_id}
        while len(children.get(current, ())) == 1:
            child_id = next(iter(children[current]))
            child = tasks.get(child_id)
            if child is None or child_id in seen or child.status != "pending":
                break
            if child.node.engine_type != task.node.engine_type:
                break
            chain.append(child)
            seen.add(child_id)
            current = child_id
        if chain:
            continuations[task_id] = chain
    return continuations


def balancing_weight(load, completion_estimates, config):
    """High load favors mean latency; a widening predicted tail lowers lambda."""
    if not config.tail_feedback or len(completion_estimates) < 2:
        return load
    median = statistics.median(completion_estimates)
    tail_pressure = max(
        1.0,
        max(completion_estimates) / max(config.epsilon, config.tail_ratio * median),
    )
    return max(min(load, config.lambda_floor), load / tail_pressure)


def choose_allocation(
    groups,
    *,
    slot,
    capacity,
    max_batch_size,
    profiles,
    queries,
    remaining,
    now,
    balance,
    busy,
    affinity,
    reservation,
    config,
    eligible,
    limit,
    continuations=None,
):
    """Enumerate executable (b,a), then roll out the rest of the ready horizon.

    `groups` retain priority order within homogeneous batches. Every candidate is
    evaluated against the same ready set and in-flight commitments. Remaining
    service after the ready frontier is retained in each query's completion cost.
    """
    if not groups or not eligible:
        return None
    continuations = continuations or {}
    ordered = sorted(groups, key=lambda group: (-group[0].priority, group[0].id))
    group_of = {task.id: i for i, group in enumerate(ordered) for task in group}
    all_tasks = [task for group in ordered for task in group]
    credited = defaultdict(float)
    scales = {
        q: remaining[q]
        / max(
            config.epsilon,
            sum(
                cost
                for name, cost in queries[q].predictions.items()
                if name not in queries[q].completed
            ),
        )
        for q in remaining
    }
    horizon = {}
    for task in all_tasks + [t for batch in busy.values() for t in batch]:
        horizon[task.id] = task
        horizon.update((t.id, t) for t in continuations.get(task.id, ()))
    for task in horizon.values():
        for q, name in task.owners:
            if q not in remaining:
                continue
            state = queries[q]
            cost = state.predictions[name] * scales[q]
            credited[q] += cost
    future = {q: max(0.0, value - credited[q]) for q, value in remaining.items()}
    median_remaining = statistics.median(remaining.values())
    median_phi = max(
        config.epsilon,
        statistics.median(sum(task.owners.values()) for task in all_tasks),
    )
    waiting_weights = {
        task.id: (
            config.srtf_penalty
            * median_remaining
            / max(
                config.epsilon,
                min(remaining[q] for q, _ in task.owners if q in remaining),
            )
            + config.topology_penalty * sum(task.owners.values()) / median_phi
        )
        for task in all_tasks
    }
    affinities = {task.id: affinity(task) for task in all_tasks}
    duration_cache = {}
    cohort_cache = {}
    # Most generation chains use the same decode quantum. Their cost depends
    # on the number of live sequences at each step, not request identity. Cache
    # that small shape so exhaustive batch search does not repeatedly rebuild
    # thousands of otherwise identical future decode batches.
    tail_nodes = [
        child
        for task in all_tasks + [t for batch in busy.values() for t in batch]
        for child in continuations.get(task.id, ())
    ]
    uniform_tail = False
    if tail_nodes and hasattr(profiles, "estimate_batch"):
        tail_node = tail_nodes[0].node
        tail_key = profiles.key(tail_node)
        tail_unit = max(float(tail_node.relative_ms or 1) / 1000, 1e-6)
        uniform_tail = len({t.id for t in tail_nodes}) == len(tail_nodes) and all(
            profiles.key(t.node) == tail_key
            and max(float(t.node.relative_ms or 1) / 1000, 1e-6) == tail_unit
            for t in tail_nodes
        )
    tail_costs = {}

    def duration(tasks):
        key = tuple(t.id for t in tasks)
        if key not in duration_cache:
            duration_cache[key] = profiles.batch_time([t.node for t in tasks])
        return duration_cache[key]

    def cohort_duration(tasks):
        key = tuple(t.id for t in tasks)
        if key not in cohort_cache:
            total = duration(tasks)
            chains = [continuations.get(t.id, ()) for t in tasks]
            if uniform_tail:
                lengths = tuple(sorted(len(chain) for chain in chains))
                if lengths not in tail_costs:
                    tail_costs[lengths] = sum(
                        profiles.estimate_batch(
                            tail_node, size, sum([tail_unit] * size)
                        )
                        for step in range(max(lengths, default=0))
                        if (size := sum(length > step for length in lengths))
                    )
                cohort_cache[key] = total + tail_costs[lengths]
                return cohort_cache[key]
            for step in range(max(map(len, chains), default=0)):
                frontier = {
                    chain[step].id: chain[step] for chain in chains if step < len(chain)
                }
                total += duration(list(frontier.values()))
            cohort_cache[key] = total
        return cohort_cache[key]

    initial_times = [0.0] * capacity
    initial_finishes = defaultdict(float)
    for worker, tasks in busy.items():
        finish = max(0.0, tasks[0].expected_finish - now)
        finish += cohort_duration(tasks) - duration(tasks)
        initial_times[worker] = finish
        for task in tasks:
            for q, _ in task.owners:
                initial_finishes[q] = max(initial_finishes[q], finish)

    def rollout(first):
        times, finishes = list(initial_times), dict(initial_finishes)
        starts = {}

        def commit(tasks, worker):
            start = times[worker]
            times[worker] += cohort_duration(tasks)
            for task in tasks:
                starts[task.id] = start
                for q, _ in task.owners:
                    finishes[q] = max(finishes.get(q, 0.0), times[worker])

        commit(first, slot)
        selected = {task.id for task in first}
        queues = [[t for t in group if t.id not in selected] for group in ordered]
        heads = [
            (-queue[0].priority, queue[0].id, i, 0)
            for i, queue in enumerate(queues)
            if queue
        ]
        heapq.heapify(heads)
        while heads:
            _, _, group_id, offset = heapq.heappop(heads)
            queue = queues[group_id]
            head = queue[offset]
            fixed = affinities[head.id]
            worker = (
                fixed
                if fixed is not None
                else min(range(capacity), key=lambda i: times[i])
            )
            batch = []
            # Evaluate the current class with one consistent batching policy.
            # Otherwise a tiny committed batch is credited with all subsequent
            # batches running at maximum efficiency, then every later decision
            # repeats the same optimistic assumption and commits another tiny one.
            batch_limit = (
                min(max_batch_size, len(first))
                if group_id == group_of[first[0].id]
                else max_batch_size
            )
            while offset < len(queue) and len(batch) < batch_limit:
                task = queue[offset]
                if affinities[task.id] not in {None, worker}:
                    break
                batch.append(task)
                offset += 1
            commit(batch, worker)
            if offset < len(queue):
                heapq.heappush(
                    heads, (-queue[offset].priority, queue[offset].id, group_id, offset)
                )
        completions = [
            now - queries[q].started + finishes.get(q, 0.0) + future[q]
            for q in remaining
        ]
        objective = balance * statistics.fmean(completions) + (1 - balance) * max(
            completions
        )
        # Optional soft priority constraints. Normalize inverse remaining time
        # and downstream contribution so each term has units of waiting seconds.
        penalty = sum(starts[task.id] * waiting_weights[task.id] for task in all_tasks)
        return objective, penalty / max(1, len(all_tasks))

    best, best_key = None, None
    for group in ordered:
        feasible = [task for task in group if task.id in eligible]
        if not feasible:
            continue
        for size in range(1, min(limit(feasible[0]), len(feasible)) + 1):
            tasks = feasible[:size]
            shares = [1.0 / size] * size
            heavy_count = sum(task.heavy for task in tasks)
            if heavy_count / size + config.epsilon < reservation:
                continue
            if heavy_count and shares[0] + config.epsilon < config.min_heavy_share:
                continue
            objective, penalty = rollout(tasks)
            key = (objective + penalty, -sum(t.priority for t in tasks), -size)
            if best_key is None or key < best_key:
                best_key = key
                best = Allocation(tasks, shares, duration(tasks), objective, penalty)
    return best
