"""HeraSys configuration, including the Appendix B.2 query-aging extension."""

import math
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class HeraConfig:
    alpha: float = 1.0
    gamma: float = 0.3
    delta: float = 0.5
    rho: float = 0.2
    eta: float = 1.0
    epsilon: float = 1e-6
    # Eq. 5: workload measured in predicted seconds; depth and branch terms unitless.
    omega_work: float = 1.0
    omega_depth: float = 0.1
    omega_branch: float = 0.1
    # Eq. 5 / A.1.1: Nv is the successor set. Reverse depth already
    # represents the serial path; a later descendant is not another branch
    # enabled by v. Keep the former transitive interpretation as an ablation.
    contribution_scope: str = "direct"
    policy: str = "joint"
    fusion: bool = True
    value_fusion: bool = True
    completed_reuse: bool = True
    resource_skew: bool = True
    adaptive_batching: bool = True
    adaptive_splitting: bool = True
    dynamic_remaining: bool = True
    joint_allocation: bool = True
    tail_feedback: bool = True
    tail_ratio: float = 2.0
    lambda_floor: float = 0.05
    min_heavy_share: float = 1 / 64
    min_heavy_splits: int = 2
    srtf_penalty: float = 0.01
    topology_penalty: float = 0.01
    fusion_window_s: float = 0.002
    max_batch_wait_s: float = 0.005
    load_scale: float = 30.0
    max_splits: int = 4
    fixed_splits: int = 2
    decode_quantum: int = 16
    cache_entries: int = 4096
    cache_ttl_s: float = 300.0
    trace: bool = True
    audit_nodes: bool = False
    max_ready_wait_s: float | None = None
    max_query_wait_s: float | None = 2.0

    def __post_init__(self):
        numeric = [
            self.alpha,
            self.gamma,
            self.delta,
            self.omega_work,
            self.omega_depth,
            self.omega_branch,
            self.eta,
            self.rho,
            self.epsilon,
            self.fusion_window_s,
            self.max_batch_wait_s,
            self.cache_ttl_s,
            self.load_scale,
            self.tail_ratio,
            self.lambda_floor,
            self.min_heavy_share,
            self.srtf_penalty,
            self.topology_penalty,
        ]
        if any(not math.isfinite(value) or value < 0 for value in numeric):
            raise ValueError("Scheduler parameters must be finite and nonnegative")
        if self.max_ready_wait_s is not None and (
            not math.isfinite(self.max_ready_wait_s) or self.max_ready_wait_s <= 0
        ):
            raise ValueError("The optional aging threshold must be finite and positive")
        if self.max_query_wait_s is not None and (
            not math.isfinite(self.max_query_wait_s) or self.max_query_wait_s <= 0
        ):
            raise ValueError("The query aging threshold must be finite and positive")
        if self.policy not in {"joint", "fcfs", "srtf"}:
            raise ValueError("policy must be joint, fcfs, or srtf")
        if self.contribution_scope not in {"direct", "transitive"}:
            raise ValueError("contribution_scope must be direct or transitive")
        if not 0 <= self.rho <= 1 or self.eta < 0 or self.epsilon <= 0:
            raise ValueError("Invalid reservation, long-query threshold, or epsilon")
        if min(self.max_splits, self.fixed_splits, self.decode_quantum) < 1:
            raise ValueError("Split factors and decode quantum must be positive")
        if min(self.fusion_window_s, self.max_batch_wait_s, self.cache_ttl_s) < 0:
            raise ValueError("Timeouts must be nonnegative")
        if self.load_scale <= 0 or self.cache_entries < 0:
            raise ValueError("Invalid load scale or cache capacity")
        if self.tail_ratio < 1 or not 0 <= self.lambda_floor <= 1:
            raise ValueError("Invalid tail feedback parameters")
        if not 0 < self.min_heavy_share <= 1 or self.min_heavy_splits < 1:
            raise ValueError("Invalid heavy-node minimum allocation")

    @classmethod
    def variant(cls, name: str):
        base = cls()
        unaged = replace(base, max_query_wait_s=None)
        variants = {
            "HeraSys": base,
            "herasys": base,
            "herasys_unaged": unaged,
            "herasys_legacy": replace(
                unaged, joint_allocation=False, tail_feedback=False
            ),
            "no_joint_allocation": replace(base, joint_allocation=False),
            "no_tail_feedback": replace(base, tail_feedback=False),
            "no_pipeline": replace(base, adaptive_splitting=False),
            "herasys_transitive": replace(unaged, contribution_scope="transitive"),
            "herasys_direct": unaged,
            "herasys_aged": replace(base, max_query_wait_s=2.0),
            "herasys_direct_aged": replace(
                base, contribution_scope="direct", max_query_wait_s=2.0
            ),
            "herasys_legacy_direct_aged": replace(
                base,
                contribution_scope="direct",
                max_query_wait_s=2.0,
                joint_allocation=False,
                tail_feedback=False,
            ),
            "herasys_node_aged": replace(unaged, max_ready_wait_s=2.0),
            "herasys_aged_no_fusion": replace(
                base, max_query_wait_s=2.0, fusion=False, completed_reuse=False
            ),
            "herasys_direct_aged_no_fusion": replace(
                base,
                contribution_scope="direct",
                max_query_wait_s=2.0,
                fusion=False,
                completed_reuse=False,
            ),
            "no_fusion": replace(base, fusion=False, completed_reuse=False),
            "fcfs": replace(
                unaged, policy="fcfs", resource_skew=False, joint_allocation=False
            ),
            "srtf": replace(
                unaged, policy="srtf", resource_skew=False, joint_allocation=False
            ),
            "no_skew": replace(base, resource_skew=False),
            "no_adaptive": replace(
                base,
                adaptive_batching=False,
                adaptive_splitting=False,
                joint_allocation=False,
            ),
            # Shared-engine control; the complete Ayo baseline has its own runner.
            "ayo_static": replace(
                unaged,
                policy="fcfs",
                fusion=False,
                completed_reuse=False,
                resource_skew=False,
                adaptive_batching=False,
                adaptive_splitting=False,
                dynamic_remaining=False,
                joint_allocation=False,
                tail_feedback=False,
            ),
        }
        if name not in variants:
            raise ValueError(f"Unknown variant {name!r}; choose {list(variants)}")
        return variants[name]
