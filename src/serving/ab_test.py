"""A/B testing framework for recommendation strategies."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from scipy import stats


class Variant(str, Enum):
    CONTROL = "control"
    TREATMENT = "treatment"


@dataclass
class VariantMetrics:
    impressions: int = 0
    clicks: int = 0
    conversions: int = 0
    total_diversity: float = 0.0
    total_novelty: float = 0.0

    @property
    def ctr(self) -> float:
        return self.clicks / max(self.impressions, 1)

    @property
    def conversion_rate(self) -> float:
        return self.conversions / max(self.impressions, 1)

    @property
    def avg_diversity(self) -> float:
        return self.total_diversity / max(self.impressions, 1)

    @property
    def avg_novelty(self) -> float:
        return self.total_novelty / max(self.impressions, 1)


@dataclass
class ABTestResult:
    test_name: str
    control: VariantMetrics
    treatment: VariantMetrics
    is_significant: bool = False
    p_value: float = 1.0
    lift: float = 0.0
    concluded: bool = False
    winner: str | None = None


class RecommendationABTest:
    """A/B test for comparing recommendation strategies."""

    def __init__(
        self,
        test_name: str,
        min_sample_size: int = 100,
        significance_level: float = 0.05,
        traffic_split: float = 0.5,
    ) -> None:
        self.test_name = test_name
        self.min_sample_size = min_sample_size
        self.significance_level = significance_level
        self.traffic_split = traffic_split
        self._metrics: dict[Variant, VariantMetrics] = {
            Variant.CONTROL: VariantMetrics(),
            Variant.TREATMENT: VariantMetrics(),
        }
        self._concluded = False
        self._winner: str | None = None

    def assign_variant(self, user_id: str) -> Variant:
        """Deterministically assign a user to a variant based on user_id hash."""
        h = hashlib.md5(f"{self.test_name}:{user_id}".encode()).hexdigest()
        bucket = int(h[:8], 16) / 0xFFFFFFFF
        return Variant.TREATMENT if bucket < self.traffic_split else Variant.CONTROL

    def record_impression(
        self,
        user_id: str,
        diversity: float = 0.0,
        novelty: float = 0.0,
    ) -> Variant:
        """Record that recommendations were shown to a user."""
        variant = self.assign_variant(user_id)
        m = self._metrics[variant]
        m.impressions += 1
        m.total_diversity += diversity
        m.total_novelty += novelty
        return variant

    def record_click(self, user_id: str) -> None:
        variant = self.assign_variant(user_id)
        self._metrics[variant].clicks += 1

    def record_conversion(self, user_id: str) -> None:
        variant = self.assign_variant(user_id)
        self._metrics[variant].conversions += 1

    def get_results(self) -> ABTestResult:
        """Compute current A/B test results with statistical significance."""
        control = self._metrics[Variant.CONTROL]
        treatment = self._metrics[Variant.TREATMENT]

        result = ABTestResult(
            test_name=self.test_name,
            control=control,
            treatment=treatment,
        )

        # Need minimum samples in both groups
        if control.impressions < self.min_sample_size or treatment.impressions < self.min_sample_size:
            return result

        # Two-proportion z-test on CTR
        p_value = self._two_proportion_z_test(
            control.clicks, control.impressions,
            treatment.clicks, treatment.impressions,
        )

        result.p_value = p_value
        result.is_significant = p_value < self.significance_level

        if control.ctr > 0:
            result.lift = (treatment.ctr - control.ctr) / control.ctr
        elif treatment.ctr > 0:
            result.lift = 1.0

        if result.is_significant:
            result.concluded = True
            result.winner = (
                Variant.TREATMENT.value if treatment.ctr > control.ctr else Variant.CONTROL.value
            )
            self._concluded = True
            self._winner = result.winner

        return result

    @staticmethod
    def _two_proportion_z_test(
        successes_a: int, n_a: int, successes_b: int, n_b: int
    ) -> float:
        """Two-proportion z-test, returns p-value."""
        if n_a == 0 or n_b == 0:
            return 1.0

        p_a = successes_a / n_a
        p_b = successes_b / n_b
        p_pool = (successes_a + successes_b) / (n_a + n_b)

        if p_pool == 0 or p_pool == 1:
            return 1.0

        se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
        if se == 0:
            return 1.0

        z = (p_b - p_a) / se
        p_value = float(2 * (1 - stats.norm.cdf(abs(z))))
        return p_value

    @property
    def is_concluded(self) -> bool:
        return self._concluded

    @property
    def winner(self) -> str | None:
        return self._winner
