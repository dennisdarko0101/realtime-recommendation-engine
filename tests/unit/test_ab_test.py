"""Tests for A/B testing framework."""

import pytest

from src.serving.ab_test import RecommendationABTest, Variant


def test_deterministic_assignment() -> None:
    ab = RecommendationABTest("test_1")
    v1 = ab.assign_variant("user_001")
    v2 = ab.assign_variant("user_001")
    assert v1 == v2  # Same user always gets same variant


def test_two_variants_exist() -> None:
    ab = RecommendationABTest("test_1", traffic_split=0.5)
    variants = set()
    for i in range(200):
        variants.add(ab.assign_variant(f"user_{i}"))
    assert Variant.CONTROL in variants
    assert Variant.TREATMENT in variants


def test_record_impression() -> None:
    ab = RecommendationABTest("test_1")
    ab.record_impression("user_001")
    result = ab.get_results()
    total = result.control.impressions + result.treatment.impressions
    assert total == 1


def test_record_click() -> None:
    ab = RecommendationABTest("test_1")
    ab.record_impression("user_001")
    ab.record_click("user_001")
    result = ab.get_results()
    total_clicks = result.control.clicks + result.treatment.clicks
    assert total_clicks == 1


def test_ctr_computation() -> None:
    ab = RecommendationABTest("test_1", min_sample_size=1)
    # Record in a controlled way
    for i in range(10):
        uid = f"ctrl_{i}"
        ab.record_impression(uid)
    result = ab.get_results()
    # No clicks → CTR should be 0
    assert result.control.ctr == 0.0 or result.treatment.ctr == 0.0


def test_not_significant_with_few_samples() -> None:
    ab = RecommendationABTest("test_1", min_sample_size=100)
    ab.record_impression("u1")
    ab.record_click("u1")
    result = ab.get_results()
    assert not result.is_significant


def test_significance_detection() -> None:
    ab = RecommendationABTest("significance_test", min_sample_size=10, traffic_split=0.5)
    # Generate enough data with clear difference
    for i in range(500):
        uid = f"user_{i}"
        variant = ab.assign_variant(uid)
        ab.record_impression(uid)
        if variant == Variant.TREATMENT:
            ab.record_click(uid)  # 100% CTR for treatment

    result = ab.get_results()
    # Treatment has much higher CTR → should be significant
    if result.control.impressions >= 10 and result.treatment.impressions >= 10:
        assert result.p_value < 0.05


def test_concluded_property() -> None:
    ab = RecommendationABTest("test_1", min_sample_size=5, traffic_split=0.5)
    for i in range(200):
        uid = f"user_{i}"
        variant = ab.assign_variant(uid)
        ab.record_impression(uid)
        if variant == Variant.TREATMENT:
            ab.record_click(uid)

    result = ab.get_results()
    if result.is_significant:
        assert result.concluded
        assert result.winner is not None


def test_conversion_tracking() -> None:
    ab = RecommendationABTest("test_1")
    ab.record_impression("u1")
    ab.record_conversion("u1")
    result = ab.get_results()
    total_conv = result.control.conversions + result.treatment.conversions
    assert total_conv == 1
