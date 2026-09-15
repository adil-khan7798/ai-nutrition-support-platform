"""Unit tests for FoodService.aggregate_intake (and its search/lookup helpers).

These are pure-logic tests: no FastAPI app, no HTTP. They run against a tiny
fixture CSV loaded via the `food_service` fixture in conftest.py.
"""

import pytest

# All eight canonical nutrient fields, with per-100g values for the two fixture
# foods, so expected scaled totals can be asserted exactly.
APPLE_PER_100G = {
    "protein_g": 0.3,
    "carbs_g": 14.0,
    "sugar_g": 10.0,
    "fiber_g": 2.4,
    "total_fat_g": 0.2,
    "saturated_fat_g": 0.03,
    "cholesterol_mg": 0.0,
    "sodium_mg": 1.0,
}
BEEF_PER_100G = {
    "protein_g": 26.0,
    "carbs_g": 0.0,
    "sugar_g": 0.0,
    "fiber_g": 0.0,
    "total_fat_g": 20.0,
    "saturated_fat_g": 8.0,
    "cholesterol_mg": 90.0,
    "sodium_mg": 75.0,
}


def scaled(per_100g, grams):
    """Expected per-item scaled nutrients: value * grams/100, rounded to 4dp."""
    factor = grams / 100.0
    return {k: round(v * factor, 4) for k, v in per_100g.items()}


def test_single_item_100g_equals_per_100g(food_service):
    """100g of a food yields exactly its per-100g values."""
    result = food_service.aggregate_intake([{"food_name": "Apple raw", "grams": 100}])
    assert result["totals"] == pytest.approx(APPLE_PER_100G)
    assert result["unmatched"] == []
    assert len(result["resolved"]) == 1
    assert result["resolved"][0]["matched_description"] == "Apple raw"


def test_grams_scaling_is_linear(food_service):
    """200g scales every per-100g nutrient by exactly 2x."""
    result = food_service.aggregate_intake([{"food_name": "Apple raw", "grams": 200}])
    expected = scaled(APPLE_PER_100G, 200)
    assert result["totals"] == pytest.approx(expected)
    assert result["resolved"][0]["scaled_nutrients"] == pytest.approx(expected)
    assert result["resolved"][0]["grams"] == 200


def test_fractional_grams_scaling(food_service):
    """Sub-100g portions scale down proportionally."""
    result = food_service.aggregate_intake(
        [{"food_name": "Beef patty cooked", "grams": 50}]
    )
    assert result["totals"] == pytest.approx(scaled(BEEF_PER_100G, 50))


def test_multiple_items_totals_are_summed(food_service):
    """Totals are the field-wise sum across all resolved items."""
    result = food_service.aggregate_intake(
        [
            {"food_name": "Apple raw", "grams": 150},
            {"food_name": "Beef patty cooked", "grams": 200},
        ]
    )
    apple = scaled(APPLE_PER_100G, 150)
    beef = scaled(BEEF_PER_100G, 200)
    expected = {k: round(apple[k] + beef[k], 4) for k in APPLE_PER_100G}
    assert result["totals"] == pytest.approx(expected)
    assert len(result["resolved"]) == 2
    assert result["unmatched"] == []


def test_unmatched_food_is_reported_not_aggregated(food_service):
    """A food name with no match lands in `unmatched` and contributes nothing."""
    result = food_service.aggregate_intake(
        [{"food_name": "zzz nonexistent food", "grams": 100}]
    )
    assert result["unmatched"] == ["zzz nonexistent food"]
    assert result["resolved"] == []
    assert result["totals"] == {k: 0.0 for k in APPLE_PER_100G}


def test_mixed_matched_and_unmatched(food_service):
    """Matched items aggregate while unmatched are collected separately."""
    result = food_service.aggregate_intake(
        [
            {"food_name": "Apple raw", "grams": 100},
            {"food_name": "zzz nonexistent food", "grams": 500},
        ]
    )
    assert result["unmatched"] == ["zzz nonexistent food"]
    assert len(result["resolved"]) == 1
    assert result["totals"] == pytest.approx(APPLE_PER_100G)


def test_substring_fallback_matches_partial_name(food_service):
    """A non-exact name falls back to best substring match (e.g. 'apple')."""
    result = food_service.aggregate_intake([{"food_name": "apple", "grams": 100}])
    assert result["unmatched"] == []
    assert result["resolved"][0]["submitted_name"] == "apple"
    assert result["resolved"][0]["matched_description"] == "Apple raw"


def test_lookup_is_case_insensitive(food_service):
    """Exact lookup ignores case."""
    result = food_service.aggregate_intake([{"food_name": "APPLE RAW", "grams": 100}])
    assert result["resolved"][0]["matched_description"] == "Apple raw"
    assert result["unmatched"] == []


def test_empty_item_list_yields_zero_totals(food_service):
    """No items -> zeroed totals, nothing resolved or unmatched."""
    result = food_service.aggregate_intake([])
    assert result["totals"] == {k: 0.0 for k in APPLE_PER_100G}
    assert result["resolved"] == []
    assert result["unmatched"] == []


def test_totals_cover_all_nutrient_fields(food_service):
    """Totals always contain exactly the canonical nutrient field set."""
    result = food_service.aggregate_intake([{"food_name": "Apple raw", "grams": 100}])
    assert set(result["totals"].keys()) == set(APPLE_PER_100G.keys())


# --- search_foods / get_food helpers that aggregate_intake relies on ---


def test_search_blank_query_returns_empty(food_service):
    assert food_service.search_foods("   ") == []


def test_search_finds_substring(food_service):
    hits = food_service.search_foods("beef")
    assert len(hits) == 1
    assert hits[0]["description"] == "Beef patty cooked"


def test_get_food_unknown_returns_none(food_service):
    assert food_service.get_food("not a food") is None
