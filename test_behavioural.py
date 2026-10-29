
"""
Automated Unit Tests for Behavioural Fraud Detection Engine.

These tests check whether the fraud detection engine behaves as expected
when customers make normal or potentially suspicious transactions.
It covers all four behavioural anomaly detectors:
1. Unusual Location Changes
2. Suddenly Large Transactions
3. Unusual Transaction Timing
4. Frequent Back-to-Back Bursts

"""
import pytest
from datetime import datetime
from ml.behavioural_engine import BehaviouralEngine, haversine_distance

@pytest.fixture
def engine():
    # Create a fresh engine for each test to keep results independent.
    return BehaviouralEngine()

def test_haversine_distance():
    # Check the approximate distance between Raleigh and New York.
    dist = haversine_distance(35.7796, -78.6382, 40.7128, -74.0060)
    assert 650.0 <= dist <= 750.0

    # The distance should be zero when both locations are identical.
    assert haversine_distance(35.0, -78.0, 35.0, -78.0) == 0.0

    # Invalid coordinates should not crash the fraud detection system.
    assert haversine_distance("bad", None, 35.0, -78.0) == 0.0

def test_location_anomaly_implausible_speed(engine):
    # A customer cannot realistically travel across the country in 30 minutes.
    curr_time = 1735689600
    last_time = curr_time - 1800

    res = engine.check_location_anomaly(
        current_merch_lat=34.0522, current_merch_long=-118.2437,
        customer_home_lat=35.7796, customer_home_long=-78.6382,
        last_tx_merch_lat=35.7796, last_tx_merch_long=-78.6382,
        current_unix_time=curr_time, last_unix_time=last_time
    )

    # Such an extreme change in location should be treated as critical.
    assert res["flag"] is True
    assert res["severity"] == "CRITICAL"
    assert res["score"] >= 90.0
    assert "implausible" in res["explanation"].lower()

def test_location_anomaly_missing_coordinates(engine):
    # Without the current merchant's location, a reliable comparison is impossible.
    res = engine.check_location_anomaly(
        current_merch_lat=None, current_merch_long=None,
        customer_home_lat=35.0, customer_home_long=-78.0
    )

    # Missing information alone should not trigger a location anomaly.
    assert res["flag"] is False
    assert res["severity"] == "NONE"

def test_sudden_large_amount_with_history(engine):
    # A customer's previous purchases help establish their normal spending range.
    history = [25.0, 30.0, 22.0, 45.0, 28.0, 35.0, 40.0]

    # This purchase is much larger than the customer's usual transactions.
    res = engine.check_sudden_large_amount(current_amt=2500.0, historical_amts=history)

    assert res["flag"] is True
    assert res["severity"] in ["CRITICAL", "HIGH"]
    assert res["score"] >= 75.0
    assert res["details"]["z_score"] > 3.0

def test_sudden_large_amount_normal(engine):
    # A small change from the customer's usual spending should be acceptable.
    history = [40.0, 50.0, 45.0, 60.0, 55.0]
    res = engine.check_sudden_large_amount(current_amt=52.0, historical_amts=history)

    assert res["flag"] is False
    assert res["severity"] in ["NONE", "LOW"]

def test_sudden_large_amount_cold_start(engine):
    # New customers have no past transactions to establish a spending baseline.
    res = engine.check_sudden_large_amount(current_amt=1800.0, historical_amts=[])

    # A large first transaction should receive extra attention.
    assert res["flag"] is True
    assert res["severity"] == "HIGH"

def test_unusual_timing_late_night(engine):
    # This customer usually shops during the day, not in the early morning.
    dt = datetime(2025, 1, 1, 3, 15, 0)
    hist_hours = [9, 10, 11, 12, 14, 15, 17, 18, 19, 20]

    res = engine.check_unusual_timing(dt, hist_hours)

    # The unusual hour should be flagged for further review.
    assert res["flag"] is True
    assert res["severity"] in ["HIGH", "MEDIUM"]
    assert "late-night" in res["explanation"].lower()

def test_unusual_timing_normal_daytime(engine):
    # Afternoon shopping matches this customer's usual behaviour.
    dt = datetime(2025, 1, 1, 14, 30, 0)
    hist_hours = [10, 12, 14, 15, 16]

    res = engine.check_unusual_timing(dt, hist_hours)

    assert res["flag"] is False
    assert res["severity"] == "NONE"

def test_burst_transactions_rapid_succession(engine):
    # Several transactions within a few minutes can indicate suspicious activity.
    now = 1735689600
    hist_timestamps = [now - 60, now - 120, now - 180, now - 240]

    res = engine.check_burst_transactions(now, hist_timestamps)

    # Confirm that the engine recognises the rapid sequence of transactions.
    assert res["flag"] is True
    assert res["severity"] in ["CRITICAL", "HIGH"]
    assert res["details"]["count_10m"] >= 4

def test_burst_transactions_isolated(engine):
    # A transaction from three days ago should not count as a recent burst.
    now = 1735689600
    hist_timestamps = [now - 259200]

    res = engine.check_burst_transactions(now, hist_timestamps)

    assert res["flag"] is False
    assert res["severity"] == "NONE"
# Overall, these tests cover normal activity and common suspicious patterns.
# They also help catch regressions when the behavioural detection rules change.
# Run the suite with pytest to check whether the engine passes these scenarios.
# Remember that these tests verify expected behaviour, not real-world fraud accuracy.
