import math

import numpy as np

from src.data.features import CancelAddTracker, mid_price, micro_price, obi


def test_feature_formulas_with_hand_computed_fixture():
    bids = np.array([100.0, 99.5, 99.0])
    asks = np.array([100.5, 101.0, 101.5])
    bid_sizes = np.array([10.0, 20.0, 15.0])
    ask_sizes = np.array([12.0, 8.0, 5.0])

    assert math.isclose(mid_price(bids, asks), 100.25, rel_tol=1e-9)
    assert math.isclose(micro_price(bids, asks, bid_sizes, ask_sizes), 100.22727272727273, rel_tol=1e-9)
    assert math.isclose(obi(bids, asks, bid_sizes, ask_sizes, k=2), 0.2, rel_tol=1e-9)


def test_cancel_add_tracker_rolls_and_ratios():
    tracker = CancelAddTracker(window=3)

    tracker.observe(side="bid", cancel=2.0, add=5.0)
    tracker.observe(side="bid", cancel=1.0, add=3.0)
    tracker.observe(side="bid", cancel=4.0, add=1.0)
    tracker.observe(side="ask", cancel=1.0, add=2.0)

    assert tracker.ratio("bid") == 0.5
    assert tracker.ratio("ask") == 0.5
    assert tracker.window_ratio("bid") == 0.5
    assert tracker.window_ratio("ask") == 0.5
