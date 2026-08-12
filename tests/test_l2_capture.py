from src.data.l2_capture_daemon import L2Book, apply_diff


def test_reconstruct_book_with_resync_on_gap():
    book = L2Book()
    book.apply_snapshot({"bids": [[100.0, 10.0]], "asks": [[101.0, 12.0]]}, seq=1)

    diff = {"u": 2, "pu": 1, "bids": [[100.0, 20.0]], "asks": [[101.0, 8.0]]}
    apply_diff(book, diff)
    assert book.best_bid() == 100.0
    assert book.best_ask() == 101.0

    gap = {"u": 4, "pu": 2, "bids": [[100.5, 6.0]], "asks": [[101.5, 4.0]]}
    apply_diff(book, gap)
    assert book.best_bid() == 100.5
    assert book.best_ask() == 101.5
