from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.l2_capture_daemon import L2CaptureConfig, L2CaptureDaemon


def _configure_logging(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)s %(name)s %(message)s"
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not root.handlers:
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setFormatter(logging.Formatter(fmt))
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter(fmt))
        root.addHandler(fh)
        root.addHandler(sh)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run live Binance L2 capture daemon")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--out-dir", default="data/raw_l2", dest="out_dir")
    p.add_argument("--top-levels", type=int, default=10, dest="top_levels")
    p.add_argument("--shard-rows", type=int, default=2000, dest="shard_rows")
    p.add_argument("--flush-interval-s", type=float, default=10.0, dest="flush_interval_s")
    p.add_argument("--log-file", default="logs/l2_capture.log", dest="log_file")
    return p.parse_args(argv)


async def _run(args: argparse.Namespace) -> None:
    _configure_logging(Path(args.log_file))
    daemon = L2CaptureDaemon(
        L2CaptureConfig(
            symbol=args.symbol,
            out_dir=Path(args.out_dir),
            top_levels=args.top_levels,
            shard_rows=args.shard_rows,
            flush_interval_s=args.flush_interval_s,
        )
    )
    await daemon.run_forever()


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    stop_event = asyncio.Event()

    def _handle_signal(signum, frame):  # noqa: ANN001
        logging.getLogger("l2_capture").warning("Signal %s received; shutting down", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    task = loop.create_task(_run(args))

    async def _wait_for_stop() -> None:
        await stop_event.wait()
        task.cancel()

    stopper = loop.create_task(_wait_for_stop())
    try:
        loop.run_until_complete(asyncio.gather(task, stopper))
    except (asyncio.CancelledError, KeyboardInterrupt):
        pass
    finally:
        pending = asyncio.all_tasks(loop)
        for item in pending:
            item.cancel()
        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
