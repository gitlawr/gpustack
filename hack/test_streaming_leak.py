#!/usr/bin/env python3
"""
Reproduction and verification script for memory leak in SSE streaming subscriptions.

This script:
1. Opens many SSE watch connections to the server
2. Abruptly closes them (simulating proxy-buffered disconnect)
3. Checks the /debug/event-bus endpoint to verify subscriber cleanup
4. Reports memory usage of the server process

Usage:
    # Against a running GPUStack server:
    python hack/test_streaming_leak.py --base-url http://localhost --token <api_key>

    # With custom parameters:
    python hack/test_streaming_leak.py --base-url http://localhost --token <api_key> \
        --connections 50 --rounds 5 --hold-seconds 3

Prerequisites:
    pip install httpx aiohttp
"""

import argparse
import asyncio
import json
import logging
import sys

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Endpoints that support ?watch=true
WATCH_ENDPOINTS = [
    "/v2/models",
    "/v2/model-instances",
    "/v2/model-routes",
    "/v2/model-route-targets",
    "/v2/workers",
]


async def get_event_bus_stats(client: httpx.AsyncClient) -> dict:
    """Fetch EventBus subscriber stats from the debug endpoint."""
    try:
        resp = await client.get("/debug/event-bus")
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.error(f"Failed to get event-bus stats: {e}")
        return {}


def summarize_stats(stats: dict) -> dict:
    """Summarize EventBus stats into a simple dict of topic -> subscriber_count."""
    summary = {}
    for topic, info in stats.items():
        summary[topic] = info.get("subscriber_count", 0)
    return summary


async def open_and_abandon_connections(
    base_url: str,
    token: str,
    num_connections: int,
    hold_seconds: float,
    endpoints: list[str],
):
    """
    Open SSE watch connections and close them abruptly after hold_seconds.
    This simulates what happens when a client disconnects behind a proxy.
    """
    tasks = []

    async def connect_and_abandon(endpoint: str, conn_id: int):
        url = f"{base_url}{endpoint}"
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(connect=5, read=None, write=5, pool=5),
                verify=False,
            ) as client:
                headers = {"Authorization": f"Bearer {token}"}
                async with client.stream(
                    "GET", url, params={"watch": "true"}, headers=headers
                ) as resp:
                    # Read a few events to confirm subscription is active
                    event_count = 0
                    async for line in resp.aiter_lines():
                        event_count += 1
                        if event_count >= 2:
                            break
                    # Hold the connection open
                    await asyncio.sleep(hold_seconds)
                    # Connection closes when exiting context manager
        except Exception as e:
            logger.debug(f"Connection {conn_id} to {endpoint}: {e}")

    for i in range(num_connections):
        endpoint = endpoints[i % len(endpoints)]
        tasks.append(connect_and_abandon(endpoint, i))

    logger.info(
        f"Opening {num_connections} watch connections (hold {hold_seconds}s)..."
    )
    await asyncio.gather(*tasks, return_exceptions=True)
    logger.info(f"All {num_connections} connections closed.")


async def main():
    parser = argparse.ArgumentParser(
        description="Test SSE streaming subscriber leak in GPUStack"
    )
    parser.add_argument(
        "--base-url",
        required=True,
        help="GPUStack server base URL (e.g., http://localhost)",
    )
    parser.add_argument(
        "--token",
        required=True,
        help="API key for authentication",
    )
    parser.add_argument(
        "--connections",
        type=int,
        default=20,
        help="Number of watch connections per round (default: 20)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Number of connect/disconnect rounds (default: 3)",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=2.0,
        help="Seconds to hold each connection before closing (default: 2.0)",
    )
    parser.add_argument(
        "--wait-after",
        type=float,
        default=5.0,
        help="Seconds to wait after closing connections before checking stats (default: 5.0)",
    )
    args = parser.parse_args()

    # Client for debug API
    debug_client = httpx.AsyncClient(
        base_url=args.base_url,
        headers={"Authorization": f"Bearer {args.token}"},
        verify=False,
    )

    try:
        # Baseline stats
        logger.info("=== Baseline EventBus stats ===")
        baseline = await get_event_bus_stats(debug_client)
        baseline_summary = summarize_stats(baseline)
        logger.info(f"Subscribers: {json.dumps(baseline_summary, indent=2)}")

        for round_num in range(1, args.rounds + 1):
            logger.info(f"\n=== Round {round_num}/{args.rounds} ===")

            # Open and close connections
            await open_and_abandon_connections(
                base_url=args.base_url,
                token=args.token,
                num_connections=args.connections,
                hold_seconds=args.hold_seconds,
                endpoints=WATCH_ENDPOINTS,
            )

            # Wait for cleanup
            logger.info(f"Waiting {args.wait_after}s for subscriber cleanup...")
            await asyncio.sleep(args.wait_after)

            # Check stats
            stats = await get_event_bus_stats(debug_client)
            summary = summarize_stats(stats)
            logger.info(
                f"Subscribers after round {round_num}: {json.dumps(summary, indent=2)}"
            )

            # Check for leaks
            leaked = False
            for topic, count in summary.items():
                baseline_count = baseline_summary.get(topic, 0)
                if count > baseline_count:
                    leaked = True
                    logger.warning(
                        f"  LEAK: topic '{topic}' has {count} subscribers "
                        f"(baseline: {baseline_count}, leaked: {count - baseline_count})"
                    )

            if not leaked:
                logger.info("  OK: No subscriber leaks detected.")

        # Final summary
        logger.info("\n=== Final check ===")
        final_stats = await get_event_bus_stats(debug_client)
        final_summary = summarize_stats(final_stats)
        logger.info(f"Final subscribers: {json.dumps(final_summary, indent=2)}")

        total_leaked = 0
        for topic, count in final_summary.items():
            baseline_count = baseline_summary.get(topic, 0)
            if count > baseline_count:
                total_leaked += count - baseline_count

        if total_leaked > 0:
            logger.error(
                f"FAIL: {total_leaked} subscriber(s) leaked across all topics "
                f"after {args.rounds} rounds x {args.connections} connections."
            )
            # Print detailed info for leaked topics
            for topic, info in final_stats.items():
                baseline_count = baseline_summary.get(topic, 0)
                if info.get("subscriber_count", 0) > baseline_count:
                    logger.error(f"  Topic '{topic}' details:")
                    for sub in info.get("subscribers", []):
                        logger.error(
                            f"    subscriber={sub['id']} "
                            f"queue_size={sub['queue_size']} "
                            f"latest_by_key_size={sub['latest_by_key_size']}"
                        )
            sys.exit(1)
        else:
            logger.info(
                f"PASS: All subscribers properly cleaned up after "
                f"{args.rounds} rounds x {args.connections} connections."
            )
    finally:
        await debug_client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
