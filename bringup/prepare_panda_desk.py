#!/usr/bin/env python3
"""Prepare a Franka Panda through Desk before starting HIL-SERL.

This mirrors the Desk auto-setup flow used by panda_data_collector:
take control, open brakes, and activate FCI.
"""

from __future__ import annotations

import argparse
import os
import ssl

_PATCHED = False


def _install_patches(host: str) -> None:
    """Patch panda_py websocket setup for older Panda Desk installations."""
    global _PATCHED

    for env_name in ("NO_PROXY", "no_proxy"):
        current = [value for value in os.environ.get(env_name, "").split(",") if value]
        for candidate in (host, "robot.franka.de"):
            if candidate not in current:
                current.append(candidate)
        os.environ[env_name] = ",".join(current)

    if _PATCHED:
        return

    try:
        import panda_py
    except ImportError as exc:
        raise RuntimeError(
            "panda_py is required for Desk auto-setup. "
            "Set AUTO_SETUP_PYTHON to a Python interpreter that has panda_py installed."
        ) from exc

    original_connect = panda_py.connect

    def _patched_connect(uri, **kwargs):
        kwargs.pop("ssl_context", None)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        kwargs["ssl"] = context
        kwargs["proxy"] = None
        return original_connect(uri, **kwargs)

    panda_py.connect = _patched_connect
    _PATCHED = True


def connect_desk(host: str, user: str, password: str):
    _install_patches(host)
    import panda_py

    return panda_py.Desk(host, user, password)


def prepare_robot(desk, force_take: bool = True, log=print) -> None:
    log("[desk] Taking control...")
    if not desk.take_control(force=False):
        if not force_take:
            raise RuntimeError(
                "Desk control is already taken. Close the other Desk session "
                "or rerun with force-take enabled."
            )
        log(
            "[desk] Control already taken. Confirm takeover with the Panda "
            "Pilot circle button (Pilot Mode must be Desk)..."
        )
        desk.take_control(force=True)

    log("[desk] Opening brakes...")
    desk.unlock()
    log("[desk] Activating FCI...")
    desk.activate_fci()
    log("[desk] Robot is ready.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--desk-host", required=True)
    parser.add_argument("--desk-user", required=True)
    parser.add_argument("--desk-pass", required=True)
    parser.add_argument(
        "--no-force-take",
        action="store_true",
        help="Do not force takeover if another Desk session already holds control.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    desk = connect_desk(args.desk_host, args.desk_user, args.desk_pass)
    prepare_robot(desk, force_take=not args.no_force_take)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
