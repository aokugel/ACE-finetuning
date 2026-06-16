#!/usr/bin/env python
"""Patch prometheus-fastapi-instrumentator for Starlette 1.3.x compatibility.

vLLM 0.23 pulls Starlette 1.3.1 + FastAPI 0.137, whose `app.routes` can contain
`_IncludedRouter` objects that have neither `.matches()` nor `.path`. The
instrumentator's `_get_route_name` (v8.0.0) assumes every route has both and
crashes with `AttributeError: '_IncludedRouter' object has no attribute 'path'`
on EVERY request, taking down the OpenAI server.

This idempotently makes `_get_route_name` skip routes without `.matches` and use
`getattr(route, "path", None)`. Run once after installing vLLM.
"""
import prometheus_fastapi_instrumentator as p
from pathlib import Path

f = Path(p.__file__).parent / "routing.py"
src = f.read_text()

GUARD = "        if not hasattr(route, \"matches\"):\n            continue\n"
if GUARD in src:
    print("Already patched.")
else:
    # add a guard as the first statement inside `for route in routes:`
    needle = "    for route in routes:\n        match, child_scope = route.matches(scope)\n"
    if needle not in src:
        raise SystemExit("instrumentator layout changed; inspect routing.py manually")
    repl = "    for route in routes:\n" + GUARD + "        match, child_scope = route.matches(scope)\n"
    src = src.replace(needle, repl, 1)
    # harden both `.path` accesses
    src = src.replace("route_name = route.path", "route_name = getattr(route, \"path\", None)")
    f.write_text(src)
    print(f"Patched {f}")
