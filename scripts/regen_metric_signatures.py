#!/usr/bin/env python3
"""Regenerate tests/fixtures/metric_signatures.json.

Run this after intentionally changing a metric's logic and bumping its
`version` class attribute (or after editing its judge prompt template).
The drift test (tests/unit/metrics/test_metric_signatures.py) compares
the current state against this fixture and fails on any unintended drift.

Usage:
    python scripts/regen_metric_signatures.py
"""

import json
import sys
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType

from typing_extensions import TypedDict


def _stub_heavy_imports() -> None:
    """Install a meta-path finder that stubs heavy packages before any eva imports.

    litellm and pipecat together account for ~4 s of import time, but
    regen_metric_signatures only needs class definitions and source hashes —
    it never calls into these libraries at runtime.

    Using a MetaPathFinder means every submodule import (no matter how deep)
    is automatically intercepted and given a lightweight stub, with no manual
    per-submodule registration required.

    The one exception is DeploymentTypedDict, which is used as a base class in
    eva.models.config and must be an actual type, not a stub object.
    """
    _STUB_PACKAGES = frozenset({"litellm"})

    class _AutoStub(ModuleType):
        """A module stub that satisfies arbitrary attribute and submodule access."""

        def __getattr__(self, name: str) -> "_AutoStub":
            child = _AutoStub(f"{self.__name__}.{name}")
            object.__setattr__(self, name, child)
            sys.modules[child.__name__] = child
            return child

        def __call__(self, *args: object, **kwargs: object) -> "_AutoStub":
            return self

        def __iter__(self):  # type: ignore[override]
            return iter([])

        # Allow use in type union expressions at module level, e.g. `Router | None`
        def __or__(self, other: object) -> object:
            return object

        def __ror__(self, other: object) -> object:
            return object

    class _StubLoader(Loader):
        def create_module(self, spec: ModuleSpec) -> _AutoStub:
            return _AutoStub(spec.name)

        def exec_module(self, module: ModuleType) -> None:
            pass  # _AutoStub handles everything via __getattr__

    class _StubFinder(MetaPathFinder):
        def find_spec(self, fullname: str, path: object, target: object = None) -> ModuleSpec | None:
            if fullname.split(".")[0] in _STUB_PACKAGES:
                return ModuleSpec(fullname, _StubLoader())
            return None

    sys.meta_path.insert(0, _StubFinder())

    # DeploymentTypedDict is inherited by ModelDeployment in eva.models.config,
    # so it must be a real class. Trigger the import so the stub is registered in
    # sys.modules, then replace the attribute with a proper TypedDict.
    import litellm.types.router  # noqa: PLC0415, F401

    class DeploymentTypedDict(TypedDict, total=False):
        pass

    sys.modules["litellm.types.router"].DeploymentTypedDict = DeploymentTypedDict  # type: ignore[attr-defined]


_stub_heavy_imports()

from eva.metrics.signatures import compute_all_metric_signatures  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "metric_signatures.json"


def main() -> None:
    signatures = compute_all_metric_signatures()
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(json.dumps(signatures, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {len(signatures)} metric signatures to {FIXTURE_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
