"""Example experiment helpers."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_PACKAGE_ROOTS = (
    REPO_ROOT / "serl_launcher",
    REPO_ROOT / "serl_robot_infra",
)

for package_root in reversed(LOCAL_PACKAGE_ROOTS):
    package_root_str = str(package_root)
    if package_root.is_dir() and package_root_str not in sys.path:
        sys.path.insert(0, package_root_str)
