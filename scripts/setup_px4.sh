#!/usr/bin/env bash
# Build PX4 SITL against the installed Gazebo, for `--backend px4`.
#
#   scripts/setup_px4.sh            # clone (if needed), patch, build, install pymavlink here
#   scripts/setup_px4.sh --check    # only report what is there and what is missing
#
# Safe to run again: every step checks first and skips what is already done.
# Settings: WAYPOINT_PX4_DIR (default ~/PX4-Autopilot), PX4_VERSION (default v1.17.0).
#
# What it does not do: install Gazebo (brew tap osrf/simulation && brew install
# gz-jetty, or gz-harmonic), or OpenCV. PX4 only needs OpenCV for its
# optical-flow plugin, which the patches below skip, and installing it with
# Homebrew can upgrade protobuf ahead of the Gazebo bottles and break Gazebo.

set -euo pipefail

PX4_DIR="${WAYPOINT_PX4_DIR:-$HOME/PX4-Autopilot}"
PX4_VERSION="${PX4_VERSION:-v1.17.0}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCHES="$ROOT/scripts/px4-patches"
CHECK_ONLY=0
[[ "${1:-}" == "--check" ]] && CHECK_ONLY=1

say() { printf '\n==> %s\n' "$*"; }
ok() { printf '    ok: %s\n' "$*"; }
todo() { printf '    missing: %s\n' "$*"; }
die() { printf '\nerror: %s\n' "$*" >&2; exit 1; }

say "Tools"
command -v git >/dev/null || die "git is not installed"
if command -v gz >/dev/null; then
  ok "gz sim $(gz sim --versions 2>/dev/null | head -n 1)"
else
  die "Gazebo is not installed: brew tap osrf/simulation && brew install gz-jetty"
fi
for tool in cmake ninja ccache; do
  if command -v "$tool" >/dev/null; then
    ok "$tool"
  elif (( CHECK_ONLY )); then
    todo "$tool"
  else
    command -v brew >/dev/null || die "$tool is missing and Homebrew is not installed to get it"
    brew install "$tool"
  fi
done

say "PX4 $PX4_VERSION in $PX4_DIR"
if [[ -d "$PX4_DIR/.git" ]]; then
  ok "checkout exists ($(git -C "$PX4_DIR" describe --tags --always 2>/dev/null))"
elif (( CHECK_ONLY )); then
  todo "no checkout"
else
  git clone --recursive --shallow-submodules --depth 1 --branch "$PX4_VERSION" \
    https://github.com/PX4/PX4-Autopilot.git "$PX4_DIR"
fi

say "PX4's Python tools (a venv inside the checkout, so Homebrew's Python is left alone)"
if [[ -x "$PX4_DIR/.venv/bin/python" ]] && "$PX4_DIR/.venv/bin/python" -c "import em, kconfiglib, jinja2" 2>/dev/null; then
  ok "$PX4_DIR/.venv"
elif (( CHECK_ONLY )); then
  todo "$PX4_DIR/.venv"
else
  python3 -m venv "$PX4_DIR/.venv"
  "$PX4_DIR/.venv/bin/pip" install -q --upgrade pip
  "$PX4_DIR/.venv/bin/pip" install -q -r "$PX4_DIR/Tools/setup/requirements.txt"
fi

say "Patches for current Homebrew protobuf, abseil and OpenCV ($PATCHES)"
for patch in "$PATCHES"/*.patch; do
  name="$(basename "$patch")"
  if [[ ! -d "$PX4_DIR/.git" ]]; then
    todo "$name (no checkout yet)"
  elif git -C "$PX4_DIR" apply --reverse --check "$patch" 2>/dev/null; then
    ok "$name (applied)"
  elif git -C "$PX4_DIR" apply --check "$patch" 2>/dev/null; then
    if (( CHECK_ONLY )); then todo "$name"; else git -C "$PX4_DIR" apply "$patch" && ok "$name (applied now)"; fi
  else
    die "$name does not apply to this checkout; PX4 has moved on from $PX4_VERSION, see README section 5c"
  fi
done

say "make px4_sitl"
if (( CHECK_ONLY )); then
  [[ -x "$PX4_DIR/build/px4_sitl_default/bin/px4" ]] && ok "built" || todo "not built"
else
  # Incremental: quick when nothing changed.
  (cd "$PX4_DIR" && PATH="$PX4_DIR/.venv/bin:$PATH" make px4_sitl) \
    || die "the build failed; if it stopped on Gazebo version detection, see README section 5c"
  ok "built"
fi

say "pymavlink in this project ($ROOT/.venv)"
if "$ROOT/.venv/bin/python" -c "import pymavlink" 2>/dev/null; then
  ok "installed"
elif (( CHECK_ONLY )); then
  todo "pymavlink"
elif command -v uv >/dev/null; then
  (cd "$ROOT" && uv pip install --python .venv/bin/python -e ".[px4]")
else
  (cd "$ROOT" && .venv/bin/pip install -e ".[px4]")
fi

say "Can waypoint find it?"
if WAYPOINT_PX4_DIR="$PX4_DIR" "$ROOT/.venv/bin/python" -c "from waypoint.backends import make_backend; make_backend('px4')" 2>/dev/null; then
  ok "make_backend('px4') works"
  printf '\nNext: %s\n' "python scripts/px4_check.py   (about 3 minutes; add --gui to watch)"
else
  todo "make_backend('px4') fails; run without --check, or see README section 5c"
  (( CHECK_ONLY )) || exit 1
fi
