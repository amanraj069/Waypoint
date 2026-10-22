"""Start PX4 SITL in a Gazebo world written from a waypoint `World`, and stop it again.

Two processes per flight, each in its own session so they can be stopped as a
group:

* `gz sim -s`, the Gazebo server, on the world `gazebo.scene` writes in its
  PX4 form (gravity, sensors, ground to stand on); and `gz sim -g`, the
  window, when one is asked for;
* PX4 itself, in standalone mode: it waits for that world, spawns its x500
  quadrotor on the launch pad and flies it through its gz_bridge.

PX4 runs in a fresh working directory every time, so no parameter or mission
from one flight leaks into the next. Gazebo is controlled with the `gz` command
line (pause, wind, spawning markers), because Gazebo's Python bindings are
built for a different Python than the project's.
"""

from __future__ import annotations

import os
import queue
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from waypoint.backends import BackendError
from waypoint.gazebo.scene import WORLD_NAME, write_scene
from waypoint.planner.models import MissionConstraints
from waypoint.world.models import World

#: Where PX4 is looked for when $WAYPOINT_PX4_DIR is not set.
DEFAULT_PX4_DIR = Path.home() / "PX4-Autopilot"
PX4_DIR_VAR = "WAYPOINT_PX4_DIR"
#: The airframes this backend flies: PX4's x500, bare or carrying a camera.
#: `x500_depth` has an OAK-D Lite (RGB and depth), `x500_mono_cam` one camera.
MODELS = ("x500", "x500_depth", "x500_mono_cam")
MODEL = "x500"


def model_name(model: str) -> str:
    """PX4 names the first spawned airframe after its model and its instance."""
    return f"{model}_0"
CACHE = Path.home() / ".cache" / "waypoint" / "px4"


@dataclass(frozen=True)
class Px4Install:
    """A PX4-Autopilot checkout with SITL built (`make px4_sitl`)."""

    root: Path

    @property
    def build(self) -> Path:
        return self.root / "build" / "px4_sitl_default"

    @property
    def binary(self) -> Path:
        return self.build / "bin" / "px4"

    @property
    def models(self) -> Path:
        return self.root / "Tools" / "simulation" / "gz" / "models"

    @property
    def plugins(self) -> Path:
        return self.build / "src" / "modules" / "simulation" / "gz_plugins"

    @classmethod
    def find(cls, root: Path | str | None = None) -> "Px4Install":
        """The install at `root`, $WAYPOINT_PX4_DIR or ~/PX4-Autopilot; says what is missing if none."""
        install = cls(Path(root or os.environ.get(PX4_DIR_VAR) or DEFAULT_PX4_DIR).expanduser())
        if not install.binary.exists():
            raise BackendError(
                f"no PX4 SITL build at {install.build}. Clone PX4-Autopilot and run "
                f"`make px4_sitl` in it, or point ${PX4_DIR_VAR} at your checkout."
            )
        for model in MODELS:
            if not (install.models / model / "model.sdf").exists():
                raise BackendError(f"PX4's {model} model is missing from {install.models}")
        if shutil.which("gz") is None:
            raise BackendError("`gz` is not on PATH; install Gazebo (brew install gz-jetty).")
        return install

    def version(self) -> str | None:
        try:
            result = subprocess.run(
                ["git", "describe", "--tags", "--always"], cwd=self.root,
                capture_output=True, text=True, timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip() or None


class Sitl:
    """One PX4 + Gazebo session: the world, the aircraft on its pad, and the processes."""

    def __init__(
        self,
        world: World | None,
        pad: tuple[float, float, float],
        *,
        constraints: MissionConstraints | None = None,
        model: str = MODEL,
        gui: bool = False,
        speed: float = 1.0,
        install: Px4Install | None = None,
        keep: bool = False,
        progress=None,
    ):
        self.world = world or World()
        self.pad = pad
        self.constraints = constraints
        if model not in MODELS:
            raise BackendError(f"the airframe must be one of {', '.join(MODELS)}, not {model!r}")
        self.model = model
        self.name = model_name(model)
        self.gui = gui
        self.speed = speed
        self.install = install or Px4Install.find()
        self.keep = keep
        #: Called with a short line at each startup stage; startup takes 20 to 30 s.
        self.progress = progress or (lambda line: None)
        self.dir: Path | None = None
        self._processes: list[subprocess.Popen] = []
        self._logs: list = []
        self._jobs: queue.Queue = queue.Queue()
        self._worker: threading.Thread | None = None
        self._spawned = 0
        self._window: subprocess.Popen | None = None

    # -- life cycle -------------------------------------------------------------

    def start(self, timeout: float = 45.0, attempts: int = 3) -> None:
        """Write the world, start Gazebo on it, then PX4; returns once both are up.

        A start normally takes about 20 s. Now and then PX4's startup script
        stalls part-way (in a different place each time, with Gazebo's clock
        still running), so a start that has not finished in `timeout` is torn
        down and tried again, up to `attempts` times.
        """
        started = time.monotonic()
        for attempt in range(1, attempts + 1):
            try:
                self._start(timeout)
                self.progress(f"PX4 and Gazebo ready in {time.monotonic() - started:.0f} s")
                return
            except _Stalled as exc:
                if attempt == attempts:
                    raise BackendError(f"{exc} (gave up after {attempts} attempts)") from None
                self.progress(f"PX4's startup stalled; starting again (attempt {attempt + 1} of {attempts})")
            except BaseException:
                # Ctrl-C, or anything else: never leave Gazebo or PX4 running behind us.
                self.stop()
                raise

    def _start(self, timeout: float) -> None:
        CACHE.mkdir(parents=True, exist_ok=True)
        self.dir = Path(tempfile.mkdtemp(prefix=time.strftime("%Y%m%dT%H%M%S-"), dir=CACHE))
        scene_dir, work = self.dir / "scene", self.dir / "px4"
        work.mkdir(parents=True)
        write_scene(self.scene(), scene_dir, px4=True)
        self.progress("starting Gazebo")
        if self._service_up(f"/world/{WORLD_NAME}/scene/info"):
            # PX4 would attach to whichever answers first, and fly in the wrong world.
            self.stop()
            raise BackendError(
                f"another Gazebo is already serving a world called '{WORLD_NAME}' (a replay, or "
                "an earlier flight that was not shut down); close it first, e.g. pkill -f gz-sim-main"
            )

        self._spawn(
            ["gz", "sim", "-s", "-r", "--verbose=1", str(scene_dir / "world.sdf")],
            cwd=scene_dir, log="gz.log",
        )
        self._await(lambda: self._service_up(f"/world/{WORLD_NAME}/scene/info"), timeout, "Gazebo's world")
        if self.gui:
            self._spawn(["gz", "sim", "-g"], cwd=scene_dir, log="gz-gui.log")
            self._window = self._processes[-1]

        x, y, z = self.pad
        self.progress(f"starting PX4 ({self.model}) on the launch pad")
        env = {
            "PX4_GZ_STANDALONE": "1",
            "PX4_GZ_WORLD": WORLD_NAME,
            "PX4_SIM_MODEL": f"gz_{self.model}",
            "PX4_GZ_MODELS": str(self.install.models),
            # A few centimetres up, so the airframe settles onto the pad rather than into it.
            "PX4_GZ_MODEL_POSE": f"{x:.3f},{y:.3f},{z + 0.05:.3f},0,0,0",
        }
        if self.speed != 1.0:
            env["PX4_SIM_SPEED_FACTOR"] = f"{self.speed:g}"
        self._px4_log = self._spawn(
            [str(self.install.binary), "-d", str(self.install.build / "etc"), "-w", str(work)],
            cwd=work, log="px4.log", extra=env,
        )
        self._await(self._px4_started, timeout, "PX4's startup script", retry=True)
        self._worker = threading.Thread(target=self._work, name="gz-commands", daemon=True)
        self._worker.start()
        if self.gui:
            threading.Thread(target=self._follow, name="gz-follow", daemon=True).start()

    def scene(self) -> dict:
        """What the world shows: the place, plus everything the verifier reasoned about.

        The geofence walls, the no-fly volumes and the planned obstacles are
        drawn exactly as the replay draws them, and like there they are glass:
        seen, never collided with. Keeping the aircraft inside them is the
        verifier's and the monitor's job, not a wall's.
        """
        run: dict = {"world": self.world.model_dump(mode="json"), "pads": {"start": list(self.pad)}}
        if self.constraints is not None:
            run["constraints"] = self.constraints.model_dump(mode="json")
            run["obstacles"] = [{**o.model_dump(mode="json"), "appeared_at": 0.0} for o in self.constraints.obstacles]
        return run

    def stop(self) -> None:
        """Stop PX4 and Gazebo, newest first, and tidy up unless asked to keep the files."""
        if self._worker is not None:
            self._jobs.put(None)
            self._worker.join(timeout=5)
            self._worker = None
        for process in reversed(self._processes):
            _stop(process)
        self._processes.clear()
        for handle in self._logs:
            handle.close()
        self._logs.clear()
        if self.dir is not None and not self.keep:
            shutil.rmtree(self.dir, ignore_errors=True)
        self.dir = None

    @property
    def running(self) -> bool:
        return bool(self._processes) and all(p.poll() is None for p in self._processes)

    def camera_topics(self) -> list[str]:
        """The image topics Gazebo is publishing for the aircraft's camera, if it has one."""
        listed = self._run(["gz", "topic", "-l"], timeout=10)
        if listed is None:
            return []
        return [t for t in listed.stdout.split() if t.endswith("/image") and self.name in t or t == "/depth_camera"]

    def wait_for_window(self) -> None:
        """Return once the Gazebo window has been closed (at once if there is none)."""
        if self.gui and self._window is not None:
            self._window.wait()

    def tail(self, name: str = "px4.log", lines: int = 15) -> str:
        """The end of one of the session's logs, for an error message."""
        path = self.dir / name if self.dir else None
        if not path or not path.exists():
            return ""
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])

    # -- controlling the world ----------------------------------------------------

    def pause(self, paused: bool) -> None:
        """Freeze or restart Gazebo; PX4 runs in lockstep with it, so its clock stops too."""
        self._service(
            f"/world/{WORLD_NAME}/control", "gz.msgs.WorldControl",
            f"pause: {'true' if paused else 'false'}",
        )

    def push(self, force: tuple[float, float, float]) -> None:
        """Hold a steady force (N, world frame) on the airframe, replacing the last; zero clears it."""
        entity = f'name: "{self.name}", type: MODEL'
        self._later(["gz", "topic", "-t", f"/world/{WORLD_NAME}/wrench/clear", "-m", "gz.msgs.Entity", "-p", entity])
        if any(abs(c) > 1e-3 for c in force):
            fx, fy, fz = force
            self._later([
                "gz", "topic", "-t", f"/world/{WORLD_NAME}/wrench/persistent", "-m", "gz.msgs.EntityWrench",
                "-p", f"entity: {{{entity}}}, wrench: {{force: {{x: {fx:.3f}, y: {fy:.3f}, z: {fz:.3f}}}}}",
            ])

    def show(self, sdf: str) -> None:
        """Add a model to the scene (a route, an obstacle that appeared); it is only drawn, never hit."""
        if self.dir is None:
            return
        self._spawned += 1
        path = self.dir / f"late_{self._spawned}.sdf"
        path.write_text(sdf)
        self._later([
            "gz", "service", "-s", f"/world/{WORLD_NAME}/create", "--reqtype", "gz.msgs.EntityFactory",
            "--reptype", "gz.msgs.Boolean", "--timeout", "3000",
            "--req", f'sdf_filename: "{path}", allow_renaming: true',
        ])

    # -- internals ------------------------------------------------------------------

    def _env(self, extra: dict | None = None) -> dict:
        env = dict(os.environ)
        # Discovery by multicast, pinned to loopback so a VPN cannot split server from PX4.
        env["GZ_IP"] = "127.0.0.1"
        env["GZ_SIM_RESOURCE_PATH"] = os.pathsep.join(
            filter(None, [str(self.install.models), env.get("GZ_SIM_RESOURCE_PATH")])
        )
        env["GZ_SIM_SYSTEM_PLUGIN_PATH"] = os.pathsep.join(
            filter(None, [str(self.install.plugins), env.get("GZ_SIM_SYSTEM_PLUGIN_PATH")])
        )
        env.pop("GZ_SIM_SERVER_CONFIG_PATH", None)  # the world lists its own systems
        if not self.gui:
            env["HEADLESS"] = "1"
        return {**env, **(extra or {})}

    def _spawn(self, args: list[str], *, cwd: Path, log: str, extra: dict | None = None) -> Path:
        path = self.dir / log
        handle = path.open("w")
        self._logs.append(handle)
        process = subprocess.Popen(
            args, cwd=cwd, env=self._env(extra), stdout=handle, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True,
        )
        self._processes.append(process)
        return path

    def _await(self, ready, timeout: float, what: str, *, retry: bool = False) -> None:
        deadline = time.monotonic() + timeout
        while not ready():
            dead = [p for p in self._processes if p.poll() is not None]
            if dead or time.monotonic() > deadline:
                why = "exited" if dead else f"did not come up within {timeout:g} s"
                detail = self.tail("px4.log") or self.tail("gz.log")
                self.stop()
                error = _Stalled if retry and not dead else BackendError
                raise error(f"{what} {why}" + (f":\n{detail}" if detail else ""))
            time.sleep(0.5)

    def _px4_started(self) -> bool:
        text = self._px4_log.read_text(errors="replace") if self._px4_log.exists() else ""
        if "Startup script returned with return value" in text or "ERROR [init]" in text:
            detail = self.tail("px4.log")
            self.stop()
            raise BackendError(f"PX4 failed to start:\n{detail}")
        return "Startup script returned successfully" in text

    def _service_up(self, name: str) -> bool:
        result = self._run(["gz", "service", "-i", "--service", name], timeout=5)
        return result is not None and "Service providers" in result.stdout

    def _service(self, name: str, reqtype: str, request: str) -> None:
        result = self._run([
            "gz", "service", "-s", name, "--reqtype", reqtype, "--reptype", "gz.msgs.Boolean",
            "--timeout", "3000", "--req", request,
        ], timeout=10)
        if result is None or "true" not in result.stdout:
            raise BackendError(f"Gazebo did not accept {name} ({request})")

    def _run(self, args: list[str], timeout: float) -> subprocess.CompletedProcess | None:
        """Run a `gz` command; None if it fails or outlives `timeout`.

        `gz` is a Ruby wrapper that starts the real tool as a child. On a timeout
        `subprocess.run` kills only the wrapper and then waits for the child to
        close the output pipe, which it may never do; so the whole process group
        is killed instead, and this can never hang.
        """
        try:
            process = subprocess.Popen(
                args, env=self._env(), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL, text=True, start_new_session=True,
            )
        except OSError:
            return None
        try:
            out, err = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_group(process)
            return None
        except BaseException:
            _kill_group(process)
            raise
        return subprocess.CompletedProcess(args, process.returncode, out, err)

    def _later(self, args: list[str]) -> None:
        """Queue a command the flight need not wait for, run in order on the worker."""
        self._jobs.put(args)

    def _follow(self, wait: float = 60.0) -> None:
        """Point the window's camera at the aircraft and keep it there.

        Jetty's GUI tracks through `/gui/track`; PX4's own PX4_GZ_FOLLOW calls
        `/gui/follow`, which this Gazebo no longer has. The window comes up well
        after the server and a new subscriber misses the first few messages, so
        this waits for the topic and then says it a few times.
        """
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline and self._processes:
            listed = self._run(["gz", "topic", "-l"], timeout=5)
            if listed is not None and "/gui/track" in listed.stdout:
                break
            time.sleep(1.0)
        else:
            return
        request = (
            f'track_mode: FOLLOW_LOOK_AT, follow_target: {{name: "{self.name}"}}, '
            f'track_target: {{name: "{self.name}"}}, follow_offset: {{x: -8, y: -8, z: 5}}, '
            "follow_pgain: 0.08, track_pgain: 0.2"
        )
        for _ in range(5):
            if not self._processes:
                return
            self._run(["gz", "topic", "-t", "/gui/track", "-m", "gz.msgs.CameraTrack", "-p", request], timeout=5)
            time.sleep(0.5)

    def _work(self) -> None:
        while (args := self._jobs.get()) is not None:
            self._run(args, timeout=10)


class _Stalled(BackendError):
    """PX4's startup ran out of time with every process still alive: worth another try."""


def _kill_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        process.communicate(timeout=2)
    except (subprocess.TimeoutExpired, ValueError, OSError):
        pass


def _stop(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=5)
    except (ProcessLookupError, subprocess.TimeoutExpired, PermissionError):
        try:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
        except (ProcessLookupError, subprocess.TimeoutExpired, PermissionError):
            pass
