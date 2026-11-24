"""Interactive slider control of the simulated Panda, with the kinematics shown live.

    python scripts/slider_control.py            # opens http://localhost:8765 in your browser
    python scripts/slider_control.py --seed 4   # another cube and target position
    python scripts/slider_control.py --no-browser --port 9000

Move the gripper with Cartesian sliders (your own IK solves the joints) or drive the seven
joints directly. Beside the 3D view: the modified-DH table with live angles, every link
transform, the product down to the tool checked against MuJoCo, the Jacobian, each IK solve
iteration by iteration, and the closed-form elbow equation. See src/ari/interactive.py.

Physics and rendering run on the main thread (macOS requires that for OpenGL); the small web
server runs in a background thread and only exchanges commands, state and images with it.
Nothing is sent anywhere but your own browser: the server listens on localhost only.
"""

from __future__ import annotations

import argparse
import io
import json
import queue
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ari.interactive import SliderSession  # isort: skip  (ari first: it picks the renderer backend)

import mujoco
import numpy as np
from PIL import Image

UI = Path(__file__).resolve().parents[1] / "src" / "ari" / "interactive_ui.html"


class Shared:
    """What the server thread and the main loop exchange (all guarded by `lock`)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.commands: queue.Queue = queue.Queue()
        self.state_json = b"{}"
        self.frame_request = threading.Condition(self.lock)
        self.camera = {"az": 145.0, "el": -25.0, "dist": 1.6, "lx": 0.45, "ly": 0.0, "lz": 0.2,
                       "frames": 1, "target": 1, "sel": -1}
        self.frame_wanted = False
        self.frame = b""
        self.frame_id = 0


def make_handler(shared: Shared):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):          # keep the terminal quiet
            pass

        def _send(self, body: bytes, ctype: str, code: int = 200):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            if url.path in ("/", "/index.html"):
                self._send(UI.read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/state":
                with shared.lock:
                    body = shared.state_json
                self._send(body, "application/json")
            elif url.path == "/frame":
                params = {k: v[0] for k, v in parse_qs(url.query).items()}
                with shared.frame_request:
                    for k in shared.camera:
                        if k in params:
                            shared.camera[k] = float(params[k])
                    last = shared.frame_id
                    shared.frame_wanted = True
                    shared.frame_request.wait_for(lambda: shared.frame_id != last, timeout=2.0)
                    body = shared.frame
                self._send(body, "image/jpeg")
            elif url.path == "/favicon.ico":
                self._send(b"", "image/x-icon", 204)
            else:
                self._send(b"not found", "text/plain", 404)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            try:
                cmd = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                self._send(b'{"ok": false}', "application/json", 400)
                return
            shared.commands.put(cmd)
            self._send(b'{"ok": true}', "application/json")

    return Handler


def apply(session: SliderSession, cmd: dict) -> None:
    kind = cmd.get("type")
    if kind == "cartesian":
        mm = lambda k: None if cmd.get(k) is None else float(cmd[k]) / 1000.0   # noqa: E731
        session.set_cartesian(x=mm("x"), y=mm("y"), z=mm("z"),
                              yaw=None if cmd.get("yaw") is None else np.radians(float(cmd["yaw"])),
                              gripper=cmd.get("gripper"), swivel=cmd.get("swivel"),
                              free_swivel=bool(cmd.get("free_swivel", False)))
    elif kind == "joint":
        q = cmd.get("q")
        session.set_joints(q=None if q is None else np.radians(np.asarray(q, dtype=float)), gripper=cmd.get("gripper"))
    elif kind == "gripper":
        session.gripper = float(np.clip(cmd.get("value", 255), 0, 255))
    elif kind == "reset":
        session.reset(int(cmd.get("seed", 0)))


def add_axes(scene, T: np.ndarray, length: float, width: float, alpha: float = 1.0) -> None:
    """An RGB triad (x red, y green, z blue) at pose T, drawn as arrows."""
    for k, rgb in enumerate(((0.86, 0.2, 0.2), (0.2, 0.7, 0.3), (0.2, 0.45, 0.9))):
        if scene.ngeom >= scene.maxgeom:
            return
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.eye(3).flatten(),
                            np.array([*rgb, alpha], dtype=np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, width, T[:3, 3], T[:3, 3] + length * T[:3, k])
        scene.ngeom += 1


def add_sphere(scene, p, radius, rgba) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([radius, 0, 0]),
                        np.asarray(p, dtype=float), np.eye(3).flatten(), np.asarray(rgba, dtype=np.float32))
    scene.ngeom += 1


def render(session: SliderSession, renderer: mujoco.Renderer, cam_cfg: dict, state: dict) -> bytes:
    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = cam_cfg["az"], cam_cfg["el"], cam_cfg["dist"]
    cam.lookat[:] = [cam_cfg["lx"], cam_cfg["ly"], cam_cfg["lz"]]
    opt = mujoco.MjvOption()
    if cam_cfg.get("frames"):                    # the frames sit inside the links: make the arm see-through
        opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 1
    renderer.update_scene(session.env.data, camera=cam, scene_option=opt)
    scene = renderer.scene
    sel = int(cam_cfg.get("sel", -1))
    if cam_cfg.get("frames"):
        frames = [np.eye(4)] + [np.array(F) for F in state["frames"]] + [np.array(state["T_fk"])]
        for i, F in enumerate(frames):              # frame 0 (base) .. 7, then the tool
            big = i == sel
            add_axes(scene, F, 0.13 if big else 0.08, 0.009 if big else 0.005)
    if cam_cfg.get("target") and state["mode"] == "cartesian":
        t = state["target"]
        R = np.array([[np.cos(t["yaw"]), np.sin(t["yaw"]), 0], [np.sin(t["yaw"]), -np.cos(t["yaw"]), 0], [0, 0, -1]])
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, [t["x"], t["y"], t["z"]]
        add_sphere(scene, T[:3, 3], 0.008, (1.0, 0.75, 0.1, 0.6))
        add_axes(scene, T, 0.05, 0.003, 0.6)
    img = Image.fromarray(renderer.render())
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=640)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    session = SliderSession(args.seed)
    m = session.env.model
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, args.width)
    m.vis.global_.offheight = max(m.vis.global_.offheight, args.height)
    renderer = mujoco.Renderer(m, height=args.height, width=args.width)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0

    shared = Shared()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(shared))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://localhost:{args.port}"
    print(f"Slider control running at {url}   (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)

    last, last_state = time.perf_counter(), 0.0
    state = session.state()
    try:
        while True:
            while not shared.commands.empty():
                apply(session, shared.commands.get())
            now = time.perf_counter()
            periods = min(int((now - last) / 0.01), 10)        # real time, 10 ms per control period
            if periods:
                session.step(periods)
                last += periods * 0.01
                if now - last > 0.1:                           # fell behind (slow machine): do not try to catch up
                    last = now
            if now - last_state > 0.066:
                state = session.state()
                body = json.dumps(state).encode()
                with shared.lock:
                    shared.state_json = body
                last_state = now
            with shared.frame_request:
                wanted, cam_cfg = shared.frame_wanted, dict(shared.camera)
            if wanted:
                jpg = render(session, renderer, cam_cfg, state)
                with shared.frame_request:
                    shared.frame, shared.frame_wanted = jpg, False
                    shared.frame_id += 1
                    shared.frame_request.notify_all()
            time.sleep(0.002)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.shutdown()
        renderer.close()


if __name__ == "__main__":
    main()
