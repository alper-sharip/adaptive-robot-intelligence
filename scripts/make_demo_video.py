"""M23: render the project demo video (outputs/demo.mp4) from live simulation.

Every clip is a real episode run by the code in this repository on a fixed seed; nothing is
edited except the captions and the playback speed (1.5x: 20 simulated frames per second shown
at 30). Needs ffmpeg.

    python scripts/make_demo_video.py                 # all scenes, then joined
    python scripts/make_demo_video.py --only language # re-render one scene and re-join
    python scripts/make_demo_video.py --stills        # one frame per scene to outputs/demo_stills/ (fast check)
"""

import argparse
import subprocess
from pathlib import Path

from ari.envs.pick_place import PickPlaceEnv, SceneConfig  # isort: skip  (ari first: headless renderer)

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1280, 720, 30
INK, MUTED, PANEL = (255, 255, 255), (190, 198, 210), (14, 20, 30)
FONT_DIRS = ["/usr/share/fonts/opentype/inter", "/usr/share/fonts/truetype/dejavu", "/System/Library/Fonts/Supplemental",
             "/Library/Fonts"]


def font(size: int, bold: bool = False):
    names = (["Inter-SemiBold.otf", "DejaVuSans-Bold.ttf", "Arial Bold.ttf"] if bold
             else ["Inter-Regular.otf", "DejaVuSans.ttf", "Arial.ttf"])
    for d in FONT_DIRS:
        for n in names:
            if (Path(d) / n).exists():
                return ImageFont.truetype(str(Path(d) / n), size)
    return ImageFont.load_default()


F_TITLE, F_BODY, F_SMALL, F_HUGE = font(30, True), font(24), font(19), font(54, True)


def caption(img: Image.Image, title: str, line: str, note: str = "") -> Image.Image:
    """Stage title + one line of result on a dark band along the bottom."""
    d = ImageDraw.Draw(img, "RGBA")
    d.rectangle([0, H - 112, W, H], fill=(*PANEL, 225))
    d.text((40, H - 100), title, font=F_TITLE, fill=INK)
    d.text((40, H - 58), line, font=F_BODY, fill=MUTED)
    if note:
        w = d.textlength(note, font=F_SMALL)
        d.text((W - 40 - w, H - 94), note, font=F_SMALL, fill=MUTED)
    return img


class Clip:
    """Collects frames from a running episode and encodes them with ffmpeg."""

    def __init__(self, env, title: str, line: str, lookat, distance, azimuth, elevation, overlay=None, stills=False):
        env.model.vis.global_.offwidth, env.model.vis.global_.offheight = W, H
        self.env, self.title, self.line, self.overlay, self.stills = env, title, line, overlay, stills
        self.renderer = mujoco.Renderer(env.model, height=H, width=W)
        # The mirror floor distracts, and the software renderer draws striped artefacts at the
        # edge of the shadow map; both are switched off.
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:] = lookat
        self.cam.distance, self.cam.azimuth, self.cam.elevation = distance, azimuth, elevation
        self.frames: list[bytes] = []
        self.images: list = []
        self.status = ""
        self.count, self.next_t = 0, 0.0

    def grab(self, env=None, *_):
        if self.env.data.time < self.next_t - 1e-6:     # one frame per 50 ms of simulation, whoever calls
            return
        self.next_t = self.env.data.time + 0.05
        self.count += 1
        if self.stills and self.count % 60 != 30:       # stills mode: one frame every 3 s of simulation
            return
        self.renderer.update_scene(self.env.data, camera=self.cam)
        img = Image.fromarray(self.renderer.render())
        if self.overlay is not None:
            self.overlay(img)
        caption(img, self.title, self.status or self.line, "live simulation, 1.5x speed")
        self.frames.append(img.tobytes())
        if self.stills:
            self.images.append(img)

    def hold(self, seconds: float):
        """Repeat the last frame (lets the viewer read the outcome)."""
        self.frames += [self.frames[-1]] * int(seconds * FPS)

    def write(self, path: Path):
        if self.stills:
            for k, img in enumerate(self.images):
                img.save(path.with_name(f"{path.stem}_{k}.png"))
        else:
            encode(self.frames, path)
        self.renderer.close()


def encode(frames: list[bytes], path: Path):
    tmp = path.with_name(path.stem + ".part.mp4")
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                          "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "20", "-preset", "veryfast",
                          "-pix_fmt", "yuv420p", "-r", str(FPS), str(tmp)], stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(f)
    p.stdin.close()
    if p.wait() != 0:
        raise SystemExit(f"ffmpeg failed on {path}")
    tmp.rename(path)


def card(lines: list[tuple[str, object, tuple]], seconds: float, path: Path, image: Path | None = None):
    """A still: centred text lines (text, font, colour), optionally above a picture."""
    img = Image.new("RGB", (W, H), PANEL)
    d = ImageDraw.Draw(img)
    y = 60 if image else (H - sum(f.size + 22 for _, f, _ in lines)) // 2
    for text, f, colour in lines:
        d.text(((W - d.textlength(text, font=f)) // 2, y), text, font=f, fill=colour)
        y += f.size + 22
    if image is not None:
        pic = Image.open(image).convert("RGB")
        scale = min((W - 120) / pic.width, (H - y - 50) / pic.height)
        pic = pic.resize((int(pic.width * scale), int(pic.height * scale)), Image.LANCZOS)
        img.paste(pic, ((W - pic.width) // 2, y + max(10, (H - y - pic.height) // 2 - 20)))
    if STILLS:
        img.save(path.with_suffix(".png"))
    else:
        encode([img.tobytes()] * int(seconds * FPS), path)


TABLE_VIEW = dict(lookat=[0.47, 0.02, 0.10], distance=1.25, azimuth=155, elevation=-28)
WALL_VIEW = dict(lookat=[0.45, 0.02, 0.18], distance=1.45, azimuth=140, elevation=-22)
TOP_VIEW = dict(lookat=[0.48, 0.02, 0.05], distance=1.0, azimuth=180, elevation=-48)


# ---------------------------------------------------------------- scenes
def scene_scripted(path, stills):
    from ari.skills.pick_place import ScriptedPickPlace, run_episode

    env = PickPlaceEnv()
    clip = Clip(env, "Stage 2  ·  Scripted pick-and-place", "Own kinematics, IK and controller. 100 of 100 seeded episodes.",
                stills=stills, **TABLE_VIEW)
    res = run_episode(env, ScriptedPickPlace(), 3, frame_callback=clip.grab, frame_every=5)
    assert res.success
    clip.hold(0.6)
    clip.write(path)


def scene_wall(path, stills):
    from ari.skills.pick_place import run_episode
    from ari.skills.planned_pick_place import PlannedPickPlace

    env = PickPlaceEnv(SceneConfig.with_wall())
    clip = Clip(env, "Stage 2  ·  Collision-aware motion planning",
                "RRT-Connect over a random wall: 96% clean, against 60% for a hand-written heuristic.", stills=stills, **WALL_VIEW)
    res = run_episode(env, PlannedPickPlace(plan_seed=1), 1, frame_callback=clip.grab, frame_every=5)
    assert res.success
    clip.hold(0.6)
    clip.write(path)


def _rollout(path, stills, spec, seed, title, line, disturb=False, joint_interp=False, overlay_factory=None, view=TABLE_VIEW,
             moved_text="cube moved by hand mid-episode: the policy re-targets"):
    from ari.learning import demos
    from ari.learning.evaluate import CubeShift, make_policy

    env = PickPlaceEnv()
    policy = make_policy(spec, env)
    clip = Clip(env, title, line, stills=stills, overlay=overlay_factory(env, policy) if overlay_factory else None, **view)
    shift = CubeShift(seed) if disturb else None

    moved_at = [None]

    def on_frame(env):
        if shift is not None and shift.applied and moved_at[0] is None:
            moved_at[0] = env.data.time
        showing = moved_at[0] is not None and env.data.time - moved_at[0] < 2.5      # then back to the result line
        clip.status = moved_text if showing else ""
        clip.grab()

    res = demos.rollout(env, policy, seed, max_time=14.0, disturbance=shift, frame_callback=on_frame, joint_interp=joint_interp)
    assert res.success, f"{spec} seed {seed} failed: {res.failure}"
    clip.hold(0.6)
    clip.write(path)


def scene_dagger(path, stills):
    _rollout(path, stills, "outputs/models/bc_dagger.pt", 3, "Stage 4  ·  Learned policy (behaviour cloning + DAgger)",
             "A neural network at 20 Hz. 99% with the cube moved mid-episode.", disturb=True)


def scene_image(path, stills):
    def overlay_factory(env, policy):
        def overlay(img):                       # what the network sees: its two 64x64 camera images
            if policy.last is None:
                return
            d = ImageDraw.Draw(img, "RGBA")
            for k, (v, name) in enumerate(zip(policy.last[0], ("front camera", "wrist camera"))):
                a = np.ascontiguousarray(v.transpose(1, 2, 0))     # (2, 3, 64, 64) uint8 -> HWC
                x = W - 40 - 192 - (1 - k) * 212
                d.rectangle([x - 4, 36, x + 196, 236], fill=(*PANEL, 255))
                img.paste(Image.fromarray(a).resize((192, 192), Image.NEAREST), (x, 40))
                d.text((x, 242), name, font=F_SMALL, fill=INK)
        return overlay

    _rollout(path, stills, "outputs/models/vision_cpu.pt", IMAGE_SEED, "Stage 4  ·  Policy from camera images only",
             "Input: the two 64x64 images shown. 43% success: it grasps reliably, placing is the weak point.",
             overlay_factory=overlay_factory)


def scene_rl(path, stills):
    _rollout(path, stills, "outputs/models/rl_residual.pt", 0, "Stage 5  ·  Reinforcement learning (own PPO)",
             "A 100-demo clone (66%) corrected by residual RL: 93%.", disturb=True, joint_interp=True)


def scene_language(path, stills):
    from ari.envs.multi_object import CUBE_COLORS, TARGET_COLORS, MultiObjectEnv
    from ari.language import evaluate as lev
    from ari.language import parser
    from ari.learning.evaluate import make_policy
    from ari.planning.task_planner import Faults, TaskExecutor, make_plan

    seed = 7
    env = MultiObjectEnv()
    env.reset(seed)
    text = lev.m20_task(seed)[0]
    steps = make_plan(text, parser.load("outputs/models/language_parser.pt"))
    clip = Clip(env, "Stage 6  ·  Language instruction, planning, recovery", "", stills=stills, **TOP_VIEW)
    faults = Faults(seed, len(steps))
    seen = [0]
    quoted = f"“{text}”"
    f_quote = next(f for f in (font(z) for z in range(26, 13, -1))          # largest size that fits the frame
                   if ImageDraw.Draw(Image.new("RGB", (8, 8))).textlength(quoted, font=f) <= W - 80)

    def overlay(img):
        d = ImageDraw.Draw(img, "RGBA")
        d.rectangle([0, 0, W, 64], fill=(*PANEL, 225))
        d.text((40, 32), quoted, font=f_quote, fill=INK, anchor="lm")

    clip.overlay = overlay

    def on_frame(env):
        if len(faults.injected) > seen[0]:
            seen[0] = len(faults.injected)
            on_frame.until = env.data.time + 2.5
            on_frame.msg = {"drop": "fault injected: cube knocked out of the gripper",
                            "bump": "fault injected: a placed cube knocked off its target"}[faults.injected[-1][0]]
        doing = f"{CUBE_COLORS[env.cube_index]} cube \u2192 {TARGET_COLORS[env.target_index]} target"
        clip.status = on_frame.msg if env.data.time < on_frame.until else f"checks every result, redoes what failed  ·  now: {doing}"
        clip.grab()

    on_frame.until, on_frame.msg = -1.0, ""
    res = TaskExecutor(env, make_policy("outputs/models/bc_dagger.pt", env), fallback=make_policy("reactive", env),
                       frame_callback=on_frame).run(steps, faults)
    assert res.success
    clip.status = "task verified complete  ·  95% of 100 two-step tasks with faults (35-45% without checking)"
    clip.grab()
    clip.hold(2.2)
    clip.write(path)


def scene_title(path, stills):
    card([("Adaptive Robot Intelligence", F_HUGE, INK),
          ("Robot manipulation in simulation, built from the kinematics up", F_TITLE, MUTED),
          ("", F_SMALL, MUTED),
          ("MuJoCo  ·  Franka Panda  ·  PyTorch  ·  no robot hardware, no GPU", F_BODY, MUTED),
          ("Alper Sharip", F_BODY, INK)], 3.5, path)


def scene_perception(path, stills):
    card([("Stage 3  ·  Perception from an RGB-D camera", F_TITLE, INK),
          ("Cube located to 0.3-0.6 mm from colour and depth; vision-driven policies match ground truth.", F_BODY, MUTED)],
         4.5, path, image=Path("outputs/perception_seed3_realistic.png"))


def scene_benchmark(path, stills):
    card([("Stage 7  ·  One benchmark for every policy", F_TITLE, INK),
          (BENCH_LINE, F_BODY, MUTED)], 6.5, path, image=Path("outputs/benchmark_heatmap.png"))


def scene_end(path, stills):
    card([("Everything shown is measured on fixed seeds and reproducible", F_TITLE, INK),
          ("", F_SMALL, MUTED),
          ("Failures and negative results are documented alongside the successes.", F_BODY, MUTED),
          ("", F_SMALL, MUTED),
          ("github.com/alper-sharip/adaptive-robot-intelligence", F_TITLE, INK)], 4.5, path)


STILLS = False
IMAGE_SEED = 3
BENCH_LINE = "6 policies x 13 conditions x 100 episodes. No policy is best everywhere; each fails in its own way."
SCENES = [("title", scene_title), ("scripted", scene_scripted), ("wall", scene_wall), ("perception", scene_perception),
          ("dagger", scene_dagger), ("image", scene_image), ("rl", scene_rl), ("language", scene_language),
          ("benchmark", scene_benchmark), ("end", scene_end)]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--stills", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("outputs/demo.mp4"))
    args = ap.parse_args()
    STILLS = args.stills
    work = Path("outputs/demo_stills" if args.stills else "outputs/demo_scenes")
    work.mkdir(parents=True, exist_ok=True)
    for name, fn in SCENES:
        if args.only and name not in args.only:
            continue
        fn(work / f"{name}.mp4", args.stills)
        print("rendered", name, flush=True)
    if not args.stills:
        parts = [work / f"{n}.mp4" for n, _ in SCENES]
        missing = [str(q) for q in parts if not q.exists()]
        if missing:
            raise SystemExit(f"scenes not rendered yet: {missing}")
        (work / "list.txt").write_text("".join(f"file '{q.name}'\n" for q in parts))
        tmp = args.out.with_name(args.out.stem + ".part.mp4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(work / "list.txt"),
                        "-c", "copy", "-movflags", "+faststart", str(tmp)], check=True)
        tmp.rename(args.out)
        print("saved", args.out)
        # README teaser: the language scene at 2x, small enough for GitHub to show inline.
        teaser = args.out.with_name("demo_teaser.gif")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(work / "language.mp4"), "-filter_complex",
                        "setpts=PTS/2,fps=10,scale=560:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];"
                        "[b][p]paletteuse=dither=bayer:bayer_scale=4", str(teaser)], check=True)
        print("saved", teaser)
