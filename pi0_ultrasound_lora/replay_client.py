#!/usr/bin/env python
"""回放客户端：纯软件验证远程推理的 wire 契约（不碰硬件）。

拿一个**从数据集回放的假机器人**接进 LeRobot 真机同款的 `RobotClient`，让整条链路
（观测序列化 → gRPC → 服务端预处理/推理/后处理 → 动作 chunk → 客户端队列 → 下发）
原样跑一遍，然后把实际下发的动作和数据集真值逐帧对比。

为什么这么做，而不是直接接机械臂：
  - 契约错了（键名、维序、单位、roll 符号）在真机上表现为"机械臂乱动"，很难归因；
    在这里表现为 MAE 从 1.19 跳到几十甚至几百，一眼可辨。
  - 这份 `ReplayRobot` 就是 xArm6 驱动的模板：把 `get_observation` 换成读相机+读
    SDK、`send_action` 换成下发 SDK 即可，接口一字不用改。

── 契约要点（都是从源码里挖出来的，不是猜的）────────────────────────────
1. **相机名必须逐字等于 policy 的键**，即 `base_0_rgb` / `left_wrist_0_rgb`。
   `helpers.py:166` 的 `policy_image_features[key]` 是直接索引，而且发生在
   preprocessor **之前** —— 所以 `rename_map` 在这条路上救不了键名不匹配，
   只有 KeyError。（实测 ckpt 的键是 observation.images.{base_0_rgb,
   left_wrist_0_rgb, right_wrist_0_rgb}；第三个槽位机器人不用给，policy 会
   自动补 mask=0 的空图。）
2. **特征表的插入顺序 = observation.state 的维序**。`feature_utils.py:93` 的
   `names = list(joint_fts)` 直接取字典顺序，而 `build_dataset_frame` 按
   `names` 逐位组装。顺序写错就是静默换轴，不报错。
3. **图像给 HWC uint8**。数据集里是 CHW float [0,1]，而
   `resize_robot_observation_image` 内部 `permute(2,0,1)`，要求入参是 HWC；
   `prepare_image` 只在 dtype==uint8 时才 /255。真机相机给的就是 uint8，
   这里也按 uint8 送，好让这条路径和真机完全一致。
4. **fps=15，不是默认的 30**。数据集就是 15fps，`chunk=50` ⇒ 3.33s 轨迹。
   `RobotClientConfig.fps` 默认 30，会让动作的时间语义整体错一倍。
5. **roll 必须是解卷绕后的值**（训练分布 [163.97, 193.04]）。机械臂 SDK 报出来的是
   −180..180，同一个物理姿态在 ±180 两侧会差 360°。这条本脚本从已修好的数据集
   读取所以天然正确，脚本会断言并打印 range —— 真机驱动必须自己实现同样的
   `unwrap_roll.py` 逻辑。

用法（先起 serve.py，再跑这个）：
    python serve.py --ckpt <ckpt> &
    python replay_client.py --episode 0 --n-frames 300
"""

import argparse
import glob
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

for _v in ("ALL_PROXY", "all_proxy"):
    os.environ.pop(_v, None)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from lerobot.async_inference.configs import RobotClientConfig  # noqa: E402
from lerobot.async_inference.robot_client import RobotClient  # noqa: E402
from lerobot.robots import RobotConfig  # noqa: E402
from lerobot.robots.robot import Robot  # noqa: E402

ROOT = "/data/vla/vla_batch_20261007_pi0_6d_rollunwrap"
REPO = "local/vla_batch_20261007_pi0_6d_rollunwrap"
DEFAULT_CKPT = (
    "pi0_ultrasound_lora/run1/checkpoints/012250/pretrained_model"
)

# 维序 = 训练时的 state / action 维序，必须逐位一致
DIMS = ["x_mm", "y_mm", "z_mm", "roll_deg", "pitch_deg", "yaw_deg"]

# 相机名 = policy config 的键去掉 "observation.images." 前缀后的部分。
# 数据集里叫 rgb / ultrasound，但这里**不能**用数据集的名字（见模块 docstring 第 1 条）。
CAM_BASE, DS_CAM_BASE = "base_0_rgb", "observation.images.rgb"
CAM_WRIST, DS_CAM_WRIST = "left_wrist_0_rgb", "observation.images.ultrasound"

# 离线在 measure_inference.py 上测到的部署口径单次 MAE。这里是它的端到端对照值。
OFFLINE_MAE = 1.191


@RobotConfig.register_subclass("replay")
@dataclass(kw_only=True)
class ReplayRobotConfig(RobotConfig):
    root: str = ROOT
    repo_id: str = REPO
    episode: int = 0
    start_frame: int = -1  # -1 = 自动挑一帧"正在运动"的（避开起停段）
    n_frames: int = 300


class ReplayRobot(Robot):
    """从 LeRobot 数据集回放的假机器人。接口与真机驱动完全一致。"""

    config_class = ReplayRobotConfig
    name = "replay"

    def __init__(self, config: ReplayRobotConfig):
        super().__init__(config)
        self.config = config
        self._connected = False
        self.ds = None
        self.A = None  # 本条 episode 的 GT action, (len, 6)
        self.task = ""
        self.ep_from = 0
        self.t = 0  # 机器人时间轴：下一个要被观测的 episode 内帧号
        self.end_frame = 0  # = 起点 + n_frames。注意 t 是绝对帧号，不能直接和计数比
        self.performed = []  # 实际下发的动作
        self.performed_frames = []
        self.observed_frames = []
        self.exhausted = threading.Event()

    # ---------- 特征表 ----------
    @property
    def observation_features(self) -> dict:
        # 顺序即 observation.state 的维序（feature_utils.py:93 取字典顺序）
        f = {n: float for n in DIMS}
        f[CAM_BASE] = (720, 1280, 3)
        f[CAM_WRIST] = (256, 256, 3)
        return f

    @property
    def action_features(self) -> dict:
        # 顺序即 RobotClient._action_tensor_to_action_dict 的逐位映射顺序
        return {n: float for n in DIMS}

    # ---------- 生命周期 ----------
    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, calibrate: bool = True) -> None:
        if self._connected:
            return
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.ds = LeRobotDataset(self.config.repo_id, root=self.config.root, video_backend="pyav")
        ep = self.ds.meta.episodes[self.config.episode]
        self.ep_from = int(ep["dataset_from_index"])
        self.ep_len = int(ep["length"])
        self.task = str(ep["tasks"][0])  # 零视频解码，直接从 episodes 元数据拿

        # GT action（parquet 里没有图像，读起来很便宜）
        files = sorted(glob.glob(f"{self.config.root}/data/*/*.parquet"))
        df = pd.concat([pq.read_table(f).to_pandas() for f in files], ignore_index=True)
        sub = df[df.episode_index == self.config.episode].sort_values("frame_index")
        self.A = np.stack(sub["action"].values).astype(np.float64)

        if self.config.start_frame < 0:
            self.t = self._pick_moving_frame()
        else:
            self.t = self.config.start_frame
            if self.t + self.config.n_frames > self.ep_len:
                raise SystemExit(
                    f"起点 {self.t} + 回放 {self.config.n_frames} 帧 超出了这条 episode "
                    f"的 {self.ep_len} 帧"
                )
        self.end_frame = self.t + self.config.n_frames
        self._connected = True

        print(f"  episode {self.config.episode}: {self.ep_len} 帧, 全局起点 {self.ep_from}")
        print(f"  任务串: {self.task!r}")
        print(f"  回放区间: episode 内第 {self.t} -> {self.end_frame} 帧"
              f"（共 {self.config.n_frames} 帧, {self.config.n_frames / 15:.1f}s @15fps）")
        self._print_contract_check()

    def _pick_moving_frame(self) -> int:
        """挑一帧"在动"的，避开起停段 —— 与 measure_inference.pick_frames 同一策略。"""
        spd = np.linalg.norm(np.diff(self.A, axis=0), axis=1)
        moving = spd > np.percentile(spd[spd > 0], 40)
        room = self.config.n_frames + 1
        ok = np.where(moving & (np.arange(len(spd)) + room < len(self.A)))[0]
        if len(ok) == 0:
            raise SystemExit("这条 episode 里找不到「在动且后面还剩够帧数」的起点")
        return int(ok[len(ok) // 2])

    def _print_contract_check(self):
        """把契约里最容易错的几条当场断言掉。"""
        st = self.A[self.t]
        seg = self.A[self.t : self.t + self.config.n_frames]
        print("  契约自检:")
        roll = seg[:, 3]
        ok_roll = bool(roll.min() > 90 or roll.max() < -90)
        print(
            f"    roll 解卷绕   : [{roll.min():.2f}, {roll.max():.2f}]  "
            f"{'OK（未跨 ±180）' if ok_roll else '!! 跨了 ±180，unwrap 没做对'}"
        )
        print(f"    state 单位    : x={st[0]:.1f} y={st[1]:.1f} z={st[2]:.1f} (mm)  "
              f"roll={st[3]:.2f} pitch={st[4]:.2f} yaw={st[5]:.2f} (deg)")

    def calibrate(self) -> None:
        pass

    def is_calibrated(self) -> bool:
        return True

    def configure(self) -> None:
        pass

    def disconnect(self) -> None:
        self._connected = False

    # ---------- 数据流 ----------
    def _frame_to_raw(self, k: int) -> dict:
        fr = self.ds[self.ep_from + k]
        st = fr["observation.state"].numpy().astype(np.float64)
        raw = {n: float(st[i]) for i, n in enumerate(DIMS)}
        for cam, key in ((CAM_BASE, DS_CAM_BASE), (CAM_WRIST, DS_CAM_WRIST)):
            img = fr[key]  # (3,H,W) float32 [0,1]
            # 必须转成 HWC：resize_robot_observation_image 内部 permute(2,0,1)，
            # 直接送 CHW 会被当成 HWC 重排，图像彻底烂掉（而且不报错）。
            hwc = img.permute(1, 2, 0).numpy()
            # uint8 是真相机的格式，prepare_image 见 uint8 才会 /255
            raw[cam] = np.ascontiguousarray((hwc * 255.0).round().clip(0, 255).astype(np.uint8))
        return raw

    def get_observation(self) -> dict:
        self.observed_frames.append(self.t)
        return self._frame_to_raw(self.t)

    def send_action(self, action: dict) -> dict:
        """真机这里换成 SDK 下发。回放时只记录，并推进机器人时间轴。

        时间轴只在 send_action 里推进：一次控制循环执行一个动作 = 机器人前进一帧。
        执行完 action[t] 之后物理状态就是 state[t+1]，所以下一帧观测 t+1 是对的，
        而这一步的 GT 就是 A[t]。
        """
        if not self.exhausted.is_set():
            self.performed.append([float(action[n]) for n in DIMS])
            self.performed_frames.append(self.t)
            self.t += 1
            if self.t >= self.end_frame:
                self.exhausted.set()
        return action


def patch_robot_factory():
    """把 ReplayRobot 接进真机同款的 RobotClient。

    `robots/utils.py:25` 的 `make_robot_from_config` 是**硬编码 if/elif**，第三方机器人
    只能靠 `lerobot_robot_*` 发行包注册，所以只能运行时替换。替换发生在**本文件内**，
    不改 src/。
    """
    import lerobot.async_inference.robot_client as rc

    original = rc.make_robot_from_config

    def make_robot_from_config(config):
        if isinstance(config, ReplayRobotConfig):
            return ReplayRobot(config)
        return original(config)

    rc.make_robot_from_config = make_robot_from_config


def run_replay(args) -> tuple[ReplayRobot, RobotClient, dict]:
    patch_robot_factory()

    robot_cfg = ReplayRobotConfig(
        id="replay",
        calibration_dir=Path("/tmp/lerobot_replay_calib"),
        root=args.root,
        repo_id=args.repo_id,
        episode=args.episode,
        start_frame=args.start_frame,
        n_frames=args.n_frames,
    )
    client_cfg = RobotClientConfig(
        policy_type="pi0",
        pretrained_name_or_path=args.ckpt,
        robot=robot_cfg,
        actions_per_chunk=args.actions_per_chunk,
        task="",  # 真值从数据集读，构造完再传（control_loop(task=...) 是显式入参）
        server_address=f"{args.host}:{args.port}",
        policy_device=args.device,
        client_device="cpu",
        chunk_size_threshold=args.chunk_size_threshold,
        fps=args.fps,
        aggregate_fn_name="latest_only",
    )

    print("\n【1】连接")
    client = RobotClient(client_cfg)
    robot: ReplayRobot = client.robot

    # 统计钩子：RobotClient 自身不暴露这些计数
    stats = {"chunks": 0, "chunk_lens": [], "starved": 0, "iters": 0}
    agg_orig = client._aggregate_action_queues

    def agg_counted(actions, fn=None):
        stats["chunks"] += 1
        stats["chunk_lens"].append(len(actions))
        return agg_orig(actions, fn)

    client._aggregate_action_queues = agg_counted
    avail_orig = client.actions_available

    def avail_counted():
        stats["iters"] += 1
        ok = avail_orig()
        if not ok:
            stats["starved"] += 1
        return ok

    client.actions_available = avail_counted

    print("\n【2】握手 + 下发策略指令（服务端在这里加载 LoRA）")
    t0 = time.perf_counter()
    if not client.start():
        raise SystemExit(
            f"连不上服务端 {args.host}:{args.port} —— 先起 serve.py，再跑本脚本"
        )
    print(f"  握手完成 {time.perf_counter() - t0:.1f}s")

    print(f"\n【3】回放 {args.n_frames} 帧 @ {args.fps}fps（含 {args.actions_per_chunk} 步 chunk）")
    rx = threading.Thread(target=client.receive_actions, daemon=True)
    rx.start()
    ctl = threading.Thread(target=client.control_loop, kwargs={"task": robot.task}, daemon=True)
    ctl.start()

    t0 = time.perf_counter()
    if not robot.exhausted.wait(timeout=args.timeout):
        print(f"  !! 超时 {args.timeout}s，只执行了 {len(robot.performed)} 步")
    wall = time.perf_counter() - t0

    # 先置停标志让两个线程自己退出，隔一会儿再拆连接。直接 client.stop() 会在
    # 线程还在 RPC 里的时候关掉 channel，LeRobot 自己会打两条 "Channel closed!" /
    # "Client not running" 的日志 —— 是无害的竞态，但会淹没真正的报错。
    # 服务端 GetActions 最长阻塞 obs_queue_timeout(2s) 就会返回 Empty，
    # 所以给 3s 足够让接收线程走到 while 判断上自己退出。
    client.shutdown_event.set()
    ctl.join(timeout=3)
    rx.join(timeout=3)
    client.stop()
    ctl.join(timeout=5)
    rx.join(timeout=5)
    print(f"  回放结束，耗时 {wall:.1f}s")
    stats["wall"] = wall
    return robot, client, stats


def report(robot: ReplayRobot, client: RobotClient, stats: dict, args):
    performed = np.array(robot.performed)
    frames = np.array(robot.performed_frames)
    if len(performed) == 0:
        raise SystemExit("一步都没执行 —— 服务端可能没返回动作，看服务端日志")

    gt = robot.A[frames]
    err = np.abs(performed - gt)

    print("\n【4】时序与队列")
    print(f"  下发动作步数   : {len(performed)}")
    print(f"  发出观测帧数   : {len(robot.observed_frames)}  "
          f"（只占执行步数的 {100 * len(robot.observed_frames) / len(performed):.0f}% "
          f"—— 客户端只在队列降到阈值以下才问策略，这是设计如此）")
    print(f"  收到动作 chunk : {stats['chunks']} 个，长度 {stats['chunk_lens']}")
    # 不能用 client.fps_tracker：它只在 control_loop(verbose=True) 时才被喂数据
    print(f"  实测观测频率   : {len(robot.observed_frames) / max(stats['wall'], 1e-9):.2f} Hz"
          f"（目标 {args.fps}）")
    print(f"  控制循环迭代   : {stats['iters']} 次，均周期 "
          f"{1000 * stats['wall'] / max(stats['iters'], 1):.1f} ms（目标 {1000 / args.fps:.1f} ms）")
    # 正常情况这个数 = 首帧推理的那段空窗（约 0.35s ≈ 5 步），真机上表现为
    # "启动后头几步没有动作可执行"。若明显大于这个数，说明推理跟不上执行。
    print(f"  队列饿死次数   : {stats['starved']}  "
          f"（首帧约 0.35s 推理空窗 = 约 5 步；明显更多说明推理跟不上执行）")
    q = client.action_queue_size
    if q:
        print(f"  队列深度       : 最小 {min(q)} 最大 {max(q)} 末尾 {q[-1]}")

    print("\n【5】动作误差：实际下发的动作 vs 数据集真值（同一帧）")
    print(f"  {'维度':<12s} {'MAE':>9s} {'最大误差':>10s} {'该段 GT 行程':>13s} {'占行程':>8s}")
    for i, nm in enumerate(DIMS):
        span = float(gt[:, i].max() - gt[:, i].min())
        # 行程阈值 0.5：低于这个值时比值是除以 ~0，会印出 255% 这种误导性数字
        # （roll 段内只动 0.078°，MAE 0.200° 就被算成 255%）。真值行程本身也印出来了。
        ratio = f"{100 * err[:, i].mean() / span:6.1f}%" if span >= 0.5 else "   n/a"
        print(f"  {nm:<12s} {err[:, i].mean():9.3f} {err[:, i].max():10.3f} "
              f"{span:13.3f} {ratio:>8s}")
    mae = float(err.mean())
    print(f"\n  整体 MAE = {mae:.3f}")
    print(f"  离线 measure_inference.py 的部署口径单次 MAE = {OFFLINE_MAE:.3f}")

    print("\n【6】量纲 / 解卷绕 sanity")
    for i, nm in ((3, "roll_deg"), (5, "yaw_deg")):
        p, g = performed[:, i], gt[:, i]
        print(f"  {nm:<9s} 下发 [{p.min():8.2f}, {p.max():8.2f}]   "
              f"GT [{g.min():8.2f}, {g.max():8.2f}]")
    print(f"  {'x_mm':<9s} 下发 [{performed[:, 0].min():8.2f}, {performed[:, 0].max():8.2f}]   "
          f"GT [{gt[:, 0].min():8.2f}, {gt[:, 0].max():8.2f}]")

    print("\n【7】判定")
    if mae < 3.0:
        print(f"  ✅ 契约对齐。端到端 MAE {mae:.3f}，与离线 {OFFLINE_MAE:.3f} 同量级")
        print("     （这里比离线略差是正常的：图像被量化成 uint8，而且走的是真实的异步队列路径）")
    elif mae < 10.0:
        print(f"  ⚠️  MAE {mae:.3f} 明显高于离线 {OFFLINE_MAE:.3f}。链路是通的但某处口径有偏")
        print("     优先查：图像 HWC/CHW、相机键名、state 维序")
    else:
        print(f"  ❌ MAE {mae:.3f} —— 契约没对齐。按这个顺序查：")
        print("     1. 服务端日志有没有 'adapter 已注入'（没有就是加载成了随机初始化模型）")
        print("     2. state 维序是否与 DIMS 一致（错了会静默换轴）")
        print("     3. roll 有没有解卷绕")
        print("     4. 相机键名是否就是 base_0_rgb / left_wrist_0_rgb")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--repo-id", default=REPO)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--start-frame", type=int, default=-1)
    ap.add_argument("--n-frames", type=int, default=300)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--actions-per-chunk", type=int, default=50)
    ap.add_argument("--chunk-size-threshold", type=float, default=0.5)
    ap.add_argument("--timeout", type=float, default=600)
    args = ap.parse_args()

    print("=" * 74)
    print("回放闭环：RobotClient（真机同款）<-> 数据集回放的假机器人 <-> gRPC <-> serve.py")
    print("=" * 74)
    robot, client, stats = run_replay(args)
    report(robot, client, stats, args)


if __name__ == "__main__":
    main()
