#!/usr/bin/env python
"""瘦客户端：在**没有 LeRobot、没有 torch** 的机器上，把回放包发过去推理，把动作收回来。

依赖只有 `grpcio` + `numpy`（+ protobuf，grpcio 会带）。对面不需要 1.8G 数据集、
不需要 LeRobot、不需要 CUDA。

它复刻的是 `lerobot/async_inference/robot_client.py` 的**线上行为**，不是那个类 ——
对面没装 LeRobot，只能用本目录里那两个 protobuf 生成文件 + 一段 pickle 垫片手搓。
两条路必须字节级同构，下面把要点写清楚，改的时候别踩。

── 为什么需要 pickle 垫片 ────────────────────────────────────────────────
观测/动作在传输时是 `pickle.dumps(TimedObservation/TimedAction)`，而 pickle 里记的是
**类的模块路径 + 类名**：`lerobot.async_inference.helpers.TimedObservation`。
对面没有这个模块，unpickle 就会 `ModuleNotFoundError`。

解法：在 `sys.modules` 里塞一个同名模块，里面放**属性名完全一致**的 dataclass。
pickle 对 dataclass 走的是 NEWOBJ + `__dict__` 状态，所以**只要属性名对得上就行**，
字段顺序、有没有基类、有没有方法都不影响。

── 与真机路径的三个差异（都是刻意的）────────────────────────────────────
1. **相机的键名**：发出去的是 `base_0_rgb` / `left_wrist_0_rgb`，即 policy 的键。
   这不是真机的做法（真机在自己那边用 rename_map 映射），而是因为
   `helpers.py:166` 的 `policy_image_features[key]` 是**直接索引**且发生在 preprocessor
   之前 —— 键名不匹配只有 KeyError，rename_map 救不了。见 replay_client.py 的说明。
2. **图像是预先缩好的 224×224**：服务端还会再缩一次，而 224→224 是精确恒等，
   所以给模型的张量与本机全分辨率路径逐位相同。省 10 倍带宽（见 make_replay_bundle.py）。
3. **服务端要用 `--wire numpy` 起**：默认格式回传的是 torch.Tensor，对面没 torch
   就 unpickle 不了。这是唯一为"对面没 torch"做的协议让步。

用法：
    # 本机
    python serve.py --wire numpy --host 0.0.0.0 --port 8080
    # 对面
    python remote_client.py --server 100.64.0.1:8080 --bundle replay_bundle.npz
"""

import argparse
import io
import json
import os
import pickle  # nosec
import sys
import threading
import time
import types
from dataclasses import dataclass, field
from queue import Queue

# 让 `import services_pb2` 能找到同目录下那两个文件（从别的目录跑也成立）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

try:
    import grpc  # noqa: E402
    import services_pb2  # noqa: E402
    import services_pb2_grpc  # noqa: E402
except ImportError as e:  # 给出比 "No module named" 更有用的提示
    raise SystemExit(
        f"缺依赖：{e}\n"
        f"先装：python3 -m pip install -r requirements.txt\n"
        f"（只需要 grpcio / numpy / protobuf，不需要 torch 和 lerobot）"
    ) from e

TransferState = services_pb2.TransferState
CHUNK_SIZE = 2 * 1024 * 1024
MAX_MESSAGE_SIZE = 4 * 1024 * 1024

# 与训练时逐位一致的维序 —— 顺序错了就是静默换轴
DIMS = ["x_mm", "y_mm", "z_mm", "roll_deg", "pitch_deg", "yaw_deg"]

# 本机 serve.py 的默认 ckpt。这里发的是**服务端本地路径**（服务端拿它加载模型），
# 不是对面的路径 —— 对面根本没有这个文件。
DEFAULT_CKPT = "pi0_ultrasound_lora/run1/checkpoints/012250/pretrained_model"

# 本机 replay_client.py 测到的端到端 MAE，作为对照
LOCAL_REPLAY_MAE = 1.478
OFFLINE_MAE = 1.191


# ══════════════════════════════════════════════════════════════════════════
#  pickle 垫片
# ══════════════════════════════════════════════════════════════════════════
# 逐字抄自 lerobot/async_inference/helpers.py:202-235。
#
# **字段和方法都要抄**：pickle 恢复的是字段，但代码调用的是方法。
# 只抄字段的话 unpickle 能成功、对象看着对，一调 get_timestep() 就
# AttributeError —— 而且是在接收线程里崩，主线程只看到"动作再也不来"。
_STUB_SRC = '''
from dataclasses import dataclass, field


@dataclass
class TimedData:
    timestamp: float
    timestep: int

    def get_timestamp(self):
        return self.timestamp

    def get_timestep(self):
        return self.timestep


@dataclass
class TimedAction(TimedData):
    action: object

    def get_action(self):
        return self.action


@dataclass
class TimedObservation(TimedData):
    observation: dict
    must_go: bool = False

    def get_observation(self):
        return self.observation


@dataclass
class RemotePolicyConfig:
    policy_type: str
    pretrained_name_or_path: str
    lerobot_features: dict
    actions_per_chunk: int
    device: str = "cpu"
    rename_map: dict = field(default_factory=dict)
'''


def _install_pickle_stubs():
    """把 lerobot.async_inference.helpers 等模块名占住，好让 pickle 能解析类路径。"""
    created = []
    for pkg in ("lerobot", "lerobot.async_inference"):
        if pkg not in sys.modules:
            m = types.ModuleType(pkg)
            m.__path__ = []  # 标成包，import 机制才不会试图去磁盘找
            sys.modules[pkg] = m
            created.append(pkg)

    helpers = types.ModuleType("lerobot.async_inference.helpers")
    helpers.__path__ = []
    helpers.__file__ = "<remote_client stub>"
    exec(compile(_STUB_SRC, "<remote_client stub>", "exec"), helpers.__dict__)
    sys.modules["lerobot.async_inference.helpers"] = helpers
    setattr(sys.modules["lerobot.async_inference"], "helpers", helpers)

    # 万一服务端在别处也引用了这些名字
    for extra in ("lerobot.async_inference.constants", "lerobot.transport",
                  "lerobot.transport.utils", "lerobot.utils.feature_utils",
                  "lerobot.utils.constants"):
        if extra not in sys.modules:
            m = types.ModuleType(extra)
            m.__path__ = []
            sys.modules[extra] = m

    return created


_install_pickle_stubs()


def check_stubs_work():
    """自检：pickle 往返之后，**字段和方法**都必须还能用。

    只查字段是不够的 —— 只抄字段的垫片能通过"往返"检查，却会在接收线程里
    以 AttributeError 崩掉。这里把要用到的方法全部调一遍。
    """
    from lerobot.async_inference.helpers import TimedAction, TimedObservation

    o = TimedObservation(timestamp=1.0, timestep=7, observation={"a": np.zeros(3, np.uint8)},
                         must_go=True)
    o2 = pickle.loads(pickle.dumps(o))
    assert o2.timestep == 7 and o2.must_go is True
    assert o2.get_timestep() == 7 and o2.get_timestamp() == 1.0
    assert o2.get_observation()["a"].shape == (3,)

    a = TimedAction(timestamp=1.0, timestep=7, action=np.zeros(6))
    a2 = pickle.loads(pickle.dumps(a))
    assert a2.get_timestep() == 7 and a2.get_action().shape == (6,)
    return True


# ══════════════════════════════════════════════════════════════════════════
#  传输层（抄自 lerobot/transport/utils.py，只留用得到的）
# ══════════════════════════════════════════════════════════════════════════
def grpc_channel_options(initial_backoff: str = "0.1s"):
    service_config = {
        "methodConfig": [
            {
                "name": [{}],
                "retryPolicy": {
                    "maxAttempts": 5,
                    "initialBackoff": initial_backoff,
                    "maxBackoff": "2s",
                    "backoffMultiplier": 2,
                    "retryableStatusCodes": ["UNAVAILABLE", "DEADLINE_EXCEEDED"],
                },
            }
        ]
    }
    return [
        ("grpc.max_receive_message_length", MAX_MESSAGE_SIZE),
        ("grpc.max_send_message_length", MAX_MESSAGE_SIZE),
        ("grpc.enable_retries", 1),
        ("grpc.service_config", json.dumps(service_config)),
    ]


def send_bytes_in_chunks(buffer: bytes, message_class):
    """按 2MB 切块。注意判定顺序：**先判 END 再判 BEGIN**，
    所以小于 2MB 的载荷只会有一个 TRANSFER_END 块（接收端从空缓冲区写起，能正确处理）。
    本客户端的观测约 0.3MB，走的正是这条路。"""
    bb = io.BytesIO(buffer)
    bb.seek(0, io.SEEK_END)
    total = bb.tell()
    bb.seek(0)
    sent = 0
    while sent < total:
        if sent + CHUNK_SIZE >= total:
            state = TransferState.TRANSFER_END
        elif sent == 0:
            state = TransferState.TRANSFER_BEGIN
        else:
            state = TransferState.TRANSFER_MIDDLE
        n = min(CHUNK_SIZE, total - sent)
        yield message_class(transfer_state=state, data=bb.read(n))
        sent += n


# ══════════════════════════════════════════════════════════════════════════
#  特征表（等价于 hw_to_dataset_features(obs_features, "observation",
#  use_video=False)，见 lerobot/utils/feature_utils.py:47）
# ══════════════════════════════════════════════════════════════════════════
def make_lerobot_features(img_size: int, cam_base: str, cam_wrist: str) -> dict:
    """服务端只用到 state 的 names（在 observations_similar 里按它组装向量），
    图像的 shape 实际由 policy config 说了算，这里填对是为了语义完整。"""
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(DIMS),),
            "names": list(DIMS),  # ← 顺序 = state 维序，唯一真正要命的地方
        },
        f"observation.images.{cam_base}": {
            "dtype": "image",
            "shape": (img_size, img_size, 3),
            "names": ["height", "width", "channels"],
            "info": {"is_depth_map": False},
        },
        f"observation.images.{cam_wrist}": {
            "dtype": "image",
            "shape": (img_size, img_size, 3),
            "names": ["height", "width", "channels"],
            "info": {"is_depth_map": False},
        },
    }


# ══════════════════════════════════════════════════════════════════════════
#  回放包
# ══════════════════════════════════════════════════════════════════════════
class Bundle:
    def __init__(self, path: str):
        z = np.load(path, allow_pickle=False)
        self.state = z["state"]          # (N,6) float32
        self.action_gt = z["action_gt"]  # (N,6) float32
        self.rgb = z["rgb"]              # (N,S,S,3) uint8
        self.us = z["us"]                # (N,S,S,3) uint8
        self.meta = json.loads(str(z["meta"]))
        self.n = int(self.state.shape[0])
        self.cam_base = self.meta["cam_base"]
        self.cam_wrist = self.meta["cam_wrist"]
        self.img_size = int(self.meta["img_size"])
        self.task = self.meta["task"]
        self.dims = self.meta.get("dims", DIMS)

    def describe(self, path: str):
        print(f"  文件     : {path}（{os.path.getsize(path) / 1e6:.1f} MB）")
        print(f"  来源     : {self.meta.get('source_root')} episode {self.meta['episode']}"
              f" 第 {self.meta['start_frame']} 帧起 {self.n} 帧")
        print(f"  导出时间 : {self.meta.get('made_at')}")
        print(f"  任务串   : {self.task!r}")
        print(f"  图像     : {self.rgb.shape} uint8（{self.img_size}×{self.img_size}，"
              f"服务端 224→224 恒等，与原分辨率路径逐位相同）")
        print(f"  相机键   : {self.cam_base} / {self.cam_wrist}")
        assert self.dims == DIMS, f"包的维序 {self.dims} 与脚本的 {DIMS} 不一致"
        # 解卷绕自检：训练分布里 roll 不会跨 ±180
        roll = self.action_gt[:, 3]
        assert roll.min() > 90 or roll.max() < -90, \
            f"roll 区间 [{roll.min():.2f}, {roll.max():.2f}] 跨了 ±180，解卷绕没做对"
        print(f"  roll 区间: [{roll.min():.2f}, {roll.max():.2f}]  OK（未跨 ±180）")
        print(f"  state[0] : x={self.state[0, 0]:.1f} y={self.state[0, 1]:.1f} "
              f"z={self.state[0, 2]:.1f} mm | roll={self.state[0, 3]:.2f} "
              f"pitch={self.state[0, 4]:.2f} yaw={self.state[0, 5]:.2f} deg")


# ══════════════════════════════════════════════════════════════════════════
#  客户端
# ══════════════════════════════════════════════════════════════════════════
class LeanRobotClient:
    """只做回放：get_observation 从包里的第 t 帧取，send_action 只记录。"""

    def __init__(self, args, bundle: Bundle):
        self.args = args
        self.b = bundle
        self.fps = args.fps
        self.environment_dt = 1.0 / args.fps

        self.t = 0                 # 机器人时间轴：下一个要观测的帧号
        self.exhausted = threading.Event()
        self.performed = []        # 实际收到的动作
        self.performed_frames = []
        self.observed_frames = []

        self.action_queue = Queue()
        self.action_queue_lock = threading.Lock()
        self.latest_action_lock = threading.Lock()
        self.latest_action = -1
        self.action_chunk_size = args.actions_per_chunk
        self.action_queue_size = []
        self.received_chunks = 0
        self.rx_error = None
        self.must_go = threading.Event()
        self.must_go.set()
        self.shutdown_event = threading.Event()
        self.start_barrier = threading.Barrier(2)
        self.channel = None
        self.stub = None

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    # ---------- 握手 ----------
    def start(self) -> bool:
        from lerobot.async_inference.helpers import RemotePolicyConfig
        import services_pb2_grpc

        self.channel = grpc.insecure_channel(
            self.args.server, grpc_channel_options(f"{self.environment_dt:.4f}s")
        )
        try:
            grpc.channel_ready_future(self.channel).result(timeout=self.args.connect_timeout)
        except grpc.FutureTimeoutError:
            return False
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)

        # 顺序和 RobotClient 一致：先 Ready 再下发策略指令
        self.stub.Ready(services_pb2.Empty())

        self.policy_config = RemotePolicyConfig(
            policy_type="pi0",
            pretrained_name_or_path=self.args.ckpt,
            lerobot_features=make_lerobot_features(
                self.b.img_size, self.b.cam_base, self.b.cam_wrist
            ),
            actions_per_chunk=self.args.actions_per_chunk,
            device=self.args.policy_device,
            rename_map={},  # 空 = 让服务端沿用 ckpt 存档的那份（serve.py 差异 2）
        )
        try:
            self.stub.SendPolicyInstructions(
                services_pb2.PolicySetup(data=pickle.dumps(self.policy_config))
            )
        except grpc.RpcError as e:
            # 服务端加载失败（ckpt 路径不对 / adapter 没注入）会在这里变成 UNKNOWN
            print(f"\n服务端拒绝了策略指令：{e.code()} {e.details()}", file=sys.stderr)
            print("去跑 serve.py 的那个终端看真正的报错。", file=sys.stderr)
            return False
        self.shutdown_event.clear()
        return True

    def stop(self):
        self.shutdown_event.set()
        if self.channel is not None:
            self.channel.close()

    # ---------- 观测 ----------
    def get_observation(self) -> dict:
        k = self.t
        self.observed_frames.append(k)
        raw = {n: float(self.b.state[k][i]) for i, n in enumerate(DIMS)}
        raw[self.b.cam_base] = self.b.rgb[k]
        raw[self.b.cam_wrist] = self.b.us[k]
        return raw

    def send_observation(self, obs) -> bool:
        from lerobot.async_inference.helpers import TimedObservation

        if not isinstance(obs, TimedObservation):
            raise ValueError("obs 必须是 TimedObservation")
        try:
            self.stub.SendObservations(
                send_bytes_in_chunks(pickle.dumps(obs), services_pb2.Observation)
            )
            return True
        except grpc.RpcError as e:
            print(f"  !! 发送观测失败: {e}", file=sys.stderr)
            return False

    def _ready_to_send_observation(self) -> bool:
        # 回放跑完就不要再发观测了：一来没意义，二来 self.t 此时已经等于帧数，
        # get_observation 会越界（上游没有这个问题，因为真机的观测没有"尽头"）。
        if self.exhausted.is_set():
            return False
        with self.action_queue_lock:
            return self.action_queue.qsize() / self.action_chunk_size <= self.args.chunk_size_threshold

    # ---------- 动作 ----------
    def actions_available(self) -> bool:
        with self.action_queue_lock:
            return not self.action_queue.empty()

    def _pop_action(self):
        with self.action_queue_lock:
            return self.action_queue.get()

    def pop_action(self):
        timed = self._pop_action()
        action = timed.get_action()  # numpy 数组（服务端 --wire numpy）
        if not self.exhausted.is_set():
            self.performed.append(np.asarray(action, dtype=np.float64))
            self.performed_frames.append(self.t)
            self.t += 1
            if self.t >= self.b.n:
                self.exhausted.set()
        with self.latest_action_lock:
            self.latest_action = timed.get_timestep()
        return action

    # ---------- 两个线程 ----------
    def control_loop(self):
        self.start_barrier.wait()
        while self.running:
            t0 = time.perf_counter()
            if self.actions_available():
                self.pop_action()
            if self._ready_to_send_observation():
                self.control_loop_observation()
            with self.action_queue_lock:
                self.action_queue_size.append(self.action_queue.qsize())
            time.sleep(max(0, self.environment_dt - (time.perf_counter() - t0)))

    def control_loop_observation(self):
        from lerobot.async_inference.helpers import TimedObservation

        raw = self.get_observation()
        raw["task"] = self.b.task
        with self.latest_action_lock:
            latest = self.latest_action
        obs = TimedObservation(
            timestamp=time.time(),  # 必须是 time.time()，跨机器比时间戳
            observation=raw,
            timestep=max(latest, 0),
        )
        with self.action_queue_lock:
            obs.must_go = self.must_go.is_set() and self.action_queue.empty()
        self.send_observation(obs)
        if obs.must_go:
            self.must_go.clear()

    def receive_actions(self):
        self.start_barrier.wait()
        while self.running:
            try:
                chunk = self.stub.GetActions(services_pb2.Empty())
                if len(chunk.data) == 0:
                    continue  # 服务端超时返回了 Empty
                timed_actions = pickle.loads(chunk.data)  # nosec
            except grpc.RpcError as e:
                if self.running:
                    print(f"  !! 收动作失败: {e}", file=sys.stderr)
                break
            except Exception as e:
                # 这个线程一旦死掉，主线程只会看到"动作再也不来"，极难归因
                # （第一次调试时就是垫片少了 get_timestep 在这里崩的）。
                self.rx_error = f"{type(e).__name__}: {e}"
                print(f"  !! 接收线程异常退出: {self.rx_error}", file=sys.stderr)
                import traceback
                traceback.print_exc()
                self.shutdown_event.set()
                break

            self.action_chunk_size = max(self.action_chunk_size, len(timed_actions))

            with self.action_queue_lock:
                current = {a.get_timestep(): a.get_action() for a in self.action_queue.queue}
            future = Queue()
            for new in timed_actions:
                with self.latest_action_lock:
                    latest = self.latest_action
                if new.get_timestep() <= latest:
                    continue
                if new.get_timestep() not in current:
                    future.put(new)
                else:
                    # aggregate_fn = latest_only ⇒ 取后到的那个
                    future.put(type(new)(
                        timestamp=new.get_timestamp(),
                        timestep=new.get_timestep(),
                        action=new.get_action(),
                    ))
            with self.action_queue_lock:
                self.action_queue = future

            # 关键：上游 robot_client.py:337 就在这里 set。
            # 它不是"优化"，是**活性机制**：队列一空，本次 obs 必带 must_go=True，
            # 服务端 _enqueue_observation 无条件收下 → 必然再跑一次推理。
            # 少了这一句，后续 obs 全部 must_go=False，只能指望 _obs_sanity_checks
            # 放行；一旦 obs 的 timestep 落进服务端的 _predicted_timesteps 就会被
            # 静默丢弃，表现为"发了一堆观测，动作再也不来"。
            self.must_go.set()
            self.received_chunks += 1


# ══════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default="100.64.0.1:8080",
                    help="本机 Tailscale 地址:端口（即跑 serve.py 那台）")
    ap.add_argument("--bundle", default="replay_bundle.npz")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT,
                    help="**服务端本地**的 ckpt 路径，服务端拿它加载模型")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--actions-per-chunk", type=int, default=50)
    ap.add_argument("--chunk-size-threshold", type=float, default=0.5)
    ap.add_argument("--policy-device", default="cuda", help="仅作占位，服务端强制用 --device")
    ap.add_argument("--connect-timeout", type=float, default=10.0)
    ap.add_argument("--timeout", type=float, default=600, help="回放总超时")
    ap.add_argument("--dry-run", action="store_true", help="只加载包并打印，不连网")
    args = ap.parse_args()

    print("=" * 74)
    print("远程推理测试：本机(serve.py) <- gRPC -> 这台机器（无 torch / 无 lerobot）")
    print("=" * 74)

    print("\n【0】pickle 垫片自检")
    check_stubs_work()
    print("  OK：TimedObservation / TimedAction 能在无 lerobot 的进程里往返")
    print(f"  实际装了 lerobot 吗: {'是' if _has_lerobot() else '否'}")

    print("\n【1】加载回放包")
    b = Bundle(args.bundle)
    b.describe(args.bundle)

    if args.dry_run:
        print("\n--dry-run：到此为止")
        return

    print(f"\n【2】连接 {args.server}")
    client = LeanRobotClient(args, b)
    t0 = time.perf_counter()
    if not client.start():
        raise SystemExit(
            f"连不上 {args.server}\n"
            f"  - 那边起了吗：python serve.py --wire numpy --host 0.0.0.0 --port 8080\n"
            f"  - 网络通吗  ：nc -vz {args.server.split(':')[0]} {args.server.split(':')[1].split('/')[0]}\n"
            f"  - Tailscale ：tailscale status"
        )
    # 服务端在这里加载 LoRA（3B 模型，要几十秒），start() 返回只代表 RPC 发出去了
    print(f"  握手 + 策略指令已下发 {time.perf_counter() - t0:.1f}s"
          f"（服务端此刻正在加载模型，见那边的日志）")

    print(f"\n【3】回放 {b.n} 帧 @ {args.fps}fps（chunk={args.actions_per_chunk}）")
    rx = threading.Thread(target=client.receive_actions, daemon=True)
    rx.start()
    ctl = threading.Thread(target=client.control_loop, daemon=True)
    ctl.start()

    # 每 5 秒报一次进度：卡住时这几行能直接告诉你卡在"没收到 chunk"还是"没在发观测"
    stop_status = threading.Event()

    def status_loop():
        while not stop_status.wait(5.0):
            with client.action_queue_lock:
                q = client.action_queue.qsize()
            print(f"    [t={time.perf_counter() - t0:5.1f}s] 已执行 {len(client.performed)}/{b.n} "
                  f"| 队列 {q} | chunk {client.received_chunks} | "
                  f"已发观测 {len(client.observed_frames)} | latest_action {client.latest_action}",
                  flush=True)

    st = threading.Thread(target=status_loop, daemon=True)
    st.start()

    t0 = time.perf_counter()
    if not client.exhausted.wait(timeout=args.timeout):
        print(f"  !! 超时 {args.timeout}s，只执行了 {len(client.performed)} 步")
    wall = time.perf_counter() - t0
    stop_status.set()

    client.shutdown_event.set()
    ctl.join(timeout=3)
    rx.join(timeout=3)
    client.stop()
    ctl.join(timeout=5)
    rx.join(timeout=5)
    print(f"  回放结束，耗时 {wall:.1f}s")

    report(client, b, wall, args)


def _has_lerobot() -> bool:
    """垫片把 sys.modules 占住了，所以不能用 importlib.util.find_spec 判断。
    看有没有真的 dist-info 更靠谱。"""
    import importlib.metadata
    try:
        importlib.metadata.version("lerobot")
        return True
    except Exception:
        return False


def report(client, b: Bundle, wall: float, args):
    performed = np.array(client.performed)
    frames = np.array(client.performed_frames)
    if len(performed) == 0:
        if client.rx_error:
            raise SystemExit(f"接收线程崩了：{client.rx_error}（往上翻 traceback）")
        raise SystemExit(
            "一步都没收到 —— 服务端可能没返回动作。\n"
            "  看服务端日志：有没有 'adapter 已注入'？是不是忘了 --wire numpy？"
        )

    gt = b.action_gt[frames]
    err = np.abs(performed - gt)

    print("\n【4】时序")
    print(f"  下发动作步数 : {len(performed)} / {b.n}")
    print(f"  发出观测帧数 : {len(client.observed_frames)} "
          f"（占执行步数 {100 * len(client.observed_frames) / len(performed):.0f}%"
          f" —— 队列降到阈值以下才问策略，设计如此）")
    # 注意别把这个数当成相机帧率：客户端只在队列降到阈值以下才发观测，
    # 所以它天然远低于 fps。真正该看的是"实测执行频率"。
    print(f"  实测执行频率 : {len(performed) / max(wall, 1e-9):.2f} Hz（目标 {args.fps}）")
    print(f"  平均发观测率 : {len(client.observed_frames) / max(wall, 1e-9):.2f} Hz"
          f"（≠帧率，队列低才发，设计如此）")
    q = client.action_queue_size
    if q:
        # 队列深度贴着 0 说明推理跟不上；这里的每一帧都要过网络，比本机回放更吃时序
        print(f"  队列深度     : 最小 {min(q)} 最大 {max(q)}")

    print("\n【5】动作误差：收到的动作 vs 数据集真值（同一帧）")
    print(f"  {'维度':<12s} {'MAE':>9s} {'最大误差':>10s} {'该段 GT 行程':>13s} {'占行程':>8s}")
    for i, nm in enumerate(DIMS):
        span = float(gt[:, i].max() - gt[:, i].min())
        ratio = f"{100 * err[:, i].mean() / span:6.1f}%" if span >= 0.5 else "   n/a"
        print(f"  {nm:<12s} {err[:, i].mean():9.3f} {err[:, i].max():10.3f} "
              f"{span:13.3f} {ratio:>8s}")
    mae = float(err.mean())
    print(f"\n  整体 MAE = {mae:.3f}")
    print(f"  对照：本机 replay_client.py 端到端 {LOCAL_REPLAY_MAE:.3f}"
          f" | 离线 measure_inference.py {OFFLINE_MAE:.3f}")

    print("\n【6】解卷绕 / 量纲 sanity")
    for i, nm in ((0, "x_mm"), (3, "roll_deg"), (5, "yaw_deg")):
        p, g = performed[:, i], gt[:, i]
        print(f"  {nm:<9s} 下发 [{p.min():8.2f}, {p.max():8.2f}]   "
              f"GT [{g.min():8.2f}, {g.max():8.2f}]")

    print("\n【7】判定")
    if mae < 3.0:
        print(f"  ✅ 远程链路通，且契约对齐。MAE {mae:.3f}，与离线 {OFFLINE_MAE:.3f} 同量级")
        print("     这条链路上唯一和真机不同的就是图像预先缩过了（224→224 恒等，无偏差）")
    elif mae < 10.0:
        print(f"  ⚠️  MAE {mae:.3f} 明显高于离线 {OFFLINE_MAE:.3f} —— 链路通但口径有偏")
        print("     优先查：state 维序、相机键名、图像 HWC/CHW、roll 解卷绕")
    else:
        print(f"  ❌ MAE {mae:.3f} —— 契约没对齐，按这个顺序查：")
        print("     1. 服务端日志有没有 'adapter 已注入'（没有 = 加载成了随机初始化模型）")
        print("     2. state 维序是否 = x,y,z,roll,pitch,yaw")
        print("     3. 相机键名是否就是 base_0_rgb / left_wrist_0_rgb")
        print("     4. roll 有没有解卷绕")


if __name__ == "__main__":
    main()
