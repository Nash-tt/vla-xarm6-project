#!/usr/bin/env python
"""远程推理服务端：加载 LoRA 微调后的 pi0，通过 gRPC 下发动作 chunk。

架构：这台机器只做推理（policy server），另一台机器驱动机械臂（robot client）。
LeRobot 自带这套 gRPC 协议，实现见 `lerobot/async_inference/`。

本文件和上游 `lerobot.async_inference.policy_server` 只有**两处**差异，但两处都是
上游在该场景下的静默 bug：

── 差异 1（致命）：加载 LoRA ────────────────────────────────────────────────
上游 `policy_server.py:152` 是：

    self.policy = policy_class.from_pretrained(policy_specs.pretrained_name_or_path)

而 `PI0Policy.from_pretrained` 只找 `model.safetensors`，LoRA checkpoint 里只有
`adapter_model.safetensors`。**找不到时它不报错**，只打印一句
"Returning model without loading pretrained weights"，然后返回一个随机初始化的
模型（measure_inference.py 里实测过：MAE 52.7，真实是 1.19）。后果是服务端会
一本正经地给真机下发垃圾动作。

正确入口是 `policies/factory.py:565` 那段 PEFT 分支，这里复刻它的三行。

── 差异 2：rename_map 被清空 ────────────────────────────────────────────────
上游 `policy_server.py:162` 无条件用 client 传来的 rename_map 覆盖 checkpoint
存档的那一份：

    preprocessor_overrides={"rename_observations_processor": {"rename_map": policy_specs.rename_map}}

但 `RobotClient` 构造 `RemotePolicyConfig` 时是**位置传参**（robot_client.py:103），
压根没传 rename_map，于是恒为 `{}`。结果是 ckpt 里存的
`{"observation.images.rgb": "observation.images.base_0_rgb", ...}` 被清空。
这里改成：client 明确传了才覆盖，否则保留 checkpoint 自带的。

用法：
    python serve.py --ckpt <LoRA ckpt 目录> [--host 0.0.0.0] [--port 8080] [--device cuda]
"""

import argparse
import logging
import os
import pickle  # nosec
import time
from concurrent import futures

# 必须在 import lerobot 之前：socks:// 这个 scheme httpx 不认，会让 HF 请求直接报错。
# 注意只删 ALL_PROXY，HTTP(S)_PROXY 要留着（本机直连 HF 不通）。
for _v in ("ALL_PROXY", "all_proxy"):
    os.environ.pop(_v, None)
# 服务端要的是确定性：底座从本地 HF 缓存读，不联网。想允许联网就 export HF_HUB_OFFLINE=0
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import grpc  # noqa: E402
import numpy as np  # noqa: E402

from lerobot.async_inference.configs import PolicyServerConfig  # noqa: E402
from lerobot.async_inference.constants import SUPPORTED_POLICIES  # noqa: E402
from lerobot.async_inference.helpers import RemotePolicyConfig  # noqa: E402
from lerobot.async_inference.policy_server import PolicyServer  # noqa: E402
from lerobot.policies import get_policy_class, make_pre_post_processors  # noqa: E402
from lerobot.transport import services_pb2, services_pb2_grpc  # noqa: E402

log = logging.getLogger("policy_server")

DEFAULT_CKPT = (
    "pi0_ultrasound_lora/run1/checkpoints/012250/pretrained_model"
)


def load_policy_checked(path: str, policy_type: str):
    """加载 checkpoint，并**硬校验** adapter 真的进来了。

    非 PEFT 的 checkpoint（全量微调 / 官方发布）沿用上游逻辑。
    """
    adapter_file = os.path.join(path, "adapter_model.safetensors")
    if not os.path.exists(adapter_file):
        log.info("没有 adapter_model.safetensors，按普通 checkpoint 加载：%s", path)
        return get_policy_class(policy_type).from_pretrained(path)

    # --- 以下复刻 policies/factory.py:565-589 的 PEFT 分支（三行核心） ---
    import lerobot.policies.pi0.configuration_pi0  # noqa: F401  注册 'pi0' 子类
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy
    from peft import PeftConfig, PeftModel

    peft_cfg = PeftConfig.from_pretrained(path)
    base = peft_cfg.base_model_name_or_path
    if not base:
        raise RuntimeError(f"{path}/adapter_config.json 里没有 base_model_name_or_path")

    # config 必须取自 **checkpoint**，且必须用**基类**读：ckpt 的 config.json 带
    # "type":"pi0"，直接调 PI0Config.from_pretrained 会因 draccus 解不了判别字段而报错。
    cfg = PreTrainedConfig.from_pretrained(path)
    log.info("底座：%s", base)
    policy = PI0Policy.from_pretrained(base, config=cfg)
    policy = PeftModel.from_pretrained(policy, path, config=peft_cfg, is_trainable=False)
    inner = policy.get_base_model()  # LoRA 层是就地注入的，取回底座即可直接用

    # 把上游的静默失败变成硬报错。PEFT 初始化时 lora_B 恒为 0，所以读进来还全零
    # ⇒ adapter 根本没生效 ⇒ 模型等价于未微调。宁可起不来，也不要给真机发垃圾动作。
    norms = [
        (n, p.detach().float().norm().item())
        for n, p in inner.named_parameters()
        if "lora_B" in n
    ]
    if not norms:
        raise RuntimeError(f"{path} 是 PEFT ckpt，但模型里找不到任何 lora_B 层 —— adapter 没注入")
    zeros = [n for n, v in norms if v == 0.0]
    if len(zeros) == len(norms):
        raise RuntimeError(
            f"全部 {len(norms)} 个 lora_B 都是 0 —— adapter 权重没加载进来。"
            "此时模型等价于未微调，绝不能拿去控真机。"
        )
    vals = [v for _, v in norms]
    log.info(
        "adapter 已注入：%d 个 lora_B，范数 [%.3f, %.3f]，全零 %d 个",
        len(norms), min(vals), max(vals), len(zeros),
    )
    return inner


class LoraPolicyServer(PolicyServer):
    """和上游只差上面那两处。其余（gRPC、观测队列、FPS 统计）全部继承。"""

    def __init__(self, config: PolicyServerConfig, device: str | None = None, wire: str = "torch"):
        super().__init__(config)
        # 设备由服务端说了算：不信 client 传的 —— 对面要是传个 cpu，
        # 3B 模型一帧要跑几十秒，表现成"卡死"而不是报错。
        self.forced_device = device
        self.wire = wire

    def _time_action_chunk(self, t_0, action_chunk, i_0):
        """wire=numpy 时把回传动作从 torch.Tensor 换成 numpy 数组。

        上游默认（wire=torch）每个 TimedAction.action 都是 torch.Tensor，对面
        unpickle 时**必须有 torch**（2GB 起）。换 numpy 后对面只要 numpy，
        代价是这条路不再是真机的字节格式 —— 只用于远程连通性测试。
        """
        if self.wire == "numpy":
            action_chunk = [
                a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)
                for a in action_chunk
            ]
        return super()._time_action_chunk(t_0, action_chunk, i_0)

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        if not self.running:
            self.logger.warning("Server is not running. Ignoring policy instructions.")
            return services_pb2.Empty()

        client_id = context.peer()
        policy_specs = pickle.loads(request.data)  # nosec

        if not isinstance(policy_specs, RemotePolicyConfig):
            raise TypeError(f"Policy specs must be a RemotePolicyConfig. Got {type(policy_specs)}")
        if policy_specs.policy_type not in SUPPORTED_POLICIES:
            raise ValueError(
                f"Policy type {policy_specs.policy_type} not supported. "
                f"Supported policies: {SUPPORTED_POLICIES}"
            )

        self.policy_type = policy_specs.policy_type
        self.lerobot_features = policy_specs.lerobot_features
        self.actions_per_chunk = policy_specs.actions_per_chunk
        self.device = self.forced_device or policy_specs.device

        path = policy_specs.pretrained_name_or_path
        self.logger.info(
            f"Client {client_id} | type={self.policy_type} | ckpt={path} | "
            f"actions_per_chunk={self.actions_per_chunk} | device={self.device}"
        )

        start = time.perf_counter()
        # ---- 差异 1 ----
        self.policy = load_policy_checked(path, self.policy_type)
        self.policy.eval()
        self.policy.to(self.device)

        # ---- 差异 2 ----
        pre_overrides = {"device_processor": {"device": self.device}}
        if policy_specs.rename_map:
            self.logger.info("client 指定了 rename_map，覆盖 ckpt 存档值：%s", policy_specs.rename_map)
            pre_overrides["rename_observations_processor"] = {"rename_map": policy_specs.rename_map}
        else:
            self.logger.info("client 未传 rename_map，沿用 checkpoint 存档的那份")

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=path,
            preprocessor_overrides=pre_overrides,
            postprocessor_overrides={"device_processor": {"device": self.device}},
        )

        self.logger.info(
            f"策略就绪，耗时 {time.perf_counter() - start:.1f}s | "
            f"图像键 {sorted(self.policy_image_features)} | "
            f"state {self.policy.config.input_features.get('observation.state')} | "
            f"action {self.policy.config.output_features}"
        )
        return services_pb2.Empty()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT, help="LoRA checkpoint 目录（含 adapter_model.safetensors）")
    ap.add_argument("--host", default="0.0.0.0", help="监听地址，远程部署用 0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--device", default="cuda", help="推理设备；写死不给 client 覆盖的机会")
    ap.add_argument("--fps", type=int, default=15, help="数据集就是 15fps，别用默认的 30")
    ap.add_argument("--obs-queue-timeout", type=float, default=2.0)
    ap.add_argument(
        "--wire",
        choices=("torch", "numpy"),
        default="torch",
        help="回传动作的类型。torch=上游原格式（真机 RobotClient 用）；"
        "numpy=回传 numpy 数组，让没装 torch 的瘦客户端也能收（见 remote/）",
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    # 先加载再起服务：宁可启动失败，也不要让 client 连上一个没加载好策略的空壳。
    server_cfg = PolicyServerConfig(
        host=args.host, port=args.port, fps=args.fps, obs_queue_timeout=args.obs_queue_timeout
    )
    policy_server = LoraPolicyServer(server_cfg, device=args.device, wire=args.wire)

    if not os.path.exists(args.ckpt):
        raise SystemExit(f"checkpoint 不存在：{args.ckpt}")

    grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, grpc_server)
    bound = grpc_server.add_insecure_port(f"{args.host}:{args.port}")
    if bound == 0:
        raise SystemExit(f"端口绑定失败：{args.host}:{args.port}")

    log.info(
        "PolicyServer 已启动 %s:%d（fps=%d, device=%s, wire=%s）",
        args.host, args.port, args.fps, args.device, args.wire,
    )
    log.info("等待 client 连接并下发策略指令……")
    grpc_server.start()
    try:
        grpc_server.wait_for_termination()
    except KeyboardInterrupt:
        log.info("收到中断，正在停止……")
        grpc_server.stop(grace=2.0)
    log.info("服务端已退出")


if __name__ == "__main__":
    main()
