# 10 · 两台机器之间的协作方式

本项目的代码天然分布在两台机器上：机械臂、相机与超声采集程序只能在工控机上开发，
而训练与推理只能在高显存的 GPU 机器上进行。本篇说明两端如何交换代码，
以及**哪些东西绝不能进仓库**。

---

## 一、分工

| 角色 | 主机 | 承担 |
|---|---|---|
| GPU 机器 | `gpu-server` | 数据集处理、训练、策略推理（`serve.py`） |
| 工控机 | `robot-pc` | xArm6 控制、RealSense 采集、PhaseLink 超声采集、真机执行 |

两侧共享的只有[文档 09](09-interface-contract.md) 所述的接口契约，不共享代码。

**两台机器上的 AI 会话互不可见。** 工控机侧的改动对 GPU 侧完全透明，
反之亦然——这正是[文档 09](09-interface-contract.md) 存在的原因：
契约必须落在仓库里，而不能依赖某一侧的上下文。

---

## 二、仓库布局

```text
vla-ultrasound/
├── tools/                  # GPU 侧：裁剪、统计量重算、校验
├── pi0_ultrasound_lora/    # GPU 侧：serve.py、训练脚本、远程客户端
├── robot/                  # 工控机侧：采集与控制（副本）
│   ├── my_code/            #   采集主链路与数据集构建
│   └── realsense_collector/#   独立 RealSense RGB-D 采集器
└── docs/                   # 两侧共同维护
```

`robot/my_code/` 已入库，是现行采集链路的 13 个模块与对应测试，
与工控机原件的差异只有两处替换（见 `robot/README.md` 第一节）。
`robot/realsense_collector/` 已入库，与原件逐字节相同（零脱敏）。

`robot/real_robot/` 尚未入库，见第七节。

**注意 `pi0_remote` 与 `pi0_ultrasound_lora/remote/` 是同一套东西的两端**
（服务端 / 客户端）。合并到一处时必须注明「两侧必须同版本」——
`services_pb2*.py` 是 proto 生成的桩，版本不一致不会报错，只会握手失败。

---

## 三、代码如何流动

GitHub 仓库作为**唯一的交换通道**。两端各自推送、各自拉取。

### 每台机器各自生成密钥对

```bash
ssh-keygen -t ed25519 -C "<主机名> -> github" -f ~/.ssh/id_ed25519_github
# 把 .pub 内容加到 https://github.com/settings/keys
ssh -T git@github.com        # 期望：Hi <用户名>! You've successfully authenticated
```

**私钥绝不跨机器复制。** 泄漏一台的私钥只影响一台；共用私钥则等于两台同时失守。
公钥可以随意分发，它本身不是秘密。

工控机是 Windows，注意两点：

- 管理员账户的 sshd 读取的是
  `C:\ProgramData\ssh\administrators_authorized_keys`，
  **不是** `%USERPROFILE%\.ssh\authorized_keys`。写错位置 sshd 会静默忽略。
- 该文件必须**只允许 `Administrators` 与 `SYSTEM`** 访问，
  且**不带 BOM**：

  ```powershell
  icacls "$env:ProgramData\ssh\administrators_authorized_keys" `
    /inheritance:r /grant "Administrators:F" /grant "SYSTEM:F"
  ```

### 提交者身份

当前提交使用 `Nash-tt@users.noreply.github.com`。
如需改为真实邮箱，在**每台机器**上分别设置：

```bash
git config --global user.name  "<名字>"
git config --global user.email "<邮箱>"
```

---

## 四、提交前消毒清单

GPU 侧的路径、主机名与内网地址在入库前统一替换为占位值。
现有规则如下，**新增脚本须一并套用**：

| 原文 | 替换为 |
|---|---|
| `/home/lab301/vla-data/` | `/data/vla/` |
| `/home/lab301/vla-training/` | 空（去掉前缀） |
| `/home/lab301/lerobot/` | `/opt/lerobot/` |
| `/home/lab301/miniconda3/envs/lerobot/bin/` | `/opt/conda/envs/lerobot/bin/` |
| `/home/lab301/` | `/home/user/` |
| GPU 机 Tailscale 地址（`100.127.*.*`） | `100.64.0.1` |
| 工控机 Tailscale 地址（`100.84.*.*`） | `100.64.0.2` |
| GPU 机主机名 | `gpu-server` |
| 工控机主机名 | `robot-pc` |
| 权重 snapshot 绝对路径 | HF repo id |

> 上面几行的原文只写成**可检索的前缀**（如 `100.127.`），不写全。
> 消毒时拿前缀 `grep -rn` 一遍即可定位；完整值不落盘是本清单能成立的前提。

另有两类需**改写**而非替换：

1. `sys.path.insert(0, "/绝对路径")` → 改为读环境变量：

   ```python
   sys.path.insert(0, os.environ.get("LEROBOT_SRC", "<默认值>"))
   ```

2. 写死的 `DATASET_ROOT=` → 改为可覆盖：

   ```bash
   DATASET_ROOT="${DATASET_ROOT:-/data/vla/<数据集名>}"
   ```

**工控机侧的路径同理**：`C:\Users\<用户名>\...`、`D:\...`、`192.168.1.209`（xArm）、
`192.168.50.205`（PhaseLink）均需替换或改为命令行参数。

### 绝不入库

| 项 | 原因 |
|---|---|
| `.codex/auth.json` | AI CLI 的**凭据文件** |
| `.codex/sessions/`、`archived_sessions/` | AI 会话记录，可能含代码与数据片段 |
| 在审论文的图表 | **学术优先权**，一旦公开难以彻底撤回 |
| 厂商 SDK（手册 PDF、DLL、`.exe`、`cs_phaselink.zip`） | 版权与体积 |
| 数据集与采集产物 | 体积；且是项目核心资产 |
| `*.npz`（回放包约 53 MB）、`*.safetensors`、`*.mp4`、`*.parquet` | 体积 |
| `.venv/`、`__pycache__/` | 可再生 |

`.gitignore` 已覆盖后四类。**前三类不在忽略规则内，只能靠提交时人工确认**——
尤其注意：在审论文的图表与采集代码位于同一台机器的同一棵目录树下，
粗粒度地「把整个目录加进来」会连它一起推上去。

### 提交前的机械检查

```bash
git diff --cached --name-only          # 逐个过一遍文件名
git diff --cached | grep -nE 'sk-|hf_|ghp_|AKIA|PRIVATE KEY|password|api_key|token'
```

对二进制文件（图片、视频）上述匹配是假阳性，只对文本文件采信。

---

## 五、文档约定

`docs/` 下每篇对应一个具体问题，统一按
**现象 → 原因 → 处理 → 验证 → 结论** 组织。
无法复现的现象也要记录，但要**明确标注为未解释**，避免日后被当作新问题重复排查
（[文档 06 第五节](06-remote-client.md)即为此例）。

写文档时的两条准则：

- **区分实测与推断。** 由公式相同而推出的结论，必须标注「未实测」。
  [文档 08](08-openpi-compat.md) 的 float32 一节即按此处理。
- **记录失败路径。** 导致方向错误的弯路（如 openpi 把「文件不存在」伪装成网络错误）
  与最终结论同样有价值。

---

## 六、两侧各自不可触碰的部分

| 侧 | 约束 |
|---|---|
| GPU | `/opt/lerobot/src/`（上游源码）不做无必要修改 |
| GPU | 执行 `uv pip` / `pip` 前必须确认 `CONDA_PREFIX` 未指向 `lerobot` 环境 |
| GPU | 代理：`GitHub` 走代理，`PyPI` 走清华镜像直连 |
| 工控机 | **`--execute` 相关的一切**：未经影子测试不得下发运动命令 |

---

## 七、待办

- [x] 建立 `robot/` 目录，明确 `my_code` 中脚本的取舍
      —— 已收入现行链路的 13 个模块 + 7 个测试 + 2 份流程说明 + 环境脚本；
      排除的 8 项及理由见 `robot/README.md` 第三节
- [x] 工控机侧的路径与内网地址按第四节规则消毒
      —— IP 与 SDK 绝对路径共两处替换，涉及 8 个文件
- [ ] 收入 `robot/remote_client.py`（`pi0_remote/`，即链路测试的瘦客户端），
      并与 `pi0_ultrasound_lora/remote/` 合并到一处；合并时必须注明「两侧同版本」，
      `services_pb2*.py` 版本不一致不会报错、只会握手失败
- [x] 收入 `robot/realsense_collector/`（独立 RealSense 采集器）
      —— 与原件逐字节相同，无需消毒
- [ ] `real_robot_client.py` 入库前**逐行审阅**，确认安全阈值
      是通用保守值而非实验室特定值；另需收录其依赖的 `export_ultrasound_images.py`
      （已随 `my_code` 入库）与 `remote_client.py`
- [ ] 决定提交者身份是否改用真实邮箱
