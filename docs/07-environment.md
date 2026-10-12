# 07 · 环境与工具链：若干消耗大量工时的配置问题

本篇的技术含量低于前几篇，但消耗的工时最多。记录于此，以备后续避免重复排查。

---

## 一、conda 环境污染（代价最高的一次）

### 经过

安装 openpi 时，`uv pip install` **写入了当时处于激活状态的 `lerobot` conda 环境**，
同时破坏了 `torch` / `transformers` / `lerobot` 三项依赖。

### 原因

`uv pip install` 在未显式指定 `--python` 时，会**优先安装至当前激活的 conda 环境**
（即 `CONDA_PREFIX` 所指向者），而非当前目录下的虚拟环境。
事发时 `CONDA_PREFIX` 正指向 lerobot 环境。

### 处理与预防

1. 执行任何 `uv pip` / `pip install` 之前，先确认 `CONDA_PREFIX`。
   无法确认时先执行 `conda deactivate`，或显式传入 `--python <目标解释器>`。
2. 训练脚本**不应依赖「当前激活的环境」**，而应以绝对路径固定解释器与入口。
   `run_train.sh` 中 `LEROBOT_BIN` 采用绝对路径，即源于此次事故。
3. 事后重建了环境，并在 `env_backup/` 保留了一份当时的 `pip freeze` 快照
   （`lerobot_env_AS_OF_2026-10-11_1122_openpi.txt`），便于追溯。

该条已纳入长期准则：**执行 `uv`/`pip` 前必须确认 `CONDA_PREFIX` 未指向 lerobot。**

---

## 二、torchcodec 失效，全面改用 pyav

### 现象

```
ImportError: undefined symbol: ffi_type_uint32, version LIBFFI_BASE_7.0
```

`lerobot-info` 与 `lerobot-dataset-viz` 无法启动。

### 原因

`torchcodec` 依赖的 `libffi` 版本与环境不一致（找不到 `LIBFFI_BASE_7.0` 符号版本），
属典型的动态链接库版本错配。

### 处理

不修复该问题，项目内**所有视频读取一律使用 `video_backend="pyav"`**。
pyav 为自带 ffmpeg 的绑定，不受系统 libffi 影响。

代价：所有脚本与数据集加载均须显式指定 `video_backend="pyav"`，
遗漏即回退至默认路径并崩溃。

### 结论

此类「系统库版本错配」问题，**绕过优于修复**：
更换不依赖该库的后端，其成本远低于调和两套 libffi，且不会因系统升级而复发。

---

## 三、代理：`ALL_PROXY` 的 scheme 导致 httpx 直接报错

环境中配置有 `ALL_PROXY=socks://127.0.0.1:7890/`。
httpx 遇到 `socks://` 这一 scheme 会**直接抛出异常**，而非降级处理。

正确配置：

```bash
unset ALL_PROXY                       # 必须移除，否则 httpx 报错
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
```

此外，**GitHub 使用代理，PyPI 使用清华镜像直连**：
pip 经代理会明显变慢，清华镜像直连更快。

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple <包名>
```

训练脚本中包含一行 `unset HTTP_PROXY HTTPS_PROXY ALL_PROXY ...`，即为此目的。

---

## 四、缺失依赖 `num2words`

### 现象

SmolVLA 冒烟测试报错：

```
ImportError: Package num2words is required to run SmolVLM processor
```

### 处理

安装 `num2words` 与 `docopt`，命令为：

```bash
pip install --no-deps -i https://pypi.tuna.tsinghua.edu.cn/simple num2words docopt
```

`--no-deps` 为有意添加，以避免再次将大量依赖写入环境（见本篇第一节）。
推测该包在 openpi 环境污染事故中丢失。

---

## 五、`~/.bashrc` 中的阻塞式 `read`

`~/.bashrc` 第 148–155 行含一段 fishros 的 ROS 环境选择器，
其中有一个**阻塞式 `read choose`**。

后果：**每个交互式 shell 均会阻塞于该处等待输入。**
通过 `tmux send-keys` 向会话发送命令时会卡住，因为 tmux 启动的 shell 是交互式的，
进入后即被该 `read` 拦截。

绕过方式是使用非交互式入口，其不读取 `.bashrc`：

```bash
bash script.sh                       # 非交互
tmux new-session -d <cmd>            # 非交互
```

### 结论

排查「命令已发出但无响应」时，**应优先怀疑 shell 启动脚本本身**，
而非检查命令书写。此类环境级阻塞会以「命令有误」的形式表现出来。
