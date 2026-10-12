# 07 · 环境与工具链：几个把时间吃掉一半的坑

这一篇不如前面几篇「有技术含量」，但它们消耗掉的时间最多。
记下来是为了下一次不再重走。

---

## 一、conda 环境污染事故（最贵的一次）

### 发生了什么

装 openpi 的时候，`uv pip install` **灌进了当时处于激活状态的 `lerobot` conda 环境**，
把 `torch` / `transformers` / `lerobot` 三件套一起毁掉了。

### 根因

`uv pip install` 在没有显式指定 `--python` 时，会**优先装进当前激活的 conda 环境**
（即 `CONDA_PREFIX` 指向的那个），而不是当前目录的虚拟环境。

当时 `CONDA_PREFIX` 正好指向 lerobot 环境。

### 解决与预防

1. **跑任何 `uv pip` / `pip install` 之前，先确认 `CONDA_PREFIX`。**
   不放心就 `conda deactivate`，或者显式传 `--python <目标解释器>`。
2. 训练脚本里**不要依赖「当前激活的环境」**，而是用绝对路径写死解释器／入口：
   `run_train.sh` 里 `LEROBOT_BIN` 之所以是绝对路径，就是因为这次事故。
3. 事后重建了环境，并在 `env_backup/` 里留了一份当时的 `pip freeze` 快照
   （`lerobot_env_AS_OF_2026-10-11_1122_openpi.txt`），方便追溯。

**这条已经写进我的长期备忘：「别碰 conda lerobot 环境」——任何 `uv`/`pip`
之前必须确认 `CONDA_PREFIX` 没指向 lerobot。**

---

## 二、torchcodec 坏掉 → 全面改用 pyav

### 现象

```
ImportError: undefined symbol: ffi_type_uint32, version LIBFFI_BASE_7.0
```

`lerobot-info` 和 `lerobot-dataset-viz` 直接起不来。

### 根因

`torchcodec` 依赖的 `libffi` 版本与环境里的不一致（`LIBFFI_BASE_7.0` 这个符号版本找不到），
也就是典型的**动态链接库版本错配**。

### 解决

不修它——项目里**所有视频读取一律用 `video_backend="pyav"`**。
pyav 是纯 Python 打包的 ffmpeg 绑定，不受系统 libffi 影响。

**代价**：所有脚本、所有数据集加载都要显式写 `video_backend="pyav"`，
忘了就会掉回默认路径然后崩。

### 教训

这类「系统库版本错配」的问题，**修它不如绕过它**：
换一个不依赖该库的后端，比去调和两套 libffi 便宜得多，而且不会因为
系统升级再次复发。

---

## 三、代理：`ALL_PROXY` 的 scheme 会让 httpx 直接报错

环境里配了 `ALL_PROXY=socks://127.0.0.1:7890/`。
`httpx` 遇到 `socks://` 这个 scheme 会**直接抛错**，而不是优雅降级。

**正确组合：**

```bash
unset ALL_PROXY                       # 必须去掉，否则 httpx 报错
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
```

另外，**GitHub 走代理，PyPI 走清华镜像直连**——
pip 走代理会明显变慢，而清华镜像直连更快：

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple <包名>
```

训练脚本里有一行 `unset HTTP_PROXY HTTPS_PROXY ALL_PROXY ...`，
就是为了避免代理干扰。

---

## 四、缺失的依赖：`num2words`

SmolVLA 冒烟测试时报：

```
ImportError: Package num2words is required to run SmolVLM processor
```

装 `num2words` + `docopt` 解决。**注意用的命令是：**

```bash
pip install --no-deps -i https://pypi.tuna.tsinghua.edu.cn/simple num2words docopt
```

`--no-deps` 是刻意的——避免再次把一堆依赖灌进环境（见上面第一节）。

推测这个包是**在 openpi 那次环境污染事故中被弄丢的**。

---

## 五、`~/.bashrc` 里有个阻塞的 `read`

`~/.bashrc` 第 148–155 行有一段 fishros 的 ROS 环境选择器，
里面有个**阻塞的 `read choose`**。

后果：**每一个交互式 shell 都会卡在那里等输入。**
我们通过 `tmux send-keys` 往会话里发命令时会卡住——
因为 tmux 起的 shell 是交互式的，一进去就被那个 `read` 拦住。

**绕过方式**：用非交互式入口，它们不读 `.bashrc`：

```bash
bash script.sh                       # 非交互
tmux new-session -d <cmd>            # 非交互
```

### 教训

排查「命令发出去了但没反应」时，**先怀疑 shell 启动脚本本身**，
而不是去查命令写错没有。这类环境级的阻塞会伪装成「命令有问题」。
