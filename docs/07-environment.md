# 07 · 环境与工具链：若干消耗大量工时的配置与工具问题

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

---

## 六、`pgrep -f` 匹配到自己的命令行，队列静默等 12 小时

### 现象

ACT 训练于 00:47:44 正常结束（日志有 `End of training`、checkpoint 到 `020000`），
但**训练队列没有接着跑下一段，也没有报错**：GPU 占用停在 585 MiB，队列进程活着、
反复 `sleep 60`，日志停在「等待 ACT ……」——整整 12 小时。第二段的日志文件
**根本没有被创建**，即它从未被拉起。

### 原因

队列用 `pgrep -f "$pattern"` 判断「训练还在不在」。为了看进度，当时另挂了一个后台监控，
其命令行里恰好**带着同一个字面量**：

```bash
while pgrep -f "output_dir=…/act_ultrasound_scan_20k" >/dev/null; do sleep 60; done
```

`pgrep -f` 匹配的是**整条命令行**，它不区分这条命令行是在「查询」这个串，还是「恰好含有」
这个串。于是：

1. 监控匹配到**自己**，条件恒为真 —— 它永不退出，哪怕 ACT 早已结束；
2. 队列的等待函数也匹配到这个监控 —— 它据此认为训练仍在进行，继续 waiting。

两个看守互相看住，而真正在训练的进程一个都没有。**全程没有报错**，
因为从工具的角度看，每一条判断都"成立"。

### 处理

1. 匹配串与进程名同时出现：`lerobot-train.*output_dir=…`。旁观者只带 `output_dir`、
   不带 `lerobot-train`，不再匹配。
2. 拉起后 45 秒复查「进程在、且日志非空」，不满足即打印日志尾部并**终止队列**——
   避免把一段没跑起来的训练当成跑完而静默跳过。这条守卫当天就拦下了一次真实的启动失败。
3. 自己临时写 watcher 时必须用 `-f` 时，把匹配串拆开拼接，使它不出现在自己的命令行里：

   ```bash
   P="chain_train_""queue.sh"     # ps 看到的命令行是 chain_train_""queue.sh
   until ! pgrep -f "$P" >/dev/null 2>&1; do sleep 120; done
   ```

### 结论

**`pgrep -f` / `pkill -f` 的匹配串绝不能出现在自己的命令行里。**
这类匹配是「文本巧合」而非「身份判断」：同一个串既是「我要找的东西」又是「我自己的名字」时，
就变成自锁，而且**不报错**——最坏的一种失败。

同日还有一次更直接的版本：`pkill -f chain_smolvla_after_act` 把**执行它的那个 shell**
一并杀掉（exit 144），因为 `-f` 匹配到了承载该命令的 bash。判断进程身份应优先用 PID；
确需按名字匹配时，加上只有目标才有的第二个特征，并复核一次「这个串会不会出现在我自己身上」。

与第五节合并来看，「命令已发出但无响应」的排查顺序应是：
**先确认是不是在等一个永远不会满足的条件，再怀疑命令本身。**
