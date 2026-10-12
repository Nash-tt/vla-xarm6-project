#!/usr/bin/env python3
"""裸 TCP 吞吐探针：只测链路，不碰 gRPC / torch / lerobot。

用途：remote_client.py 慢的时候，先分清是"链路本身就慢"还是"gRPC 没谈好流控"。
只用标准库，对面只要有 python3 就能跑。

服务端（本机）：
    python3 netprobe.py recv --host 100.64.0.1 --port 9099
客户端（另一台机器）：
    python3 netprobe.py send --host 100.64.0.1 --port 9099 --mb 5

对照：参考值 ≈ 100 Mbps 以上算正常；只有几 Mbps 就是链路问题。
"""

import argparse
import socket
import sys
import time

BLOCK = 64 * 1024


def _rate(nbytes: int, secs: float) -> str:
    if secs <= 0:
        return "n/a"
    mbps = nbytes * 8 / secs / 1e6
    mbs = nbytes / secs / 1e6
    return f"{mbs:8.2f} MB/s  ({mbps:7.1f} Mbps)"


def _path_label(ip: str) -> str:
    """从对端源地址判断这次走的是哪条路。"""
    if ip.startswith("100."):
        return "Tailscale 路径"
    return "LAN 直连（绕开 Tailscale）"


def do_recv(args) -> int:
    """连续接多次：测 Tailscale 路径和 LAN 直连路径可以共用这一个进程。"""
    deadline = time.perf_counter() + args.timeout
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(4)
    print(
        f"[recv] 监听 {args.host}:{args.port}，{args.timeout:.0f}s 内可连续接多次……",
        flush=True,
    )

    n = 0
    while True:
        left = deadline - time.perf_counter()
        if left <= 0:
            break
        srv.settimeout(left)
        try:
            conn, peer = srv.accept()
        except socket.timeout:
            break
        except OSError as e:
            print(f"[recv] accept 失败：{type(e).__name__}: {e}", file=sys.stderr)
            break

        n += 1
        print(f"\n[recv #{n}] 来自 {peer[0]}:{peer[1]}  →  {_path_label(peer[0])}", flush=True)
        conn.settimeout(max(5.0, deadline - time.perf_counter()))
        total = 0
        t0 = time.perf_counter()
        note = ""
        try:
            while True:
                buf = conn.recv(BLOCK * 4)
                if not buf:
                    break
                total += len(buf)
        except socket.timeout:
            note = "读超时，对面可能卡住了"
        except OSError as e:
            # 对面中途 RST、进程被杀、或中间设备重置连接。
            # 端口探测（Test-NetConnection 之类）就会这样，不能让整个接收端跟着死。
            note = f"{type(e).__name__}: {e}"
        wall = time.perf_counter() - t0
        try:
            conn.close()
        except OSError:
            pass

        tail = f"   ⚠ {note}" if note else ""
        print(f"[recv #{n}] 收到 {total} 字节，用时 {wall:.2f}s{tail}")
        if total:
            print(f"[recv #{n}] 吞吐 {_rate(total, wall)}", flush=True)
        elif note:
            print(f"[recv #{n}] 没拿到数据，继续等下一个连接……", flush=True)

    srv.close()
    if n == 0:
        print(f"[recv] {args.timeout:.0f}s 内没人来，退出", file=sys.stderr)
        return 1
    print(f"\n[recv] 共测了 {n} 次，退出")
    return 0


def do_send(args) -> int:
    total_target = args.mb * 1024 * 1024
    payload = b"\0" * BLOCK

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(args.timeout)
    t_conn = time.perf_counter()
    try:
        s.connect((args.host, args.port))
    except OSError as e:
        print(f"[send] 连不上 {args.host}:{args.port} —— {e}", file=sys.stderr)
        return 1
    print(f"[send] 已连接（建连 {(time.perf_counter() - t_conn) * 1e3:.1f}ms），"
          f"发 {args.mb} MB……", flush=True)

    sent = 0
    t0 = time.perf_counter()
    try:
        while sent < total_target:
            n = min(BLOCK, total_target - sent)
            s.sendall(payload[:n])
            sent += n
    except OSError as e:
        print(f"[send] 发到一半断了：{e}", file=sys.stderr)
    wall = time.perf_counter() - t0
    s.shutdown(socket.SHUT_WR)  # 告诉对面发完了
    s.close()

    print(f"[send] 发出 {sent} 字节，用时 {wall:.2f}s")
    print(f"[send] 吞吐 {_rate(sent, wall)}")
    print()
    print("  对照：>=100 Mbps 说明链路正常，慢的在别处；")
    print("        只有几 Mbps 就是这条链路的问题（多半是 Wi-Fi 信号/功率管理）。")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="裸 TCP 吞吐探针（只用标准库）")
    sub = ap.add_subparsers(dest="mode", required=True)

    r = sub.add_parser("recv", help="本机：收")
    r.add_argument("--host", default="0.0.0.0")
    r.add_argument("--port", type=int, default=9099)
    r.add_argument("--timeout", type=float, default=120.0)

    s = sub.add_parser("send", help="对面：发")
    s.add_argument("--host", required=True)
    s.add_argument("--port", type=int, default=9099)
    s.add_argument("--mb", type=int, default=5, help="发多少 MB（默认 5）")
    s.add_argument("--timeout", type=float, default=120.0)

    args = ap.parse_args()
    return do_recv(args) if args.mode == "recv" else do_send(args)


if __name__ == "__main__":
    sys.exit(main())
