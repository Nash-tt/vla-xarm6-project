#!/usr/bin/env python3
"""Read-only xArm force/torque sensor diagnostic.

This utility deliberately does not enable motion, change robot mode/state,
enable/disable the FT sensor, zero the sensor, or issue any motion command.
"""

import argparse
import math
import os
import statistics
import sys
import time

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from xarm.wrapper import XArmAPI


def vector6(value):
    if not isinstance(value, (list, tuple)) or len(value) < 6:
        return None
    result = [float(item) for item in value[:6]]
    return result if all(math.isfinite(item) for item in result) else None


def summarize(name, values):
    if not values:
        print("{}: no valid samples".format(name))
        return False
    columns = list(zip(*values))
    means = [statistics.fmean(column) for column in columns]
    ranges = [max(column) - min(column) for column in columns]
    maximum = max(abs(item) for row in values for item in row)
    print("{} sample count: {}".format(name, len(values)))
    print("{} mean [Fx,Fy,Fz,Tx,Ty,Tz]: {}".format(
        name, [round(item, 6) for item in means]
    ))
    print("{} axis range: {}".format(name, [round(item, 6) for item in ranges]))
    print("{} maximum absolute value: {:.6f}".format(name, maximum))
    return maximum > 1e-6


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-ip", default="192.168.1.100")
    parser.add_argument("--report-type", choices=("real", "rich"), default="real")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--interval", type=float, default=0.2)
    args = parser.parse_args()
    if args.duration <= 0 or args.interval <= 0:
        raise ValueError("duration and interval must be positive")

    print("READ-ONLY force diagnostic")
    print("Robot IP:", args.robot_ip)
    print("Report type:", args.report_type)
    print("No motion/configuration commands will be sent.")

    arm = XArmAPI(
        args.robot_ip,
        enable_report=True,
        report_type=args.report_type,
    )
    direct_ext = []
    direct_raw = []
    report_ext = []
    report_raw = []
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not (
            arm.connected and getattr(arm._arm, "reported", False)
        ):
            time.sleep(0.05)
        print(
            "Connected:",
            arm.connected,
            "Reported:",
            getattr(arm._arm, "reported", False),
        )
        if not arm.connected:
            raise RuntimeError("failed to connect to xArm")

        config_code, config = arm.get_ft_sensor_config()
        print("get_ft_sensor_config code:", config_code)
        if config_code == 0 and len(config) >= 5:
            print("FT mode:", config[0])
            print("FT started/enabled:", config[1])
            print("FT type:", config[2])
            print("FT id:", config[3])
            print("FT frequency:", config[4])
        error_code, sensor_error = arm.get_ft_sensor_error()
        print("get_ft_sensor_error code:", error_code, "sensor error:", sensor_error)

        end_time = time.monotonic() + args.duration
        index = 0
        while time.monotonic() < end_time:
            ext_code, ext_value = arm.get_ft_sensor_data(is_raw=False)
            raw_code, raw_value = arm.get_ft_sensor_data(is_raw=True)
            ext = vector6(ext_value) if ext_code == 0 else None
            raw = vector6(raw_value) if raw_code == 0 else None
            rep_ext = vector6(arm.ft_ext_force)
            rep_raw = vector6(arm.ft_raw_force)
            if ext is not None:
                direct_ext.append(ext)
            if raw is not None:
                direct_raw.append(raw)
            if rep_ext is not None:
                report_ext.append(rep_ext)
            if rep_raw is not None:
                report_raw.append(rep_raw)
            if index % max(1, int(round(1.0 / args.interval))) == 0:
                print(
                    "sample {} direct_ext={} direct_raw={} report_ext={} report_raw={}".format(
                        index, ext, raw, rep_ext, rep_raw
                    )
                )
            index += 1
            time.sleep(args.interval)

        print("\nSummary (forces N, torques N*m):")
        direct_ext_nonzero = summarize("direct external", direct_ext)
        direct_raw_nonzero = summarize("direct raw", direct_raw)
        report_ext_nonzero = summarize("report external", report_ext)
        report_raw_nonzero = summarize("report raw", report_raw)
        print("\nDirect sensor values nonzero:", direct_ext_nonzero or direct_raw_nonzero)
        print("Collection report values nonzero:", report_ext_nonzero or report_raw_nonzero)
        if direct_ext_nonzero or direct_raw_nonzero:
            if report_ext_nonzero or report_raw_nonzero:
                print("RESULT: PASS - force values are available in collection reports.")
            else:
                print("RESULT: PARTIAL - sensor reads directly, but selected report type is still zero.")
        else:
            print("RESULT: FAIL - direct and report force values are all zero.")
    finally:
        arm.disconnect()


if __name__ == "__main__":
    main()
