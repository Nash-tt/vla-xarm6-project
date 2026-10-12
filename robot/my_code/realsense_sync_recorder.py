"""RealSense RGB-D recorder with device and host timestamps."""

import json
import os
import queue
import threading
import time


# One 1280x720 BGR8 + Z16 pair occupies about 4.4 MiB before encoding.
# Keeping 120 raw pairs could retain more than 500 MiB and starve the
# acquisition thread.  A short bounded queue absorbs normal disk jitter while
# keeping memory use predictable.
FRAME_QUEUE_CAPACITY = 16
PREVIEW_QUEUE_CAPACITY = 1
DEPTH_PNG_COMPRESSION = 1


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def intrinsics_dict(profile):
    intrinsics = profile.get_intrinsics()
    return {
        "width": intrinsics.width,
        "height": intrinsics.height,
        "fx": intrinsics.fx,
        "fy": intrinsics.fy,
        "ppx": intrinsics.ppx,
        "ppy": intrinsics.ppy,
        "model": str(intrinsics.model),
        "coeffs": list(intrinsics.coeffs),
    }


def device_info(rs, device):
    fields = {
        "name": rs.camera_info.name,
        "serial_number": rs.camera_info.serial_number,
        "firmware_version": rs.camera_info.firmware_version,
        "product_line": rs.camera_info.product_line,
        "usb_type": rs.camera_info.usb_type_descriptor,
    }
    result = {}
    for name, field in fields.items():
        try:
            if device.supports(field):
                result[name] = device.get_info(field)
        except Exception:
            pass
    return result


class RealSenseRecorder(threading.Thread):
    def __init__(
        self,
        episode_dir,
        stop_event,
        width=1280,
        height=720,
        fps=30,
        warmup_frames=30,
        jpeg_quality=95,
        color_only=False,
        serial="",
        queue_capacity=FRAME_QUEUE_CAPACITY,
        depth_png_compression=DEPTH_PNG_COMPRESSION,
        preview=False,
        preview_window_name="RealSense RGB Preview",
    ):
        super().__init__(daemon=True)
        self.episode_dir = episode_dir
        self.stop_event = stop_event
        self.width = width
        self.height = height
        self.fps = fps
        self.warmup_frames = warmup_frames
        self.jpeg_quality = jpeg_quality
        self.color_only = color_only
        self.serial = serial
        self.queue_capacity = queue_capacity
        self.depth_png_compression = depth_png_compression
        self.preview = preview
        self.preview_window_name = preview_window_name
        self.color_dir = os.path.join(episode_dir, "color")
        self.depth_dir = os.path.join(episode_dir, "depth")
        self.frames_path = os.path.join(episode_dir, "realsense_frames.jsonl")
        self.metadata_path = os.path.join(episode_dir, "realsense_metadata.json")
        self.started_event = threading.Event()
        self.error = None
        self.frame_count = 0
        self.received_count = 0
        self.dropped_count = 0
        self.writer_error = None
        self.preview_error = None
        self.preview_started_event = threading.Event()
        self.preview_closed_event = threading.Event()
        self.preview_closed_by_user = False

    def _writer(self, frame_queue, stop_token, cv2):
        try:
            with open(self.frames_path, "w", encoding="utf-8", buffering=1) as frames_file:
                while True:
                    item = frame_queue.get()
                    try:
                        if item is stop_token:
                            return

                        record, color, depth = item
                        color_path = os.path.join(
                            self.episode_dir, record["color_path"]
                        )
                        if not cv2.imwrite(
                            color_path,
                            color,
                            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
                        ):
                            raise RuntimeError(
                                "Failed to write {}".format(color_path)
                            )

                        if depth is not None:
                            depth_path = os.path.join(
                                self.episode_dir, record["depth_path"]
                            )
                            if not cv2.imwrite(
                                depth_path,
                                depth,
                                [
                                    cv2.IMWRITE_PNG_COMPRESSION,
                                    self.depth_png_compression,
                                ],
                            ):
                                raise RuntimeError(
                                    "Failed to write {}".format(depth_path)
                                )

                        frames_file.write(
                            json.dumps(
                                record,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            )
                            + "\n"
                        )
                        self.frame_count += 1
                    finally:
                        frame_queue.task_done()
        except Exception as exc:
            self.writer_error = exc

    def _preview(self, preview_queue, stop_token, cv2):
        try:
            cv2.namedWindow(self.preview_window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.preview_window_name, self.width, self.height)
            self.preview_started_event.set()
            while True:
                try:
                    item = preview_queue.get(timeout=0.05)
                except queue.Empty:
                    item = None

                if item is stop_token:
                    return
                if item is not None:
                    # Draw only on a preview copy so saved JPEG data stays raw.
                    display = item.copy()
                    cv2.putText(
                        display,
                        "RealSense live preview - Q/ESC closes preview only",
                        (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )
                    cv2.imshow(self.preview_window_name, display)

                key = cv2.waitKey(1) & 0xFF
                try:
                    window_visible = cv2.getWindowProperty(
                        self.preview_window_name, cv2.WND_PROP_VISIBLE
                    ) >= 1
                except Exception:
                    window_visible = False
                if key in (27, ord("q"), ord("Q")) or not window_visible:
                    self.preview_closed_by_user = True
                    print(
                        "RealSense preview closed; RGB-D recording continues."
                    )
                    return
        except Exception as exc:
            self.preview_error = exc
            self.preview_started_event.set()
        finally:
            self.preview_closed_event.set()
            try:
                cv2.destroyWindow(self.preview_window_name)
            except Exception:
                pass

    @staticmethod
    def _offer_latest(preview_queue, frame):
        """Replace a stale preview frame without blocking acquisition."""
        try:
            preview_queue.put_nowait(frame)
            return
        except queue.Full:
            pass
        try:
            preview_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            preview_queue.put_nowait(frame)
        except queue.Full:
            pass

    def run(self):
        try:
            self._run()
        except Exception as exc:
            self.error = exc
            self.started_event.set()

    def _run(self):
        import cv2
        import numpy as np
        import pyrealsense2 as rs

        ensure_dir(self.color_dir)
        if not self.color_only:
            ensure_dir(self.depth_dir)

        devices = rs.context().query_devices()
        if not devices:
            raise RuntimeError("No RealSense camera found")
        selected = None
        if self.serial:
            for device in devices:
                if device.get_info(rs.camera_info.serial_number) == self.serial:
                    selected = device
                    break
            if selected is None:
                raise RuntimeError(
                    "RealSense serial {} was not found".format(self.serial)
                )
        else:
            selected = devices[0]
        serial = selected.get_info(rs.camera_info.serial_number)

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(
            rs.stream.color,
            self.width,
            self.height,
            rs.format.bgr8,
            self.fps,
        )
        if not self.color_only:
            config.enable_stream(
                rs.stream.depth,
                self.width,
                self.height,
                rs.format.z16,
                self.fps,
            )

        profile = None
        frame_queue = queue.Queue(maxsize=self.queue_capacity)
        stop_token = object()
        writer_thread = None
        preview_queue = queue.Queue(maxsize=PREVIEW_QUEUE_CAPACITY)
        preview_stop_token = object()
        preview_thread = None
        cleanup_error = None
        try:
            profile = pipeline.start(config)
            align = None if self.color_only else rs.align(rs.stream.color)
            color_profile = profile.get_stream(
                rs.stream.color
            ).as_video_stream_profile()
            depth_scale = None
            native_depth_intrinsics = None
            if not self.color_only:
                depth_profile = profile.get_stream(
                    rs.stream.depth
                ).as_video_stream_profile()
                native_depth_intrinsics = intrinsics_dict(depth_profile)
                depth_scale = (
                    profile.get_device().first_depth_sensor().get_depth_scale()
                )

            metadata = {
                "schema": "realsense_capture_v2",
                "device": device_info(rs, profile.get_device()),
                "capture": {
                    "width": self.width,
                    "height": self.height,
                    "fps": self.fps,
                    "warmup_frames": self.warmup_frames,
                    "jpeg_quality": self.jpeg_quality,
                    "depth_png_compression": self.depth_png_compression,
                    "raw_frame_queue_capacity": self.queue_capacity,
                    "color_format": "BGR8 saved as JPEG",
                    "depth_format": (
                        None if self.color_only else "Z16 saved as uint16 PNG"
                    ),
                    "depth_aligned_to_color": not self.color_only,
                    "depth_scale_meters_per_unit": depth_scale,
                    "color_intrinsics": intrinsics_dict(color_profile),
                    "native_depth_intrinsics": native_depth_intrinsics,
                    "live_preview_enabled": self.preview,
                },
            }
            with open(self.metadata_path, "w", encoding="utf-8") as handle:
                json.dump(metadata, handle, ensure_ascii=False, indent=2)

            for _ in range(self.warmup_frames):
                if self.stop_event.is_set():
                    break
                pipeline.wait_for_frames(5000)

            writer_thread = threading.Thread(
                target=self._writer,
                args=(frame_queue, stop_token, cv2),
                name="realsense-disk-writer",
                daemon=True,
            )
            writer_thread.start()
            if self.preview:
                preview_thread = threading.Thread(
                    target=self._preview,
                    args=(preview_queue, preview_stop_token, cv2),
                    name="realsense-preview",
                    daemon=True,
                )
                preview_thread.start()
                self.preview_started_event.wait(timeout=5)
                if not self.preview_started_event.is_set():
                    raise RuntimeError("RealSense preview startup timed out")
                if self.preview_error is not None:
                    raise RuntimeError(
                        "RealSense preview failed: {}".format(self.preview_error)
                    )
            self.started_event.set()
            while not self.stop_event.is_set():
                if self.writer_error is not None:
                    raise RuntimeError(
                        "RealSense writer failed: {}".format(self.writer_error)
                    )
                if self.preview_error is not None:
                    raise RuntimeError(
                        "RealSense preview failed: {}".format(self.preview_error)
                    )
                try:
                    frames = pipeline.wait_for_frames(5000)
                except RuntimeError:
                    self.dropped_count += 1
                    continue

                # Timestamp immediately after the SDK returns the frameset.
                host_monotonic_ns = time.monotonic_ns()
                host_unix_ns = time.time_ns()
                if align is not None:
                    frames = align.process(frames)

                color_frame = frames.get_color_frame()
                depth_frame = (
                    None if self.color_only else frames.get_depth_frame()
                )
                if not color_frame or (not self.color_only and not depth_frame):
                    self.dropped_count += 1
                    continue

                frame_index = self.received_count
                self.received_count += 1
                color_name = "{:06d}.jpg".format(frame_index)

                # Do not allocate another ~4.4 MiB raw RGB-D pair when the
                # bounded disk queue is already full.  The preview may still
                # receive the latest color image, but depth allocation is
                # skipped for this dropped recording frame.
                if frame_queue.full():
                    self.dropped_count += 1
                    if self.preview and not self.preview_closed_event.is_set():
                        preview_color = np.asanyarray(
                            color_frame.get_data()
                        ).copy()
                        self._offer_latest(preview_queue, preview_color)
                    continue

                # Copy SDK-owned frame buffers before returning to wait_for_frames.
                # JPEG/PNG compression and filesystem writes happen in _writer.
                color = np.asanyarray(color_frame.get_data()).copy()

                depth_relative = None
                depth = None
                if depth_frame is not None:
                    depth_name = "{:06d}.png".format(frame_index)
                    depth = np.asanyarray(depth_frame.get_data()).copy()
                    depth_relative = "depth/{}".format(depth_name)

                record = {
                    "schema": "realsense_frame_v2",
                    "frame_index": frame_index,
                    "host_unix_ns": host_unix_ns,
                    "host_monotonic_ns": host_monotonic_ns,
                    "device_timestamp_ms": color_frame.get_timestamp(),
                    "device_timestamp_domain": str(
                        color_frame.get_frame_timestamp_domain()
                    ),
                    "device_frame_number": color_frame.get_frame_number(),
                    "color_path": "color/{}".format(color_name),
                    "depth_path": depth_relative,
                }
                try:
                    frame_queue.put_nowait((record, color, depth))
                except queue.Full:
                    self.dropped_count += 1
                if self.preview and not self.preview_closed_event.is_set():
                    self._offer_latest(preview_queue, color)
        finally:
            if profile is not None:
                pipeline.stop()
            if preview_thread is not None:
                while preview_thread.is_alive():
                    try:
                        preview_queue.put_nowait(preview_stop_token)
                        break
                    except queue.Full:
                        try:
                            preview_queue.get_nowait()
                        except queue.Empty:
                            pass
                preview_thread.join(timeout=5)
                if preview_thread.is_alive():
                    cleanup_error = RuntimeError(
                        "RealSense preview did not stop"
                    )
            if writer_thread is not None:
                while True:
                    try:
                        frame_queue.put(stop_token, timeout=0.5)
                        break
                    except queue.Full:
                        if self.writer_error is not None:
                            break
                writer_thread.join(timeout=30)
                if writer_thread.is_alive():
                    raise RuntimeError("RealSense writer did not stop")
                if self.writer_error is not None:
                    raise RuntimeError(
                        "RealSense writer failed: {}".format(self.writer_error)
                    )
            if cleanup_error is not None:
                raise cleanup_error
