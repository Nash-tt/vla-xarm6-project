#!/usr/bin/env python3
"""Export PhaseLink raw packets as aligned S-scan PNG images.

The decoder mirrors ViewWidget::showEScan in the PhaseLink Qt application:
each packet contains one row per beam, each beam contains the effective A-scan
samples followed by 16 int16 measurement words, and the packet ends with the
32-byte device tail.  The displayed S-scan is the transposed beam/sample
matrix colored by the PhaseLink amplitude palette.
"""

import argparse
import json
import os
import re

import cv2
import numpy as np


CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
WORKSPACE_DIR = os.path.abspath(os.path.join(ROOT_DIR, "..", ".."))
DEFAULT_PALETTE_SOURCE = os.path.join(
    WORKSPACE_DIR,
    "PAU_SDK - dt",
    "Phaselink",
    "dialog",
    "amplitudpalette.cpp",
)
TAIL_SIZE_BYTES = 32
MEASUREMENT_WORDS_PER_BEAM = 16


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def resolve_ultrasound_settings(episode_dir, ultrasound_meta):
    """Return settings and their provenance.

    Newer PhaseLink builds persist the actual applied settings in
    ultrasound_metadata.json.  Older captures only contain the settings that
    the Python collector requested in metadata.json.  The fallback keeps
    those episodes decodable, while making the weaker provenance explicit.
    """
    settings = ultrasound_meta.get("applied_settings")
    if settings:
        return settings, "ultrasound_metadata.applied_settings"

    episode_meta_path = os.path.join(episode_dir, "metadata.json")
    if os.path.exists(episode_meta_path):
        episode_meta = read_json(episode_meta_path)
        settings = (
            episode_meta.get("ultrasound_preset", {})
            .get("requested_settings")
        )
        if settings:
            return settings, "metadata.ultrasound_preset.requested_settings"

    raise RuntimeError(
        "no ultrasound settings found in ultrasound_metadata.json or metadata.json"
    )


def calculate_beam_count(settings):
    array = settings["array"]
    first = int(array["first_element"])
    last = int(array["last_element"])
    aperture = int(array["aperture"])
    step = int(array["element_step"])
    if step <= 0 or aperture <= 0 or last < first:
        raise ValueError("invalid array settings for beam-count calculation")
    usable_starts = last - first - aperture + 2
    if usable_starts <= 0:
        raise ValueError("aperture is larger than the configured element range")
    return (usable_starts - 1) // step + 1


def load_phaselink_palette(path):
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        source = handle.read()
    match = re.search(
        r"static\s+const\s+uint32_t\s+color_Amplitude\[\]\s*=\s*\{(.*?)\};",
        source,
        flags=re.DOTALL,
    )
    if not match:
        raise ValueError("could not find PhaseLink color_Amplitude palette")
    values = [int(token, 16) for token in re.findall(r"0x[0-9a-fA-F]+", match.group(1))]
    if len(values) < 256:
        raise ValueError("PhaseLink amplitude palette has fewer than 256 colors")
    # QRgb is 0xAARRGGBB. OpenCV PNG input is BGR.
    return np.asarray(
        [[value & 0xFF, (value >> 8) & 0xFF, (value >> 16) & 0xFF]
         for value in values[:256]],
        dtype=np.uint8,
    )


def decode_s_scan(packet, beam_count, display_max_amplitude_percent):
    if len(packet) <= TAIL_SIZE_BYTES:
        raise ValueError("packet is too small to contain beam data and tail")
    payload_size = len(packet) - TAIL_SIZE_BYTES
    denominator = beam_count * np.dtype("<i2").itemsize
    if payload_size % denominator:
        raise ValueError(
            "packet payload {} is not divisible by {} beams of int16 data".format(
                payload_size, beam_count
            )
        )
    words_per_beam = payload_size // denominator
    sample_count = words_per_beam - MEASUREMENT_WORDS_PER_BEAM
    if sample_count <= 0:
        raise ValueError("packet has no A-scan samples")

    words = np.frombuffer(packet, dtype="<i2", count=beam_count * words_per_beam)
    beams = words.reshape(beam_count, words_per_beam)[:, :sample_count]
    # This is the exact scaling used by ViewWidget::showEScan. For the 800%
    # display range it reduces to abs(raw)/16, clipped to palette index 255.
    scaled = (
        np.abs(beams.astype(np.int32))
        * float(display_max_amplitude_percent)
        / 12800.0
    )
    amplitude_u8 = np.clip(scaled, 0, 255).astype(np.uint8).T
    return amplitude_u8


def write_png(path, image):
    ok = cv2.imwrite(path, image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise OSError("OpenCV failed to write {}".format(path))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", required=True)
    parser.add_argument(
        "--valid-only",
        action="store_true",
        help="Export only samples whose alignment quality.valid is true.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing ultrasound_images.jsonl manifest.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Export at most this many images (useful for decoder checks).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    episode_dir = os.path.abspath(args.episode)
    ultrasound_meta = read_json(os.path.join(episode_dir, "ultrasound_metadata.json"))
    alignment_meta = read_json(os.path.join(episode_dir, "alignment.json"))
    settings, settings_source = resolve_ultrasound_settings(
        episode_dir, ultrasound_meta
    )
    if settings_source != "ultrasound_metadata.applied_settings":
        print(
            "WARNING: PhaseLink actual settings were not saved; using requested "
            "episode settings for decoding."
        )

    beam_count = calculate_beam_count(settings)
    display_max = settings["basic"]["display_max_amplitude_percent"]
    range_start = settings["basic"]["range_start_mm"]
    range_end = settings["basic"]["range_end_mm"]
    palette = load_phaselink_palette(DEFAULT_PALETTE_SOURCE)

    native_color_dir = os.path.join(episode_dir, "ultrasound_s_scan_color")
    native_gray_dir = os.path.join(episode_dir, "ultrasound_s_scan_gray")
    display_color_dir = os.path.join(episode_dir, "ultrasound_s_scan_display_color")
    display_gray_dir = os.path.join(episode_dir, "ultrasound_s_scan_display_gray")
    for directory in (
        native_color_dir,
        native_gray_dir,
        display_color_dir,
        display_gray_dir,
    ):
        os.makedirs(directory, exist_ok=True)
    manifest_path = os.path.join(episode_dir, "ultrasound_images.jsonl")
    metadata_path = os.path.join(episode_dir, "ultrasound_image_metadata.json")
    if os.path.exists(manifest_path) and not args.overwrite:
        raise FileExistsError(
            "{} already exists; pass --overwrite to regenerate".format(manifest_path)
        )

    packet_path = os.path.join(episode_dir, ultrasound_meta["packet_file"])
    aligned_path = os.path.join(episode_dir, alignment_meta["output"])
    exported = 0
    skipped_invalid = 0
    sample_count = None
    display_width = None
    display_height = None
    cached_key = None
    cached_gray = None
    cached_display_gray = None

    with open(packet_path, "rb") as packets, open(
        manifest_path, "w", encoding="utf-8", buffering=1
    ) as manifest:
        for sample in read_jsonl(aligned_path):
            if args.valid_only and not sample["quality"]["valid"]:
                skipped_invalid += 1
                continue
            ultrasound = sample["observation"].get("ultrasound")
            if ultrasound is None:
                skipped_invalid += 1
                continue
            offset = int(ultrasound["binary_offset"])
            size = int(ultrasound["binary_size"])
            key = (offset, size)
            if key != cached_key:
                packets.seek(offset)
                packet = packets.read(size)
                if len(packet) != size:
                    raise EOFError(
                        "short read at offset {}: expected {}, received {}".format(
                            offset, size, len(packet)
                        )
                    )
                cached_gray = decode_s_scan(packet, beam_count, display_max)
                depth_span_mm = float(range_end) - float(range_start)
                if depth_span_mm <= 0:
                    raise ValueError("ultrasound depth range must be positive")
                pitch_mm = float(settings["basic"]["probe_pitch_mm"])
                element_step = int(settings["array"]["element_step"])
                lateral_span_mm = (beam_count - 1) * pitch_mm * element_step
                # Keep all native depth samples and expand the sparse beam
                # axis so one pixel represents the same physical distance in
                # both directions. This matches the aspect seen in the Qt
                # S-scan much better than the raw 49 x 888 sampling grid.
                depth_mm_per_pixel = depth_span_mm / cached_gray.shape[0]
                target_width = max(
                    beam_count,
                    int(round(lateral_span_mm / depth_mm_per_pixel)),
                )
                cached_display_gray = cv2.resize(
                    cached_gray,
                    (target_width, cached_gray.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
                cached_key = key
            gray = cached_gray
            display_gray = cached_display_gray
            if sample_count is None:
                sample_count = int(gray.shape[0])
                display_height, display_width = map(int, display_gray.shape)
            elif sample_count != gray.shape[0]:
                raise ValueError("effective sample count changed within the episode")
            color = palette[gray]
            display_color = palette[display_gray]

            sample_index = int(sample["sample_index"])
            filename = "{:06d}.png".format(sample_index)
            native_color_rel = os.path.join(
                "ultrasound_s_scan_color", filename
            ).replace("\\", "/")
            native_gray_rel = os.path.join(
                "ultrasound_s_scan_gray", filename
            ).replace("\\", "/")
            display_color_rel = os.path.join(
                "ultrasound_s_scan_display_color", filename
            ).replace("\\", "/")
            display_gray_rel = os.path.join(
                "ultrasound_s_scan_display_gray", filename
            ).replace("\\", "/")
            write_png(os.path.join(episode_dir, native_color_rel), color)
            write_png(os.path.join(episode_dir, native_gray_rel), gray)
            write_png(os.path.join(episode_dir, display_color_rel), display_color)
            write_png(os.path.join(episode_dir, display_gray_rel), display_gray)

            record = {
                "schema": "phaselink_s_scan_image_v1",
                "image_index": exported,
                "aligned_sample_index": sample_index,
                "timestamp_unix_ns": sample["timestamp_unix_ns"],
                "timestamp_from_start_s": sample["timestamp_from_start_s"],
                "quality_valid": sample["quality"]["valid"],
                "color_path": display_color_rel,
                "gray_path": display_gray_rel,
                "native_color_path": native_color_rel,
                "native_gray_path": native_gray_rel,
                "display_height_pixels": int(display_gray.shape[0]),
                "display_width_pixels": int(display_gray.shape[1]),
                "native_height_depth_samples": int(gray.shape[0]),
                "native_width_beams": int(gray.shape[1]),
                "depth_range_mm": [range_start, range_end],
                "source_packet_index": ultrasound["packet_index"],
                "binary_offset": offset,
                "binary_size": size,
                "device_frame_number": ultrasound.get("device_frame_number"),
                "device_timestamp_ticks_200mhz": ultrasound.get(
                    "device_timestamp_ticks_200mhz"
                ),
                "alignment_skew_ms": ultrasound.get("alignment_skew_ms"),
            }
            manifest.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            exported += 1
            if args.limit is not None and exported >= args.limit:
                break

    metadata = {
        "schema": "phaselink_s_scan_export_v1",
        "source_episode": episode_dir,
        "source_packet_file": ultrasound_meta["packet_file"],
        "source_alignment_file": alignment_meta["output"],
        "output_rate_hz": alignment_meta["fps"],
        "exported_image_count": exported,
        "skipped_invalid_count": skipped_invalid,
        "valid_only": args.valid_only,
        "beam_count": beam_count,
        "effective_depth_sample_count": sample_count,
        "display_width_pixels": display_width,
        "display_height_pixels": display_height,
        "measurement_words_per_beam": MEASUREMENT_WORDS_PER_BEAM,
        "packet_tail_size_bytes": TAIL_SIZE_BYTES,
        "depth_range_mm": [range_start, range_end],
        "display_max_amplitude_percent": display_max,
        "settings_source": settings_source,
        "settings_are_actual_readback": (
            settings_source == "ultrasound_metadata.applied_settings"
        ),
        "orientation": "rows=depth, columns=linear-array scan position",
        "display_aspect": "equal physical millimetres per pixel on both axes",
        "amplitude_scaling": "PhaseLink ViewWidget::showEScan",
        "palette_source": os.path.abspath(DEFAULT_PALETTE_SOURCE),
        "color_directory": "ultrasound_s_scan_display_color",
        "gray_directory": "ultrasound_s_scan_display_gray",
        "native_color_directory": "ultrasound_s_scan_color",
        "native_gray_directory": "ultrasound_s_scan_gray",
        "manifest": "ultrasound_images.jsonl",
    }
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    print("Exported S-scan images:", exported)
    print("Native shape: {} depth samples x {} beams".format(sample_count, beam_count))
    print("Display shape: {} x {} pixels".format(display_width, display_height))
    print("Display color:", display_color_dir)
    print("Display gray:", display_gray_dir)
    print("Manifest:", manifest_path)


if __name__ == "__main__":
    main()
