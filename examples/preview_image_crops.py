#!/usr/bin/env python3

import argparse
import copy
import os
from pathlib import Path
import sys
import time
from collections import OrderedDict
from importlib import import_module

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "serl_robot_infra", REPO_ROOT / "serl_launcher"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from franka_env.camera.rs_capture import RSCapture
from franka_env.camera.video_capture import VideoCapture


INSTRUCTIONS = [
    "Drag with left mouse button to define a crop.",
    "Keys: p=print crop lambdas, c=clear active crop, s=save snapshots, q=quit.",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Preview raw RealSense feeds and interactively define IMAGE_CROP "
            "rectangles for an experiment config."
        )
    )
    parser.add_argument(
        "--config_module",
        required=True,
        help="Python module path for the experiment config, e.g. experiments.workpiece_pickup.config",
    )
    parser.add_argument(
        "--display_width",
        type=int,
        default=960,
        help="Maximum width used for each preview window.",
    )
    parser.add_argument(
        "--snapshot_dir",
        default="crop_debug",
        help="Directory to save annotated snapshots when pressing 's'.",
    )
    return parser.parse_args()


def get_env_config(config_module):
    module = import_module(config_module)
    env_config = getattr(module, "EnvConfig")
    return env_config() if callable(env_config) else env_config


def get_crop_tool_camera_configs(env_config):
    camera_configs = OrderedDict(getattr(env_config, "REALSENSE_CAMERAS", {}))
    extra_camera_configs = getattr(env_config, "CROP_TOOL_CAMERAS", {})
    for camera_name, config in extra_camera_configs.items():
        camera_configs[camera_name] = copy.deepcopy(config)
    return camera_configs


def get_crop_tool_image_crop_keys(env_config):
    image_crop = OrderedDict(getattr(env_config, "IMAGE_CROP", {}))
    extra_image_crop = getattr(env_config, "CROP_TOOL_IMAGE_CROP", {})
    for camera_name, crop in extra_image_crop.items():
        image_crop[camera_name] = crop
    return image_crop


def dedupe_camera_configs(camera_configs):
    resolved = RSCapture.resolve_camera_configs(camera_configs)
    physical_configs = OrderedDict()
    aliases_by_serial = OrderedDict()

    for camera_name, config in resolved.items():
        serial = config["serial_number"]
        aliases_by_serial.setdefault(serial, []).append(camera_name)

        if serial not in physical_configs:
            physical_configs[serial] = copy.deepcopy(config)
            continue

        existing = physical_configs[serial]
        comparable_existing = {k: v for k, v in existing.items() if k != "serial_number"}
        comparable_new = {k: v for k, v in config.items() if k != "serial_number"}
        if comparable_existing != comparable_new:
            raise ValueError(
                "The same physical camera is configured with conflicting settings. "
                f"Serial {serial} is used by {aliases_by_serial[serial]} with configs "
                f"{comparable_existing} and {comparable_new}. Keep dim/fps/exposure consistent "
                "when reusing a camera for multiple logical views."
            )

    return resolved, physical_configs, aliases_by_serial


class CropWindow:
    def __init__(self, name, serial_number, display_width):
        self.name = name
        self.serial_number = serial_number
        self.display_width = display_width
        self.window_name = f"crop::{name}"
        self.scale = 1.0
        self.frame_shape = None
        self.drag_start = None
        self.drag_current = None
        self.crop = None
        self.cursor_xy = None
        self.latest_frame = None
        self.last_interaction_ts = 0.0

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

    def _mouse_callback(self, event, x, y, flags, param):
        del flags, param
        self.last_interaction_ts = time.monotonic()
        self.cursor_xy = self._display_to_raw((x, y))

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = self._display_to_raw((x, y))
            self.drag_current = self.drag_start
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_start is not None:
            self.drag_current = self._display_to_raw((x, y))
        elif event == cv2.EVENT_LBUTTONUP and self.drag_start is not None:
            self.drag_current = self._display_to_raw((x, y))
            crop = self._normalize_crop(self.drag_start, self.drag_current)
            self.crop = crop if crop is not None else self.crop
            self.drag_start = None
            self.drag_current = None

    def _display_to_raw(self, point):
        if self.frame_shape is None:
            return (0, 0)
        x = int(round(point[0] / self.scale))
        y = int(round(point[1] / self.scale))
        h, w = self.frame_shape[:2]
        return (int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1)))

    @staticmethod
    def _normalize_crop(start, end):
        x0, y0 = start
        x1, y1 = end
        left, right = sorted((x0, x1))
        top, bottom = sorted((y0, y1))
        if right - left < 2 or bottom - top < 2:
            return None
        return (top, bottom, left, right)

    def clear_crop(self):
        self.crop = None

    def render(self, frame):
        self.frame_shape = frame.shape
        _, raw_w = frame.shape[:2]
        self.scale = min(self.display_width / raw_w, 1.0)
        display = frame.copy()

        active_crop = self.crop
        if self.drag_start is not None and self.drag_current is not None:
            active_crop = self._normalize_crop(self.drag_start, self.drag_current)

        if active_crop is not None:
            top, bottom, left, right = active_crop
            cv2.rectangle(display, (left, top), (right, bottom), (0, 255, 0), 3)
            crop_h = bottom - top
            crop_w = right - left
            cv2.putText(
                display,
                f"x={left}:{right}  y={top}:{bottom}  size={crop_w}x{crop_h}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            preview = frame[top:bottom, left:right]
            preview_max_w = min(280, preview.shape[1])
            preview_scale = preview_max_w / max(preview.shape[1], 1)
            preview_size = (
                max(1, int(preview.shape[1] * preview_scale)),
                max(1, int(preview.shape[0] * preview_scale)),
            )
            preview = cv2.resize(preview, preview_size)
            pad = 12
            y1 = pad
            y2 = pad + preview.shape[0]
            x2 = display.shape[1] - pad
            x1 = x2 - preview.shape[1]
            if x1 >= 0 and y2 <= display.shape[0]:
                display[y1:y2, x1:x2] = preview
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

        overlay_y = display.shape[0] - 22 * (len(INSTRUCTIONS) + 1)
        overlay_y = max(30, overlay_y)
        for line in INSTRUCTIONS:
            cv2.putText(
                display,
                line,
                (20, overlay_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            overlay_y += 22

        if self.cursor_xy is not None:
            cv2.putText(
                display,
                f"cursor raw xy={self.cursor_xy}",
                (20, overlay_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        if self.scale != 1.0:
            display = cv2.resize(
                display,
                (int(display.shape[1] * self.scale), int(display.shape[0] * self.scale)),
            )

        cv2.imshow(self.window_name, display)
        return display

    def crop_lambda(self):
        if self.crop is None:
            return None
        top, bottom, left, right = self.crop
        return f'"{self.name}": lambda img: img[{top}:{bottom}, {left}:{right}]'


def print_current_crops(windows):
    print("\nIMAGE_CROP = {")
    for window in windows.values():
        crop_lambda = window.crop_lambda()
        if crop_lambda is not None:
            print(f"    {crop_lambda},")
    print("}")


def save_snapshots(windows, snapshot_dir):
    os.makedirs(snapshot_dir, exist_ok=True)
    for window in windows.values():
        frame = window.latest_frame
        if frame is None:
            continue
        frame = frame.copy()
        if window.crop is not None:
            top, bottom, left, right = window.crop
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)
        path = os.path.join(snapshot_dir, f"{window.name}.png")
        cv2.imwrite(path, frame)
        print(f"Saved snapshot to {path}")


def main():
    args = parse_args()
    env_config = get_env_config(args.config_module)

    camera_configs = get_crop_tool_camera_configs(env_config)
    image_crop_keys = set(get_crop_tool_image_crop_keys(env_config).keys())
    camera_keys = set(camera_configs.keys())
    unknown_crop_keys = sorted(image_crop_keys - camera_keys)
    missing_crop_keys = sorted(camera_keys - image_crop_keys)

    if unknown_crop_keys:
        print(
            "Warning: IMAGE_CROP contains keys that are not present in REALSENSE_CAMERAS: "
            + ", ".join(unknown_crop_keys)
        )
    if missing_crop_keys:
        print(
            "Note: these camera keys do not currently have IMAGE_CROP entries: "
            + ", ".join(missing_crop_keys)
        )

    resolved_cameras, physical_configs, aliases_by_serial = dedupe_camera_configs(camera_configs)
    captures_by_serial = {}
    windows = OrderedDict()

    try:
        for serial, config in physical_configs.items():
            alias = aliases_by_serial[serial][0]
            captures_by_serial[serial] = VideoCapture(RSCapture(name=alias, **config))

        for camera_name, config in resolved_cameras.items():
            serial = config["serial_number"]
            windows[camera_name] = CropWindow(
                name=camera_name,
                serial_number=serial,
                display_width=args.display_width,
            )
        active_window_name = next(iter(windows))
        print("Connected crop preview windows:")
        for camera_name, config in resolved_cameras.items():
            print(
                f"  {camera_name}: serial={config['serial_number']} "
                f"dim={config.get('dim', 'default')} exposure={config.get('exposure', 'default')}"
            )
        print("\nUse the mouse in any window to make it the active crop window.")

        while True:
            latest_frames = {}
            for serial, capture in captures_by_serial.items():
                latest_frames[serial] = capture.read()

            for name, window in windows.items():
                frame = latest_frames[window.serial_number]
                window.latest_frame = frame
                window.render(frame)

            latest_active = max(
                windows,
                key=lambda name: windows[name].last_interaction_ts,
            )
            if windows[latest_active].last_interaction_ts > 0.0:
                active_window_name = latest_active

            key = cv2.waitKey(10) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("p"):
                print_current_crops(windows)
            elif key == ord("c"):
                windows[active_window_name].clear_crop()
                print(f"Cleared crop for {active_window_name}")
            elif key == ord("s"):
                save_snapshots(windows, args.snapshot_dir)
    finally:
        for capture in captures_by_serial.values():
            capture.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
