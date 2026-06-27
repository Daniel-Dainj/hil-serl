import copy
from collections import OrderedDict
import os
import numpy as np
import pyrealsense2 as rs  # Intel RealSense cross-platform open-source API


class RSCapture:
    @classmethod
    def get_connected_devices(cls):
        devices = []
        try:
            context = rs.context()
            for dev in context.devices:
                devices.append(
                    {
                        "name": dev.get_info(rs.camera_info.name),
                        "serial_number": dev.get_info(rs.camera_info.serial_number),
                    }
                )
        except Exception as exc:
            raise RuntimeError(
                "Failed to enumerate RealSense devices. Check that librealsense/udev "
                "is set up correctly and that the cameras are connected."
            ) from exc
        return devices

    @classmethod
    def get_device_serial_numbers(cls):
        return [device["serial_number"] for device in cls.get_connected_devices()]

    @staticmethod
    def _camera_env_var(camera_name):
        sanitized = camera_name.upper().replace("-", "_")
        return f"SERL_CAMERA_{sanitized}_SERIAL"

    @classmethod
    def _resolve_serial_assignments(cls, requested_serials, available_serials, preferred_serials):
        if preferred_serials:
            unknown = [serial for serial in preferred_serials if serial not in available_serials]
            if unknown:
                raise ValueError(
                    "SERL_REALSENSE_SERIALS contains serial numbers that are not connected: "
                    + ", ".join(unknown)
                )

        ordered_candidates = list(preferred_serials)
        ordered_candidates.extend(
            serial for serial in available_serials if serial not in ordered_candidates
        )

        assignments = OrderedDict()
        used_serials = set()

        for requested_serial in requested_serials:
            if requested_serial in available_serials:
                assignments[requested_serial] = requested_serial
                used_serials.add(requested_serial)

        for requested_serial in requested_serials:
            if requested_serial in assignments:
                continue

            replacement = next(
                (serial for serial in ordered_candidates if serial not in used_serials),
                None,
            )
            if replacement is None:
                return None
            assignments[requested_serial] = replacement
            used_serials.add(replacement)

        return assignments

    @classmethod
    def resolve_camera_configs(cls, camera_configs):
        resolved = OrderedDict()
        for camera_name, config in camera_configs.items():
            if isinstance(config, dict):
                resolved[camera_name] = copy.deepcopy(config)
            else:
                resolved[camera_name] = {"serial_number": config}

            override = os.getenv(cls._camera_env_var(camera_name))
            if override:
                resolved[camera_name]["serial_number"] = override.strip()

        requested_serials = OrderedDict()
        for camera_name, config in resolved.items():
            serial_number = config.get("serial_number")
            if not serial_number:
                raise ValueError(
                    f"Camera '{camera_name}' has no serial_number configured. "
                    f"Set {cls._camera_env_var(camera_name)} to override it."
                )
            requested_serials.setdefault(serial_number, []).append(camera_name)

        available_devices = cls.get_connected_devices()
        available_serials = [device["serial_number"] for device in available_devices]
        preferred_serials = [
            serial.strip()
            for serial in os.getenv("SERL_REALSENSE_SERIALS", "").split(",")
            if serial.strip()
        ]

        assignments = cls._resolve_serial_assignments(
            list(requested_serials.keys()),
            available_serials,
            preferred_serials,
        )
        if assignments is None:
            available_summary = ", ".join(
                f"{device['name']} ({device['serial_number']})" for device in available_devices
            ) or "none"
            requested_summary = ", ".join(
                f"{camera_name}={config['serial_number']}" for camera_name, config in resolved.items()
            )
            raise ValueError(
                "Connected RealSense devices do not match the configured camera serial numbers. "
                f"Requested: {requested_summary}. Connected: {available_summary}. "
                "Set SERL_CAMERA_<CAMERA_NAME>_SERIAL or SERL_REALSENSE_SERIALS to override."
            )

        remapped = False
        for original_serial, camera_names in requested_serials.items():
            resolved_serial = assignments[original_serial]
            if resolved_serial != original_serial:
                remapped = True
            for camera_name in camera_names:
                resolved[camera_name]["serial_number"] = resolved_serial

        if remapped:
            print("Remapped RealSense serial numbers:")
            for camera_name, config in resolved.items():
                print(f"  {camera_name}: {config['serial_number']}")

        return resolved

    def __init__(self, name, serial_number, dim=(640, 480), fps=15, depth=False, exposure=40000):
        self.name = name
        available_serials = self.get_device_serial_numbers()
        if serial_number not in available_serials:
            raise ValueError(
                f"Camera '{name}' expected serial {serial_number}, but connected serials are "
                f"{available_serials}. Set {self._camera_env_var(name)} or "
                "SERL_REALSENSE_SERIALS to override."
            )
        self.serial_number = serial_number
        self.depth = depth
        self.pipe = rs.pipeline()
        self.cfg = rs.config()
        self.cfg.enable_device(self.serial_number)
        self.cfg.enable_stream(rs.stream.color, dim[0], dim[1], rs.format.bgr8, fps)
        if self.depth:
            self.cfg.enable_stream(rs.stream.depth, dim[0], dim[1], rs.format.z16, fps)
        self.profile = self.pipe.start(self.cfg)
        self.s = self.profile.get_device().query_sensors()[0]
        self.s.set_option(rs.option.exposure, exposure)

        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        align_to = rs.stream.color
        self.align = rs.align(align_to)

    def read(self):
        frames = self.pipe.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        if self.depth:
            depth_frame = aligned_frames.get_depth_frame()

        if color_frame.is_video_frame():
            image = np.asarray(color_frame.get_data())
            if self.depth and depth_frame.is_depth_frame():
                depth = np.expand_dims(np.asarray(depth_frame.get_data()), axis=2)
                return True, np.concatenate((image, depth), axis=-1)
            else:
                return True, image
        else:
            return False, None

    def close(self):
        self.pipe.stop()
        self.cfg.disable_all_streams()
