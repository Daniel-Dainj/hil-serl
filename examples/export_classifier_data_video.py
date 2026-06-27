#!/usr/bin/env python3

import argparse
import pickle
from collections.abc import Mapping
from pathlib import Path

import cv2
import numpy as np

try:
    import imageio.v2 as imageio
except ImportError as exc:
    raise ImportError(
        "imageio[ffmpeg] is required for H.264 export. "
        "Run this script with `uv run` or install project dependencies first."
    ) from exc


STATE_GROUP_PRESETS = {
    19: [
        ("tcp_pose", ["x", "y", "z", "roll", "pitch", "yaw"]),
        ("tcp_vel", ["vx", "vy", "vz", "wx", "wy", "wz"]),
        ("gripper_pose", ["grip"]),
        ("tcp_force", ["fx", "fy", "fz"]),
        ("tcp_torque", ["tx", "ty", "tz"]),
    ],
}

ACTION_GROUP_PRESETS = {
    7: [("action", ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "grip"])],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert classifier_data pickle transitions into an annotated mp4. "
            "Each frame shows observations, next_observations, and recorded "
            "transition fields such as action, reward, mask, done, and state. "
            "Videos are encoded as H.264 mp4 via imageio[ffmpeg]."
        )
    )
    parser.add_argument("input_path", help="Path to the classifier-data pkl file.")
    parser.add_argument(
        "--output",
        default=None,
        help=("Output mp4 path. Defaults to examples/classifier_video/<input-stem>.mp4 next to the repository root."),
    )
    parser.add_argument("--fps", type=float, default=10.0, help="Output video FPS.")
    parser.add_argument(
        "--tile-size",
        type=int,
        default=256,
        help="Rendered size for each camera tile.",
    )
    parser.add_argument(
        "--panel-width",
        type=int,
        default=760,
        help="Width of the right-hand information panel.",
    )
    parser.add_argument(
        "--image-keys",
        nargs="+",
        default=None,
        help="Optional ordered subset of image keys to render.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Index of the first transition to render.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of transitions to render.",
    )
    return parser.parse_args()


def load_transitions(path: Path):
    with path.open("rb") as f:
        transitions = pickle.load(f)

    if not isinstance(transitions, list):
        raise TypeError(f"Expected a list of transitions, got {type(transitions).__name__}.")
    if not transitions:
        raise ValueError("The pickle file contains no transitions.")
    if not isinstance(transitions[0], Mapping):
        raise TypeError("Expected each transition to be a mapping with observations/actions/etc.")
    return transitions


def extract_image_dict(observation):
    if "images" in observation and isinstance(observation["images"], Mapping):
        return observation["images"]
    return {key: value for key, value in observation.items() if key != "state" and not isinstance(value, Mapping)}


def infer_image_keys(sample, requested_keys=None):
    observation = sample["observations"]
    image_dict = extract_image_dict(observation)
    available_keys = list(image_dict.keys())
    if not available_keys:
        raise ValueError("No image keys were found under observations.")

    if requested_keys is None:
        return available_keys

    missing = [key for key in requested_keys if key not in image_dict]
    if missing:
        raise KeyError(f"Requested image keys are missing from observations: {', '.join(missing)}")
    return requested_keys


def latest_array(value):
    array = np.asarray(value)
    if array.ndim >= 1 and array.shape[0] == 1:
        array = array[0]
    elif array.ndim >= 4:
        array = array[-1]
    return np.asarray(array)


def normalize_image(image):
    array = latest_array(image)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    elif array.ndim == 3 and array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=2)
    elif array.ndim == 3 and array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
        array = np.transpose(array, (1, 2, 0))

    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Unsupported image shape: {array.shape}")

    if array.dtype == np.uint8:
        return np.ascontiguousarray(array)

    array = array.astype(np.float32)
    if array.size and array.max() <= 1.5 and array.min() >= 0.0:
        array = array * 255.0
    array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def extract_state_vector(observation):
    state = observation.get("state")
    if state is None:
        return np.array([], dtype=np.float32)

    if isinstance(state, Mapping):
        pieces = []
        for value in state.values():
            piece = np.asarray(value)
            if piece.ndim >= 1 and piece.shape[0] == 1:
                piece = piece[0]
            pieces.append(np.asarray(piece).reshape(-1))
        if not pieces:
            return np.array([], dtype=np.float32)
        return np.concatenate(pieces, axis=0)

    state = np.asarray(state)
    if state.ndim >= 1 and state.shape[0] == 1:
        state = state[0]
    return np.asarray(state).reshape(-1)


def extract_vector(value):
    array = np.asarray(value)
    if array.ndim >= 1 and array.shape[0] == 1:
        array = array[0]
    return np.asarray(array).reshape(-1)


def format_scalar(value):
    if isinstance(value, (np.generic,)):
        value = value.item()
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def group_spec_for_vector(vector, presets, default_prefix):
    group_spec = presets.get(len(vector))
    if group_spec is not None:
        return group_spec
    return [(default_prefix, [f"{default_prefix}[{idx}]" for idx in range(len(vector))])]


def format_group_lines(title, vector, group_spec, values_per_line=3):
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    lines = [title]
    cursor = 0
    for group_name, labels in group_spec:
        values = vector[cursor : cursor + len(labels)]
        cursor += len(labels)
        if values.size == 0:
            continue

        for start in range(0, len(labels), values_per_line):
            line_labels = labels[start : start + values_per_line]
            line_values = values[start : start + values_per_line]
            prefix = group_name if start == 0 else " " * len(group_name)
            formatted = "  ".join(f"{label}={float(value):+0.3f}" for label, value in zip(line_labels, line_values))
            lines.append(f"{prefix}: {formatted}")
    return lines


def join_horizontal(images, pad=10, fill_value=24):
    if len(images) == 1:
        return images[0]
    spacer = np.full((images[0].shape[0], pad, 3), fill_value, dtype=np.uint8)
    pieces = []
    for index, image in enumerate(images):
        if index:
            pieces.append(spacer)
        pieces.append(image)
    return np.concatenate(pieces, axis=1)


def join_vertical(images, pad=10, fill_value=24):
    if len(images) == 1:
        return images[0]
    spacer = np.full((pad, images[0].shape[1], 3), fill_value, dtype=np.uint8)
    pieces = []
    for index, image in enumerate(images):
        if index:
            pieces.append(spacer)
        pieces.append(image)
    return np.concatenate(pieces, axis=0)


def build_image_tile(image, label, tile_size, border_color):
    image = normalize_image(image)
    height, width = image.shape[:2]
    scale = min(tile_size / max(width, 1), tile_size / max(height, 1))
    resized = cv2.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )

    tile = np.full((tile_size, tile_size, 3), 18, dtype=np.uint8)
    y0 = (tile_size - resized.shape[0]) // 2
    x0 = (tile_size - resized.shape[1]) // 2
    tile[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized

    cv2.rectangle(tile, (0, 0), (tile_size - 1, tile_size - 1), border_color, 2)
    cv2.rectangle(tile, (0, 0), (tile_size - 1, 28), (0, 0, 0), thickness=-1)
    cv2.putText(
        tile,
        label,
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return tile


def build_panel(sample, sample_index, total_samples, input_name, panel_shape):
    panel = np.full(panel_shape, (26, 26, 26), dtype=np.uint8)
    text_x = 18
    text_y = 28
    line_height = 19
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5

    reward = sample.get("rewards")
    mask = sample.get("masks")
    done = sample.get("dones")
    actions = extract_vector(sample.get("actions", []))
    obs_state = extract_state_vector(sample["observations"])
    next_state = extract_state_vector(sample["next_observations"])
    state_delta = next_state - obs_state if obs_state.size == next_state.size else np.array([])

    lines = [
        f"file: {input_name}",
        f"sample: {sample_index + 1}/{total_samples}",
        f"reward={format_scalar(reward)}  mask={format_scalar(mask)}  done={format_scalar(done)}",
    ]

    if actions.size:
        lines.extend(
            format_group_lines(
                "action",
                actions,
                group_spec_for_vector(actions, ACTION_GROUP_PRESETS, "action"),
            )
        )

    if obs_state.size:
        lines.extend(
            format_group_lines(
                "obs.state",
                obs_state,
                group_spec_for_vector(obs_state, STATE_GROUP_PRESETS, "state"),
            )
        )

    if state_delta.size:
        lines.extend(
            format_group_lines(
                "delta(next-obs)",
                state_delta,
                group_spec_for_vector(state_delta, STATE_GROUP_PRESETS, "dstate"),
            )
        )

    for line_index, line in enumerate(lines):
        color = (0, 220, 255) if line_index == 0 else (235, 235, 235)
        if "done=True" in line:
            color = (80, 160, 255)
        cv2.putText(
            panel,
            line,
            (text_x, text_y),
            font,
            font_scale,
            color,
            1,
            cv2.LINE_AA,
        )
        text_y += line_height
        if text_y >= panel.shape[0] - 10:
            break

    return panel


def build_frame(sample, sample_index, total_samples, image_keys, tile_size, panel_width, input_name):
    observation_images = extract_image_dict(sample["observations"])
    next_observation_images = extract_image_dict(sample["next_observations"])

    obs_tiles = [
        build_image_tile(observation_images[key], f"obs/{key}", tile_size, (40, 200, 80)) for key in image_keys
    ]
    next_tiles = [
        build_image_tile(next_observation_images[key], f"next/{key}", tile_size, (220, 140, 50)) for key in image_keys
    ]

    left_canvas = join_vertical(
        [join_horizontal(obs_tiles), join_horizontal(next_tiles)],
        pad=10,
    )
    panel = build_panel(
        sample=sample,
        sample_index=sample_index,
        total_samples=total_samples,
        input_name=input_name,
        panel_shape=(left_canvas.shape[0], panel_width, 3),
    )

    frame = join_horizontal([left_canvas, panel], pad=10)
    if bool(sample.get("dones", False)):
        cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), (0, 0, 255), 6)
    return frame


def ensure_h264_frame_compatibility(frame):
    pad_bottom = frame.shape[0] % 2
    pad_right = frame.shape[1] % 2
    if not pad_bottom and not pad_right:
        return frame
    return cv2.copyMakeBorder(
        frame,
        0,
        pad_bottom,
        0,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )


def open_h264_writer(output_path, fps):
    return imageio.get_writer(
        output_path,
        format="FFMPEG",
        mode="I",
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=1,
        output_params=[
            "-movflags",
            "+faststart",
        ],
    )


def main():
    args = parse_args()
    input_path = Path(args.input_path).resolve()
    transitions = load_transitions(input_path)

    start_index = max(0, args.start_index)
    end_index = len(transitions)
    if args.max_frames is not None:
        end_index = min(end_index, start_index + max(0, args.max_frames))
    selected = transitions[start_index:end_index]
    if not selected:
        raise ValueError("No transitions were selected for rendering.")

    image_keys = infer_image_keys(selected[0], requested_keys=args.image_keys)

    if args.output is None:
        output_dir = input_path.parents[1] / "classifier_video"
        output_path = output_dir / f"{input_path.stem}.mp4"
    else:
        output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ensure_h264_frame_compatibility(
        build_frame(
            selected[0],
            sample_index=start_index,
            total_samples=len(transitions),
            image_keys=image_keys,
            tile_size=args.tile_size,
            panel_width=args.panel_width,
            input_name=input_path.name,
        )
    )
    writer = open_h264_writer(str(output_path), args.fps)

    try:
        for offset, sample in enumerate(selected):
            sample_index = start_index + offset
            frame = build_frame(
                sample,
                sample_index=sample_index,
                total_samples=len(transitions),
                image_keys=image_keys,
                tile_size=args.tile_size,
                panel_width=args.panel_width,
                input_name=input_path.name,
            )
            frame = ensure_h264_frame_compatibility(frame)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            writer.append_data(frame_rgb)
            if (offset + 1) % 50 == 0 or offset == len(selected) - 1:
                print(f"Rendered {offset + 1}/{len(selected)} frames...")
    finally:
        writer.close()

    print(f"Saved H.264 video to {output_path}")


if __name__ == "__main__":
    main()
