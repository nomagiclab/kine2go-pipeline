import os

import cv2
import tyro

"""Overlay frame numbers on a video or exported image sequence.

add_timestamp.py

Very small and simple tool to overlay a frame number on a video.

Usage (CLI):
    python add_timestamp.py -i input.mp4 -o output.mp4

Only options: input, output, and optional starting index.
All visual options (color, size, thickness, background) are fixed.
"""

VIDEO_EXTENSIONS = (".mp4", ".m4v", ".mov")
AVI_EXTENSIONS = (".avi",)
DEFAULT_VIDEO_CODEC = "MP4V"
AVI_VIDEO_CODEC = "XVID"
DEFAULT_FPS = 30.0
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
THICKNESS = 1
TEXT_COLOR = (255, 255, 255)
BACKGROUND_COLOR = (0, 0, 0)
MARGIN = (8, 8)
POSITION = "bottom-right"
BACKGROUND_PADDING = 4


def _choose_codec_for_path(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in VIDEO_EXTENSIONS:
        return DEFAULT_VIDEO_CODEC
    if ext in AVI_EXTENSIONS:
        return AVI_VIDEO_CODEC
    return DEFAULT_VIDEO_CODEC


def _compute_text_position(
    frame_w: int,
    frame_h: int,
    text_size: tuple[int, int],
    margin: tuple[int, int],
    position: str,
) -> tuple[int, int]:
    text_w, text_h = text_size
    pos = position.lower()
    x = margin[0] if "left" in pos else frame_w - margin[0] - text_w
    y = margin[1] + text_h if "top" in pos else frame_h - margin[1]
    return int(x), int(y)


def add_frame_numbers(input_path: str, output_path: str, start_index: int = 0, frames_dir: str | None = None):
    """Add frame numbers to a video (very small, opinionated defaults).

    Parameters:
    - input_path: source video path
    - output_path: destination video path
    - start_index: starting frame number (default 0)
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if frames_dir is None:
        codec = _choose_codec_for_path(output_path)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Cannot open output video for writing: {output_path}")
    else:
        os.makedirs(frames_dir, exist_ok=True)

    frame_idx = start_index
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        text = str(frame_idx)
        (text_w, text_h), baseline = cv2.getTextSize(text, FONT, FONT_SCALE, THICKNESS)
        x, y = _compute_text_position(w, h, (text_w, text_h), MARGIN, POSITION)

        rect_tl = (x - BACKGROUND_PADDING, y - text_h - BACKGROUND_PADDING)
        rect_br = (x + text_w + BACKGROUND_PADDING, y + baseline + BACKGROUND_PADDING)
        cv2.rectangle(frame, rect_tl, rect_br, BACKGROUND_COLOR, thickness=-1)

        cv2.putText(frame, text, (x, y), FONT, FONT_SCALE, TEXT_COLOR, THICKNESS, lineType=cv2.LINE_AA)

        if frames_dir is None:
            writer.write(frame)
        else:
            img_name = os.path.join(frames_dir, f"{frame_idx:06d}.png")
            cv2.imwrite(img_name, frame)

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()


def main(
    input: tyro.conf.Positional[str],
    output: tyro.conf.Positional[str],
    start: int = 0,
    frames_dir: str | None = None,
):
    """CLI: input and output are positional. Provide --frames-dir to save as images."""
    add_frame_numbers(input, output, start_index=start, frames_dir=frames_dir)


if __name__ == "__main__":
    tyro.cli(main)
