from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import importlib
import os
import queue
import select
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Protocol


if os.environ.get("XDG_SESSION_TYPE") == "wayland":
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.*=false")

for font_dir in (
    "/usr/share/fonts/google-noto",
    "/usr/share/fonts/liberation-sans-fonts",
    "/usr/share/fonts",
):
    if os.path.isdir(font_dir):
        os.environ.setdefault("QT_QPA_FONTDIR", font_dir)
        break

import cv2
import mss
import numpy as np
from dbus_next import Variant
from dbus_next.aio import MessageBus
from dbus_next.constants import BusType, MessageType
from dbus_next.message import Message
from ollama import Client


DEFAULT_PROMPT = (
    "Briefly describe the visible desktop. Mention important text, windows, "
    "and user interface state. Be concise."
)
DEFAULT_MODEL = "gemma3:4b"
QUIET_PREVIEW_STARTUP_FRAMES = 10
BLANK_CAPTURE_WARNING = (
    "Capture warning: the desktop backend is returning black frames. "
    "Try --capture portal on GNOME Wayland."
)
PORTAL_DESTINATION = "org.freedesktop.portal.Desktop"
PORTAL_PATH = "/org/freedesktop/portal/desktop"
PORTAL_SCREENCAST = "org.freedesktop.portal.ScreenCast"
PORTAL_REQUEST = "org.freedesktop.portal.Request"
PORTAL_SESSION = "org.freedesktop.portal.Session"
PORTAL_SOURCE_TYPES = {
    "screen": 1,
    "window": 2,
    "selection": 1,
}
SOURCE_LABELS = {
    "screen": "Screen",
    "window": "Window",
    "selection": "Selection",
}
GSTREAMER_HELPER = r"""
import os
import sys

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstVideo", "1.0")
from gi.repository import Gst, GstVideo

fd = int(sys.argv[1])
node_id = sys.argv[2]
width = int(sys.argv[3])
height = int(sys.argv[4])
frame_size = width * height * 3

Gst.init(None)
pipeline = Gst.parse_launch(
    "pipewiresrc fd={fd} path={node_id} do-timestamp=true ! "
    "videoconvert ! videoscale ! "
    "video/x-raw,format=BGR,width={width},height={height} ! "
    "appsink name=sink sync=false max-buffers=1 drop=true"
    .format(fd=fd, node_id=node_id, width=width, height=height)
)
sink = pipeline.get_by_name("sink")
pipeline.set_state(Gst.State.PLAYING)

try:
    while True:
        sample = sink.emit("pull-sample")
        if sample is None:
            raise RuntimeError("appsink returned no sample")

        caps = sample.get_caps()
        info = GstVideo.VideoInfo.new_from_caps(caps)
        buffer = sample.get_buffer()
        ok, mapped = buffer.map(Gst.MapFlags.READ)
        if not ok:
            raise RuntimeError("failed to map GStreamer buffer")

        try:
            data = mapped.data
            stride = info.stride[0]
            offset = info.offset[0]
            row_size = width * 3
            if stride == row_size and len(data) >= frame_size:
                os.write(1, data[offset : offset + frame_size])
            else:
                for row in range(height):
                    start = offset + row * stride
                    os.write(1, data[start : start + row_size])
        finally:
            buffer.unmap(mapped)
finally:
    pipeline.set_state(Gst.State.NULL)
"""


@dataclass(frozen=True)
class FrameJob:
    frame: np.ndarray
    captured_at: float


class CaptureBackend(Protocol):
    def read(self) -> np.ndarray:
        pass

    def close(self) -> None:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture the desktop and send sampled frames to a local VLM via Ollama."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Ollama vision model name, for example llama3.2-vision, moondream, or llava.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt sent with each sampled desktop frame.",
    )
    parser.add_argument(
        "--monitor",
        type=int,
        default=1,
        help="mss monitor index. 1 is usually the primary display. Used by --capture mss.",
    )
    parser.add_argument(
        "--capture",
        choices=("auto", "mss", "portal"),
        default="auto",
        help="Screen capture backend. Use portal on GNOME Wayland.",
    )
    parser.add_argument(
        "--wayland-source",
        type=parse_wayland_source,
        default="screen",
        metavar="{Selection,Window,Screen}",
        help="Wayland portal source mode: Selection, Window, or Screen.",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=960,
        help="Downsample captured frames to this maximum width for preview/processing.",
    )
    parser.add_argument(
        "--analysis-width",
        type=int,
        default=512,
        help="Downsample frames to this maximum width before sending them to the VLM.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Seconds between VLM requests. Lower values need a fast model/GPU.",
    )
    parser.add_argument(
        "--num-predict",
        type=int,
        default=80,
        help="Maximum response tokens generated per frame.",
    )
    parser.add_argument(
        "--vlm-timeout",
        type=float,
        default=120.0,
        help="Seconds to wait for one Ollama vision request before showing a timeout.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=80,
        choices=range(20, 101),
        metavar="20-100",
        help="JPEG quality for frames sent to Ollama.",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Run headless without an OpenCV preview window.",
    )
    parser.add_argument(
        "--ollama-host",
        default=None,
        help="Optional Ollama host URL, for example http://127.0.0.1:11434.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch the desktop GUI instead of the OpenCV preview loop.",
    )
    return parser.parse_args()


def parse_wayland_source(value: str) -> str:
    normalized = value.lower()
    if normalized not in PORTAL_SOURCE_TYPES:
        choices = ", ".join(SOURCE_LABELS.values())
        raise argparse.ArgumentTypeError(f"choose one of: {choices}")

    return normalized


class MSSCapture:
    def __init__(self, monitor_index: int) -> None:
        self.screen = mss.MSS()
        if monitor_index < 1 or monitor_index >= len(self.screen.monitors):
            self.screen.close()
            raise ValueError(
                f"Monitor {monitor_index} is unavailable. "
                f"Choose 1-{len(self.screen.monitors) - 1}."
            )

        self.monitor = self.screen.monitors[monitor_index]

    def read(self) -> np.ndarray:
        raw = np.array(self.screen.grab(self.monitor))
        return cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)

    def close(self) -> None:
        self.screen.close()


class SelectionCapture:
    def __init__(self, capture: CaptureBackend) -> None:
        self.capture = capture
        self.roi: tuple[int, int, int, int] | None = None

    def read(self) -> np.ndarray:
        frame = self.capture.read()
        if self.roi is None:
            self.roi = select_capture_region(frame)

        x, y, width, height = self.roi
        return frame[y : y + height, x : x + width].copy()

    def close(self) -> None:
        self.capture.close()


class PortalPipeWireCapture:
    def __init__(self, max_width: int, source: str) -> None:
        if not shutil.which("/usr/bin/python3"):
            raise RuntimeError(
                "The portal backend needs /usr/bin/python3 with GStreamer bindings."
            )

        self.max_width = max_width
        self.source = source
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self._run_loop, daemon=True)
        self.loop_thread.start()
        self.bus: MessageBus | None = None
        self.session_handle = ""
        self.pipewire_fd = -1
        self.process: subprocess.Popen[bytes] | None = None
        self.frame_size = 0
        self.width = 0
        self.height = 0

        future = asyncio.run_coroutine_threadsafe(self._open(), self.loop)
        future.result(timeout=90)
        print("Portal: capture backend ready.", flush=True)

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    async def _open(self) -> None:
        self.bus = await MessageBus(
            bus_type=BusType.SESSION, negotiate_unix_fd=True
        ).connect()
        print(
            f"Requesting GNOME {SOURCE_LABELS[self.source]} capture permission...",
            flush=True,
        )
        create_token = self._token("create")
        session_token = self._token("session")
        print("Portal: creating capture session...", flush=True)
        create_results = await self._portal_request(
            "CreateSession",
            "a{sv}",
            [
                {
                    "handle_token": Variant("s", create_token),
                    "session_handle_token": Variant("s", session_token),
                }
            ],
            create_token,
        )
        self.session_handle = create_results["session_handle"].value

        select_token = self._token("select")
        print(f"Portal: selecting {SOURCE_LABELS[self.source]} source...", flush=True)
        await self._portal_request(
            "SelectSources",
            "oa{sv}",
            [
                self.session_handle,
                {
                    "handle_token": Variant("s", select_token),
                    "types": Variant("u", PORTAL_SOURCE_TYPES[self.source]),
                    "multiple": Variant("b", False),
                    "cursor_mode": Variant("u", 2),
                    "persist_mode": Variant("u", 1),
                },
            ],
            select_token,
        )
        start_token = self._token("start")
        print("Portal: opening GNOME chooser...", flush=True)
        start_results = await self._portal_request(
            "Start",
            "osa{sv}",
            [self.session_handle, "", {"handle_token": Variant("s", start_token)}],
            start_token,
        )

        streams = start_results["streams"].value
        if not streams:
            raise RuntimeError("The portal did not return a PipeWire stream.")
        print("Portal: screen stream approved.", flush=True)

        node_id, properties = streams[0]
        source_width, source_height = self._stream_size(properties)
        self.width, self.height = self._scaled_size(source_width, source_height)
        self.frame_size = self.width * self.height * 3

        print("Portal: opening PipeWire remote...", flush=True)
        fd_message = await self._call_portal(
            "OpenPipeWireRemote", "oa{sv}", [self.session_handle, {}]
        )
        fd_index = fd_message.body[0]
        self.pipewire_fd = fd_message.unix_fds[fd_index]

        print("Portal: starting GStreamer reader...", flush=True)
        self.process = self._start_gstreamer(node_id, properties)
        assert self.process.stdout is not None
        os.set_blocking(self.process.stdout.fileno(), False)
        print(
            f"Portal: waiting for frames at {self.width}x{self.height}.",
            flush=True,
        )

    async def _portal_request(
        self, member: str, signature: str, body: list[object], handle_token: str
    ) -> dict[str, Variant]:
        expected_handle = self._expected_request_handle(handle_token)
        response_task = asyncio.create_task(self._wait_for_response(expected_handle))
        reply = await self._call_portal(member, signature, body)
        handle = reply.body[0]
        if handle != expected_handle:
            response_task.cancel()
            response_task = asyncio.create_task(self._wait_for_response(handle))

        return await response_task

    async def _call_portal(
        self, member: str, signature: str, body: list[object]
    ) -> Message:
        assert self.bus is not None
        reply = await self.bus.call(
            Message(
                destination=PORTAL_DESTINATION,
                path=PORTAL_PATH,
                interface=PORTAL_SCREENCAST,
                member=member,
                signature=signature,
                body=body,
            )
        )
        if reply is None:
            raise RuntimeError(f"{member} returned no D-Bus reply.")
        if reply.message_type == MessageType.ERROR:
            raise RuntimeError(f"{member} failed: {reply.body}")
        return reply

    async def _wait_for_response(self, handle: str) -> dict[str, Variant]:
        assert self.bus is not None
        future: asyncio.Future[dict[str, Variant]] = self.loop.create_future()

        def handler(message: Message) -> bool | None:
            if (
                message.message_type == MessageType.SIGNAL
                and message.path == handle
                and message.interface == PORTAL_REQUEST
                and message.member == "Response"
            ):
                response, results = message.body
                if future.done():
                    return True
                if response == 0:
                    future.set_result(results)
                else:
                    future.set_exception(RuntimeError("Screen capture was cancelled."))
                return True
            return None

        self.bus.add_message_handler(handler)
        match_rule = (
            "type='signal',"
            f"path='{handle}',"
            f"interface='{PORTAL_REQUEST}',"
            "member='Response'"
        )
        await self.bus.call(
            Message(
                destination="org.freedesktop.DBus",
                path="/org/freedesktop/DBus",
                interface="org.freedesktop.DBus",
                member="AddMatch",
                signature="s",
                body=[match_rule],
            )
        )
        try:
            return await asyncio.wait_for(future, timeout=90)
        finally:
            self.bus.remove_message_handler(handler)
            await self.bus.call(
                Message(
                    destination="org.freedesktop.DBus",
                    path="/org/freedesktop/DBus",
                    interface="org.freedesktop.DBus",
                    member="RemoveMatch",
                    signature="s",
                    body=[match_rule],
                )
            )

    def _start_gstreamer(
        self, node_id: int, properties: dict[str, Variant]
    ) -> subprocess.Popen[bytes]:
        command = [
            "/usr/bin/python3",
            "-c",
            GSTREAMER_HELPER,
            str(self.pipewire_fd),
            str(node_id),
            str(self.width),
            str(self.height),
        ]
        return subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            pass_fds=(self.pipewire_fd,),
        )

    def read(self) -> np.ndarray:
        if self.process is None or self.process.stdout is None:
            raise RuntimeError("The portal capture stream is not running.")

        data = self._read_exact(self.process.stdout, self.frame_size)
        frame = np.frombuffer(data, dtype=np.uint8).reshape((self.height, self.width, 3))
        return frame.copy()

    def close(self) -> None:
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()

        if self.pipewire_fd >= 0:
            os.close(self.pipewire_fd)
            self.pipewire_fd = -1

        if self.bus is not None and self.session_handle:
            future = asyncio.run_coroutine_threadsafe(self._close_session(), self.loop)
            try:
                future.result(timeout=2)
            except Exception:
                pass

        self.loop.call_soon_threadsafe(self.loop.stop)
        self.loop_thread.join(timeout=2)

    async def _close_session(self) -> None:
        assert self.bus is not None
        await self.bus.call(
            Message(
                destination=PORTAL_DESTINATION,
                path=self.session_handle,
                interface=PORTAL_SESSION,
                member="Close",
            )
        )
        self.bus.disconnect()

    def _token(self, prefix: str) -> str:
        return f"vplayer_{prefix}_{uuid.uuid4().hex}"

    def _expected_request_handle(self, token: str) -> str:
        assert self.bus is not None and self.bus.unique_name is not None
        sender = self.bus.unique_name.removeprefix(":").replace(".", "_")
        return f"/org/freedesktop/portal/desktop/request/{sender}/{token}"

    def _scaled_size(self, width: int, height: int) -> tuple[int, int]:
        if width <= self.max_width:
            return width, height

        scale = self.max_width / width
        return self.max_width, max(1, int(height * scale))

    def _stream_size(self, properties: dict[str, Variant]) -> tuple[int, int]:
        if "size" not in properties:
            raise RuntimeError("The portal stream did not include a size.")

        width, height = properties["size"].value
        return int(width), int(height)

    def _read_exact(self, stream: object, size: int) -> bytes:
        chunks: list[bytes] = []
        remaining = size
        started = time.monotonic()
        bytes_read = 0
        while remaining > 0:
            if time.monotonic() - started > 10.0:
                error = self._gstreamer_error()
                raise RuntimeError(
                    f"Timed out waiting for a full frame from PipeWire/GStreamer "
                    f"({bytes_read}/{size} bytes read)."
                    + (f" GStreamer said: {error}" if error else "")
                )

            ready, _, _ = select.select([stream], [], [], 0.5)
            if not ready:
                continue

            try:
                chunk = os.read(stream.fileno(), remaining)
            except BlockingIOError:
                continue
            if not chunk:
                error = self._gstreamer_error()
                if error:
                    raise RuntimeError(f"The PipeWire/GStreamer stream ended: {error}")
                raise RuntimeError("The PipeWire/GStreamer stream ended.")
            chunks.append(chunk)
            remaining -= len(chunk)
            bytes_read += len(chunk)
        return b"".join(chunks)

    def _gstreamer_error(self) -> str:
        if self.process is None or self.process.stderr is None:
            return ""

        ready, _, _ = select.select([self.process.stderr], [], [], 0)
        if not ready:
            return ""

        data = os.read(self.process.stderr.fileno(), 8192)
        return data.decode("utf-8", errors="replace").strip()


def create_capture_backend(args: argparse.Namespace) -> CaptureBackend:
    capture = args.capture
    if capture == "auto":
        capture = "portal" if os.environ.get("XDG_SESSION_TYPE") == "wayland" else "mss"

    if capture == "portal":
        backend: CaptureBackend = PortalPipeWireCapture(
            max_width=args.max_width,
            source=args.wayland_source,
        )
        if args.wayland_source == "selection":
            return SelectionCapture(backend)

        return backend

    return MSSCapture(monitor_index=args.monitor)


def select_capture_region(frame: np.ndarray) -> tuple[int, int, int, int]:
    print("Selection: draw a region, then press Enter or Space. Press Esc for full frame.", flush=True)
    with suppress_stderr():
        x, y, width, height = cv2.selectROI(
            "vplayer selection",
            frame,
            showCrosshair=True,
            fromCenter=False,
        )
        cv2.destroyWindow("vplayer selection")

    if width <= 0 or height <= 0:
        print("Selection: no region selected; using full frame.", flush=True)
        frame_height, frame_width = frame.shape[:2]
        return 0, 0, frame_width, frame_height

    print(f"Selection: using region {width}x{height} at ({x}, {y}).", flush=True)
    return int(x), int(y), int(width), int(height)


def resize_to_width(frame: np.ndarray, max_width: int) -> np.ndarray:
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame

    scale = max_width / width
    new_size = (max_width, max(1, int(height * scale)))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def encode_jpeg_base64(frame: np.ndarray, quality: int) -> str:
    ok, buffer = cv2.imencode(
        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    )
    if not ok:
        raise RuntimeError("Failed to encode captured frame as JPEG.")

    return base64.b64encode(buffer).decode("ascii")


def frame_is_blank(frame: np.ndarray) -> bool:
    return int(frame.max()) <= 2 and float(frame.mean()) < 1.0


def draw_status(frame: np.ndarray, status: str, latest_response: str) -> np.ndarray:
    preview = frame.copy()
    lines = [status]
    if latest_response:
        lines.extend(wrap_text(latest_response, max_chars=95)[:4])

    y = 28
    for line in lines:
        cv2.putText(
            preview,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            preview,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 24

    return preview


def wrap_text(text: str, max_chars: int) -> list[str]:
    words = text.replace("\n", " ").split()
    lines: list[str] = []
    current: list[str] = []

    for word in words:
        candidate = " ".join([*current, word])
        if len(candidate) > max_chars and current:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)

    if current:
        lines.append(" ".join(current))

    return lines


@contextlib.contextmanager
def suppress_stderr() -> object:
    saved_stderr = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved_stderr, 2)
        os.close(saved_stderr)
        os.close(devnull)


def vlm_worker(
    client: Client,
    model: str,
    prompt: str,
    analysis_width: int,
    jpeg_quality: int,
    num_predict: int,
    jobs: queue.Queue[FrameJob],
    stop_event: threading.Event,
    response_lock: threading.Lock,
    latest_response: dict[str, str],
) -> None:
    while not stop_event.is_set():
        try:
            job = jobs.get(timeout=0.2)
        except queue.Empty:
            continue

        try:
            with response_lock:
                latest_response["text"] = (
                    f"VLM: thinking with {model} on "
                    f"{job.frame.shape[1]}x{job.frame.shape[0]} frame..."
                )

            analysis_frame = resize_to_width(job.frame, analysis_width)
            image = encode_jpeg_base64(analysis_frame, jpeg_quality)
            started = time.perf_counter()
            response = client.chat(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [image],
                    }
                ],
                options={
                    "num_predict": num_predict,
                    "temperature": 0,
                },
            )
            elapsed = time.perf_counter() - started
            content = response["message"]["content"].strip()
            age = time.time() - job.captured_at
            text = (
                f"VLM: {content} "
                f"({elapsed:.1f}s inference, {analysis_frame.shape[1]}px input, "
                f"frame age {age:.1f}s)"
            )
        except Exception as exc:
            text = f"VLM error: {exc}"

        with response_lock:
            latest_response["text"] = text

        jobs.task_done()


def submit_latest(jobs: queue.Queue[FrameJob], job: FrameJob) -> None:
    try:
        jobs.put_nowait(job)
        return
    except queue.Full:
        pass

    try:
        jobs.get_nowait()
        jobs.task_done()
    except queue.Empty:
        pass

    try:
        jobs.put_nowait(job)
    except queue.Full:
        pass


def validate_args(args: argparse.Namespace) -> None:
    if args.interval <= 0:
        raise ValueError("--interval must be greater than 0.")
    if args.max_width <= 0:
        raise ValueError("--max-width must be greater than 0.")
    if args.analysis_width <= 0:
        raise ValueError("--analysis-width must be greater than 0.")
    if args.num_predict <= 0:
        raise ValueError("--num-predict must be greater than 0.")
    if args.vlm_timeout <= 0:
        raise ValueError("--vlm-timeout must be greater than 0.")
    if args.capture == "mss" and args.wayland_source != "screen":
        raise ValueError("--wayland-source is only supported with --capture portal/auto.")


def run_cli(args: argparse.Namespace) -> int:
    validate_args(args)

    stop_event = threading.Event()
    capture = create_capture_backend(args)

    client_kwargs = {"timeout": args.vlm_timeout}
    client = (
        Client(host=args.ollama_host, **client_kwargs)
        if args.ollama_host
        else Client(**client_kwargs)
    )
    jobs: queue.Queue[FrameJob] = queue.Queue(maxsize=1)
    response_lock = threading.Lock()
    latest_response = {"text": f"VLM: waiting for first frame ({args.model})..."}

    worker = threading.Thread(
        target=vlm_worker,
        args=(
            client,
            args.model,
            args.prompt,
            args.analysis_width,
            args.jpeg_quality,
            args.num_predict,
            jobs,
            stop_event,
            response_lock,
            latest_response,
        ),
        daemon=True,
    )
    worker.start()

    next_submit_at = 0.0
    last_printed_response = ""
    preview_frames_seen = 0
    blank_capture_warned = False
    first_capture_read = True
    try:
        try:
            while not stop_event.is_set():
                if first_capture_read:
                    print("Capture: reading first frame...", flush=True)
                    first_capture_read = False
                frame = capture.read()
                frame = resize_to_width(frame, args.max_width)
                capture_warning = BLANK_CAPTURE_WARNING if frame_is_blank(frame) else ""

                if capture_warning and not blank_capture_warned:
                    print(capture_warning, flush=True)
                    last_printed_response = capture_warning
                    blank_capture_warned = True

                now = time.monotonic()
                if now >= next_submit_at and not capture_warning:
                    submit_latest(jobs, FrameJob(frame=frame.copy(), captured_at=time.time()))
                    next_submit_at = now + args.interval

                with response_lock:
                    response_text = latest_response["text"]

                if capture_warning:
                    response_text = capture_warning

                source_label = SOURCE_LABELS[args.wayland_source]
                status = (
                    f"{args.model} | {source_label} | "
                    f"sample every {args.interval:.1f}s | q to quit"
                )
                if not args.no_preview:
                    preview = draw_status(frame, status, response_text)
                    if preview_frames_seen < QUIET_PREVIEW_STARTUP_FRAMES:
                        with suppress_stderr():
                            cv2.imshow("vplayer", preview)
                            key = cv2.waitKey(1)
                    else:
                        cv2.imshow("vplayer", preview)
                        key = cv2.waitKey(1)

                    preview_frames_seen += 1
                    if key & 0xFF == ord("q"):
                        stop_event.set()
                        break
                else:
                    if response_text and response_text != last_printed_response:
                        print(response_text, flush=True)
                        last_printed_response = response_text
                    time.sleep(0.03)
        except KeyboardInterrupt:
            stop_event.set()
    finally:
        with contextlib.suppress(Exception):
            capture.close()

    stop_event.set()
    worker.join(timeout=2.0)
    cv2.destroyAllWindows()
    return 0


def run_gui(args: argparse.Namespace) -> int:
    try:
        QtCore = importlib.import_module("PySide6.QtCore")
        QtGui = importlib.import_module("PySide6.QtGui")
        QtWidgets = importlib.import_module("PySide6.QtWidgets")
    except ImportError as exc:
        raise RuntimeError(
            "GUI mode requires PySide6. Install dependencies with `uv sync`."
        ) from exc

    validate_args(args)

    class VPlayerWindow(QtWidgets.QMainWindow):
        def __init__(self, initial: argparse.Namespace) -> None:
            super().__init__()
            self.setWindowTitle("vplayer")
            self.resize(1200, 760)

            self._args = initial
            self._stop_event = threading.Event()
            self._capture: CaptureBackend | None = None
            self._capture_thread: threading.Thread | None = None
            self._worker_thread: threading.Thread | None = None
            self._jobs: queue.Queue[FrameJob] | None = None
            self._response_lock = threading.Lock()
            self._frame_lock = threading.Lock()
            self._latest_response = {"text": "Ready."}
            self._latest_frame: np.ndarray | None = None
            self._next_submit_at = 0.0
            self._is_stopping = False

            self._build_ui()

            self._ui_timer = QtCore.QTimer(self)
            self._ui_timer.setInterval(33)
            self._ui_timer.timeout.connect(self._refresh_preview)
            self._ui_timer.start()

        def _build_ui(self) -> None:
            central = QtWidgets.QWidget(self)
            self.setCentralWidget(central)

            root = QtWidgets.QHBoxLayout(central)
            root.setContentsMargins(16, 16, 16, 16)
            root.setSpacing(16)

            control_card = QtWidgets.QFrame()
            control_card.setObjectName("controlCard")
            control_card.setMinimumWidth(340)
            controls = QtWidgets.QVBoxLayout(control_card)
            controls.setContentsMargins(16, 16, 16, 16)
            controls.setSpacing(10)

            title = QtWidgets.QLabel("Desktop Vision Player")
            title.setObjectName("title")
            subtitle = QtWidgets.QLabel("Live capture + local Ollama VLM")
            subtitle.setObjectName("subtitle")
            controls.addWidget(title)
            controls.addWidget(subtitle)

            self.model_edit = QtWidgets.QLineEdit(self._args.model)
            self.prompt_edit = QtWidgets.QPlainTextEdit(self._args.prompt)
            self.prompt_edit.setMinimumHeight(96)
            self.capture_combo = QtWidgets.QComboBox()
            self.capture_combo.addItems(["auto", "portal", "mss"])
            self.capture_combo.setCurrentText(self._args.capture)
            self.source_combo = QtWidgets.QComboBox()
            self.source_combo.addItems(["screen", "window", "selection"])
            self.source_combo.setCurrentText(self._args.wayland_source)
            self.interval_spin = QtWidgets.QDoubleSpinBox()
            self.interval_spin.setRange(0.2, 300.0)
            self.interval_spin.setValue(self._args.interval)
            self.interval_spin.setSingleStep(0.2)
            self.max_width_spin = QtWidgets.QSpinBox()
            self.max_width_spin.setRange(128, 4096)
            self.max_width_spin.setValue(self._args.max_width)
            self.analysis_width_spin = QtWidgets.QSpinBox()
            self.analysis_width_spin.setRange(64, 2048)
            self.analysis_width_spin.setValue(self._args.analysis_width)
            self.monitor_spin = QtWidgets.QSpinBox()
            self.monitor_spin.setRange(1, 16)
            self.monitor_spin.setValue(self._args.monitor)
            self.num_predict_spin = QtWidgets.QSpinBox()
            self.num_predict_spin.setRange(8, 2048)
            self.num_predict_spin.setValue(self._args.num_predict)
            self.jpeg_spin = QtWidgets.QSpinBox()
            self.jpeg_spin.setRange(20, 100)
            self.jpeg_spin.setValue(self._args.jpeg_quality)
            self.timeout_spin = QtWidgets.QDoubleSpinBox()
            self.timeout_spin.setRange(5.0, 600.0)
            self.timeout_spin.setValue(self._args.vlm_timeout)

            fields: list[tuple[str, object]] = [
                ("Model", self.model_edit),
                ("Prompt", self.prompt_edit),
                ("Capture", self.capture_combo),
                ("Source", self.source_combo),
                ("Monitor (mss)", self.monitor_spin),
                ("Interval (s)", self.interval_spin),
                ("Max Width", self.max_width_spin),
                ("Analysis Width", self.analysis_width_spin),
                ("Num Predict", self.num_predict_spin),
                ("JPEG Quality", self.jpeg_spin),
                ("VLM Timeout (s)", self.timeout_spin),
            ]
            for label_text, widget in fields:
                label = QtWidgets.QLabel(label_text)
                label.setObjectName("fieldLabel")
                controls.addWidget(label)
                controls.addWidget(widget)

            self.start_button = QtWidgets.QPushButton("Start")
            self.stop_button = QtWidgets.QPushButton("Stop")
            self.stop_button.setEnabled(False)
            buttons = QtWidgets.QHBoxLayout()
            buttons.addWidget(self.start_button)
            buttons.addWidget(self.stop_button)
            controls.addLayout(buttons)
            controls.addStretch(1)

            right = QtWidgets.QVBoxLayout()
            right.setSpacing(12)
            self.preview = QtWidgets.QLabel("Press Start to begin capture")
            self.preview.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.preview.setMinimumSize(720, 405)
            self.preview.setObjectName("preview")
            self.response = QtWidgets.QPlainTextEdit()
            self.response.setReadOnly(True)
            self.response.setPlaceholderText("VLM output appears here...")
            self.response.setObjectName("response")
            self.response.setMinimumHeight(160)

            right.addWidget(self.preview, stretch=1)
            right.addWidget(self.response, stretch=0)

            root.addWidget(control_card)
            root.addLayout(right, stretch=1)

            self.status_label = QtWidgets.QLabel("Idle")
            self.statusBar().addPermanentWidget(self.status_label)

            self.start_button.clicked.connect(self._start)
            self.stop_button.clicked.connect(self._stop)
            self.capture_combo.currentTextChanged.connect(self._update_control_state)
            self._update_control_state()

            self.setStyleSheet(
                """
                QMainWindow { background: #f3ede3; }
                #controlCard {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                        stop:0 #1f3a5f, stop:1 #27496d);
                    border-radius: 16px;
                }
                #title { color: #f8f5ee; font-size: 22px; font-weight: 700; }
                #subtitle { color: #d4dfeb; font-size: 13px; }
                #fieldLabel { color: #e6edf5; font-size: 12px; margin-top: 6px; }
                QLineEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                    background: rgba(255, 255, 255, 0.95);
                    border: 1px solid rgba(0, 0, 0, 0.16);
                    border-radius: 10px;
                    padding: 6px;
                    color: #0f1720;
                }
                QPushButton {
                    background: #e9c46a;
                    border: none;
                    border-radius: 10px;
                    padding: 8px 14px;
                    color: #1a1a1a;
                    font-weight: 600;
                }
                QPushButton:disabled {
                    background: #b9bfc6;
                    color: #5d646d;
                }
                #preview {
                    background: qradialgradient(cx:0.5, cy:0.4, radius:1.0,
                        fx:0.45, fy:0.35, stop:0 #3a4f61, stop:1 #121a24);
                    border-radius: 16px;
                    color: #e7ecf1;
                    font-size: 15px;
                    padding: 10px;
                }
                #response {
                    background: #fffefc;
                    border: 1px solid #d1c6b5;
                    border-radius: 14px;
                    color: #192430;
                }
                QStatusBar { background: #ece2d2; color: #243040; }
                """
            )

        def _update_control_state(self) -> None:
            is_mss = self.capture_combo.currentText() == "mss"
            self.monitor_spin.setEnabled(is_mss)
            self.source_combo.setEnabled(not is_mss)

        def _collect_args(self) -> argparse.Namespace:
            return argparse.Namespace(
                model=self.model_edit.text().strip() or DEFAULT_MODEL,
                prompt=self.prompt_edit.toPlainText().strip() or DEFAULT_PROMPT,
                monitor=int(self.monitor_spin.value()),
                capture=self.capture_combo.currentText(),
                wayland_source=self.source_combo.currentText(),
                max_width=int(self.max_width_spin.value()),
                analysis_width=int(self.analysis_width_spin.value()),
                interval=float(self.interval_spin.value()),
                num_predict=int(self.num_predict_spin.value()),
                vlm_timeout=float(self.timeout_spin.value()),
                jpeg_quality=int(self.jpeg_spin.value()),
                no_preview=True,
                ollama_host=self._args.ollama_host,
                gui=True,
            )

        def _start(self) -> None:
            if self._capture_thread and self._capture_thread.is_alive():
                return

            args_now = self._collect_args()
            try:
                validate_args(args_now)
                client_kwargs = {"timeout": args_now.vlm_timeout}
                client = (
                    Client(host=args_now.ollama_host, **client_kwargs)
                    if args_now.ollama_host
                    else Client(**client_kwargs)
                )
                capture = create_capture_backend(args_now)
            except Exception as exc:
                QtWidgets.QMessageBox.critical(self, "Unable to start", str(exc))
                return

            self._args = args_now
            self._stop_event.clear()
            self._capture = capture
            self._jobs = queue.Queue(maxsize=1)
            self._next_submit_at = 0.0
            with self._response_lock:
                self._latest_response["text"] = (
                    f"VLM: waiting for first frame ({self._args.model})..."
                )

            self._worker_thread = threading.Thread(
                target=vlm_worker,
                args=(
                    client,
                    self._args.model,
                    self._args.prompt,
                    self._args.analysis_width,
                    self._args.jpeg_quality,
                    self._args.num_predict,
                    self._jobs,
                    self._stop_event,
                    self._response_lock,
                    self._latest_response,
                ),
                daemon=True,
            )
            self._worker_thread.start()

            self._capture_thread = threading.Thread(
                target=self._capture_loop,
                daemon=True,
            )
            self._capture_thread.start()

            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.status_label.setText("Running")

        def _capture_loop(self) -> None:
            assert self._capture is not None
            while not self._stop_event.is_set():
                try:
                    frame = self._capture.read()
                    frame = resize_to_width(frame, self._args.max_width)
                except Exception as exc:
                    with self._response_lock:
                        self._latest_response["text"] = f"Capture error: {exc}"
                    self._stop_event.set()
                    break

                capture_warning = BLANK_CAPTURE_WARNING if frame_is_blank(frame) else ""
                with self._frame_lock:
                    self._latest_frame = frame.copy()

                now = time.monotonic()
                if not capture_warning and self._jobs is not None and now >= self._next_submit_at:
                    submit_latest(
                        self._jobs,
                        FrameJob(frame=frame.copy(), captured_at=time.time()),
                    )
                    self._next_submit_at = now + self._args.interval

                if capture_warning:
                    with self._response_lock:
                        self._latest_response["text"] = capture_warning

        def _refresh_preview(self) -> None:
            with self._frame_lock:
                frame = None if self._latest_frame is None else self._latest_frame.copy()
            with self._response_lock:
                response_text = self._latest_response["text"]

            self.response.setPlainText(response_text)
            source_label = SOURCE_LABELS[self._args.wayland_source]
            self.status_label.setText(
                f"{self._args.model} | {source_label} | every {self._args.interval:.1f}s"
            )

            if frame is None:
                return

            status = f"{self._args.model} | {source_label} | q in CLI only"
            preview = draw_status(frame, status, response_text)
            rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
            height, width, _ = rgb.shape
            image = QtGui.QImage(
                rgb.data,
                width,
                height,
                width * 3,
                QtGui.QImage.Format.Format_RGB888,
            ).copy()
            pixmap = QtGui.QPixmap.fromImage(image)
            self.preview.setPixmap(
                pixmap.scaled(
                    self.preview.size(),
                    QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                    QtCore.Qt.TransformationMode.SmoothTransformation,
                )
            )

            if self._stop_event.is_set() and self.stop_button.isEnabled():
                self._stop(final=True)

        def _stop(self, final: bool = False) -> None:
            if self._is_stopping:
                return
            self._is_stopping = True

            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(False)
            self.status_label.setText("Stopping...")
            self._stop_event.set()

            capture = self._capture
            self._capture = None
            if capture is not None:
                # Close capture first to unblock any pending read() call.
                with contextlib.suppress(Exception):
                    capture.close()

            if self._capture_thread and self._capture_thread.is_alive():
                self._capture_thread.join(timeout=2.0)
            if self._worker_thread and self._worker_thread.is_alive():
                self._worker_thread.join(timeout=2.0)

            self._capture_thread = None
            self._worker_thread = None
            self._jobs = None
            with self._frame_lock:
                self._latest_frame = None
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.status_label.setText("Stopped" if not final else "Stopped after error")
            self._is_stopping = False

        def closeEvent(self, event: object) -> None:
            self._stop()
            super().closeEvent(event)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = VPlayerWindow(args)
    window.show()
    return int(app.exec())


def main() -> int:
    args = parse_args()
    if args.gui:
        return run_gui(args)

    return run_cli(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
