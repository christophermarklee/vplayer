# vplayer

Real-time desktop capture for GNOME Wayland/X11 with frames sent to a local
Ollama vision model.

The current working setup is:

- Fedora/GNOME Wayland capture through `xdg-desktop-portal` + PipeWire
- GPU-backed Ollama
- `gemma3:4b` as the default local vision model
- OpenCV preview with the latest VLM response drawn on top

## Requirements

- Python managed by `uv`
- Ollama running locally at `127.0.0.1:11434`
- A vision-capable Ollama model, currently tested with:

```bash
ollama pull gemma3:4b
```

For NVIDIA GPU acceleration, verify:

```bash
ollama ps
```

The model should show `100% GPU`, not `100% CPU`.

## Run

Whole screen:

```bash
uv run python main.py --capture portal --wayland-source Screen
```

Specific window:

```bash
uv run python main.py --capture portal --wayland-source Window
```

Custom desktop region:

```bash
uv run python main.py --capture portal --wayland-source Selection
```

`Selection` first asks GNOME for screen capture permission, then opens a
one-time OpenCV selector. Draw the region, then press Enter or Space. Press Esc
to use the full frame.

## Useful Options

```bash
uv run python main.py \
  --capture portal \
  --wayland-source Selection \
  --model gemma3:4b \
  --analysis-width 384 \
  --interval 5 \
  --num-predict 80
```

- `--wayland-source`: `Selection`, `Window`, or `Screen`
- `--analysis-width`: frame width sent to the VLM; higher is sharper but slower
- `--max-width`: frame width used for preview/processing
- `--interval`: seconds between VLM requests
- `--num-predict`: maximum response tokens per frame
- `--no-preview`: print VLM output in the terminal instead of opening the player

## X11 / mss Fallback

The `mss` backend is still available for X11:

```bash
uv run python main.py --capture mss
```

On GNOME Wayland, `mss` usually returns black frames by design. Use
`--capture portal`.

## Troubleshooting

If the player is black:

```bash
uv run python main.py --capture portal --wayland-source Screen
```

If the model is slow:

```bash
ollama ps
```

Make sure the model shows `100% GPU`. If it shows CPU, Ollama is not using the
NVIDIA backend.

If `llama3.2-vision` fails with `unknown model architecture: 'mllama'`, use
`gemma3:4b`. The current upstream Ollama install works with `gemma3:4b` on the
RTX 4090 in this project.
