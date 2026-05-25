# rust_wthisper

Rust microphone capture with `cpal 0.17.3`, followed by transcription through the
local Whisper CLI.

## Setup

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -U pip openai-whisper
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip openai-whisper
.\.venv\Scripts\python.exe -c "import whisper; whisper.load_model('tiny.en')"
```

For text-to-speech on the second page, the server auto-installs piper-tts and
auto-downloads `en_US-ryan-medium` on first start. You can also pre-install:

```powershell
.\.venv\Scripts\python.exe -m pip install piper-tts
```

The TTS page lets you switch between piper voice models (en_US-amy-low,
en_US-amy-medium, en_US-ryan-medium, en_US-lessac-medium, en_US-libritts-high)
and download additional ones on demand. Models are stored under `models/tts/`.

The third page (Voice Clone, F5-TTS) auto-installs `f5-tts` on first start.
Heads-up: that pulls in heavy deps (`transformers`, `accelerate`, etc.) and the
F5-TTS base model (~1.4 GB) is fetched on the first synthesis. Upload a clean
5–30 s mono reference WAV, paste its transcript, then paste the text you want
spoken in that voice.

## Run

```sh
cargo run
```

By default, the app starts the web UI at `http://127.0.0.1:3030`.
Recording waits for loud speech, keeps the speech clip, then stops
after 1.5 seconds of quiet audio before sending the WAV to Whisper.
The web page also includes a clickable activation meter from `1` to `100`; the
browser maps that to the Rust voice threshold range used by `/api/threshold`.
There is also a silence tail slider from `1.0` to `5.0` seconds backed by
`/api/silence-tail`.

## Web UI

Build the WASM frontend after changing `frontend/src/lib.rs`:

```powershell
cargo build --manifest-path frontend\Cargo.toml --target wasm32-unknown-unknown --release
wasm-bindgen --target web --out-dir pkg frontend\target\wasm32-unknown-unknown\release\rust_wthisper_frontend.wasm
```

Then build the native server:

```powershell
cargo build --release
```

```sh
cargo run
```

Open `http://127.0.0.1:3030`.

After building with `cargo build --release`, run:

```powershell
.\target\release\rust_wthisper.exe
```

Useful options:

```sh
cargo run -- --model tiny --language en
cargo run -- --model tiny.en
cargo run -- --model small.en
cargo run -- --model medium.en
cargo run -- --model large
cargo run -- --model large-v3-turbo
cargo run -- --model models/tiny.en.pt
cargo run -- --voice-threshold 0.08 --silence-tail 1.5
cargo run -- --fixed-duration --seconds 5 --output target/test.wav
cargo run -- --detect-language
cargo run -- --whisper-bin .venv/Scripts/whisper.exe
cargo run -- --port 4040
```

The default model is `tiny.en` for faster transcription. If a local `tiny.en.pt` file is in the
project folder, `models/`, `whisper/`, or the parent folder, the program uses
that file automatically. Otherwise Whisper uses its normal model cache/download
behavior.
