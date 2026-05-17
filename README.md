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
