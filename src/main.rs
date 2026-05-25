use std::{
    env, fs, io,
    io::{BufRead, BufReader, BufWriter, Read, Write},
    net::{TcpListener, TcpStream},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::Duration,
};

use anyhow::{Context, Result, anyhow, bail};
use clap::Parser;
use cpal::{
    SampleFormat, StreamConfig,
    traits::{DeviceTrait, HostTrait, StreamTrait},
};
use hound::{SampleFormat as WavSampleFormat, WavSpec, WavWriter};

static ASSISTANT_STATE: OnceLock<Mutex<AssistantState>> = OnceLock::new();
static CURRENT_MODEL: OnceLock<Mutex<String>> = OnceLock::new();
static DOWNLOAD_STATE: OnceLock<Mutex<Option<DownloadState>>> = OnceLock::new();
static CURRENT_TTS_MODEL: OnceLock<Mutex<String>> = OnceLock::new();
static TTS_DOWNLOAD_STATE: OnceLock<Mutex<Option<DownloadState>>> = OnceLock::new();
static LISTENING_ACTIVE: AtomicBool = AtomicBool::new(false);
static UPLOAD_PENDING: AtomicBool = AtomicBool::new(false);
static HEARING_PAUSED: AtomicBool = AtomicBool::new(false);
static PIPER_INSTALLING: AtomicBool = AtomicBool::new(false);
static PIPER_INSTALL_ERROR: OnceLock<Mutex<Option<String>>> = OnceLock::new();
static F5_INSTALLING: AtomicBool = AtomicBool::new(false);
static F5_INSTALL_ERROR: OnceLock<Mutex<Option<String>>> = OnceLock::new();
static CLONE_SYNTHESIZING: AtomicBool = AtomicBool::new(false);
static SHUTTING_DOWN: AtomicBool = AtomicBool::new(false);

const DEFAULT_TTS_MODEL: &str = "en_US-ryan-medium";

const SIGINT: i32 = 2;
const SIGTERM: i32 = 15;
const MODEL_OPTIONS: [(&str, &str, u32, &str); 10] = [
    (
        "tiny",
        "tiny.en",
        75,
        "https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt",
    ),
    (
        "base",
        "base.en",
        142,
        "https://openaipublic.azureedge.net/main/whisper/models/25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt",
    ),
    (
        "small",
        "small.en",
        466,
        "https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt",
    ),
    (
        "medium",
        "medium.en",
        1500,
        "https://openaipublic.azureedge.net/main/whisper/models/d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f/medium.en.pt",
    ),
    (
        "large",
        "large",
        2900,
        "https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt",
    ),
    (
        "turbo",
        "large-v3-turbo",
        1550,
        "https://openaipublic.azureedge.net/main/whisper/models/aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a/large-v3-turbo.pt",
    ),
    (
        "any-tiny",
        "tiny",
        75,
        "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e54f811345b9bb1ace5d51b15/tiny.pt",
    ),
    (
        "any-base",
        "base",
        142,
        "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt",
    ),
    (
        "any-small",
        "small",
        466,
        "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19c1b0bb1c4eb6e7b15b2c/small.pt",
    ),
    (
        "any-medium",
        "medium",
        1500,
        "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt",
    ),
];

// (label, model id, size_mb, onnx_url, onnx_json_url)
const TTS_MODEL_OPTIONS: [(&str, &str, u32, &str, &str); 9] = [
    (
        "amy",
        "en_US-amy-low",
        22,
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/low/en_US-amy-low.onnx",
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/low/en_US-amy-low.onnx.json",
    ),
    (
        "amy-md",
        "en_US-amy-medium",
        63,
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx",
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json",
    ),
    (
        "ryan",
        "en_US-ryan-medium",
        63,
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx",
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx.json",
    ),
    (
        "lessac",
        "en_US-lessac-medium",
        63,
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
    ),
    (
        "libritts",
        "en_US-libritts-high",
        124,
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/libritts/high/en_US-libritts-high.onnx",
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/libritts/high/en_US-libritts-high.onnx.json",
    ),
    (
        "ru-irina",
        "ru_RU-irina-medium",
        63,
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/irina/medium/ru_RU-irina-medium.onnx",
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/irina/medium/ru_RU-irina-medium.onnx.json",
    ),
    (
        "ru-denis",
        "ru_RU-denis-medium",
        63,
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/denis/medium/ru_RU-denis-medium.onnx",
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/denis/medium/ru_RU-denis-medium.onnx.json",
    ),
    (
        "ru-dmitri",
        "ru_RU-dmitri-medium",
        63,
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/dmitri/medium/ru_RU-dmitri-medium.onnx",
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/dmitri/medium/ru_RU-dmitri-medium.onnx.json",
    ),
    (
        "ru-ruslan",
        "ru_RU-ruslan-medium",
        63,
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/ruslan/medium/ru_RU-ruslan-medium.onnx",
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/ruslan/medium/ru_RU-ruslan-medium.onnx.json",
    ),
];

unsafe extern "C" {
    fn signal(signum: i32, handler: extern "C" fn(i32)) -> usize;
}

extern "C" fn shutdown_signal_handler(_signal: i32) {
    SHUTTING_DOWN.store(true, Ordering::SeqCst);
}

#[derive(Debug, Clone)]
struct AssistantState {
    status: AssistantStatus,
    level: f32,
    left_level: f32,
    right_level: f32,
    channels: u16,
    sample_rate: u32,
    bits_per_sample: u16,
    sample_format: &'static str,
    device_name: String,
    voice_threshold: f32,
    silence_tail: f32,
}

#[derive(Debug, Clone)]
struct DownloadState {
    model: String,
    progress: f32,
    active: bool,
    error: Option<String>,
}

impl Default for AssistantState {
    fn default() -> Self {
        Self {
            status: AssistantStatus::Idle,
            level: 0.0,
            left_level: 0.0,
            right_level: 0.0,
            channels: 1,
            sample_rate: 0,
            bits_per_sample: 16,
            sample_format: "unknown",
            device_name: "unknown input".to_string(),
            voice_threshold: 0.08,
            silence_tail: 1.5,
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum AssistantStatus {
    Idle,
    Waiting,
    Speaking,
    Transcribing,
}

impl AssistantStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Waiting => "waiting",
            Self::Speaking => "speaking",
            Self::Transcribing => "transcribing",
        }
    }
}

#[derive(Clone, Debug, Parser)]
#[command(
    version,
    about = "Record microphone audio and transcribe it with Whisper."
)]
struct Args {
    /// Address used by the web server.
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Port used by the web server.
    #[arg(long, default_value_t = 3030)]
    port: u16,

    /// Use fixed-duration recording instead of voice activation.
    #[arg(long)]
    fixed_duration: bool,

    /// Seconds of microphone audio to record with --fixed-duration.
    #[arg(short, long, default_value_t = 10)]
    seconds: u64,

    /// RMS volume needed to activate recording. Lower is more sensitive.
    #[arg(long, default_value_t = 0.08)]
    voice_threshold: f32,

    /// Seconds of quiet audio after speech before the utterance is finished.
    #[arg(long, default_value_t = 1.5)]
    silence_tail: f32,

    /// Maximum seconds to keep one activated speech clip.
    #[arg(long, default_value_t = 30)]
    max_speech_seconds: u64,

    /// WAV file written before transcription. Defaults to the project target directory.
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Whisper model name or local .pt path, for example tiny.en, small.en, medium.en, or models/tiny.en.pt.
    #[arg(short, long, default_value = "tiny.en")]
    model: String,

    /// Spoken language hint passed to Whisper.
    #[arg(short, long, default_value = "en")]
    language: String,

    /// Let Whisper auto-detect the language instead of using --language.
    #[arg(long)]
    detect_language: bool,

    /// Whisper executable path. Auto-detects the project .venv when present.
    #[arg(long)]
    whisper_bin: Option<PathBuf>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    set_current_model(&initial_model(&args.model));
    serve(&args)
}

fn serve(args: &Args) -> Result<()> {
    install_shutdown_signal_handlers();
    SHUTTING_DOWN.store(false, Ordering::SeqCst);
    set_voice_threshold(args.voice_threshold);
    set_silence_tail(args.silence_tail);
    let address = format!("{}:{}", args.host, args.port);
    let listener =
        TcpListener::bind(&address).with_context(|| format!("failed to bind {address}"))?;
    listener
        .set_nonblocking(true)
        .context("failed to set listener nonblocking")?;

    let url = format!("http://{address}");
    println!("Serving {url}");
    println!("Press Ctrl+C to stop.");

    setup_tts(args.clone());
    setup_clone(args.clone());

    while !SHUTTING_DOWN.load(Ordering::SeqCst) {
        match listener.accept() {
            Ok((stream, _address)) => {
                stream
                    .set_nonblocking(false)
                    .context("failed to set client socket blocking")?;
                let args = args.clone();
                thread::spawn(move || {
                    if let Err(err) = handle_connection(stream, &args) {
                        eprintln!("request failed: {err:#}");
                    }
                });
            }
            Err(err) if err.kind() == io::ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(100));
            }
            Err(err) => eprintln!("connection failed: {err}"),
        }
    }

    set_assistant_status(AssistantStatus::Idle);
    set_assistant_level(0.0);
    eprintln!("server stopped");
    Ok(())
}

fn handle_connection(mut stream: TcpStream, args: &Args) -> Result<()> {
    let mut reader = BufReader::new(stream.try_clone().context("failed to clone TCP stream")?);
    let mut request_line = String::new();
    reader
        .read_line(&mut request_line)
        .context("failed to read request line")?;

    let mut content_length = 0usize;
    loop {
        let mut line = String::new();
        let bytes = reader
            .read_line(&mut line)
            .context("failed to read request header")?;
        if bytes == 0 || line == "\r\n" {
            break;
        }
        if let Some(value) = line.strip_prefix("Content-Length:") {
            content_length = value.trim().parse().unwrap_or(0);
        }
    }

    let mut request_body_bytes = Vec::new();
    if content_length > 0 {
        request_body_bytes = vec![0; content_length];
        reader
            .read_exact(&mut request_body_bytes)
            .context("failed to read request body")?;
    }
    let request_body = std::str::from_utf8(&request_body_bytes)
        .map(|s| s.to_string())
        .unwrap_or_default();

    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or_default();
    let path = parts.next().unwrap_or_default();
    let path_only = path.split('?').next().unwrap_or(path);

    match (method, path_only) {
        ("GET", "/") | ("GET", "/index.html") => write_response(
            &mut stream,
            "200 OK",
            "text/html; charset=utf-8",
            INDEX_HTML,
        ),
        ("GET", path) if path.starts_with("/pkg/") => write_static_file(&mut stream, path),
        ("GET", "/api/status") => write_response(
            &mut stream,
            "200 OK",
            "application/json",
            &assistant_state_json(),
        ),
        ("GET", "/api/models") => write_response(
            &mut stream,
            "200 OK",
            "application/json",
            &models_json(),
        ),
        ("POST", "/api/listen") => {
            if listening_is_available() {
                match listen_once_exclusive(args) {
                    Ok(transcript) => {
                        let body = format!(r#"{{"transcript":"{}"}}"#, json_escape(transcript.trim()));
                        write_response(&mut stream, "200 OK", "application/json", &body)
                    }
                    Err(err) if !listening_is_available() => {
                        set_assistant_status(AssistantStatus::Idle);
                        set_assistant_level(0.0);
                        let body = format!(
                            r#"{{"success":false,"error":"{}"}}"#,
                            json_escape(&err.to_string())
                        );
                        write_response(&mut stream, "409 Conflict", "application/json", &body)
                    }
                    Err(err) if err.to_string().contains("listener is already running") => {
                        write_response(
                            &mut stream,
                            "409 Conflict",
                            "application/json",
                            r#"{"success":false,"error":"Listener is already running"}"#,
                        )
                    }
                    Err(err) => Err(err),
                }
            } else {
                set_assistant_status(AssistantStatus::Idle);
                set_assistant_level(0.0);
                write_response(
                    &mut stream,
                    "409 Conflict",
                    "application/json",
                    r#"{"success":false,"error":"Selected model is not ready"}"#,
                )
            }
        }
        ("POST", "/api/model") => {
            if let Ok(model) = parse_string_value_from_body(&request_body, "model") {
                if MODEL_OPTIONS
                    .iter()
                    .any(|(_label, option, _size_mb, _download_url)| {
                        *option == model && model_is_available(option)
                    })
                {
                    set_current_model(&model);
                    write_response(
                        &mut stream,
                        "200 OK",
                        "application/json",
                        r#"{"success":true}"#,
                    )
                } else {
                    write_response(
                        &mut stream,
                        "400 Bad Request",
                        "application/json",
                        r#"{"success":false,"error":"Model is not available"}"#,
                    )
                }
            } else {
                write_response(
                    &mut stream,
                    "400 Bad Request",
                    "application/json",
                    r#"{"success":false,"error":"Invalid model value"}"#,
                )
            }
        }
        ("POST", "/api/download-model") => {
            if let Ok(model) = parse_string_value_from_body(&request_body, "model") {
                if MODEL_OPTIONS
                    .iter()
                    .any(|(_label, option, _size_mb, _download_url)| *option == model)
                {
                    match start_model_download(args.clone(), model.clone()) {
                        Ok(()) => write_response(
                            &mut stream,
                            "200 OK",
                            "application/json",
                            r#"{"success":true,"started":true}"#,
                        ),
                        Err(err) => {
                            let body = format!(
                                r#"{{"success":false,"error":"{}"}}"#,
                                json_escape(&err.to_string())
                            );
                            write_response(&mut stream, "500 Internal Server Error", "application/json", &body)
                        }
                    }
                } else {
                    write_response(
                        &mut stream,
                        "400 Bad Request",
                        "application/json",
                        r#"{"success":false,"error":"Unknown model"}"#,
                    )
                }
            } else {
                write_response(
                    &mut stream,
                    "400 Bad Request",
                    "application/json",
                    r#"{"success":false,"error":"Invalid model value"}"#,
                )
            }
        }
        ("POST", "/api/threshold") => {
            if let Ok(threshold) = parse_config_value_from_body(&request_body, "threshold") {
                set_voice_threshold(threshold);
                write_response(
                    &mut stream,
                    "200 OK",
                    "application/json",
                    r#"{"success":true}"#,
                )
            } else {
                write_response(
                    &mut stream,
                    "400 Bad Request",
                    "application/json",
                    r#"{"success":false,"error":"Invalid threshold value"}"#,
                )
            }
        }
        ("POST", "/api/silence-tail") => {
            if let Ok(silence_tail) = parse_config_value_from_body(&request_body, "silence_tail") {
                set_silence_tail(silence_tail);
                write_response(
                    &mut stream,
                    "200 OK",
                    "application/json",
                    r#"{"success":true}"#,
                )
            } else {
                write_response(
                    &mut stream,
                    "400 Bad Request",
                    "application/json",
                    r#"{"success":false,"error":"Invalid silence tail value"}"#,
                )
            }
        }
        ("POST", "/api/pause") => {
            HEARING_PAUSED.store(true, Ordering::SeqCst);
            write_response(
                &mut stream,
                "200 OK",
                "application/json",
                r#"{"success":true,"paused":true}"#,
            )
        }
        ("POST", "/api/resume") => {
            HEARING_PAUSED.store(false, Ordering::SeqCst);
            write_response(
                &mut stream,
                "200 OK",
                "application/json",
                r#"{"success":true,"paused":false}"#,
            )
        }
        ("GET", "/api/clone/status") => write_response(
            &mut stream,
            "200 OK",
            "application/json",
            &clone_status_json(),
        ),
        ("POST", path) if path.split('?').next() == Some("/api/clone/reference") => {
            if request_body_bytes.is_empty() {
                return write_response(
                    &mut stream,
                    "400 Bad Request",
                    "application/json",
                    r#"{"success":false,"error":"Empty reference audio"}"#,
                );
            }
            let text = parse_query_param(path, "text").unwrap_or_default();
            let text = text.trim();
            if text.is_empty() {
                return write_response(
                    &mut stream,
                    "400 Bad Request",
                    "application/json",
                    r#"{"success":false,"error":"Missing reference transcript"}"#,
                );
            }
            let ext = parse_query_param(path, "ext").unwrap_or_default();
            match save_clone_reference(&request_body_bytes, text, &ext) {
                Ok(()) => write_response(
                    &mut stream,
                    "200 OK",
                    "application/json",
                    r#"{"success":true}"#,
                ),
                Err(err) => {
                    let body = format!(
                        r#"{{"success":false,"error":"{}"}}"#,
                        json_escape(&err.to_string())
                    );
                    write_response(
                        &mut stream,
                        "500 Internal Server Error",
                        "application/json",
                        &body,
                    )
                }
            }
        }
        ("POST", "/api/clone/synthesize") => {
            let text = request_body.trim();
            if text.is_empty() {
                return write_response(
                    &mut stream,
                    "400 Bad Request",
                    "application/json",
                    r#"{"success":false,"error":"Empty text"}"#,
                );
            }
            if F5_INSTALLING.load(Ordering::SeqCst) {
                return write_response(
                    &mut stream,
                    "409 Conflict",
                    "application/json",
                    r#"{"success":false,"error":"f5-tts is still installing"}"#,
                );
            }
            if !has_clone_reference() {
                return write_response(
                    &mut stream,
                    "409 Conflict",
                    "application/json",
                    r#"{"success":false,"error":"Upload a reference voice first"}"#,
                );
            }
            match synthesize_clone(args, text) {
                Ok(wav) => write_binary_response(&mut stream, "200 OK", "audio/wav", &wav),
                Err(err) => {
                    let body = format!(
                        r#"{{"success":false,"error":"{}"}}"#,
                        json_escape(&err.to_string())
                    );
                    write_response(
                        &mut stream,
                        "500 Internal Server Error",
                        "application/json",
                        &body,
                    )
                }
            }
        }
        ("GET", "/api/tts/models") => write_response(
            &mut stream,
            "200 OK",
            "application/json",
            &tts_models_json(),
        ),
        ("POST", "/api/tts/model") => {
            if let Ok(model) = parse_string_value_from_body(&request_body, "model") {
                if TTS_MODEL_OPTIONS
                    .iter()
                    .any(|(_l, m, _s, _o, _j)| *m == model && tts_model_is_available(m))
                {
                    set_current_tts_model(&model);
                    write_response(
                        &mut stream,
                        "200 OK",
                        "application/json",
                        r#"{"success":true}"#,
                    )
                } else {
                    write_response(
                        &mut stream,
                        "400 Bad Request",
                        "application/json",
                        r#"{"success":false,"error":"TTS model is not available"}"#,
                    )
                }
            } else {
                write_response(
                    &mut stream,
                    "400 Bad Request",
                    "application/json",
                    r#"{"success":false,"error":"Invalid model value"}"#,
                )
            }
        }
        ("POST", "/api/tts/download-model") => {
            if let Ok(model) = parse_string_value_from_body(&request_body, "model") {
                if TTS_MODEL_OPTIONS
                    .iter()
                    .any(|(_l, m, _s, _o, _j)| *m == model)
                {
                    match start_tts_model_download(args.clone(), model.clone()) {
                        Ok(()) => write_response(
                            &mut stream,
                            "200 OK",
                            "application/json",
                            r#"{"success":true,"started":true}"#,
                        ),
                        Err(err) => {
                            let body = format!(
                                r#"{{"success":false,"error":"{}"}}"#,
                                json_escape(&err.to_string())
                            );
                            write_response(
                                &mut stream,
                                "500 Internal Server Error",
                                "application/json",
                                &body,
                            )
                        }
                    }
                } else {
                    write_response(
                        &mut stream,
                        "400 Bad Request",
                        "application/json",
                        r#"{"success":false,"error":"Unknown TTS model"}"#,
                    )
                }
            } else {
                write_response(
                    &mut stream,
                    "400 Bad Request",
                    "application/json",
                    r#"{"success":false,"error":"Invalid model value"}"#,
                )
            }
        }
        ("POST", "/api/tts/synthesize") => {
            let text = request_body.trim();
            if text.is_empty() {
                return write_response(
                    &mut stream,
                    "400 Bad Request",
                    "application/json",
                    r#"{"success":false,"error":"Empty text"}"#,
                );
            }
            let current = get_current_tts_model();
            if !tts_model_is_available(&current) {
                return write_response(
                    &mut stream,
                    "409 Conflict",
                    "application/json",
                    r#"{"success":false,"error":"TTS model is not downloaded"}"#,
                );
            }
            match synthesize_text(&current, text) {
                Ok(wav) => write_binary_response(&mut stream, "200 OK", "audio/wav", &wav),
                Err(err) => {
                    let body = format!(
                        r#"{{"success":false,"error":"{}"}}"#,
                        json_escape(&err.to_string())
                    );
                    write_response(
                        &mut stream,
                        "500 Internal Server Error",
                        "application/json",
                        &body,
                    )
                }
            }
        }
        ("POST", "/api/transcribe-file") => {
            if !model_is_available(&get_current_model()) {
                return write_response(
                    &mut stream,
                    "409 Conflict",
                    "application/json",
                    r#"{"success":false,"error":"Selected model is not ready"}"#,
                );
            }
            if request_body_bytes.is_empty() {
                return write_response(
                    &mut stream,
                    "400 Bad Request",
                    "application/json",
                    r#"{"success":false,"error":"Empty upload"}"#,
                );
            }
            let extension = parse_query_param(path, "ext")
                .map(|raw| sanitize_extension(&raw))
                .filter(|ext| !ext.is_empty())
                .unwrap_or_else(|| "wav".to_string());
            match transcribe_upload_exclusive(args, &extension, &request_body_bytes) {
                Ok(transcript) => {
                    let body = format!(
                        r#"{{"transcript":"{}"}}"#,
                        json_escape(transcript.trim())
                    );
                    write_response(&mut stream, "200 OK", "application/json", &body)
                }
                Err(err) => {
                    let body = format!(
                        r#"{{"success":false,"error":"{}"}}"#,
                        json_escape(&err.to_string())
                    );
                    write_response(
                        &mut stream,
                        "500 Internal Server Error",
                        "application/json",
                        &body,
                    )
                }
            }
        }
        _ => write_response(
            &mut stream,
            "404 Not Found",
            "text/plain; charset=utf-8",
            "Not found",
        ),
    }
}

fn install_shutdown_signal_handlers() {
    unsafe {
        signal(SIGINT, shutdown_signal_handler);
        signal(SIGTERM, shutdown_signal_handler);
    }
}

fn listen_once_exclusive(args: &Args) -> Result<String> {
    if LISTENING_ACTIVE
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        bail!("listener is already running");
    }

    let result = listen_once(args);
    LISTENING_ACTIVE.store(false, Ordering::SeqCst);
    result
}

fn listen_once(args: &Args) -> Result<String> {
    let output = args.output.clone().unwrap_or_else(default_output_path);
    let result = (|| {
        record_input(&output, args)?;
        set_assistant_status(AssistantStatus::Transcribing);
        set_assistant_level(0.0);
        transcribe(args, &output)
    })();
    set_assistant_status(AssistantStatus::Idle);
    set_assistant_level(0.0);
    result
}

fn record_input(output: &Path, args: &Args) -> Result<()> {
    if args.fixed_duration {
        record_fixed_duration(output, args.seconds)
    } else {
        record_voice_activated(output, args)
    }
}

fn record_voice_activated(output: &Path, args: &Args) -> Result<()> {
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create output directory {}", parent.display()))?;
    }

    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or_else(|| anyhow!("no default input device found"))?;
    let config = device
        .default_input_config()
        .context("failed to read default input config")?;

    set_assistant_device(&device, &config);
    set_assistant_status(AssistantStatus::Waiting);
    eprintln!(
        "waiting for speech from '{}' at {} Hz",
        device
            .description()
            .map(|description| description.name().to_string())
            .unwrap_or_else(|_| "unknown input".to_string()),
        config.sample_rate()
    );

    let (samples, sample_rate, channels) = collect_voice_activated_samples(&device, &config, args)?;
    write_wav(output, sample_rate, channels, samples)
}

fn collect_voice_activated_samples(
    device: &cpal::Device,
    config: &cpal::SupportedStreamConfig,
    args: &Args,
) -> Result<(Vec<i16>, u32, u16)> {
    let (tx, rx) = mpsc::channel();

    let stream_config: StreamConfig = config.clone().into();
    let err_fn = |err| eprintln!("input stream error: {err}");

    let stream = match config.sample_format() {
        SampleFormat::F32 => device.build_input_stream(
            &stream_config,
            move |data: &[f32], _| {
                let _ = tx.send(data.iter().copied().map(f32_to_i16).collect());
            },
            err_fn,
            None,
        ),
        SampleFormat::I16 => device.build_input_stream(
            &stream_config,
            move |data: &[i16], _| {
                let _ = tx.send(data.to_vec());
            },
            err_fn,
            None,
        ),
        SampleFormat::U16 => device.build_input_stream(
            &stream_config,
            move |data: &[u16], _| {
                let _ = tx.send(data.iter().copied().map(u16_to_i16).collect());
            },
            err_fn,
            None,
        ),
        other => bail!("unsupported input sample format: {other:?}"),
    }
    .context("failed to build input stream")?;

    stream.play().context("failed to start input stream")?;

    let sample_rate = config.sample_rate();
    let channels = config.channels();
    let max_samples = args.max_speech_seconds as usize * sample_rate as usize * channels as usize;
    let mut samples = Vec::new();
    let mut active = false;
    let mut quiet_samples = 0usize;

    while let Ok(chunk) = rx.recv() {
        if !listening_is_available() {
            set_assistant_status(AssistantStatus::Idle);
            set_assistant_level(0.0);
            bail!("listening paused until model download finishes");
        }

        let levels = channel_levels(&chunk, channels);
        let rms = levels.combined;
        set_assistant_levels(levels);
        let voice_threshold = get_voice_threshold();
        let silence_threshold = voice_threshold * 0.55;
        let silence_tail = get_silence_tail();
        let silence_limit = (silence_tail * sample_rate as f32 * channels as f32) as usize;

        if !active {
            if rms >= voice_threshold {
                eprintln!("speech activated at rms {:.3}", rms);
                set_assistant_status(AssistantStatus::Speaking);
                active = true;
                samples.extend_from_slice(&chunk);
            }
            continue;
        }

        if rms < silence_threshold {
            quiet_samples += chunk.len();
        } else {
            quiet_samples = 0;
        }

        samples.extend_from_slice(&chunk);

        if quiet_samples >= silence_limit {
            eprintln!("speech finished after {:.1}s silence", silence_tail);
            break;
        }

        if samples.len() >= max_samples {
            eprintln!(
                "speech reached max clip length of {}s",
                args.max_speech_seconds
            );
            break;
        }
    }

    drop(stream);
    set_assistant_level(0.0);

    if samples.is_empty() {
        bail!("no speech samples were recorded");
    }

    Ok((samples, sample_rate, channels))
}

fn record_fixed_duration(output: &Path, seconds: u64) -> Result<()> {
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create output directory {}", parent.display()))?;
    }

    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or_else(|| anyhow!("no default input device found"))?;
    let config = device
        .default_input_config()
        .context("failed to read default input config")?;

    set_assistant_status(AssistantStatus::Speaking);
    set_assistant_device(&device, &config);
    eprintln!(
        "recording {}s from '{}' at {} Hz",
        seconds,
        device
            .description()
            .map(|description| description.name().to_string())
            .unwrap_or_else(|_| "unknown input".to_string()),
        config.sample_rate()
    );

    let wav_spec = WavSpec {
        channels: config.channels(),
        sample_rate: config.sample_rate(),
        bits_per_sample: 16,
        sample_format: WavSampleFormat::Int,
    };
    let writer = WavWriter::create(output, wav_spec)
        .with_context(|| format!("failed to create {}", output.display()))?;
    let writer = Arc::new(Mutex::new(Some(writer)));
    let writer_for_stream = Arc::clone(&writer);
    let stream = build_writer_stream(&device, &config, writer_for_stream)?;

    stream.play().context("failed to start input stream")?;
    thread::sleep(Duration::from_secs(seconds));
    drop(stream);
    set_assistant_level(0.0);

    finalize_writer(writer)
}

fn build_writer_stream(
    device: &cpal::Device,
    config: &cpal::SupportedStreamConfig,
    writer: Arc<Mutex<Option<WavWriter<BufWriter<fs::File>>>>>,
) -> Result<cpal::Stream> {
    let stream_config: StreamConfig = config.clone().into();
    let err_fn = |err| eprintln!("input stream error: {err}");

    let stream = match config.sample_format() {
        SampleFormat::F32 => device.build_input_stream(
            &stream_config,
            move |data: &[f32], _| write_samples(data.iter().copied().map(f32_to_i16), &writer),
            err_fn,
            None,
        ),
        SampleFormat::I16 => device.build_input_stream(
            &stream_config,
            move |data: &[i16], _| write_samples(data.iter().copied(), &writer),
            err_fn,
            None,
        ),
        SampleFormat::U16 => device.build_input_stream(
            &stream_config,
            move |data: &[u16], _| write_samples(data.iter().copied().map(u16_to_i16), &writer),
            err_fn,
            None,
        ),
        other => bail!("unsupported input sample format: {other:?}"),
    }
    .context("failed to build input stream")?;

    Ok(stream)
}

fn finalize_writer(writer: Arc<Mutex<Option<WavWriter<BufWriter<fs::File>>>>>) -> Result<()> {
    let mut guard = writer
        .lock()
        .map_err(|_| anyhow!("failed to lock WAV writer after recording"))?;
    let writer = guard
        .take()
        .ok_or_else(|| anyhow!("WAV writer was already finalized"))?;
    writer.finalize().context("failed to finalize WAV file")?;

    Ok(())
}

fn write_wav(output: &Path, sample_rate: u32, channels: u16, samples: Vec<i16>) -> Result<()> {
    let wav_spec = WavSpec {
        channels,
        sample_rate,
        bits_per_sample: 16,
        sample_format: WavSampleFormat::Int,
    };
    let mut writer = WavWriter::create(output, wav_spec)
        .with_context(|| format!("failed to create {}", output.display()))?;

    for sample in samples {
        writer
            .write_sample(sample)
            .context("failed to write WAV sample")?;
    }

    writer.finalize().context("failed to finalize WAV file")
}

#[derive(Debug, Clone, Copy)]
struct AudioLevels {
    combined: f32,
    left: f32,
    right: f32,
}

fn channel_levels(samples: &[i16], channels: u16) -> AudioLevels {
    if channels <= 1 {
        let combined = rms_level(samples);
        return AudioLevels {
            combined,
            left: combined,
            right: 0.0,
        };
    }

    let channels = channels as usize;
    let mut left_sum = 0.0f32;
    let mut right_sum = 0.0f32;
    let mut left_count = 0usize;
    let mut right_count = 0usize;

    for frame in samples.chunks(channels) {
        if let Some(sample) = frame.first() {
            let normalized = *sample as f32 / i16::MAX as f32;
            left_sum += normalized * normalized;
            left_count += 1;
        }

        if let Some(sample) = frame.get(1) {
            let normalized = *sample as f32 / i16::MAX as f32;
            right_sum += normalized * normalized;
            right_count += 1;
        }
    }

    let left = rms_from_sum(left_sum, left_count);
    let right = rms_from_sum(right_sum, right_count);

    AudioLevels {
        combined: left.max(right),
        left,
        right,
    }
}

fn rms_from_sum(sum: f32, count: usize) -> f32 {
    if count == 0 {
        0.0
    } else {
        (sum / count as f32).sqrt()
    }
}

fn rms_level(samples: &[i16]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }

    let sum = samples
        .iter()
        .map(|sample| {
            let normalized = *sample as f32 / i16::MAX as f32;
            normalized * normalized
        })
        .sum::<f32>();

    (sum / samples.len() as f32).sqrt()
}

fn write_samples<I>(samples: I, writer: &Arc<Mutex<Option<WavWriter<BufWriter<fs::File>>>>>)
where
    I: IntoIterator<Item = i16>,
{
    let Ok(mut guard) = writer.lock() else {
        return;
    };
    let Some(writer) = guard.as_mut() else {
        return;
    };

    for sample in samples {
        if writer.write_sample(sample).is_err() {
            break;
        }
    }
}

fn f32_to_i16(sample: f32) -> i16 {
    (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16
}

fn u16_to_i16(sample: u16) -> i16 {
    (sample as i32 - 32768) as i16
}

fn transcribe(args: &Args, output: &Path) -> Result<String> {
    let whisper_bin = args
        .whisper_bin
        .clone()
        .or_else(default_local_whisper)
        .unwrap_or_else(|| PathBuf::from("whisper"));

    let transcript_dir = output
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));

    let mut command = Command::new(&whisper_bin);
    let current_model_name = get_current_model();
    let model = resolve_model(&current_model_name);
    let is_english_only = current_model_name.ends_with(".en");

    command
        .arg(output)
        .arg("--model")
        .arg(&model)
        .arg("--output_format")
        .arg("txt")
        .arg("--output_dir")
        .arg(&transcript_dir)
        .arg("--fp16")
        .arg("False")
        .arg("--verbose")
        .arg("False")
        .stdout(Stdio::null())
        .stderr(Stdio::null());

    if !args.detect_language {
        if is_english_only {
            // .en checkpoints are English-only — always force the language flag.
            command.arg("--language").arg(&args.language);
        } else if args.language != "en" {
            // Multilingual model: only force a language if the user explicitly
            // chose a non-default one on the CLI; otherwise let whisper detect.
            command.arg("--language").arg(&args.language);
        }
    }

    let status = command
        .status()
        .with_context(|| {
            format!(
                "failed to run {}. Install openai-whisper in .venv or pass --whisper-bin .venv/Scripts/whisper.exe",
                whisper_bin.display()
            )
        })?;
    if !status.success() {
        bail!("Whisper exited with status {status}");
    }

    let transcript_path = output.with_extension("txt");
    fs::read_to_string(&transcript_path)
        .with_context(|| format!("failed to read transcript {}", transcript_path.display()))
}

fn start_model_download(args: Args, model: String) -> Result<()> {
    if model_is_available(&model) {
        set_download_finished(&model);
        return Ok(());
    }

    if let Some(state) = download_state()
        && state.active
    {
        return Ok(());
    }

    set_download_started(&model);
    thread::spawn(move || {
        if let Err(err) = download_model(&args, &model) {
            set_download_error(&model, err.to_string());
        }
    });

    Ok(())
}

fn download_model(args: &Args, model: &str) -> Result<()> {
    let python = args
        .whisper_bin
        .as_ref()
        .and_then(python_for_whisper_bin)
        .or_else(default_local_python)
        .unwrap_or_else(|| PathBuf::from("python"));

    let mut child = Command::new(&python)
        .arg("-c")
        .arg("import sys, whisper; whisper.load_model(sys.argv[1])")
        .arg(model)
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("failed to run {}", python.display()))?;

    if let Some(stderr) = child.stderr.take() {
        let mut reader = BufReader::new(stderr);
        let mut chunk = Vec::new();

        loop {
            chunk.clear();
            let read = reader
                .read_until(b'\r', &mut chunk)
                .context("failed to read model download progress")?;
            if read == 0 {
                break;
            }

            let text = String::from_utf8_lossy(&chunk);
            if let Some(progress) = parse_progress_percent(&text) {
                set_download_progress(model, progress);
            }
        }
    }

    let status = child
        .wait()
        .context("failed to wait for model download process")?;

    if !status.success() {
        bail!("model download exited with status {status}");
    }

    set_download_finished(model);
    Ok(())
}

fn parse_progress_percent(text: &str) -> Option<f32> {
    let percent_index = text.find('%')?;
    let before_percent = &text[..percent_index];
    let digits = before_percent
        .chars()
        .rev()
        .skip_while(|ch| ch.is_whitespace())
        .take_while(|ch| ch.is_ascii_digit() || *ch == '.')
        .collect::<String>();
    let value = digits.chars().rev().collect::<String>().parse::<f32>().ok()?;
    Some(value.clamp(0.0, 100.0))
}

fn python_for_whisper_bin(whisper_bin: &PathBuf) -> Option<PathBuf> {
    let parent = whisper_bin.parent()?;
    [
        parent.join("python.exe"),
        parent.join("python"),
        parent.parent()?.join("bin").join("python"),
    ]
    .into_iter()
    .find(|candidate| candidate.exists())
}

fn default_local_python() -> Option<PathBuf> {
    let mut starts = Vec::new();

    if let Ok(cwd) = env::current_dir() {
        starts.push(cwd);
    }

    if let Ok(exe) = env::current_exe()
        && let Some(parent) = exe.parent()
    {
        starts.push(parent.to_path_buf());
    }

    starts.into_iter().find_map(|start| {
        find_upward(start.clone(), ".venv/Scripts/python.exe")
            .or_else(|| find_upward(start.clone(), ".venv/Scripts/python"))
            .or_else(|| find_upward(start, ".venv/bin/python"))
    })
}

fn default_local_whisper() -> Option<PathBuf> {
    let mut starts = Vec::new();

    if let Ok(cwd) = env::current_dir() {
        starts.push(cwd);
    }

    if let Ok(exe) = env::current_exe()
        && let Some(parent) = exe.parent()
    {
        starts.push(parent.to_path_buf());
    }

    starts
        .into_iter()
        .find_map(|start| {
            find_upward(start.clone(), ".venv/Scripts/whisper.exe")
                .or_else(|| find_upward(start.clone(), ".venv/Scripts/whisper"))
                .or_else(|| find_upward(start.clone(), ".venv/bin/whisper"))
                .or_else(|| find_nearby_file(start, "whisper.exe"))
        })
}

fn find_upward(start: PathBuf, relative: &str) -> Option<PathBuf> {
    start
        .ancestors()
        .map(|ancestor| ancestor.join(relative))
        .find(|candidate| candidate.exists())
}

fn resolve_model(model: &str) -> String {
    let model_path = PathBuf::from(model);
    if model_path.exists() {
        return model_path.display().to_string();
    }

    if model_path.extension().is_some() {
        return model.to_string();
    }

    find_model_file(model)
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| model.to_string())
}

fn model_is_available(model: &str) -> bool {
    let model_path = PathBuf::from(model);
    model_path.exists() || find_model_file(model).is_some()
}

fn initial_model(requested_model: &str) -> String {
    if model_is_available(requested_model) {
        return requested_model.to_string();
    }

    MODEL_OPTIONS
        .iter()
        .map(|(_label, model, _size_mb, _download_url)| *model)
        .find(|model| model_is_available(model))
        .unwrap_or(requested_model)
        .to_string()
}

fn find_model_file(model: &str) -> Option<PathBuf> {
    let model_file = format!("{model}.pt");
    let mut starts = Vec::new();

    if let Ok(cwd) = env::current_dir() {
        starts.push(cwd);
    }

    if let Some(root) = project_root() {
        starts.push(root);
    }

    if let Ok(exe) = env::current_exe()
        && let Some(parent) = exe.parent()
    {
        starts.push(parent.to_path_buf());
    }

    starts
        .into_iter()
        .find_map(|start| find_nearby_file(start, &model_file))
        .or_else(|| cached_model_file(&model_file))
}

fn cached_model_file(file_name: &str) -> Option<PathBuf> {
    let mut cache_dirs = Vec::new();

    if let Ok(user_profile) = env::var("USERPROFILE") {
        cache_dirs.push(PathBuf::from(user_profile).join(".cache").join("whisper"));
    }

    if let Ok(home) = env::var("HOME") {
        cache_dirs.push(PathBuf::from(home).join(".cache").join("whisper"));
    }

    if let Ok(cache_home) = env::var("XDG_CACHE_HOME") {
        cache_dirs.push(PathBuf::from(cache_home).join("whisper"));
    }

    cache_dirs
        .into_iter()
        .map(|dir| dir.join(file_name))
        .find(|candidate| candidate.exists())
}

fn find_nearby_file(start: PathBuf, file_name: &str) -> Option<PathBuf> {
    let parent = start.parent();
    [
        start.join(file_name),
        start.join("models").join(file_name),
        start.join("whisper").join(file_name),
        parent.map(|path| path.join(file_name)).unwrap_or_default(),
        parent
            .map(|path| path.join("models").join(file_name))
            .unwrap_or_default(),
        parent
            .map(|path| path.join("whisper").join(file_name))
            .unwrap_or_default(),
    ]
    .into_iter()
    .find(|candidate| candidate.exists())
}

fn default_output_path() -> PathBuf {
    project_root()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("target/mic.wav")
}

fn project_root() -> Option<PathBuf> {
    let mut starts = Vec::new();

    if let Ok(cwd) = env::current_dir() {
        starts.push(cwd);
    }

    if let Ok(exe) = env::current_exe()
        && let Some(parent) = exe.parent()
    {
        starts.push(parent.to_path_buf());
    }

    starts.into_iter().find_map(|start| {
        start
            .ancestors()
            .find(|ancestor| ancestor.join("Cargo.toml").exists())
            .map(Path::to_path_buf)
    })
}

fn write_response(
    stream: &mut TcpStream,
    status: &str,
    content_type: &str,
    body: &str,
) -> Result<()> {
    write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
    .context("failed to write HTTP response")
}

fn write_binary_response(
    stream: &mut TcpStream,
    status: &str,
    content_type: &str,
    body: &[u8],
) -> Result<()> {
    write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    )
    .context("failed to write HTTP headers")?;
    stream.write_all(body).context("failed to write HTTP body")
}

fn write_static_file(stream: &mut TcpStream, request_path: &str) -> Result<()> {
    let relative = request_path
        .trim_start_matches('/')
        .split('?')
        .next()
        .unwrap_or_default();

    if relative.contains("..") || relative.contains('\\') {
        return write_response(
            stream,
            "400 Bad Request",
            "text/plain; charset=utf-8",
            "Bad request",
        );
    }

    let path = project_root()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(relative);

    if !path.exists() {
        return write_response(
            stream,
            "404 Not Found",
            "text/plain; charset=utf-8",
            "Not found",
        );
    }

    let content_type = match path.extension().and_then(|extension| extension.to_str()) {
        Some("js") => "text/javascript; charset=utf-8",
        Some("wasm") => "application/wasm",
        _ => "application/octet-stream",
    };

    let body = fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
    write_binary_response(stream, "200 OK", content_type, &body)
}

fn json_escape(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            ch if ch.is_control() => escaped.push_str(&format!("\\u{:04x}", ch as u32)),
            ch => escaped.push(ch),
        }
    }
    escaped
}

fn parse_config_value_from_body(body: &str, key: &str) -> Result<f32> {
    let body = body.trim();
    if body.is_empty() {
        bail!("empty request body");
    }

    if let Ok(value) = body.parse::<f32>() {
        return Ok(value);
    }

    let prefix = format!("{{\"{key}\":");
    if let Some(value_str) = body.strip_prefix(&prefix).and_then(|s| s.strip_suffix("}")) {
        value_str
            .trim()
            .parse::<f32>()
            .with_context(|| format!("failed to parse {key} as f32"))
    } else {
        bail!("invalid JSON format");
    }
}

fn parse_string_value_from_body(body: &str, key: &str) -> Result<String> {
    let body = body.trim();
    if body.is_empty() {
        bail!("empty request body");
    }

    let prefix = format!("{{\"{key}\":\"");
    if let Some(value) = body.strip_prefix(&prefix).and_then(|s| s.strip_suffix("\"}")) {
        Ok(value.replace("\\\"", "\"").replace("\\\\", "\\"))
    } else {
        bail!("invalid JSON format");
    }
}

fn state_cell() -> &'static Mutex<AssistantState> {
    ASSISTANT_STATE.get_or_init(|| Mutex::new(AssistantState::default()))
}

fn current_model_cell() -> &'static Mutex<String> {
    CURRENT_MODEL.get_or_init(|| Mutex::new("tiny.en".to_string()))
}

fn download_state_cell() -> &'static Mutex<Option<DownloadState>> {
    DOWNLOAD_STATE.get_or_init(|| Mutex::new(None))
}

fn download_state() -> Option<DownloadState> {
    download_state_cell()
        .lock()
        .ok()
        .and_then(|state| state.clone())
}

fn set_download_started(model: &str) {
    set_assistant_status(AssistantStatus::Idle);
    set_assistant_level(0.0);

    if let Ok(mut state) = download_state_cell().lock() {
        *state = Some(DownloadState {
            model: model.to_string(),
            progress: 0.0,
            active: true,
            error: None,
        });
    }
}

fn set_download_progress(model: &str, progress: f32) {
    if let Ok(mut state) = download_state_cell().lock()
        && let Some(current) = state.as_mut()
        && current.model == model
    {
        current.progress = progress.clamp(0.0, 100.0);
    }
}

fn set_download_finished(model: &str) {
    set_current_model(model);

    if let Ok(mut state) = download_state_cell().lock() {
        *state = Some(DownloadState {
            model: model.to_string(),
            progress: 100.0,
            active: false,
            error: None,
        });
    }
}

fn set_download_error(model: &str, error: String) {
    if let Ok(mut state) = download_state_cell().lock() {
        *state = Some(DownloadState {
            model: model.to_string(),
            progress: 0.0,
            active: false,
            error: Some(error),
        });
    }
}

fn set_current_model(model: &str) {
    if let Ok(mut current) = current_model_cell().lock() {
        *current = model.to_string();
    }
}

fn get_current_model() -> String {
    current_model_cell()
        .lock()
        .map(|model| model.clone())
        .unwrap_or_else(|_| "tiny.en".to_string())
}

fn listening_is_available() -> bool {
    if UPLOAD_PENDING.load(Ordering::SeqCst) {
        return false;
    }

    if HEARING_PAUSED.load(Ordering::SeqCst) {
        return false;
    }

    if let Some(state) = download_state()
        && state.active
    {
        return false;
    }

    model_is_available(&get_current_model())
}

fn transcribe_upload_exclusive(args: &Args, extension: &str, bytes: &[u8]) -> Result<String> {
    UPLOAD_PENDING.store(true, Ordering::SeqCst);

    for _ in 0..200 {
        if !LISTENING_ACTIVE.load(Ordering::SeqCst) {
            break;
        }
        thread::sleep(Duration::from_millis(50));
    }

    if LISTENING_ACTIVE
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        UPLOAD_PENDING.store(false, Ordering::SeqCst);
        bail!("listener is still active");
    }

    let result = (|| -> Result<String> {
        let path = save_upload(extension, bytes)?;
        set_assistant_status(AssistantStatus::Transcribing);
        set_assistant_level(0.0);
        transcribe(args, &path)
    })();

    LISTENING_ACTIVE.store(false, Ordering::SeqCst);
    UPLOAD_PENDING.store(false, Ordering::SeqCst);
    set_assistant_status(AssistantStatus::Idle);
    set_assistant_level(0.0);
    result
}

fn save_upload(extension: &str, bytes: &[u8]) -> Result<PathBuf> {
    let target_dir = project_root()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("target");
    fs::create_dir_all(&target_dir)
        .with_context(|| format!("failed to create upload directory {}", target_dir.display()))?;
    let path = target_dir.join(format!("upload.{extension}"));
    fs::write(&path, bytes)
        .with_context(|| format!("failed to write upload {}", path.display()))?;
    Ok(path)
}

fn sanitize_extension(value: &str) -> String {
    value
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .take(8)
        .collect::<String>()
        .to_ascii_lowercase()
}

fn parse_query_param(path: &str, key: &str) -> Option<String> {
    let query = path.split('?').nth(1)?;
    for pair in query.split('&') {
        let mut parts = pair.splitn(2, '=');
        let candidate = parts.next()?;
        let value = parts.next().unwrap_or("");
        if candidate == key {
            return Some(url_decode(value));
        }
    }
    None
}

fn url_decode(input: &str) -> String {
    let mut output = String::with_capacity(input.len());
    let bytes = input.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        let byte = bytes[index];
        if byte == b'+' {
            output.push(' ');
            index += 1;
            continue;
        }
        if byte == b'%' && index + 2 < bytes.len() {
            if let (Some(hi), Some(lo)) = (
                hex_value(bytes[index + 1]),
                hex_value(bytes[index + 2]),
            ) {
                output.push(char::from(hi * 16 + lo));
                index += 3;
                continue;
            }
        }
        output.push(char::from(byte));
        index += 1;
    }
    output
}

fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn set_assistant_status(status: AssistantStatus) {
    if let Ok(mut current) = state_cell().lock() {
        current.status = status;
    }
}

fn set_assistant_level(level: f32) {
    if let Ok(mut current) = state_cell().lock() {
        current.level = level.clamp(0.0, 1.0);
        current.left_level = current.level;
        current.right_level = 0.0;
    }
}

fn set_assistant_levels(levels: AudioLevels) {
    if let Ok(mut current) = state_cell().lock() {
        current.level = levels.combined.clamp(0.0, 1.0);
        current.left_level = levels.left.clamp(0.0, 1.0);
        current.right_level = levels.right.clamp(0.0, 1.0);
    }
}

fn set_assistant_device(device: &cpal::Device, config: &cpal::SupportedStreamConfig) {
    if let Ok(mut current) = state_cell().lock() {
        current.channels = config.channels().max(1);
        current.sample_rate = config.sample_rate();
        current.bits_per_sample = match config.sample_format() {
            SampleFormat::F32 => 32,
            SampleFormat::I16 | SampleFormat::U16 => 16,
            _ => 0,
        };
        current.sample_format = match config.sample_format() {
            SampleFormat::F32 => "float",
            SampleFormat::I16 => "signed int",
            SampleFormat::U16 => "unsigned int",
            _ => "unknown",
        };
        current.device_name = device
            .description()
            .map(|description| description.name().to_string())
            .unwrap_or_else(|_| "unknown input".to_string());
    }
}

fn set_voice_threshold(threshold: f32) {
    if let Ok(mut current) = state_cell().lock() {
        current.voice_threshold = threshold.clamp(0.001, 0.1);
    }
}

fn set_silence_tail(silence_tail: f32) {
    if let Ok(mut current) = state_cell().lock() {
        current.silence_tail = silence_tail.clamp(1.0, 5.0);
    }
}

fn get_voice_threshold() -> f32 {
    state_cell()
        .lock()
        .map(|state| state.voice_threshold)
        .unwrap_or(0.08)
}

fn get_silence_tail() -> f32 {
    state_cell()
        .lock()
        .map(|state| state.silence_tail)
        .unwrap_or(1.5)
}

fn assistant_state() -> AssistantState {
    state_cell()
        .lock()
        .map(|state| state.clone())
        .unwrap_or_default()
}

fn assistant_state_json() -> String {
    let state = assistant_state();
    let paused = HEARING_PAUSED.load(Ordering::SeqCst);
    format!(
        r#"{{"status":"{}","level":{:.5},"left_level":{:.5},"right_level":{:.5},"channels":{},"sample_rate":{},"bits_per_sample":{},"sample_format":"{}","device_name":"{}","voice_threshold":{:.5},"silence_tail":{:.1},"paused":{}}}"#,
        state.status.as_str(),
        state.level,
        state.left_level,
        state.right_level,
        state.channels,
        state.sample_rate,
        state.bits_per_sample,
        json_escape(state.sample_format),
        json_escape(&state.device_name),
        state.voice_threshold,
        state.silence_tail,
        paused
    )
}

fn models_json() -> String {
    let current_model = get_current_model();
    let download = download_state();
    let models = MODEL_OPTIONS
        .iter()
        .map(|(label, model, size_mb, download_url)| {
            let downloading = download
                .as_ref()
                .map(|state| state.active && state.model == *model)
                .unwrap_or(false);
            let progress = download
                .as_ref()
                .filter(|state| state.model == *model)
                .map(|state| state.progress)
                .unwrap_or(0.0);
            let error = download
                .as_ref()
                .filter(|state| state.model == *model)
                .and_then(|state| state.error.as_ref())
                .map(|error| format!(r#""{}""#, json_escape(error)))
                .unwrap_or_else(|| "null".to_string());

            format!(
                r#"{{"label":"{}","model":"{}","size_mb":{},"download_url":"{}","available":{},"downloading":{},"progress":{:.1},"error":{}}}"#,
                json_escape(label),
                json_escape(model),
                size_mb,
                json_escape(download_url),
                model_is_available(model),
                downloading,
                progress,
                error
            )
        })
        .collect::<Vec<_>>()
        .join(",");

    format!(
        r#"{{"current":"{}","models":[{}]}}"#,
        json_escape(&current_model),
        models
    )
}

fn current_tts_model_cell() -> &'static Mutex<String> {
    CURRENT_TTS_MODEL.get_or_init(|| Mutex::new(DEFAULT_TTS_MODEL.to_string()))
}

fn tts_download_state_cell() -> &'static Mutex<Option<DownloadState>> {
    TTS_DOWNLOAD_STATE.get_or_init(|| Mutex::new(None))
}

fn tts_download_state() -> Option<DownloadState> {
    tts_download_state_cell()
        .lock()
        .ok()
        .and_then(|state| state.clone())
}

fn set_current_tts_model(model: &str) {
    if let Ok(mut current) = current_tts_model_cell().lock() {
        *current = model.to_string();
    }
}

fn get_current_tts_model() -> String {
    current_tts_model_cell()
        .lock()
        .map(|model| model.clone())
        .unwrap_or_else(|_| DEFAULT_TTS_MODEL.to_string())
}

fn piper_install_error_cell() -> &'static Mutex<Option<String>> {
    PIPER_INSTALL_ERROR.get_or_init(|| Mutex::new(None))
}

fn set_piper_install_error(error: Option<String>) {
    if let Ok(mut current) = piper_install_error_cell().lock() {
        *current = error;
    }
}

fn get_piper_install_error() -> Option<String> {
    piper_install_error_cell()
        .lock()
        .ok()
        .and_then(|inner| inner.clone())
}

fn install_piper(args: &Args) -> Result<()> {
    let python = args
        .whisper_bin
        .as_ref()
        .and_then(python_for_whisper_bin)
        .or_else(default_local_python)
        .ok_or_else(|| anyhow!("could not find .venv python; create it first"))?;

    eprintln!("Installing piper-tts via {}...", python.display());
    let status = Command::new(&python)
        .arg("-m")
        .arg("pip")
        .arg("install")
        .arg("--upgrade")
        .arg("piper-tts")
        .status()
        .with_context(|| format!("failed to run {}", python.display()))?;

    if !status.success() {
        bail!("pip install piper-tts exited with status {status}");
    }
    Ok(())
}

fn setup_tts(args: Args) {
    thread::spawn(move || {
        if default_local_piper().is_none() {
            PIPER_INSTALLING.store(true, Ordering::SeqCst);
            set_piper_install_error(None);
            let result = install_piper(&args);
            PIPER_INSTALLING.store(false, Ordering::SeqCst);
            match result {
                Ok(()) => {
                    eprintln!("piper-tts installed");
                }
                Err(err) => {
                    let msg = format!("{err:#}");
                    eprintln!("piper install failed: {msg}");
                    set_piper_install_error(Some(msg));
                    return;
                }
            }
        }

        let any_available = TTS_MODEL_OPTIONS
            .iter()
            .any(|(_l, m, _s, _o, _j)| tts_model_is_available(m));
        if any_available {
            return;
        }

        eprintln!("Auto-downloading default TTS model: {DEFAULT_TTS_MODEL}");
        if let Err(err) = start_tts_model_download(args, DEFAULT_TTS_MODEL.to_string()) {
            eprintln!("Failed to start default TTS model download: {err:#}");
        }
    });
}

fn f5_install_error_cell() -> &'static Mutex<Option<String>> {
    F5_INSTALL_ERROR.get_or_init(|| Mutex::new(None))
}

fn set_f5_install_error(error: Option<String>) {
    if let Ok(mut current) = f5_install_error_cell().lock() {
        *current = error;
    }
}

fn get_f5_install_error() -> Option<String> {
    f5_install_error_cell()
        .lock()
        .ok()
        .and_then(|inner| inner.clone())
}

fn clone_dir() -> PathBuf {
    project_root()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("target")
        .join("clone")
}

fn clone_reference_text_path() -> PathBuf {
    clone_dir().join("reference.txt")
}

fn clone_output_path() -> PathBuf {
    clone_dir().join("output.wav")
}

fn find_clone_reference_audio() -> Option<PathBuf> {
    let dir = clone_dir();
    let entries = fs::read_dir(&dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if !name.starts_with("reference.") || name == "reference.txt" {
            continue;
        }
        if path.is_file() {
            return Some(path);
        }
    }
    None
}

fn has_clone_reference() -> bool {
    find_clone_reference_audio().is_some() && clone_reference_text_path().exists()
}

fn get_clone_reference_text() -> Option<String> {
    fs::read_to_string(clone_reference_text_path()).ok()
}

fn default_local_f5tts() -> Option<PathBuf> {
    let mut starts = Vec::new();

    if let Ok(cwd) = env::current_dir() {
        starts.push(cwd);
    }
    if let Ok(exe) = env::current_exe()
        && let Some(parent) = exe.parent()
    {
        starts.push(parent.to_path_buf());
    }

    starts.into_iter().find_map(|start| {
        find_upward(start.clone(), ".venv/Scripts/f5-tts_infer-cli.exe")
            .or_else(|| find_upward(start.clone(), ".venv/Scripts/f5-tts_infer-cli"))
            .or_else(|| find_upward(start, ".venv/bin/f5-tts_infer-cli"))
    })
}

fn install_f5tts(args: &Args) -> Result<()> {
    let python = args
        .whisper_bin
        .as_ref()
        .and_then(python_for_whisper_bin)
        .or_else(default_local_python)
        .ok_or_else(|| anyhow!("could not find .venv python; create it first"))?;

    eprintln!("Installing f5-tts via {}... (this may take several minutes)", python.display());
    let status = Command::new(&python)
        .arg("-m")
        .arg("pip")
        .arg("install")
        .arg("--upgrade")
        .arg("f5-tts")
        .status()
        .with_context(|| format!("failed to run {}", python.display()))?;

    if !status.success() {
        bail!("pip install f5-tts exited with status {status}");
    }
    Ok(())
}

fn setup_clone(args: Args) {
    thread::spawn(move || {
        if default_local_f5tts().is_none() {
            F5_INSTALLING.store(true, Ordering::SeqCst);
            set_f5_install_error(None);
            let result = install_f5tts(&args);
            F5_INSTALLING.store(false, Ordering::SeqCst);
            match result {
                Ok(()) => eprintln!("f5-tts installed"),
                Err(err) => {
                    let msg = format!("{err:#}");
                    eprintln!("f5-tts install failed: {msg}");
                    set_f5_install_error(Some(msg));
                }
            }
        }
    });
}

fn save_clone_reference(audio_bytes: &[u8], text: &str, extension: &str) -> Result<()> {
    let dir = clone_dir();
    fs::create_dir_all(&dir)
        .with_context(|| format!("failed to create clone dir {}", dir.display()))?;

    // Clear any previous reference audio (could be a different extension).
    if let Ok(entries) = fs::read_dir(&dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            if name.starts_with("reference.") && name != "reference.txt" {
                let _ = fs::remove_file(&path);
            }
        }
    }

    let ext = {
        let clean = sanitize_extension(extension);
        if clean.is_empty() {
            "wav".to_string()
        } else {
            clean
        }
    };
    let ref_path = dir.join(format!("reference.{ext}"));
    fs::write(&ref_path, audio_bytes)
        .with_context(|| format!("failed to save reference audio {}", ref_path.display()))?;
    fs::write(clone_reference_text_path(), text)
        .context("failed to save clone reference text")?;
    Ok(())
}

fn synthesize_clone(args: &Args, gen_text: &str) -> Result<Vec<u8>> {
    if !has_clone_reference() {
        bail!("No reference audio uploaded");
    }
    if default_local_f5tts().is_none() {
        bail!("f5-tts is not installed yet");
    }

    let ref_audio = find_clone_reference_audio()
        .ok_or_else(|| anyhow!("reference audio file not found"))?;
    let ref_text = get_clone_reference_text()
        .ok_or_else(|| anyhow!("failed to read reference transcript"))?;
    let out_dir = clone_dir();
    let out_file = "output.wav";

    // Use Python to invoke the f5-tts API. The CLI flags vary across versions,
    // but the Python API is stable.
    let python = args
        .whisper_bin
        .as_ref()
        .and_then(python_for_whisper_bin)
        .or_else(default_local_python)
        .ok_or_else(|| anyhow!("could not find .venv python"))?;

    let script = r#"
import sys, os
from pathlib import Path
ref_audio, ref_text, gen_text, out_dir, out_file = sys.argv[1:6]
try:
    from f5_tts.api import F5TTS
except Exception as exc:
    print(f"failed to import f5_tts: {exc}", file=sys.stderr)
    sys.exit(2)

os.makedirs(out_dir, exist_ok=True)
out_path = str(Path(out_dir) / out_file)

try:
    f5 = F5TTS()
except Exception as exc:
    print(f"failed to init F5TTS: {exc}", file=sys.stderr)
    sys.exit(3)

try:
    f5.infer(
        ref_file=ref_audio,
        ref_text=ref_text,
        gen_text=gen_text,
        file_wave=out_path,
        seed=-1,
    )
except Exception as exc:
    print(f"f5 infer failed: {exc}", file=sys.stderr)
    sys.exit(4)
"#;

    CLONE_SYNTHESIZING.store(true, Ordering::SeqCst);
    let result = (|| -> Result<Vec<u8>> {
        let output = Command::new(&python)
            .arg("-c")
            .arg(script)
            .arg(&ref_audio)
            .arg(&ref_text)
            .arg(gen_text)
            .arg(&out_dir)
            .arg(out_file)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .with_context(|| format!("failed to run {}", python.display()))?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            bail!("f5-tts failed: {stderr}");
        }
        let wav_path = clone_output_path();
        let wav = fs::read(&wav_path)
            .with_context(|| format!("failed to read generated WAV {}", wav_path.display()))?;
        Ok(wav)
    })();
    CLONE_SYNTHESIZING.store(false, Ordering::SeqCst);
    result
}

fn clone_status_json() -> String {
    let installing = F5_INSTALLING.load(Ordering::SeqCst);
    let install_error = get_f5_install_error()
        .map(|e| format!(r#""{}""#, json_escape(&e)))
        .unwrap_or_else(|| "null".to_string());
    let available = default_local_f5tts().is_some();
    let has_ref = has_clone_reference();
    let ref_text = get_clone_reference_text()
        .map(|t| format!(r#""{}""#, json_escape(&t)))
        .unwrap_or_else(|| "null".to_string());
    let synthesizing = CLONE_SYNTHESIZING.load(Ordering::SeqCst);
    format!(
        r#"{{"installing":{},"install_error":{},"available":{},"has_reference":{},"reference_text":{},"synthesizing":{}}}"#,
        installing, install_error, available, has_ref, ref_text, synthesizing
    )
}

fn tts_models_root() -> PathBuf {
    project_root()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("models")
        .join("tts")
}

fn tts_model_is_available(model: &str) -> bool {
    find_tts_model_file(model).is_some()
}

fn find_tts_model_file(model: &str) -> Option<PathBuf> {
    let onnx_name = format!("{model}.onnx");
    let json_name = format!("{model}.onnx.json");

    let mut dirs: Vec<PathBuf> = Vec::new();
    dirs.push(tts_models_root());
    if let Some(root) = project_root() {
        dirs.push(root.join("models"));
        dirs.push(root);
    }
    if let Ok(cwd) = env::current_dir() {
        dirs.push(cwd.join("models").join("tts"));
        dirs.push(cwd.join("models"));
        dirs.push(cwd);
    }
    if let Ok(exe) = env::current_exe()
        && let Some(parent) = exe.parent()
    {
        dirs.push(parent.join("models").join("tts"));
        dirs.push(parent.to_path_buf());
    }

    dirs.into_iter().find_map(|dir| {
        let onnx = dir.join(&onnx_name);
        let json = dir.join(&json_name);
        if onnx.exists() && json.exists() {
            Some(onnx)
        } else {
            None
        }
    })
}

fn set_tts_download_started(model: &str) {
    if let Ok(mut state) = tts_download_state_cell().lock() {
        *state = Some(DownloadState {
            model: model.to_string(),
            progress: 0.0,
            active: true,
            error: None,
        });
    }
}

fn set_tts_download_progress(model: &str, progress: f32) {
    if let Ok(mut state) = tts_download_state_cell().lock()
        && let Some(current) = state.as_mut()
        && current.model == model
    {
        current.progress = progress.clamp(0.0, 100.0);
    }
}

fn set_tts_download_finished(model: &str) {
    set_current_tts_model(model);
    if let Ok(mut state) = tts_download_state_cell().lock() {
        *state = Some(DownloadState {
            model: model.to_string(),
            progress: 100.0,
            active: false,
            error: None,
        });
    }
}

fn set_tts_download_error(model: &str, error: String) {
    if let Ok(mut state) = tts_download_state_cell().lock() {
        *state = Some(DownloadState {
            model: model.to_string(),
            progress: 0.0,
            active: false,
            error: Some(error),
        });
    }
}

fn start_tts_model_download(args: Args, model: String) -> Result<()> {
    if tts_model_is_available(&model) {
        set_tts_download_finished(&model);
        return Ok(());
    }

    if let Some(state) = tts_download_state()
        && state.active
    {
        return Ok(());
    }

    let model_info = TTS_MODEL_OPTIONS
        .iter()
        .find(|(_label, m, _size, _onnx, _json)| *m == model)
        .ok_or_else(|| anyhow!("Unknown TTS model"))?;
    let onnx_url = model_info.3.to_string();
    let json_url = model_info.4.to_string();

    set_tts_download_started(&model);
    thread::spawn(move || {
        if let Err(err) = download_tts_model(&args, &model, &onnx_url, &json_url) {
            set_tts_download_error(&model, err.to_string());
        }
    });

    Ok(())
}

const TTS_DOWNLOAD_SCRIPT: &str = r#"
import sys, urllib.request
def progress(blocks, block_size, total):
    if total > 0:
        pct = min(100.0, blocks * block_size * 100.0 / total)
        sys.stderr.write(f"\r{pct:.1f}%")
        sys.stderr.flush()
url, output = sys.argv[1], sys.argv[2]
urllib.request.urlretrieve(url, output, progress)
"#;

fn download_tts_model(args: &Args, model: &str, onnx_url: &str, json_url: &str) -> Result<()> {
    let target_dir = tts_models_root();
    fs::create_dir_all(&target_dir).with_context(|| {
        format!("failed to create TTS models directory {}", target_dir.display())
    })?;

    let python = args
        .whisper_bin
        .as_ref()
        .and_then(python_for_whisper_bin)
        .or_else(default_local_python)
        .unwrap_or_else(|| PathBuf::from("python"));

    let onnx_path = target_dir.join(format!("{model}.onnx"));
    let json_path = target_dir.join(format!("{model}.onnx.json"));

    let mut child = Command::new(&python)
        .arg("-c")
        .arg(TTS_DOWNLOAD_SCRIPT)
        .arg(onnx_url)
        .arg(&onnx_path)
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("failed to run {}", python.display()))?;

    if let Some(stderr) = child.stderr.take() {
        let mut reader = BufReader::new(stderr);
        let mut chunk = Vec::new();
        loop {
            chunk.clear();
            let read = reader
                .read_until(b'\r', &mut chunk)
                .context("failed to read TTS model download progress")?;
            if read == 0 {
                break;
            }

            let text = String::from_utf8_lossy(&chunk);
            if let Some(progress) = parse_progress_percent(&text) {
                set_tts_download_progress(model, progress * 0.95);
            }
        }
    }

    let status = child
        .wait()
        .context("failed to wait for TTS .onnx download")?;
    if !status.success() {
        bail!("TTS .onnx download exited with status {status}");
    }

    set_tts_download_progress(model, 96.0);

    let status = Command::new(&python)
        .arg("-c")
        .arg(TTS_DOWNLOAD_SCRIPT)
        .arg(json_url)
        .arg(&json_path)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .context("failed to run TTS .onnx.json download")?;
    if !status.success() {
        bail!("TTS .onnx.json download exited with status {status}");
    }

    set_tts_download_finished(model);
    Ok(())
}

fn default_local_piper() -> Option<PathBuf> {
    let mut starts = Vec::new();

    if let Ok(cwd) = env::current_dir() {
        starts.push(cwd);
    }

    if let Ok(exe) = env::current_exe()
        && let Some(parent) = exe.parent()
    {
        starts.push(parent.to_path_buf());
    }

    starts.into_iter().find_map(|start| {
        find_upward(start.clone(), ".venv/Scripts/piper.exe")
            .or_else(|| find_upward(start.clone(), ".venv/Scripts/piper"))
            .or_else(|| find_upward(start, ".venv/bin/piper"))
    })
}

fn synthesize_text(model: &str, text: &str) -> Result<Vec<u8>> {
    let onnx = find_tts_model_file(model)
        .ok_or_else(|| anyhow!("TTS model {model} is not downloaded"))?;
    let target_dir = project_root()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("target");
    fs::create_dir_all(&target_dir)
        .with_context(|| format!("failed to create output dir {}", target_dir.display()))?;
    let output = target_dir.join("tts.wav");

    let piper_bin = default_local_piper().unwrap_or_else(|| PathBuf::from("piper"));

    let mut child = Command::new(&piper_bin)
        .arg("--model")
        .arg(&onnx)
        .arg("--output_file")
        .arg(&output)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| {
            format!(
                "failed to run {}. Install piper-tts in .venv with: .venv/Scripts/pip install piper-tts",
                piper_bin.display()
            )
        })?;

    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(text.as_bytes())
            .context("failed to write text to piper stdin")?;
    }

    let output_result = child
        .wait_with_output()
        .context("failed to wait for piper process")?;
    if !output_result.status.success() {
        let stderr = String::from_utf8_lossy(&output_result.stderr);
        bail!("piper failed: {stderr}");
    }

    let wav = fs::read(&output)
        .with_context(|| format!("failed to read generated WAV {}", output.display()))?;
    Ok(wav)
}

fn tts_models_json() -> String {
    let current_model = get_current_tts_model();
    let download = tts_download_state();
    let models = TTS_MODEL_OPTIONS
        .iter()
        .map(|(label, model, size_mb, onnx_url, _json_url)| {
            let downloading = download
                .as_ref()
                .map(|state| state.active && state.model == *model)
                .unwrap_or(false);
            let progress = download
                .as_ref()
                .filter(|state| state.model == *model)
                .map(|state| state.progress)
                .unwrap_or(0.0);
            let error = download
                .as_ref()
                .filter(|state| state.model == *model)
                .and_then(|state| state.error.as_ref())
                .map(|error| format!(r#""{}""#, json_escape(error)))
                .unwrap_or_else(|| "null".to_string());

            format!(
                r#"{{"label":"{}","model":"{}","size_mb":{},"download_url":"{}","available":{},"downloading":{},"progress":{:.1},"error":{}}}"#,
                json_escape(label),
                json_escape(model),
                size_mb,
                json_escape(onnx_url),
                tts_model_is_available(model),
                downloading,
                progress,
                error
            )
        })
        .collect::<Vec<_>>()
        .join(",");

    let piper_installing = PIPER_INSTALLING.load(Ordering::SeqCst);
    let piper_install_error = get_piper_install_error()
        .map(|e| format!(r#""{}""#, json_escape(&e)))
        .unwrap_or_else(|| "null".to_string());
    let piper_available = default_local_piper().is_some();

    format!(
        r#"{{"current":"{}","piper_installing":{},"piper_install_error":{},"piper_available":{},"models":[{}]}}"#,
        json_escape(&current_model),
        piper_installing,
        piper_install_error,
        piper_available,
        models
    )
}

const INDEX_HTML: &str = include_str!("../index.html");
