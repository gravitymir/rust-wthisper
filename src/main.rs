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
static LISTENING_ACTIVE: AtomicBool = AtomicBool::new(false);
static SHUTTING_DOWN: AtomicBool = AtomicBool::new(false);

const SIGINT: i32 = 2;
const SIGTERM: i32 = 15;
const MODEL_OPTIONS: [(&str, &str, u32); 6] = [
    ("tiny", "tiny.en", 75),
    ("base", "base.en", 142),
    ("small", "small.en", 466),
    ("medium", "medium.en", 1500),
    ("large", "large", 2900),
    ("turbo", "large-v3-turbo", 1550),
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

    let mut request_body = String::new();
    if content_length > 0 {
        let mut body = vec![0; content_length];
        reader
            .read_exact(&mut body)
            .context("failed to read request body")?;
        request_body = String::from_utf8(body).context("invalid UTF-8 in request body")?;
    }

    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or_default();
    let path = parts.next().unwrap_or_default();

    match (method, path) {
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
                    .any(|(_label, option, _size_mb)| {
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
                    .any(|(_label, option, _size_mb)| *option == model)
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
    let model = resolve_model(&get_current_model());

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
        command.arg("--language").arg(&args.language);
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
        .map(|(_label, model, _size_mb)| *model)
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
    if let Some(state) = download_state()
        && state.active
    {
        return false;
    }

    model_is_available(&get_current_model())
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
    format!(
        r#"{{"status":"{}","level":{:.5},"left_level":{:.5},"right_level":{:.5},"channels":{},"sample_rate":{},"bits_per_sample":{},"sample_format":"{}","device_name":"{}","voice_threshold":{:.5},"silence_tail":{:.1}}}"#,
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
        state.silence_tail
    )
}

fn models_json() -> String {
    let current_model = get_current_model();
    let download = download_state();
    let models = MODEL_OPTIONS
        .iter()
        .map(|(label, model, size_mb)| {
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
                r#"{{"label":"{}","model":"{}","size_mb":{},"available":{},"downloading":{},"progress":{:.1},"error":{}}}"#,
                json_escape(label),
                json_escape(model),
                size_mb,
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

const INDEX_HTML: &str = include_str!("../index.html");
