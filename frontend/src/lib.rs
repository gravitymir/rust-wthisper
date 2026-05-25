use std::cell::RefCell;
use std::rc::Rc;

use js_sys::{Array, JSON, Reflect};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::{spawn_local, JsFuture};
use web_sys::{
    Document, Element, Event, File, HtmlAnchorElement, HtmlAudioElement, HtmlButtonElement,
    HtmlElement, HtmlInputElement, HtmlTextAreaElement, KeyboardEvent, Request, RequestInit,
    RequestMode, Response, Url, Window,
};

const METER_SEGMENTS: usize = 100;

#[wasm_bindgen(start)]
pub fn start() -> Result<(), JsValue> {
    App::new()?.start()
}

#[derive(Clone)]
struct App {
    window: Window,
    document: Document,
    tiles: Element,
    state_tiles: Vec<Element>,
    waiting_timer: Element,
    speaking_timer: Element,
    transcribing_timer: Element,
    transcript: Element,
    copy_transcript_button: HtmlButtonElement,
    meter: Element,
    meter_track: Element,
    meter_tooltip: Element,
    meter_hover_tooltip: Element,
    silence_tail_slider: HtmlInputElement,
    silence_tail_value: Element,
    model_buttons: Element,
    device_details: Element,
    waiting_tile: Element,
    tile_action: Element,
    upload_input: HtmlInputElement,
    upload_button: HtmlButtonElement,
    download_modal: Element,
    download_modal_title: Element,
    download_modal_text: Element,
    download_model_button: HtmlButtonElement,
    pages: Element,
    nav_left: HtmlButtonElement,
    nav_right: HtmlButtonElement,
    tts_input: HtmlTextAreaElement,
    tts_char_count: Element,
    tts_combine_button: HtmlButtonElement,
    tts_download_button: HtmlButtonElement,
    tts_audio: HtmlAudioElement,
    tts_status: Element,
    tts_model_buttons: Element,
    clone_ref_input: HtmlInputElement,
    clone_ref_button: HtmlButtonElement,
    clone_ref_name: Element,
    clone_ref_text: HtmlTextAreaElement,
    clone_ref_text_count: Element,
    clone_gen_text: HtmlTextAreaElement,
    clone_gen_text_count: Element,
    clone_combine_button: HtmlButtonElement,
    clone_download_button: HtmlButtonElement,
    clone_audio: HtmlAudioElement,
    clone_status: Element,
    state: Rc<RefCell<State>>,
}

struct State {
    active: bool,
    in_flight: bool,
    current_state: String,
    state_started_at: f64,
    can_listen: bool,
    downloading_model: bool,
    uploading_file: bool,
    paused: bool,
    pause_in_flight: bool,
    device_details_set: bool,
    status_failures: u32,
    pending_download_model: Option<String>,
    pending_download_label: Option<String>,
    selected_activation_block: usize,
    current_threshold: f64,
    models_signature: String,
    current_page: usize,
    tts_audio_url: Option<String>,
    tts_synthesizing: bool,
    tts_models_signature: String,
    tts_current_model: String,
    tts_downloading_model: bool,
    clone_ref_file: Option<File>,
    clone_ref_file_name: Option<String>,
    clone_audio_url: Option<String>,
    clone_synthesizing: bool,
    clone_status_signature: String,
    clone_f5_installing: bool,
    clone_f5_available: bool,
}

#[derive(Clone)]
struct ModelOption {
    label: String,
    model: String,
    size_mb: f64,
    download_url: String,
    available: bool,
    downloading: bool,
    progress: f64,
}

struct ModelsData {
    current: String,
    models: Vec<ModelOption>,
}

struct TtsModelsData {
    inner: ModelsData,
    piper_installing: bool,
    piper_install_error: Option<String>,
    piper_available: bool,
}

impl App {
    fn new() -> Result<Self, JsValue> {
        let window = web_sys::window().ok_or_else(|| js_err("missing window"))?;
        let document = window
            .document()
            .ok_or_else(|| js_err("missing document"))?;

        Ok(Self {
            window: window.clone(),
            document: document.clone(),
            tiles: query(&document, "#tiles")?,
            state_tiles: query_all(&document, "[data-state-tile]")?,
            waiting_timer: query(&document, "[data-timer=\"waiting\"]")?,
            speaking_timer: query(&document, "[data-timer=\"speaking\"]")?,
            transcribing_timer: query(&document, "[data-timer=\"transcribing\"]")?,
            transcript: query(&document, "#transcript")?,
            copy_transcript_button: query::<HtmlButtonElement>(&document, "#copy-transcript")?,
            meter: query(&document, "#meter")?,
            meter_track: query(&document, "#meter-track")?,
            meter_tooltip: query(&document, "#meter-tooltip")?,
            meter_hover_tooltip: query(&document, "#meter-hover-tooltip")?,
            silence_tail_slider: query::<HtmlInputElement>(&document, "#silence-tail-slider")?,
            silence_tail_value: query(&document, "#silence-tail-value")?,
            model_buttons: query(&document, "#model-buttons")?,
            device_details: query(&document, "#device-details")?,
            waiting_tile: query(&document, "[data-state-tile=\"waiting\"]")?,
            tile_action: query(&document, "#tile-action")?,
            upload_input: query::<HtmlInputElement>(&document, "#upload-input")?,
            upload_button: query::<HtmlButtonElement>(&document, "#upload-button")?,
            download_modal: query(&document, "#download-modal")?,
            download_modal_title: query(&document, "#download-modal-title")?,
            download_modal_text: query(&document, "#download-modal-text")?,
            download_model_button: query::<HtmlButtonElement>(&document, "#download-model-button")?,
            pages: query(&document, "#pages")?,
            nav_left: query::<HtmlButtonElement>(&document, "#nav-left")?,
            nav_right: query::<HtmlButtonElement>(&document, "#nav-right")?,
            tts_input: query::<HtmlTextAreaElement>(&document, "#tts-input")?,
            tts_char_count: query(&document, "#tts-char-count")?,
            tts_combine_button: query::<HtmlButtonElement>(&document, "#tts-combine")?,
            tts_download_button: query::<HtmlButtonElement>(&document, "#tts-download")?,
            tts_audio: query::<HtmlAudioElement>(&document, "#tts-audio")?,
            tts_status: query(&document, "#tts-status")?,
            tts_model_buttons: query(&document, "#tts-model-buttons")?,
            clone_ref_input: query::<HtmlInputElement>(&document, "#clone-ref-input")?,
            clone_ref_button: query::<HtmlButtonElement>(&document, "#clone-ref-button")?,
            clone_ref_name: query(&document, "#clone-ref-name")?,
            clone_ref_text: query::<HtmlTextAreaElement>(&document, "#clone-ref-text")?,
            clone_ref_text_count: query(&document, "#clone-ref-text-count")?,
            clone_gen_text: query::<HtmlTextAreaElement>(&document, "#clone-gen-text")?,
            clone_gen_text_count: query(&document, "#clone-gen-text-count")?,
            clone_combine_button: query::<HtmlButtonElement>(&document, "#clone-combine")?,
            clone_download_button: query::<HtmlButtonElement>(&document, "#clone-download")?,
            clone_audio: query::<HtmlAudioElement>(&document, "#clone-audio")?,
            clone_status: query(&document, "#clone-status")?,
            state: Rc::new(RefCell::new(State {
                active: true,
                in_flight: false,
                current_state: "idle".to_string(),
                state_started_at: now(&window),
                can_listen: false,
                downloading_model: false,
                uploading_file: false,
                paused: false,
                pause_in_flight: false,
                device_details_set: false,
                status_failures: 0,
                pending_download_model: None,
                pending_download_label: None,
                selected_activation_block: 80,
                current_threshold: 0.08,
                models_signature: String::new(),
                current_page: 0,
                tts_audio_url: None,
                tts_synthesizing: false,
                tts_models_signature: String::new(),
                tts_current_model: "en_US-ryan-medium".to_string(),
                tts_downloading_model: false,
                clone_ref_file: None,
                clone_ref_file_name: None,
                clone_audio_url: None,
                clone_synthesizing: false,
                clone_status_signature: String::new(),
                clone_f5_installing: false,
                clone_f5_available: false,
            })),
        })
    }

    fn start(&self) -> Result<(), JsValue> {
        self.build_meter_track()?;
        self.install_handlers()?;
        self.set_state("idle", 0.0)?;
        self.update_char_count();
        self.update_clone_char_counts();
        App::fetch_models(self.clone());
        App::fetch_tts_models(self.clone());
        App::fetch_clone_status(self.clone());
        self.interval(250, App::poll_status)?;
        self.interval(500, App::fetch_models)?;
        self.interval(500, App::fetch_tts_models)?;
        self.interval(700, App::fetch_clone_status)?;
        self.interval(50, App::update_timers)?;
        Ok(())
    }

    fn install_handlers(&self) -> Result<(), JsValue> {
        let app = self.clone();
        let closure = Closure::<dyn FnMut(Event)>::new(move |event: Event| {
            let Some(target) = event.target() else {
                return;
            };
            if target == app.download_modal.clone().into()
                && !app.download_model_button.disabled()
            {
                let _ = app.close_download_modal();
            }
        });
        self.download_modal
            .add_event_listener_with_callback("click", closure.as_ref().unchecked_ref())?;
        closure.forget();

        let app = self.clone();
        let closure = Closure::<dyn FnMut(Event)>::new(move |_| {
            app.download_selected_model();
        });
        self.download_model_button
            .add_event_listener_with_callback("click", closure.as_ref().unchecked_ref())?;
        closure.forget();

        let app = self.clone();
        let closure = Closure::<dyn FnMut(Event)>::new(move |_| {
            app.copy_transcript();
        });
        self.copy_transcript_button
            .add_event_listener_with_callback("click", closure.as_ref().unchecked_ref())?;
        closure.forget();

        let app = self.clone();
        let closure = Closure::<dyn FnMut(Event)>::new(move |_| {
            let value = app.silence_tail_slider.value();
            app.set_silence_tail_display(value.parse::<f64>().unwrap_or(1.5));
            app.update_silence_tail(value);
        });
        self.silence_tail_slider
            .add_event_listener_with_callback("input", closure.as_ref().unchecked_ref())?;
        closure.forget();

        let app = self.clone();
        let closure = Closure::<dyn FnMut(Event)>::new(move |_| {
            if app.state.borrow().uploading_file || app.state.borrow().downloading_model {
                return;
            }
            app.upload_input.set_value("");
            app.upload_input.click();
        });
        self.upload_button
            .add_event_listener_with_callback("click", closure.as_ref().unchecked_ref())?;
        closure.forget();

        let app = self.clone();
        let closure = Closure::<dyn FnMut(Event)>::new(move |_| {
            let Some(files) = app.upload_input.files() else {
                return;
            };
            let Some(file) = files.get(0) else {
                return;
            };
            app.upload_file(file);
        });
        self.upload_input
            .add_event_listener_with_callback("change", closure.as_ref().unchecked_ref())?;
        closure.forget();

        let app = self.clone();
        let closure = Closure::<dyn FnMut(Event)>::new(move |_| {
            app.toggle_hearing();
        });
        self.waiting_tile
            .add_event_listener_with_callback("click", closure.as_ref().unchecked_ref())?;
        closure.forget();

        let app = self.clone();
        let closure = Closure::<dyn FnMut(Event)>::new(move |_| {
            let current = app.state.borrow().current_page;
            if current > 0 {
                app.navigate_to_page(current - 1);
            }
        });
        self.nav_left
            .add_event_listener_with_callback("click", closure.as_ref().unchecked_ref())?;
        closure.forget();

        let app = self.clone();
        let closure = Closure::<dyn FnMut(Event)>::new(move |_| {
            let current = app.state.borrow().current_page;
            if current < 2 {
                app.navigate_to_page(current + 1);
            }
        });
        self.nav_right
            .add_event_listener_with_callback("click", closure.as_ref().unchecked_ref())?;
        closure.forget();

        let app = self.clone();
        let closure = Closure::<dyn FnMut(KeyboardEvent)>::new(move |event: KeyboardEvent| {
            let key = event.key();
            let target = event.target();
            let is_text_input = target
                .as_ref()
                .and_then(|t| t.dyn_ref::<Element>())
                .map(|el| {
                    let tag = el.tag_name();
                    tag.eq_ignore_ascii_case("textarea") || tag.eq_ignore_ascii_case("input")
                })
                .unwrap_or(false);
            if is_text_input {
                return;
            }
            let current = app.state.borrow().current_page;
            match key.as_str() {
                "ArrowLeft" if current > 0 => app.navigate_to_page(current - 1),
                "ArrowRight" if current < 2 => app.navigate_to_page(current + 1),
                _ => {}
            }
        });
        self.document
            .add_event_listener_with_callback("keydown", closure.as_ref().unchecked_ref())?;
        closure.forget();

        let app = self.clone();
        let closure = Closure::<dyn FnMut(Event)>::new(move |_| {
            app.update_char_count();
        });
        self.tts_input
            .add_event_listener_with_callback("input", closure.as_ref().unchecked_ref())?;
        closure.forget();

        let app = self.clone();
        let closure = Closure::<dyn FnMut(Event)>::new(move |_| {
            app.synthesize_tts();
        });
        self.tts_combine_button
            .add_event_listener_with_callback("click", closure.as_ref().unchecked_ref())?;
        closure.forget();

        let app = self.clone();
        let closure = Closure::<dyn FnMut(Event)>::new(move |_| {
            app.download_tts_audio();
        });
        self.tts_download_button
            .add_event_listener_with_callback("click", closure.as_ref().unchecked_ref())?;
        closure.forget();

        let app = self.clone();
        let closure = Closure::<dyn FnMut(Event)>::new(move |_| {
            if app.state.borrow().clone_synthesizing {
                return;
            }
            app.clone_ref_input.set_value("");
            app.clone_ref_input.click();
        });
        self.clone_ref_button
            .add_event_listener_with_callback("click", closure.as_ref().unchecked_ref())?;
        closure.forget();

        let app = self.clone();
        let closure = Closure::<dyn FnMut(Event)>::new(move |_| {
            let Some(files) = app.clone_ref_input.files() else {
                return;
            };
            let Some(file) = files.get(0) else {
                return;
            };
            let name = file.name();
            {
                let mut state = app.state.borrow_mut();
                state.clone_ref_file = Some(file);
                state.clone_ref_file_name = Some(name.clone());
            }
            app.clone_ref_name.set_text_content(Some(&name));
            let _ = app.clone_ref_name.class_list().add_1("set");
        });
        self.clone_ref_input
            .add_event_listener_with_callback("change", closure.as_ref().unchecked_ref())?;
        closure.forget();

        let app = self.clone();
        let closure = Closure::<dyn FnMut(Event)>::new(move |_| {
            app.update_clone_char_counts();
        });
        self.clone_ref_text
            .add_event_listener_with_callback("input", closure.as_ref().unchecked_ref())?;
        closure.forget();

        let app = self.clone();
        let closure = Closure::<dyn FnMut(Event)>::new(move |_| {
            app.update_clone_char_counts();
        });
        self.clone_gen_text
            .add_event_listener_with_callback("input", closure.as_ref().unchecked_ref())?;
        closure.forget();

        let app = self.clone();
        let closure = Closure::<dyn FnMut(Event)>::new(move |_| {
            app.synthesize_clone();
        });
        self.clone_combine_button
            .add_event_listener_with_callback("click", closure.as_ref().unchecked_ref())?;
        closure.forget();

        let app = self.clone();
        let closure = Closure::<dyn FnMut(Event)>::new(move |_| {
            app.download_clone_audio();
        });
        self.clone_download_button
            .add_event_listener_with_callback("click", closure.as_ref().unchecked_ref())?;
        closure.forget();

        Ok(())
    }

    fn interval(&self, ms: i32, callback: fn(App)) -> Result<(), JsValue> {
        let app = self.clone();
        let closure = Closure::<dyn FnMut()>::new(move || callback(app.clone()));
        self.window
            .set_interval_with_callback_and_timeout_and_arguments_0(
                closure.as_ref().unchecked_ref(),
                ms,
            )?;
        closure.forget();
        Ok(())
    }

    fn build_meter_track(&self) -> Result<(), JsValue> {
        self.meter_track.set_inner_html("");
        for index in 0..METER_SEGMENTS {
            let segment = self.document.create_element("span")?;
            segment.set_class_name("meter-segment");
            set_style(&segment, "--segment-color", &segment_color(index));

            let app = self.clone();
            let block = index + 1;
            let closure = Closure::<dyn FnMut(Event)>::new(move |_| {
                let _ = app.show_meter_hover_tooltip(block);
            });
            segment.add_event_listener_with_callback("mouseenter", closure.as_ref().unchecked_ref())?;
            closure.forget();

            let app = self.clone();
            let closure = Closure::<dyn FnMut(Event)>::new(move |_| {
                let _ = app.hide_meter_hover_tooltip();
            });
            segment.add_event_listener_with_callback("mouseleave", closure.as_ref().unchecked_ref())?;
            closure.forget();

            let app = self.clone();
            let closure = Closure::<dyn FnMut(Event)>::new(move |event: Event| {
                event.stop_propagation();
                app.set_activation_from_meter_block(block);
            });
            segment.add_event_listener_with_callback("click", closure.as_ref().unchecked_ref())?;
            closure.forget();

            self.meter_track.append_child(&segment)?;
        }
        Ok(())
    }

    fn set_activation_from_meter_block(&self, block_number: usize) {
        let selected = block_number.clamp(1, METER_SEGMENTS);
        let backend_threshold = selected as f64 / 1000.0;
        let _ = self.set_selected_activation_block(selected);
        self.set_current_threshold(backend_threshold);
        self.update_threshold(backend_threshold);
    }

    fn show_meter_hover_tooltip(&self, block_number: usize) -> Result<(), JsValue> {
        let selected = block_number.clamp(1, METER_SEGMENTS);
        set_style(&self.meter, "--hover-position", &format!("{selected}%"));
        self.meter_hover_tooltip
            .set_text_content(Some(&format!("Set activation level {selected}")));
        self.meter_hover_tooltip.class_list().add_1("visible")
    }

    fn hide_meter_hover_tooltip(&self) -> Result<(), JsValue> {
        self.meter_hover_tooltip.class_list().remove_1("visible")
    }

    fn set_selected_activation_block(&self, block_number: usize) -> Result<(), JsValue> {
        let selected = block_number.clamp(1, METER_SEGMENTS);
        self.state.borrow_mut().selected_activation_block = selected;
        set_style(&self.meter, "--threshold-position", &format!("{selected}%"));
        self.meter_tooltip.set_text_content(Some(&selected.to_string()));
        self.update_selected_meter_segments()
    }

    fn set_current_threshold(&self, threshold: f64) {
        self.state.borrow_mut().current_threshold = threshold.clamp(0.001, 0.1);
    }

    fn set_silence_tail_display(&self, value: f64) {
        self.silence_tail_value
            .set_text_content(Some(&format!("{value:.1} s")));
    }

    fn set_device_details(&self, data: &JsValue) -> Result<(), JsValue> {
        {
            let state = self.state.borrow();
            if state.device_details_set || number_prop(data, "sample_rate") <= 0.0 {
                return Ok(());
            }
        }

        let channels = number_prop(data, "channels").max(1.0) as u32;
        let sample_rate = number_prop(data, "sample_rate") as u32;
        let bits = number_prop(data, "bits_per_sample") as u32;
        let format = string_prop(data, "sample_format", "input");
        let device = string_prop(data, "device_name", "input");
        let channel_text = if channels == 1 {
            "1 channel".to_string()
        } else {
            format!("{channels} channels")
        };
        let rate_text = if sample_rate > 0 {
            format!("{sample_rate} Hz")
        } else {
            "unknown Hz".to_string()
        };
        self.device_details.set_text_content(Some(&format!(
            "{device} - {channel_text}, {bits} bit, {rate_text} ({format})"
        )));
        self.state.borrow_mut().device_details_set = true;
        Ok(())
    }

    fn set_model_buttons(&self, data: ModelsData, signature: String) -> Result<(), JsValue> {
        {
            let mut state = self.state.borrow_mut();
            if state.models_signature == signature {
                return Ok(());
            }
            state.models_signature = signature;
        }

        self.model_buttons.set_inner_html("");

        let any_downloading = data.models.iter().any(|model| model.downloading);
        let current_available = data
            .models
            .iter()
            .find(|option| option.model == data.current)
            .map(|option| option.available)
            .unwrap_or(false);

        let uploading;
        {
            let mut state = self.state.borrow_mut();
            state.downloading_model = any_downloading;
            state.can_listen = current_available && !any_downloading;
            if any_downloading {
                state.active = false;
            }
            uploading = state.uploading_file;
        }

        self.upload_button.set_disabled(any_downloading || uploading);
        self.update_nav_visibility();

        if any_downloading {
            self.set_state("idle", 0.0)?;
        }

        for option in data.models {
            let button: HtmlButtonElement = self.document.create_element("button")?.dyn_into()?;
            button.set_type("button");
            button.set_class_name("model-button");

            let progress = option.progress.clamp(0.0, 100.0);
            if option.downloading {
                button.set_text_content(Some(&format!("{} {}%", option.label, progress.round())));
            } else {
                button.set_text_content(Some(&option.label));
            }
            set_style(&button, "--download-progress", &format!("{progress}%"));
            toggle_class(&button, "unavailable", !option.available && !option.downloading)?;
            toggle_class(&button, "downloading", option.downloading)?;
            toggle_class(
                &button,
                "locked",
                (any_downloading && !option.downloading) || uploading,
            )?;
            toggle_class(&button, "active", option.model == data.current)?;

            let app = self.clone();
            let option_for_click = option.clone();
            let closure = Closure::<dyn FnMut(Event)>::new(move |_| {
                if any_downloading
                    || option_for_click.downloading
                    || app.state.borrow().uploading_file
                {
                    return;
                }
                if option_for_click.available {
                    app.update_model(option_for_click.model.clone());
                } else {
                    let _ = app.open_download_modal(&option_for_click);
                }
            });
            button.add_event_listener_with_callback("click", closure.as_ref().unchecked_ref())?;
            closure.forget();

            self.model_buttons.append_child(&button)?;
        }

        {
            let state = self.state.borrow();
            let should_start = state.can_listen
                && !state.in_flight
                && !state.paused
                && !state.uploading_file;
            drop(state);
            if should_start {
                self.state.borrow_mut().active = true;
                self.listen_loop();
            }
        }

        Ok(())
    }

    fn open_download_modal(&self, option: &ModelOption) -> Result<(), JsValue> {
        {
            let mut state = self.state.borrow_mut();
            state.active = false;
            state.can_listen = false;
            state.pending_download_model = Some(option.model.clone());
            state.pending_download_label = Some(option.label.clone());
        }
        self.set_state("idle", 0.0)?;
        self.reset_timers();
        self.download_modal_title
            .set_text_content(Some(&format!("Download {} MB?", option.size_mb)));
        self.download_modal_text.set_text_content(Some(&format!(
            "Model \"{}\" is not downloaded yet. URL: {}",
            option.label, option.download_url
        )));
        self.download_model_button
            .set_text_content(Some("Start download model"));
        self.download_model_button.set_disabled(false);
        self.download_modal.class_list().remove_1("hidden")
    }

    fn close_download_modal(&self) -> Result<(), JsValue> {
        self.download_modal.class_list().add_1("hidden")?;
        let mut state = self.state.borrow_mut();
        state.pending_download_model = None;
        state.pending_download_label = None;
        Ok(())
    }

    fn scale_level(&self, level: f64) -> f64 {
        let state = self.state.borrow();
        let relative = level.max(0.0) / state.current_threshold.max(0.001);
        (relative * (state.selected_activation_block as f64 / METER_SEGMENTS as f64))
            .clamp(0.0, 1.0)
    }

    fn set_level(&self, level: f64) -> Result<(), JsValue> {
        let scaled = self.scale_level(level);
        set_style(&self.tiles, "--level", &format!("{scaled:.3}"));
        self.set_meter_segments(scaled)?;
        self.update_selected_meter_segments()
    }

    fn set_meter_segments(&self, level: f64) -> Result<(), JsValue> {
        let active_count = (level.clamp(0.0, 1.0) * METER_SEGMENTS as f64).round() as usize;
        let children = self.meter_track.children();
        for index in 0..children.length() {
            if let Some(segment) = children.item(index) {
                toggle_class(&segment, "active", index < active_count as u32)?;
            }
        }
        Ok(())
    }

    fn update_selected_meter_segments(&self) -> Result<(), JsValue> {
        let selected = self.state.borrow().selected_activation_block - 1;
        let children = self.meter_track.children();
        for index in 0..children.length() {
            if let Some(segment) = children.item(index) {
                toggle_class(&segment, "selected", index == selected as u32)?;
            }
        }
        Ok(())
    }

    fn set_state(&self, status: &str, level: f64) -> Result<(), JsValue> {
        self.set_level(level)?;
        let reset_waiting_timer;
        {
            let mut state = self.state.borrow_mut();
            if status != state.current_state {
                state.current_state = status.to_string();
                state.state_started_at = now(&self.window);
                if let Some(timer) = self.timer_for(status) {
                    timer.set_text_content(Some("0"));
                }
            }
            reset_waiting_timer = status == "idle" && state.paused;
        }
        if reset_waiting_timer {
            self.waiting_timer.set_text_content(Some("0"));
        }

        for tile in &self.state_tiles {
            let matches = tile
                .get_attribute("data-state-tile")
                .map(|value| value == status)
                .unwrap_or(false);
            toggle_class(tile, "active", matches)?;
        }

        let label = match status {
            "waiting" => "Waiting for speech",
            "speaking" => "Speech detected",
            "transcribing" => "Transcribing audio",
            _ => "Listening loop active",
        };
        let result = self.tiles.set_attribute("aria-label", label);
        self.update_tile_action();
        self.update_nav_visibility();
        result
    }

    fn timer_for(&self, status: &str) -> Option<&Element> {
        match status {
            "waiting" => Some(&self.waiting_timer),
            "speaking" => Some(&self.speaking_timer),
            "transcribing" => Some(&self.transcribing_timer),
            _ => None,
        }
    }

    fn update_timers(app: App) {
        let state = app.state.borrow();
        if state.paused || !state.active {
            return;
        }
        if let Some(timer) = app.timer_for(&state.current_state) {
            let elapsed = (now(&app.window) - state.state_started_at).max(0.0).round();
            timer.set_text_content(Some(&format!("{elapsed:.0}")));
        }
    }

    fn reset_timers(&self) {
        self.waiting_timer.set_text_content(Some("0"));
        self.speaking_timer.set_text_content(Some("0"));
        self.transcribing_timer.set_text_content(Some("0"));
        self.state.borrow_mut().state_started_at = now(&self.window);
    }

    fn set_transcript(&self, text: &str) {
        self.transcript.set_text_content(Some(text));
        let show_copy = is_copyable_transcript(text);
        let _ = self
            .copy_transcript_button
            .class_list()
            .toggle_with_force("hidden", !show_copy);
    }

    fn copy_transcript(&self) {
        let text = self.transcript.text_content().unwrap_or_default();
        if !is_copyable_transcript(&text) {
            return;
        }

        let clipboard = self.window.navigator().clipboard();
        let button = self.copy_transcript_button.clone();
        spawn_local(async move {
            let result = JsFuture::from(clipboard.write_text(text.trim())).await;
            match result {
                Ok(_) => {
                    button.set_text_content(Some("copied"));
                    let _ = button.class_list().add_1("copied");
                    wait(900).await;
                    button.set_text_content(Some("copy"));
                    let _ = button.class_list().remove_1("copied");
                }
                Err(error) => {
                    log_error(&format!("Failed to copy transcript: {error:?}"));
                }
            }
        });
    }

    fn listen_loop(&self) {
        {
            let mut state = self.state.borrow_mut();
            if state.in_flight || !state.can_listen {
                return;
            }
            state.in_flight = true;
        }

        let app = self.clone();
        spawn_local(async move {
            while app.state.borrow().active && app.state.borrow().can_listen {
                match post_json("/api/listen", None).await {
                    Ok(data) => {
                        let text = string_prop(&data, "transcript", "").trim().to_string();
                        app.set_transcript(if text.is_empty() { "[no speech detected]" } else { &text });
                    }
                    Err(error) => {
                        if error == "HTTP 409" {
                            wait(350).await;
                            continue;
                        }

                        if !is_fetch_error(&error) {
                            app.set_transcript(&display_error(&error));
                        }
                        app.state.borrow_mut().active = false;
                    }
                }
            }

            app.state.borrow_mut().in_flight = false;
            let _ = app.set_state("idle", 0.0);

            let should_restart = {
                let state = app.state.borrow();
                state.active
                    && state.can_listen
                    && !state.paused
                    && !state.uploading_file
                    && !state.downloading_model
            };
            if should_restart {
                app.listen_loop();
            }
        });
    }

    fn poll_status(app: App) {
        if app.state.borrow().downloading_model {
            let _ = app.set_state("idle", 0.0);
            return;
        }

        spawn_local(async move {
            match get_json("/api/status").await {
                Ok(data) => {
                    app.state.borrow_mut().status_failures = 0;
                    let status = string_prop(&data, "status", "idle");
                    let level = number_prop(&data, "level");
                    let _ = app.set_state(&status, level);
                    if has_prop(&data, "voice_threshold") {
                        let threshold = number_prop(&data, "voice_threshold");
                        let _ = app.set_selected_activation_block(threshold_to_block(threshold));
                        app.set_current_threshold(threshold);
                    }
                    if has_prop(&data, "silence_tail") {
                        let value = number_prop(&data, "silence_tail");
                        app.silence_tail_slider.set_value(&format!("{value:.1}"));
                        app.set_silence_tail_display(value);
                    }
                    if has_prop(&data, "paused") && !app.state.borrow().pause_in_flight {
                        let server_paused = bool_prop(&data, "paused");
                        let mut state = app.state.borrow_mut();
                        if state.paused != server_paused {
                            state.paused = server_paused;
                            drop(state);
                            app.update_tile_action();
                        }
                    }
                    let _ = app.set_device_details(&data);
                }
                Err(error) => {
                    let _ = app.set_state("idle", 0.0);
                    let mut state = app.state.borrow_mut();
                    state.status_failures += 1;
                    if state.status_failures >= 4 && is_fetch_error(&error) {
                        app.set_transcript("Server stopped");
                    }
                }
            }
        });
    }

    fn fetch_models(app: App) {
        spawn_local(async move {
            match get_json_with_signature("/api/models").await {
                Ok((data, signature)) => match parse_models(&data) {
                    Ok(models) => {
                        let uploading = app.state.borrow().uploading_file;
                        let combined = format!("{uploading}|{signature}");
                        let _ = app.set_model_buttons(models, combined);
                    }
                    Err(error) => log_error(&format!("Failed to parse models: {error:?}")),
                },
                Err(error) => log_error(&format!("Failed to fetch models: {error}")),
            }
        });
    }

    fn update_threshold(&self, value: f64) {
        let body = format!(r#"{{"threshold":{value}}}"#);
        spawn_local(async move {
            if let Err(error) = post_json("/api/threshold", Some(body)).await {
                log_error(&format!("Failed to update threshold: {error}"));
            }
        });
    }

    fn update_silence_tail(&self, value: String) {
        let parsed = value.parse::<f64>().unwrap_or(1.5);
        let body = format!(r#"{{"silence_tail":{parsed}}}"#);
        spawn_local(async move {
            if let Err(error) = post_json("/api/silence-tail", Some(body)).await {
                log_error(&format!("Failed to update silence tail: {error}"));
            }
        });
    }

    fn update_model(&self, model: String) {
        let app = self.clone();
        spawn_local(async move {
            let body = format!(r#"{{"model":"{}"}}"#, json_string_escape(&model));
            match post_json("/api/model", Some(body)).await {
                Ok(_) => App::fetch_models(app),
                Err(error) => log_error(&format!("Failed to update model: {error}")),
            }
        });
    }

    fn tile_action_label(&self) -> Option<&'static str> {
        let state = self.state.borrow();
        if state.uploading_file || state.downloading_model {
            return None;
        }
        if state.paused || !state.active {
            return Some("start");
        }
        match state.current_state.as_str() {
            "waiting" | "speaking" => Some("stop"),
            _ => None,
        }
    }

    fn update_tile_action(&self) {
        let label = self.tile_action_label();
        let _ = self.tile_action.class_list().remove_2("start", "stop");
        match label {
            Some(text) => {
                self.tile_action.set_text_content(Some(text));
                let _ = self.tile_action.class_list().remove_1("hidden");
                let _ = self.tile_action.class_list().add_1(text);
            }
            None => {
                let _ = self.tile_action.class_list().add_1("hidden");
            }
        }
    }

    fn toggle_hearing(&self) {
        if self.tile_action_label().is_none() {
            return;
        }
        if self.state.borrow().pause_in_flight {
            return;
        }
        let paused = self.state.borrow().paused || !self.state.borrow().active;
        if paused {
            self.resume_hearing();
        } else {
            self.pause_hearing();
        }
    }

    fn pause_hearing(&self) {
        {
            let mut state = self.state.borrow_mut();
            state.paused = true;
            state.active = false;
            state.pause_in_flight = true;
        }
        self.update_tile_action();

        let app = self.clone();
        spawn_local(async move {
            let _ = post_json("/api/pause", None).await;
            app.state.borrow_mut().pause_in_flight = false;
            app.update_tile_action();
        });
    }

    fn resume_hearing(&self) {
        {
            let mut state = self.state.borrow_mut();
            state.paused = false;
            state.active = true;
            state.pause_in_flight = true;
        }
        self.update_tile_action();

        let app = self.clone();
        spawn_local(async move {
            let _ = post_json("/api/resume", None).await;
            app.state.borrow_mut().pause_in_flight = false;
            app.update_tile_action();
            app.listen_loop();
        });
    }

    fn upload_file(&self, file: File) {
        if self.state.borrow().uploading_file {
            return;
        }

        let name = file.name();
        let extension = name
            .rsplit_once('.')
            .map(|(_, ext)| ext.to_string())
            .unwrap_or_else(|| "wav".to_string());
        let safe_extension: String = extension
            .chars()
            .filter(|ch| ch.is_ascii_alphanumeric())
            .take(8)
            .collect::<String>()
            .to_ascii_lowercase();
        let extension = if safe_extension.is_empty() {
            "wav".to_string()
        } else {
            safe_extension
        };

        {
            let mut state = self.state.borrow_mut();
            state.uploading_file = true;
            state.active = false;
        }
        self.upload_button.set_disabled(true);
        self.upload_button
            .set_text_content(Some(&format!("Uploading {name}...")));
        self.set_transcript("Uploading file...");
        self.update_nav_visibility();
        App::fetch_models(self.clone());

        let app = self.clone();
        spawn_local(async move {
            let url = format!("/api/transcribe-file?ext={}", extension);
            let result = post_blob(&url, &file).await;
            match result {
                Ok(data) => {
                    let text = string_prop(&data, "transcript", "").trim().to_string();
                    app.set_transcript(if text.is_empty() {
                        "[no speech detected]"
                    } else {
                        &text
                    });
                }
                Err(error) => {
                    app.set_transcript(&display_error(&error));
                }
            }

            {
                let mut state = app.state.borrow_mut();
                state.uploading_file = false;
                state.active = true;
                state.paused = false;
            }
            app.upload_button.set_disabled(false);
            app.upload_button
                .set_text_content(Some("Upload audio file"));
            App::fetch_models(app.clone());
            app.update_tile_action();
            app.update_nav_visibility();
            app.listen_loop();
        });
    }

    fn navigate_to_page(&self, page: usize) {
        let target = page.min(2);
        let prev = self.state.borrow().current_page;
        if prev == target {
            return;
        }
        // Leaving STT to anywhere right: pause hearing if waiting/speaking
        if prev == 0 && target > 0 {
            let should_pause = {
                let state = self.state.borrow();
                !state.paused
                    && state.active
                    && (state.current_state == "waiting"
                        || state.current_state == "speaking")
            };
            if should_pause {
                self.pause_hearing();
            }
        }
        self.state.borrow_mut().current_page = target;
        let _ = self.pages.set_attribute("data-page", &target.to_string());
        self.update_nav_visibility();
    }

    fn update_nav_visibility(&self) {
        let (current_page, stt_busy, tts_busy) = {
            let state = self.state.borrow();
            let stt_busy = state.current_state == "speaking"
                || state.current_state == "transcribing"
                || state.uploading_file
                || state.downloading_model;
            let tts_busy = state.tts_synthesizing || state.tts_downloading_model;
            (state.current_page, stt_busy, tts_busy)
        };
        let hide_left = current_page == 0;
        let hide_right = current_page == 2
            || (current_page == 0 && stt_busy)
            || (current_page == 1 && tts_busy);
        let _ = self
            .nav_left
            .class_list()
            .toggle_with_force("hidden", hide_left);
        let _ = self
            .nav_right
            .class_list()
            .toggle_with_force("hidden", hide_right);
    }

    fn update_char_count(&self) {
        let text = self.tts_input.value();
        let count = text.chars().count();
        let label = if count == 1 {
            "1 character".to_string()
        } else {
            format!("{count} characters")
        };
        self.tts_char_count.set_text_content(Some(&label));
    }

    fn set_tts_status(&self, message: &str) {
        self.tts_status.set_text_content(Some(message));
    }

    fn revoke_audio_url(&self) {
        let url = self.state.borrow_mut().tts_audio_url.take();
        if let Some(url) = url {
            let _ = Url::revoke_object_url(&url);
        }
    }

    fn synthesize_tts(&self) {
        if self.state.borrow().tts_synthesizing {
            return;
        }
        let text = self.tts_input.value().trim().to_string();
        if text.is_empty() {
            self.set_tts_status("Paste some text first.");
            return;
        }
        let current = self.state.borrow().tts_current_model.clone();
        if current.is_empty() {
            self.set_tts_status("Pick a TTS voice first.");
            return;
        }

        self.state.borrow_mut().tts_synthesizing = true;
        self.tts_combine_button.set_disabled(true);
        self.tts_combine_button.set_text_content(Some("Synthesizing..."));
        self.tts_download_button.set_disabled(true);
        self.set_tts_status("Synthesizing audio with the current voice...");

        let app = self.clone();
        spawn_local(async move {
            let result = post_text_for_blob("/api/tts/synthesize", &text).await;
            match result {
                Ok(blob) => match Url::create_object_url_with_blob(&blob) {
                    Ok(url) => {
                        app.revoke_audio_url();
                        app.tts_audio.set_src(&url);
                        let _ = app.tts_audio.class_list().remove_1("hidden");
                        app.tts_audio.load();
                        let _ = app.tts_audio.play();
                        app.tts_download_button.set_disabled(false);
                        app.state.borrow_mut().tts_audio_url = Some(url);
                        app.set_tts_status("Done.");
                    }
                    Err(error) => {
                        app.set_tts_status(&format!("Failed to make audio URL: {error:?}"));
                    }
                },
                Err(error) => {
                    app.set_tts_status(&format!("Synthesis failed: {error}"));
                }
            }
            app.state.borrow_mut().tts_synthesizing = false;
            app.tts_combine_button.set_disabled(false);
            app.tts_combine_button.set_text_content(Some("Combine"));
        });
    }

    fn download_tts_audio(&self) {
        let Some(url) = self.state.borrow().tts_audio_url.clone() else {
            self.set_tts_status("Click Combine first to generate audio.");
            return;
        };
        let Ok(anchor) = self.document.create_element("a") else {
            return;
        };
        let Ok(anchor) = anchor.dyn_into::<HtmlAnchorElement>() else {
            return;
        };
        anchor.set_href(&url);
        anchor.set_download("tts.wav");
        let _ = anchor.style().set_property("display", "none");
        let body = match self.document.body() {
            Some(b) => b,
            None => return,
        };
        let _ = body.append_child(&anchor);
        anchor.click();
        let _ = body.remove_child(&anchor);
    }

    fn fetch_tts_models(app: App) {
        spawn_local(async move {
            match get_json_with_signature("/api/tts/models").await {
                Ok((data, signature)) => match parse_tts_models(&data) {
                    Ok(models) => {
                        let synth = app.state.borrow().tts_synthesizing;
                        let combined = format!("{synth}|{signature}");
                        let _ = app.set_tts_model_buttons(models, combined);
                    }
                    Err(error) => log_error(&format!("Failed to parse TTS models: {error:?}")),
                },
                Err(error) => log_error(&format!("Failed to fetch TTS models: {error}")),
            }
        });
    }

    fn set_tts_model_buttons(
        &self,
        data: TtsModelsData,
        signature: String,
    ) -> Result<(), JsValue> {
        {
            let mut state = self.state.borrow_mut();
            if state.tts_models_signature == signature {
                return Ok(());
            }
            state.tts_models_signature = signature;
            state.tts_current_model = data.inner.current.clone();
        }

        self.tts_model_buttons.set_inner_html("");

        let any_downloading = data.inner.models.iter().any(|model| model.downloading);
        let synthesizing = self.state.borrow().tts_synthesizing;
        let piper_installing = data.piper_installing;
        let piper_available = data.piper_available;

        self.state.borrow_mut().tts_downloading_model = any_downloading;
        self.tts_combine_button
            .set_disabled(synthesizing || any_downloading || piper_installing || !piper_available);

        if let Some(err) = data.piper_install_error.as_ref() {
            self.set_tts_status(&format!("piper install failed: {err}"));
        } else if piper_installing {
            self.set_tts_status("Installing piper-tts into .venv... (one-time setup, ~1-2 min)");
        } else if !piper_available {
            self.set_tts_status("piper-tts is not installed.");
        } else if any_downloading {
            let downloading_option = data
                .inner
                .models
                .iter()
                .find(|m| m.downloading)
                .map(|m| (m.label.clone(), m.progress));
            if let Some((label, progress)) = downloading_option {
                self.set_tts_status(&format!(
                    "Downloading voice {label}... {progress:.0}%"
                ));
            }
        }

        for option in data.inner.models {
            let button: HtmlButtonElement = self.document.create_element("button")?.dyn_into()?;
            button.set_type("button");
            button.set_class_name("model-button");

            let progress = option.progress.clamp(0.0, 100.0);
            if option.downloading {
                button.set_text_content(Some(&format!("{} {}%", option.label, progress.round())));
            } else {
                button.set_text_content(Some(&option.label));
            }
            set_style(&button, "--download-progress", &format!("{progress}%"));
            toggle_class(&button, "unavailable", !option.available && !option.downloading)?;
            toggle_class(&button, "downloading", option.downloading)?;
            toggle_class(
                &button,
                "locked",
                (any_downloading && !option.downloading) || synthesizing || piper_installing,
            )?;
            toggle_class(&button, "active", option.model == data.inner.current)?;

            let app = self.clone();
            let option_for_click = option.clone();
            let closure = Closure::<dyn FnMut(Event)>::new(move |_| {
                if any_downloading
                    || option_for_click.downloading
                    || app.state.borrow().tts_synthesizing
                    || piper_installing
                {
                    return;
                }
                if option_for_click.available {
                    app.update_tts_model(option_for_click.model.clone());
                } else {
                    app.start_tts_download(option_for_click.model.clone(), option_for_click.label.clone());
                }
            });
            button.add_event_listener_with_callback("click", closure.as_ref().unchecked_ref())?;
            closure.forget();

            self.tts_model_buttons.append_child(&button)?;
        }

        Ok(())
    }

    fn update_tts_model(&self, model: String) {
        let app = self.clone();
        spawn_local(async move {
            let body = format!(r#"{{"model":"{}"}}"#, json_string_escape(&model));
            match post_json("/api/tts/model", Some(body)).await {
                Ok(_) => App::fetch_tts_models(app),
                Err(error) => log_error(&format!("Failed to update TTS model: {error}")),
            }
        });
    }

    fn start_tts_download(&self, model: String, label: String) {
        self.set_tts_status(&format!("Downloading {label} ({model})..."));
        let app = self.clone();
        spawn_local(async move {
            let body = format!(r#"{{"model":"{}"}}"#, json_string_escape(&model));
            match post_json("/api/tts/download-model", Some(body)).await {
                Ok(_) => App::fetch_tts_models(app),
                Err(error) => {
                    app.set_tts_status(&format!("TTS download failed: {error}"));
                }
            }
        });
    }

    fn update_clone_char_counts(&self) {
        let ref_count = self.clone_ref_text.value().chars().count();
        self.clone_ref_text_count
            .set_text_content(Some(&format!("{ref_count} characters")));
        let gen_count = self.clone_gen_text.value().chars().count();
        self.clone_gen_text_count
            .set_text_content(Some(&format!("{gen_count} characters")));
    }

    fn set_clone_status(&self, message: &str) {
        self.clone_status.set_text_content(Some(message));
    }

    fn revoke_clone_audio_url(&self) {
        let url = self.state.borrow_mut().clone_audio_url.take();
        if let Some(url) = url {
            let _ = Url::revoke_object_url(&url);
        }
    }

    fn synthesize_clone(&self) {
        if self.state.borrow().clone_synthesizing {
            return;
        }
        if !self.state.borrow().clone_f5_available {
            if self.state.borrow().clone_f5_installing {
                self.set_clone_status("f5-tts is still installing — please wait.");
            } else {
                self.set_clone_status("f5-tts is not ready yet.");
            }
            return;
        }

        let Some(ref_file) = self.state.borrow().clone_ref_file.clone() else {
            self.set_clone_status("Upload a reference voice first.");
            return;
        };
        let ref_text = self.clone_ref_text.value().trim().to_string();
        if ref_text.is_empty() {
            self.set_clone_status("Type what is said in the reference audio.");
            return;
        }
        let gen_text = self.clone_gen_text.value().trim().to_string();
        if gen_text.is_empty() {
            self.set_clone_status("Paste text to synthesize.");
            return;
        }

        self.state.borrow_mut().clone_synthesizing = true;
        self.clone_combine_button.set_disabled(true);
        self.clone_combine_button
            .set_text_content(Some("Synthesizing..."));
        self.clone_download_button.set_disabled(true);
        self.set_clone_status("Uploading reference voice and synthesizing... (this can take a while on CPU)");

        let ref_name = ref_file.name();
        let ref_ext = ref_name
            .rsplit_once('.')
            .map(|(_, ext)| ext.to_string())
            .unwrap_or_else(|| "wav".to_string());
        let ref_ext: String = ref_ext
            .chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .take(8)
            .collect::<String>()
            .to_ascii_lowercase();
        let ref_ext = if ref_ext.is_empty() {
            "wav".to_string()
        } else {
            ref_ext
        };

        let app = self.clone();
        spawn_local(async move {
            let ref_url = format!(
                "/api/clone/reference?text={}&ext={}",
                url_encode_component(&ref_text),
                ref_ext
            );
            if let Err(error) = post_blob(&ref_url, &ref_file).await {
                app.set_clone_status(&format!("Reference upload failed: {error}"));
                app.state.borrow_mut().clone_synthesizing = false;
                app.clone_combine_button.set_disabled(false);
                app.clone_combine_button.set_text_content(Some("Combine"));
                return;
            }

            let result = post_text_for_blob("/api/clone/synthesize", &gen_text).await;
            match result {
                Ok(blob) => match Url::create_object_url_with_blob(&blob) {
                    Ok(url) => {
                        app.revoke_clone_audio_url();
                        app.clone_audio.set_src(&url);
                        let _ = app.clone_audio.class_list().remove_1("hidden");
                        app.clone_audio.load();
                        let _ = app.clone_audio.play();
                        app.clone_download_button.set_disabled(false);
                        app.state.borrow_mut().clone_audio_url = Some(url);
                        app.set_clone_status("Done.");
                    }
                    Err(error) => {
                        app.set_clone_status(&format!("Failed to make audio URL: {error:?}"));
                    }
                },
                Err(error) => {
                    app.set_clone_status(&format!("Synthesis failed: {error}"));
                }
            }

            app.state.borrow_mut().clone_synthesizing = false;
            app.clone_combine_button.set_disabled(false);
            app.clone_combine_button.set_text_content(Some("Combine"));
        });
    }

    fn download_clone_audio(&self) {
        let Some(url) = self.state.borrow().clone_audio_url.clone() else {
            self.set_clone_status("Click Combine first to generate audio.");
            return;
        };
        let Ok(anchor) = self.document.create_element("a") else {
            return;
        };
        let Ok(anchor) = anchor.dyn_into::<HtmlAnchorElement>() else {
            return;
        };
        anchor.set_href(&url);
        anchor.set_download("clone.wav");
        let _ = anchor.style().set_property("display", "none");
        let body = match self.document.body() {
            Some(b) => b,
            None => return,
        };
        let _ = body.append_child(&anchor);
        anchor.click();
        let _ = body.remove_child(&anchor);
    }

    fn fetch_clone_status(app: App) {
        spawn_local(async move {
            match get_json_with_signature("/api/clone/status").await {
                Ok((data, signature)) => {
                    let installing = bool_prop(&data, "installing");
                    let available = bool_prop(&data, "available");
                    let install_error = string_prop(&data, "install_error", "");
                    let has_ref = bool_prop(&data, "has_reference");
                    let ref_text = string_prop(&data, "reference_text", "");

                    let prev_signature = app.state.borrow().clone_status_signature.clone();
                    if prev_signature == signature {
                        return;
                    }
                    {
                        let mut state = app.state.borrow_mut();
                        state.clone_status_signature = signature;
                        state.clone_f5_installing = installing;
                        state.clone_f5_available = available;
                    }

                    let synthesizing = app.state.borrow().clone_synthesizing;
                    app.clone_combine_button.set_disabled(
                        synthesizing || installing || !available,
                    );
                    app.clone_ref_button.set_disabled(synthesizing);

                    if !install_error.is_empty() {
                        app.set_clone_status(&format!("f5-tts install failed: {install_error}"));
                    } else if installing {
                        app.set_clone_status(
                            "Installing f5-tts and pulling deps (one-time, can take several minutes)...",
                        );
                    } else if !available {
                        app.set_clone_status("f5-tts is not installed yet.");
                    } else if !synthesizing {
                        // Show reference state when not actively synthesizing
                        if has_ref && app.state.borrow().clone_ref_file.is_none() {
                            // Pre-populate fields from server's saved reference
                            if app.clone_ref_text.value().trim().is_empty()
                                && !ref_text.is_empty()
                            {
                                app.clone_ref_text.set_value(&ref_text);
                                app.update_clone_char_counts();
                            }
                            app.clone_ref_name
                                .set_text_content(Some("(server has a saved reference)"));
                            let _ = app.clone_ref_name.class_list().add_1("set");
                        }
                        app.set_clone_status("Ready.");
                    }

                    app.update_nav_visibility();
                }
                Err(error) => log_error(&format!("Failed to fetch clone status: {error}")),
            }
        });
    }

    fn download_selected_model(&self) {
        let Some(model) = self.state.borrow().pending_download_model.clone() else {
            return;
        };
        let label = self
            .state
            .borrow()
            .pending_download_label
            .clone()
            .unwrap_or_else(|| model.clone());

        self.download_model_button.set_disabled(true);
        self.download_model_button
            .set_text_content(Some(&format!("Downloading {label}...")));
        {
            let mut state = self.state.borrow_mut();
            state.active = false;
            state.can_listen = false;
            state.downloading_model = true;
        }
        let _ = self.set_state("idle", 0.0);

        let app = self.clone();
        spawn_local(async move {
            let body = format!(r#"{{"model":"{}"}}"#, json_string_escape(&model));
            match post_json("/api/download-model", Some(body)).await {
                Ok(_) => {
                    let _ = app.close_download_modal();
                    app.state.borrow_mut().active = false;
                    app.state.borrow_mut().can_listen = false;
                    App::fetch_models(app);
                }
                Err(error) => {
                    app.download_model_button.set_disabled(false);
                    app.download_model_button
                        .set_text_content(Some("Start download model"));
                    log_error(&format!("Failed to download model: {error}"));
                }
            }
        });
    }
}

fn query<T: JsCast>(document: &Document, selector: &str) -> Result<T, JsValue> {
    document
        .query_selector(selector)?
        .ok_or_else(|| js_err(&format!("missing selector {selector}")))?
        .dyn_into::<T>()
        .map_err(|_| js_err(&format!("wrong element type for {selector}")))
}

fn query_all(document: &Document, selector: &str) -> Result<Vec<Element>, JsValue> {
    let list = document.query_selector_all(selector)?;
    let mut elements = Vec::new();
    for index in 0..list.length() {
        if let Some(node) = list.item(index) {
            elements.push(node.dyn_into::<Element>()?);
        }
    }
    Ok(elements)
}

fn now(window: &Window) -> f64 {
    window.performance().map(|perf| perf.now()).unwrap_or(0.0)
}

fn segment_color(index: usize) -> String {
    let hue = 120.0 - (index as f64 / (METER_SEGMENTS - 1) as f64) * 120.0;
    format!("hsl({hue}, 100%, 50%)")
}

fn threshold_to_block(threshold: f64) -> usize {
    ((threshold * 1000.0).round() as isize).clamp(1, METER_SEGMENTS as isize) as usize
}

fn set_style<T: AsRef<JsValue>>(element: &T, name: &str, value: &str) {
    if let Some(html) = element.as_ref().dyn_ref::<HtmlElement>() {
        let _ = html.style().set_property(name, value);
    } else if let Some(element) = element.as_ref().dyn_ref::<Element>() {
        let _ = element
            .dyn_ref::<HtmlElement>()
            .map(|html| html.style().set_property(name, value));
    }
}

fn toggle_class<T: AsRef<JsValue>>(element: &T, class_name: &str, enabled: bool) -> Result<(), JsValue> {
    let element = element
        .as_ref()
        .dyn_ref::<Element>()
        .ok_or_else(|| js_err("expected Element"))?;
    if enabled {
        element.class_list().add_1(class_name)
    } else {
        element.class_list().remove_1(class_name)
    }
}

async fn get_json(url: &str) -> Result<JsValue, String> {
    get_json_with_signature(url).await.map(|(value, _)| value)
}

async fn get_json_with_signature(url: &str) -> Result<(JsValue, String), String> {
    let window = web_sys::window().ok_or_else(|| "missing window".to_string())?;
    let response_value = JsFuture::from(window.fetch_with_str(url))
        .await
        .map_err(debug_js)?;
    let response: Response = response_value.dyn_into().map_err(debug_js)?;
    if !response.ok() {
        return Err(format!("HTTP {}", response.status()));
    }
    let text_value = JsFuture::from(response.text().map_err(debug_js)?)
        .await
        .map_err(debug_js)?;
    let text = text_value.as_string().unwrap_or_default();
    let parsed = JSON::parse(&text).map_err(debug_js)?;
    Ok((parsed, text))
}

async fn post_text_for_blob(url: &str, text: &str) -> Result<web_sys::Blob, String> {
    let window = web_sys::window().ok_or_else(|| "missing window".to_string())?;
    let init = RequestInit::new();
    init.set_method("POST");
    init.set_mode(RequestMode::SameOrigin);
    init.set_body(&JsValue::from_str(text));

    let request = Request::new_with_str_and_init(url, &init).map_err(debug_js)?;
    request
        .headers()
        .set("Content-Type", "text/plain; charset=utf-8")
        .map_err(debug_js)?;

    let response_value = JsFuture::from(window.fetch_with_request(&request))
        .await
        .map_err(debug_js)?;
    let response: Response = response_value.dyn_into().map_err(debug_js)?;
    if !response.ok() {
        let status = response.status();
        let text = match response.text() {
            Ok(promise) => JsFuture::from(promise)
                .await
                .ok()
                .and_then(|value| value.as_string())
                .unwrap_or_default(),
            Err(_) => String::new(),
        };
        if let Ok(parsed) = JSON::parse(&text) {
            let message = string_prop(&parsed, "error", "");
            if !message.is_empty() {
                return Err(message);
            }
        }
        if text.is_empty() {
            return Err(format!("HTTP {status}"));
        }
        return Err(format!("HTTP {status}: {text}"));
    }
    let blob_value = JsFuture::from(response.blob().map_err(debug_js)?)
        .await
        .map_err(debug_js)?;
    blob_value.dyn_into::<web_sys::Blob>().map_err(debug_js)
}

async fn post_blob(url: &str, file: &File) -> Result<JsValue, String> {
    let window = web_sys::window().ok_or_else(|| "missing window".to_string())?;
    let init = RequestInit::new();
    init.set_method("POST");
    init.set_mode(RequestMode::SameOrigin);
    init.set_body(file.as_ref());

    let request = Request::new_with_str_and_init(url, &init).map_err(debug_js)?;
    let mime = file.type_();
    let content_type = if mime.is_empty() {
        "application/octet-stream"
    } else {
        mime.as_str()
    };
    request
        .headers()
        .set("Content-Type", content_type)
        .map_err(debug_js)?;

    let response_value = JsFuture::from(window.fetch_with_request(&request))
        .await
        .map_err(debug_js)?;
    let response: Response = response_value.dyn_into().map_err(debug_js)?;
    if !response.ok() {
        let status = response.status();
        let text = match response.text() {
            Ok(promise) => JsFuture::from(promise)
                .await
                .ok()
                .and_then(|value| value.as_string())
                .unwrap_or_default(),
            Err(_) => String::new(),
        };
        if text.is_empty() {
            return Err(format!("HTTP {status}"));
        }
        if let Ok(parsed) = JSON::parse(&text) {
            let message = string_prop(&parsed, "error", "");
            if !message.is_empty() {
                return Err(message);
            }
        }
        return Err(format!("HTTP {status}: {text}"));
    }
    response
        .json()
        .map(JsFuture::from)
        .map_err(debug_js)?
        .await
        .map_err(debug_js)
}

async fn post_json(url: &str, body: Option<String>) -> Result<JsValue, String> {
    let window = web_sys::window().ok_or_else(|| "missing window".to_string())?;
    let init = RequestInit::new();
    init.set_method("POST");
    init.set_mode(RequestMode::SameOrigin);
    if let Some(body) = body {
        init.set_body(&JsValue::from_str(&body));
    }

    let request = Request::new_with_str_and_init(url, &init).map_err(debug_js)?;
    request
        .headers()
        .set("Content-Type", "application/json")
        .map_err(debug_js)?;

    let response_value = JsFuture::from(window.fetch_with_request(&request))
        .await
        .map_err(debug_js)?;
    let response: Response = response_value.dyn_into().map_err(debug_js)?;
    if !response.ok() {
        return Err(format!("HTTP {}", response.status()));
    }
    response.json().map(JsFuture::from).map_err(debug_js)?.await.map_err(debug_js)
}

async fn wait(ms: i32) {
    let promise = js_sys::Promise::new(&mut |resolve, _reject| {
        if let Some(window) = web_sys::window() {
            let _ = window.set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, ms);
        }
    });
    let _ = JsFuture::from(promise).await;
}

fn parse_tts_models(data: &JsValue) -> Result<TtsModelsData, JsValue> {
    let inner = parse_models(data)?;
    let piper_installing = bool_prop(data, "piper_installing");
    let piper_available = bool_prop(data, "piper_available");
    let raw_error = string_prop(data, "piper_install_error", "");
    let piper_install_error = if raw_error.is_empty() {
        None
    } else {
        Some(raw_error)
    };
    Ok(TtsModelsData {
        inner,
        piper_installing,
        piper_install_error,
        piper_available,
    })
}

fn parse_models(data: &JsValue) -> Result<ModelsData, JsValue> {
    let current = string_prop(data, "current", "");
    let models_value = Reflect::get(data, &JsValue::from_str("models"))?;
    let models_array = Array::from(&models_value);
    let mut models = Vec::new();

    for item in models_array.iter() {
        models.push(ModelOption {
            label: string_prop(&item, "label", ""),
            model: string_prop(&item, "model", ""),
            size_mb: number_prop(&item, "size_mb"),
            download_url: string_prop(&item, "download_url", ""),
            available: bool_prop(&item, "available"),
            downloading: bool_prop(&item, "downloading"),
            progress: number_prop(&item, "progress"),
        });
    }

    Ok(ModelsData { current, models })
}

fn has_prop(value: &JsValue, key: &str) -> bool {
    Reflect::has(value, &JsValue::from_str(key)).unwrap_or(false)
}

fn is_copyable_transcript(text: &str) -> bool {
    let text = text.trim();
    !text.is_empty()
        && text != "[no speech detected]"
        && !text.starts_with("Uploading ")
}

fn string_prop(value: &JsValue, key: &str, fallback: &str) -> String {
    Reflect::get(value, &JsValue::from_str(key))
        .ok()
        .and_then(|value| value.as_string())
        .unwrap_or_else(|| fallback.to_string())
}

fn number_prop(value: &JsValue, key: &str) -> f64 {
    Reflect::get(value, &JsValue::from_str(key))
        .ok()
        .and_then(|value| value.as_f64())
        .unwrap_or(0.0)
}

fn bool_prop(value: &JsValue, key: &str) -> bool {
    Reflect::get(value, &JsValue::from_str(key))
        .ok()
        .and_then(|value| value.as_bool())
        .unwrap_or(false)
}

fn json_string_escape(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

fn url_encode_component(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric()
            || byte == b'-'
            || byte == b'_'
            || byte == b'.'
            || byte == b'~'
        {
            out.push(byte as char);
        } else {
            out.push_str(&format!("%{:02X}", byte));
        }
    }
    out
}

fn log_error(message: &str) {
    web_sys::console::error_1(&JsValue::from_str(message));
}

fn js_err(message: &str) -> JsValue {
    JsValue::from_str(message)
}

fn debug_js(value: JsValue) -> String {
    format!("{value:?}")
}

fn display_error(error: &str) -> String {
    if is_fetch_error(error) {
        "Server stopped".to_string()
    } else {
        format!("Error: {error}")
    }
}

fn is_fetch_error(error: &str) -> bool {
    error.contains("Failed to fetch")
}
