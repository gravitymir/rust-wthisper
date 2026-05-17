use std::cell::RefCell;
use std::rc::Rc;

use js_sys::{Array, JSON, Reflect};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::{spawn_local, JsFuture};
use web_sys::{
    Document, Element, Event, HtmlButtonElement, HtmlElement, HtmlInputElement, Request,
    RequestInit, RequestMode, Response, Window,
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
    meter: Element,
    meter_track: Element,
    meter_tooltip: Element,
    meter_hover_tooltip: Element,
    silence_tail_slider: HtmlInputElement,
    silence_tail_value: Element,
    model_buttons: Element,
    device_details: Element,
    download_modal: Element,
    download_modal_title: Element,
    download_modal_text: Element,
    download_model_button: HtmlButtonElement,
    state: Rc<RefCell<State>>,
}

struct State {
    active: bool,
    in_flight: bool,
    current_state: String,
    state_started_at: f64,
    can_listen: bool,
    downloading_model: bool,
    device_details_set: bool,
    status_failures: u32,
    pending_download_model: Option<String>,
    pending_download_label: Option<String>,
    selected_activation_block: usize,
    current_threshold: f64,
    models_signature: String,
}

#[derive(Clone)]
struct ModelOption {
    label: String,
    model: String,
    size_mb: f64,
    available: bool,
    downloading: bool,
    progress: f64,
}

struct ModelsData {
    current: String,
    models: Vec<ModelOption>,
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
            meter: query(&document, "#meter")?,
            meter_track: query(&document, "#meter-track")?,
            meter_tooltip: query(&document, "#meter-tooltip")?,
            meter_hover_tooltip: query(&document, "#meter-hover-tooltip")?,
            silence_tail_slider: query::<HtmlInputElement>(&document, "#silence-tail-slider")?,
            silence_tail_value: query(&document, "#silence-tail-value")?,
            model_buttons: query(&document, "#model-buttons")?,
            device_details: query(&document, "#device-details")?,
            download_modal: query(&document, "#download-modal")?,
            download_modal_title: query(&document, "#download-modal-title")?,
            download_modal_text: query(&document, "#download-modal-text")?,
            download_model_button: query::<HtmlButtonElement>(&document, "#download-model-button")?,
            state: Rc::new(RefCell::new(State {
                active: true,
                in_flight: false,
                current_state: "idle".to_string(),
                state_started_at: now(&window),
                can_listen: false,
                downloading_model: false,
                device_details_set: false,
                status_failures: 0,
                pending_download_model: None,
                pending_download_label: None,
                selected_activation_block: 80,
                current_threshold: 0.08,
                models_signature: String::new(),
            })),
        })
    }

    fn start(&self) -> Result<(), JsValue> {
        self.build_meter_track()?;
        self.install_handlers()?;
        self.set_state("idle", 0.0)?;
        App::fetch_models(self.clone());
        self.interval(250, App::poll_status)?;
        self.interval(500, App::fetch_models)?;
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
            let value = app.silence_tail_slider.value();
            app.set_silence_tail_display(value.parse::<f64>().unwrap_or(1.5));
            app.update_silence_tail(value);
        });
        self.silence_tail_slider
            .add_event_listener_with_callback("input", closure.as_ref().unchecked_ref())?;
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

        {
            let mut state = self.state.borrow_mut();
            state.downloading_model = any_downloading;
            state.can_listen = current_available && !any_downloading;
            if any_downloading {
                state.active = false;
            }
        }

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
            toggle_class(&button, "locked", any_downloading && !option.downloading)?;
            toggle_class(&button, "active", option.model == data.current)?;

            let app = self.clone();
            let option_for_click = option.clone();
            let closure = Closure::<dyn FnMut(Event)>::new(move |_| {
                if any_downloading || option_for_click.downloading {
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

        if self.state.borrow().can_listen && !self.state.borrow().in_flight {
            self.state.borrow_mut().active = true;
            self.listen_loop();
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
            "Model \"{}\" is not downloaded yet.",
            option.label
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
        {
            let mut state = self.state.borrow_mut();
            if status != state.current_state {
                state.current_state = status.to_string();
                state.state_started_at = now(&self.window);
                if let Some(timer) = self.timer_for(status) {
                    timer.set_text_content(Some("0"));
                }
            }
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
        self.tiles.set_attribute("aria-label", label)
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
                        app.transcript
                            .set_text_content(Some(if text.is_empty() { "[no speech detected]" } else { &text }));
                    }
                    Err(error) => {
                        if error == "HTTP 409" {
                            wait(350).await;
                            continue;
                        }

                        if !is_fetch_error(&error) {
                            app.transcript
                                .set_text_content(Some(&display_error(&error)));
                        }
                        app.state.borrow_mut().active = false;
                    }
                }
            }

            app.state.borrow_mut().in_flight = false;
            let _ = app.set_state("idle", 0.0);
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
                    let _ = app.set_device_details(&data);
                }
                Err(error) => {
                    let _ = app.set_state("idle", 0.0);
                    let mut state = app.state.borrow_mut();
                    state.status_failures += 1;
                    if state.status_failures >= 4 && is_fetch_error(&error) {
                        app.transcript.set_text_content(Some("Server stopped"));
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
                        let _ = app.set_model_buttons(models, signature);
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
