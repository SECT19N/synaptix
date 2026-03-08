use eframe::egui;
use egui::{Color32, CornerRadius, FontId, RichText, Stroke, Vec2};

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Snapticks")
            .with_inner_size([1200.0, 780.0])
            .with_min_inner_size([900.0, 600.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Snapticks",
        options,
        Box::new(|cc| {
            // Set a dark theme with custom colors
            cc.egui_ctx.set_visuals(dark_visuals());
            Ok(Box::new(SnaptickApp::default()))
        }),
    )
}

// ─── App State ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
enum ActiveTab {
    Build,
    Train,
    Inspect,
}

impl Default for ActiveTab {
    fn default() -> Self {
        ActiveTab::Build
    }
}

#[derive(Debug, Clone)]
struct LayerConfig {
    neurons: usize,
    activation: ActivationFn,
}

#[derive(Debug, Clone, PartialEq)]
enum ActivationFn {
    ReLU,
    LeakyReLU,
    Sigmoid,
    Tanh,
    Softmax,
    Linear,
    GELU,
    Swish,
    Mish,
    SELU,
    ELU,
}

impl ActivationFn {
    fn label(&self) -> &str {
        match self {
            ActivationFn::ReLU => "ReLU",
            ActivationFn::LeakyReLU => "Leaky ReLU",
            ActivationFn::Sigmoid => "Sigmoid",
            ActivationFn::Tanh => "Tanh",
            ActivationFn::Softmax => "Softmax",
            ActivationFn::Linear => "Linear",
            ActivationFn::GELU => "GELU",
            ActivationFn::Swish => "Swish",
            ActivationFn::Mish => "Mish",
            ActivationFn::SELU => "SELU",
            ActivationFn::ELU => "ELU",
        }
    }

    fn all() -> Vec<ActivationFn> {
        vec![
            ActivationFn::ReLU,
            ActivationFn::LeakyReLU,
            ActivationFn::Sigmoid,
            ActivationFn::Tanh,
            ActivationFn::Softmax,
            ActivationFn::Linear,
            ActivationFn::GELU,
            ActivationFn::Swish,
            ActivationFn::Mish,
            ActivationFn::SELU,
            ActivationFn::ELU,
        ]
    }
}

#[derive(Debug, Clone, PartialEq)]
enum LossFn {
    MSE,
    BinaryCrossEntropy,
    CategoricalCrossEntropy,
    Huber,
}

impl LossFn {
    fn label(&self) -> &str {
        match self {
            LossFn::MSE => "Mean Squared Error",
            LossFn::BinaryCrossEntropy => "Binary Cross-Entropy",
            LossFn::CategoricalCrossEntropy => "Categorical Cross-Entropy",
            LossFn::Huber => "Huber Loss",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
enum LrScheduler {
    Constant,
    StepDecay,
    ExponentialDecay,
    CosineAnnealing,
    ReduceOnPlateau,
}

impl LrScheduler {
    fn label(&self) -> &str {
        match self {
            LrScheduler::Constant => "Constant",
            LrScheduler::StepDecay => "Step Decay",
            LrScheduler::ExponentialDecay => "Exponential Decay",
            LrScheduler::CosineAnnealing => "Cosine Annealing",
            LrScheduler::ReduceOnPlateau => "Reduce on Plateau",
        }
    }
}

#[derive(Debug)]
struct SnaptickApp {
    // Navigation
    active_tab: ActiveTab,

    // Network architecture config
    input_count: usize,
    output_count: usize,
    hidden_layers: Vec<LayerConfig>,

    // Training config
    learning_rate: f64,
    epochs: usize,
    batch_size: usize,
    loss_fn: LossFn,
    lr_scheduler: LrScheduler,

    // Training state (placeholders for now)
    is_training: bool,
    current_epoch: usize,
    loss_history: Vec<f32>,

    // Dataset
    dataset_path: Option<String>,

    // Inspect panel
    inspect_inputs: Vec<f64>,
}

impl Default for SnaptickApp {
    fn default() -> Self {
        Self {
            active_tab: ActiveTab::Build,
            input_count: 2,
            output_count: 1,
            hidden_layers: vec![
                LayerConfig {
                    neurons: 4,
                    activation: ActivationFn::ReLU,
                },
                LayerConfig {
                    neurons: 4,
                    activation: ActivationFn::ReLU,
                },
            ],
            learning_rate: 0.01,
            epochs: 100,
            batch_size: 32,
            loss_fn: LossFn::MSE,
            lr_scheduler: LrScheduler::Constant,
            is_training: false,
            current_epoch: 0,
            loss_history: vec![],
            dataset_path: None,
            inspect_inputs: vec![0.0, 0.0],
        }
    }
}

// ─── Visuals ──────────────────────────────────────────────────────────────────

fn dark_visuals() -> egui::Visuals {
    let mut v = egui::Visuals::dark();
    v.panel_fill = Color32::from_rgb(13, 13, 20);
    v.window_fill = Color32::from_rgb(18, 18, 28);
    v.faint_bg_color = Color32::from_rgb(22, 22, 35);
    v.extreme_bg_color = Color32::from_rgb(8, 8, 14);
    v.override_text_color = Some(Color32::from_rgb(220, 220, 240));
    v.widgets.noninteractive.bg_fill = Color32::from_rgb(28, 28, 44);
    v.widgets.inactive.bg_fill = Color32::from_rgb(35, 35, 55);
    v.widgets.hovered.bg_fill = Color32::from_rgb(55, 50, 100);
    v.widgets.active.bg_fill = Color32::from_rgb(99, 88, 200);
    v.selection.bg_fill = Color32::from_rgb(80, 70, 180);
    v
}

// Accent colors
const INDIGO: Color32 = Color32::from_rgb(99, 88, 200);
const INDIGO_DIM: Color32 = Color32::from_rgb(60, 55, 120);
const TEAL: Color32 = Color32::from_rgb(45, 200, 170);
const SURFACE: Color32 = Color32::from_rgb(22, 22, 35);
const SURFACE2: Color32 = Color32::from_rgb(28, 28, 44);
const TEXT_DIM: Color32 = Color32::from_rgb(120, 120, 160);
const TEXT: Color32 = Color32::from_rgb(220, 220, 240);
const DANGER: Color32 = Color32::from_rgb(220, 80, 80);

// ─── App impl ─────────────────────────────────────────────────────────────────

impl eframe::App for SnaptickApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.render_topbar(ctx);
        self.render_left_panel(ctx);
        self.render_central_panel(ctx);
        self.render_bottom_bar(ctx);
    }
}

impl SnaptickApp {
    // ── Top bar ───────────────────────────────────────────────────────────────
    fn render_topbar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("topbar")
            .exact_height(52.0)
            .frame(
                egui::Frame::none()
                    .fill(Color32::from_rgb(10, 10, 18))
                    .inner_margin(egui::Margin::symmetric(16, 16)),
            )
            .show(ctx, |ui| {
                ui.horizontal_centered(|ui| {
                    // Logo
                    ui.label(
                        RichText::new("⚡ Snapticks")
                            .font(FontId::proportional(20.0))
                            .color(INDIGO)
                            .strong(),
                    );

                    ui.add_space(32.0);

                    // Tab buttons
                    for (tab, label, icon) in [
                        (ActiveTab::Build, "Build", "🧱"),
                        (ActiveTab::Train, "Train", "📈"),
                        (ActiveTab::Inspect, "Inspect", "🔍"),
                    ] {
                        let is_active = self.active_tab == tab;
                        let text = RichText::new(format!("{} {}", icon, label))
                            .font(FontId::proportional(14.0))
                            .color(if is_active { TEXT } else { TEXT_DIM });

                        let btn = ui.add(
                            egui::Button::new(text)
                                .fill(if is_active {
                                    INDIGO_DIM
                                } else {
                                    Color32::TRANSPARENT
                                })
                                .stroke(Stroke::new(if is_active { 1.0 } else { 0.0 }, INDIGO))
                                .corner_radius(CornerRadius::same(6))
                                .min_size(Vec2::new(100.0, 34.0)),
                        );

                        if btn.clicked() {
                            self.active_tab = tab;
                        }
                    }

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(
                            RichText::new("v0.0.1-dev")
                                .font(FontId::proportional(12.0))
                                .color(TEXT_DIM),
                        );
                    });
                });
            });
    }

    // ── Left config panel ─────────────────────────────────────────────────────
    fn render_left_panel(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("config_panel")
            .resizable(true)
            .default_width(280.0)
            .min_width(220.0)
            .max_width(380.0)
            .frame(
                egui::Frame::none()
                    .fill(Color32::from_rgb(15, 15, 24))
                    .inner_margin(egui::Margin::same(16)),
            )
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| match self.active_tab {
                    ActiveTab::Build => self.render_build_config(ui),
                    ActiveTab::Train => self.render_train_config(ui),
                    ActiveTab::Inspect => self.render_inspect_config(ui),
                });
            });
    }

    // ── Build config sidebar ──────────────────────────────────────────────────
    fn render_build_config(&mut self, ui: &mut egui::Ui) {
        section_header(ui, "Architecture");

        config_row(ui, "Inputs", |ui| {
            ui.add(
                egui::DragValue::new(&mut self.input_count)
                    .range(1..=64)
                    .speed(1),
            );
        });

        config_row(ui, "Outputs", |ui| {
            ui.add(
                egui::DragValue::new(&mut self.output_count)
                    .range(1..=64)
                    .speed(1),
            );
        });

        ui.add_space(12.0);
        section_header(ui, "Hidden Layers");

        let mut layer_to_remove: Option<usize> = None;

        for (i, layer) in self.hidden_layers.iter_mut().enumerate() {
            ui.push_id(i, |ui| {
                egui::Frame::none()
                    .fill(SURFACE2)
                    .corner_radius(CornerRadius::same(8))
                    .inner_margin(egui::Margin::same(10))
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.label(
                                RichText::new(format!("Layer {}", i + 1))
                                    .font(FontId::proportional(12.0))
                                    .color(TEXT_DIM),
                            );
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    if ui
                                        .add(
                                            egui::Button::new(
                                                RichText::new("✕").color(DANGER).size(11.0),
                                            )
                                            .fill(Color32::TRANSPARENT)
                                            .frame(false),
                                        )
                                        .clicked()
                                    {
                                        layer_to_remove = Some(i);
                                    }
                                },
                            );
                        });

                        ui.add_space(6.0);

                        config_row(ui, "Neurons", |ui| {
                            ui.add(
                                egui::DragValue::new(&mut layer.neurons)
                                    .range(1..=512)
                                    .speed(1),
                            );
                        });

                        config_row(ui, "Activation", |ui| {
                            egui::ComboBox::from_id_salt(format!("act_{}", i))
                                .selected_text(layer.activation.label())
                                .width(130.0)
                                .show_ui(ui, |ui| {
                                    for act in ActivationFn::all() {
                                        let label = act.label().to_string();
                                        ui.selectable_value(&mut layer.activation, act, label);
                                    }
                                });
                        });
                    });

                ui.add_space(6.0);
            });
        }

        if let Some(idx) = layer_to_remove {
            self.hidden_layers.remove(idx);
        }

        if ui
            .add(
                egui::Button::new(
                    RichText::new("＋  Add Layer")
                        .font(FontId::proportional(13.0))
                        .color(TEAL),
                )
                .fill(Color32::TRANSPARENT)
                .stroke(Stroke::new(1.0, TEAL))
                .corner_radius(CornerRadius::same(6))
                .min_size(Vec2::new(ui.available_width(), 32.0)),
            )
            .clicked()
        {
            self.hidden_layers.push(LayerConfig {
                neurons: 4,
                activation: ActivationFn::ReLU,
            });
        }

        ui.add_space(16.0);
        section_header(ui, "Loss Function");

        egui::ComboBox::from_id_salt("loss_fn")
            .selected_text(self.loss_fn.label())
            .width(ui.available_width())
            .show_ui(ui, |ui| {
                for (loss, label) in [
                    (LossFn::MSE, "Mean Squared Error"),
                    (LossFn::BinaryCrossEntropy, "Binary Cross-Entropy"),
                    (LossFn::CategoricalCrossEntropy, "Categorical Cross-Entropy"),
                    (LossFn::Huber, "Huber Loss"),
                ] {
                    ui.selectable_value(&mut self.loss_fn, loss, label);
                }
            });
    }

    // ── Train config sidebar ──────────────────────────────────────────────────
    fn render_train_config(&mut self, ui: &mut egui::Ui) {
        section_header(ui, "Dataset");

        let dataset_label = self.dataset_path.as_deref().unwrap_or("No file loaded");

        ui.label(
            RichText::new(dataset_label)
                .font(FontId::proportional(12.0))
                .color(if self.dataset_path.is_some() {
                    TEAL
                } else {
                    TEXT_DIM
                }),
        );

        ui.add_space(4.0);

        if ui
            .add(
                egui::Button::new(RichText::new("📂  Load CSV").color(TEXT))
                    .fill(SURFACE2)
                    .corner_radius(CornerRadius::same(6))
                    .min_size(Vec2::new(ui.available_width(), 30.0)),
            )
            .clicked()
        {
            // rfd file dialog will go here in Phase 2
            self.dataset_path = Some("dataset.csv (stub)".to_string());
        }

        ui.add_space(16.0);
        section_header(ui, "Hyperparameters");

        config_row(ui, "Learning Rate", |ui| {
            ui.add(
                egui::DragValue::new(&mut self.learning_rate)
                    .range(0.0001..=10.0)
                    .speed(0.0001)
                    .max_decimals(6),
            );
        });

        config_row(ui, "Epochs", |ui| {
            ui.add(
                egui::DragValue::new(&mut self.epochs)
                    .range(1..=10000)
                    .speed(1),
            );
        });

        config_row(ui, "Batch Size", |ui| {
            ui.add(
                egui::DragValue::new(&mut self.batch_size)
                    .range(1..=1024)
                    .speed(1),
            );
        });

        ui.add_space(16.0);
        section_header(ui, "LR Scheduler");

        egui::ComboBox::from_id_salt("lr_scheduler")
            .selected_text(self.lr_scheduler.label())
            .width(ui.available_width())
            .show_ui(ui, |ui| {
                for (sched, label) in [
                    (LrScheduler::Constant, "Constant"),
                    (LrScheduler::StepDecay, "Step Decay"),
                    (LrScheduler::ExponentialDecay, "Exponential Decay"),
                    (LrScheduler::CosineAnnealing, "Cosine Annealing"),
                    (LrScheduler::ReduceOnPlateau, "Reduce on Plateau"),
                ] {
                    ui.selectable_value(&mut self.lr_scheduler, sched, label);
                }
            });

        ui.add_space(20.0);

        let train_label = if self.is_training {
            "⏹  Stop Training"
        } else {
            "▶  Start Training"
        };
        let train_color = if self.is_training { DANGER } else { INDIGO };

        if ui
            .add(
                egui::Button::new(RichText::new(train_label).color(TEXT).size(14.0))
                    .fill(train_color)
                    .corner_radius(CornerRadius::same(8))
                    .min_size(Vec2::new(ui.available_width(), 40.0)),
            )
            .clicked()
        {
            self.is_training = !self.is_training;
        }
    }

    // ── Inspect config sidebar ────────────────────────────────────────────────
    fn render_inspect_config(&mut self, ui: &mut egui::Ui) {
        section_header(ui, "Manual Inputs");

        ui.label(
            RichText::new("Enter input values to step through a single forward pass.")
                .font(FontId::proportional(12.0))
                .color(TEXT_DIM),
        );

        ui.add_space(8.0);

        // Sync input fields to current input_count
        self.inspect_inputs.resize(self.input_count, 0.0);

        for i in 0..self.input_count {
            config_row(ui, &format!("x{}", i + 1), |ui| {
                ui.add(
                    egui::DragValue::new(&mut self.inspect_inputs[i])
                        .speed(0.01)
                        .max_decimals(4),
                );
            });
        }

        ui.add_space(16.0);

        if ui
            .add(
                egui::Button::new(RichText::new("▶  Run Forward Pass").color(TEXT).size(13.0))
                    .fill(INDIGO)
                    .corner_radius(CornerRadius::same(8))
                    .min_size(Vec2::new(ui.available_width(), 36.0)),
            )
            .clicked()
        {
            // Forward pass logic goes here in Phase 1
        }

        ui.add_space(8.0);

        if ui
            .add(
                egui::Button::new(
                    RichText::new("🔬  Run Backprop Step")
                        .color(TEXT)
                        .size(13.0),
                )
                .fill(SURFACE2)
                .corner_radius(CornerRadius::same(8))
                .min_size(Vec2::new(ui.available_width(), 36.0)),
            )
            .clicked()
        {
            // Backprop inspection goes here in Phase 1
        }
    }

    // ── Central panel ─────────────────────────────────────────────────────────
    fn render_central_panel(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default()
            .frame(
                egui::Frame::none()
                    .fill(Color32::from_rgb(13, 13, 20))
                    .inner_margin(egui::Margin::same(20)),
            )
            .show(ctx, |ui| match self.active_tab {
                ActiveTab::Build => self.render_network_diagram(ui),
                ActiveTab::Train => self.render_train_view(ui),
                ActiveTab::Inspect => self.render_inspect_view(ui),
            });
    }

    // ── Network diagram ───────────────────────────────────────────────────────
    fn render_network_diagram(&self, ui: &mut egui::Ui) {
        panel_title(ui, "Network Diagram");

        let painter = ui.painter();
        let rect = ui.available_rect_before_wrap();

        // Build layer sizes: input → hidden layers → output
        let mut all_layers: Vec<usize> = vec![self.input_count];
        for l in &self.hidden_layers {
            all_layers.push(l.neurons);
        }
        all_layers.push(self.output_count);

        let n_layers = all_layers.len();
        let h_padding = 60.0;
        let v_padding = 40.0;
        let x_step = (rect.width() - h_padding * 2.0) / (n_layers as f32 - 1.0).max(1.0);
        let max_neurons = *all_layers.iter().max().unwrap_or(&1);
        let node_radius =
            (((rect.height() - v_padding * 2.0) / (max_neurons as f32 * 2.2)).min(22.0)).max(6.0);
        let v_step_fn =
            |count: usize| -> f32 { (rect.height() - v_padding * 2.0) / (count as f32 + 1.0) };

        // Compute node positions
        let mut positions: Vec<Vec<egui::Pos2>> = Vec::new();
        for (li, &count) in all_layers.iter().enumerate() {
            let x = rect.left() + h_padding + li as f32 * x_step;
            let vstep = v_step_fn(count);
            let col: Vec<egui::Pos2> = (1..=count)
                .map(|ni| egui::pos2(x, rect.top() + v_padding + ni as f32 * vstep))
                .collect();
            positions.push(col);
        }

        // Draw edges first (behind nodes)
        for li in 0..positions.len().saturating_sub(1) {
            for &src in &positions[li] {
                for &dst in &positions[li + 1] {
                    painter.line_segment(
                        [src, dst],
                        Stroke::new(0.6, Color32::from_rgba_premultiplied(80, 75, 160, 60)),
                    );
                }
            }
        }

        // Draw nodes
        for (li, col) in positions.iter().enumerate() {
            let is_input = li == 0;
            let is_output = li == all_layers.len() - 1;

            let node_color = if is_input {
                TEAL
            } else if is_output {
                Color32::from_rgb(200, 100, 220)
            } else {
                INDIGO
            };

            for &pos in col {
                // Glow ring
                painter.circle_filled(
                    pos,
                    node_radius + 3.0,
                    Color32::from_rgba_premultiplied(
                        node_color.r(),
                        node_color.g(),
                        node_color.b(),
                        30,
                    ),
                );
                // Node fill
                painter.circle_filled(pos, node_radius, SURFACE2);
                // Node border
                painter.circle_stroke(pos, node_radius, Stroke::new(1.5, node_color));
            }
        }

        // Layer labels
        let label_y = rect.bottom() - 18.0;
        for (li, col) in positions.iter().enumerate() {
            let x = col[0].x;
            let label = if li == 0 {
                "Input".to_string()
            } else if li == all_layers.len() - 1 {
                "Output".to_string()
            } else {
                format!("H{}", li)
            };

            painter.text(
                egui::pos2(x, label_y),
                egui::Align2::CENTER_CENTER,
                label,
                FontId::proportional(11.0),
                TEXT_DIM,
            );
        }
    }

    // ── Train view ────────────────────────────────────────────────────────────
    fn render_train_view(&self, ui: &mut egui::Ui) {
        panel_title(ui, "Training");

        // Status row
        ui.horizontal(|ui| {
            let status_color = if self.is_training { TEAL } else { TEXT_DIM };
            let status_text = if self.is_training {
                format!(
                    "● Training  —  Epoch {}/{}",
                    self.current_epoch, self.epochs
                )
            } else {
                "◌  Idle".to_string()
            };
            ui.label(RichText::new(status_text).color(status_color).size(13.0));
        });

        ui.add_space(16.0);

        // Loss chart placeholder
        placeholder_box(
            ui,
            "Loss Curve — egui_plot will render here in Phase 2",
            280.0,
        );

        ui.add_space(16.0);

        // LR schedule chart placeholder
        placeholder_box(
            ui,
            "Learning Rate Schedule — egui_plot will render here in Phase 2",
            180.0,
        );
    }

    // ── Inspect view ──────────────────────────────────────────────────────────
    fn render_inspect_view(&self, ui: &mut egui::Ui) {
        panel_title(ui, "Forward Pass Inspector");

        ui.label(
            RichText::new(
                "Run a forward pass from the left panel to see layer-by-layer values here.",
            )
            .color(TEXT_DIM)
            .size(13.0),
        );

        ui.add_space(16.0);

        // Per-layer value tables (placeholder)
        let mut all_layers: Vec<String> = vec!["Input".to_string()];
        for (i, _) in self.hidden_layers.iter().enumerate() {
            all_layers.push(format!("Hidden {}", i + 1));
        }
        all_layers.push("Output".to_string());

        for layer_name in &all_layers {
            egui::Frame::none()
                .fill(SURFACE2)
                .corner_radius(CornerRadius::same(8))
                .inner_margin(egui::Margin::same(12))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(layer_name).strong().color(TEXT).size(13.0));
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.label(
                                RichText::new("z / a values shown after forward pass")
                                    .color(TEXT_DIM)
                                    .size(11.0),
                            );
                        });
                    });
                });
            ui.add_space(6.0);
        }
    }

    // ── Bottom status bar ─────────────────────────────────────────────────────
    fn render_bottom_bar(&self, ctx: &egui::Context) {
        egui::TopBottomPanel::bottom("status_bar")
            .exact_height(28.0)
            .frame(
                egui::Frame::none()
                    .fill(Color32::from_rgb(8, 8, 14))
                    .inner_margin(egui::Margin::symmetric(16, 0)),
            )
            .show(ctx, |ui| {
                ui.horizontal_centered(|ui| {
                    let arch: Vec<String> = {
                        let mut a = vec![format!("{}", self.input_count)];
                        for l in &self.hidden_layers {
                            a.push(format!("{}", l.neurons));
                        }
                        a.push(format!("{}", self.output_count));
                        a
                    };
                    ui.label(
                        RichText::new(format!("Architecture: [{}]", arch.join(" → ")))
                            .font(FontId::proportional(11.0))
                            .color(TEXT_DIM),
                    );

                    ui.separator();

                    ui.label(
                        RichText::new(format!("Loss: {}", self.loss_fn.label()))
                            .font(FontId::proportional(11.0))
                            .color(TEXT_DIM),
                    );

                    ui.separator();

                    ui.label(
                        RichText::new(format!(
                            "LR: {:.4}  ·  Scheduler: {}",
                            self.learning_rate,
                            self.lr_scheduler.label()
                        ))
                        .font(FontId::proportional(11.0))
                        .color(TEXT_DIM),
                    );
                });
            });
    }
}

// ─── UI Helpers ───────────────────────────────────────────────────────────────

fn section_header(ui: &mut egui::Ui, title: &str) {
    ui.add_space(4.0);
    ui.label(
        RichText::new(title.to_uppercase())
            .font(FontId::proportional(10.5))
            .color(TEXT_DIM)
            .strong(),
    );
    ui.add(egui::Separator::default().spacing(8.0));
    ui.add_space(4.0);
}

fn panel_title(ui: &mut egui::Ui, title: &str) {
    ui.label(
        RichText::new(title)
            .font(FontId::proportional(18.0))
            .color(TEXT)
            .strong(),
    );
    ui.add_space(12.0);
}

fn config_row(ui: &mut egui::Ui, label: &str, widget: impl FnOnce(&mut egui::Ui)) {
    ui.horizontal(|ui| {
        ui.label(
            RichText::new(label)
                .color(TEXT_DIM)
                .font(FontId::proportional(13.0)),
        );
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), widget);
    });
    ui.add_space(4.0);
}

fn placeholder_box(ui: &mut egui::Ui, label: &str, height: f32) {
    let (rect, _) = ui.allocate_exact_size(
        Vec2::new(ui.available_width(), height),
        egui::Sense::hover(),
    );

    ui.painter().rect(
        rect,
        CornerRadius::same(8),
        SURFACE,
        Stroke::new(1.0, INDIGO_DIM),
        egui::StrokeKind::Inside,
    );

    ui.painter().text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        label,
        FontId::proportional(12.0),
        TEXT_DIM,
    );
}
