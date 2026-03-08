use crate::structs::*;
use crate::theme::*;
use crate::types::*;
use crate::ui::*;
use eframe::egui;
use egui::{Color32, CornerRadius, FontId, RichText, Stroke, Vec2};

#[derive(Debug)]
pub struct SynaptixApp {
    pub active_tab: ActiveTab,

    pub input_count: usize,
    pub output_count: usize,
    pub hidden_layers: Vec<LayerConfig>,

    pub learning_rate: f64,
    pub epochs: usize,
    pub batch_size: usize,
    pub loss_function: LossFunction,
    pub lr_scheduler: LrScheduler,

    pub is_training: bool,
    pub current_epoch: usize,
    pub loss_history: Vec<f64>,

    pub dataset_path: Option<String>,

    pub inspect_inputs: Vec<f64>,
}

impl Default for SynaptixApp {
    fn default() -> Self {
        Self {
            active_tab: ActiveTab::Build,
            input_count: 2,
            output_count: 1,
            hidden_layers: vec![
                LayerConfig {
                    neurons: 3,
                    activation: ActivationFunction::ReLU,
                },
                LayerConfig {
                    neurons: 2,
                    activation: ActivationFunction::ReLU,
                },
            ],
            learning_rate: 0.1,
            epochs: 100,
            batch_size: 32,
            loss_function: LossFunction::MSE,
            lr_scheduler: LrScheduler::Constant,
            is_training: false,
            current_epoch: 0,
            loss_history: vec![],
            dataset_path: None,
            inspect_inputs: vec![],
        }
    }
}

impl eframe::App for SynaptixApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.render_topbar(ctx);
        self.render_left_panel(ctx);
        self.render_central_panel(ctx);
        self.render_bottom_bar(ctx);
    }
}

impl SynaptixApp {
    pub fn render_build_config(&mut self, ui: &mut egui::Ui) {
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
                egui::Frame::new()
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
                                    for act in ActivationFunction::all() {
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
                activation: ActivationFunction::ReLU,
            });
        }

        ui.add_space(16.0);
        section_header(ui, "Loss Function");

        egui::ComboBox::from_id_salt("loss_fn")
            .selected_text(self.loss_function.label())
            .width(ui.available_width())
            .show_ui(ui, |ui| {
                for (loss, label) in [
                    (LossFunction::MSE, "Mean Squared Error"),
                    (LossFunction::BinaryCrossEntropy, "Binary Cross-Entropy"),
                    (
                        LossFunction::CategoricalCrossEntropy,
                        "Categorical Cross-Entropy",
                    ),
                    (LossFunction::Huber, "Huber Loss"),
                ] {
                    ui.selectable_value(&mut self.loss_function, loss, label);
                }
            });
    }

    // ── Train config sidebar ──────────────────────────────────────────────────
    pub fn render_train_config(&mut self, ui: &mut egui::Ui) {
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
    pub fn render_inspect_config(&mut self, ui: &mut egui::Ui) {
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
                egui::Frame::new()
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
            egui::Frame::new()
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
                egui::Frame::new()
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
                        RichText::new(format!("Loss: {}", self.loss_function.label()))
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
