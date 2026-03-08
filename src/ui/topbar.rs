use crate::app::SynaptixApp;
use crate::theme::*;
use crate::types::ActiveTab;
use eframe::egui::{Color32, CornerRadius, FontId, RichText, Stroke, Vec2};

impl SynaptixApp {
    pub fn render_topbar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("topbar")
            .exact_height(52.0)
            .frame(
                egui::Frame::new()
                    .fill(Color32::from_rgb(10, 10, 18))
                    .inner_margin(egui::Margin::symmetric(16, 16)),
            )
            .show(ctx, |ui| {
                ui.horizontal_centered(|ui| {
                    // Logo
                    ui.label(
                        RichText::new("⚡ Synaptix")
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
}
