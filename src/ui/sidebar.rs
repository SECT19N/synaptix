use crate::app::SynaptixApp;
use crate::types::*;
use eframe::egui::Color32;

impl SynaptixApp {
    pub fn render_left_panel(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("config_panel")
            .resizable(true)
            .default_width(280.0)
            .min_width(220.0)
            .max_width(380.0)
            .frame(
                egui::Frame::new()
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
}
