mod app;
mod structs;
mod theme;
mod types;
mod ui;

use crate::theme::dark_visuals;
use app::SynaptixApp;
use eframe::egui;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Synaptix")
            .with_inner_size([1200.0, 780.0])
            .with_min_inner_size([900.0, 600.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Synaptix",
        options,
        Box::new(|cc| {
            // Set a dark theme with custom colors
            cc.egui_ctx.set_visuals(dark_visuals());
            Ok(Box::new(SynaptixApp::default()))
        }),
    )
}
