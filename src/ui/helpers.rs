use crate::theme::*;
use egui::{CornerRadius, FontId, RichText, Stroke, Vec2};

pub fn section_header(ui: &mut egui::Ui, title: &str) {
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

pub fn panel_title(ui: &mut egui::Ui, title: &str) {
    ui.label(
        RichText::new(title)
            .font(FontId::proportional(18.0))
            .color(TEXT)
            .strong(),
    );
    ui.add_space(12.0);
}

pub fn config_row(ui: &mut egui::Ui, label: &str, widget: impl FnOnce(&mut egui::Ui)) {
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

pub fn placeholder_box(ui: &mut egui::Ui, label: &str, height: f32) {
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
