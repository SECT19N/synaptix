use eframe::egui::Color32;

pub fn dark_visuals() -> egui::Visuals {
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

pub const INDIGO: Color32 = Color32::from_rgb(99, 88, 200);
pub const INDIGO_DIM: Color32 = Color32::from_rgb(60, 55, 120);
pub const TEAL: Color32 = Color32::from_rgb(45, 200, 170);
pub const SURFACE: Color32 = Color32::from_rgb(22, 22, 35);
pub const SURFACE2: Color32 = Color32::from_rgb(28, 28, 44);
pub const TEXT: Color32 = Color32::from_rgb(220, 220, 240);
pub const TEXT_DIM: Color32 = Color32::from_rgb(120, 120, 160);
pub const DANGER: Color32 = Color32::from_rgb(220, 80, 80);
