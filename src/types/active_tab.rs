#[derive(Debug, Clone, PartialEq)]
pub enum ActiveTab {
    Build,
    Train,
    Inspect,
}

impl Default for ActiveTab {
    fn default() -> Self {
        ActiveTab::Build
    }
}
