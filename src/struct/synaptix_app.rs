#[derive(Debug)]
pub struct SynaptixApp {
    active_tab: ActiveTab,

    input_count: usize,
    output_count: usize,
    hidden_layers: Vec<LayerConfig>,

    learning_rate: f64,
    epochs: usize,
    batch_size: usize,
    loss_function: LossFunction,
    lr_scheduler: LrScheduler,

    is_training: bool,
    current_epoch: usize,
    loss_history: Vec<f64>,

    dataset_path: Option<String>,

    inspect_inputs: Vec<f64>,
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
