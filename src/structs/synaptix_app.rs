use crate::structs::*;
use crate::types::*;

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
