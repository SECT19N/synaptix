use crate::types::ActivationFunction;

#[derive(Debug, Clone)]
pub struct LayerConfig {
    pub neurons: usize,
    pub activation: ActivationFunction,
}
