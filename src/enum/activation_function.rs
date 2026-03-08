#[derive(Debug, Clone, PartialEq)]
pub enum ActivationFunction {
    ReLU,
    LeakyReLU,
    Sigmoid,
    Tanh,
    Softmax,
    Linear,
    GELU,
    Swish,
    Mish,
    SELU,
    ELU,
}

impl ActivationFunction {
    fn label(&self) -> &str {
        match self {
            ActivationFunction::ReLU => "ReLU",
            ActivationFunction::LeakyReLU => "LeakyReLU",
            ActivationFunction::Sigmoid => "Sigmoid",
            ActivationFunction::Tanh => "Tanh",
            ActivationFunction::Softmax => "Softmax",
            ActivationFunction::Linear => "Linear",
            ActivationFunction::GELU => "GELU",
            ActivationFunction::Swish => "Swish",
            ActivationFunction::Mish => "Mish",
            ActivationFunction::SELU => "SELU",
            ActivationFunction::ELU => "ELU",
        }
    }

    fn all() -> Vec<ActivationFunction> {
        vec![
            ActivationFunction::ReLU,
            ActivationFunction::LeakyReLU,
            ActivationFunction::Sigmoid,
            ActivationFunction::Tanh,
            ActivationFunction::Softmax,
            ActivationFunction::Linear,
            ActivationFunction::GELU,
            ActivationFunction::Swish,
            ActivationFunction::Mish,
            ActivationFunction::SELU,
            ActivationFunction::ELU,
        ]
    }
}
