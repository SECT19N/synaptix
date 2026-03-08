#[derive(Debug, Clone, PartialEq)]
pub enum LossFunction {
    MSE,
    BinaryCrossEntropy,
    CategoricalCrossEntropy,
    Huber,
}

impl LossFunction {
    fn label(&self) -> &str {
        match self {
            LossFunction::MSE => "MSE",
            LossFunction::BinaryCrossEntropy => "Binary Cross Entropy",
            LossFunction::CategoricalCrossEntropy => "Categorical Cross Entropy",
            LossFunction::Huber => "Huber",
        }
    }
}
