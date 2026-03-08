#[derive(Debug, Clone, PartialEq)]
pub enum LrScheduler {
    Constant,
    StepDecay,
    ExponentialDecay,
    CosineAnnealing,
    ReduceOnPlateau,
}

impl LrScheduler {
    pub fn label(&self) -> &str {
        match self {
            LrScheduler::Constant => "Constant",
            LrScheduler::StepDecay => "Step Decay",
            LrScheduler::ExponentialDecay => "Exponential Decay",
            LrScheduler::CosineAnnealing => "Cosine Annealing",
            LrScheduler::ReduceOnPlateau => "Reduce On Plateau",
        }
    }
}
