use crate::types::LossFunction;

pub fn compute_loss(pred: &[f64], target: &[f64], loss_function: LossFunction) -> f64 {
    match loss_function {
        LossFunction::MSE => mse(pred, target),
        LossFunction::BinaryCrossEntropy => binary_cross_entropy(pred, target),
        LossFunction::CategoricalCrossEntropy => categorical_cross_entropy(pred, target),
        LossFunction::Huber => huber(pred, target, 1.0),
    }
}

pub fn compute_loss_grad(pred: &[f64], target: &[f64], loss_function: LossFunction) -> Vec<f64> {
    assert_eq!(
        pred.len(),
        target.len(),
        "pred and target must have the same length."
    );

    match loss_function {
        LossFunction::MSE => mse_grad(pred, target),
        LossFunction::BinaryCrossEntropy => binary_cross_entropy_grad(pred, target),
        LossFunction::CategoricalCrossEntropy => categorical_cross_entropy_grad(pred, target),
        LossFunction::Huber => huber_grad(pred, target, 1.0),
    }
}

fn mse(pred: &[f64], target: &[f64]) -> f64 {
    let n = pred.len() as f64;

    pred.iter()
        .zip(target.iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f64>()
        / n
}

fn mse_grad(pred: &[f64], target: &[f64]) -> Vec<f64> {
    let n = pred.len() as f64;

    pred.iter()
        .zip(target.iter())
        .map(|(p, t)| 2.0 * (p - t) / n)
        .collect()
}

fn binary_cross_entropy(pred: &[f64], target: &[f64]) -> f64 {
    let n = pred.len() as f64;

    pred.iter()
        .zip(target.iter())
        .map(|(p, t)| {
            let p = p.clamp(1e-12, 1.0 - 1e-12); // prevent log(0)
            -(t * p.ln() + (1.0 - t) * (1.0 - p).ln())
        })
        .sum::<f64>()
        / n
}

fn binary_cross_entropy_grad(pred: &[f64], target: &[f64]) -> Vec<f64> {
    let n = pred.len() as f64;

    pred.iter()
        .zip(target.iter())
        .map(|(p, t)| {
            let p = p.clamp(1e-12, 1.0 - 1e-12); // prevent log(0)
            (-(t / p) + (1.0 - t) / (1.0 - p)) / n
        })
        .collect()
}

fn categorical_cross_entropy(pred: &[f64], target: &[f64]) -> f64 {
    let n = pred.len() as f64;

    pred.iter()
        .zip(target.iter())
        .map(|(p, t)| {
            let p = p.max(1e-12); // prevent log(0)
            -t * p.ln()
        })
        .sum::<f64>()
        / n
}

fn categorical_cross_entropy_grad(pred: &[f64], target: &[f64]) -> Vec<f64> {
    let n = pred.len() as f64;

    pred.iter()
        .zip(target.iter())
        .map(|(p, t)| {
            let p = p.max(1e-12);
            (-t / p) / n
        })
        .collect()
}

fn huber(pred: &[f64], target: &[f64], delta: f64) -> f64 {
    let n = pred.len() as f64;

    pred.iter()
        .zip(target.iter())
        .map(|(p, t)| {
            let err = (p - t).abs();
            if err <= delta {
                0.5 * err.powi(2)
            } else {
                delta * (err - 0.5 * delta)
            }
        })
        .sum::<f64>()
        / n
}

fn huber_grad(pred: &[f64], target: &[f64], delta: f64) -> Vec<f64> {
    let n = pred.len() as f64;

    pred.iter()
        .zip(target.iter())
        .map(|(p, t)| {
            let err = p - t;
            if err.abs() <= delta {
                err / n
            } else {
                delta * err.signum() / n
            }
        })
        .collect()
}
