use crate::types::ActivationFunction;

pub fn activate(x: f64, func: &ActivationFunction) -> f64 {
    match func {
        ActivationFunction::ReLU => relu(x),
        ActivationFunction::LeakyReLU => leaky_relu(x, 0.01),
        ActivationFunction::Sigmoid => sigmoid(x),
        ActivationFunction::Tanh => tanh(x),
        ActivationFunction::Softmax => x, // handled at activate_layer()
        ActivationFunction::Linear => x,
        ActivationFunction::GELU => gelu(x),
        ActivationFunction::Swish => swish(x),
        ActivationFunction::Mish => mish(x),
        ActivationFunction::SELU => selu(x),
        ActivationFunction::ELU => elu(x, 1.0),
    }
}

pub fn activate_layer(layer: &[f64], func: &ActivationFunction) -> Vec<f64> {
    match func {
        ActivationFunction::Softmax => softmax(layer),
        _ => layer.iter().map(|&x| activate(x, func)).collect(),
    }
}

pub fn activate_grad(x: f64, func: &ActivationFunction) -> f64 {
    match func {
        ActivationFunction::ReLU => relu_grad(x),
        ActivationFunction::LeakyReLU => leaky_relu_grad(x, 0.01),
        ActivationFunction::Sigmoid => sigmoid_grad(x),
        ActivationFunction::Tanh => tanh_grad(x),
        ActivationFunction::Softmax => 1.0, // handled at loss level
        ActivationFunction::Linear => 1.0,
        ActivationFunction::GELU => gelu_grad(x),
        ActivationFunction::Swish => swish_grad(x),
        ActivationFunction::Mish => mish_grad(x),
        ActivationFunction::SELU => selu_grad(x),
        ActivationFunction::ELU => elu_grad(x, 1.0),
    }
}

fn relu(x: f64) -> f64 {
    x.max(0.0)
}

fn relu_grad(x: f64) -> f64 {
    if x > 0.0 { 1.0 } else { 0.0 }
}

fn leaky_relu(x: f64, alpha: f64) -> f64 {
    if x > 0.0 { x } else { alpha * x }
}

fn leaky_relu_grad(x: f64, alpha: f64) -> f64 {
    if x > 0.0 { 1.0 } else { alpha }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn sigmoid_grad(x: f64) -> f64 {
    let s = sigmoid(x);
    s * (1.0 - s)
}

fn tanh(x: f64) -> f64 {
    x.tanh()
}

fn tanh_grad(x: f64) -> f64 {
    1.0 - x.tanh().powi(2)
}

fn softmax(layer: &[f64]) -> Vec<f64> {
    let max = layer.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = layer.iter().map(|&x| (x - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

fn elu(x: f64, alpha: f64) -> f64 {
    if x > 0.0 { x } else { alpha * (x.exp() - 1.0) }
}

fn elu_grad(x: f64, alpha: f64) -> f64 {
    if x > 0.0 { 1.0 } else { elu(x, alpha) + alpha }
}

const SELU_ALPHA: f64 = 1.6732632423543772;
const SELU_SCALE: f64 = 1.0507009873554805;

fn selu(x: f64) -> f64 {
    SELU_SCALE
        * if x > 0.0 {
            x
        } else {
            SELU_ALPHA * (x.exp() - 1.0)
        }
}

fn selu_grad(x: f64) -> f64 {
    SELU_SCALE * if x > 0.0 { 1.0 } else { SELU_ALPHA * x.exp() }
}

fn gelu(x: f64) -> f64 {
    0.5 * x * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
}

fn gelu_grad(x: f64) -> f64 {
    let pi = std::f64::consts::PI;
    let c = (2.0 / pi).sqrt();
    let inner = c * (x + 0.044715 * x.powi(3));
    let tanh_v = inner.tanh();
    let sech2 = 1.0 - tanh_v.powi(2);
    0.5 * (1.0 + tanh_v) + 0.5 * x * sech2 * c * (1.0 + 3.0 * 0.044715 * x.powi(2))
}

fn swish(x: f64) -> f64 {
    x * sigmoid(x)
}

fn swish_grad(x: f64) -> f64 {
    let s = sigmoid(x);
    s + x * s * (1.0 - s)
}

fn mish(x: f64) -> f64 {
    x * (((1.0 + x.exp()).ln()).tanh())
}

fn mish_grad(x: f64) -> f64 {
    let sp = (1.0 + x.exp()).ln(); // softplus
    let tanh_sp = sp.tanh();
    let sech2 = 1.0 - tanh_sp.powi(2);
    let delta = x.exp() / (1.0 + x.exp()); // sigmoid(x)
    tanh_sp + x * sech2 * delta
}
