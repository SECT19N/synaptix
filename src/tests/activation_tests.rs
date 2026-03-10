use crate::neural::{activate, activate_grad, activate_layer};
use crate::types::ActivationFunction;

#[cfg(test)]
mod tests {
    use super::*;

    const SELU_ALPHA: f64 = 1.6732632423543772;
    const SELU_SCALE: f64 = 1.0507009873554805;

    // ReLU
    #[test]
    fn test_relu_positive() {
        let activation = activate(2.0, &ActivationFunction::ReLU);
        assert_eq!(activation, 2.0);
    }

    #[test]
    fn test_relu_negative() {
        let activation = activate(-3.0, &ActivationFunction::ReLU);
        assert_eq!(activation, 0.0);
    }

    #[test]
    fn test_relu_grad() {
        assert_eq!(activate_grad(1.0, &ActivationFunction::ReLU), 1.0);
        assert_eq!(activate_grad(-1.0, &ActivationFunction::ReLU), 0.0);
    }

    // Sigmoid
    #[test]
    fn test_sigmoid_zero() {
        let activation = activate(0.0, &ActivationFunction::Sigmoid);
        assert!((activation - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_sigmoid_range() {
        assert!(activate(100.0, &ActivationFunction::Sigmoid) <= 1.0);
        assert!(activate(-100.0, &ActivationFunction::Sigmoid) > 0.0);
    }

    // Tanh
    #[test]
    fn test_tanh_zero() {
        let activation = activate(0.0, &ActivationFunction::Tanh);
        assert!((activation - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_tanh_grad_at_zero() {
        let activation = activate_grad(0.0, &ActivationFunction::Tanh);
        assert!((activation - 1.0).abs() < 1e-10);
    }

    // Softmax
    #[test]
    fn test_softmax_sums_to_one() {
        let activation = activate_layer(&[1.0, 2.0, 3.0], &ActivationFunction::Softmax);
        let out: f64 = activation.iter().sum();
        assert!((out - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_softmax_largest_wins() {
        let activation = activate_layer(&[1.0, 2.0, 10.0], &ActivationFunction::Softmax);
        let out: f64 = activation.iter().sum();
        assert!(activation[2] > activation[1]);
        assert!(activation[2] > activation[0]);
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // should not produce NaN or inf with large values
        let activation = activate_layer(&[1000.0, 1001.0, 1002.0], &ActivationFunction::Softmax);
        assert!(activation.iter().all(|x| x.is_finite()));
    }

    // SELU constants
    #[test]
    fn test_selu_positive_is_scaled() {
        let out = activate_layer(&[1.0], &ActivationFunction::SELU);
        assert!((out[0] - SELU_SCALE * 1.0).abs() < 1e-10);
    }

    // GELU
    #[test]
    fn test_gelu_near_zero() {
        // gelu(0) should be 0
        let activation = activate_layer(&[0.0], &ActivationFunction::GELU);
        assert!(activation[0].abs() < 1e-6);
    }

    // Swish
    #[test]
    fn test_swish_zero() {
        let activation = activate_layer(&[0.0], &ActivationFunction::Swish);
        assert!(activation[0].abs() < 1e-10);
    }

    #[test]
    fn test_swish_positive_approx_linear() {
        // for large x, swish(x) ≈ x
        let activation = activate_layer(&[10.0], &ActivationFunction::Swish);
        assert!((activation[0] - 10.0).abs() < 0.01);
    }

    // Mish
    #[test]
    fn test_mish_zero() {
        let activation = activate_layer(&[0.0], &ActivationFunction::Mish);
        assert!(activation[0].abs() < 1e-10);
    }

    // activate_layer
    #[test]
    fn test_activate_layer_non_softmax() {
        let input = vec![1.0, -1.0, 0.0];
        let output = activate_layer(&input, &ActivationFunction::ReLU);
        assert_eq!(output, vec![1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_activate_layer_softmax() {
        let input = vec![1.0, 2.0, 3.0];
        let out: f64 = activate_layer(&input, &ActivationFunction::Softmax)
            .iter()
            .sum();
        assert!((out - 1.0).abs() < 1e-10);
    }
}
