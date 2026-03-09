use crate::neural::{compute_loss, compute_loss_grad};
use crate::types::LossFunction;

#[cfg(test)]
mod loss_tests {
    use super::*;

    #[test]
    fn test_mse_perfect_prediction() {
        let pred = vec![1.0, 2.0, 3.0];
        let target = vec![1.0, 2.0, 3.0];
        let loss = compute_loss(&pred, &target, LossFunction::MSE);
        assert!((loss - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_mse_known_value() {
        // errors are [1, 1, 1], squared = [1,1,1], mean = 1.0
        let pred = vec![2.0, 3.0, 4.0];
        let target = vec![1.0, 2.0, 3.0];
        let loss = compute_loss(&pred, &target, LossFunction::MSE);
        assert!((loss - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_mse_grad_direction() {
        // grad should be positive when pred > target
        let pred = vec![1.5];
        let target = vec![1.0];
        let grad = compute_loss_grad(&pred, &target, LossFunction::MSE);
        assert!(grad[0] > 0.0);
    }

    #[test]
    fn test_bce_clamping() {
        // should not panic or produce NaN on edge values
        let pred = vec![0.0, 1.0];
        let target = vec![0.0, 1.0];
        let loss = compute_loss(&pred, &target, LossFunction::BinaryCrossEntropy);
        assert!(!loss.is_nan());
    }

    #[test]
    fn test_huber_small_error_equals_mse() {
        // when error < delta, huber should equal 0.5 * err²
        let pred = vec![1.1];
        let target = vec![1.0];
        let loss = compute_loss(&pred, &target, LossFunction::Huber);
        assert!((loss - 0.5 * 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_loss_and_grad_same_length() {
        let pred = vec![0.5, 0.3, 0.8];
        let target = vec![1.0, 0.0, 1.0];
        let grad = compute_loss_grad(&pred, &target, LossFunction::BinaryCrossEntropy);
        assert_eq!(grad.len(), pred.len());
    }
}
