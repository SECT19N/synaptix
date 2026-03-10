pub mod activation;
pub mod backward;
pub mod forward;
pub mod loss;
pub mod network;

pub use activation::{activate, activate_grad, activate_layer};
pub use loss::{compute_loss, compute_loss_grad};
