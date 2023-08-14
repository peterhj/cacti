use crate::prelude::*;

/// An implementation of AdamW for 32-bit floating point.
#[derive(Clone, Copy, Debug)]
pub struct AdamW32 {
  /// A multiplicative un-scaling factor to apply to the
  /// gradients _before_ adding them to the exponential
  /// moving average moments.
  pub grad_unscale: f32,

  /// The learning rate.
  pub lr: f32,

  /// The weight decay.
  pub wd: f32,

  /// The dampening factor for the 1st order moment
  /// (relation to the original: `a1 + b1 = 1`).
  pub a1: f32,

  /// The dampening factor for the 2nd order moment
  /// (relation to the original: `a2 + b2 = 1`).
  pub a2: f32,

  /// The epsilon for the 2nd order moment.
  pub eps: f32,
}

impl Default for AdamW32 {
  fn default() -> AdamW32 {
    AdamW32{
      grad_unscale: 1.0,
      lr: 1.0e-3,
      wd: 0.0,
      a1: 1.0e-1,
      a2: 1.0e-3,
      eps: 1.0e-8,
      //dtype: f32::dtype(),
    }
  }
}

impl AdamW32 {
  pub fn step(&self, iter_nr: i32, master: &StableCell, grad1_avg: &StableCell, grad2_avg: &StableCell, grad: &StableCell) {
    grad1_avg.init_online_average_scale(grad, self.grad_unscale, self.a1);
    grad2_avg.init_online_average_square_scale(grad, self.grad_unscale, self.a2);
    master.init_online_adamw_update32(grad1_avg, grad2_avg,
        iter_nr,
        self.lr,
        self.wd,
        self.a1,
        self.a2,
        self.eps);
  }
}

#[derive(Clone, Copy, Debug)]
pub struct ScaledAdamW16 {
  pub grad_unscale: f16,
  //pub loss_upscale: f16,
  //pub loss_grad_scale_prod: f16,
  pub lr: f16,
  pub wd: f16,
  pub a1: f16,
  pub a2: f16,
  pub eps: f16,
}
