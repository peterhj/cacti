use crate::prelude::*;

#[derive(Clone, Copy, Debug)]
pub struct SGD32 {
  pub grad_unscale: f32,
  pub lr: f32,
}

impl SGD32 {
  pub fn step(&self, master: &StableCell, grad: &StableCell) {
    master.init_online_add_scale2(grad,
        -self.lr * self.grad_unscale,
        1.0_f32);
  }
}

#[derive(Clone, Copy, Debug)]
pub struct MomentumSGD32 {
  // TODO: momentum, nesterov, wd, etc.
  pub lr: f32,
  pub wd: f32,
  pub mu: f32,
  pub nesterov: bool,
}
