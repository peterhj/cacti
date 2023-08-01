use crate::prelude::*;
//use crate::algo::{HashMap, HashSet};

#[derive(Clone, Copy, Debug)]
pub struct AdamW32 {
  pub grad_downscale: f32,
  pub lr: f32,
  pub wd: f32,
  pub a1: f32,
  pub a2: f32,
  pub eps: f32,
  //pub dtype: Dtype,
}

impl Default for AdamW32 {
  fn default() -> AdamW32 {
    AdamW32{
      grad_downscale: 1.0,
      lr: 1.0e-3,
      wd: 1.0e-2,
      a1: 1.0e-1,
      a2: 1.0e-3,
      eps: 1.0e-12,
      //dtype: f32::dtype(),
    }
  }
}

impl AdamW32 {
  pub fn step(&self, iter_nr: i32, master: &StableCell, grad1_avg: &StableCell, grad2_avg: &StableCell, grad: &StableCell) {
    grad1_avg.init_online_average_scale(grad, self.grad_downscale, self.a1);
    grad2_avg.init_online_average_square_scale(grad, self.grad_downscale, self.a2);
    master.init_online_adamw_update32(grad1_avg, grad2_avg,
        iter_nr,
        self.lr,
        self.wd,
        self.a1,
        self.a2,
        self.eps);
  }
}

/*pub struct AdamWState {
  pub iter_nr: i32,
  pub lr: f32,
  pub alpha1: f32,
  pub alpha2: f32,
  pub lamda: f32,
  pub eps: f32,
}

impl From<AdamWConfig> for AdamWState {
  fn from(cfg: AdamWConfig) -> AdamWState {
    AdamWState{
      iter_nr: 0,
      lr: cfg.lr,
      alpha1: cfg.alpha1,
      alpha2: cfg.alpha2,
      lamda: cfg.lamda,
      eps: cfg.eps,
    }
  }
}

pub struct AdamW {
  pub state: AdamWState,
  pub master: Vec<StableCell>,
  pub grad_avg: Vec<StableCell>,
  pub grad2_avg: Vec<StableCell>,
  /*pub master: HashSet<StableCell>,
  pub grad_avg: HashMap<StableCell, StableCell>,
  pub grad2_avg: HashMap<StableCell, StableCell>,*/
}

impl From<AdamWConfig> for AdamW {
  fn from(cfg: AdamWConfig) -> AdamW {
    let state = AdamWState::from(cfg);
    AdamW{
      state,
      master: Vec::new(),
      grad_avg: Vec::new(),
      grad2_avg: Vec::new(),
      /*master: HashSet::new(),
      grad_avg: HashMap::new(),
      grad2_avg: HashMap::new(),*/
    }
  }
}

impl AdamW {
  // TODO

  /*pub fn step(&mut self, grads: &HashMap<StableCell, StableCell>) {
    let n = self.state.iter_nr;
    for w in self.master.iter() {
      w.cache_init();
      let g = grads.get(&w).unwrap().clone();
      let mut g_avg = self.grad_avg.get(&w).unwrap().clone();
      // FIXME: coefficient.
      /*g_avg.cache_init();
      g_avg += (g - g_avg.clone()) * self.state.beta;*/
      g_avg.init_add_diff_scale(g, self.state.beta);
      let mut g2_avg = self.grad2_avg.get(&w).unwrap().clone();
      // FIXME: coefficient.
      /*g2_avg.cache_init();
      g2_avg += (g - g_avg.clone()).square() * self.state.gamma;*/
      g2_avg.init_add_square_diff_scale(g, self.state.gamma);
      // FIXME FIXME: weight decay.
      let mut w = w.clone();
      w += (g_avg / (g2_avg + self.state.eps).sqrt()) * self.state.lr;
    }
    self.state.iter_nr += 1;
  }*/
}*/

#[derive(Clone, Copy, Debug)]
pub struct ScaledAdamW16 {
  //pub iter_nr: i32,
  pub loss_upscale: f16,
  pub grad_downscale: f16,
  pub loss_grad_scale_prod: f16,
  pub lr: f16,
  pub wd: f16,
  pub a1: f16,
  pub a2: f16,
  pub eps: f16,
}
