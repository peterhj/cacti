use crate::prelude::*;
use crate::algo::{HashMap, HashSet};

#[derive(Clone, Copy)]
pub struct AdamWConfig {
  pub lr: f32,
  pub beta: f32,
  pub gamma: f32,
  pub eps: f32,
  pub lamda: f32,
  pub dtype: Dtype,
}

impl Default for AdamWConfig {
  fn default() -> AdamWConfig {
    AdamWConfig{
      // FIXME: check these numbers.
      lr: 1.0e-4,
      beta: 1.0e-1,
      gamma: 1.0e-3,
      eps: 1.0e-12,
      lamda: 1.0e-2,
      dtype: f32::dtype(),
    }
  }
}

pub struct AdamWState {
  pub lr: f32,
  pub beta: f32,
  pub gamma: f32,
  pub eps: f32,
  pub lamda: f32,
  pub iter_nr: i64,
}

impl From<AdamWConfig> for AdamWState {
  fn from(cfg: AdamWConfig) -> AdamWState {
    AdamWState{
      lr: cfg.lr,
      beta: cfg.beta,
      gamma: cfg.gamma,
      eps: cfg.eps,
      // FIXME
      lamda: 0.0,
      iter_nr: 0,
    }
  }
}

pub struct AdamW {
  pub state: AdamWState,
  //pub master: Vec<StableCell>,
  pub master: HashSet<StableCell>,
  pub grad_avg: HashMap<StableCell, StableCell>,
  pub grad2_avg: HashMap<StableCell, StableCell>,
}

impl From<AdamWConfig> for AdamW {
  fn from(cfg: AdamWConfig) -> AdamW {
    let state = AdamWState::from(cfg);
    AdamW{
      state,
      master: HashSet::new(),
      grad_avg: HashMap::new(),
      grad2_avg: HashMap::new(),
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
}
