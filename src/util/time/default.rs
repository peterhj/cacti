use std::time::{Instant};

#[derive(Clone, Copy)]
pub struct Stopwatch {
  t:  [Instant; 2],
}

impl Stopwatch {
  pub fn new() -> Stopwatch {
    let t0 = Instant::now();
    let t1 = t0.clone();
    let t = [t0, t1];
    Stopwatch{t}
  }

  pub fn lap(&mut self) -> f64 {
    self.t[1] = Instant::now();
    let dt = self.t[1] - self.t[0];
    let d = 1.0e-9 * dt.subsec_nanos() as f64 + dt.as_secs() as f64;
    self.t.swap(0, 1);
    d
  }
}
