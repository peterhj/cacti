use std::ops::{Sub};
use std::time::{Instant};

#[derive(Clone, Copy)]
pub struct Stopwatch {
  t:  [Instant; 2],
}

pub struct Timestamp {
  t: Instant,
}

impl Stopwatch {
  pub fn new() -> Stopwatch {
    let t0 = Instant::now();
    let t1 = t0.clone();
    let t = [t0, t1];
    Stopwatch{t}
  }

  pub fn stamp(&mut self) -> Timestamp {
    self.t[1] = Instant::now();
    self.t.swap(0, 1);
    Timestamp{t: self.t[0]}
  }

  pub fn lap(&mut self) -> f64 {
    self.t[1] = Instant::now();
    let dt = self.t[1] - self.t[0];
    let d = 1.0e-9 * dt.subsec_nanos() as f64 + dt.as_secs() as f64;
    self.t.swap(0, 1);
    d
  }
}

impl Timestamp {
  pub fn s(&self) -> i32 {
    unimplemented!();
  }

  pub fn sub_ns(&self) -> i32 {
    unimplemented!();
  }
}

impl Sub<Timestamp> for Timestamp {
  type Output = f64;

  fn sub(self, rhs: Timestamp) -> f64 {
    let dt = self.t - rhs.t;
    let d = 1.0e-9 * dt.subsec_nanos() as f64 + dt.as_secs() as f64;
    d
  }
}
