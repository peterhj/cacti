use libc::{CLOCK_BOOTTIME, clock_gettime, timespec};

use std::mem::{zeroed};
use std::ops::{Sub};

#[derive(Clone, Copy)]
pub struct Stopwatch {
  t:  [timespec; 2],
}

#[derive(Clone, Copy)]
pub struct Timestamp {
  t:  timespec,
}

impl Stopwatch {
  pub fn new() -> Stopwatch {
    unsafe {
      let mut t = [zeroed(), zeroed()];
      let res = clock_gettime(CLOCK_BOOTTIME, &mut t[0]);
      if res != 0 {
        panic!("bug");
      }
      Stopwatch{t}
    }
  }

  pub fn stamp(&mut self) -> Timestamp {
    unsafe {
      let res = clock_gettime(CLOCK_BOOTTIME, &mut self.t[1]);
      if res != 0 {
        panic!("bug");
      }
    }
    self.t.swap(0, 1);
    Timestamp{t: self.t[0]}
  }

  pub fn lap(&mut self) -> f64 {
    unsafe {
      let res = clock_gettime(CLOCK_BOOTTIME, &mut self.t[1]);
      if res != 0 {
        panic!("bug");
      }
    }
    let d = 1.0e-9 * (self.t[1].tv_nsec as f64 - self.t[0].tv_nsec as f64)
                + (self.t[1].tv_sec as f64 - self.t[0].tv_sec as f64);
    self.t.swap(0, 1);
    d
  }
}

impl Timestamp {
  pub fn s(&self) -> i32 {
    self.t.tv_sec as _
  }

  pub fn sub_ns(&self) -> i32 {
    self.t.tv_nsec as _
  }
}

impl Sub<Timestamp> for Timestamp {
  type Output = f64;

  fn sub(self, rhs: Timestamp) -> f64 {
    let d = 1.0e-9 * (self.t.tv_nsec as f64 - rhs.t.tv_nsec as f64)
                + (self.t.tv_sec as f64 - rhs.t.tv_sec as f64);
    d
  }
}
