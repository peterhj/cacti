use libc::{CLOCK_BOOTTIME, clock_gettime, timespec};

use std::mem::{zeroed};

#[derive(Clone, Copy)]
pub struct Stopwatch {
  t:  [timespec; 2],
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
