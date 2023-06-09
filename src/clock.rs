use std::cmp::{Ordering};
use std::fmt::{Debug, Formatter, Result as FmtResult};

pub fn rst_signed_distance(a: u32, b: u32) -> Option<i32> {
  match (a, b) {
    (0, 0) => return None,
    (0, _) => return Some(i32::min_value()),
    (_, 0) => return Some(i32::max_value()),
    _ => {}
  }
  if a == b {
    return Some(0);
  }
  let a = a - 1;
  let b = b - 1;
  let ab = a.wrapping_sub(b);
  let ba = b.wrapping_sub(a);
  let (ab, ba) = if a > b {
    (ab, ba - 1)
  } else {
    (ab - 1, ba)
  };
  let cutoff = (i32::max_value() as u32) >> 1;
  if ab > cutoff && ba > cutoff {
    return None;
  }
  Some(if ab < ba {
    ab as i32
  } else {
    -(ba as i32)
  })
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct Counter {
  pub rst:  u32,
}

impl Default for Counter {
  fn default() -> Counter {
    Counter{rst: 0}
  }
}

impl From<Clock> for Counter {
  #[inline(always)]
  fn from(clk: Clock) -> Counter {
    Counter{rst: clk.rst}
  }
}

impl Debug for Counter {
  fn fmt(&self, f: &mut Formatter) -> FmtResult {
    write!(f, "Counter(rst={})", self.rst)
  }
}

impl Counter {
  pub fn is_nil(&self) -> bool {
    self.rst == 0
  }

  pub fn advance(&self) -> Counter {
    let mut next_rst = self.rst.wrapping_add(1);
    if next_rst == 0 {
      next_rst = next_rst.wrapping_add(1);
    }
    Counter{rst: next_rst}
  }

  pub fn succeeds<Ctr: Into<Counter>>(&self, r_ctr: Ctr) -> bool {
    let r_ctr = r_ctr.into();
    if self.rst > 1 {
      r_ctr.rst.wrapping_add(1) == self.rst
    } else if self.rst == 1 {
      r_ctr.rst == u32::max_value()
    } else {
      false
    }
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Clock {
  pub rst:  u32,
  pub tup:  u32,
}

impl Default for Clock {
  fn default() -> Clock {
    Clock{rst: 0, tup: 0}
  }
}

impl From<Counter> for Clock {
  #[inline(always)]
  fn from(ctr: Counter) -> Clock {
    Clock{rst: ctr.rst, tup: 0}
  }
}

impl Debug for Clock {
  fn fmt(&self, f: &mut Formatter) -> FmtResult {
    write!(f, "Clock(rst={}, tup={})", self.rst, self.tup)
  }
}

impl Clock {
  pub fn ctr(&self) -> Counter {
    Counter{rst: self.rst}
  }

  pub fn advance(&self) -> Clock {
    let mut next_rst = self.rst.wrapping_add(1);
    if next_rst == 0 {
      next_rst = next_rst.wrapping_add(1);
    }
    Clock{rst: next_rst, tup: 0}
  }

  pub fn finish(&self) -> Clock {
    assert!(self.tup != u32::max_value());
    Clock{rst: self.rst, tup: u32::max_value()}
  }

  pub fn update(&self) -> Clock {
    let next_tup = self.tup + 1;
    assert!(next_tup != u32::max_value());
    Clock{rst: self.rst, tup: next_tup}
  }

  /*pub fn happens_after<Clk: Into<Clock>>(&self, r_clk: Clk) -> Option<bool> {
    let r_clk = r_clk.into();
    let diff = self.rst.wrapping_sub(r_clk.rst);
    if diff == 0 {
      if self.tup > r_clk.tup {
        return Some(true);
      } else {
        return Some(false);
      }
    // FIXME FIXME: these conditions are wrong; use intervals.
    } else if diff < 0x4000 {
      return Some(true);
    } else if diff > 0xc000 {
      return Some(false);
    }
    None
  }

  pub fn happens_before<Clk: Into<Clock>>(&self, r_clk: Clk) -> Option<bool> {
    let r_clk = r_clk.into();
    let diff = r_clk.rst.wrapping_sub(self.rst);
    if diff == 0 {
      if r_clk.tup > self.tup {
        return Some(true);
      } else {
        return Some(false);
      }
    // FIXME FIXME: these conditions are wrong; use intervals.
    } else if diff < 0x4000 {
      return Some(true);
    } else if diff > 0xc000 {
      return Some(false);
    }
    None
  }*/

  pub fn partial_cmp_<Clk: Into<Clock>>(&self, r_clk: Clk) -> Option<Ordering> {
    self.partial_cmp(&r_clk.into())
  }
}

impl PartialOrd for Clock {
  fn partial_cmp(&self, r_clk: &Clock) -> Option<Ordering> {
    match rst_signed_distance(self.rst, r_clk.rst) {
      None => None,
      Some(0) => {
        Some(self.tup.cmp(&r_clk.tup))
      }
      Some(d) => {
        if d > 0 {
          Some(Ordering::Greater)
        } else {
          Some(Ordering::Less)
        }
      }
    }
  }
}
