#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(transparent)]
pub struct Counter {
  pub rst:  u16,
}

impl Default for Counter {
  fn default() -> Counter {
    Counter{rst: 0}
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
}

#[derive(Clone, Copy)]
pub struct Clock {
  pub rst:  u16,
  pub tup:  u16,
}

impl Default for Clock {
  fn default() -> Clock {
    Clock{rst: 0, tup: 0}
  }
}

impl From<Counter> for Clock {
  #[inline(always)]
  fn from(ctr: Counter) -> Clock {
    Clock{rst: ctr.rst, tup:  0}
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
    assert!(self.tup != u16::max_value());
    Clock{rst: self.rst, tup: u16::max_value()}
  }

  pub fn update(&self) -> Clock {
    let next_tup = self.tup + 1;
    assert!(next_tup != u16::max_value());
    Clock{rst: self.rst, tup: next_tup}
  }

  pub fn happens_after<Clk: Into<Clock>>(&self, r_clk: Clk) -> Option<bool> {
    let r_clk = r_clk.into();
    let diff = self.rst.wrapping_sub(r_clk.rst);
    if diff == 0 {
      if self.tup > r_clk.tup {
        return Some(true);
      } else {
        return Some(false);
      }
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
    } else if diff < 0x4000 {
      return Some(true);
    } else if diff > 0xc000 {
      return Some(false);
    }
    None
  }
}
