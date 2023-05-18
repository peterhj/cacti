use crate::clock::*;

use std::cell::{Cell};

pub struct SwapInnerCell {
  pub clk:  Cell<Clock>,
}
