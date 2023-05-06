use crate::clock::*;

use std::cell::{Cell};

pub struct SwapInnerCell {
  clk:  Cell<Clock>,
}
