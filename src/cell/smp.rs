#[cfg(feature = "gpu")]
use crate::cell::gpu::{GpuOuterCell};
use crate::clock::*;

use std::cell::{Cell};
use std::rc::{Rc};

pub struct SmpInnerCell {
  clk:      Cell<Clock>,
  // FIXME
  #[cfg(feature = "gpu")]
  gpu:      Option<GpuOuterCell>,
  // TODO
}

impl SmpInnerCell {
  pub fn wait_gpu(&self) {
    match self.gpu.as_ref() {
      None => {}
      Some(cel) => {
        // FIXME FIXME: query spin wait.
        cel.write.event.sync().unwrap();
      }
    }
  }
}
