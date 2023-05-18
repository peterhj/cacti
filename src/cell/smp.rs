#[cfg(feature = "gpu")]
use crate::cell::gpu::{GpuOuterCell};
use crate::clock::*;

use cacti_smp_c_ffi::*;

use std::cell::{Cell};
//use std::rc::{Rc};

pub struct SmpInnerCell {
  pub clk:      Cell<Clock>,
  // FIXME
  #[cfg(feature = "gpu")]
  pub gpu:      Option<GpuOuterCell>,
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

pub struct SmpCtx {
}

impl SmpCtx {
  pub fn new() -> SmpCtx {
    let n = unsafe {
      (LIBCBLAS.openblas_get_num_threads.as_ref().unwrap())()
    };
    println!("DEBUG: SmpCtx::new: blas num threads={}", n);
    let n = 4;
    unsafe {
      (LIBCBLAS.openblas_set_num_threads.as_ref().unwrap())(n)
    };
    println!("DEBUG: SmpCtx::new: blas set num threads={}", n);
    let n = unsafe {
      (LIBCBLAS.openblas_get_num_threads.as_ref().unwrap())()
    };
    println!("DEBUG: SmpCtx::new: blas num threads={}", n);
    SmpCtx{
    }
  }
}
