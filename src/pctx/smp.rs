use super::*;
use crate::clock::*;
#[cfg(feature = "gpu")]
use crate::pctx::nvgpu::{GpuOuterCell};

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

pub struct SmpPCtx {
}

impl SmpPCtx {
  pub fn new() -> SmpPCtx {
    let n = unsafe {
      (LIBCBLAS.openblas_get_num_threads.as_ref().unwrap())()
    };
    println!("DEBUG: SmpPCtx::new: blas num threads={}", n);
    // FIXME FIXME: debugging.
    let n = 1;
    unsafe {
      (LIBCBLAS.openblas_set_num_threads.as_ref().unwrap())(n)
    };
    println!("DEBUG: SmpPCtx::new: blas set num threads={}", n);
    let n = unsafe {
      (LIBCBLAS.openblas_get_num_threads.as_ref().unwrap())()
    };
    println!("DEBUG: SmpPCtx::new: blas num threads={}", n);
    SmpPCtx{
    }
  }

  pub fn append_matrix(&self, lp: &mut Vec<(Locus, PMach)>, pl: &mut Vec<(PMach, Locus)>) {
    lp.push((Locus::Mem, PMach::Smp));
    pl.push((PMach::Smp, Locus::Mem));
  }
}
