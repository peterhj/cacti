use super::*;
use crate::algo::{RevSortMap8};
use crate::cell::*;
use crate::clock::*;
/*#[cfg(feature = "nvgpu")]
use crate::pctx::nvgpu::{GpuOuterCell};
#[cfg(feature = "nvgpu")]
use cacti_gpu_cu_ffi::{cuda_mem_free_host, cuda_mem_alloc_host};
#[cfg(feature = "nvgpu")]
use cacti_gpu_cu_ffi::types::{CUDA_ERROR_OUT_OF_MEMORY, CUDA_ERROR_DEINITIALIZED};*/
use cacti_smp_c_ffi::*;

#[cfg(target_os = "linux")]
use libc::{__errno_location};
#[cfg(all(unix, not(target_os = "linux")))]
use libc::{__errno as __errno_location};
use libc::{ENOMEM, free, malloc, c_char, c_int, c_void};

use std::cell::{Cell};
//use std::rc::{Rc};

#[repr(C)]
pub struct MemCell {
  pub ptr:  *mut c_void,
  pub mask: usize,
}

impl Drop for MemCell {
  fn drop(&mut self) {
    if self.ptr.is_null() {
      return;
    }
    match (self.mask as u8) {
      0 => {
        unsafe {
          free(self.ptr);
        }
      }
      /*1 => {
        #[cfg(feature = "nvgpu")]
        unsafe {
          match cuda_mem_free_host(self.ptr) {
            Ok(_) => {}
            Err(CUDA_ERROR_DEINITIALIZED) => {}
            Err(_) => panic!("bug"),
          }
        }
      }*/
      _ => unreachable!()
    }
  }
}

impl MemCell {
  pub fn try_alloc(sz: usize) -> Result<MemCell, PMemErr> {
    // FIXME: assure 64-bit ptr.
    assert!(sz <= 0x00ff_ffff_ffff_ffff);
    unsafe {
      let ptr = malloc(sz);
      if ptr.is_null() {
        let e = *(__errno_location)();
        if e == ENOMEM {
          return Err(PMemErr::Oom);
        } else {
          return Err(PMemErr::Bot);
        }
      }
      let mask = (sz << 8);
      Ok(MemCell{ptr, mask})
    }
  }

  /*#[cfg(not(feature = "nvgpu"))]
  pub fn try_alloc_page_locked(sz: usize) -> Result<MemCell, PMemErr> {
    unimplemented!();
  }

  #[cfg(feature = "nvgpu")]
  pub fn try_alloc_page_locked(sz: usize) -> Result<MemCell, PMemErr> {
    // FIXME: assure 64-bit ptr.
    assert!(sz <= 0x00ff_ffff_ffff_ffff);
    let ptr = match cuda_mem_alloc_host(sz) {
      Err(CUDA_ERROR_OUT_OF_MEMORY) => {
        return Err(PMemErr::Oom);
      }
      Err(_) => {
        return Err(PMemErr::Bot);
      }
      Ok(ptr) => ptr
    };
    if ptr.is_null() {
      return Err(PMemErr::Bot);
    }
    let mask = (sz << 8) | 1;
    Ok(MemCell{ptr, mask})
  }*/

  pub fn as_reg(&self) -> MemReg {
    MemReg{
      ptr:  self.ptr,
      sz:   self.size_bytes(),
    }
  }

  pub fn size_bytes(&self) -> usize {
    self.mask >> 8
  }
}

pub struct SmpInnerCell {
  pub clk:  Cell<Clock>,
  pub mem:  MemCell,
  // FIXME
  //#[cfg(feature = "nvgpu")]
  //pub gpu:  Option<GpuOuterCell>,
  // TODO
}

impl SmpInnerCell {
  /*pub fn wait_gpu(&self) {
    match self.gpu.as_ref() {
      None => {}
      Some(cel) => {
        // FIXME FIXME: query spin wait.
        cel.write.event.sync().unwrap();
      }
    }
  }*/
}

impl InnerCell for SmpInnerCell {}

pub struct SmpPCtx {
}

impl PCtxImpl for SmpPCtx {
  //type ICel = SmpInnerCell;

  fn pmach(&self) -> PMach {
    PMach::Smp
  }

  fn fastest_locus(&self) -> Locus {
    Locus::Mem
  }

  fn append_matrix(&self, lp: &mut RevSortMap8<(Locus, PMach), ()>, pl: &mut RevSortMap8<(PMach, Locus), ()>) {
    lp.insert((Locus::Mem, PMach::Smp), ());
    pl.insert((PMach::Smp, Locus::Mem), ());
  }

  //fn try_alloc(&self, x: CellPtr, sz: usize, /*pmset: PMachSet,*/ locus: Locus) -> Result<Rc<dyn InnerCell_>, PMemErr> {}
  fn try_alloc(&self, pctr: &PCtxCtr, /*pmset: PMachSet,*/ locus: Locus, ty: &CellType) -> Result<PAddr, PMemErr> {
    unimplemented!();
  }
}

impl SmpPCtx {
  pub fn new() -> SmpPCtx {
    let n = (LIBCBLAS.openblas_get_num_threads.as_ref().unwrap())();
    println!("DEBUG: SmpPCtx::new: blas num threads={}", n);
    // FIXME FIXME: debugging.
    let n = 1;
    (LIBCBLAS.openblas_set_num_threads.as_ref().unwrap())(n);
    println!("DEBUG: SmpPCtx::new: blas set num threads={}", n);
    let n = (LIBCBLAS.openblas_get_num_threads.as_ref().unwrap())();
    println!("DEBUG: SmpPCtx::new: blas num threads={}", n);
    SmpPCtx{
    }
  }

  pub fn phy_core_ct(&self) -> u32 {
    // FIXME
    1
  }

  /*pub fn append_matrix(&self, lp: &mut Vec<(Locus, PMach)>, pl: &mut Vec<(PMach, Locus)>) {
    lp.push((Locus::Mem, PMach::Smp));
    pl.push((PMach::Smp, Locus::Mem));
  }*/

  /*pub fn try_mem_alloc(&self, sz: usize, pmset: PMachSet) -> Result<MemCell, PMemErr> {
    if pmset.contains(PMach::NvGpu) {
      MemCell::try_alloc_page_locked(sz)
    } else {
      MemCell::try_alloc(sz)
    }
  }*/
}

pub extern "C" fn tl_pctx_smp_mem_alloc_hook(ptr: *mut *mut c_void, sz: usize, raw_tag: *const c_char) -> c_int {
  // FIXME
  unimplemented!();
  /*
  assert!(!ptr.is_null());
  unsafe {
    let mem = malloc(sz);
    if mem.is_null() {
      return 1;
    }
    write(ptr, mem);
    0
  }
  */
}

pub extern "C" fn tl_pctx_smp_mem_free_hook(ptr: *mut c_void) -> c_int {
  // FIXME
  unimplemented!();
  /*
  assert!(!ptr.is_null());
  unsafe {
    free(ptr);
  }
  */
}

pub extern "C" fn tl_pctx_smp_mem_unify_hook(lhs_raw_tag: *mut c_char, rhs_raw_tag: *mut c_char) {
  // FIXME
}
