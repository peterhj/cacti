use super::*;
use crate::algo::{RevSortMap8};
use crate::algo::fp::*;
use crate::cell::*;
use crate::clock::*;
use crate::panick::*;
#[cfg(feature = "gpu")]
use crate::pctx::nvgpu::{GpuOuterCell};
#[cfg(feature = "gpu")]
use cacti_gpu_cu_ffi::{cuda_mem_free_host, cuda_mem_alloc_host};
#[cfg(feature = "gpu")]
use cacti_gpu_cu_ffi::types::{CUDA_ERROR_OUT_OF_MEMORY, CUDA_ERROR_DEINITIALIZED};
use cacti_smp_c_ffi::*;

#[cfg(target_os = "linux")]
use libc::{__errno_location};
#[cfg(all(unix, not(target_os = "linux")))]
use libc::{__errno as __errno_location};
use libc::{ENOMEM, free, malloc};

use std::borrow::{Borrow};
use std::cell::{Cell};
use std::cmp::{min};
use std::ffi::{c_void};
use std::io::{Read};
use std::mem::{align_of};
//use std::rc::{Rc};
use std::slice::{from_raw_parts, from_raw_parts_mut};

#[derive(Clone, Copy)]
#[repr(C)]
pub struct MemReg {
  pub ptr:  *mut c_void,
  pub sz:   usize,
}

impl MemReg {
  #[track_caller]
  pub fn copy_from_slice<T: DtypeExt + Copy/*, Buf: Borrow<[T]>*/>(&self, src_buf: &[T]) {
    panick_wrap(|| self._copy_from_slice(src_buf))
  }

  pub fn _copy_from_slice<T: DtypeExt + Copy/*, Buf: Borrow<[T]>*/>(&self, src_buf: &[T]) {
    //let src_buf = src_buf.borrow();
    let src_len = src_buf.len();
    let dsz = <T as DtypeExt>::dtype().size_bytes();
    let src_sz = dsz * src_len;
    assert_eq!(self.sz, src_sz);
    let src_start = src_buf.as_ptr() as usize;
    let src_end = src_start + src_sz;
    let dst_start = self.ptr as usize;
    let dst_end = dst_start + self.sz;
    if !(src_end <= dst_start || dst_end <= src_start) {
      panic!("bug: MemReg::_copy_from: overlapping src and dst");
    }
    unsafe {
      std::intrinsics::copy_nonoverlapping(src_buf.as_ptr() as *const u8, self.ptr as *mut u8, self.sz);
    }
  }

  #[track_caller]
  pub fn copy_from_reader<R: Read>(&self, src: R) {
    panick_wrap(|| self._copy_from_reader(src))
  }

  pub fn _copy_from_reader<R: Read>(&self, mut src: R) {
    let dst_buf = unsafe { from_raw_parts_mut(self.ptr as *mut u8, self.sz) };
    let mut dst_off = 0;
    loop {
      match src.read(&mut dst_buf[dst_off .. ]) {
        Err(_) => panic!("ERROR: I/O error"),
        Ok(0) => break,
        Ok(n) => {
          dst_off += n;
        }
      }
    }
  }

  pub fn _debug_dump_f32(&self) {
    let len = self.sz / 4;
    assert_eq!(0, self.sz % 4);
    assert_eq!(0, (self.ptr as usize) % align_of::<f32>());
    let buf = unsafe { from_raw_parts(self.ptr as *mut u8 as *const u8 as *const f32, len) };
    let start = 0;
    print!("DEBUG: MemReg: {:08x} :", start * 4);
    for i in start .. min(start + 8, len) {
      let x = buf[i];
      print!(" {}", x);
    }
    println!();
    if len <= 0 {
      return;
    }
    let start = (len - 1) - ((len - 1) & (8 - 1));
    print!("DEBUG: MemReg: {:08x} :", start * 4);
    for i in start .. min(start + 8, len) {
      let x = buf[i];
      print!(" {}", x);
    }
    println!();
  }

  pub fn _debug_dump_f16(&self) {
    let len = self.sz / 2;
    assert_eq!(0, self.sz % 2);
    assert_eq!(0, (self.ptr as usize) % align_of::<u16>());
    let buf = unsafe { from_raw_parts(self.ptr as *mut u8 as *const u8 as *const u16, len) };
    let start = 0;
    print!("DEBUG: MemReg: {:08x} :", start * 2);
    for i in start .. min(start + 8, len) {
      let x = f16::from_bits(buf[i]);
      print!(" {}", x);
    }
    println!();
    if len <= 0 {
      return;
    }
    let start = (len - 1) - ((len - 1) & (8 - 1));
    print!("DEBUG: MemReg: {:08x} :", start * 2);
    for i in start .. min(start + 8, len) {
      let x = f16::from_bits(buf[i]);
      print!(" {}", x);
    }
    println!();
  }
}

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
      1 => {
        #[cfg(feature = "gpu")]
        unsafe {
          match cuda_mem_free_host(self.ptr) {
            Ok(_) => {}
            Err(CUDA_ERROR_DEINITIALIZED) => {}
            Err(_) => panic!("bug"),
          }
        }
      }
      _ => unreachable!()
    }
  }
}

impl MemCell {
  pub fn try_alloc(sz: usize) -> Result<MemCell, PMemErr> {
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

  #[cfg(not(feature = "gpu"))]
  pub fn try_alloc_page_locked(sz: usize) -> Result<MemCell, PMemErr> {
    unimplemented!();
  }

  #[cfg(feature = "gpu")]
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
  }

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
  //#[cfg(feature = "gpu")]
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

  /*pub fn append_matrix(&self, lp: &mut Vec<(Locus, PMach)>, pl: &mut Vec<(PMach, Locus)>) {
    lp.push((Locus::Mem, PMach::Smp));
    pl.push((PMach::Smp, Locus::Mem));
  }*/

  pub fn try_mem_alloc(&self, sz: usize, pmset: PMachSet) -> Result<MemCell, PMemErr> {
    if pmset.contains(PMach::NvGpu) {
      MemCell::try_alloc_page_locked(sz)
    } else {
      MemCell::try_alloc(sz)
    }
  }
}
