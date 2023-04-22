#[cfg(feature = "gpu")]
use self::gpu::*;

use std::alloc::{Layout};
use std::cell::{Cell};
use std::rc::{Rc, Weak};

#[cfg(feature = "gpu")]
pub mod gpu;

pub struct TlsCtx {
  default_primary:  Cell<PMachSpec>,
  default_compute:  Cell<PMachSpec>,
  swap_cap:         Cell<usize>,
  gpu_reserve:      Cell<u16>,
  gpu_workspace:    Cell<u16>,
  ptr_ctr:          Cell<u32>,
}

impl Default for TlsCtx {
  fn default() -> TlsCtx {
    TlsCtx{
      default_primary:  Cell::new(PMachSpec::Cpu),
      default_compute:  Cell::new(PMachSpec::Cpu),
      swap_cap:         Cell::new(0),
      gpu_reserve:      Cell::new(9001),
      gpu_workspace:    Cell::new(111),
      ptr_ctr:          Cell::new(0),
    }
  }
}

thread_local! {
  static TL_CTX: TlsCtx = TlsCtx::default();
}

pub fn tl_get_default_primary() -> PMachSpec {
  TL_CTX.with(|ctx| ctx.default_primary.get())
}

pub fn tl_set_default_primary(spec: PMachSpec) {
  TL_CTX.with(|ctx| ctx.default_primary.set(spec))
}

pub fn tl_get_default_compute() -> PMachSpec {
  TL_CTX.with(|ctx| ctx.default_compute.get())
}

pub fn tl_set_default_compute(spec: PMachSpec) {
  TL_CTX.with(|ctx| ctx.default_compute.set(spec))
}

pub fn tl_get_swapfile_max_bytes() -> usize {
  TL_CTX.with(|ctx| ctx.swap_cap.get())
}

pub fn tl_set_swapfile_max_bytes(sz: usize) {
  TL_CTX.with(|ctx| ctx.swap_cap.set(sz))
}

pub fn tl_get_gpu_reserve_mem_per_10k() -> u16 {
  TL_CTX.with(|ctx| ctx.gpu_reserve.get())
}

pub fn tl_set_gpu_reserve_mem_per_10k(m: u16) {
  if m <= 0 {
    panic!("bug: gpu reserve too small: {}/10000", m);
  }
  if m >= 10000 {
    panic!("bug: gpu reserve too big: {}/10000", m);
  }
  TL_CTX.with(|ctx| ctx.gpu_reserve.set(m))
}

pub fn tl_get_gpu_workspace_mem_per_10k() -> u16 {
  TL_CTX.with(|ctx| ctx.gpu_workspace.get())
}

pub fn tl_set_gpu_workspace_mem_per_10k(m: u16) {
  if m <= 0 {
    panic!("bug: gpu workspace too small: {}/10000", m);
  }
  if m >= 10000 {
    panic!("bug: gpu workspace too big: {}/10000", m);
  }
  TL_CTX.with(|ctx| ctx.gpu_workspace.set(m))
}

pub fn tl_fresh_ptr() -> StablePtr {
  TL_CTX.with(|ctx| {
    let next = ctx.ptr_ctr.get() + 1;
    assert!(next >= 0);
    assert!(next < u32::max_value());
    ctx.ptr_ctr.set(next);
    StablePtr(next)
  })
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct StablePtr(u32);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Dtype {
  _Top,
  Float64,
  Float32,
  Float16,
  BFloat16,
  Int64,
  Int32,
  Int16,
  Int8,
  UInt64,
  UInt32,
  UInt16,
  UInt8,
}

#[derive(Clone, Copy)]
pub struct PMachFlag {
  bits: u8,
}

impl Default for PMachFlag {
  fn default() -> PMachFlag {
    PMachFlag{bits: 0}
  }
}

impl PMachFlag {
  pub fn set_primary(self) -> PMachFlag {
    PMachFlag{bits: self.bits | 1}
  }

  pub fn set_compute(self) -> PMachFlag {
    PMachFlag{bits: self.bits | 2}
  }

  pub fn primary(self) -> bool {
    (self.bits & 1) != 0
  }

  pub fn compute(self) -> bool {
    (self.bits & 2) != 0
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PMachSpec {
  _Top,
  Cpu,
  //Cpu(CpuSet),
  Swap,
  //Swap(SwapSet),
  #[cfg(feature = "gpu")]
  Gpu,
  //Gpu(GpuSet),
}

impl PMachSpec {
  pub fn flag(&self) -> PMachFlag {
    match self {
      &PMachSpec::Cpu => {
        PMachFlag::default().set_primary().set_compute()
      }
      &PMachSpec::Swap => {
        PMachFlag::default().set_primary()
      }
      #[cfg(feature = "gpu")]
      &PMachSpec::Gpu => {
        PMachFlag::default().set_primary().set_compute()
      }
      _ => panic!("bug: unimplemented")
    }
  }
}

pub struct PCell {
  pub ptr:      StablePtr,
  pub dtype:    Dtype,
  //pub shape:    _,
  pub primary:  InnerCell,
  pub compute:  InnerCell,
}

impl PCell {
  pub fn fresh(stableptr: StablePtr, dtype: Dtype) -> PCell {
    let primary = match tl_get_default_primary() {
      PMachSpec::Cpu => {
        InnerCell::Cpu(None)
      }
      _ => unimplemented!()
    };
    let compute = match tl_get_default_compute() {
      PMachSpec::Gpu => {
        InnerCell::Gpu(None, None)
      }
      _ => unimplemented!()
    };
    PCell{
      ptr: stableptr,
      dtype,
      primary,
      compute,
    }
  }
}

pub enum InnerCell {
  Primary,
  Cpu(Option<Weak<CpuInnerCell>>),
  Swap(Option<Weak<SwapInnerCell>>),
  #[cfg(feature = "gpu")]
  Gpu(Option<GpuInnerRef>, Option<Weak<GpuInnerCell>>),
}

/*
pub enum PCellState {
  Unalloc,
  Uninit,
  Init(u16),
}

pub enum PCell {
  Primary,
  Cpu(CpuPCell),
  Swap(SwapPCell),
  #[cfg(feature = "gpu")]
  Gpu(GpuPCell),
}

impl PCell {
  pub fn fresh(spec: PMachSpec) -> PCell {
    match spec {
      PMachSpec::Cpu => {
        PCell::Cpu(CpuPCell::default())
      }
      #[cfg(feature = "gpu")]
      PMachSpec::Gpu => {
        PCell::Gpu(GpuPCell::default())
      }
      _ => unimplemented!()
    }
  }

  pub fn primary_or_fresh(primary: PMachSpec, spec: PMachSpec) -> PCell {
    if primary == spec {
      PCell::Primary
    } else {
      PCell::fresh(spec)
    }
  }
}

pub struct CpuPCell {
  // FIXME FIXME: just use malloc'd buffer.
  //layout:   Layout,
  //ptr:      *mut u8,
  //sz:       usize,
  //pitch:    Vec<usize>,
  //shape:    Vec<usize>,
  dtype:        Dtype,
  laststate:    PCellState,
}

impl Default for CpuPCell {
  fn default() -> CpuPCell {
    CpuPCell{
      dtype:        Dtype::_Top,
      laststate:    PCellState::Unalloc,
    }
  }
}

pub struct SwapPCell {
  dtype:        Dtype,
  laststate:    PCellState,
}

impl Default for SwapPCell {
  fn default() -> SwapPCell {
    SwapPCell{
      dtype:        Dtype::_Top,
      laststate:    PCellState::Unalloc,
    }
  }
}
*/

pub struct CpuInnerCell {
  // FIXME
  //gpu_dep:  Option<Rc<GpuInnerCell>>,
  gpu_dep:  Option<Rc<GpuOuterCell>>,
  // TODO
}

impl CpuInnerCell {
  pub fn wait_gpu(&self) {
    match self.gpu_dep.as_ref() {
      None => {}
      Some(cel) => {
        cel.copy_.sync();
      }
    }
  }
}

pub struct SwapInnerCell {
}
