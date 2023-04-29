#[cfg(feature = "gpu")]
use crate::cell::gpu::*;
use crate::clock::*;
use crate::ctx::*;
use crate::ptr::*;
use crate::thunk::*;

//use std::alloc::{Layout};
use std::cell::{Cell};
use std::rc::{Rc, Weak};

#[cfg(feature = "gpu")]
pub mod gpu;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
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

#[derive(Clone, Copy)]
#[repr(u8)]
pub enum CellMode {
  _Top,
  Aff,
  Semi,
  Fin,
}

impl Default for CellMode {
  fn default() -> CellMode {
    CellMode::_Top
  }
}

#[derive(Clone, Copy, Default)]
pub struct CellFlag {
  bits: u8,
}

impl CellFlag {
  pub fn reset(&mut self) {
    self.bits = 0;
  }

  pub fn set_intro(&mut self) {
    self.bits |= 1;
  }

  pub fn set_seal(&mut self) {
    self.bits |= 2;
  }

  pub fn intro(&self) -> bool {
    (self.bits & 1) != 0
  }

  pub fn seal(&self) -> bool {
    (self.bits & 2) != 0
  }
}

/*#[derive(Clone, Copy)]
pub struct CellState {
  pub mode: CellMode,
  pub flag: CellFlag,
  // FIXME: clock per InnerCell.
  //pub clk:  Clock,
}*/

#[derive(Clone)]
pub struct CellType {
  pub dtype:    Dtype,
  pub shape:    Vec<i64>,
}

pub struct PCell {
  // FIXME FIXME: to implement fusion w/ unique cells, simply change
  // the PCell's owning ptr; the original ptr then becomes dangling.
  pub ptr:  CellPtr,
  //pub ptr:  Cell<CellPtr>,
  // FIXME FIXME: this is the original ty; aliases may have another ty.
  pub ty:   CellType,
  pub mode: CellMode,
  pub flag: CellFlag,
  pub clk:  Clock,
  pub primary:  InnerCell,
  pub compute:  InnerCell,
}

impl PCell {
  pub fn fresh(stableptr: StablePtr, dtype: Dtype, shape: Vec<i64>) -> PCell {
    let primary = match tl_ctx_get_default_primary() {
      PMachSpec::Cpu => {
        // FIXME
        //InnerCell::Cpu(None)
        unimplemented!();
      }
      _ => unimplemented!()
    };
    let compute = match tl_ctx_get_default_compute() {
      PMachSpec::Gpu => {
        // FIXME
        //InnerCell::Gpu(None, None)
        unimplemented!();
      }
      _ => unimplemented!()
    };
    PCell{
      ptr:  stableptr.into(),
      ty:   CellType{dtype, shape},
      mode: CellMode::default(),
      flag: CellFlag::default(),
      // FIXME
      clk:  Clock::default(),
      primary,
      compute,
    }
  }
}

pub enum InnerCell {
  Uninit,
  Cpu(Weak<CpuInnerCell>),
  Swap(Weak<SwapInnerCell>),
  #[cfg(feature = "gpu")]
  Gpu(Weak<GpuInnerCell>),
  Primary,
}

impl InnerCell {
  pub fn clk(&self) -> Clock {
    match self {
      &InnerCell::Uninit => {
        panic!("bug");
      }
      &InnerCell::Cpu(ref cel) => {
        unimplemented!();
      }
      &InnerCell::Gpu(ref cel) => {
        match Weak::upgrade(cel) {
          Some(cel) => {
            cel.clk.get()
          }
          None => {
            unimplemented!();
          }
        }
      }
      _ => unimplemented!()
    }
  }

  /*pub fn advance_clk(&self, rst: u16) {
    match self.as_ref() {
      InnerCell::Uninit => {
        panic!("bug");
      }
      _ => unimplemented!()
    }
  }

  pub fn update_clk(&self) {
    match self.as_ref() {
      InnerCell::Uninit => {
        panic!("bug");
      }
      _ => unimplemented!()
    }
  }*/

  pub fn synced(&self, ty: &CellType, clk: Clock) -> bool {
    match self {
      &InnerCell::Uninit => false,
      &InnerCell::Cpu(ref cel) => {
        unimplemented!();
      }
      &InnerCell::Gpu(ref cel) => {
        match Weak::upgrade(cel) {
          Some(cel) => {
            unimplemented!();
          }
          None => {
            unimplemented!();
          }
        }
      }
      _ => unimplemented!()
    }
  }

  pub fn sync_cell(&self, ty: &CellType, cel: &InnerCell, clk: Clock) {
    if self.synced(ty, clk) {
      return;
    }
    match self {
      &InnerCell::Uninit => {}
      &InnerCell::Cpu(ref cel) => {
        unimplemented!();
      }
      &InnerCell::Gpu(ref cel) => {
        match Weak::upgrade(cel) {
          Some(cel) => {
            unimplemented!();
          }
          None => {
            unimplemented!();
          }
        }
      }
      &InnerCell::Primary => {
        panic!("bug");
      }
      _ => unimplemented!()
    }
  }

  pub fn sync_thunk(&self, ty: &CellType, thunk: &PThunk, clk: Clock) {
    if self.synced(ty, clk) {
      return;
    }
    match self {
      &InnerCell::Uninit => {}
      &InnerCell::Cpu(ref cel) => {
        unimplemented!();
      }
      &InnerCell::Gpu(ref cel) => {
        match Weak::upgrade(cel) {
          Some(cel) => {
            unimplemented!();
          }
          None => {
            unimplemented!();
          }
        }
      }
      &InnerCell::Primary => {
        panic!("bug");
      }
      _ => unimplemented!()
    }
  }

  pub fn unsync(&self, clk: Clock) {
    unimplemented!();
  }
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
  clk:  Cell<Clock>,
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
        cel.copy_.sync().unwrap();
      }
    }
  }
}

pub struct SwapInnerCell {
  clk:  Cell<Clock>,
}
