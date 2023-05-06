#[cfg(feature = "gpu")]
use crate::cell::gpu::*;
use crate::cell::smp::*;
use crate::cell::swap::*;
use crate::clock::*;
use crate::ctx::*;
use crate::ptr::*;
use crate::thunk::*;

//use std::alloc::{Layout};
use std::cell::{Cell, RefCell};
use std::mem::{size_of};
use std::ops::{Deref};
use std::rc::{Rc, Weak};
use std::slice::{from_raw_parts};
use std::str::{FromStr};

#[cfg(feature = "gpu")]
pub mod gpu;
pub mod smp;
pub mod swap;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(transparent)]
pub struct CellPtr(i32);

impl<'a> From<&'a CellPtr> for CellPtr {
  fn from(x: &'a CellPtr) -> CellPtr {
    *x
  }
}

impl<'a> From<&'a StableCell> for CellPtr {
  fn from(x: &'a StableCell) -> CellPtr {
    x.as_ptr()
  }
}

impl From<StableCell> for CellPtr {
  fn from(x: StableCell) -> CellPtr {
    x.into_ptr()
  }
}

impl CellPtr {
  pub fn from_unchecked(p: i32) -> CellPtr {
    CellPtr(p)
  }

  pub fn to_unchecked(&self) -> i32 {
    self.0
  }

  pub fn as_bytes_repr(&self) -> &[u8] {
    // SAFETY: This should be safe as the type is `Copy`.
    let ptr = (self as *const CellPtr) as *const u8;
    let len = size_of::<CellPtr>();
    assert_eq!(len, 4);
    unsafe { from_raw_parts(ptr, len) }
  }
}

#[repr(transparent)]
pub struct StableCell {
  ptr:  CellPtr,
}

impl Drop for StableCell {
  fn drop(&mut self) {
    ctx_release(self.ptr);
  }
}

impl Clone for StableCell {
  fn clone(&self) -> StableCell {
    StableCell::from(self.ptr)
  }
}

impl Deref for StableCell {
  type Target = CellPtr;

  fn deref(&self) -> &CellPtr {
    // SAFETY: The following should be safe as `StableCell` has the same
    // (transparent) repr as `CellPtr`.
    unsafe { &*((self as *const StableCell) as *const CellPtr) as &CellPtr }
  }
}

impl From<CellPtr> for StableCell {
  fn from(ptr: CellPtr) -> StableCell {
    ctx_retain(ptr);
    StableCell{ptr}
  }
}

impl StableCell {
  pub fn new_array<S: Into<Vec<i64>>>(shape: S, dtype: Dtype) -> StableCell {
    let shape: Vec<i64> = shape.into();
    unimplemented!();
  }

  pub fn into_ptr(self) -> CellPtr {
    self.ptr
  }

  pub fn as_ptr(&self) -> CellPtr {
    self.ptr
  }
}

#[derive(Clone, Copy, Debug)]
pub enum CellSpec {
  Primary,
  Compute,
  //Backup,
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
  _Top = 0,
  //Cpu = 1,
  //Cpu(CpuSet),
  Smp = 1,
  //Smp(CpuSet),
  Swap = 2,
  //Swap(SwapSet),
  #[cfg(feature = "gpu")]
  Gpu = 3,
  //Gpu(GpuSet),
}

impl PMachSpec {
  pub fn flag(&self) -> PMachFlag {
    match self {
      &PMachSpec::Smp => {
        PMachFlag::default().set_primary().set_compute()
      }
      &PMachSpec::Swap => {
        PMachFlag::default().set_primary()
      }
      #[cfg(feature = "gpu")]
      &PMachSpec::Gpu => {
        PMachFlag::default().set_compute()
      }
      _ => panic!("bug: unimplemented")
    }
  }
}

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

impl FromStr for Dtype {
  type Err = String;

  fn from_str(s: &str) -> Result<Dtype, String> {
    Ok(match s {
      "float64" => Dtype::Float64,
      "float32" => Dtype::Float32,
      "float16" => Dtype::Float16,
      "bfloat16" => Dtype::BFloat16,
      "int64" => Dtype::Int64,
      "int32" => Dtype::Int32,
      "int16" => Dtype::Int16,
      "int8" => Dtype::Int8,
      "uint64" => Dtype::UInt64,
      "uint32" => Dtype::UInt32,
      "uint16" => Dtype::UInt16,
      "uint8" => Dtype::UInt8,
      _ => return Err(s.to_owned())
    })
  }
}

impl Dtype {
  pub fn size_bytes(self) -> usize {
    match self {
      Dtype::_Top       => panic!("bug"),
      Dtype::Float64    => 8,
      Dtype::Float32    => 4,
      Dtype::Float16    => 2,
      Dtype::BFloat16   => 2,
      Dtype::Int64      => 8,
      Dtype::Int32      => 4,
      Dtype::Int16      => 2,
      Dtype::Int8       => 1,
      Dtype::UInt64     => 8,
      Dtype::UInt32     => 4,
      Dtype::UInt16     => 2,
      Dtype::UInt8      => 1,
    }
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum ShapeCompat {
  Equal,
  NewShape,
  AlignedPrefix,
  UnalignedPrefix,
  Incompat,
}

#[derive(Clone)]
pub struct CellType {
  pub shape:    Vec<i64>,
  pub dtype:    Dtype,
}

impl CellType {
  pub fn shape_compat(&self, orig: &CellType) -> ShapeCompat {
    if &self.shape == &orig.shape {
      return ShapeCompat::Equal;
    }
    let nd = self.shape.len();
    let o_nd = orig.shape.len();
    let mut span = Vec::with_capacity(nd);
    let mut o_span = Vec::with_capacity(o_nd);
    if nd == 0 {
      span.push(1);
    } else {
      span.push(self.shape[0]);
    }
    for d in 1 .. nd {
      span.push(self.shape[(nd - 1) - d] * span[d - 1]);
    }
    span.reverse();
    if o_nd == 0 {
      o_span.push(1);
    } else {
      o_span.push(orig.shape[0]);
    }
    for d in 1 .. o_nd {
      o_span.push(orig.shape[(o_nd - 1) - d] * o_span[d - 1]);
    }
    o_span.reverse();
    if span[0] == o_span[0] {
      return ShapeCompat::NewShape;
    } else if span[0] > o_span[0] {
      return ShapeCompat::Incompat;
    }
    let mut d = 1;
    let mut o_d = 1;
    while d < nd && o_d < o_nd {
      if span[d] == o_span[o_d] {
        return ShapeCompat::AlignedPrefix;
      } else if span[d] > o_span[o_d] {
        d += 1;
      } else if span[d] < o_span[o_d] {
        o_d += 1;
      } else {
        unreachable!();
      }
    }
    ShapeCompat::UnalignedPrefix
  }
}

#[derive(Clone)]
pub struct CellLayout {
  pub offset:   u64,
  pub pitch:    Vec<u64>,
}

impl CellLayout {
  pub fn new_packed(ty: &CellType) -> CellLayout {
    let offset = 0;
    let nd = ty.shape.len();
    let mut pitch = Vec::with_capacity(nd);
    pitch.push(ty.dtype.size_bytes() as u64);
    for d in 1 .. nd {
      let ds = ty.shape[nd - d];
      assert!(ds >= 0);
      pitch.push(ds as u64 * pitch[d - 1]);
    }
    pitch.reverse();
    CellLayout{offset, pitch}
  }

  pub fn is_packed(&self, ty: &CellType) -> bool {
    if self.offset != 0 {
      return false;
    }
    let np = self.pitch.len();
    let nd = ty.shape.len();
    let mut p = ty.dtype.size_bytes() as u64;
    if self.pitch[np - 1] != p {
      return false;
    }
    for d in 1 .. nd {
      let ds = ty.shape[nd - d];
      assert!(ds >= 0);
      p *= ds as u64;;
      if self.pitch[(np - 1) - d] != p {
        return false;
      }
    }
    true
  }
}

#[derive(Clone, Copy)]
#[repr(u8)]
pub enum CellMode {
  Top,
  Aff,
  Mux,
  //Fin,
}

impl Default for CellMode {
  fn default() -> CellMode {
    CellMode::Top
  }
}

#[derive(Clone, Copy, Default)]
pub struct CellFlag {
  bits: u8,
}

impl CellFlag {
  pub fn reset(&mut self) {
    self.bits &= 0xf0;
  }

  pub fn set_intro(&mut self) {
    self.bits |= 1;
  }

  pub fn intro(&self) -> bool {
    (self.bits & 1) != 0
  }

  pub fn set_seal(&mut self) {
    self.bits |= 2;
  }

  pub fn seal(&self) -> bool {
    (self.bits & 2) != 0
  }

  pub fn set_cache(&mut self) {
    self.bits |= 4;
  }

  pub fn cache(&self) -> bool {
    (self.bits & 4) != 0
  }

  pub fn set_eval(&mut self) {
    self.bits |= 8;
  }

  pub fn eval(&self) -> bool {
    (self.bits & 8) != 0
  }
}

pub struct PCell {
  // FIXME FIXME: to implement fusion w/ unique cells, simply change
  // the PCell's owning ptr; the original ptr then becomes dangling.
  pub ptr:  CellPtr,
  //pub ptr:  Cell<CellPtr>,
  pub ogty: CellType,
  pub lay:  CellLayout,
  pub mode: CellMode,
  pub flag: CellFlag,
  pub clk:  Clock,
  //pub flag: RefCell<CellFlag>,
  //pub clk:  Cell<Clock>,
  pub primary:  InnerCell,
  pub compute:  InnerCell,
}

impl PCell {
  pub fn fresh(ptr: CellPtr, shape: Vec<i64>, dtype: Dtype) -> PCell {
    //let ptr = Cell::new(ptr);
    let ogty = CellType{shape, dtype};
    let lay = CellLayout::new_packed(&ogty);
    let mode = CellMode::default();
      // FIXME FIXME
    let flag = CellFlag::default();
    let clk = Clock::default();
    let primary = InnerCell::empty(CellSpec::Primary, ctx_get_default_primary());
    let compute = InnerCell::empty(CellSpec::Compute, ctx_get_default_compute());
    PCell{
      ptr,
      ogty,
      lay,
      mode,
      flag,
      clk,
      //flag: RefCell::new(CellFlag::default()),
      //clk:  Cell::new(Clock::default()),
      primary,
      compute,
    }
  }
}

pub enum InnerCell {
  Uninit,
  Smp(Weak<SmpInnerCell>),
  Swap(Weak<SwapInnerCell>),
  #[cfg(feature = "gpu")]
  Gpu(Weak<GpuInnerCell>),
  Primary,
}

impl InnerCell {
  pub fn empty(spec: CellSpec, pmspec: PMachSpec) -> InnerCell {
    let pmflag = pmspec.flag();
    let valid = match spec {
      CellSpec::Primary => pmflag.primary(),
      CellSpec::Compute => pmflag.compute(),
    };
    if !valid {
      panic!("bug: InnerCell::empty: PMachSpec::{:?} does not support CellSpec::{:?}",
          pmspec, spec);
    }
    match pmspec {
      PMachSpec::_Top => panic!("bug"),
      PMachSpec::Smp  => InnerCell::Smp(Weak::default()),
      PMachSpec::Swap => InnerCell::Swap(Weak::default()),
      #[cfg(feature = "gpu")]
      PMachSpec::Gpu  => InnerCell::Gpu(Weak::default()),
    }
  }

  pub fn clk(&self) -> Clock {
    match self {
      &InnerCell::Uninit => {
        panic!("bug");
      }
      &InnerCell::Smp(ref cel) => {
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
      &InnerCell::Smp(ref cel) => {
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
      &InnerCell::Smp(ref cel) => {
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
      &InnerCell::Smp(ref cel) => {
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
