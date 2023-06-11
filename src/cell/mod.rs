use crate::algo::{RevSortKey8, RevSortMap8};
use crate::algo::fp::*;
use crate::clock::*;
use crate::ctx::*;
use crate::pctx::{Locus, PMach};
use crate::thunk::*;
use crate::thunk::op::{SetScalarFutThunkSpec};
use crate::util::pickle::{TorchDtype};

use std::any::{Any};
use std::cell::{Cell};
use std::collections::{HashMap};
use std::convert::{TryFrom, TryInto};
use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::mem::{size_of, swap};
use std::ops::{Deref};
use std::rc::{Weak};
use std::slice::{from_raw_parts};
use std::str::{FromStr};

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct CellPtr(pub i32);

impl From<StableCell> for CellPtr {
  fn from(x: StableCell) -> CellPtr {
    x.into_ptr()
  }
}

impl<'a> From<&'a StableCell> for CellPtr {
  fn from(x: &'a StableCell) -> CellPtr {
    x.as_ptr()
  }
}

impl<'a> From<&'a CellPtr> for CellPtr {
  fn from(x: &'a CellPtr) -> CellPtr {
    *x
  }
}

impl AsRef<CellPtr> for CellPtr {
  fn as_ref(&self) -> &CellPtr {
    self
  }
}

impl Debug for CellPtr {
  fn fmt(&self, f: &mut Formatter) -> FmtResult {
    write!(f, "CellPtr({})", self.0)
  }
}

impl CellPtr {
  pub fn nil() -> CellPtr {
    CellPtr(0)
  }

  pub fn from_unchecked(p: i32) -> CellPtr {
    CellPtr(p)
  }

  pub fn to_unchecked(&self) -> i32 {
    self.0
  }

  pub fn is_nil(&self) -> bool {
    self.0 == 0
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
  pub ptr_: CellPtr,
}

impl From<CellPtr> for StableCell {
  fn from(ptr: CellPtr) -> StableCell {
    ctx_retain(ptr);
    StableCell{ptr_: ptr}
  }
}

impl From<f32> for StableCell {
  fn from(value: f32) -> StableCell {
    StableCell::scalar(value)
  }
}

impl AsRef<CellPtr> for StableCell {
  fn as_ref(&self) -> &CellPtr {
    self.as_ptr_ref()
  }
}

impl Deref for StableCell {
  type Target = CellPtr;

  fn deref(&self) -> &CellPtr {
    self.as_ptr_ref()
  }
}

impl Debug for StableCell {
  fn fmt(&self, f: &mut Formatter) -> FmtResult {
    write!(f, "StableCell({})", self.ptr_.0)
  }
}

impl Clone for StableCell {
  fn clone(&self) -> StableCell {
    StableCell::from(self.ptr_)
  }
}

impl Drop for StableCell {
  fn drop(&mut self) {
    ctx_release(self.ptr_);
  }
}

impl StableCell {
  pub fn retain(env: &CtxEnv, ptr: CellPtr) -> StableCell {
    assert!(!ptr.is_nil());
    env.retain(ptr);
    StableCell{ptr_: ptr}
  }

  pub fn scalar<T: ThunkValExt>(value: T) -> StableCell {
    ctx_pop_thunk(SetScalarFutThunkSpec{val: value.into_thunk_val()}).into()
  }

  pub fn array<S: Into<Vec<i64>>, D: TryInto<Dtype>>(shape: S, dtype: D) -> StableCell {
    let shape: Vec<i64> = shape.into();
    let dtype: Dtype = match dtype.try_into() {
      Ok(d) => d,
      Err(_) => panic!("bug: StableCell::new_array: invalid dtype")
    };
    let ty = CellType{shape, dtype};
    ctx_insert(ty).into()
  }

  pub fn release(mut self, env: &CtxEnv) -> CellPtr {
    let mut ptr = CellPtr::nil();
    swap(&mut ptr, &mut self.ptr_);
    env.release(ptr);
    ptr
  }

  #[inline]
  pub fn into_ptr(self) -> CellPtr {
    self.ptr_
  }

  #[inline]
  pub fn as_ptr(&self) -> CellPtr {
    self.ptr_
  }

  #[inline]
  pub fn as_ptr_ref(&self) -> &CellPtr {
    // SAFETY: The following is safe as `StableCell` has the same
    // (transparent) repr as `CellPtr`.
    unsafe { &*((self as *const StableCell) as *const CellPtr) as &CellPtr }
  }
}

/*pub struct Snapshot {
  pub ptr_: CellPtr,
  pub clk:  Clock,
}

pub struct StableSnapshot {
}*/

/*pub struct Checkpoint {
}

pub type StableCheckpoint = Checkpoint;*/

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Atom(pub i32);

impl Atom {
  pub fn nil() -> Atom {
    Atom(0)
  }

  pub fn from_unchecked(p: i32) -> Atom {
    Atom(p)
  }

  pub fn to_unchecked(&self) -> i32 {
    self.0
  }

  pub fn is_nil(&self) -> bool {
    self.0 == 0
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct MCellPtr(pub i32);

impl MCellPtr {
  pub fn nil() -> MCellPtr {
    MCellPtr(0)
  }

  pub fn from_unchecked(p: i32) -> MCellPtr {
    MCellPtr(p)
  }

  pub fn to_unchecked(&self) -> i32 {
    self.0
  }

  pub fn is_nil(&self) -> bool {
    self.0 == 0
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
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

impl TryFrom<TorchDtype> for Dtype {
  type Error = String;

  fn try_from(t: TorchDtype) -> Result<Dtype, String> {
    Ok(match t {
      TorchDtype::Float64 => Dtype::Float64,
      TorchDtype::Float32 => Dtype::Float32,
      TorchDtype::Float16 => Dtype::Float16,
      TorchDtype::BFloat16 => Dtype::BFloat16,
      TorchDtype::Int64 => Dtype::Int64,
      TorchDtype::Int32 => Dtype::Int32,
      TorchDtype::Int16 => Dtype::Int16,
      TorchDtype::Int8 => Dtype::Int8,
      TorchDtype::UInt64 => Dtype::UInt64,
      TorchDtype::UInt32 => Dtype::UInt32,
      TorchDtype::UInt16 => Dtype::UInt16,
      TorchDtype::UInt8 => Dtype::UInt8,
    })
  }
}

impl<'a> TryFrom<&'a str> for Dtype {
  type Error = String;

  fn try_from(s: &str) -> Result<Dtype, String> {
    Dtype::from_str(s)
  }
}

impl FromStr for Dtype {
  type Err = String;

  fn from_str(s: &str) -> Result<Dtype, String> {
    Ok(match s {
      "f64"     |
      "float64" => Dtype::Float64,
      "f32"     |
      "float32" => Dtype::Float32,
      "f16"     |
      "float16" => Dtype::Float16,
      "bfloat16" => Dtype::BFloat16,
      "i64"     |
      "int64"   => Dtype::Int64,
      "i32"     |
      "int32"   => Dtype::Int32,
      "i16"     |
      "int16"   => Dtype::Int16,
      "i8"      |
      "int8"    => Dtype::Int8,
      "u64"     |
      "uint64"  => Dtype::UInt64,
      "u32"     |
      "uint32"  => Dtype::UInt32,
      "u16"     |
      "uint16"  => Dtype::UInt16,
      "u8"      |
      "uint8"   => Dtype::UInt8,
      _ => return Err(s.to_owned())
    })
  }
}

impl Dtype {
  pub fn format_futhark(self) -> &'static str {
    match self {
      Dtype::_Top       => panic!("bug"),
      Dtype::Float64    => "f64",
      Dtype::Float32    => "f32",
      Dtype::Float16    => "f16",
      Dtype::BFloat16   => unimplemented!(),
      Dtype::Int64      => "i64",
      Dtype::Int32      => "i32",
      Dtype::Int16      => "i16",
      Dtype::Int8       => "i8",
      Dtype::UInt64     => "u64",
      Dtype::UInt32     => "u32",
      Dtype::UInt16     => "u16",
      Dtype::UInt8      => "u8",
    }
  }

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

  pub fn is_float(self) -> bool {
    match self {
      Dtype::Float64    |
      Dtype::Float32    |
      Dtype::Float16    |
      Dtype::BFloat16   => true,
      _ => false
    }
  }

  pub fn is_signed_int(self) -> bool {
    match self {
      Dtype::Int64      |
      Dtype::Int32      |
      Dtype::Int16      |
      Dtype::Int8       => true,
      _ => false
    }
  }

  pub fn is_unsigned_int(self) -> bool {
    match self {
      Dtype::UInt64     |
      Dtype::UInt32     |
      Dtype::UInt16     |
      Dtype::UInt8      => true,
      _ => false
    }
  }

  pub fn is_uint(self) -> bool {
    self.is_unsigned_int()
  }

  pub fn max(self, rhs: Dtype) -> Option<Dtype> {
    match (self, rhs) {
      (Dtype::_Top, _) |
      (_, Dtype::_Top) => Some(Dtype::_Top),
      (Dtype::Float32, Dtype::Float32) |
      (Dtype::Float32, Dtype::Float16) |
      (Dtype::Float16, Dtype::Float32) => Some(Dtype::Float32),
      (Dtype::Float16, Dtype::Float16) => Some(Dtype::Float16),
      _ => None
    }
  }
}

/*pub trait DtypeExt: Any {
  fn as_any(&self) -> &dyn Any { &self }
  fn _dtype(&self) -> Dtype where Self: Sized { <Self as DtypeExt>::dtype() }
  fn dtype() -> Dtype where Self: Sized;
}*/

pub trait DtypeExt {
  fn dtype() -> Dtype where Self: Sized;
  // FIXME FIXME
  //fn is_zero(&self) -> bool;
}

impl DtypeExt for TotalOrd<f32> { fn dtype() -> Dtype { Dtype::Float32 } }
impl DtypeExt for NonNan<f32>   { fn dtype() -> Dtype { Dtype::Float32 } }

impl DtypeExt for f64 { fn dtype() -> Dtype { Dtype::Float64 } }
impl DtypeExt for f32 { fn dtype() -> Dtype { Dtype::Float32 } }
impl DtypeExt for f16 { fn dtype() -> Dtype { Dtype::Float16 } }
impl DtypeExt for i64 { fn dtype() -> Dtype { Dtype::Int64 } }
impl DtypeExt for i32 { fn dtype() -> Dtype { Dtype::Int32 } }
impl DtypeExt for i16 { fn dtype() -> Dtype { Dtype::Int16 } }
impl DtypeExt for i8  { fn dtype() -> Dtype { Dtype::Int8 } }
impl DtypeExt for u64 { fn dtype() -> Dtype { Dtype::UInt64 } }
impl DtypeExt for u32 { fn dtype() -> Dtype { Dtype::UInt32 } }
impl DtypeExt for u16 { fn dtype() -> Dtype { Dtype::UInt16 } }
impl DtypeExt for u8  { fn dtype() -> Dtype { Dtype::UInt8 } }

pub fn dtype<T: DtypeExt>() -> Dtype {
  T::dtype()
}

pub fn dtype_of<T: DtypeExt>(_: T) -> Dtype {
  T::dtype()
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Dim {
  pub ndim:     u8,
  pub dtype:    Dtype,
}

impl Dim {
  pub fn ndim(&self) -> u8 {
    self.ndim
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

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct CellType {
  pub shape:    Vec<i64>,
  pub dtype:    Dtype,
}

impl CellType {
  pub fn top() -> CellType {
    CellType{
      shape:    Vec::new(),
      dtype:    Dtype::_Top,
    }
  }

  pub fn to_dim(&self) -> Dim {
    assert!(self.dtype != Dtype::_Top, "bug");
    Dim{ndim: self.ndim(), dtype: self.dtype}
  }

  pub fn ndim(&self) -> u8 {
    assert!(self.dtype != Dtype::_Top, "bug");
    assert!(self.shape.len() <= u8::max_value() as usize);
    self.shape.len() as u8
  }

  pub fn is_scalar(&self) -> bool {
    self.ndim() == 0
  }

  pub fn shape_compat(&self, orig: &CellType) -> ShapeCompat {
    assert!(self.dtype != Dtype::_Top, "bug");
    assert!(orig.dtype != Dtype::_Top, "bug");
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
    if ty.dtype == Dtype::_Top {
      let offset = 0;
      let pitch = Vec::new();
      return CellLayout{offset, pitch};
    }
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
    if ty.dtype == Dtype::_Top {
      return true;
    }
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
      p *= ds as u64;
      if self.pitch[(np - 1) - d] != p {
        return false;
      }
    }
    true
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum CellMode {
  _Top,
  Aff,
  Init,
  //Fin,
}

impl Default for CellMode {
  fn default() -> CellMode {
    CellMode::_Top
  }
}

impl CellMode {
  pub fn set_aff(&mut self) -> Result<bool, ()> {
    match *self {
      CellMode::_Top => {
        *self = CellMode::Aff;
        Ok(false)
      }
      CellMode::Aff => {
        Ok(true)
      }
      _ => {
        return Err(());
      }
    }
  }

  pub fn set_init(&mut self) -> Result<bool, ()> {
    match *self {
      CellMode::_Top => {
        *self = CellMode::Init;
        Ok(false)
      }
      CellMode::Init => {
        Ok(true)
      }
      _ => {
        return Err(());
      }
    }
  }

  pub fn set_mux(&mut self) -> Result<bool, ()> {
    self.set_init()
  }
}

#[derive(Clone, Copy, Default)]
#[repr(transparent)]
pub struct CellFlag {
  bits: u8,
}

impl CellFlag {
  pub fn reset(&mut self) {
    // FIXME FIXME
    /*self.bits &= 0xf0;*/
    self.bits = 0;
  }

  pub fn intro(&self) -> bool {
    (self.bits & 1) != 0
  }

  pub fn set_intro(&mut self) -> bool {
    let prev = self.intro();
    self.bits |= 1;
    prev
  }

  pub fn seal(&self) -> bool {
    (self.bits & 2) != 0
  }

  pub fn set_seal(&mut self) -> bool {
    let prev = self.seal();
    self.bits |= 2;
    prev
  }

  pub fn unset_seal(&mut self) -> bool {
    let prev = self.seal();
    self.bits &= !2;
    prev
  }

  pub fn cache(&self) -> bool {
    (self.bits & 4) != 0
  }

  pub fn set_cache(&mut self) -> bool {
    let prev = self.cache();
    self.bits |= 4;
    prev
  }

  /*pub fn eval(&self) -> bool {
    (self.bits & 8) != 0
  }

  pub fn set_eval(&mut self) -> bool {
    let prev = self.eval();
    self.bits |= 8;
    prev
  }*/
}

#[derive(Clone, Default)]
pub struct CellState {
  pub mode: CellMode,
  pub flag: CellFlag,
  pub clk:  Clock,
}

pub struct PCell {
  // FIXME FIXME: to implement fusion w/ unique cells, simply change
  // the PCell's owning ptr; the original ptr then becomes dangling.
  pub optr: CellPtr,
  //pub optr: Cell<CellPtr>,
  pub ogty: CellType,
  pub olay: CellLayout,
  pub primary:  Locus,
  //pub replicas: Vec<(Locus, PMach, Option<Weak<dyn InnerCell_>>)>,
  pub pm_index: RevSortMap8<(PMach, Locus), RevSortKey8<(Locus, PMach)>>,
  pub replicas: RevSortMap8<(Locus, PMach), Option<Weak<dyn InnerCell_>>>,
}

impl PCell {
  pub fn new(ptr: CellPtr, ty: CellType) -> PCell {
    let lay = CellLayout::new_packed(&ty);
    let primary = Locus::fastest();
    //let replicas = Vec::new();
    let pm_index = RevSortMap8::new();
    let replicas = RevSortMap8::new();
    PCell{
      optr: ptr,
      //optr: Cell::new(ptr),
      ogty: ty,
      olay: lay,
      primary,
      pm_index,
      replicas,
    }
  }

  pub fn push_new_replica(&mut self, locus: Locus, pmach: PMach, cel: Option<Weak<dyn InnerCell_>>) {
    /*for &(o_locus, o_pmach, _) in self.replicas.iter() {
      if (o_locus, o_pmach) == (locus, pmach) {
        panic!("bug");
      }
    }
    self.replicas.push((locus, pmach, cel));
    self.replicas.sort_by(|&(l_locus, l_pmach, _), &(r_locus, r_pmach, _)| {
      (r_locus, r_pmach).cmp(&(l_locus, l_pmach))
    });*/
    match self.replicas.find((locus, pmach)) {
      None => {}
      Some(_) => panic!("bug")
    }
    let key = self.replicas.insert((locus, pmach), cel);
    self.pm_index.insert((pmach, locus), key);
  }

  pub fn lookup_replica(&mut self, q_locus: Locus) -> Option<(PMach, &mut Option<Weak<dyn InnerCell_>>)> {
    /*for &mut (locus, pmach, ref mut cel) in self.replicas.iter_mut() {
      if locus == q_locus {
        return Some((pmach, cel));
      }
    }
    None*/
    match self.replicas.find_lub_mut((q_locus, PMach::_Bot)) {
      None => None,
      Some((key, cel)) => {
        let key = key.key.as_ref();
        if key.0 != q_locus {
          None
        } else {
          Some((key.1, cel))
        }
      }
    }
  }

  pub fn get(&mut self, q_pmach: PMach) -> Option<&mut Weak<dyn InnerCell_>> {
    /*let mut key = None;
    for &mut (locus, pmach, ref mut cel_) in self.replicas.iter_mut() {
      if pmach == q_pmach {
        match cel_ {
          &mut Some(ref mut cel_) => {
            return Some(cel_);
          }
          &mut None => {
            key = Some(locus);
          }
        }
      }
    }*/
    match self.pm_index.find_lub((q_pmach, Locus::_Bot)) {
      None => {}
      Some((_, key)) => {
        if key.key.as_ref().1 == q_pmach {
          match self.replicas.get_mut(key) {
            None => {}
            Some(cel) => {
              return Some(cel);
            }
          }
        }
      }
    }
    // FIXME FIXME: if it doesn't exist, then create it.
    unimplemented!();
  }

  pub fn hardcopy(&self) -> PCell {
    // FIXME FIXME
    unimplemented!();
  }
}

pub trait InnerCell {
  // TODO TODO
}

pub trait InnerCell_ {
  // TODO TODO
  fn as_any(&self) -> &dyn Any;
}

impl<C: InnerCell + Any> InnerCell_ for C {
  fn as_any(&self) -> &dyn Any {
    self
  }
}

/*pub struct CellSet {
  // TODO TODO
  pub ptr_: CellPtr,
}

pub struct CellMap {
  // TODO TODO
  pub ptr_: CellPtr,
}*/

pub struct MSet {
  // TODO TODO
  pub ptr_: MCellPtr,
}

pub struct MMap {
  // TODO TODO
  pub ptr_: MCellPtr,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum MValue {
  Cell(CellPtr),
  /*Atom(Atom),*/
  // FIXME FIXME: recursive MCell.
  /*MCel(MCellPtr),*/
}

#[derive(Clone, Copy)]
pub enum MValueRef<'a> {
  Cell(&'a CellPtr),
}

impl<'p, P: AsRef<CellPtr>> From<&'p P> for MValueRef<'p> {
  fn from(p: &'p P) -> MValueRef<'p> {
    MValueRef::Cell(p.as_ref())
  }
}

#[derive(Clone)]
pub struct MCellSetEntry {
  pub item: MValue,
  pub clk:  Clock,
  pub rev_: Cell<bool>,
}

#[derive(Clone)]
pub struct MCellSet {
  // FIXME
  pub idx:  HashMap<(MValue, Clock), u32>,
  pub log:  Vec<MCellSetEntry>,
}

impl Default for MCellSet {
  fn default() -> MCellSet {
    MCellSet{
      idx:  HashMap::new(),
      log:  Vec::new(),
    }
  }
}

impl MCellSet {
  pub fn add(&mut self, item: MValue, clk: Clock) {
    match self.idx.get(&(item, clk)) {
      None => {}
      Some(&idx) => {
        assert!((idx as usize) < self.log.len());
        assert!(!self.log[idx as usize].rev_.get());
        self.log[idx as usize].rev_.set(true);
      }
    }
    assert!(self.log.len() < u32::max_value() as usize);
    let idx = self.log.len() as u32;
    self.idx.insert((item, clk), idx);
    self.log.push(MCellSetEntry{item, clk, rev_: Cell::new(false)});
  }
}

#[derive(Clone)]
pub struct MCellMapEntry {
  pub key:  MValue,
  pub val:  MValue,
  pub kclk: Clock,
  pub vclk: Clock,
  pub rev_: Cell<bool>,
}

#[derive(Clone)]
pub struct MCellMap {
  // FIXME
  pub kidx: HashMap<(MValue, Clock), u32>,
  pub log:  Vec<MCellMapEntry>,
}

impl Default for MCellMap {
  fn default() -> MCellMap {
    MCellMap{
      kidx: HashMap::new(),
      log:  Vec::new(),
    }
  }
}

impl MCellMap {
  pub fn add(&mut self, key: MValue, kclk: Clock, val: MValue, vclk: Clock) {
    match self.kidx.get(&(key, kclk)) {
      None => {}
      Some(&idx) => {
        assert!((idx as usize) < self.log.len());
        assert!(!self.log[idx as usize].rev_.get());
        assert!(self.log[idx as usize].vclk <= vclk);
        self.log[idx as usize].rev_.set(true);
      }
    }
    assert!(self.log.len() < u32::max_value() as usize);
    let idx = self.log.len() as u32;
    self.kidx.insert((key, kclk), idx);
    self.log.push(MCellMapEntry{key, val, kclk, vclk, rev_: Cell::new(false)});
  }
}
