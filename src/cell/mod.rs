use crate::algo::{HashMap, HashSet, RevSortKey8, RevSortMap8};
use crate::algo::fp::*;
use crate::algo::int::*;
use crate::clock::*;
use crate::ctx::*;
use crate::nd::{IRange};
use crate::panick::*;
use crate::pctx::{TL_PCTX, Locus, PMach, PAddr, MemReg, InnerCell_};
use crate::thunk::*;
use crate::thunk::op::{SetScalarFutThunkSpec};
use crate::util::mmap::{MmapFileSlice};
use crate::util::pickle::{TorchDtype};
use crate::util::safetensor::{TensorDtype};
use cacti_cfg_env::*;

use futhark_ffi::{AbiScalar as FutAbiScalar};
use rustc_serialize::*;
use smol_str::{SmolStr};

use std::borrow::{Borrow};
use std::cell::{Cell, RefCell};
use std::cmp::{Ordering};
use std::convert::{TryFrom, TryInto};
use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::hash::{Hash, Hasher};
use std::mem::{align_of, size_of, swap};
use std::ops::{Deref, Neg};
use std::rc::{Rc, Weak};
use std::slice::{from_raw_parts, from_raw_parts_mut};
use std::str::{FromStr};

//#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct CellPtr {
  pub raw_: i64,
}

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

impl CellDeref for CellPtr {
  fn _deref(&self) -> CellPtr {
    *self
  }
}

impl<'a> CellDeref for &'a CellPtr {
  fn _deref(&self) -> CellPtr {
    **self
  }
}

/*impl<T: CellStoreTo + ?Sized> CellStore<T> for CellPtr {
  fn _store(&self, src: &T) {
    src._store_to(*self);
  }
}*/

impl Debug for CellPtr {
  fn fmt(&self, f: &mut Formatter) -> FmtResult {
    write!(f, "CellPtr({})", self.raw_)
  }
}

impl CellPtr {
  pub fn nil() -> CellPtr {
    CellPtr{raw_: 0}
  }

  pub fn from_unchecked(raw_: i64) -> CellPtr {
    CellPtr{raw_}
  }

  pub fn to_unchecked(&self) -> i64 {
    self.raw_
  }

  pub fn is_nil(&self) -> bool {
    self.raw_ == 0
  }

  /*pub fn as_bytes_repr(&self) -> &[u8] {
    // SAFETY: This should be safe as the type is `Copy`.
    let ptr = (self as *const CellPtr) as *const u8;
    let len = size_of::<CellPtr>();
    assert_eq!(len, 8);
    unsafe { from_raw_parts(ptr, len) }
  }*/

  pub fn _into_mcel_ptr(self) -> MCellPtr {
    MCellPtr{raw_: self.raw_}
  }
}

impl<R: ?Sized + CellDeref> PartialEq<R> for CellPtr {
  fn eq(&self, rhs: &R) -> bool {
    self.raw_.eq(&rhs._deref().raw_)
  }
}

impl Eq for CellPtr {}

impl<R: ?Sized + CellDeref> PartialOrd<R> for CellPtr {
  fn partial_cmp(&self, rhs: &R) -> Option<Ordering> {
    Some(self.raw_.cmp(&rhs._deref().raw_))
  }
}

impl Ord for CellPtr {
  fn cmp(&self, rhs: &CellPtr) -> Ordering {
    self.partial_cmp(rhs).unwrap()
  }
}

impl Hash for CellPtr {
  fn hash<H: Hasher>(&self, hasher: &mut H) {
    self.raw_.hash(hasher);
  }
}

//#[derive(PartialEq, Eq, PartialOrd, Ord, Hash)]
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
    StableCell::set_scalar(value)
  }
}

impl<'a> Borrow<CellPtr> for &'a StableCell {
  fn borrow(&self) -> &CellPtr {
    self.as_ptr_ref()
  }
}

impl Borrow<CellPtr> for StableCell {
  fn borrow(&self) -> &CellPtr {
    self.as_ptr_ref()
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

impl CellDeref for StableCell {
  fn _deref(&self) -> CellPtr {
    *self.as_ptr_ref()
  }
}

impl<'a> CellDeref for &'a StableCell {
  fn _deref(&self) -> CellPtr {
    *self.as_ptr_ref()
  }
}

/*impl<'a, T: CellStoreTo + ?Sized> CellStore<T> for &'a StableCell {
  fn _store(&self, src: &T) {
    src._store_to(*self.as_ptr_ref());
  }
}

impl<T: CellStoreTo + ?Sized> CellStore<T> for StableCell {
  fn _store(&self, src: &T) {
    src._store_to(*self.as_ptr_ref());
  }
}*/

impl Debug for StableCell {
  fn fmt(&self, f: &mut Formatter) -> FmtResult {
    write!(f, "StableCell({})", self.ptr_.raw_)
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

impl From<CellType> for StableCell {
  fn from(ty: CellType) -> StableCell {
    StableCell::array(ty.shape, ty.dtype)
  }
}

impl StableCell {
  pub fn _retain(env: &CtxEnv, ptr: CellPtr) -> StableCell {
    assert!(!ptr.is_nil());
    env._retain_probe(ptr);
    StableCell{ptr_: ptr}
  }

  pub fn top() -> StableCell {
    let ty = CellType::top();
    TL_CTX.with(|ctx| {
      let x = ctx.ctr.fresh_cel();
      ctx.env.borrow_mut().insert_top(x, ty);
      x.into()
    })
  }

  pub fn scalar<D: TryInto<Dtype>>(dtype: D) -> StableCell {
    let dtype: Dtype = match dtype.try_into() {
      Ok(d) => d,
      Err(_) => panic!("bug: StableCell::scalar: invalid dtype")
    };
    let ty = CellType{shape: Box::new([]), dtype};
    TL_CTX.with(|ctx| {
      let x = ctx.ctr.fresh_cel();
      ctx.env.borrow_mut().insert_top(x, ty);
      x.into()
    })
  }

  pub fn set_scalar<T: IntoScalarValExt>(value: T) -> StableCell {
    ctx_pop_thunk(SetScalarFutThunkSpec{val: value.into_scalar_val_()}).into()
  }

  pub fn array<S: Into<Box<[i64]>>, D: TryInto<Dtype>>(shape: S, dtype: D) -> StableCell {
    let shape: Box<[i64]> = shape.into();
    let dtype: Dtype = match dtype.try_into() {
      Ok(d) => d,
      Err(_) => panic!("bug: StableCell::array: invalid dtype")
    };
    let ty = CellType{shape, dtype};
    TL_CTX.with(|ctx| {
      let x = ctx.ctr.fresh_cel();
      ctx.env.borrow_mut().insert_top(x, ty);
      x.into()
    })
  }

  pub fn _release(mut self, env: &CtxEnv) -> CellPtr {
    let mut ptr = CellPtr::nil();
    swap(&mut ptr, &mut self.ptr_);
    env._release_probe(ptr);
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
    unsafe { &*((self as *const StableCell) as *const CellPtr) }
  }

  #[inline]
  pub fn as_ptr_mut(&mut self) -> &mut CellPtr {
    // SAFETY: The following is safe as `StableCell` has the same
    // (transparent) repr as `CellPtr`.
    unsafe { &mut *((self as *mut StableCell) as *mut CellPtr) }
  }
}

impl<R: ?Sized + CellDeref> PartialEq<R> for StableCell {
  fn eq(&self, rhs: &R) -> bool {
    self.ptr_.eq(&rhs._deref())
  }
}

impl Eq for StableCell {}

impl<R: ?Sized + CellDeref> PartialOrd<R> for StableCell {
  fn partial_cmp(&self, rhs: &R) -> Option<Ordering> {
    Some(self.ptr_.cmp(&rhs._deref()))
  }
}

impl Ord for StableCell {
  fn cmp(&self, rhs: &StableCell) -> Ordering {
    self.partial_cmp(rhs).unwrap()
  }
}

impl Hash for StableCell {
  fn hash<H: Hasher>(&self, hasher: &mut H) {
    self.ptr_.hash(hasher);
  }
}

/*pub struct Snapshot {
  pub ptr_: CellPtr,
  pub clk:  Clock,
}

pub struct StableSnapshot {
}

pub struct Checkpoint {
}

pub type StableCheckpoint = Checkpoint;*/

/*#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Atom {
  pub raw_: i64,
}

impl Atom {
  pub fn nil() -> Atom {
    Atom{raw_: 0}
  }

  pub fn from_unchecked(raw_: i64) -> Atom {
    Atom{raw_}
  }

  pub fn to_unchecked(&self) -> i64 {
    self.raw_
  }

  pub fn is_nil(&self) -> bool {
    self.raw_ == 0
  }
}*/

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct MCellPtr {
  pub raw_: i64,
}

impl Debug for MCellPtr {
  fn fmt(&self, f: &mut Formatter) -> FmtResult {
    write!(f, "MCellPtr({})", self.raw_)
  }
}

impl MCellPtr {
  pub fn nil() -> MCellPtr {
    MCellPtr{raw_: 0}
  }

  pub fn from_unchecked(raw_: i64) -> MCellPtr {
    MCellPtr{raw_}
  }

  pub fn to_unchecked(&self) -> i64 {
    self.raw_
  }

  pub fn is_nil(&self) -> bool {
    self.raw_ == 0
  }

  pub fn _into_cel_ptr(self) -> CellPtr {
    CellPtr{raw_: self.raw_}
  }
}

#[repr(transparent)]
pub struct CellSet {
  pub ptr_: MCellPtr,
}

impl Debug for CellSet {
  fn fmt(&self, f: &mut Formatter) -> FmtResult {
    write!(f, "CellSet({})", self.ptr_.raw_)
  }
}

impl Borrow<MCellPtr> for CellSet {
  fn borrow(&self) -> &MCellPtr {
    &self.ptr_
  }
}

impl Clone for CellSet {
  fn clone(&self) -> CellSet {
    CellSet::from(self.ptr_)
  }
}

/*impl Drop for CellSet {
  fn drop(&mut self) {
    //ctx_release(self.ptr_._into_cel_ptr());
  }
}*/

impl From<MCellPtr> for CellSet {
  fn from(ptr: MCellPtr) -> CellSet {
    //ctx_retain(ptr._into_cel_ptr());
    CellSet{ptr_: ptr}
  }
}

impl CellSet {
  #[track_caller]
  pub fn new() -> CellSet {
    panick_wrap(|| TL_CTX.with(|ctx| {
      ctx.ctr.fresh_mcel().into()
    }))
  }

  pub fn as_ptr(&self) -> MCellPtr {
    self.ptr_
  }

  #[track_caller]
  pub fn add<X: CellDeref>(&self, x: X) {
    panick_wrap(|| TL_CTX.with(|ctx| {
      let spine = ctx.spine.borrow();
      spine.add(self.as_ptr(), x._deref());
    }))
  }

  #[track_caller]
  pub fn contains<X: CellDeref>(&self, x: X) -> bool {
    panick_wrap(|| TL_CTX.with(|ctx| {
      let spine = ctx.spine.borrow();
      spine._contains(self.as_ptr(), x._deref())
    }))
  }
}

#[repr(transparent)]
pub struct CellMap {
  pub ptr_: MCellPtr,
}

impl Debug for CellMap {
  fn fmt(&self, f: &mut Formatter) -> FmtResult {
    write!(f, "CellMap({})", self.ptr_.raw_)
  }
}

impl Borrow<MCellPtr> for CellMap {
  fn borrow(&self) -> &MCellPtr {
    &self.ptr_
  }
}

impl Clone for CellMap {
  fn clone(&self) -> CellMap {
    CellMap::from(self.ptr_)
  }
}

/*impl Drop for CellMap {
  fn drop(&mut self) {
    //ctx_release(self.ptr_._into_cel_ptr());
  }
}*/

impl From<MCellPtr> for CellMap {
  fn from(ptr: MCellPtr) -> CellMap {
    //ctx_retain(ptr._into_cel_ptr());
    CellMap{ptr_: ptr}
  }
}

impl CellMap {
  #[track_caller]
  pub fn new() -> CellMap {
    panick_wrap(|| TL_CTX.with(|ctx| {
      ctx.ctr.fresh_mcel().into()
    }))
  }

  pub fn as_ptr(&self) -> MCellPtr {
    self.ptr_
  }

  #[track_caller]
  pub fn add<K: CellDeref, V: CellDeref>(&self, k: K, v: V) {
    panick_wrap(|| TL_CTX.with(|ctx| {
      let spine = ctx.spine.borrow();
      spine.add2(self.as_ptr(), k._deref(), v._deref());
    }))
  }

  /*#[track_caller]
  pub fn vadd<K: CellDeref, V: CellDeref>(&self, k: &[K], v: &[V]) {
    panick_wrap(|| TL_CTX.with(|ctx| {
      let spine = ctx.spine.borrow();
      for (k, v) in k.iter().zip(v.iter()) {
        spine.add2(self.ptr_, k._deref(), v._deref());
      }
    }))
  }*/

  #[track_caller]
  pub fn get<K: CellDeref>(&self, key: K) -> CellPtr {
    panick_wrap(|| TL_CTX.with(|ctx| {
      let spine = ctx.spine.borrow();
      let key = key._deref();
      let kclk = spine._version(key).unwrap_or_else(|| Clock::default());
      spine._get(self.as_ptr(), key, kclk).map(|(v, _)| v).unwrap_or_else(|| CellPtr::nil())
    }))
  }

  #[track_caller]
  pub fn maybe_get<K: CellDeref>(&self, key: K) -> Option<CellPtr> {
    let v = self.get(key);
    if v.is_nil() {
      None
    } else {
      Some(v)
    }
  }

  /*#[track_caller]
  pub fn vget(&self, vkey: &[CellPtr]) -> Vec<CellPtr> {
    panick_wrap(|| TL_CTX.with(|ctx| {
      let mut vval = Vec::with_capacity(vkey.len());
      let spine = ctx.spine.borrow();
      for &key in vkey.iter() {
        let kclk = spine._version(key).unwrap_or_else(|| Clock::default());
        let v = spine._get(self.as_ptr(), key, kclk).map(|(v, _)| v).unwrap_or_else(|| CellPtr::nil());
        vval.push(v);
      }
      vval
    }))
  }*/
}

enum _Void {}

pub type CellViewHandle = CellViewHandle_;

#[repr(transparent)]
pub struct CellViewHandle_([_Void]);

/*impl Debug for CellViewHandle_ {
  fn fmt(&self, f: &mut Formatter) -> FmtResult {
    write!(f, "CellViewHandle({})", self._deref().raw_)
  }
}*/

impl CellViewHandle_ {
  pub fn _from<'a>(this: &'a CellPtr) -> &'a CellViewHandle_ {
    let ptr = { this.raw_ as _ };
    unsafe { &*(from_raw_parts(this as *const CellPtr as *const _Void, ptr) as *const [_Void] as *const CellViewHandle_) }
  }

  pub fn _from2<'a>(this: &'a CellPtr, other: CellPtr) -> &'a CellViewHandle_ {
    let ptr = { other.raw_ as _ };
    unsafe { &*(from_raw_parts(this as *const CellPtr as *const _Void, ptr) as *const [_Void] as *const CellViewHandle_) }
  }

  pub fn _from_mut<'a>(this: &'a mut CellPtr) -> &'a mut CellViewHandle_ {
    let ptr = { this.raw_ as _ };
    unsafe { &mut *(from_raw_parts_mut(this as *mut CellPtr as *mut _Void, ptr) as *mut [_Void] as *mut CellViewHandle_) }
  }

  pub fn _from2_mut<'a>(this: &'a mut CellPtr, other: CellPtr) -> &'a mut CellViewHandle_ {
    let ptr = { other.raw_ as _ };
    unsafe { &mut *(from_raw_parts_mut(this as *mut CellPtr as *mut _Void, ptr) as *mut [_Void] as *mut CellViewHandle_) }
  }
}

impl CellDeref for CellViewHandle_ {
  fn _deref(&self) -> CellPtr {
    CellPtr::from_unchecked(self.0.len() as _)
  }
}

impl<'a> CellDeref for &'a CellViewHandle_ {
  fn _deref(&self) -> CellPtr {
    CellPtr::from_unchecked(self.0.len() as _)
  }
}

impl<'a> CellDeref for &'a mut CellViewHandle_ {
  fn _deref(&self) -> CellPtr {
    CellPtr::from_unchecked(self.0.len() as _)
  }
}

pub trait CellDeref {
  fn _deref(&self) -> CellPtr;
}

/*pub trait CellStore<T: ?Sized> {
  fn _store(&self, src: &T);
}*/

pub trait CellStoreTo {
  fn _store_to(&self, dst: CellPtr, clk: Clock, env: &CtxEnv);
}

/*impl<'a> CellStoreTo for &'a [u8] {
  fn _store_to(&self, dst: CellPtr, clk: Clock, env: &CtxEnv) {
    unimplemented!();
  }
}*/

impl CellStoreTo for MmapFileSlice {
  fn _store_to(&self, dst: CellPtr, clk: Clock, env: &CtxEnv) {
    assert!(!dst.is_nil());
    match env._lookup_view(dst) {
      Err(_) => panic!("bug"),
      Ok(e) => {
        let root = e.root();
        let v_ty = match e.view().eval_contiguous(e.root_ty) {
          Err(_) => {
            println!("ERROR:  MmapFileSlice: dst={:?} is not a zero-copy (contiguous) view", dst);
            panic!();
          }
          Ok(ty) => ty
        };
        assert_eq!(e.ty, v_ty.as_ref());
        TL_PCTX.with(|pctx| {
          let mut cel_ = e.cel_.borrow_mut();
          match &mut *cel_ {
            &mut Cell_::Top(ref state, optr) => {
              assert_eq!(root, optr);
              let oclk = state.clk;
              assert_eq!(clk, oclk);
              if e.root_ty.span_bytes() != e.ty.span_bytes() {
                // FIXME: view: need to do a raw zero-pad.
                unimplemented!();
              } else {
                let addr = pctx.ctr.fresh_addr();
                let icel = pctx.swap._try_alloc_cow(addr, &e.ty, self.clone()).unwrap();
                let state = state.clone();
                let clo = CellClosure::default();
                let mut pcel = PCell::new(optr, e.root_ty.clone());
                pcel.push(optr, oclk, Locus::Mem, PMach::Swap, addr);
                *cel_ = Cell_::Phy(state, clo, pcel);
              }
            }
            &mut Cell_::Phy(ref state, .., ref mut pcel) => {
              let optr = pcel.optr;
              assert_eq!(root, optr);
              let oclk = state.clk;
              assert_eq!(clk, oclk);
              if e.root_ty.span_bytes() != e.ty.span_bytes() {
                // FIXME: view.
                unimplemented!();
              } else {
                let addr = pctx.ctr.fresh_addr();
                let icel = pctx.swap._try_alloc_cow(addr, &e.ty, self.clone()).unwrap();
                match pcel.swap(optr, oclk, Locus::Mem, PMach::Swap, addr) {
                  None => {}
                  Some((_, prev_addr)) => {
                    if prev_addr != addr {
                      let _ = pctx.release(prev_addr);
                    }
                  }
                }
              }
            }
            _ => panic!("bug")
          }
        });
      }
    }
  }
}

pub type CellVOp = CellViewOp;

#[derive(Clone, Debug)]
pub enum CellViewOp {
  Nop,
  //Slice(Rc<IRange>),
  //Slice2(Rc<[IRange; 2]>),
  //Slice3(Rc<[IRange; 3]>),
  //Slice4(Rc<[IRange; 4]>),
  Slice(Rc<[IRange]>),
  Proj(u8),
  Swap(i8, i8),
  Transpose(Rc<[i8]>),
  //NewShape(Vec<i64>),
  NewShape(Rc<[i64]>),
  BitAlias(Dtype),
}

impl CellViewOp {
  pub fn nop() -> CellViewOp {
    CellViewOp::Nop
  }

  pub fn slice(idx: &[IRange]) -> CellViewOp {
    CellViewOp::Slice(idx.into())
  }

  pub fn proj(mask: &[bool]) -> CellViewOp {
    assert!(mask.len() <= 8);
    let mut maskbits = 0;
    for d in 0 .. mask.len() {
      if mask[d] {
        maskbits |= (1 << d);
      }
    }
    CellViewOp::Proj(maskbits)
  }

  pub fn swap(ld: i8, rd: i8) -> CellViewOp {
    CellViewOp::Swap(ld, rd)
  }

  pub fn transpose(perm: &[i8]) -> CellViewOp {
    CellViewOp::Transpose(perm.into())
  }

  pub fn new_shape(shape: &[i64]) -> CellViewOp {
    CellViewOp::NewShape(shape.into())
  }

  pub fn bit_alias(dtype: Dtype) -> CellViewOp {
    CellViewOp::BitAlias(dtype)
  }
}

#[derive(Clone, Debug)]
pub struct CellView {
  // TODO
  pub root: CellPtr,
  pub vlog: Vec<CellViewOp>,
}

impl Default for CellView {
  fn default() -> CellView {
    CellView{
      root: CellPtr::nil(),
      vlog: Vec::new(),
    }
  }
}

impl From<CellPtr> for CellView {
  fn from(root: CellPtr) -> CellView {
    CellView{
      root,
      vlog: Vec::new(),
    }
  }
}

impl CellView {
  pub fn root(&self) -> CellPtr {
    self.root
  }

  pub fn type_eval(&self, root_ty: &CellType) -> Result<CellType, ()> {
    assert!(!root_ty.is_top());
    let mut shape = root_ty.shape.to_vec();
    let mut dtype = root_ty.dtype;
    if self.vlog.is_empty() {
      let shape = shape.into();
      return Ok(CellType{shape, dtype});
    }
    let mut offset = Vec::with_capacity(root_ty.shape.len());
    offset.resize(root_ty.shape.len(), 0);
    let mut end_offset = root_ty.shape.to_vec();
    let mut root_shape = root_ty.shape.to_vec();
    for vop in self.vlog.iter() {
      match vop {
        &CellViewOp::Nop => {}
        &CellViewOp::Slice(ref idx) => {
          for d in (0 .. shape.len()).rev() {
            shape[d] = idx[d].end - idx[d].start;
            offset[d] += idx[d].start;
            end_offset[d] = offset[d] + (idx[d].end - idx[d].start);
          }
        }
        &CellViewOp::Proj(maskbits) => {
          //let mut new_rank = 0;
          let mut new_shape = Vec::new();
          for d in 0 .. shape.len() {
            if ((maskbits >> d) & 1) != 0 {
              if shape[d] != 1 {
                return Err(());
              }
            } else {
              //new_rank += 1;
              new_shape.push(shape[d]);
            }
          }
          //let mut new_shape = Vec::with_capacity(new_rank);
          // FIXME
          unimplemented!();
        }
        &CellViewOp::NewShape(ref new_shape) => {
          for d in 0 .. shape.len() {
            if offset[d] == 0 && end_offset[d] == shape[d] {
              continue;
            }
            return Err(());
          }
          match (CellType::_shape_compat(&**new_shape, &root_shape),
                 CellType::_shape_compat(&**new_shape, &shape)) {
            (ShapeCompat::Equal, ShapeCompat::Equal) |
            (ShapeCompat::Equal, ShapeCompat::NewShape) |
            (ShapeCompat::NewShape, ShapeCompat::Equal) |
            (ShapeCompat::NewShape, ShapeCompat::NewShape) => {
              offset.clear();
              offset.resize(new_shape.len(), 0);
              shape.clear();
              shape.extend_from_slice(new_shape);
              end_offset.clear();
              end_offset.extend_from_slice(new_shape);
              root_shape.clear();
              root_shape.extend_from_slice(new_shape);
            }
            _ => return Err(())
          }
        }
        &CellViewOp::BitAlias(new_dtype) => {
          if dtype.size_bits() != new_dtype.size_bits() {
            return Err(());
          }
          dtype = new_dtype;
        }
        _ => unimplemented!()
      }
    }
    let shape = shape.into();
    Ok(CellType{shape, dtype})
  }

  pub fn eval_contiguous(&self, root_ty: &CellType) -> Result<CellSliceType, ()> {
    assert!(!root_ty.is_top());
    let mut offset = Vec::with_capacity(root_ty.shape.len());
    offset.resize(root_ty.shape.len(), 0);
    let mut shape = root_ty.shape.to_vec();
    let mut dtype = root_ty.dtype;
    if self.vlog.is_empty() {
      let shape = shape.into();
      return Ok(CellSliceType{offset, type_: CellType{shape, dtype}});
    }
    let mut end_offset = root_ty.shape.to_vec();
    let mut root_shape = root_ty.shape.to_vec();
    for vop in self.vlog.iter() {
      loop {
        match vop {
          &CellViewOp::Nop => {
            break;
          }
          &CellViewOp::Slice(ref idx) => {
            let mut outer = None;
            for d in 0 .. shape.len() {
              if idx[d].start + 1 == idx[d].end {
                continue;
              } else if outer.is_none() && idx[d].start < idx[d].end {
                outer = Some(d);
                continue;
              } else if outer.is_some() && idx[d].start == 0 && idx[d].end == shape[d] {
                continue;
              }
              return Err(());
            }
            // FIXME: asserts below are kinda ad hoc.
            let outer = outer.unwrap_or(shape.len());
            for d in (0 .. shape.len()).rev() {
              if !(idx[d].end <= shape[d] + idx[d].start) {
                println!("DEBUG: CellView::eval_contiguous: Slice: root ty={:?}", root_ty);
                println!("DEBUG: CellView::eval_contiguous: Slice: vlog   ={:?}", &self.vlog);
                println!("DEBUG: CellView::eval_contiguous: Slice: offset ={:?}", &offset);
                println!("DEBUG: CellView::eval_contiguous: Slice: shape  ={:?}", &shape);
                println!("DEBUG: CellView::eval_contiguous: Slice: eoffset={:?}", &end_offset);
                println!("DEBUG: CellView::eval_contiguous: Slice: rshape ={:?}", &root_shape);
                println!("DEBUG: CellView::eval_contiguous: Slice: idx    ={:?}", idx);
                println!("DEBUG: CellView::eval_contiguous: Slice: outer  ={:?}", outer);
              }
              assert!(idx[d].start <= idx[d].end);
              assert!(idx[d].end <= shape[d] + idx[d].start);
              assert!(offset[d] + idx[d].start <= idx[d].end);
              if d <= outer {
                shape[d] = idx[d].end - idx[d].start;
                offset[d] += idx[d].start;
                end_offset[d] = offset[d] + (idx[d].end - idx[d].start);
              } else {
                assert_eq!(offset[d], 0);
              }
            }
            break;
          }
          &CellViewOp::NewShape(ref new_shape) => {
            for d in 0 .. shape.len() {
              if offset[d] == 0 && end_offset[d] == shape[d] {
                continue;
              }
              return Err(());
            }
            match (CellType::_shape_compat(&**new_shape, &root_shape),
                   CellType::_shape_compat(&**new_shape, &shape)) {
              (ShapeCompat::Equal, ShapeCompat::Equal) |
              (ShapeCompat::Equal, ShapeCompat::NewShape) |
              (ShapeCompat::NewShape, ShapeCompat::Equal) |
              (ShapeCompat::NewShape, ShapeCompat::NewShape) => {
                offset.clear();
                offset.resize(new_shape.len(), 0);
                shape.clear();
                shape.extend_from_slice(new_shape);
                end_offset.clear();
                end_offset.extend_from_slice(new_shape);
                root_shape.clear();
                root_shape.extend_from_slice(new_shape);
              }
              _ => return Err(())
            }
            break;
          }
          &CellViewOp::BitAlias(new_dtype) => {
            //assert_eq!(dtype.size_bytes(), new_dtype.size_bytes());
            if dtype.size_bits() != new_dtype.size_bits() {
              return Err(());
            }
            dtype = new_dtype;
            break;
          }
          _ => unimplemented!()
        }
      }
    }
    // NB: check contiguous wrt root.
    let mut flat_len = 1;
    let mut start = 0;
    let mut fin = 0;
    let mut stride = 1;
    for d in (0 .. root_shape.len()).rev() {
      let ds = shape[d];
      let o = offset[d];
      let o2 = end_offset[d];
      let root_ds = root_shape[d];
      assert!(ds >= 0);
      assert!(o >= 0);
      assert!(o2 >= 0);
      assert!(root_ds >= 0);
      assert!(o <= o2);
      if !(o2 <= root_ds) {
        // NB: this failure may indicate a user indexing bug.
        println!("DEBUG: CellView::eval_contiguous:   root ty={:?}", root_ty);
        println!("DEBUG: CellView::eval_contiguous:   rshape ={:?}", root_shape);
        println!("DEBUG: CellView::eval_contiguous:   offset ={:?}", offset);
        println!("DEBUG: CellView::eval_contiguous:   shape  ={:?}", shape);
        println!("DEBUG: CellView::eval_contiguous:   eoffset={:?}", end_offset);
        println!("DEBUG: CellView::eval_contiguous:   dtype  ={:?}", dtype);
        println!("DEBUG: CellView::eval_contiguous:   flat   ={:?}", flat_len);
        println!("DEBUG: CellView::eval_contiguous:   start  ={:?}", start);
        println!("DEBUG: CellView::eval_contiguous:   fin    ={:?}", fin);
        println!("DEBUG: CellView::eval_contiguous:   d      ={:?}", d);
        println!("DEBUG: CellView::eval_contiguous:   ds     ={:?}", ds);
        println!("DEBUG: CellView::eval_contiguous:   o      ={:?}", o);
        println!("DEBUG: CellView::eval_contiguous:   o2     ={:?}", o2);
        println!("DEBUG: CellView::eval_contiguous:   root_ds={:?}", root_ds);
      }
      assert!(o2 <= root_ds);
      assert!(ds <= root_ds);
      flat_len *= ds as u64;
      start += (o as u64) * stride;
      fin += ((o2 - 1) as u64) * stride;
      stride *= root_ds as u64;
    }
    if !(start + flat_len == fin + 1) {
      println!("BUG:   CellView::eval_contiguous: unexpected non-contiguous view:");
      println!("DEBUG: CellView::eval_contiguous:   root ty={:?}", root_ty);
      println!("DEBUG: CellView::eval_contiguous:   rshape ={:?}", root_shape);
      println!("DEBUG: CellView::eval_contiguous:   offset ={:?}", offset);
      println!("DEBUG: CellView::eval_contiguous:   shape  ={:?}", shape);
      println!("DEBUG: CellView::eval_contiguous:   eoffset={:?}", end_offset);
      println!("DEBUG: CellView::eval_contiguous:   dtype  ={:?}", dtype);
      println!("DEBUG: CellView::eval_contiguous:   flat   ={:?}", flat_len);
      println!("DEBUG: CellView::eval_contiguous:   start  ={:?}", start);
      println!("DEBUG: CellView::eval_contiguous:   fin    ={:?}", fin);
    }
    assert_eq!(start + flat_len, fin + 1);
    let shape = shape.into();
    Ok(CellSliceType{offset, type_: CellType{shape, dtype}})
  }

  pub fn eval_contiguous_transposed(&self, root_ty: &CellType) -> Result<(CellSliceType, CellIndexPerm), ()> {
    assert!(!root_ty.is_top());
    assert!(root_ty.shape.len() <= 7);
    let nd = root_ty.shape.len() as i8;
    let mut offset = Vec::with_capacity(nd as usize);
    offset.resize(nd as usize, 0);
    let mut shape = root_ty.shape.to_vec();
    let mut dtype = root_ty.dtype;
    if self.vlog.is_empty() {
      let shape = shape.into();
      return Ok((CellSliceType{offset, type_: CellType{shape, dtype}}, CellIndexPerm::new(nd)));
    }
    let mut end_offset = root_ty.shape.to_vec();
    let mut root_shape = root_ty.shape.to_vec();
    let mut seal_slice = false;
    /*let mut idx_perm = Vec::with_capacity(nd as usize);
    for d in 0 .. nd {
      idx_perm.push(d);
    }*/
    let mut perm_state = CellIndexPermState::new(nd);
    // TODO
    for vop in self.vlog.iter() {
      match vop {
        &CellViewOp::Nop => {}
        &CellViewOp::Slice(ref idx) => {
          assert!(!seal_slice);
          let mut outer = None;
          for d in 0 .. shape.len() {
            if idx[d].start + 1 == idx[d].end {
              continue;
            } else if outer.is_none() && idx[d].start < idx[d].end {
              outer = Some(d);
              continue;
            } else if outer.is_some() && idx[d].start == 0 && idx[d].end == shape[d] {
              continue;
            }
            return Err(());
          }
          // FIXME: asserts below are kinda ad hoc.
          let outer = outer.unwrap_or(shape.len());
          for d in (0 .. shape.len()).rev() {
            if !(idx[d].end <= shape[d] + idx[d].start) {
              println!("DEBUG: CellView::eval_contiguous: Slice: root ty={:?}", root_ty);
              println!("DEBUG: CellView::eval_contiguous: Slice: vlog   ={:?}", &self.vlog);
              println!("DEBUG: CellView::eval_contiguous: Slice: offset ={:?}", &offset);
              println!("DEBUG: CellView::eval_contiguous: Slice: shape  ={:?}", &shape);
              println!("DEBUG: CellView::eval_contiguous: Slice: eoffset={:?}", &end_offset);
              println!("DEBUG: CellView::eval_contiguous: Slice: rshape ={:?}", &root_shape);
              println!("DEBUG: CellView::eval_contiguous: Slice: idx    ={:?}", idx);
              println!("DEBUG: CellView::eval_contiguous: Slice: outer  ={:?}", outer);
            }
            assert!(idx[d].start <= idx[d].end);
            assert!(idx[d].end <= shape[d] + idx[d].start);
            assert!(offset[d] + idx[d].start <= idx[d].end);
            if d <= outer {
              shape[d] = idx[d].end - idx[d].start;
              offset[d] += idx[d].start;
              end_offset[d] = offset[d] + (idx[d].end - idx[d].start);
            } else {
              assert_eq!(offset[d], 0);
            }
          }
        }
        &CellViewOp::NewShape(ref new_shape) => {
          assert!(!seal_slice);
          for d in 0 .. shape.len() {
            if offset[d] == 0 && end_offset[d] == shape[d] {
              continue;
            }
            return Err(());
          }
          match (CellType::_shape_compat(&**new_shape, &root_shape),
                 CellType::_shape_compat(&**new_shape, &shape)) {
            (ShapeCompat::Equal, ShapeCompat::Equal) |
            (ShapeCompat::Equal, ShapeCompat::NewShape) |
            (ShapeCompat::NewShape, ShapeCompat::Equal) |
            (ShapeCompat::NewShape, ShapeCompat::NewShape) => {
              offset.clear();
              offset.resize(new_shape.len(), 0);
              shape.clear();
              shape.extend_from_slice(new_shape);
              end_offset.clear();
              end_offset.extend_from_slice(new_shape);
              root_shape.clear();
              root_shape.extend_from_slice(new_shape);
            }
            _ => return Err(())
          }
        }
        &CellViewOp::BitAlias(new_dtype) => {
          //assert_eq!(dtype.size_bytes(), new_dtype.size_bytes());
          if dtype.size_bits() != new_dtype.size_bits() {
            return Err(());
          }
          dtype = new_dtype;
        }
        &CellViewOp::Swap(_ld, _rd) => {
          seal_slice = true;
          perm_state._step(vop);
        }
        _ => unimplemented!()
      }
    }
    // NB: check contiguous wrt root.
    let mut flat_len = 1;
    let mut start = 0;
    let mut fin = 0;
    let mut stride = 1;
    for d in (0 .. root_shape.len()).rev() {
      let ds = shape[d];
      let o = offset[d];
      let o2 = end_offset[d];
      let root_ds = root_shape[d];
      assert!(ds >= 0);
      assert!(o >= 0);
      assert!(o2 >= 0);
      assert!(root_ds >= 0);
      assert!(o <= o2);
      assert!(o2 <= root_ds);
      assert!(ds <= root_ds);
      flat_len *= ds as u64;
      start += (o as u64) * stride;
      fin += ((o2 - 1) as u64) * stride;
      stride *= root_ds as u64;
    }
    assert_eq!(start + flat_len, fin + 1);
    let shape = shape.into();
    Ok((CellSliceType{offset, type_: CellType{shape, dtype}}, CellIndexPerm{perm: perm_state.perm, ndim: nd}))
  }

  pub fn eval_strided(&self, root_ty: &CellType) -> Result<CellStridedSliceType, ()> {
    assert!(!root_ty.is_top());
    unimplemented!();
  }

  pub fn eval_strided_transposed(&self, root_ty: &CellType) -> Result<(CellStridedSliceType, CellIndexPerm), ()> {
    assert!(!root_ty.is_top());
    unimplemented!();
  }
}

#[derive(Clone, Copy, Debug)]
pub struct CellIndexPerm {
  pub perm: [i8; 7],
  pub ndim: i8,
}

impl CellIndexPerm {
  pub fn new(ndim: i8) -> CellIndexPerm {
    CellIndexPerm{
      perm: [0, 1, 2, 3, 4, 5, 6],
      ndim,
    }
  }
}

/*pub type CellViewStep = CellIndexPermStep;

#[derive(Clone, Copy, Debug)]
#[repr(u8)]
pub enum CellIndexPermStep {
  Break,
  Swap,
  // TODO
  Halt,
}*/

pub type CellViewPermState = CellIndexPermState;

#[derive(Clone, Copy, Debug)]
pub struct CellIndexPermState {
  pub perm: [i8; 7],
  pub ndim: i8,
  //pub swap: bool,
}

impl CellIndexPermState {
  pub fn new(ndim: i8) -> CellIndexPermState {
    assert!(ndim >= 0);
    assert!(ndim <= 7);
    CellIndexPermState{
      perm: [0, 1, 2, 3, 4, 5, 6],
      ndim: ndim,
      //swap: false,
    }
  }

  /*pub fn _reset_swap(&mut self) {
    self.swap = false;
    self.perm = [0, 1, 2, 3, 4, 5, 6, 7];
  }*/

  /*pub fn _swap(&mut self) -> bool {
    if self.swap {
      let mut noswap = true;
      for d in 0 .. self.ndim {
        if self.perm[d as usize] != d {
          noswap = false;
          break;
        }
      }
      if noswap {
        self.swap = false;
      }
    }
    self.swap
  }*/

  pub fn _step(&mut self, vop: &CellVOp) /*-> CellViewStep */{
    match vop {
      /*&CellVOp::Nop => {
        if self._swap() {
          return CellViewStep::Swap;
        }
        // TODO
        return CellViewStep::Halt;
      }*/
      &CellVOp::Swap(ld, rd) => {
        assert!(ld < self.ndim);
        assert!(ld >= -self.ndim);
        assert!(rd < self.ndim);
        assert!(rd >= -self.ndim);
        let lidx = if ld < 0 { self.ndim + ld } else { ld } as usize;
        let ridx = if rd < 0 { self.ndim + rd } else { rd } as usize;
        self.perm.swap(lidx, ridx);
        //self.swap = true;
      }
      _ => {
        /*if self._swap() {
          return CellViewStep::Swap;
        }*/
        // TODO
        println!("DEBUG: CellViewState::_step: unimplemented: vop={:?}", vop);
        panic!("bug");
      }
    }
    //CellViewStep::Break
  }
}

pub type ScalarVal = ScalarVal_;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[non_exhaustive]
pub enum ScalarVal_ {
  F64(TotalOrd<f64>),
  F32(TotalOrd<f32>),
  F16(TotalOrd<f16>),
  I64(i64),
  I32(i32),
  I16(i16),
  I8(i8),
  U64(u64),
  U32(u32),
  U16(u16),
  U8(u8),
  Bot,
  // TODO
}

impl Neg for ScalarVal_ {
  type Output = ScalarVal_;

  fn neg(self) -> ScalarVal_ {
    match self {
      ScalarVal_::F64(x) => ScalarVal_::F64(-x),
      ScalarVal_::F32(x) => ScalarVal_::F32(-x),
      ScalarVal_::F16(x) => ScalarVal_::F16(-x),
      _ => unimplemented!()
    }
  }
}

impl ScalarVal_ {
  pub fn zero(dtype: Dtype) -> ScalarVal_ {
    match dtype {
      Dtype::F64 => ScalarVal_::F64(<f64 as FpConstExt>::zero().into()),
      Dtype::F32 => ScalarVal_::F32(<f32 as FpConstExt>::zero().into()),
      Dtype::F16 => ScalarVal_::F16(<f16 as FpConstExt>::zero().into()),
      Dtype::I64 => ScalarVal_::I64(<i64 as UintConstExt>::zero().into()),
      Dtype::I32 => ScalarVal_::I32(<i32 as UintConstExt>::zero().into()),
      Dtype::I16 => ScalarVal_::I16(<i16 as UintConstExt>::zero().into()),
      Dtype::I8 => ScalarVal_::I8(<i8 as UintConstExt>::zero().into()),
      Dtype::U64 => ScalarVal_::U64(<u64 as UintConstExt>::zero().into()),
      Dtype::U32 => ScalarVal_::U32(<u32 as UintConstExt>::zero().into()),
      Dtype::U16 => ScalarVal_::U16(<u16 as UintConstExt>::zero().into()),
      Dtype::U8 => ScalarVal_::U8(<u8 as UintConstExt>::zero().into()),
      _ => unimplemented!()
    }
  }

  pub fn one(dtype: Dtype) -> ScalarVal_ {
    match dtype {
      Dtype::F64 => ScalarVal_::F64(<f64 as FpConstExt>::one().into()),
      Dtype::F32 => ScalarVal_::F32(<f32 as FpConstExt>::one().into()),
      Dtype::F16 => ScalarVal_::F16(<f16 as FpConstExt>::one().into()),
      Dtype::I64 => ScalarVal_::I64(<i64 as UintConstExt>::one().into()),
      Dtype::I32 => ScalarVal_::I32(<i32 as UintConstExt>::one().into()),
      Dtype::I16 => ScalarVal_::I16(<i16 as UintConstExt>::one().into()),
      Dtype::I8 => ScalarVal_::I8(<i8 as UintConstExt>::one().into()),
      Dtype::U64 => ScalarVal_::U64(<u64 as UintConstExt>::one().into()),
      Dtype::U32 => ScalarVal_::U32(<u32 as UintConstExt>::one().into()),
      Dtype::U16 => ScalarVal_::U16(<u16 as UintConstExt>::one().into()),
      Dtype::U8 => ScalarVal_::U8(<u8 as UintConstExt>::one().into()),
      _ => unimplemented!()
    }
  }

  pub fn cast_to(&self, dtype: Dtype) -> ScalarVal_ {
    // TODO
    unimplemented!();
  }

  pub fn is_bot(self) -> bool {
    match self {
      ScalarVal_::Bot => true,
      _ => false
    }
  }

  pub fn dtype(self) -> Dtype {
    match self {
      ScalarVal_::F64(_) => Dtype::F64,
      ScalarVal_::F32(_) => Dtype::F32,
      ScalarVal_::F16(_) => Dtype::F16,
      ScalarVal_::I64(_) => Dtype::I64,
      ScalarVal_::I32(_) => Dtype::I32,
      ScalarVal_::I16(_) => Dtype::I16,
      ScalarVal_::I8(_) => Dtype::I8,
      ScalarVal_::U64(_) => Dtype::U64,
      ScalarVal_::U32(_) => Dtype::U32,
      ScalarVal_::U16(_) => Dtype::U16,
      ScalarVal_::U8(_) => Dtype::U8,
      ScalarVal_::Bot => Dtype::_Bot,
      _ => unimplemented!()
    }
  }

  pub fn to_abi_scalar(&self) -> FutAbiScalar {
    match self {
      &ScalarVal_::F64(x) => FutAbiScalar::F64(x.0.into()),
      &ScalarVal_::F32(x) => FutAbiScalar::F32(x.0.into()),
      &ScalarVal_::F16(x) => FutAbiScalar::F16(x.0.to_bits().into()),
      &ScalarVal_::I64(x) => FutAbiScalar::I64(x.into()),
      &ScalarVal_::I32(x) => FutAbiScalar::I32(x.into()),
      &ScalarVal_::I16(x) => FutAbiScalar::I16(x.into()),
      &ScalarVal_::I8(x) => FutAbiScalar::I8(x.into()),
      &ScalarVal_::U64(x) => FutAbiScalar::U64(x.into()),
      &ScalarVal_::U32(x) => FutAbiScalar::U32(x.into()),
      &ScalarVal_::U16(x) => FutAbiScalar::U16(x.into()),
      &ScalarVal_::U8(x) => FutAbiScalar::U8(x.into()),
      &ScalarVal_::Bot => panic!("bug"),
      _ => unimplemented!()
    }
  }

  pub fn format_futhark(&self) -> SmolStr {
    // FIXME: use the formatter.
    match self {
      ScalarVal_::F64(x) => {
        if x.0.is_nan() {
          unimplemented!();
        } else if x.0.is_infinite() {
          if x.0 < 0.0 {
            format!("-f64.inf").into()
          } else {
            format!("f64.inf").into()
          }
        } else {
          format!("{}f64", x.0).into()
        }
      }
      ScalarVal_::F32(x) => {
        if x.0.is_nan() {
          unimplemented!();
        } else if x.0.is_infinite() {
          if x.0 < 0.0 {
            format!("-f32.inf").into()
          } else {
            format!("f32.inf").into()
          }
        } else {
          format!("{}f32", x.0).into()
        }
      }
      ScalarVal_::F16(x) => {
        // FIXME FIXME
        /*if x.0.to_bits() == 0 {
          "0.0f16".into()
        } else if x.0.to_bits() == 0x3c00 {
          "1.0f16".into()
        } else if x.0.to_bits() == 0x8000 {
          "-0.0f16".into()
        } else if x.0.to_bits() == 0xbc00 {
          "-1.0f16".into()
        } else {
          unimplemented!();
        }*/
        if x.0.is_nan() {
          unimplemented!();
        } else if x.0.is_infinite() {
          if x.0 < f16::from_bits(0) {
            format!("-f16.inf").into()
          } else {
            format!("f16.inf").into()
          }
        } else {
          format!("{}f16", x.0).into()
        }
      }
      ScalarVal_::Bot => {
        panic!("bug");
      }
      _ => {
        unimplemented!();
      }
    }
  }
}

pub trait IntoScalarValExt {
  fn into_scalar_val_(self) -> ScalarVal_;

  fn into_scalar_val(self) -> ScalarVal_ where Self: Sized {
    self.into_scalar_val_()
  }
}

impl IntoScalarValExt for ScalarVal_ {
  fn into_scalar_val_(self) -> ScalarVal_ {
    self
  }
}

impl IntoScalarValExt for f16 {
  fn into_scalar_val_(self) -> ScalarVal_ {
    ScalarVal_::F16(self.into())
  }
}

impl IntoScalarValExt for f32 {
  fn into_scalar_val_(self) -> ScalarVal_ {
    ScalarVal_::F32(self.into())
  }
}

impl IntoScalarValExt for f64 {
  fn into_scalar_val_(self) -> ScalarVal_ {
    ScalarVal_::F64(self.into())
  }
}

impl IntoScalarValExt for i32 {
  fn into_scalar_val_(self) -> ScalarVal_ {
    ScalarVal_::I32(self)
  }
}

impl IntoScalarValExt for i64 {
  fn into_scalar_val_(self) -> ScalarVal_ {
    ScalarVal_::I64(self)
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[non_exhaustive]
pub enum DtypeKind_ {
  Fp(u8, u8),
  Int,
  Uint,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
#[non_exhaustive]
pub enum Dtype {
  _Top,
  F64,
  F32,
  I64,
  I32,
  I16,
  I8,
  U64,
  U32,
  U16,
  U8,
  F16,
  Bf16,
  _Bot,
}

impl Decodable for Dtype {
  fn decode<D: Decoder>(d: &mut D) -> Result<Dtype, D::Error> {
    Dtype::from_str(d.read_str()?.as_str()).map_err(|_| d.error("invalid dtype"))
  }
}

impl Encodable for Dtype {
  fn encode<E: Encoder>(&self, e: &mut E) -> Result<(), E::Error> {
    e.emit_str(self._to_str())
  }
}

impl TryFrom<TensorDtype> for Dtype {
  type Error = SmolStr;

  fn try_from(t: TensorDtype) -> Result<Dtype, SmolStr> {
    Ok(match t {
      TensorDtype::F64 => Dtype::F64,
      TensorDtype::F32 => Dtype::F32,
      TensorDtype::I64 => Dtype::I64,
      TensorDtype::I32 => Dtype::I32,
      TensorDtype::I16 => Dtype::I16,
      TensorDtype::I8 => Dtype::I8,
      TensorDtype::U64 => Dtype::U64,
      TensorDtype::U32 => Dtype::U32,
      TensorDtype::U16 => Dtype::U16,
      TensorDtype::U8 => Dtype::U8,
      TensorDtype::Bool => Dtype::U8,
      TensorDtype::F16 => Dtype::F16,
      TensorDtype::Bf16 => Dtype::Bf16,
      _ => unimplemented!()
    })
  }
}

impl TryFrom<TorchDtype> for Dtype {
  //type Error = String;
  type Error = SmolStr;

  fn try_from(t: TorchDtype) -> Result<Dtype, SmolStr> {
    Ok(match t {
      TorchDtype::Float64 => Dtype::F64,
      TorchDtype::Float32 => Dtype::F32,
      TorchDtype::Float16 => Dtype::F16,
      TorchDtype::Bfloat16 => Dtype::Bf16,
      TorchDtype::Int64 => Dtype::I64,
      TorchDtype::Int32 => Dtype::I32,
      TorchDtype::Int16 => Dtype::I16,
      TorchDtype::Int8 => Dtype::I8,
      TorchDtype::UInt64 => Dtype::U64,
      TorchDtype::UInt32 => Dtype::U32,
      TorchDtype::UInt16 => Dtype::U16,
      TorchDtype::UInt8 => Dtype::U8,
    })
  }
}

impl TryFrom<SmolStr> for Dtype {
  type Error = SmolStr;

  fn try_from(s: SmolStr) -> Result<Dtype, SmolStr> {
    Dtype::from_str(s.as_str())
  }
}

impl TryFrom<String> for Dtype {
  type Error = SmolStr;

  fn try_from(s: String) -> Result<Dtype, SmolStr> {
    Dtype::from_str(s.as_str())
  }
}

impl<'a> TryFrom<&'a str> for Dtype {
  type Error = SmolStr;

  fn try_from(s: &'a str) -> Result<Dtype, SmolStr> {
    Dtype::from_str(s)
  }
}

impl FromStr for Dtype {
  type Err = SmolStr;

  fn from_str(s: &str) -> Result<Dtype, SmolStr> {
    Ok(match s {
      "f64"     |
      "float64" => Dtype::F64,
      "f32"     |
      "float32" => Dtype::F32,
      "f16"     |
      "float16" => Dtype::F16,
      "bf16"    |
      "bfloat16" => Dtype::Bf16,
      "i64"     |
      "int64"   => Dtype::I64,
      "i32"     |
      "int32"   => Dtype::I32,
      "i16"     |
      "int16"   => Dtype::I16,
      "i8"      |
      "int8"    => Dtype::I8,
      "u64"     |
      "uint64"  => Dtype::U64,
      "u32"     |
      "uint32"  => Dtype::U32,
      "u16"     |
      "uint16"  => Dtype::U16,
      "u8"      |
      "uint8"   => Dtype::U8,
      _ => return Err(s.into())
    })
  }
}

impl Dtype {
  pub fn top() -> Dtype {
    Dtype::_Top
  }

  pub fn _to_str(self) -> &'static str {
    match self {
      Dtype::_Top   => "top",
      Dtype::F64    => "f64",
      Dtype::F32    => "f32",
      Dtype::F16    => "f16",
      Dtype::Bf16   => "bf16",
      Dtype::I64    => "i64",
      Dtype::I32    => "i32",
      Dtype::I16    => "i16",
      Dtype::I8     => "i8",
      Dtype::U64    => "u64",
      Dtype::U32    => "u32",
      Dtype::U16    => "u16",
      Dtype::U8     => "u8",
      Dtype::_Bot   => "bot",
    }
  }

  pub fn format_futhark(self) -> &'static str {
    match self {
      Dtype::_Top   => panic!("bug"),
      Dtype::F64    => "f64",
      Dtype::F32    => "f32",
      Dtype::F16    => "f16",
      Dtype::Bf16   => {
        println!("ERROR:  Dtype::format_futhark: unimplemented: bfloat16 is not currently supported by Futhark");
        panic!();
      }
      Dtype::I64    => "i64",
      Dtype::I32    => "i32",
      Dtype::I16    => "i16",
      Dtype::I8     => "i8",
      Dtype::U64    => "u64",
      Dtype::U32    => "u32",
      Dtype::U16    => "u16",
      Dtype::U8     => "u8",
      Dtype::_Bot   => panic!("bug"),
    }
  }

  pub fn parts(self) -> (DtypeKind_, u8) {
    match self {
      Dtype::_Top   => panic!("bug"),
      Dtype::F64    => (DtypeKind_::Fp(11, 52), 64),
      Dtype::F32    => (DtypeKind_::Fp(8, 23), 32),
      Dtype::F16    => (DtypeKind_::Fp(5, 10), 16),
      Dtype::Bf16   => (DtypeKind_::Fp(8, 7), 16),
      Dtype::I64    => (DtypeKind_::Int, 64),
      Dtype::I32    => (DtypeKind_::Int, 32),
      Dtype::I16    => (DtypeKind_::Int, 16),
      Dtype::I8     => (DtypeKind_::Int, 8),
      Dtype::U64    => (DtypeKind_::Uint, 64),
      Dtype::U32    => (DtypeKind_::Uint, 32),
      Dtype::U16    => (DtypeKind_::Uint, 16),
      Dtype::U8     => (DtypeKind_::Uint, 8),
      Dtype::_Bot   => panic!("bug"),
    }
  }

  pub fn size_bits(self) -> u8 {
    let (_, nbits) = self.parts();
    nbits
  }

  pub fn size_bytes(self) -> usize {
    let (_, nbits) = self.parts();
    assert_eq!(nbits % 8, 0);
    (nbits / 8) as usize
  }

  pub fn align_bytes(self) -> usize {
    // FIXME
    self.size_bytes()
  }

  pub fn is_float(self) -> bool {
    let (kind, _) = self.parts();
    match kind {
      DtypeKind_::Fp(..) => true,
      _ => false
    }
  }

  pub fn is_signed_int(self) -> bool {
    let (kind, _) = self.parts();
    kind == DtypeKind_::Int
  }

  pub fn is_unsigned_int(self) -> bool {
    let (kind, _) = self.parts();
    kind == DtypeKind_::Uint
  }

  pub fn is_uint(self) -> bool {
    self.is_unsigned_int()
  }

  pub fn max(self, rhs: Dtype) -> Option<Dtype> {
    let (lkind, lnbits) = self.parts();
    let (rkind, rnbits) = rhs.parts();
    match (lkind, rkind) {
      (DtypeKind_::Fp(le, lm), DtypeKind_::Fp(re, rm)) => {
        if le >= re && lm >= rm {
          return Some(self);
        } else if le <= re && lm <= rm {
          return Some(rhs);
        }
      }
      (DtypeKind_::Int, DtypeKind_::Int) |
      (DtypeKind_::Uint, DtypeKind_::Uint) => {
        if lnbits >= rnbits {
          return Some(self);
        } else {
          return Some(rhs);
        }
      }
      _ => {}
    }
    None
  }
}

/*pub trait DtypeExt: Any {
  fn as_any(&self) -> &dyn Any { &self }
  fn _dtype(&self) -> Dtype where Self: Sized { <Self as DtypeExt>::dtype() }
  fn dtype() -> Dtype where Self: Sized;
}*/

pub trait DtypeExt {
  // FIXME FIXME
  //fn is_zero(&self) -> bool;
}

pub trait DtypeConstExt {
  fn dtype_() -> Dtype where Self: Sized;
}

impl DtypeConstExt for TotalOrd<f64> { fn dtype_() -> Dtype { Dtype::F64 } }
impl DtypeConstExt for TotalOrd<f32> { fn dtype_() -> Dtype { Dtype::F32 } }
impl DtypeConstExt for TotalOrd<f16> { fn dtype_() -> Dtype { Dtype::F16 } }
//impl DtypeConstExt for TotalOrd<bf16> { fn dtype_() -> Dtype { Dtype::Bf16 } }

impl DtypeConstExt for f64 { fn dtype_() -> Dtype { Dtype::F64 } }
impl DtypeConstExt for f32 { fn dtype_() -> Dtype { Dtype::F32 } }
impl DtypeConstExt for f16 { fn dtype_() -> Dtype { Dtype::F16 } }
impl DtypeConstExt for bf16 { fn dtype_() -> Dtype { Dtype::Bf16 } }
impl DtypeConstExt for i64 { fn dtype_() -> Dtype { Dtype::I64 } }
impl DtypeConstExt for i32 { fn dtype_() -> Dtype { Dtype::I32 } }
impl DtypeConstExt for i16 { fn dtype_() -> Dtype { Dtype::I16 } }
impl DtypeConstExt for i8  { fn dtype_() -> Dtype { Dtype::I8 } }
impl DtypeConstExt for u64 { fn dtype_() -> Dtype { Dtype::U64 } }
impl DtypeConstExt for u32 { fn dtype_() -> Dtype { Dtype::U32 } }
impl DtypeConstExt for u16 { fn dtype_() -> Dtype { Dtype::U16 } }
impl DtypeConstExt for u8  { fn dtype_() -> Dtype { Dtype::U8 } }

pub fn dtype_of<T: DtypeConstExt>() -> Dtype {
  T::dtype_()
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Dim {
  pub ndim:     i8,
  pub dtype:    Dtype,
}

impl Dim {
  pub fn ndim(&self) -> i8 {
    self.ndim
  }

  pub fn dtype(&self) -> Dtype {
    self.dtype
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

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct CellType {
  pub shape:    Box<[i64]>,
  pub dtype:    Dtype,
}

impl Debug for CellType {
  fn fmt(&self, f: &mut Formatter) -> FmtResult {
    write!(f, "CellType({:?}{})", self.shape, self.dtype._to_str())
  }
}

impl CellType {
  pub fn top() -> CellType {
    CellType{
      shape:    Box::new([]),
      dtype:    Dtype::_Top,
    }
  }

  pub fn cast(&self, new_dtype: Dtype) -> CellType {
    CellType{shape: self.shape.clone(), dtype: new_dtype}
  }

  pub fn lossy_cast(&self, new_dtype: Dtype) -> CellType {
    // FIXME FIXME
    CellType{shape: self.shape.clone(), dtype: new_dtype}
  }

  pub fn to_dim(&self) -> Dim {
    assert!(self.dtype != Dtype::_Top);
    Dim{ndim: self.ndim(), dtype: self.dtype}
  }

  pub fn ndim(&self) -> i8 {
    assert!(self.dtype != Dtype::_Top);
    assert!(self.shape.len() <= i8::max_value() as usize);
    self.shape.len() as i8
  }

  pub fn shape(&self) -> &[i64] {
    &self.shape
  }

  pub fn dtype(&self) -> Dtype {
    self.dtype
  }

  pub fn is_top(&self) -> bool {
    if self.dtype == Dtype::_Top {
      assert_eq!(self.shape.len(), 0);
      true
    } else {
      false
    }
  }

  pub fn is_scalar(&self) -> bool {
    self.ndim() == 0
  }

  pub fn outer_len(&self) -> i64 {
    assert!(self.shape.len() > 0);
    self.shape[0]
  }

  pub fn inner_len(&self) -> i64 {
    assert!(self.shape.len() > 0);
    self.shape[self.shape.len() - 1]
  }

  pub fn flat_len(&self) -> i64 {
    let nd = self.ndim() as usize;
    let mut span = if nd == 0 { 1 } else { self.shape[(nd - 1)] };
    for d in 1 .. nd {
      span = self.shape[(nd - 1) - d] * span;
    }
    span
  }

  pub fn span_bytes(&self) -> u64 {
    self.packed_span_bytes()
  }

  pub fn packed_span_bytes(&self) -> u64 {
    let span = self.flat_len();
    (span * self.dtype.size_bytes() as i64) as u64
  }

  pub fn packed_stride_and_span_bytes(&self) -> (Vec<i64>, u64) {
    let nd = self.ndim() as usize;
    let mut stride = Vec::with_capacity(nd);
    if nd > 0 {
      stride.push(1);
    }
    let mut span = if nd == 0 { 1 } else { self.shape[(nd - 1)] };
    for d in 1 .. nd {
      stride.push(span);
      span = self.shape[(nd - 1) - d] * span;
    }
    stride.reverse();
    (stride, (span * self.dtype.size_bytes() as i64) as u64)
  }

  pub fn shape_compat(&self, orig: &CellType) -> ShapeCompat {
    assert!(self.dtype != Dtype::_Top);
    assert!(orig.dtype != Dtype::_Top);
    CellType::_shape_compat(&self.shape, &orig.shape)
  }

  pub fn _shape_compat(shape: &[i64], orig_shape: &[i64]) -> ShapeCompat {
    if &shape == &orig_shape {
      return ShapeCompat::Equal;
    }
    //let nd = self.ndim() as usize;
    //let o_nd = orig.ndim() as usize;
    let nd = shape.len();
    let o_nd = orig_shape.len();
    let mut span = Vec::with_capacity(nd);
    let mut o_span = Vec::with_capacity(o_nd);
    if nd == 0 {
      span.push(1);
    } else {
      span.push(shape[(nd - 1)]);
    }
    for d in 1 .. nd {
      span.push(shape[(nd - 1) - d] * span[d - 1]);
    }
    span.reverse();
    if o_nd == 0 {
      o_span.push(1);
    } else {
      o_span.push(orig_shape[(o_nd - 1)]);
    }
    for d in 1 .. o_nd {
      o_span.push(orig_shape[(o_nd - 1) - d] * o_span[d - 1]);
    }
    o_span.reverse();
    //println!("DEBUG: CellType:;shape_compat: shape={:?} o_shape={:?}", &shape, &orig_shape);
    //println!("DEBUG: CellType:;shape_compat: span={:?} o_span={:?}", &span, &o_span);
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

  pub fn unbroadcast(&self) -> CellType {
    let mut shape = Vec::new();
    let nd = self.ndim() as usize;
    for d in 0 .. nd {
      if self.shape[d] <= 0 {
        unimplemented!();
      } else if self.shape[d] > 1 {
        shape.push(self.shape[d]);
      }
    }
    let shape = shape.into();
    CellType{shape, dtype: self.dtype}
  }

  pub fn prepare_bytes<'this, 'b, T: DtypeConstExt + Copy>(&'this self, buf: &'b [u8]) -> Option<&'b [T]> {
    if self.dtype != T::dtype_() {
      return None;
    }
    if buf.as_ptr().align_offset(align_of::<T>()) != 0 {
      return None;
    }
    if buf.len() % size_of::<T>() != 0 {
      return None;
    }
    let dlen = buf.len() / size_of::<T>();
    let idlen: i64 = dlen.try_into().unwrap();
    if idlen != self.flat_len() {
      return None;
    }
    unsafe {
      Some(from_raw_parts(buf.as_ptr() as *const T, dlen))
    }
  }

  pub fn prepare_bytes_mut<'this, 'b, T: DtypeConstExt + Copy>(&'this self, buf: &'b mut [u8]) -> Option<&'b mut [T]> {
    if self.dtype != T::dtype_() {
      return None;
    }
    if buf.as_ptr().align_offset(align_of::<T>()) != 0 {
      return None;
    }
    if buf.len() % size_of::<T>() != 0 {
      return None;
    }
    let dlen = buf.len() / size_of::<T>();
    let idlen: i64 = dlen.try_into().unwrap();
    if idlen != self.flat_len() {
      return None;
    }
    unsafe {
      Some(from_raw_parts_mut(buf.as_mut_ptr() as *mut T, dlen))
    }
  }
}

#[derive(Clone, Debug)]
pub struct TypedMem<'a> {
  pub ty_:  CellType,
  pub buf:  &'a [u8],
}

impl<'a> TypedMem<'a> {
  pub fn type_(&self) -> &CellType {
    &self.ty_
  }

  pub fn as_bytes(&self) -> &'a [u8] {
    self.buf
  }

  pub fn into_bytes(self) -> &'a [u8] {
    self.buf
  }

  pub fn as_slice<T: DtypeConstExt + Copy>(&self) -> Option<&'a [T]> {
    self.ty_.prepare_bytes::<T>(self.buf)
  }

  pub fn into_slice<T: DtypeConstExt + Copy>(self) -> Option<&'a [T]> {
    self.ty_.prepare_bytes::<T>(self.buf)
  }
}

#[derive(Debug)]
pub struct TypedMemMut<'a> {
  pub ty_:  CellType,
  pub buf:  &'a mut [u8],
}

impl<'a> TypedMemMut<'a> {
  pub fn type_(&self) -> &CellType {
    &self.ty_
  }

  pub fn into_bytes(self) -> &'a [u8] {
    self.buf
  }

  pub fn into_mut_bytes(self) -> &'a mut [u8] {
    self.buf
  }

  pub fn into_slice<T: DtypeConstExt + Copy>(self) -> Option<&'a [T]> {
    self.ty_.prepare_bytes::<T>(self.buf)
  }

  pub fn into_mut_slice<T: DtypeConstExt + Copy>(self) -> Option<&'a mut [T]> {
    self.ty_.prepare_bytes_mut::<T>(self.buf)
  }
}

#[derive(Clone, Debug)]
pub struct CellSliceType {
  pub offset:   Vec<i64>,
  //pub shape:    Vec<i64>,
  //pub dtype:    Dtype,
  pub type_:    CellType,
}

impl AsRef<CellType> for CellSliceType {
  fn as_ref(&self) -> &CellType {
    &self.type_
  }
}

impl CellSliceType {
  pub fn pointer_offset(&self) -> u64 {
    let mut ptroff = 0;
    let mut pitch = self.type_.dtype.size_bytes() as u64;
    for d in (0 .. self.type_.shape.len()).rev() {
      let o = self.offset[d];
      assert!(o >= 0);
      ptroff += (o as u64) * pitch;
      pitch *= self.type_.shape[d] as u64;
    }
    ptroff
  }
}

#[derive(Clone, Debug)]
pub struct CellStridedSliceType {
  pub stride:   Vec<i64>,
  pub offset:   Vec<i64>,
  pub shape:    Vec<i64>,
  pub dtype:    Dtype,
  //pub slice:    CellSliceType,
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

#[derive(Clone, Debug)]
pub struct CellTransposeType {
  pub perm: Vec<i8>,
}

impl CellTransposeType {
  pub fn is_identity(&self) -> bool {
    let nd = self.perm.len();
    for d in 1 .. nd {
      if self.perm[d - 1] == self.perm[d] {
        panic!("bug");
      } else if self.perm[d - 1] > self.perm[d] {
        return false;
      }
    }
    true
  }

  pub fn is_new_shape(&self, shape: &[i64]) -> bool {
    let nd = self.perm.len();
    assert_eq!(nd, shape.len());
    let mut tag = Vec::with_capacity(nd);
    let mut rank: i8 = 1;
    for d in 0 .. nd {
      if shape[d] < 0 {
        panic!("bug");
      } else if shape[d] == 0 {
        return true;
      } else if shape[d] == 1 {
        tag.push(0);
      } else {
        tag.push(rank);
        rank += 1;
      }
    }
    let mut perm_tag = Vec::with_capacity(nd);
    for d in 0 .. nd {
      perm_tag.push(tag[self.perm[d] as usize]);
    }
    rank = 1;
    for d in 0 .. nd {
      if perm_tag[d] < 0 {
        panic!("bug");
      } else if perm_tag[d] == 0 {
      } else if perm_tag[d] != rank {
        return false;
      } else {
        rank += 1;
      }
    }
    true
  }
}

#[derive(Clone, Default, Debug)]
pub struct CellState {
  pub clk:  Clock,
  // FIXME
  //pub seal: Clock,
}

pub struct PCellReplica {
  pub clk:  Cell<Clock>,
  pub addr: Cell<PAddr>,
}

pub struct PCell {
  pub optr: CellPtr,
  pub ogty: CellType,
  pub replicas: RevSortMap8<(Locus, PMach), PCellReplica>,
}

impl PCell {
  pub fn new(ptr: CellPtr, ty: CellType) -> PCell {
    if cfg_debug() { println!("DEBUG: PCell::new: optr={:?} ogty={:?}", ptr, &ty); }
    let replicas = RevSortMap8::new();
    PCell{
      optr: ptr,
      ogty: ty,
      replicas,
    }
  }

  pub fn _dump_replicas(&self) {
    print!("DEBUG:  PCell::_dump_replicas: optr={:?} ogty={:?} reps=[", self.optr, &self.ogty);
    for (i, (key, rep)) in self.replicas.iter().enumerate() {
      let &(loc, pm) = key.as_ref();
      let addr = rep.addr.get();
      let clk = rep.clk.get();
      if i == 0 {
        print!("({:?} {:?} {:?} {:?})", loc, pm, addr, clk);
      } else {
        print!(", ({:?} {:?} {:?} {:?})", loc, pm, addr, clk);
      }
    }
    println!("]");
  }

  pub fn _push(&mut self, clk: Clock, locus: Locus, pmach: PMach, addr: PAddr) {
    let rep = PCellReplica{clk: Cell::new(clk), addr: Cell::new(addr)};
    let key = self.replicas.insert((locus, pmach), rep);
  }

  pub fn push(&mut self, root: CellPtr, clk: Clock, locus: Locus, pmach: PMach, addr: PAddr) {
    match self.replicas.find((locus, pmach)) {
      None => {}
      Some(_) => panic!("bug")
    }
    TL_PCTX.with(|pctx| {
      assert_eq!(pctx.set_root(addr, root), None);
    });
    self._push(clk, locus, pmach, addr);
  }

  pub fn push_noroot(&mut self, clk: Clock, locus: Locus, pmach: PMach, addr: PAddr) {
    match self.replicas.find((locus, pmach)) {
      None => {}
      Some(_) => panic!("bug")
    }
    self._push(clk, locus, pmach, addr);
  }

  /*pub fn push_new_replica(&mut self, root: CellPtr, clk: Clock, locus: Locus, pmach: PMach, addr: PAddr) {
    self.push(root, clk, locus, pmach, addr)
  }*/

  pub fn swap(&mut self, root: CellPtr, clk: Clock, locus: Locus, pmach: PMach, addr: PAddr) -> Option<(Clock, PAddr)> {
    match self.replicas.find((locus, pmach)) {
      None => {
        TL_PCTX.with(|pctx| {
          assert_eq!(pctx.set_root(addr, root), None);
        });
        self._push(clk, locus, pmach, addr);
        None
      }
      Some((_, rep)) => {
        let o_clk = rep.clk.get();
        let o_addr = rep.addr.get();
        assert!(o_clk <= clk);
        TL_PCTX.with(|pctx| {
          assert_eq!(pctx.unset_root(o_addr), Some(root));
          assert_eq!(pctx.set_root(addr, root), None);
        });
        rep.clk.set(clk);
        rep.addr.set(addr);
        Some((o_clk, o_addr))
      }
    }
  }

  /*pub fn _pop(&self, x: CellPtr, /*xclk: Clock,*/ q_addr: PAddr) {
    for (key, rep) in self.replicas.iter() {
      if rep.addr.get() == q_addr {
        TL_PCTX.with(|pctx| {
          assert_eq!(pctx.unset_root(rep.addr.get()), Some(x));
          // FIXME FIXME: also free the addr?
        });
        rep.addr.set(PAddr::nil());
        rep.clk.set(Clock::default());
        return;
      }
    }
    panic!("bug");
  }*/

  pub fn yeet(&mut self, /*root: CellPtr, q_clk: Clock,*/ q_locus: Locus, q_pmach: PMach) -> Option<(Clock, PAddr, Rc<dyn InnerCell_>)> {
    let mut ret: Option<(Clock, PAddr, Rc<dyn InnerCell_>)> = None;
    for (key, rep) in self.replicas.iter() {
      let &(loc, pm) = key.as_ref();
      if loc == q_locus && pm == q_pmach {
        let addr = rep.addr.get();
        if let Some((loc2, pm2, icel)) = TL_PCTX.with(|pctx| {
          pctx.yeet(addr)
        }) {
          assert_eq!(loc, loc2);
          assert_eq!(pm, pm2);
          ret = Some((rep.clk.get(), addr, icel));
          break;
        }
      }
    }
    if let Some(ret) = ret {
      self.replicas.remove((q_locus, q_pmach));
      return Some(ret);
    }
    return None;
  }

  pub fn _lookup(&self, q_locus: Locus, q_pmach: PMach) -> Option<&PCellReplica> {
    'retry: loop {
      for (key, rep) in self.replicas.iter() {
        let &(loc, pm) = key.as_ref();
        if loc == q_locus && pm == q_pmach {
          let addr = rep.addr.get();
          if TL_PCTX.with(|pctx| {
            pctx.lookup_pm(pm, addr).is_some()
          }) {
            return Some(rep);
          }
        }
      }
      return None;
    }
  }

  pub fn lookup(&mut self, q_locus: Locus, q_pmach: PMach) -> Option<(Clock, PAddr)> {
    'retry: loop {
      let mut unkey: Option<(Locus, PMach)> = None;
      for (key, rep) in self.replicas.iter() {
        let &(loc, pm) = key.as_ref();
        if loc == q_locus && pm == q_pmach {
          let addr = rep.addr.get();
          if TL_PCTX.with(|pctx| {
            pctx.lookup_pm(pm, addr).is_some()
          }) {
            return Some((rep.clk.get(), addr));
          } else {
            unkey = Some(*key.as_ref());
            break;
          }
        }
      }
      if let Some(key) = unkey {
        self.replicas.remove(key);
        continue 'retry;
      }
      return None;
    }
  }

  pub fn lookup_loc(&mut self, q_locus: Locus) -> Option<(Clock, PMach, PAddr)> {
    'retry: loop {
      let mut unkey: Option<(Locus, PMach)> = None;
      for (key, rep) in self.replicas.iter() {
        let &(loc, pm) = key.as_ref();
        if loc == q_locus {
          let addr = rep.addr.get();
          if TL_PCTX.with(|pctx| {
            pctx.lookup_pm(pm, addr).is_some()
          }) {
            return Some((rep.clk.get(), pm, addr));
          } else {
            unkey = Some(*key.as_ref());
            break;
          }
        }
      }
      if let Some(key) = unkey {
        self.replicas.remove(key);
        continue 'retry;
      }
      return None;
    }
  }

  pub fn find_any(&mut self, q_clk: Clock) -> Option<(Locus, PMach, PAddr)> {
    'retry: loop {
      let mut unkey: Option<(Locus, PMach)> = None;
      for (key, rep) in self.replicas.iter() {
        let &(loc, pm) = key.as_ref();
        let clk = rep.clk.get();
        if clk > q_clk {
          panic!("bug");
        } else if clk == q_clk {
          let addr = rep.addr.get();
          if TL_PCTX.with(|pctx| {
            pctx.lookup_pm(pm, addr).is_some()
          }) {
            return Some((loc, pm, addr));
          } else {
            unkey = Some(*key.as_ref());
            break;
          }
        }
      }
      if let Some(key) = unkey {
        self.replicas.remove(key);
        continue 'retry;
      }
      return None;
    }
  }

  pub fn find_any_other(&mut self, q_clk: Clock, neq_locus: Locus, neq_pmach: PMach) -> Option<(Locus, PMach, PAddr)> {
    'retry: loop {
      let mut unkey: Option<(Locus, PMach)> = None;
      for (key, rep) in self.replicas.iter() {
        let &(loc, pm) = key.as_ref();
        let clk = rep.clk.get();
        if clk > q_clk {
          panic!("bug");
        } else if clk == q_clk && !(loc == neq_locus && pm == neq_pmach) {
          let addr = rep.addr.get();
          if TL_PCTX.with(|pctx| {
            pctx.lookup_pm(pm, addr).is_some()
          }) {
            return Some((loc, pm, addr));
          } else {
            unkey = Some(*key.as_ref());
            break;
          }
        }
      }
      if let Some(key) = unkey {
        self.replicas.remove(key);
        continue 'retry;
      }
      return None;
    }
  }

  pub fn find_loc_nocow(&mut self, q_clk: Clock, q_locus: Locus) -> Option<(PMach, PAddr)> {
    'retry: loop {
      let mut unkey: Option<(Locus, PMach)> = None;
      for (key, rep) in self.replicas.iter() {
        let &(loc, pm) = key.as_ref();
        let clk = rep.clk.get();
        if clk > q_clk {
          panic!("bug");
        } else if clk == q_clk && loc == q_locus {
          let addr = rep.addr.get();
          if TL_PCTX.with(|pctx| {
            pctx.lookup_pm(pm, addr).is_some()
          }) {
            if TL_PCTX.with(|pctx| !pctx.lookup_cow(addr)) {
              return Some((pm, addr));
            }
          } else {
            unkey = Some(*key.as_ref());
            break;
          }
        }
      }
      if let Some(key) = unkey {
        self.replicas.remove(key);
        continue 'retry;
      }
      return None;
    }
  }

  pub fn read_loc(&mut self, root: CellPtr, q_clk: Clock, ty: &CellType, q_locus: Locus) -> (PMach, PAddr) {
    let mut f_pmach = match self.lookup_loc(q_locus) {
      None => None,
      Some((prev_clk, pmach, addr)) => {
        if prev_clk > q_clk {
          println!("DEBUG: PCell::read_loc: root={:?} prev clk={:?} clk={:?} ty={:?} loc={:?} pm={:?} addr={:?}",
              root, prev_clk, q_clk, ty, q_locus, pmach, addr);
          println!("ERROR: PCell::read_loc: read failure");
          panic!("bug");
        } else if prev_clk == q_clk {
          return (pmach, addr);
        }
        Some(pmach)
      }
    };
    if cfg_debug() {
    println!("DEBUG: PCell::read_loc: root={:?} clk={:?} ty={:?} loc={:?} found pm? {:?}",
        root, q_clk, ty, q_locus, f_pmach);
    }
    match self.find_any(q_clk) {
      None => {
        println!("DEBUG: PCell::read_loc: optr={:?} ogty={:?} root={:?} clk={:?} ty={:?} loc={:?} pm={:?}",
            self.optr, &self.ogty,
            root, q_clk, ty, q_locus, f_pmach,
        );
        println!("ERROR: PCell::read_loc: no replica to copy from");
        panic!();
      }
      Some(_) => {}
    }
    if f_pmach.is_none() {
      let (pmach, addr) = TL_PCTX.with(|pctx| {
        pctx.alloc_loc(ty, q_locus)
      });
      self.push(root, Clock::default(), q_locus, pmach, addr);
      f_pmach = Some(pmach);
    }
    let f_pmach = f_pmach.unwrap();
    match self.lookup(q_locus, f_pmach) {
      None => panic!("bug"),
      Some((prev_clk, addr)) => {
        //let addr = rep.addr.get();
        //let prev_clk = rep.clk.get();
        if prev_clk >= q_clk {
          panic!("bug");
        } else /*if prev_clk < q_clk */{
          match self.find_any(q_clk) {
            None => panic!("bug"),
            Some((o_loc, o_pm, o_addr)) => {
              if cfg_debug() {
              println!("DEBUG: PCell::read_loc: optr={:?} ogty={:?} root={:?} prev clk={:?} clk={:?} ty={:?} loc={:?} pm={:?} addr={:?} found o_loc={:?} o_pm={:?} o_addr={:?}",
                  self.optr, &self.ogty,
                  root, prev_clk, q_clk, ty, q_locus, f_pmach, addr,
                  o_loc, o_pm, o_addr,
              );
              }
              TL_PCTX.with(|pctx| {
                pctx.hard_copy(q_locus, f_pmach, addr, o_loc, o_pm, o_addr, ty.packed_span_bytes() as usize);
              });
            }
          }
          match self._lookup(q_locus, f_pmach) {
            None => panic!("bug"),
            Some(rep) => {
              rep.clk.set(q_clk);
            }
          }
        }
        (f_pmach, addr)
      }
    }
  }

  pub fn write_loc(&mut self, root: CellPtr, q_clk: Clock, ty: &CellType, q_locus: Locus) -> (PMach, PAddr) {
    let mut f_pmach = match self.lookup_loc(q_locus) {
      None => None,
      Some((prev_clk, pmach, addr)) => {
        if prev_clk >= q_clk {
          println!("DEBUG: PCell::write_loc: root={:?} prev clk={:?} clk={:?} ty={:?} loc={:?} pm={:?} addr={:?}",
              root, prev_clk, q_clk, ty, q_locus, pmach, addr);
          println!("ERROR: PCell::write_loc: write failure");
          panic!("bug");
        }
        Some(pmach)
      }
    };
    if f_pmach.is_none() {
      let (pmach, addr) = TL_PCTX.with(|pctx| {
        pctx.alloc_loc(ty, q_locus)
      });
      self.push(root, Clock::default(), q_locus, pmach, addr);
      f_pmach = Some(pmach);
    }
    TL_PCTX.with(|pctx| {
      let mut f_pmach = f_pmach.unwrap();
      let mut retry = false;
      'retry: loop {
        match self._lookup(q_locus, f_pmach) {
          None => panic!("bug"),
          Some(rep) => {
            let addr = rep.addr.get();
            let prev_clk = rep.clk.get();
            if pctx.lookup_cow(addr) {
              drop(rep);
              if cfg_debug() {
                println!("DEBUG: PCell::write_loc: cow: old root={:?} prev clk={:?} clk={:?} ty={:?} loc={:?} pm={:?} addr={:?} retry?{:?}",
                    root, prev_clk, q_clk, ty, q_locus, f_pmach, addr, retry);
              }
              if retry {
                panic!("bug");
              }
              let o_pm = f_pmach;
              let o_loc = q_locus;
              let o_addr = addr;
              let (pmach, addr) = pctx.alloc_loc(ty, q_locus);
              f_pmach = pmach;
              match self.swap(root, q_clk, q_locus, f_pmach, addr) {
                None => {}
                Some((_, prev_addr)) => {
                  let _ = pctx.release(prev_addr);
                }
              }
              pctx.hard_copy(q_locus, f_pmach, addr, o_loc, o_pm, o_addr, ty.packed_span_bytes() as usize);
              // FIXME: release w/ paddr refct.
              //let _ = pctx.release(o_addr);
              if cfg_debug() {
                println!("DEBUG: PCell::write_loc: cow: new root={:?} loc={:?} pm={:?} addr={:?}",
                    root, q_locus, f_pmach, addr);
              }
              retry = true;
              continue 'retry;
            }
            if prev_clk > q_clk || (!retry && prev_clk == q_clk) {
              println!("DEBUG:  PCell::write_loc: root={:?} prev clk={:?} q clk={:?} ty={:?} q loc={:?} pmach={:?} addr={:?}",
                  root, prev_clk, q_clk, ty, q_locus, f_pmach, addr);
              panic!("bug");
            } else /*if prev_clk < q_clk */{
              rep.clk.set(q_clk);
            }
            return (f_pmach, addr);
          }
        }
      }
    })
  }
}
