use crate::algo::{HashMap, HashSet, RevSortKey8, RevSortMap8};
use crate::algo::fp::*;
use crate::algo::int::*;
use crate::clock::*;
use crate::ctx::*;
use crate::nd::{IRange};
use crate::panick::*;
use crate::pctx::{TL_PCTX, Locus, PMach, PAddr, MemReg};
use crate::thunk::*;
use crate::thunk::op::{SetScalarFutThunkSpec};
use crate::util::mmap::{MmapBuf};
use crate::util::pickle::{TorchDtype};
use cacti_cfg_env::*;

use smol_str::{SmolStr};

use std::any::{Any};
use std::borrow::{Borrow};
use std::cell::{Cell, RefCell};
use std::convert::{TryFrom, TryInto};
use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::mem::{forget, size_of, swap};
use std::ops::{Deref, Neg};
use std::rc::{Rc, Weak};
use std::slice::{from_raw_parts, from_raw_parts_mut};
use std::str::{FromStr};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct CellPtr{
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

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash)]
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

impl<'a> CellDeref for &'a StableCell {
  fn _deref(&self) -> CellPtr {
    *self.as_ptr_ref()
  }
}

impl CellDeref for StableCell {
  fn _deref(&self) -> CellPtr {
    *self.as_ptr_ref()
  }
}

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
  pub fn retain(env: &CtxEnv, ptr: CellPtr) -> StableCell {
    assert!(!ptr.is_nil());
    env.retain(ptr);
    StableCell{ptr_: ptr}
  }

  pub fn new() -> StableCell {
    let ty = CellType{shape: Vec::new(), dtype: Dtype::_Top};
    ctx_insert(ty).into()
  }

  pub fn scalar<D: TryInto<Dtype>>(dtype: D) -> StableCell {
    let dtype: Dtype = match dtype.try_into() {
      Ok(d) => d,
      Err(_) => panic!("bug: StableCell::scalar: invalid dtype")
    };
    let ty = CellType{shape: Vec::new(), dtype};
    ctx_insert(ty).into()
  }

  pub fn set_scalar<T: IntoScalarValExt>(value: T) -> StableCell {
    ctx_pop_thunk(SetScalarFutThunkSpec{val: value.into_scalar_val_()}).into()
  }

  pub fn array<S: Into<Vec<i64>>, D: TryInto<Dtype>>(shape: S, dtype: D) -> StableCell {
    let shape: Vec<i64> = shape.into();
    let dtype: Dtype = match dtype.try_into() {
      Ok(d) => d,
      Err(_) => panic!("bug: StableCell::array: invalid dtype")
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
    unsafe { &*((self as *const StableCell) as *const CellPtr) }
  }

  #[inline]
  pub fn as_ptr_mut(&mut self) -> &mut CellPtr {
    // SAFETY: The following is safe as `StableCell` has the same
    // (transparent) repr as `CellPtr`.
    unsafe { &mut *((self as *mut StableCell) as *mut CellPtr) }
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
}

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

/*pub struct CellSet {
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

impl Drop for CellSet {
  fn drop(&mut self) {
    //ctx_release(self.ptr_._into_cel_ptr());
  }
}

impl From<MCellPtr> for CellSet {
  fn from(ptr: MCellPtr) -> CellSet {
    //ctx_retain(ptr._into_cel_ptr());
    CellSet{ptr_: ptr}
  }
}

impl CellSet {
  #[track_caller]
  pub fn new() -> CellSet {
    panick_wrap(|| {
      ctx_fresh_mset().into()
    })
  }

  pub fn as_ptr(&self) -> MCellPtr {
    self.ptr_
  }

  #[track_caller]
  pub fn add<X: Borrow<CellPtr>>(&self, x: X) {
    panick_wrap(|| TL_CTX.with(|ctx| {
      // FIXME: retain?
      let spine = ctx.spine.borrow();
      spine.add(self.ptr_, *x.borrow());
    }))
  }
}*/

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

impl Drop for CellMap {
  fn drop(&mut self) {
    //ctx_release(self.ptr_._into_cel_ptr());
  }
}

impl From<MCellPtr> for CellMap {
  fn from(ptr: MCellPtr) -> CellMap {
    //ctx_retain(ptr._into_cel_ptr());
    CellMap{ptr_: ptr}
  }
}

impl CellMap {
  #[track_caller]
  pub fn new() -> CellMap {
    panick_wrap(|| {
      ctx_fresh_mmap().into()
    })
  }

  pub fn as_ptr(&self) -> MCellPtr {
    self.ptr_
  }

  #[track_caller]
  pub fn add<K: Borrow<CellPtr>, V: Borrow<CellPtr>>(&self, k: K, v: V) {
    panick_wrap(|| TL_CTX.with(|ctx| {
      let spine = ctx.spine.borrow();
      spine.add2(self.ptr_, *k.borrow(), *v.borrow());
    }))
  }

  #[track_caller]
  pub fn vadd<K: Borrow<CellPtr>, V: Borrow<CellPtr>>(&self, k: &[K], v: &[V]) {
    panick_wrap(|| TL_CTX.with(|ctx| {
      let spine = ctx.spine.borrow();
      for (k, v) in k.iter().zip(v.iter()) {
        spine.add2(self.ptr_, *k.borrow(), *v.borrow());
      }
    }))
  }

  #[track_caller]
  pub fn get(&self, key: CellPtr) -> CellPtr {
    panick_wrap(|| TL_CTX.with(|ctx| {
      let spine = ctx.spine.borrow();
      let kclk = spine._version(key).unwrap_or_else(|| Clock::default());
      spine._get(self.as_ptr(), key, kclk).map(|(v, _)| v).unwrap_or_else(|| CellPtr::nil())
    }))
  }

  #[track_caller]
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
  }
}

pub trait CellDeref {
  fn _deref(&self) -> CellPtr;
}

enum _Void {}

/*const VOID_MAP_SIZE: usize = 0x1_0000_0000;
const VOID_MAP_HI: usize = 0x8000_0000;

thread_local! {
  static TL_HANDLE_TAB: RefCell<CellViewHandleTab> = RefCell::new(CellViewHandleTab::new());
}

struct CellViewHandleTab {
  base: MmapBuf,
  lock: HashSet<CellPtr>,
  read: HashMap<CellPtr, u32>,
}

impl CellViewHandleTab {
  pub fn new() -> CellViewHandleTab {
    assert_eq!(VOID_MAP_SIZE, VOID_MAP_HI << 1);
    let base = MmapBuf::new_noderef(VOID_MAP_SIZE).unwrap();
    let lock = HashSet::new();
    let read = HashMap::new();
    CellViewHandleTab{base, lock, read}
  }
}

#[repr(transparent)]
pub struct CellViewHandle([_Void]);

impl Debug for CellViewHandle {
  fn fmt(&self, f: &mut Formatter) -> FmtResult {
    write!(f, "CellViewHandle({})", self._deref().raw_)
  }
}

impl Drop for CellViewHandle {
  fn drop(&mut self) {
    TL_HANDLE_TAB.with(|tab| {
      let mut tab = tab.borrow_mut();
      let base = tab.base.as_ptr() as usize;
      let val = self.0.as_ptr() as usize;
      let mut raw = val - base;
      if (raw & VOID_MAP_HI) != 0 {
        raw ^= VOID_MAP_HI;
        let x = CellPtr::from_unchecked(raw as i32);
        match tab.lock.remove(&x) {
          false => {
            panic!("ERROR: CellViewHandle::drop: expired mutable borrow");
          }
          true => {}
        }
      } else {
        let x = CellPtr::from_unchecked(raw as i32);
        match tab.read.get_mut(&x) {
          None => {
            panic!("ERROR: CellViewHandle::drop: expired immutable borrow");
          }
          Some(ref_ct) => {
            *ref_ct -= 1;
            if *ref_ct <= 0 {
              assert_eq!(tab.read.remove(&x), Some(0));
            }
          }
        }
      }
    });
  }
}

impl CellViewHandle {
  pub fn _from<'a>(x: CellPtr) -> &'a CellViewHandle {
    let base = TL_HANDLE_TAB.with(|tab| {
      let mut tab = tab.borrow_mut();
      match tab.lock.contains(&x) {
        false => {}
        true => {
          panic!("ERROR: CellViewHandle::_from: existing mutable borrow");
        }
      }
      match tab.read.get_mut(&x) {
        None => {
          tab.read.insert(x, 1);
        }
        Some(ref_ct) => {
          *ref_ct += 1;
        }
      }
      tab.base.as_ptr() as usize
    });
    let raw = x.raw_ as usize;
    assert!(raw < VOID_MAP_HI);
    let val = base + raw;
    unsafe { &*(from_raw_parts(val as *const _Void, 0) as *const [_Void] as *const CellViewHandle) }
  }

  pub fn _from_mut<'a>(x: CellPtr) -> &'a mut CellViewHandle {
    let base = TL_HANDLE_TAB.with(|tab| {
      let mut tab = tab.borrow_mut();
      match tab.read.get(&x) {
        None => {}
        Some(ref_ct) => {
          assert!(*ref_ct > 0);
          panic!("ERROR: CellViewHandle::_from_mut: existing immutable borrow");
        }
      }
      match tab.lock.contains(&x) {
        false => {
          tab.lock.insert(x);
        }
        true => {
          panic!("ERROR: CellViewHandle::_from_mut: double mutable borrow");
        }
      }
      tab.base.as_ptr() as usize
    });
    let raw = x.raw_ as usize;
    assert!(raw < VOID_MAP_HI);
    let val = base + (raw ^ VOID_MAP_HI);
    unsafe { &mut *(from_raw_parts_mut(val as *mut _Void, 0) as *mut [_Void] as *mut CellViewHandle) }
  }

  pub fn materialize(&self) -> CellPtr {
    // FIXME FIXME
    self._deref()
    //unimplemented!();
  }
}

impl CellDeref for CellViewHandle {
  fn _deref(&self) -> CellPtr {
    let base = TL_HANDLE_TAB.with(|tab| {
      let tab = tab.borrow();
      tab.base.as_ptr() as usize
    });
    //println!("DEBUG: CellViewHandle::_deref: base=0x{:016x}", base);
    let val = self.0.as_ptr() as usize;
    //println!("DEBUG: CellViewHandle::_deref: val =0x{:016x}", val);
    let mut raw = val - base;
    //println!("DEBUG: CellViewHandle::_deref: raw =0x{:016x}", raw);
    if (raw & VOID_MAP_HI) != 0 {
      raw ^= VOID_MAP_HI;
    }
    //println!("DEBUG: CellViewHandle::_deref: raw2=0x{:016x}", raw);
    //println!("DEBUG: CellViewHandle::_deref: ptr ={}", raw as i32);
    CellPtr::from_unchecked(raw as i32)
  }
}*/

pub type CellViewHandle = CellViewHandle_;

#[repr(transparent)]
pub struct CellViewHandle_([_Void]);

/*impl Debug for CellViewHandle_ {
  fn fmt(&self, f: &mut Formatter) -> FmtResult {
    write!(f, "CellViewHandle({})", self._deref().raw_)
  }
}*/

/*impl Borrow<CellPtr> for CellViewHandle_ {
  fn borrow(&self) -> &CellPtr {
    unsafe { &*(self.0.as_ptr() as *const CellPtr) }
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

/*#[derive(Clone, Copy)]
#[repr(C)]
pub struct CellViewHandleEx(usize, usize);

impl Debug for CellViewHandleEx {
  fn fmt(&self, f: &mut Formatter) -> FmtResult {
    write!(f, "CellViewHandleEx({})", self._deref().raw_)
  }
}

impl CellViewHandleEx {
  pub fn _from(x: CellPtr) -> CellViewHandleEx {
    let raw = x.raw_ as usize;
    //assert!(raw < VOID_MAP_HI);
    //CellViewHandleEx(raw, 0)
    CellViewHandleEx(0, raw)
  }

  pub fn materialize(&self) -> CellPtr {
    // FIXME FIXME
    self._deref()
    //unimplemented!();
  }
}

impl CellDeref for CellViewHandleEx {
  fn _deref(&self) -> CellPtr {
    //CellPtr::from_unchecked(self.0 as i32)
    CellPtr::from_unchecked(self.1 as i64)
  }
}*/

/*#[derive(Clone, Debug)]
pub struct CellView(pub CellPtr, pub Vec<CellVOp>);

impl From<CellPtr> for CellView {
  fn from(x: CellPtr) -> CellView {
    CellView(x, Vec::new())
  }
}

impl<'a> From<&'a CellPtr> for CellView {
  fn from(x: &'a CellPtr) -> CellView {
    CellView(*x, Vec::new())
  }
}

impl From<StableCell> for CellView {
  fn from(x: StableCell) -> CellView {
    CellView(*x.as_ptr_ref(), Vec::new())
  }
}

impl<'a> From<&'a StableCell> for CellView {
  fn from(x: &'a StableCell) -> CellView {
    CellView(*x.as_ptr_ref(), Vec::new())
  }
}

#[derive(Clone, Copy, Debug)]
pub struct CellViewRef<'a>(pub &'a CellPtr, pub Option<&'a [CellVOp]>);

pub trait BorrowCellView {
  fn _borrow<'a>(&'a self) -> CellViewRef<'a>;
}

impl BorrowCellView for CellPtr {
  fn _borrow<'a>(&'a self) -> CellViewRef<'a> {
    CellViewRef(self, None)
  }
}

impl<'r> BorrowCellView for &'r CellPtr {
  fn _borrow<'a>(&'a self) -> CellViewRef<'a> {
    CellViewRef(*self, None)
  }
}

impl BorrowCellView for StableCell {
  fn _borrow<'a>(&'a self) -> CellViewRef<'a> {
    CellViewRef(self.as_ptr_ref(), None)
  }
}

impl<'r> BorrowCellView for &'r StableCell {
  fn _borrow<'a>(&'a self) -> CellViewRef<'a> {
    CellViewRef(self.as_ptr_ref(), None)
  }
}

impl BorrowCellView for CellView {
  fn _borrow<'a>(&'a self) -> CellViewRef<'a> {
    CellViewRef(&self.0, Some(&self.1))
  }
}

impl<'r> BorrowCellView for &'r CellView {
  fn _borrow<'a>(&'a self) -> CellViewRef<'a> {
    CellViewRef(&self.0, Some(&self.1))
  }
}

impl<'r> BorrowCellView for CellViewRef<'r> {
  fn _borrow<'a>(&'a self) -> CellViewRef<'a> {
    CellViewRef(self.0, self.1)
  }
}*/

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

#[derive(Debug)]
pub struct CellView {
  // TODO
  pub root: CellPtr,
  //pub r_ty: CellType,
  pub vlog: Vec<CellViewOp>,
}

impl Default for CellView {
  fn default() -> CellView {
    CellView{
      root: CellPtr::nil(),
      //r_ty: CellType::top(),
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
  /*pub fn new(root: CellPtr, r_ty: CellType) -> CellView {
    CellView{
      root,
      r_ty,
      vlog: Vec::new(),
    }
  }*/

  pub fn root(&self) -> CellPtr {
    self.root
  }

  pub fn type_eval(&self, root_ty: &CellType) -> Result<CellType, ()> {
    assert!(!root_ty.is_top());
    let mut shape = root_ty.shape.clone();
    let mut dtype = root_ty.dtype;
    if self.vlog.is_empty() {
      return Ok(CellType{shape, dtype});
    }
    let mut offset = Vec::with_capacity(root_ty.shape.len());
    offset.resize(root_ty.shape.len(), 0);
    let mut end_offset = root_ty.shape.clone();
    let mut root_shape = root_ty.shape.clone();
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
    Ok(CellType{shape, dtype})
  }

  pub fn eval_contiguous(&self, root_ty: &CellType) -> Result<CellSliceType, ()> {
    assert!(!root_ty.is_top());
    let mut offset = Vec::with_capacity(root_ty.shape.len());
    offset.resize(root_ty.shape.len(), 0);
    let mut shape = root_ty.shape.clone();
    let mut dtype = root_ty.dtype;
    if self.vlog.is_empty() {
      return Ok(CellSliceType{offset, type_: CellType{shape, dtype}});
    }
    let mut end_offset = root_ty.shape.clone();
    let mut root_shape = root_ty.shape.clone();
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
              assert!(shape[d] + idx[d].start <= idx[d].end);
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
    Ok(CellSliceType{offset, type_: CellType{shape, dtype}})
  }

  pub fn eval_contiguous_transposed(&self, root_ty: &CellType) -> Result<CellSliceType, ()> {
    assert!(!root_ty.is_top());
    unimplemented!();
  }

  pub fn eval_strided(&self, root_ty: &CellType) -> Result<CellStridedSliceType, ()> {
    assert!(!root_ty.is_top());
    unimplemented!();
  }

  pub fn eval_strided_transposed(&self, root_ty: &CellType) -> Result<CellStridedSliceType, ()> {
    assert!(!root_ty.is_top());
    unimplemented!();
  }
}

#[derive(Clone, Copy, Debug)]
#[repr(u8)]
pub enum CellViewStep {
  Break,
  Swap,
  // TODO
  Halt,
}

pub struct CellViewPermState {
  pub ndim: i8,
  pub swap: bool,
  pub perm: [i8; 8],
}

impl CellViewPermState {
  pub fn new(ndim: i8) -> CellViewPermState {
    /*assert!(ndim <= i8::max_value() as u8);*/
    assert!(ndim >= 0);
    CellViewPermState{
      ndim: ndim,
      swap: false,
      perm: [0, 1, 2, 3, 4, 5, 6, 7],
    }
  }

  pub fn _reset_swap(&mut self) {
    self.swap = false;
    self.perm = [0, 1, 2, 3, 4, 5, 6, 7];
  }

  pub fn _swap(&mut self) -> bool {
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
  }

  pub fn _step(&mut self, vop: &CellVOp) -> CellViewStep {
    match vop {
      &CellVOp::Nop => {
        if self._swap() {
          return CellViewStep::Swap;
        }
        // TODO
        return CellViewStep::Halt;
      }
      &CellVOp::Swap(ld, rd) => {
        assert!(ld < self.ndim);
        assert!(ld >= -self.ndim);
        assert!(rd < self.ndim);
        assert!(rd >= -self.ndim);
        let lidx = if ld < 0 { self.ndim + ld } else { ld } as usize;
        let ridx = if rd < 0 { self.ndim + rd } else { rd } as usize;
        self.perm.swap(lidx, ridx);
        self.swap = true;
      }
      _ => {
        if self._swap() {
          return CellViewStep::Swap;
        }
        // TODO
        println!("DEBUG: CellViewState::_step: unimplemented: vop={:?}", vop);
        panic!("bug");
      }
    }
    CellViewStep::Break
  }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
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
      /*Dtype::Fp64 => ScalarVal_::F64(f64::zero().into()),
      Dtype::Fp32 => ScalarVal_::F32(f32::zero().into()),
      Dtype::Fp16 => ScalarVal_::F16(f16::zero().into()),
      Dtype::Int64 => ScalarVal_::I64(i64::zero().into()),
      Dtype::Int32 => ScalarVal_::I32(i32::zero().into()),
      Dtype::Int16 => ScalarVal_::I16(i16::zero().into()),
      Dtype::Int8 => ScalarVal_::I8(i8::zero().into()),
      Dtype::UInt64 => ScalarVal_::U64(u64::zero().into()),
      Dtype::UInt32 => ScalarVal_::U32(u32::zero().into()),
      Dtype::UInt16 => ScalarVal_::U16(u16::zero().into()),
      Dtype::UInt8 => ScalarVal_::U8(u8::zero().into()),*/
      Dtype::Fp64 => ScalarVal_::F64(<f64 as FpConstExt>::zero().into()),
      Dtype::Fp32 => ScalarVal_::F32(<f32 as FpConstExt>::zero().into()),
      Dtype::Fp16 => ScalarVal_::F16(<f16 as FpConstExt>::zero().into()),
      Dtype::Int64 => ScalarVal_::I64(<i64 as UintConstExt>::zero().into()),
      Dtype::Int32 => ScalarVal_::I32(<i32 as UintConstExt>::zero().into()),
      Dtype::Int16 => ScalarVal_::I16(<i16 as UintConstExt>::zero().into()),
      Dtype::Int8 => ScalarVal_::I8(<i8 as UintConstExt>::zero().into()),
      Dtype::UInt64 => ScalarVal_::U64(<u64 as UintConstExt>::zero().into()),
      Dtype::UInt32 => ScalarVal_::U32(<u32 as UintConstExt>::zero().into()),
      Dtype::UInt16 => ScalarVal_::U16(<u16 as UintConstExt>::zero().into()),
      Dtype::UInt8 => ScalarVal_::U8(<u8 as UintConstExt>::zero().into()),
      _ => unimplemented!()
    }
  }

  pub fn one(dtype: Dtype) -> ScalarVal_ {
    match dtype {
      Dtype::Fp64 => ScalarVal_::F64(<f64 as FpConstExt>::one().into()),
      Dtype::Fp32 => ScalarVal_::F32(<f32 as FpConstExt>::one().into()),
      Dtype::Fp16 => ScalarVal_::F16(<f16 as FpConstExt>::one().into()),
      Dtype::Int64 => ScalarVal_::I64(<i64 as UintConstExt>::one().into()),
      Dtype::Int32 => ScalarVal_::I32(<i32 as UintConstExt>::one().into()),
      Dtype::Int16 => ScalarVal_::I16(<i16 as UintConstExt>::one().into()),
      Dtype::Int8 => ScalarVal_::I8(<i8 as UintConstExt>::one().into()),
      Dtype::UInt64 => ScalarVal_::U64(<u64 as UintConstExt>::one().into()),
      Dtype::UInt32 => ScalarVal_::U32(<u32 as UintConstExt>::one().into()),
      Dtype::UInt16 => ScalarVal_::U16(<u16 as UintConstExt>::one().into()),
      Dtype::UInt8 => ScalarVal_::U8(<u8 as UintConstExt>::one().into()),
      _ => unimplemented!()
    }
  }

  pub fn is_bot(self) -> bool {
    match self {
      ScalarVal_::Bot => true,
      _ => false
    }
  }

  pub fn dtype(self) -> Dtype {
    match self {
      ScalarVal_::F64(_) => Dtype::Fp64,
      ScalarVal_::F32(_) => Dtype::Fp32,
      ScalarVal_::F16(_) => Dtype::Fp16,
      ScalarVal_::I64(_) => Dtype::Int64,
      ScalarVal_::I32(_) => Dtype::Int32,
      ScalarVal_::I16(_) => Dtype::Int16,
      ScalarVal_::I8(_) => Dtype::Int8,
      ScalarVal_::U64(_) => Dtype::UInt64,
      ScalarVal_::U32(_) => Dtype::UInt32,
      ScalarVal_::U16(_) => Dtype::UInt16,
      ScalarVal_::U8(_) => Dtype::UInt8,
      ScalarVal_::Bot => Dtype::_Bot,
      _ => unimplemented!()
    }
  }

  pub fn format_futhark(&self) -> SmolStr {
    // FIXME: use the formatter.
    match self {
      ScalarVal_::F64(x) => {
        if x.0.is_infinite() {
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
        if x.0.is_infinite() {
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
        if x.0.to_bits() == 0 {
          "0.0f16".into()
        } else if x.0.to_bits() == 0x3c00 {
          "1.0f16".into()
        } else if x.0.to_bits() == 0x8000 {
          "-0.0f16".into()
        } else if x.0.to_bits() == 0xbc00 {
          "-1.0f16".into()
        } else {
          unimplemented!();
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

pub trait IntoScalarValExt/*: DtypeExt*/ {
  /*type Val: DtypeExt + Copy + Eq + Any;

  fn into_scalar_val(self) -> Self::Val;*/
  fn into_scalar_val_(self) -> ScalarVal_;
}

impl IntoScalarValExt for ScalarVal_ {
  fn into_scalar_val_(self) -> ScalarVal_ {
    self
  }
}

impl IntoScalarValExt for f16 {
  /*type Val = TotalOrd<f16>;

  fn into_scalar_val(self) -> Self::Val {
    self.into()
  }*/

  fn into_scalar_val_(self) -> ScalarVal_ {
    ScalarVal_::F16(self.into())
  }
}

impl IntoScalarValExt for f32 {
  /*type Val = TotalOrd<f32>;

  fn into_scalar_val(self) -> Self::Val {
    self.into()
  }*/

  fn into_scalar_val_(self) -> ScalarVal_ {
    ScalarVal_::F32(self.into())
  }
}

impl IntoScalarValExt for f64 {
  /*type Val = TotalOrd<f64>;

  fn into_scalar_val(self) -> Self::Val {
    self.into()
  }*/

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
#[repr(u8)]
pub enum Dtype {
  _Top,
  Fp64,
  Fp32,
  Fp16,
  Bfloat16,
  Int64,
  Int32,
  Int16,
  Int8,
  UInt64,
  UInt32,
  UInt16,
  UInt8,
  _Bot,
}

impl TryFrom<TorchDtype> for Dtype {
  //type Error = String;
  type Error = SmolStr;

  fn try_from(t: TorchDtype) -> Result<Dtype, SmolStr> {
    Ok(match t {
      TorchDtype::Float64 => Dtype::Fp64,
      TorchDtype::Float32 => Dtype::Fp32,
      TorchDtype::Float16 => Dtype::Fp16,
      TorchDtype::Bfloat16 => Dtype::Bfloat16,
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

impl TryFrom<SmolStr> for Dtype {
  //type Error = String;
  type Error = SmolStr;

  fn try_from(s: SmolStr) -> Result<Dtype, SmolStr> {
    Dtype::from_str(s.as_str())
  }
}

impl TryFrom<String> for Dtype {
  //type Error = String;
  type Error = SmolStr;

  fn try_from(s: String) -> Result<Dtype, SmolStr> {
    Dtype::from_str(s.as_str())
  }
}

impl<'a> TryFrom<&'a str> for Dtype {
  //type Error = String;
  type Error = SmolStr;

  fn try_from(s: &'a str) -> Result<Dtype, SmolStr> {
    Dtype::from_str(s)
  }
}

impl FromStr for Dtype {
  //type Err = String;
  type Err = SmolStr;

  fn from_str(s: &str) -> Result<Dtype, SmolStr> {
    Ok(match s {
      "f64"     |
      "float64" => Dtype::Fp64,
      "f32"     |
      "float32" => Dtype::Fp32,
      "f16"     |
      "float16" => Dtype::Fp16,
      "bf16"    |
      "bfloat16" => Dtype::Bfloat16,
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
      _ => return Err(s.into())
    })
  }
}

impl Dtype {
  pub fn top() -> Dtype {
    Dtype::_Top
  }

  pub fn format_futhark(self) -> &'static str {
    match self {
      Dtype::_Top       => panic!("bug"),
      Dtype::Fp64       => "f64",
      Dtype::Fp32       => "f32",
      Dtype::Fp16       => "f16",
      Dtype::Bfloat16   => unimplemented!(),
      Dtype::Int64      => "i64",
      Dtype::Int32      => "i32",
      Dtype::Int16      => "i16",
      Dtype::Int8       => "i8",
      Dtype::UInt64     => "u64",
      Dtype::UInt32     => "u32",
      Dtype::UInt16     => "u16",
      Dtype::UInt8      => "u8",
      Dtype::_Bot       => panic!("bug"),
    }
  }

  pub fn size_bits(self) -> u64 {
    match self {
      Dtype::_Top       => panic!("bug"),
      Dtype::Fp64       => 64,
      Dtype::Fp32       => 32,
      Dtype::Fp16       => 16,
      Dtype::Bfloat16   => 16,
      Dtype::Int64      => 64,
      Dtype::Int32      => 32,
      Dtype::Int16      => 16,
      Dtype::Int8       => 8,
      Dtype::UInt64     => 64,
      Dtype::UInt32     => 32,
      Dtype::UInt16     => 16,
      Dtype::UInt8      => 8,
      Dtype::_Bot       => panic!("bug"),
    }
  }

  pub fn size_bytes(self) -> usize {
    match self {
      Dtype::_Top       => panic!("bug"),
      Dtype::Fp64       => 8,
      Dtype::Fp32       => 4,
      Dtype::Fp16       => 2,
      Dtype::Bfloat16   => 2,
      Dtype::Int64      => 8,
      Dtype::Int32      => 4,
      Dtype::Int16      => 2,
      Dtype::Int8       => 1,
      Dtype::UInt64     => 8,
      Dtype::UInt32     => 4,
      Dtype::UInt16     => 2,
      Dtype::UInt8      => 1,
      Dtype::_Bot       => panic!("bug"),
    }
  }

  pub fn align_bytes(self) -> usize {
    // FIXME
    self.size_bytes()
  }

  pub fn is_float(self) -> bool {
    match self {
      Dtype::Fp64 |
      Dtype::Fp32 |
      Dtype::Fp16 |
      Dtype::Bfloat16 => true,
      _ => false
    }
  }

  pub fn is_signed_int(self) -> bool {
    match self {
      Dtype::Int64 |
      Dtype::Int32 |
      Dtype::Int16 |
      Dtype::Int8 => true,
      _ => false
    }
  }

  pub fn is_unsigned_int(self) -> bool {
    match self {
      Dtype::UInt64 |
      Dtype::UInt32 |
      Dtype::UInt16 |
      Dtype::UInt8 => true,
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
      (Dtype::Fp32, Dtype::Fp32) |
      (Dtype::Fp32, Dtype::Fp16) |
      (Dtype::Fp32, Dtype::Bfloat16) |
      (Dtype::Fp16, Dtype::Fp32) |
      (Dtype::Bfloat16, Dtype::Fp32) => Some(Dtype::Fp32),
      (Dtype::Fp16, Dtype::Fp16) => Some(Dtype::Fp16),
      (Dtype::Bfloat16, Dtype::Bfloat16) => Some(Dtype::Bfloat16),
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
  // FIXME FIXME
  //fn is_zero(&self) -> bool;
}

pub trait DtypeConstExt {
  fn dtype() -> Dtype where Self: Sized;
}

impl DtypeConstExt for TotalOrd<f64> { fn dtype() -> Dtype { Dtype::Fp64 } }
impl DtypeConstExt for TotalOrd<f32> { fn dtype() -> Dtype { Dtype::Fp32 } }
//impl DtypeConstExt for NonNan<f32>   { fn dtype() -> Dtype { Dtype::Fp32 } }
impl DtypeConstExt for TotalOrd<f16> { fn dtype() -> Dtype { Dtype::Fp16 } }

impl DtypeConstExt for f64 { fn dtype() -> Dtype { Dtype::Fp64 } }
impl DtypeConstExt for f32 { fn dtype() -> Dtype { Dtype::Fp32 } }
impl DtypeConstExt for f16 { fn dtype() -> Dtype { Dtype::Fp16 } }
impl DtypeConstExt for bf16 { fn dtype() -> Dtype { Dtype::Bfloat16 } }
impl DtypeConstExt for i64 { fn dtype() -> Dtype { Dtype::Int64 } }
impl DtypeConstExt for i32 { fn dtype() -> Dtype { Dtype::Int32 } }
impl DtypeConstExt for i16 { fn dtype() -> Dtype { Dtype::Int16 } }
impl DtypeConstExt for i8  { fn dtype() -> Dtype { Dtype::Int8 } }
impl DtypeConstExt for u64 { fn dtype() -> Dtype { Dtype::UInt64 } }
impl DtypeConstExt for u32 { fn dtype() -> Dtype { Dtype::UInt32 } }
impl DtypeConstExt for u16 { fn dtype() -> Dtype { Dtype::UInt16 } }
impl DtypeConstExt for u8  { fn dtype() -> Dtype { Dtype::UInt8 } }

pub fn dtype<T: DtypeConstExt>() -> Dtype {
  T::dtype()
}

pub fn dtype_of<T: DtypeConstExt>(_: T) -> Dtype {
  T::dtype()
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

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
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

  pub fn cast(&self, new_dtype: Dtype) -> CellType {
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
    CellType{shape, dtype: self.dtype}
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

/*// FIXME
pub type CellViewType = CellSliceType;

#[derive(Clone, Debug)]
pub struct CellSliceType {
  pub base:     Vec<i64>,
  pub shape:    Vec<i64>,
  pub oshape:   Vec<i64>,
  pub dtype:    Dtype,
}

impl CellSliceType {
  pub fn is_packed(&self) -> bool {
    let nd = self.shape.len();
    let mut fakestride = Vec::with_capacity(nd);
    let mut origstride = Vec::with_capacity(nd);
    if nd > 1 {
      fakestride.push(1);
      origstride.push(1);
    }
    for d in 1 .. nd {
      let s = self.shape[nd - d];
      assert!(s > 0);
      let og_s = self.oshape[nd - d];
      assert!(og_s > 0);
      fakestride.push(s * fakestride[d - 1]);
      origstride.push(og_s * origstride[d - 1]);
      if fakestride[d] != origstride[d] {
        return false;
      }
    }
    true
  }
}*/

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

/*#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum CellMode {
  _Top,
  Aff,
  Init,
  //Fin,
  //Unsafe,
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
}

#[derive(Clone, Copy, Default, Debug)]
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

  pub fn unset_intro(&mut self) -> bool {
    let prev = self.intro();
    self.bits &= !1;
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
}*/

#[derive(Clone, Default, Debug)]
pub struct CellState {
  //pub mode: CellMode,
  //pub flag: CellFlag,
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
  //pub olay: CellLayout,
  pub pm_index: RevSortMap8<(PMach, Locus), RevSortKey8<(Locus, PMach)>>,
  pub replicas: RevSortMap8<(Locus, PMach), PCellReplica>,
}

impl PCell {
  pub fn new(ptr: CellPtr, ty: CellType) -> PCell {
    if cfg_debug() { println!("DEBUG: PCell::new: optr={:?} ogty={:?}", ptr, &ty); }
    //let lay = CellLayout::new_packed(&ty);
    let pm_index = RevSortMap8::new();
    let replicas = RevSortMap8::new();
    PCell{
      optr: ptr,
      //optr: Cell::new(ptr),
      ogty: ty,
      //olay: lay,
      pm_index,
      replicas,
    }
  }

  pub fn _push(&mut self, clk: Clock, locus: Locus, pmach: PMach, addr: PAddr) {
    let rep = PCellReplica{clk: Cell::new(clk), addr: Cell::new(addr)};
    let key = self.replicas.insert((locus, pmach), rep);
    self.pm_index.insert((pmach, locus), key);
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

  pub fn _pop(&self, x: CellPtr, /*xclk: Clock,*/ q_addr: PAddr) {
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
  }

  pub fn lookup(&self, q_locus: Locus, q_pmach: PMach) -> Option<&PCellReplica> {
    match self.replicas.find((q_locus, q_pmach)) {
      None => None,
      Some((_, rep)) => Some(rep)
    }
  }

  pub fn lookup_loc(&self, q_locus: Locus) -> Option<(PMach, &PCellReplica)> {
    match self.replicas.find_lub((q_locus, PMach::_Bot)) {
      None => None,
      Some((key, rep)) => {
        let &(loc, pm) = key.key.as_ref();
        if loc != q_locus {
          None
        } else {
          //Some((key.1, rep.clk, &mut rep.icel))
          Some((pm, rep))
        }
      }
    }
  }

  pub fn find_any(&self, q_clk: Clock, /*ty: &CellType*/) -> Option<(Locus, PMach, PAddr)> {
    for (key, rep) in self.replicas.iter() {
      let &(loc, pm) = key.as_ref();
      if rep.clk.get() == q_clk {
        return Some((loc, pm, rep.addr.get()));
      }
    }
    None
  }

  pub fn read_loc(&mut self, root: CellPtr, q_clk: Clock, ty: &CellType, q_locus: Locus) -> (PMach, PAddr) {
    let mut f_pmach = match self.lookup_loc(q_locus) {
      None => None,
      Some((pmach, rep)) => {
        let prev_clk = rep.clk.get();
        if prev_clk > q_clk {
          println!("DEBUG: PCell::read_loc: root={:?} prev clk={:?} clk={:?} ty={:?} loc={:?} pm={:?}",
              root, prev_clk, q_clk, ty, q_locus, pmach);
          println!("ERROR: PCell::read_loc: read failure");
          panic!("bug");
        } else if prev_clk == q_clk {
          return (pmach, rep.addr.get());
        }
        Some(pmach)
      }
    };
    if cfg_debug() {
    println!("DEBUG: PCell::get_loc: root={:?} clk={:?} ty={:?} loc={:?} found pm? {:?}",
        root, q_clk, ty, q_locus, f_pmach);
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
      Some(rep) => {
        let prev_clk = rep.clk.get();
        if prev_clk >= q_clk {
          panic!("bug");
        } else /*if prev_clk < q_clk */{
          match self.find_any(q_clk) {
            None => {
              println!("DEBUG: PCell::read_loc: optr={:?} ogty={:?} root={:?} prev clk={:?} clk={:?} ty={:?} loc={:?} pm={:?} addr={:?}",
                  self.optr, &self.ogty,
                  root, prev_clk, q_clk, ty, q_locus, f_pmach, rep.addr.get(),
              );
              println!("ERROR: PCell::read_loc: no replica to copy from");
              panic!();
            }
            Some((o_loc, o_pm, o_addr)) => {
              if cfg_debug() {
              println!("DEBUG: PCell::get_loc: optr={:?} ogty={:?} root={:?} prev clk={:?} clk={:?} ty={:?} loc={:?} pm={:?} addr={:?} found o_loc={:?} o_pm={:?} o_addr={:?}",
                  self.optr, &self.ogty,
                  root, prev_clk, q_clk, ty, q_locus, f_pmach, rep.addr.get(),
                  o_loc, o_pm, o_addr,
              );
              }
              TL_PCTX.with(|pctx| {
                pctx.hard_copy(q_locus, f_pmach, rep.addr.get(), o_loc, o_pm, o_addr, ty.packed_span_bytes() as usize);
              });
            }
          }
          rep.clk.set(q_clk);
        }
        (f_pmach, rep.addr.get())
      }
    }
  }

  pub fn write_loc(&mut self, root: CellPtr, q_clk: Clock, ty: &CellType, q_locus: Locus) -> (PMach, PAddr) {
    let mut f_pmach = match self.lookup_loc(q_locus) {
      None => None,
      Some((pmach, rep)) => {
        let prev_clk = rep.clk.get();
        if prev_clk >= q_clk {
          println!("DEBUG: PCell::write_loc: root={:?} prev clk={:?} clk={:?} ty={:?} loc={:?} pm={:?}",
              root, prev_clk, q_clk, ty, q_locus, pmach);
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
    let f_pmach = f_pmach.unwrap();
    match self.lookup(q_locus, f_pmach) {
      None => panic!("bug"),
      Some(rep) => {
        let prev_clk = rep.clk.get();
        if prev_clk >= q_clk {
          panic!("bug");
        } else /*if prev_clk < q_clk */{
          rep.clk.set(q_clk);
        }
        (f_pmach, rep.addr.get())
      }
    }
  }

  pub fn hardcopy(&self) -> PCell {
    // FIXME FIXME
    unimplemented!();
  }
}

pub trait InnerCell {
  // TODO
  //fn try_borrow(&self) -> () { unimplemented!(); }
  //fn try_borrow_mut(&self) -> () { unimplemented!(); }
  fn as_mem_reg(&self) -> Option<MemReg> { None }
  //fn as_reg(&self) -> Option<MemReg> { self.as_mem_reg() }
  fn size(&self) -> usize { unimplemented!(); }
  fn root(&self) -> Option<CellPtr> { unimplemented!(); }
  fn set_root(&self, _root: Option<CellPtr>) { unimplemented!(); }
  fn pin(&self) -> bool { unimplemented!(); }
  fn set_pin(&self, _flag: bool) { unimplemented!(); }
  fn tag(&self) -> Option<u32> { unimplemented!(); }
  fn set_tag(&self, _tag: Option<u32>) { unimplemented!(); }
}

pub trait InnerCell_ {
  fn as_any(&self) -> &dyn Any;
  // TODO
  fn as_mem_reg(&self) -> Option<MemReg>;
  //fn as_reg(&self) -> Option<MemReg> { self.as_mem_reg() }
  fn size(&self) -> usize;
  fn root(&self) -> Option<CellPtr>;
  fn set_root(&self, _root: Option<CellPtr>);
  fn pin(&self) -> bool;
  fn set_pin(&self, _flag: bool);
  fn tag(&self) -> Option<u32>;
  fn set_tag(&self, _tag: Option<u32>);
}

impl<C: InnerCell + Any> InnerCell_ for C {
  fn as_any(&self) -> &dyn Any {
    self
  }

  fn as_mem_reg(&self) -> Option<MemReg> {
    InnerCell::as_mem_reg(self)
  }

  fn size(&self) -> usize {
    InnerCell::size(self)
  }

  fn root(&self) -> Option<CellPtr> {
    InnerCell::root(self)
  }

  fn set_root(&self, root: Option<CellPtr>) {
    InnerCell::set_root(self, root)
  }

  fn pin(&self) -> bool {
    InnerCell::pin(self)
  }

  fn set_pin(&self, flag: bool) {
    InnerCell::set_pin(self, flag)
  }

  fn tag(&self) -> Option<u32> {
    InnerCell::tag(self)
  }

  fn set_tag(&self, tag: Option<u32>) {
    InnerCell::set_tag(self, tag)
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

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
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

#[derive(Clone, Debug)]
pub struct MCellSetEntry {
  pub item: MValue,
  pub clk:  Clock,
  pub rev_: Cell<bool>,
}

#[derive(Clone, Debug)]
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

#[derive(Clone, Debug)]
pub struct MCellMapEntry {
  pub key:  MValue,
  pub val:  MValue,
  pub kclk: Clock,
  pub vclk: Clock,
  pub rev_: Cell<bool>,
}

#[derive(Clone, Debug)]
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
