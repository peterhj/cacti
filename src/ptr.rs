pub use crate::cell::{CellPtr};
use crate::cell::{CellType, Dtype};
use crate::ctx::*;
use crate::thunk::ops::*;

use std::convert::{TryInto};
use std::ops::{Deref, Add, Sub, Mul, Div};

/*impl<P: Into<CellPtr> + Sized> Add<P> for f32 {
  type Output = CellPtr;

  fn add(self, rhs: P) -> CellPtr {
    unimplemented!();
    /*
    let op = AddScalarF32ThunkOp{scalar: self};
    assert!(ctx_clean_arg());
    ctx_push_cell_arg(rhs.into());
    //ctx_push_cell_out(_);
    ctx_pop_thunk(op)
    */
  }
}*/

/*impl<P: Into<CellPtr> + Sized> Add<f32> for P {
  type Output = CellPtr;

  fn add(self, rhs: f32) -> CellPtr {
    unimplemented!();
    /*
    let op = AddScalarF32ThunkOp{scalar: rhs};
    assert!(ctx_clean_arg());
    ctx_push_cell_arg(self.into());
    //ctx_push_cell_out(_);
    ctx_pop_thunk(op)
    */
  }
}*/

impl<'p> Add<f32> for &'p CellPtr {
  type Output = CellPtr;

  fn add(self, rhs: f32) -> CellPtr {
    let op = AddScalarF32ThunkOp{scalar: rhs.try_into().unwrap()};
    assert!(ctx_clean_arg());
    ctx_push_cell_arg(self.into());
    /*ctx_push_cell_tmp_out();*/
    ctx_pop_thunk_(op)
  }
}

/*impl<P: Into<CellPtr> + Sized, Q: Into<CellPtr> + Sized> Add<Q> for P {
  type Output = CellPtr;

  fn add(self, rhs: Q) -> CellPtr {
    unimplemented!();
    /*
    let p = self.into();
    let q = rhs.into();
    let op = AddThunkOp::default();
    assert!(ctx_clean_arg());
    ctx_push_cell_arg(p);
    ctx_push_cell_arg(q);
    //ctx_push_cell_out(_);
    ctx_pop_thunk(op)
    */
  }
}*/

impl<'p, Q: Into<CellPtr> + Sized> Add<Q> for &'p CellPtr {
  type Output = CellPtr;

  fn add(self, rhs: Q) -> CellPtr {
    unimplemented!();
    /*
    let p = self.into();
    let q = rhs.into();
    let op = AddThunkOp::default();
    assert!(ctx_clean_arg());
    ctx_push_cell_arg(p);
    ctx_push_cell_arg(q);
    //ctx_push_cell_out(_);
    ctx_pop_thunk(op)
    */
  }
}

/*impl<P: Into<CellPtr> + Sized> Mul<P> for f32 {
  type Output = CellPtr;

  fn mul(self, rhs: P) -> CellPtr {
    unimplemented!();
    /*
    let op = MulScalarF32ThunkOp{scalar: self};
    assert!(ctx_clean_arg());
    ctx_push_cell_arg(rhs.into());
    //ctx_push_cell_out(_);
    ctx_pop_thunk(op)
    */
  }
}*/

/*impl<P: Into<CellPtr> + Sized> Mul<f32> for P {
  type Output = CellPtr;

  fn mul(self, rhs: f32) -> CellPtr {
    unimplemented!();
    /*
    let op = MulScalarF32ThunkOp{scalar: rhs};
    assert!(ctx_clean_arg());
    ctx_push_cell_arg(self.into());
    //ctx_push_cell_out(_);
    ctx_pop_thunk(op)
    */
  }
}*/

impl<'p> Mul<f32> for &'p CellPtr {
  type Output = CellPtr;

  fn mul(self, rhs: f32) -> CellPtr {
    unimplemented!();
    /*
    let op = MulScalarF32ThunkOp{scalar: rhs};
    assert!(ctx_clean_arg());
    ctx_push_cell_arg(self.into());
    //ctx_push_cell_out(_);
    ctx_pop_thunk(op)
    */
  }
}

/*impl<P: Into<CellPtr> + Sized> Div<f32> for P {
  type Output = CellPtr;

  fn div(self, rhs: f32) -> CellPtr {
    unimplemented!();
    /*
    let op = DivScalarF32ThunkOp{scalar: rhs};
    assert!(ctx_clean_arg());
    ctx_push_cell_arg(self.into());
    //ctx_push_cell_out(_);
    ctx_pop_thunk(op)
    */
  }
}*/

impl<'p> Div<f32> for &'p CellPtr {
  type Output = CellPtr;

  fn div(self, rhs: f32) -> CellPtr {
    unimplemented!();
    /*
    let op = DivScalarF32ThunkOp{scalar: rhs};
    assert!(ctx_clean_arg());
    ctx_push_cell_arg(self.into());
    //ctx_push_cell_out(_);
    ctx_pop_thunk(op)
    */
  }
}

/*impl<P: Into<CellPtr> + Sized, Q: Into<CellPtr> + Sized> Div<Q> for P {
  type Output = CellPtr;

  fn div(self, rhs: Q) -> CellPtr {
    unimplemented!();
    /*
    let p = self.into();
    let q = rhs.into();
    let op = DivThunkOp::default();
    assert!(ctx_clean_arg());
    ctx_push_cell_arg(p);
    ctx_push_cell_arg(q);
    //ctx_push_cell_out(_);
    ctx_pop_thunk(op)
    */
  }
}*/

impl<'p, Q: Into<CellPtr> + Sized> Div<Q> for &'p CellPtr {
  type Output = CellPtr;

  fn div(self, rhs: Q) -> CellPtr {
    unimplemented!();
    /*
    let p = self.into();
    let q = rhs.into();
    let op = DivThunkOp::default();
    assert!(ctx_clean_arg());
    ctx_push_cell_arg(p);
    ctx_push_cell_arg(q);
    //ctx_push_cell_out(_);
    ctx_pop_thunk(op)
    */
  }
}

pub trait MathOps: Into<CellPtr> + Sized {
  fn cast(self, dtype: Dtype) -> CellPtr {
    unimplemented!();
  }

  /*fn upcast_f32(self) -> CellPtr {
    unimplemented!();
  }

  fn downcast_f16(self) -> CellPtr {
    unimplemented!();
  }*/

  fn sqrt(self) -> CellPtr {
    unimplemented!();
    /*
    let op = SqrtThunkOp::default();
    assert!(ctx_clean_arg());
    ctx_push_cell_arg(self.into());
    //ctx_push_cell_out(_);
    ctx_pop_thunk(op)
    */
  }

  fn rsqrt(self) -> CellPtr {
    unimplemented!();
    /*
    let op = RsqrtThunkOp::default();
    assert!(ctx_clean_arg());
    ctx_push_cell_arg(self.into());
    //ctx_push_cell_out(_);
    ctx_pop_thunk(op)
    */
  }

  fn powi(self, exp: i64) -> CellPtr {
    unimplemented!();
    /*
    let op = PowiThunkOp{exp};
    assert!(ctx_clean_arg());
    ctx_push_cell_arg(self.into());
    //ctx_push_cell_out(_);
    ctx_pop_thunk(op)
    */
  }

  fn inner_max(self) -> CellPtr {
    unimplemented!();
    /*
    let p = self.into();
    let ty = ctx_lookup_type(p);
    match ty.shape.len() {
      3 => {
        let op = InnerMax3dThunkOp::default();
        assert!(ctx_clean_arg());
        ctx_push_cell_arg(p);
        //ctx_push_cell_out(_);
        ctx_pop_thunk(op)
      }
      _ => unimplemented!()
    }
    */
  }

  fn inner_mean(self) -> CellPtr {
    unimplemented!();
    /*
    let p = self.into();
    let ty = ctx_lookup_type(p);
    match ty.shape.len() {
      3 => {
        let op = InnerMean3dThunkOp::default();
        assert!(ctx_clean_arg());
        ctx_push_cell_arg(p);
        //ctx_push_cell_out(_);
        ctx_pop_thunk(op)
      }
      _ => unimplemented!()
    }
    */
  }

  fn inner_sum(self) -> CellPtr {
    unimplemented!();
    /*
    let p = self.into();
    let ty = ctx_lookup_type(p);
    match ty.shape.len() {
      3 => {
        let op = InnerSum3dThunkOp::default();
        assert!(ctx_clean_arg());
        ctx_push_cell_arg(p);
        //ctx_push_cell_out(_);
        ctx_pop_thunk(op)
      }
      _ => unimplemented!()
    }
    */
  }

  /*fn mean_1d(self, dim: i64) -> CellPtr {
    unimplemented!();
    /*
    let op = Mean1dThunkOp{dim};
    assert!(ctx_clean_arg());
    ctx_push_cell_arg(self.into());
    //ctx_push_cell_out(_);
    ctx_pop_thunk(op)
    */
  }

  fn sum_1d(self, dim: i64) -> CellPtr {
    unimplemented!();
    /*
    let op = Sum1dThunkOp{dim};
    assert!(ctx_clean_arg());
    ctx_push_cell_arg(self.into());
    //ctx_push_cell_out(_);
    ctx_pop_thunk(op)
    */
  }*/

  fn dot<Q: Into<CellPtr> + Sized>(self, rhs: Q) -> CellPtr {
    unimplemented!();
    /*
    let p = self.into();
    let q = rhs.into();
    let op = DotThunkOp::default();
    assert!(ctx_clean_arg());
    ctx_push_cell_arg(p);
    ctx_push_cell_arg(q);
    //ctx_push_cell_out(_);
    ctx_pop_thunk(op)
    */
  }
}

impl<P: Into<CellPtr> + Sized> MathOps for P {}

pub trait GradOps<Q: Into<CellPtr> + Sized>: Into<CellPtr> + Sized {
  fn gradl(self, tg: Q) -> CellPtr {
    ctx_lookup_gradl(self.into(), tg.into())
  }

  fn gradr(self, x: Q) -> CellPtr {
    ctx_lookup_gradr(self.into(), x.into())
  }
}

/*impl<P: Into<CellPtr> + Sized, Q: Into<CellPtr> + Sized> GradOps<Q> for P {
  fn gradl(self, tg: Q) -> CellPtr {
    ctx_lookup_gradl(self.into(), tg.into())
  }

  fn gradr(self, x: Q) -> CellPtr {
    ctx_lookup_gradr(self.into(), x.into())
  }
}*/

impl<P: Into<CellPtr> + Sized, Q: Into<CellPtr> + Sized> GradOps<Q> for P {}

pub trait ArrayOps: Into<CellPtr> + Sized {
  /*
  fn dtype(&self) -> Dtype;
  fn shape(&self) -> Vec<i64>;
  fn bit_alias(self, new_dtype: Dtype) -> Self;
  fn new_shape(self, new_shape: Vec<i64>) -> Self;
  fn reshape(self, new_shape: Vec<i64>) -> Self { self.new_shape(new_shape) }
  */

  fn type_(self) -> CellType {
    ctx_lookup_type(self.into())
  }

  /*fn dtype(self) -> Dtype {
    unimplemented!();
  }

  fn shape(self) -> Vec<i64> {
    unimplemented!();
  }*/

  fn bit_alias(self, new_dtype: Dtype) -> CellPtr {
    ctx_alias_bits(self.into(), new_dtype)
  }

  fn new_shape(self, new_shape: Vec<i64>) -> CellPtr {
    ctx_alias_new_shape(self.into(), new_shape)
  }

  fn reshape(self, new_shape: Vec<i64>) -> CellPtr { self.new_shape(new_shape) }
}

impl<P: Into<CellPtr> + Sized> ArrayOps for P {}

pub trait Ops: Into<CellPtr> + Sized {
  fn tag(self, /*_: ???*/) -> Self {
    unimplemented!();
  }

  fn profile(self) -> Self {
    unimplemented!();
  }

  fn trace(self) -> Self {
    unimplemented!();
  }

  fn break_(self) -> Self {
    unimplemented!();
  }

  fn opaque(self) -> Self {
    unimplemented!();
  }

  fn bar(self) -> Self {
    unimplemented!();
  }

  fn set_futhark(self, fut_str: &str) -> Self {
    unimplemented!();
  }

  //fn set(self, set_f: Box<FnOnce() -> _>) -> Self;
  //fn set_in_place(self, set_f: Box<FnOnce(_)>) -> Self;

  fn init_futhark(self, fut_str: &str) -> Self {
    unimplemented!();
  }

  //fn init(self, init_f: Box<FnOnce() -> _>) -> Self;

  fn apply_futhark(self, fut_str: &str) -> Self {
    unimplemented!();
  }

  //fn apply_fut(self, fut_str: &[u8]) -> Self { self.apply_futhark(fut_str) }

  fn cache(self) -> Self {
    unimplemented!();
  }

  fn init_cache(self) -> Self {
    unimplemented!();
  }

  fn cache_init(self) -> Self { self.init_cache() }

  /*fn cache_init_futhark(self, fut_str: &str) -> Self {
    // TODO: ???
    unimplemented!();
  }*/

  fn unseal_init(self) -> Self {
    unimplemented!();
  }

  fn eval(self) -> Self {
    unimplemented!();
  }
}

impl<P: Into<CellPtr> + Sized> Ops for P {}

/*pub trait MaybeStableOps: Sized {
  // FIXME FIXME
  fn is_stable(self) -> bool;
  fn maybe_eval(self) -> Option<StablePtr>;
  fn eval(self) -> StablePtr;
}

pub trait StableOps: Sized {
  // FIXME FIXME
  fn cache(self) -> Self;
  fn cache_init_futhark(self, fut_str: &str) -> Self;
}*/

/*#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct UnstablePtr(i32);

impl Deref for UnstablePtr {
  type Target = CellPtr;

  #[inline(always)]
  fn deref<'a>(&'a self) -> &'a CellPtr {
    unsafe { &*((self as *const UnstablePtr) as *const CellPtr) as &CellPtr }
  }
}

impl UnstablePtr {
  pub fn from_cell_unchecked(x: CellPtr) -> UnstablePtr {
    UnstablePtr(x.to_unchecked())
  }

  pub fn from_unchecked(p: i32) -> UnstablePtr {
    UnstablePtr(p)
  }

  pub fn to_unchecked(&self) -> i32 {
    self.0
  }
}*/

/*impl Ops for UnstablePtr {
  fn dtype(&self) -> Dtype {
    unimplemented!();
  }

  fn shape(&self) -> Vec<i64> {
    unimplemented!();
  }

  fn bit_alias(self, new_dtype: Dtype) -> UnstablePtr {
    unimplemented!();
  }

  fn new_shape(self, new_shape: Vec<i64>) -> UnstablePtr {
    unimplemented!();
  }

  fn tag(self, /*_: ???*/) -> UnstablePtr {
    unimplemented!();
  }

  fn profile(self) -> UnstablePtr {
    unimplemented!();
  }

  fn trace(self) -> UnstablePtr {
    unimplemented!();
  }

  fn break_(self) -> UnstablePtr {
    unimplemented!();
  }

  fn opaque(self) -> UnstablePtr {
    unimplemented!();
  }

  fn bar(self) -> UnstablePtr {
    unimplemented!();
  }

  fn set_futhark(self, fut_str: &str) -> UnstablePtr {
    unimplemented!();
  }

  fn init_futhark(self, fut_str: &str) -> UnstablePtr {
    unimplemented!();
  }

  fn apply_futhark(self, fut_str: &str) -> UnstablePtr {
    unimplemented!();
  }
}*/

/*impl MaybeStableOps for UnstablePtr {
  fn is_stable(self) -> bool {
    ctx_lookup_flag((*self).into()).eval()
  }

  fn maybe_eval(self) -> Option<StablePtr> {
    let p = self.into();
    if ctx_lookup_flag(p).eval() {
      Some(StablePtr::from_cell_unchecked(p))
    } else {
      None
    }
  }

  fn eval(self) -> StablePtr {
    let p = self.into();
    ctx_set_eval(p);
    StablePtr::from_cell_unchecked(p)
  }
}*/

/*#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct StablePtr(i32);

impl Deref for StablePtr {
  type Target = CellPtr;

  #[inline(always)]
  fn deref<'a>(&'a self) -> &'a CellPtr {
    unsafe { &*((self as *const StablePtr) as *const CellPtr) as &CellPtr }
  }
}

impl StablePtr {
  pub fn from_cell_unchecked(x: CellPtr) -> StablePtr {
    StablePtr(x.to_unchecked())
  }

  pub fn from_unchecked(p: i32) -> StablePtr {
    StablePtr(p)
  }

  pub fn to_unchecked(&self) -> i32 {
    self.0
  }
}*/

/*impl Ops for StablePtr {
  fn dtype(&self) -> Dtype {
    unimplemented!();
  }

  fn shape(&self) -> Vec<i64> {
    unimplemented!();
  }

  fn bit_alias(self, new_dtype: Dtype) -> StablePtr {
    unimplemented!();
  }

  fn new_shape(self, new_shape: Vec<i64>) -> StablePtr {
    unimplemented!();
  }

  fn tag(self, /*_: ???*/) -> StablePtr {
    unimplemented!();
  }

  fn profile(self) -> StablePtr {
    unimplemented!();
  }

  fn trace(self) -> StablePtr {
    unimplemented!();
  }

  fn break_(self) -> StablePtr {
    unimplemented!();
  }

  fn opaque(self) -> StablePtr {
    unimplemented!();
  }

  fn bar(self) -> StablePtr {
    unimplemented!();
  }

  fn set_futhark(self, fut_str: &str) -> StablePtr {
    unimplemented!();
  }

  fn init_futhark(self, fut_str: &str) -> StablePtr {
    unimplemented!();
  }

  fn apply_futhark(self, fut_str: &str) -> StablePtr {
    unimplemented!();
  }
}*/

/*impl MaybeStableOps for StablePtr {
  fn is_stable(self) -> bool {
    if !ctx_lookup_flag((*self).into()).eval() {
      panic!("bug");
    }
    true
  }

  fn maybe_eval(self) -> Option<StablePtr> {
    if !ctx_lookup_flag(self.into()).eval() {
      panic!("bug");
    }
    Some(self)
  }

  fn eval(self) -> StablePtr {
    if !ctx_lookup_flag(self.into()).eval() {
      panic!("bug");
    }
    self
  }
}

impl StableOps for StablePtr {
  fn cache(self) -> StablePtr {
    unimplemented!();
  }

  fn cache_init_futhark(self, fut_str: &str) -> StablePtr {
    unimplemented!();
  }
}

impl StablePtr {
  pub fn retain(self) -> StablePtr {
    unimplemented!();
  }
}*/
