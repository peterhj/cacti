use crate::cell::{Dtype};
//use crate::thunk::ops::*;

use std::mem::{transmute};
use std::ops::{Deref, Add, Sub, Mul, Div};

/*impl<P: Into<CellPtr> + Sized> Add<P> for f32 {
  type Output = UnstablePtr;

  fn add(self, rhs: P) -> UnstablePtr {
    unimplemented!();
    /*
    let op = AddScalarF32ThunkOp{scalar: self};
    assert!(tl_ctx_clean_arg());
    tl_ctx_push_cell_arg(rhs.into());
    //tl_ctx_push_cell_out(_);
    tl_ctx_pop_thunk(op)
    */
  }
}*/

/*impl<P: Into<CellPtr> + Sized> Add<f32> for P {
  type Output = UnstablePtr;

  fn add(self, rhs: f32) -> UnstablePtr {
    unimplemented!();
    /*
    let op = AddScalarF32ThunkOp{scalar: rhs};
    assert!(tl_ctx_clean_arg());
    tl_ctx_push_cell_arg(self.into());
    //tl_ctx_push_cell_out(_);
    tl_ctx_pop_thunk(op)
    */
  }
}*/

impl<'p> Add<f32> for &'p CellPtr {
  type Output = UnstablePtr;

  fn add(self, rhs: f32) -> UnstablePtr {
    unimplemented!();
    /*
    let op = AddScalarF32ThunkOp{scalar: rhs};
    assert!(tl_ctx_clean_arg());
    tl_ctx_push_cell_arg(self.into());
    //tl_ctx_push_cell_out(_);
    tl_ctx_pop_thunk(op)
    */
  }
}

/*impl<P: Into<CellPtr> + Sized, Q: Into<CellPtr> + Sized> Add<Q> for P {
  type Output = UnstablePtr;

  fn add(self, rhs: Q) -> UnstablePtr {
    unimplemented!();
    /*
    let p = self.into();
    let q = rhs.into();
    let op = AddThunkOp::default();
    assert!(tl_ctx_clean_arg());
    tl_ctx_push_cell_arg(p);
    tl_ctx_push_cell_arg(q);
    //tl_ctx_push_cell_out(_);
    tl_ctx_pop_thunk(op)
    */
  }
}*/

impl<'p, Q: Into<CellPtr> + Sized> Add<Q> for &'p CellPtr {
  type Output = UnstablePtr;

  fn add(self, rhs: Q) -> UnstablePtr {
    unimplemented!();
    /*
    let p = self.into();
    let q = rhs.into();
    let op = AddThunkOp::default();
    assert!(tl_ctx_clean_arg());
    tl_ctx_push_cell_arg(p);
    tl_ctx_push_cell_arg(q);
    //tl_ctx_push_cell_out(_);
    tl_ctx_pop_thunk(op)
    */
  }
}

/*impl<P: Into<CellPtr> + Sized> Mul<P> for f32 {
  type Output = UnstablePtr;

  fn mul(self, rhs: P) -> UnstablePtr {
    unimplemented!();
    /*
    let op = MulScalarF32ThunkOp{scalar: self};
    assert!(tl_ctx_clean_arg());
    tl_ctx_push_cell_arg(rhs.into());
    //tl_ctx_push_cell_out(_);
    tl_ctx_pop_thunk(op)
    */
  }
}*/

/*impl<P: Into<CellPtr> + Sized> Mul<f32> for P {
  type Output = UnstablePtr;

  fn mul(self, rhs: f32) -> UnstablePtr {
    unimplemented!();
    /*
    let op = MulScalarF32ThunkOp{scalar: rhs};
    assert!(tl_ctx_clean_arg());
    tl_ctx_push_cell_arg(self.into());
    //tl_ctx_push_cell_out(_);
    tl_ctx_pop_thunk(op)
    */
  }
}*/

impl<'p> Mul<f32> for &'p CellPtr {
  type Output = UnstablePtr;

  fn mul(self, rhs: f32) -> UnstablePtr {
    unimplemented!();
    /*
    let op = MulScalarF32ThunkOp{scalar: rhs};
    assert!(tl_ctx_clean_arg());
    tl_ctx_push_cell_arg(self.into());
    //tl_ctx_push_cell_out(_);
    tl_ctx_pop_thunk(op)
    */
  }
}

/*impl<P: Into<CellPtr> + Sized> Div<f32> for P {
  type Output = UnstablePtr;

  fn div(self, rhs: f32) -> UnstablePtr {
    unimplemented!();
    /*
    let op = DivScalarF32ThunkOp{scalar: rhs};
    assert!(tl_ctx_clean_arg());
    tl_ctx_push_cell_arg(self.into());
    //tl_ctx_push_cell_out(_);
    tl_ctx_pop_thunk(op)
    */
  }
}*/

impl<'p> Div<f32> for &'p CellPtr {
  type Output = UnstablePtr;

  fn div(self, rhs: f32) -> UnstablePtr {
    unimplemented!();
    /*
    let op = DivScalarF32ThunkOp{scalar: rhs};
    assert!(tl_ctx_clean_arg());
    tl_ctx_push_cell_arg(self.into());
    //tl_ctx_push_cell_out(_);
    tl_ctx_pop_thunk(op)
    */
  }
}

/*impl<P: Into<CellPtr> + Sized, Q: Into<CellPtr> + Sized> Div<Q> for P {
  type Output = UnstablePtr;

  fn div(self, rhs: Q) -> UnstablePtr {
    unimplemented!();
    /*
    let p = self.into();
    let q = rhs.into();
    let op = DivThunkOp::default();
    assert!(tl_ctx_clean_arg());
    tl_ctx_push_cell_arg(p);
    tl_ctx_push_cell_arg(q);
    //tl_ctx_push_cell_out(_);
    tl_ctx_pop_thunk(op)
    */
  }
}*/

impl<'p, Q: Into<CellPtr> + Sized> Div<Q> for &'p CellPtr {
  type Output = UnstablePtr;

  fn div(self, rhs: Q) -> UnstablePtr {
    unimplemented!();
    /*
    let p = self.into();
    let q = rhs.into();
    let op = DivThunkOp::default();
    assert!(tl_ctx_clean_arg());
    tl_ctx_push_cell_arg(p);
    tl_ctx_push_cell_arg(q);
    //tl_ctx_push_cell_out(_);
    tl_ctx_pop_thunk(op)
    */
  }
}

pub trait CellPtrOps: Into<CellPtr> + Sized {
  fn cast(self, dtype: Dtype) -> UnstablePtr {
    unimplemented!();
  }

  /*fn upcast_f32(self) -> UnstablePtr {
    unimplemented!();
  }

  fn downcast_f16(self) -> UnstablePtr {
    unimplemented!();
  }*/

  fn sqrt(self) -> UnstablePtr {
    unimplemented!();
    /*
    let op = SqrtThunkOp::default();
    assert!(tl_ctx_clean_arg());
    tl_ctx_push_cell_arg(self.into());
    //tl_ctx_push_cell_out(_);
    tl_ctx_pop_thunk(op)
    */
  }

  fn rsqrt(self) -> UnstablePtr {
    unimplemented!();
    /*
    let op = RsqrtThunkOp::default();
    assert!(tl_ctx_clean_arg());
    tl_ctx_push_cell_arg(self.into());
    //tl_ctx_push_cell_out(_);
    tl_ctx_pop_thunk(op)
    */
  }

  fn powi(self, exp: i64) -> UnstablePtr {
    unimplemented!();
    /*
    let op = PowiThunkOp{exp};
    assert!(tl_ctx_clean_arg());
    tl_ctx_push_cell_arg(self.into());
    //tl_ctx_push_cell_out(_);
    tl_ctx_pop_thunk(op)
    */
  }

  fn inner_max(self) -> UnstablePtr {
    unimplemented!();
    /*
    let p = self.into();
    let ty = tl_ctx_lookup_type(p);
    match ty.shape.len() {
      3 => {
        let op = InnerMax3dThunkOp::default();
        assert!(tl_ctx_clean_arg());
        tl_ctx_push_cell_arg(p);
        //tl_ctx_push_cell_out(_);
        tl_ctx_pop_thunk(op)
      }
      _ => unimplemented!()
    }
    */
  }

  fn inner_mean(self) -> UnstablePtr {
    unimplemented!();
    /*
    let p = self.into();
    let ty = tl_ctx_lookup_type(p);
    match ty.shape.len() {
      3 => {
        let op = InnerMean3dThunkOp::default();
        assert!(tl_ctx_clean_arg());
        tl_ctx_push_cell_arg(p);
        //tl_ctx_push_cell_out(_);
        tl_ctx_pop_thunk(op)
      }
      _ => unimplemented!()
    }
    */
  }

  fn inner_sum(self) -> UnstablePtr {
    unimplemented!();
    /*
    let p = self.into();
    let ty = tl_ctx_lookup_type(p);
    match ty.shape.len() {
      3 => {
        let op = InnerSum3dThunkOp::default();
        assert!(tl_ctx_clean_arg());
        tl_ctx_push_cell_arg(p);
        //tl_ctx_push_cell_out(_);
        tl_ctx_pop_thunk(op)
      }
      _ => unimplemented!()
    }
    */
  }

  /*fn mean_1d(self, dim: i64) -> UnstablePtr {
    unimplemented!();
    /*
    let op = Mean1dThunkOp{dim};
    assert!(tl_ctx_clean_arg());
    tl_ctx_push_cell_arg(self.into());
    //tl_ctx_push_cell_out(_);
    tl_ctx_pop_thunk(op)
    */
  }

  fn sum_1d(self, dim: i64) -> UnstablePtr {
    unimplemented!();
    /*
    let op = Sum1dThunkOp{dim};
    assert!(tl_ctx_clean_arg());
    tl_ctx_push_cell_arg(self.into());
    //tl_ctx_push_cell_out(_);
    tl_ctx_pop_thunk(op)
    */
  }*/

  fn dot<Q: Into<CellPtr> + Sized>(self, rhs: Q) -> UnstablePtr {
    unimplemented!();
    /*
    let p = self.into();
    let q = rhs.into();
    let op = DotThunkOp::default();
    assert!(tl_ctx_clean_arg());
    tl_ctx_push_cell_arg(p);
    tl_ctx_push_cell_arg(q);
    //tl_ctx_push_cell_out(_);
    tl_ctx_pop_thunk(op)
    */
  }
}

//impl CellPtrOps for UnstablePtr {}
//impl CellPtrOps for StablePtr {}
impl<P: Into<CellPtr> + Sized> CellPtrOps for P {}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct CellPtr(i32);

impl From<UnstablePtr> for CellPtr {
  #[inline(always)]
  fn from(x: UnstablePtr) -> CellPtr {
    CellPtr(x.0)
  }
}

impl Deref for UnstablePtr {
  type Target = CellPtr;

  #[inline(always)]
  fn deref(&self) -> &CellPtr {
    unsafe { transmute((self as *const UnstablePtr) as *const CellPtr) }
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct UnstablePtr(i32);

impl UnstablePtr {
  pub fn from_unchecked(p: i32) -> UnstablePtr {
    UnstablePtr(p)
  }

  pub fn stabilize(self) -> StablePtr {
    unimplemented!();
  }

  pub fn eval(self) -> StablePtr {
    unimplemented!();
  }

  pub fn dtype(self) -> Dtype {
    unimplemented!();
  }

  pub fn shape(self) -> Vec<i64> {
    unimplemented!();
  }

  pub fn bit_alias(self, new_dtype: Dtype) -> UnstablePtr {
    unimplemented!();
  }

  pub fn new_shape(self, new_shape: Vec<i64>) -> UnstablePtr {
    unimplemented!();
  }

  pub fn reshape(self, new_shape: Vec<i64>) -> UnstablePtr {
    self.new_shape(new_shape)
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct StablePtr(i32);

impl From<StablePtr> for CellPtr {
  #[inline(always)]
  fn from(x: StablePtr) -> CellPtr {
    CellPtr(x.0)
  }
}

impl Deref for StablePtr {
  type Target = CellPtr;

  #[inline(always)]
  fn deref(&self) -> &CellPtr {
    unsafe { transmute((self as *const StablePtr) as *const CellPtr) }
  }
}

impl StablePtr {
  pub fn from_unchecked(p: i32) -> StablePtr {
    StablePtr(p)
  }

  pub fn stabilize(self) -> StablePtr {
    self
  }

  pub fn eval(self) -> StablePtr {
    unimplemented!();
  }

  pub fn grad(self, tg: StablePtr) -> StablePtr {
    unimplemented!();
  }

  pub fn cache(self) -> StablePtr {
    unimplemented!();
  }

  pub fn retain(self) -> StablePtr {
    unimplemented!();
  }
}
