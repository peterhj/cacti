use crate::cell::{CellPtr, StableCell, CellType, Dtype, IntoScalarValExt, MSet, MMap, MValueRef};
use crate::clock::{Clock};
use crate::ctx::*;
use crate::panick::*;
use crate::pctx::{Locus};
use crate::spine::{SpineRet};
use crate::thunk::op::*;

use futhark_syntax::*;

use std::borrow::{Borrow, Cow};
use std::convert::{TryInto};
use std::ops::{Deref, AddAssign, Add, Sub, Mul, Div};
use std::rc::{Rc};

impl AddAssign<f32> for CellPtr {
  #[track_caller]
  fn add_assign(&mut self, rhs: f32) {
    // FIXME FIXME
    unimplemented!();
    /*panick_wrap(|| {
      let op = AddScalarF32FutThunkSpec{val: rhs.try_into().unwrap()};
      assert!(ctx_clean_arg());
      /*ctx_push_cell_tmp_out();*/
      ctx_pop_thunk_mux(op, self.into())
    })*/
  }
}

impl<'l> Add<f32> for &'l CellPtr {
  type Output = CellPtr;

  #[track_caller]
  fn add(self, rhs: f32) -> CellPtr {
    panick_wrap(|| {
      let op = AddScalarF32FutThunkSpec{val: rhs.try_into().unwrap()};
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(self.into());
      /*ctx_push_cell_tmp_out();*/
      ctx_pop_thunk(op)
    })
  }
}

impl Add<f32> for CellPtr {
  type Output = CellPtr;

  #[track_caller]
  fn add(self, rhs: f32) -> CellPtr {
    panick_wrap(|| (&self).add(rhs))
  }
}

impl<'l> Add<f32> for &'l StableCell {
  type Output = CellPtr;

  #[track_caller]
  fn add(self, rhs: f32) -> CellPtr {
    panick_wrap(|| self.as_ptr_ref().add(rhs))
  }
}

impl Add<f32> for StableCell {
  type Output = CellPtr;

  #[track_caller]
  fn add(self, rhs: f32) -> CellPtr {
    panick_wrap(|| self.as_ptr_ref().add(rhs))
  }
}

impl<'l, R: Borrow<CellPtr>> Add<R> for &'l CellPtr {
  type Output = CellPtr;

  fn add(self, rhs: R) -> CellPtr {
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
    panick_wrap(|| {
      let op = AddFutThunkSpec;
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(*self.borrow());
      ctx_push_cell_arg(*rhs.borrow());
      ctx_pop_thunk(op)
    })
  }
}

impl<R: Borrow<CellPtr>> Add<R> for CellPtr {
  type Output = CellPtr;

  fn add(self, rhs: R) -> CellPtr {
    panick_wrap(|| (&self).add(rhs))
  }
}

impl<'l, R: Borrow<CellPtr>> Add<R> for &'l StableCell {
  type Output = CellPtr;

  fn add(self, rhs: R) -> CellPtr {
    panick_wrap(|| self.as_ptr_ref().add(rhs))
  }
}

impl<R: Borrow<CellPtr>> Add<R> for StableCell {
  type Output = CellPtr;

  fn add(self, rhs: R) -> CellPtr {
    panick_wrap(|| self.as_ptr_ref().add(rhs))
  }
}

impl<'l> Mul<f32> for &'l CellPtr {
  type Output = CellPtr;

  #[track_caller]
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

impl Mul<f32> for CellPtr {
  type Output = CellPtr;

  #[track_caller]
  fn mul(self, rhs: f32) -> CellPtr {
    panick_wrap(|| (&self).mul(rhs))
  }
}

impl<'l, R: Borrow<CellPtr>> Mul<R> for &'l CellPtr {
  type Output = CellPtr;

  #[track_caller]
  fn mul(self, rhs: R) -> CellPtr {
    panick_wrap(|| {
      let op = MulFutThunkSpec;
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(*self.borrow());
      ctx_push_cell_arg(*rhs.borrow());
      ctx_pop_thunk(op)
    })
  }
}

impl<R: Borrow<CellPtr>> Mul<R> for CellPtr {
  type Output = CellPtr;

  #[track_caller]
  fn mul(self, rhs: R) -> CellPtr {
    panick_wrap(|| (&self).mul(rhs))
  }
}

impl<'l, R: Borrow<CellPtr>> Mul<R> for &'l StableCell {
  type Output = CellPtr;

  #[track_caller]
  fn mul(self, rhs: R) -> CellPtr {
    panick_wrap(|| self.as_ptr_ref().mul(rhs))
  }
}

impl<R: Borrow<CellPtr>> Mul<R> for StableCell {
  type Output = CellPtr;

  #[track_caller]
  fn mul(self, rhs: R) -> CellPtr {
    panick_wrap(|| self.as_ptr_ref().mul(rhs))
  }
}

impl<'l> Div<f32> for &'l CellPtr {
  type Output = CellPtr;

  #[track_caller]
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

impl Div<f32> for CellPtr {
  type Output = CellPtr;

  fn div(self, rhs: f32) -> CellPtr {
    (&self).div(rhs)
  }
}

impl<'l, R: Borrow<CellPtr>> Div<R> for &'l CellPtr {
  type Output = CellPtr;

  #[track_caller]
  fn div(self, rhs: R) -> CellPtr {
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

impl<R: Borrow<CellPtr>> Div<R> for CellPtr {
  type Output = CellPtr;

  #[track_caller]
  fn div(self, rhs: R) -> CellPtr {
    panick_wrap(|| (&self).div(rhs))
  }
}

pub trait MathBinaryOps<R: Borrow<CellPtr>>: Borrow<CellPtr> {
  #[track_caller]
  fn pow(&self, rhs: R) -> CellPtr {
    unimplemented!();
  }

  /*#[track_caller]
  fn dot(self, rhs: Q) -> CellPtr {
    let p = self.into();
    let q = rhs.into();
    let op = DotThunkOp::default();
    assert!(ctx_clean_arg());
    ctx_push_cell_arg(p);
    ctx_push_cell_arg(q);
    //ctx_push_cell_out(_);
    ctx_pop_thunk(op)
  }*/

  /*#[track_caller]
  fn mm(&self, lt: bool, rhs: R, rt: bool) -> CellPtr {
    unimplemented!();
  }*/

  /*#[track_caller]
  fn blockl_mm(&self, l_block: [i64; 2], lt: bool, rhs: R, rt: bool) -> CellPtr {
    panick_wrap(|| {
      unimplemented!();
      // FIXME FIXME
      /*
      let o_dtype = p_ty.dtype.max(q_ty.dtype).unwrap();
      let op = BlockLMatrixMulThunkSpec{
        l_block,
        lt,
        rt,
        l_dtype: p_ty.dtype,
        r_dtype: q_ty.dtype,
        o_dtype,
      };
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(p);
      ctx_push_cell_arg(q);
      ctx_pop_thunk(op)
      */
    })
  }*/

  #[track_caller]
  fn block_mm(&self, l_block: [i64; 2], l_blk_t: bool, rhs: R, r_block: [i64; 2], r_blk_t: bool) -> CellPtr {
    panick_wrap(|| {
      self.block_mm_scale(l_block, l_blk_t, rhs, r_block, r_blk_t, 1.0_f32)
    })
  }

  #[track_caller]
  fn block_mm_scale<T: IntoScalarValExt>(&self, l_block: [i64; 2], l_blk_t: bool, rhs: R, r_block: [i64; 2], r_blk_t: bool, scale: T) -> CellPtr {
    panick_wrap(|| {
      let p = *self.borrow();
      let q = *rhs.borrow();
      let p_ty = ctx_lookup_type(p);
      let q_ty = ctx_lookup_type(q);
      //println!("DEBUG: block_mm_scale: l_ty={:?} r_ty={:?}", p_ty, q_ty);
      //println!("DEBUG: block_mm_scale: lblk={:?} rblk={:?}", l_block, r_block);
      println!("DEBUG: block_mm_scale: ({:?} / {:?}{}) x ({:?} / {:?}{})",
          &p_ty.shape, l_block, if l_blk_t { " T" } else { "" },
          &q_ty.shape, r_block, if r_blk_t { " T" } else { "" },
      );
      // FIXME: can relax the ndim requirement, with well documented semantics.
      assert_eq!(p_ty.ndim(), 2);
      assert_eq!(q_ty.ndim(), 2);
      let l_nrow = p_ty.shape[0] / l_block[0];
      let l_ncol = p_ty.shape[1] / l_block[1];
      let r_nrow = q_ty.shape[0] / r_block[0];
      let r_ncol = q_ty.shape[1] / r_block[1];
      assert_eq!(p_ty.shape[0] % l_block[0], 0);
      assert_eq!(p_ty.shape[1] % l_block[1], 0);
      assert_eq!(q_ty.shape[0] % r_block[0], 0);
      assert_eq!(q_ty.shape[1] % r_block[1], 0);
      let l_blk_inner = if l_blk_t { l_block[0] } else { l_block[1] };
      let r_blk_inner = if r_blk_t { r_block[1] } else { r_block[0] };
      if l_blk_inner != r_blk_inner {
        println!("ERROR: block_mm_scale: incompatible blocks:");
        println!("ERROR: block_mm_scale:   {:?}{} x {:?}{}",
            l_block, if l_blk_t { " T" } else { "" },
            r_block, if r_blk_t { " T" } else { "" },
        );
        panic!();
      }
      // FIXME FIXME: modulo/round robin blocking in the thunk impl.
      //let (l_nrow_t, l_ncol_t) = if lt { (l_ncol, l_nrow) } else { (l_nrow, l_ncol) };
      //let (r_nrow_t, r_ncol_t) = if rt { (r_ncol, r_nrow) } else { (r_nrow, r_ncol) };
      if !(l_nrow == r_nrow || l_nrow == 1 || r_nrow == 1) ||
         !(l_ncol == r_ncol || l_ncol == 1 || r_ncol == 1)
      {
        println!("ERROR: block_mm_scale: incompatible shapes:");
        println!("ERROR: block_mm_scale:   ({:?} / {:?}{}) x ({:?} / {:?}{})",
            &p_ty.shape, l_block, if l_blk_t { " T" } else { "" },
            &q_ty.shape, r_block, if r_blk_t { " T" } else { "" },
        );
        panic!();
      }
      /*let (_m, _n) = match (lt, rt) {
        (false, false) => {
          assert_eq!(p_ty.shape[1], q_ty.shape[0]);
          (p_ty.shape[0], q_ty.shape[1])
        }
        (true, false) => {
          assert_eq!(p_ty.shape[0], q_ty.shape[0]);
          (p_ty.shape[1], q_ty.shape[1])
        }
        (false, true) => {
          assert_eq!(p_ty.shape[1], q_ty.shape[1]);
          (p_ty.shape[0], q_ty.shape[0])
        }
        (true, true) => {
          assert_eq!(p_ty.shape[0], q_ty.shape[1]);
          (p_ty.shape[1], q_ty.shape[0])
        }
      };*/
      let o_dtype = match p_ty.dtype.max(q_ty.dtype) {
        None => {
          println!("ERROR: block_mm_scale: incompatible dtypes: {:?} x {:?}", p_ty.dtype, q_ty.dtype);
          panic!();
        }
        Some(dty) => dty
      };
      let o_scale = scale.into_scalar_val_();
      let o_scale_dty = o_scale.dtype();
      match o_scale_dty {
        Dtype::Float32 => {}
        _ => {
          println!("ERROR: block_mm_scale: unsupported scale dtype: {:?}", o_scale_dty);
          panic!();
        }
      }
      let op = BlockMatrixMulThunkSpec{
        //l_shape: [p_ty.shape[0], p_ty.shape[1]],
        //r_shape: [q_ty.shape[0], q_ty.shape[1]],
        l_block,
        r_block,
        //l_nblock: [l_nrow, l_ncol],
        //r_nblock: [r_nrow, r_ncol],
        l_blk_t,
        r_blk_t,
        l_dtype: p_ty.dtype,
        r_dtype: q_ty.dtype,
        o_dtype,
        o_scale,
      };
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(p);
      ctx_push_cell_arg(q);
      ctx_pop_thunk(op)
    })
  }
}

impl<L: Borrow<CellPtr>, R: Borrow<CellPtr>> MathBinaryOps<R> for L {}

pub trait MathUnaryOps: Borrow<CellPtr> {
  #[track_caller]
  fn sqrt(&self) -> CellPtr {
    panick_wrap(|| {
      let op = SqrtFutThunkSpec;
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(*self.borrow());
      /*ctx_push_cell_tmp_out();*/
      ctx_pop_thunk(op)
    })
  }

  #[track_caller]
  fn rsqrt(&self) -> CellPtr {
    panick_wrap(|| {
      let op = RsqrtFutThunkSpec;
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(*self.borrow());
      /*ctx_push_cell_tmp_out();*/
      ctx_pop_thunk(op)
    })
  }

  #[track_caller]
  fn cos(&self) -> CellPtr {
    panick_wrap(|| {
      let op = CosFutThunkSpec;
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(*self.borrow());
      /*ctx_push_cell_tmp_out();*/
      ctx_pop_thunk(op)
    })
  }

  #[track_caller]
  fn sin(&self) -> CellPtr {
    panick_wrap(|| {
      let op = SinFutThunkSpec;
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(*self.borrow());
      /*ctx_push_cell_tmp_out();*/
      ctx_pop_thunk(op)
    })
  }

  #[track_caller]
  fn exp(&self) -> CellPtr {
    panick_wrap(|| {
      let op = ExpFutThunkSpec;
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(*self.borrow());
      /*ctx_push_cell_tmp_out();*/
      ctx_pop_thunk(op)
    })
  }

  #[track_caller]
  fn tanh(&self) -> CellPtr {
    panick_wrap(|| {
      let op = TanhFutThunkSpec;
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(*self.borrow());
      /*ctx_push_cell_tmp_out();*/
      ctx_pop_thunk(op)
    })
  }

  #[track_caller]
  fn powi(&self, exp: i64) -> CellPtr {
    panick_wrap(|| {
      let p = *self.borrow();
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(p);
      /*ctx_push_cell_tmp_out();*/
      match ctx_lookup_dtype(p) {
        // FIXME FIXME
        Dtype::Float32 => {
          let op = PowiF32FutThunkSpec{exp};
          ctx_pop_thunk(op)
        }
        _ => unimplemented!()
      }
    })
  }

  #[track_caller]
  fn inner_max(&self) -> CellPtr {
    unimplemented!();
    /*
    let p = *self.borrow();
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

  #[track_caller]
  fn inner_mean(&self) -> CellPtr {
    unimplemented!();
    /*
    let p = *self.borrow();
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

  #[track_caller]
  fn inner_sum(&self) -> CellPtr {
    unimplemented!();
    /*
    let p = *self.borrow();
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

  #[track_caller]
  fn inner_softmax(&self) -> CellPtr {
    panick_wrap(|| {
      let op = InnerSoftmaxFutThunkSpec;
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(*self.borrow());
      ctx_pop_thunk(op)
    })
  }

  #[track_caller]
  fn flat_sum(&self) -> CellPtr {
    panick_wrap(|| {
      let p = *self.borrow();
      let ty_ = ctx_lookup_type(p);
      match ty_.ndim() {
        0 => p,
        1 => {
          assert!(ctx_clean_arg());
          ctx_push_cell_arg(p);
          let op = Sum1dFutThunkSpec;
          ctx_pop_thunk(op)
        }
        2 => {
          assert!(ctx_clean_arg());
          ctx_push_cell_arg(p);
          let op = Sum2dFutThunkSpec;
          ctx_pop_thunk(op)
        }
        3 => {
          assert!(ctx_clean_arg());
          ctx_push_cell_arg(p);
          let op = Sum3dFutThunkSpec;
          ctx_pop_thunk(op)
        }
        4 => {
          assert!(ctx_clean_arg());
          ctx_push_cell_arg(p);
          let op = Sum4dFutThunkSpec;
          ctx_pop_thunk(op)
        }
        _ => unimplemented!()
      }
    })
  }
}

impl<P: Borrow<CellPtr>> MathUnaryOps for P {}

pub fn zeros<S: Into<Vec<i64>>, D: Into<Dtype>>(shape: S, dtype: D) -> CellPtr {
  unimplemented!();
}

pub fn ones<S: Into<Vec<i64>>, D: Into<Dtype>>(shape: S, dtype: D) -> CellPtr {
  unimplemented!();
}

pub trait CastOps: Borrow<CellPtr> {
  /*fn upcast_f32(self) -> CellPtr {
    unimplemented!();
  }

  fn downcast_f16(self) -> CellPtr {
    unimplemented!();
  }*/

  #[track_caller]
  fn cast(&self, new_dtype: Dtype) -> CellPtr {
    panick_wrap(|| {
      let x = *self.borrow();
      let org_dtype = ctx_lookup_dtype(x);
      if org_dtype == new_dtype {
        return x;
      }
      match (org_dtype, new_dtype) {
        (Dtype::Float32, Dtype::Float16) |
        (Dtype::Float16, Dtype::Float32) => {
          let op = CastFutThunkSpec{org_dtype, new_dtype};
          assert!(ctx_clean_arg());
          ctx_push_cell_arg(x);
          ctx_pop_thunk(op)
        }
        (Dtype::Float32, Dtype::BFloat16) => {
          let op = CastF32Bf16FutThunkSpec;
          assert!(ctx_clean_arg());
          ctx_push_cell_arg(x);
          ctx_pop_thunk(op)
        }
        (Dtype::BFloat16, Dtype::Float16) => {
          let op = CastBf16F16FutThunkSpec;
          assert!(ctx_clean_arg());
          ctx_push_cell_arg(x);
          ctx_pop_thunk(op)
        }
        (Dtype::BFloat16, Dtype::Float32) => {
          let op = CastBf16F32FutThunkSpec;
          assert!(ctx_clean_arg());
          ctx_push_cell_arg(x);
          ctx_pop_thunk(op)
        }
        _ => unimplemented!()
      }
    })
  }
}

impl<L: Borrow<CellPtr>> CastOps for L {}

pub trait GradOps<R: Borrow<CellPtr>>: Borrow<CellPtr> {
  #[track_caller]
  fn grad(&self, x: R) -> CellPtr { self.gradr(x) }

  #[track_caller]
  fn gradr(&self, x: R) -> CellPtr {
    // FIXME FIXME
    //ctx_lookup_or_insert_gradr(self.into(), x.into())
    unimplemented!();
  }

  #[track_caller]
  fn gradl(&self, y: R) -> CellPtr {
    // FIXME FIXME
    //ctx_lookup_or_insert_gradl(self.into(), tg.into())
    unimplemented!();
  }
}

impl<L: Borrow<CellPtr>, R: Borrow<CellPtr>> GradOps<R> for L {}

pub trait ArrayOps: Borrow<CellPtr> + Sized {
  #[track_caller]
  fn type_(&self) -> CellType {
    panick_wrap(|| {
      ctx_lookup_type(*self.borrow())
    })
  }

  #[track_caller]
  fn shape(&self) -> Vec<i64> {
    panick_wrap(|| {
      self.type_().shape
    })
  }

  #[track_caller]
  fn dtype(&self) -> Dtype {
    panick_wrap(|| {
      ctx_lookup_dtype(*self.borrow())
    })
  }

  #[track_caller]
  fn bit_alias(&self, new_dtype: Dtype) -> CellPtr {
    panick_wrap(|| {
      ctx_alias_bits(*self.borrow(), new_dtype)
    })
  }

  #[track_caller]
  fn new_shape<S: Into<Vec<i64>>>(&self, new_shape: S) -> CellPtr {
    panick_wrap(|| {
      ctx_alias_new_shape(*self.borrow(), new_shape.into())
    })
  }

  #[track_caller]
  fn reshape<S: Into<Vec<i64>>>(&self, new_shape: S) -> CellPtr { self.new_shape(new_shape) }

  #[track_caller]
  fn inner_one_hot(&self, inner_len: i64, new_dtype: Dtype) -> CellPtr {
    panick_wrap(|| {
      if !(inner_len > 0) {
        panic!("ERROR: inner_one_hot: invalid parameter: expected inner_len > 0, actual {:?}", inner_len);
      }
      let x = *self.borrow();
      let org_dtype = ctx_lookup_dtype(x);
      if !org_dtype.is_uint() {
        panic!("ERROR: inner_one_hot: invalid argument: expected dtype uint, actual {:?}", org_dtype);
      }
      let op = InnerOneHotFutThunkSpec{inner_len, /*org_dtype,*/ new_dtype};
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(x);
      ctx_pop_thunk(op)
    })
  }
}

impl<L: Borrow<CellPtr> + Sized> ArrayOps for L {}

pub trait CtlOps: Borrow<CellPtr> + Sized {
  /*
  #[track_caller]
  fn yield_(self) -> CellPtr {
    unimplemented!();
  }

  #[track_caller]
  fn break_(self) -> CellPtr {
    unimplemented!();
  }
  */

  /*#[track_caller]
  fn trace(self) -> CellPtr {
    // FIXME FIXME
    ctx_trace_val(*self.as_ref())
  }*/

  /*#[track_caller]
  fn profile(self) -> CellPtr {
    // FIXME FIXME
    ctx_profile_val(*self.as_ref())
  }*/

  #[track_caller]
  fn opaque(self) -> CellPtr {
    panick_wrap(|| ctx_opaque(*self.borrow()))
  }

  #[track_caller]
  fn const_(self) -> CellPtr {
    unimplemented!();
  }
}

impl<L: Borrow<CellPtr> + Sized> CtlOps for L {}

pub trait Ops: Borrow<CellPtr> + Sized {
  /*fn bar(self) -> Self {
    unimplemented!();
  }*/

  /*fn set_mem(self, ) -> Self {
    unimplemented!();
  }*/

  /*fn set_futhark(self, fut_str: &str) -> Self {
    unimplemented!();
  }*/

  //fn set(self, set_f: Box<FnOnce() -> _>) -> Self;
  //fn set_in_place(self, set_f: Box<FnOnce(_)>) -> Self;

  /*#[track_caller]
  fn init_futhark(self, fut_str: &str) -> Self {
    unimplemented!();
  }*/

  //fn init(self, init_f: Box<FnOnce() -> _>) -> Self;

  #[track_caller]
  fn apply_futhark(&self, lam_src: Cow<'static, str>) -> CellPtr {
    apply_futhark(lam_src, &[self.borrow()])
  }

  //fn apply_fut(self, fut_str: &[u8]) -> Self { self.apply_futhark(fut_str) }

  #[track_caller]
  fn cache(&self) /*-> Self */{
    panick_wrap(|| TL_CTX.with(|ctx| {
      let mut spine = ctx.spine.borrow_mut();
      spine.cache_aff(*self.borrow());
      //self
    }))
  }

  #[track_caller]
  fn cache_init(&self) /*-> Self */{
    panick_wrap(|| TL_CTX.with(|ctx| {
      let mut spine = ctx.spine.borrow_mut();
      spine.init_cache_mux(*self.borrow());
      //self
    }))
  }

  #[track_caller]
  fn init_cache(&self) /*-> Self */{ self.cache_init() }

  /*fn cache_init_futhark(self, fut_str: &str) -> Self {
    // TODO: ???
    unimplemented!();
  }*/

  #[track_caller]
  fn keep(&self) -> StableCell {
    panick_wrap(|| StableCell::from(*self.borrow()))
  }

  #[track_caller]
  fn set<X: AsRef<CellPtr>>(&self, _x: X) where Self: Sized {
    unimplemented!();
  }

  #[track_caller]
  fn unsafe_unseal(self) -> Self {
    unimplemented!();
  }

  #[track_caller]
  fn unsafe_unseal_init(self) -> Self {
    panick_wrap(|| TL_CTX.with(|ctx| {
      let mut spine = ctx.spine.borrow_mut();
      spine.unseal_mux(*self.borrow());
      self
    }))
  }

  /*#[track_caller]
  fn yield_get_and_set(self) -> Self {
    panick_wrap(|| {
      // FIXME
      unimplemented!();
    })
  }*/

  #[track_caller]
  fn mem_set_yield_(&self) {
    panick_wrap(|| TL_CTX.with(|ctx| {
      let mut spine = ctx.spine.borrow_mut();
      spine.yield_set(*self.borrow(), Locus::Mem);
    }))
  }

  #[track_caller]
  fn mem_set_yield_with(&self, _: ()) {
    panick_wrap(|| self.mem_set_yield_())
  }

  #[track_caller]
  fn mem_init_yield_(&self) {
    panick_wrap(|| TL_CTX.with(|ctx| {
      let mut spine = ctx.spine.borrow_mut();
      spine.yield_init(*self.borrow(), Locus::Mem);
    }))
  }

  #[track_caller]
  fn mem_init_yield_with(&self, _: ()) {
    panick_wrap(|| self.mem_init_yield_())
  }

  /*#[track_caller]
  fn eval(self) -> Self {
    panick_wrap(|| {
      let ret = eval(*self.borrow());
      match ret {
        SpineRet::Bot => panic!("EXCEPTION"),
        _ => {}
      }
      self
    })
  }

  #[track_caller]
  fn try_eval(self) -> Result<Self, ()> {
    panick_wrap(|| {
      let ret = eval(*self.as_ref());
      match ret {
        SpineRet::Bot => return Err(()),
        _ => {}
      }
      Ok(self)
    })
  }*/

  /*#[track_caller]
  fn tag(self, /*_: ???*/) -> Self {
    unimplemented!();
  }*/

  #[track_caller]
  fn version(&self) -> Clock {
    panick_wrap(|| {
      ctx_lookup_clk(*self.borrow())
    })
  }

  #[track_caller]
  fn snapshot(&self) -> CellPtr {
    panick_wrap(|| TL_CTX.with(|ctx| {
      let mut env = ctx.env.borrow_mut();
      env.snapshot(&ctx.ctr, *self.borrow())
    }))
  }

  /*#[track_caller]
  fn checkpoint(self) -> CellPtr {
    unimplemented!();
  }*/
}

impl<L: Borrow<CellPtr> + Sized> Ops for L {}

impl MSet {
  #[track_caller]
  pub fn add<'x, X: Into<MValueRef<'x>>>(&self, x: X) {
    unimplemented!();
  }
}

impl MMap {
  #[track_caller]
  pub fn add<'k, 'v, K: Into<MValueRef<'k>>, V: Into<MValueRef<'v>>>(&self, k: K, v: V) {
    unimplemented!();
  }
}

pub fn vjp(y_dy: &MMap, x: &MSet) -> MMap {
  unimplemented!();
}

pub fn jvp(y: &MSet, x_dx: &MMap) -> MMap {
  unimplemented!();
}

#[track_caller]
pub fn apply_futhark(lam_src: Cow<'static, str>, arg: &[&CellPtr]) -> CellPtr {
  panick_wrap(|| {
    let result = TL_CTX.with(|ctx| {
      let mut fut = ctx.futhark.borrow_mut();
      if fut.trie.is_none() {
        fut.trie = Some(Rc::new(tokenizer_trie()));
      }
      drop(fut);
      let trie = ctx.futhark.borrow().trie.as_ref().unwrap().clone();
      let tokens = Tokenizer::new(trie, &*lam_src);
      let parser = ExpParser::new(tokens);
      parser.parse()
    });
    let mut wrap_parens = false;
    match result.as_ref() {
      Ok(&Exp::Lam(..)) => {}
      Ok(&Exp::LamAlt(..)) => {
        wrap_parens = true;
      }
      Ok(_) => panic!("ERROR: expected futhark lambda expression, e.g. (\\x y -> x + y)"),
      Err(_) => panic!("ERROR: invalid futhark"),
    }
    let lam_exp = result.unwrap();
    assert!(arg.len() < u16::max_value() as usize);
    match &lam_exp {
      &Exp::Lam(ref vars, ..) |
      &Exp::LamAlt(ref vars, ..) => {
        assert_eq!(arg.len(), vars.len());
      }
      _ => unreachable!()
    }
    let ar_in = arg.len() as u16;
    // NB: arity out must be one because of the type signature (-> CellPtr).
    let ar_out = 1;
    assert!(ctx_clean_arg());
    for &&x in arg.iter() {
      // FIXME FIXME
      let xty_ = ctx_lookup_dtype(x);
      ctx_push_cell_arg(x);
    }
    let op = LamFutExpThunkSpec{lam_src, lam_exp, ar_in, ar_out, wrap_parens};
    // FIXME FIXME: here we have to build the object, because we don't know
    // the output dim or type. if the lam is annotated, double check the actual
    // dim/type we read out of the manifest.
    //let oty_ = _;
    ctx_pop_thunk(op)
    //ctx_pop_thunk_(op, oty_)
  })
}

#[track_caller]
pub fn apply2_futhark(lam_src: Cow<'static, str>, arg: &[&CellPtr]) -> (CellPtr, CellPtr) {
  unimplemented!();
}

/*#[track_caller]
pub fn apply_futhark_unverified(lam_src: &str, arg: &[&CellPtr]) -> CellPtr {
  // FIXME FIXME
  unimplemented!();
}*/
