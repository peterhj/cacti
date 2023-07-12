//use crate::cell::{CellPtr, StableCell, CellType, Dtype, ScalarVal_, IntoScalarValExt, MSet, MMap, MValueRef};
use crate::cell::*;
use crate::clock::{Clock};
use crate::ctx::*;
use crate::panick::*;
use crate::pctx::{TL_PCTX, Locus, PMach, MemReg};
use crate::spine::{SpineRet};
use crate::thunk::op::*;

use futhark_syntax::*;

use std::borrow::{Borrow, Cow};
use std::convert::{TryInto};
use std::iter::{repeat};
use std::ops::{AddAssign, BitXor, Index, IndexMut, RangeFull, Add, Sub, Mul, Div, Neg};
use std::rc::{Rc};

impl AddAssign<f32> for CellPtr {
  #[track_caller]
  fn add_assign(&mut self, rhs: f32) {
    // FIXME FIXME
    unimplemented!();
    /*panick_wrap(|| {
      let op = AddScalarF32FutThunkSpec{val: rhs.try_into().unwrap()};
      assert!(ctx_clean_arg());
      ctx_pop_thunk_mux(op, self.into())
    })*/
  }
}

impl<R: Borrow<CellPtr>> AddAssign<R> for CellPtr {
  #[track_caller]
  fn add_assign(&mut self, rhs: R) {
    panick_wrap(|| {
      let this = *self;
      let rhs = *rhs.borrow();
      if TL_CTX.with(|ctx| {
        let mut success = false;
        let mut thunkenv = ctx.thunkenv.borrow_mut();
        if thunkenv.accumulate_in_place {
          let spine = ctx.spine.borrow();
          let mut this_clk = spine._version(this).unwrap();
          let rhs_clk = spine._version(rhs).unwrap();
          // FIXME: should also check the thunk mode.
          if rhs_clk.up == 1 &&
             !thunkenv.update.contains_key(&(rhs, rhs_clk.init()))
          {
            this_clk.up += 1;
            let tclo = thunkenv.update.remove(&(rhs, rhs_clk)).unwrap();
            assert!(thunkenv.update.insert((this, this_clk), tclo).is_none());
            success = true;
          }
        }
        success
      }) {
        return;
      }
      let op = NopFutThunkSpec;
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(rhs);
      ctx_pop_accumulate_thunk(op, this)
    })
  }
}

impl<'l, R: Borrow<CellPtr>> AddAssign<R> for &'l mut CellPtr {
  #[track_caller]
  fn add_assign(&mut self, rhs: R) {
    panick_wrap(|| (*self).add_assign(rhs))
  }
}

impl<R: Borrow<CellPtr>> AddAssign<R> for StableCell {
  #[track_caller]
  fn add_assign(&mut self, rhs: R) {
    panick_wrap(|| self.as_ptr_mut().add_assign(rhs))
  }
}

impl<'l, R: Borrow<CellPtr>> AddAssign<R> for &'l mut StableCell {
  #[track_caller]
  fn add_assign(&mut self, rhs: R) {
    panick_wrap(|| self.as_ptr_mut().add_assign(rhs))
  }
}

#[derive(Clone, Copy, Debug)]
pub struct T;

#[derive(Clone, Copy, Debug)]
pub struct T_(pub i8, pub i8);

impl<'l> BitXor<T> for &'l CellPtr {
  type Output = &'l CellViewHandle;

  #[track_caller]
  fn bitxor(self, _: T) -> &'l CellViewHandle {
    panick_wrap(|| {
      // FIXME FIXME
      //CellView(*self, vec![CellVOp::Swap(-2, -1)])
      CellViewHandle::_from(*self)
    })
  }
}

impl<'l> BitXor<T_> for &'l CellPtr {
  type Output = &'l CellViewHandle;

  #[track_caller]
  fn bitxor(self, t: T_) -> &'l CellViewHandle {
    panick_wrap(|| {
      // FIXME FIXME
      //CellView(*self, vec![CellVOp::Swap(t.0, t.1)])
      CellViewHandle::_from(*self)
    })
  }
}

/*impl BitXor<T> for CellPtr {
  type Output = CellViewHandle;

  #[track_caller]
  fn bitxor(self, _: T) -> CellViewHandle {
    panick_wrap(|| (&self).bitxor(T))
  }
}*/

impl<'l> BitXor<T> for &'l StableCell {
  type Output = &'l CellViewHandle;

  #[track_caller]
  fn bitxor(self, _: T) -> &'l CellViewHandle {
    panick_wrap(|| self.as_ptr_ref().bitxor(T))
  }
}

impl BitXor<T> for StableCell {
  type Output = CellViewHandle2;

  #[track_caller]
  fn bitxor(self, _: T) -> CellViewHandle2 {
    panick_wrap(|| {
      // FIXME FIXME
      //CellView(*self, vec![CellVOp::Swap(t.0, t.1)])
      CellViewHandle2::_from(*self.as_ptr_ref())
    })
  }
}

/*impl BitXor<T_> for StableCell {
  type Output = CellViewHandle;

  #[track_caller]
  fn bitxor(self, t: T_) -> CellViewHandle {
    panick_wrap(|| (&self).bitxor(t))
  }
}*/

impl<'l> BitXor<T> for &'l CellViewHandle {
  type Output = &'l CellViewHandle;

  #[track_caller]
  fn bitxor(mut self, _: T) -> &'l CellViewHandle {
    panick_wrap(|| {
      // FIXME FIXME
      //self.1.push(CellVOp::Swap(-2, -1));
      self
    })
  }
}

impl<'l> BitXor<T_> for &'l CellViewHandle {
  type Output = &'l CellViewHandle;

  #[track_caller]
  fn bitxor(mut self, t: T_) -> &'l CellViewHandle {
    panick_wrap(|| {
      // FIXME FIXME
      //self.1.push(CellVOp::Swap(t.0, t.1));
      self
    })
  }
}

impl Index<RangeFull> for CellPtr {
  type Output = CellPtr;

  #[track_caller]
  fn index(&self, _: RangeFull) -> &CellPtr {
    panick_wrap(|| self)
  }
}

impl IndexMut<RangeFull> for CellPtr {
  #[track_caller]
  fn index_mut(&mut self, _: RangeFull) -> &mut CellPtr {
    panick_wrap(|| self)
  }
}

impl Index<IRange> for CellPtr {
  type Output = CellViewHandle;

  #[track_caller]
  fn index(&self, _: IRange) -> &CellViewHandle {
    panick_wrap(|| {
      // FIXME
      CellViewHandle::_from(*self)
      //unimplemented!();
    })
  }
}

impl IndexMut<IRange> for CellPtr {
  #[track_caller]
  fn index_mut(&mut self, _: IRange) -> &mut CellViewHandle {
    panick_wrap(|| {
      // FIXME
      CellViewHandle::_from_mut(*self)
      //unimplemented!();
    })
  }
}

impl Index<[IRange; 2]> for CellPtr {
  type Output = CellViewHandle;

  #[track_caller]
  fn index(&self, _: [IRange; 2]) -> &CellViewHandle {
    panick_wrap(|| {
      // FIXME
      //CellViewHandle::_from(*self)
      unimplemented!();
    })
  }
}

pub trait IntoCellViewOps: Into<CellView> {
  fn inner_transpose(self) -> CellView {
    panick_wrap(|| {
      let mut this = self.into();
      this.1.push(CellVOp::Swap(-2, -1));
      this
    })
  }

  fn transpose(self, ld: i8, rd: i8) -> CellView {
    panick_wrap(|| {
      let mut this = self.into();
      this.1.push(CellVOp::Swap(ld, rd));
      this
    })
  }
}

impl IntoCellViewOps for CellPtr {}
impl<'l> IntoCellViewOps for &'l CellPtr {}
impl IntoCellViewOps for StableCell {}
impl<'l> IntoCellViewOps for &'l StableCell {}
impl IntoCellViewOps for CellView {}

pub trait BorrowCellViewOps: BorrowCellView {
  fn materialize(self) -> CellPtr where Self: Sized {
    panick_wrap(|| {
      let view = self._borrow();
      let x = *(view.0);
      let vops = view.1.unwrap_or(&[]);
      if vops.is_empty() {
        return x;
      }
      let mut x = x;
      let x_ty = x.type_();
      let x_nd = x_ty.ndim();
      let nop = CellVOp::Nop;
      let mut state = CellViewState::new(x_nd);
      for vop in vops.into_iter().chain(repeat(&nop)) {
        loop {
          match state._step(vop) {
            CellViewStep::Break => {
              break;
            }
            CellViewStep::Swap => {
              if state.ndim >= 2 {
                let mut try_inner = true;
                for d in 0 .. state.ndim - 2 {
                  if state.perm[d as usize] != d {
                    try_inner = false;
                    break;
                  }
                }
                if try_inner &&
                   state.perm[state.ndim as usize - 2] == state.ndim - 1 &&
                   state.perm[state.ndim as usize - 1] == state.ndim - 2
                {
                  let op = InnerTransposeFutThunkSpec;
                  assert!(ctx_clean_arg());
                  ctx_push_cell_arg(x);
                  x = ctx_pop_thunk(op);
                  state._reset_swap();
                  continue;
                }
              }
              println!("ERROR: materialize: general transposition is currently unimplemented");
              panic!("bug");
            }
            CellViewStep::Halt => {
              return x;
            }
            _ => unimplemented!()
          }
        }
      }
      unreachable!();
    })
  }
}

impl BorrowCellViewOps for CellPtr {}
impl<'l> BorrowCellViewOps for &'l CellPtr {}
impl BorrowCellViewOps for StableCell {}
impl<'l> BorrowCellViewOps for &'l StableCell {}
impl BorrowCellViewOps for CellView {}
impl<'l> BorrowCellViewOps for &'l CellView {}
impl<'l> BorrowCellViewOps for CellViewRef<'l> {}

impl<'l> Add<f32> for &'l CellPtr {
  type Output = CellPtr;

  #[track_caller]
  fn add(self, rhs: f32) -> CellPtr {
    panick_wrap(|| {
      let op = AddScalarF32FutThunkSpec{val: rhs.try_into().unwrap()};
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(*self);
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
    panick_wrap(|| {
      let x0 = *self.borrow();
      let x1 = *rhs.borrow();
      let x0_ty = x0.type_();
      let x1_ty = x1.type_();
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(x0);
      ctx_push_cell_arg(x1);
      if &x0_ty.shape != &x1_ty.shape || x0_ty.dtype != x1_ty.dtype {
        ctx_pop_thunk(BroadcastAddFutThunkSpec)
      } else {
        ctx_pop_thunk(AddFutThunkSpec)
      }
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

impl<'l> Sub<CellPtr> for &'l f32 {
  type Output = CellPtr;

  #[track_caller]
  fn sub(self, rhs: CellPtr) -> CellPtr {
    panick_wrap(|| {
      let op = LSubScalarF32FutThunkSpec{val: (*self).try_into().unwrap()};
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(rhs);
      ctx_pop_thunk(op)
    })
  }
}

impl Sub<CellPtr> for f32 {
  type Output = CellPtr;

  #[track_caller]
  fn sub(self, rhs: CellPtr) -> CellPtr {
    panick_wrap(|| (&self).sub(rhs))
  }
}

impl<'l> Sub<f32> for &'l CellPtr {
  type Output = CellPtr;

  #[track_caller]
  fn sub(self, rhs: f32) -> CellPtr {
    panick_wrap(|| {
      let op = RSubScalarF32FutThunkSpec{val: rhs.try_into().unwrap()};
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(*self);
      ctx_pop_thunk(op)
    })
  }
}

impl Sub<f32> for CellPtr {
  type Output = CellPtr;

  #[track_caller]
  fn sub(self, rhs: f32) -> CellPtr {
    panick_wrap(|| (&self).sub(rhs))
  }
}

impl<'l, R: Borrow<CellPtr>> Sub<R> for &'l CellPtr {
  type Output = CellPtr;

  fn sub(self, rhs: R) -> CellPtr {
    panick_wrap(|| {
      let x0 = *self.borrow();
      let x1 = *rhs.borrow();
      let x0_ty = x0.type_();
      let x1_ty = x1.type_();
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(x0);
      ctx_push_cell_arg(x1);
      if &x0_ty.shape != &x1_ty.shape || x0_ty.dtype != x1_ty.dtype {
        ctx_pop_thunk(BroadcastSubFutThunkSpec)
      } else {
        ctx_pop_thunk(SubFutThunkSpec)
      }
    })
  }
}

impl<R: Borrow<CellPtr>> Sub<R> for CellPtr {
  type Output = CellPtr;

  fn sub(self, rhs: R) -> CellPtr {
    panick_wrap(|| (&self).sub(rhs))
  }
}

impl<'l, R: Borrow<CellPtr>> Sub<R> for &'l StableCell {
  type Output = CellPtr;

  fn sub(self, rhs: R) -> CellPtr {
    panick_wrap(|| self.as_ptr_ref().sub(rhs))
  }
}

impl<R: Borrow<CellPtr>> Sub<R> for StableCell {
  type Output = CellPtr;

  fn sub(self, rhs: R) -> CellPtr {
    panick_wrap(|| self.as_ptr_ref().sub(rhs))
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
      let x0 = *self.borrow();
      let x1 = *rhs.borrow();
      let x0_ty = x0.type_();
      let x1_ty = x1.type_();
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(x0);
      ctx_push_cell_arg(x1);
      if &x0_ty.shape != &x1_ty.shape || x0_ty.dtype != x1_ty.dtype {
        ctx_pop_thunk(BroadcastMulFutThunkSpec)
      } else {
        ctx_pop_thunk(MulFutThunkSpec)
      }
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
    panick_wrap(|| {
      let op = RDivScalarF32FutThunkSpec{val: rhs.try_into().unwrap()};
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(*self.borrow());
      ctx_pop_thunk(op)
    })
  }
}

impl Div<f32> for CellPtr {
  type Output = CellPtr;

  fn div(self, rhs: f32) -> CellPtr {
    panick_wrap(|| (&self).div(rhs))
  }
}

impl<'l, R: Borrow<CellPtr>> Div<R> for &'l CellPtr {
  type Output = CellPtr;

  #[track_caller]
  fn div(self, rhs: R) -> CellPtr {
    panick_wrap(|| {
      let x0 = *self.borrow();
      let x1 = *rhs.borrow();
      let x0_ty = x0.type_();
      let x1_ty = x1.type_();
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(x0);
      ctx_push_cell_arg(x1);
      if &x0_ty.shape != &x1_ty.shape || x0_ty.dtype != x1_ty.dtype {
        ctx_pop_thunk(BroadcastDivFutThunkSpec)
      } else {
        ctx_pop_thunk(DivFutThunkSpec)
      }
    })
  }
}

impl<R: Borrow<CellPtr>> Div<R> for CellPtr {
  type Output = CellPtr;

  #[track_caller]
  fn div(self, rhs: R) -> CellPtr {
    panick_wrap(|| (&self).div(rhs))
  }
}

impl<'l, R: Borrow<CellPtr>> Div<R> for &'l StableCell {
  type Output = CellPtr;

  #[track_caller]
  fn div(self, rhs: R) -> CellPtr {
    panick_wrap(|| self.as_ptr_ref().div(rhs))
  }
}

impl<R: Borrow<CellPtr>> Div<R> for StableCell {
  type Output = CellPtr;

  #[track_caller]
  fn div(self, rhs: R) -> CellPtr {
    panick_wrap(|| self.as_ptr_ref().div(rhs))
  }
}

impl<'l> Neg for &'l CellPtr {
  type Output = CellPtr;

  #[track_caller]
  fn neg(self) -> CellPtr {
    panick_wrap(|| {
      let x = *self.borrow();
      let x_dtype = ctx_lookup_dtype(x);
      match x_dtype {
        Dtype::Fp16 => {
          let op = NegF16FutThunkSpec;
          assert!(ctx_clean_arg());
          ctx_push_cell_arg(x);
          ctx_pop_thunk(op)
        }
        _ => {
          let op = NegFutThunkSpec;
          assert!(ctx_clean_arg());
          ctx_push_cell_arg(x);
          ctx_pop_thunk(op)
        }
      }
    })
  }
}

impl Neg for CellPtr {
  type Output = CellPtr;

  #[track_caller]
  fn neg(self) -> CellPtr {
    panick_wrap(|| (&self).neg())
  }
}

impl<'l> Neg for &'l StableCell {
  type Output = CellPtr;

  #[track_caller]
  fn neg(self) -> CellPtr {
    panick_wrap(|| self.as_ptr_ref().neg())
  }
}

impl Neg for StableCell {
  type Output = CellPtr;

  #[track_caller]
  fn neg(self) -> CellPtr {
    panick_wrap(|| self.as_ptr_ref().neg())
  }
}

pub trait MathBinaryOps<R: Borrow<CellPtr>>: Borrow<CellPtr> {
  #[track_caller]
  fn pow(&self, rhs: R) -> CellPtr {
    panick_wrap(|| {
      unimplemented!();
    })
  }

  #[track_caller]
  fn inner_concat(&self, rhs: R) -> CellPtr {
    panick_wrap(|| {
      let op = InnerConcatFutThunkSpec;
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(*self.borrow());
      ctx_push_cell_arg(*rhs.borrow());
      ctx_pop_thunk(op)
    })
  }

  #[track_caller]
  fn inner_select(&self, rank: R) -> CellPtr {
    panick_wrap(|| {
      let rank = *rank.borrow();
      let rank_dtype = ctx_lookup_dtype(rank);
      if !rank_dtype.is_uint() {
        panic!("ERROR: inner_select: invalid argument: expected dtype uint, actual {:?}", rank_dtype);
      }
      let op = InnerSelectFutThunkSpec;
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(*self.borrow());
      ctx_push_cell_arg(rank);
      ctx_pop_thunk(op)
    })
  }

  #[track_caller]
  fn inner_inv_select(&self, rank: R, inner_len: i64) -> CellPtr {
    panick_wrap(|| {
      let rank = *rank.borrow();
      let rank_dtype = ctx_lookup_dtype(rank);
      if !rank_dtype.is_uint() {
        panic!("ERROR: inner_inv_select: invalid argument: expected dtype uint, actual {:?}", rank_dtype);
      }
      let op = InnerInvSelectFutThunkSpec{inner_len};
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(*self.borrow());
      ctx_push_cell_arg(rank);
      ctx_pop_thunk(op)
    })
  }

  #[track_caller]
  fn outer_select(&self, rank: R) -> CellPtr {
    panick_wrap(|| {
      let rank = *rank.borrow();
      let rank_dtype = ctx_lookup_dtype(rank);
      if !rank_dtype.is_uint() {
        panic!("ERROR: outer_select: invalid argument: expected dtype uint, actual {:?}", rank_dtype);
      }
      let op = OuterSelectFutThunkSpec;
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(*self.borrow());
      ctx_push_cell_arg(rank);
      ctx_pop_thunk(op)
    })
  }

  #[track_caller]
  fn inner_softmax_categorical_nll(&self, rank: R) -> CellPtr {
    panick_wrap(|| {
      let op = InnerSoftmaxCategoricalNLLFutThunkSpec;
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(*self.borrow());
      ctx_push_cell_arg(*rank.borrow());
      ctx_pop_thunk(op)
    })
  }

  #[track_caller]
  fn outer_mul(&self, rhs: R) -> CellPtr {
    panick_wrap(|| {
      let op = OuterMulFutThunkSpec;
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(*self.borrow());
      ctx_push_cell_arg(*rhs.borrow());
      ctx_pop_thunk(op)
    })
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
        Dtype::Fp32 => {}
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
  fn square(&self) -> CellPtr {
    panick_wrap(|| {
      let op = SquareFutThunkSpec;
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(*self.borrow());
      ctx_pop_thunk(op)
    })
  }

  #[track_caller]
  fn recip(&self) -> CellPtr {
    panick_wrap(|| {
      let op = RecipFutThunkSpec;
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(*self.borrow());
      ctx_pop_thunk(op)
    })
  }

  #[track_caller]
  fn reciprocal(&self) -> CellPtr {
    panick_wrap(|| self.recip())
  }

  #[track_caller]
  fn sqrt(&self) -> CellPtr {
    panick_wrap(|| {
      let op = SqrtFutThunkSpec;
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(*self.borrow());
      ctx_pop_thunk(op)
    })
  }

  #[track_caller]
  fn rsqrt(&self) -> CellPtr {
    panick_wrap(|| {
      let op = RsqrtFutThunkSpec;
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(*self.borrow());
      ctx_pop_thunk(op)
    })
  }

  #[track_caller]
  fn cos(&self) -> CellPtr {
    panick_wrap(|| {
      let op = CosFutThunkSpec;
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(*self.borrow());
      ctx_pop_thunk(op)
    })
  }

  #[track_caller]
  fn sin(&self) -> CellPtr {
    panick_wrap(|| {
      let op = SinFutThunkSpec;
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(*self.borrow());
      ctx_pop_thunk(op)
    })
  }

  #[track_caller]
  fn exp(&self) -> CellPtr {
    panick_wrap(|| {
      let op = ExpFutThunkSpec;
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(*self.borrow());
      ctx_pop_thunk(op)
    })
  }

  #[track_caller]
  fn logistic(&self) -> CellPtr {
    panick_wrap(|| {
      let op = LogisticFutThunkSpec;
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(*self.borrow());
      ctx_pop_thunk(op)
    })
  }

  #[track_caller]
  fn sigmoid(&self) -> CellPtr {
    panick_wrap(|| self.logistic())
  }

  #[track_caller]
  fn standard_silu(&self) -> CellPtr {
    panick_wrap(|| {
      let op = StandardSiluFutThunkSpec;
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(*self.borrow());
      ctx_pop_thunk(op)
    })
  }

  #[track_caller]
  fn standard_swish(&self) -> CellPtr {
    panick_wrap(|| self.standard_silu())
  }

  #[track_caller]
  fn tanh(&self) -> CellPtr {
    panick_wrap(|| {
      let op = TanhFutThunkSpec;
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(*self.borrow());
      ctx_pop_thunk(op)
    })
  }

  #[track_caller]
  fn powi(&self, exp: i64) -> CellPtr {
    panick_wrap(|| {
      let p = *self.borrow();
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(p);
      match ctx_lookup_dtype(p) {
        // FIXME FIXME
        Dtype::Fp32 => {
          let op = PowiF32FutThunkSpec{exp};
          ctx_pop_thunk(op)
        }
        _ => unimplemented!()
      }
    })
  }

  #[track_caller]
  fn log(&self) -> CellPtr {
    panick_wrap(|| {
      let op = LogFutThunkSpec;
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(*self.borrow());
      ctx_pop_thunk(op)
    })
  }

  #[track_caller]
  fn ln(&self) -> CellPtr {
    panick_wrap(|| self.log())
  }

  #[track_caller]
  fn inner_max(&self) -> CellPtr {
    panick_wrap(|| {
      unimplemented!();
      /*let op = InnerMaxFutThunkSpec;
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(*self.borrow());
      ctx_pop_thunk(op)*/
    })
  }

  #[track_caller]
  fn inner_sum(&self) -> CellPtr {
    panick_wrap(|| {
      unimplemented!();
      /*let op = InnerSumFutThunkSpec;
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(*self.borrow());
      ctx_pop_thunk(op)*/
    })
  }

  #[track_caller]
  fn inner_mean(&self) -> CellPtr {
    panick_wrap(|| {
      let op = InnerMeanFutThunkSpec;
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(*self.borrow());
      ctx_pop_thunk(op)
    })
  }

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
  fn inner_symplectic_map(&self) -> CellPtr {
    panick_wrap(|| {
      let op = InnerSymplecticMapFutThunkSpec;
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(*self.borrow());
      ctx_pop_thunk(op)
    })
  }

  /*#[track_caller]
  fn inner_transpose(&self) -> CellPtr {
    panick_wrap(|| {
      let op = InnerTransposeFutThunkSpec;
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(*self.borrow());
      ctx_pop_thunk(op)
    })
  }*/

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

  #[track_caller]
  fn flat_sum(&self) -> CellPtr {
    panick_wrap(|| {
      let op = FlatSumFutThunkSpec;
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(*self.borrow());
      ctx_pop_thunk(op)
      /*// FIXME FIXME
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
      }*/
    })
  }

  #[track_caller]
  fn block_pad<T: IntoScalarValExt>(&self, new_block: [i64; 2], pad_val: T) -> CellPtr where Self: Sized {
    panick_wrap(|| {
      let x = *self.borrow();
      let x_ty = ctx_lookup_type(x);
      let x_nd = x_ty.ndim() as usize;
      assert!(x_nd >= 3);
      let org_block = [x_ty.shape[x_nd - 3], x_ty.shape[x_nd - 1]];
      //println!("DEBUG: block_pad: org block={:?} new_block={:?}", org_block, new_block);
      if org_block == new_block {
        //println!("DEBUG: block_unpad:   snapshot");
        return x.snapshot();
      }
      let pad_val = pad_val.into_scalar_val_();
      let op = BlockPadFutThunkSpec{org_block, new_block, pad_val};
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(x);
      ctx_pop_thunk(op)
    })
  }

  #[track_caller]
  fn block_unpad(&self, new_block: [i64; 2]) -> CellPtr {
    panick_wrap(|| {
      let x = *self.borrow();
      let x_ty = ctx_lookup_type(x);
      let x_nd = x_ty.ndim() as usize;
      assert!(x_nd >= 3);
      let org_block = [x_ty.shape[x_nd - 3], x_ty.shape[x_nd - 1]];
      //println!("DEBUG: block_unpad: org block={:?} new_block={:?}", org_block, new_block);
      if org_block == new_block {
        //println!("DEBUG: block_unpad:   snapshot");
        return x.snapshot();
      }
      let pad_val = ScalarVal_::Bot;
      let op = BlockPadFutThunkSpec{org_block, new_block, pad_val};
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(x);
      ctx_pop_thunk(op)
    })
  }

  #[track_caller]
  fn block_tri_elem_affine<T: IntoScalarValExt>(&self, diag_scale: T, diag_shift: T, lo_scale: T, lo_shift: T, up_scale: T, up_shift: T) -> CellPtr where Self: Sized {
    panick_wrap(|| {
      let diag_scale = diag_scale.into_scalar_val_();
      let diag_shift = diag_shift.into_scalar_val_();
      let lo_scale = lo_scale.into_scalar_val_();
      let lo_shift = lo_shift.into_scalar_val_();
      let up_scale = up_scale.into_scalar_val_();
      let up_shift = up_shift.into_scalar_val_();
      let op = BlockTriElemAffineFutThunkSpec{
        diag_scale,
        diag_shift,
        lo_scale,
        lo_shift,
        up_scale,
        up_shift,
      };
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(*self.borrow());
      ctx_pop_thunk(op)
    })
  }
}

impl<P: Borrow<CellPtr>> MathUnaryOps for P {}

pub fn zeros<S: Into<Vec<i64>>, D: Into<Dtype>>(shape: S, dtype: D) -> CellPtr {
  // FIXME
  unimplemented!();
  /*panick_wrap(|| {
    let ty = CellType{shape: shape.into(), dtype: dtype.into()};
    assert!(ctx_clean_arg());
    ctx_pop_thunk_(SetScalarFutThunkSpec{val: zero()}, ty)
  })*/
}

pub fn ones<S: Into<Vec<i64>>, D: Into<Dtype>>(shape: S, dtype: D) -> CellPtr {
  // FIXME
  unimplemented!();
  /*panick_wrap(|| {
    let ty = CellType{shape: shape.into(), dtype: dtype.into()};
    assert!(ctx_clean_arg());
    ctx_pop_thunk_(SetScalarFutThunkSpec{val: one()}, ty)
  })*/
}

pub fn iota(len: i64) -> CellPtr {
  panick_wrap(|| {
    assert!(ctx_clean_arg());
    ctx_pop_thunk(IotaFutThunkSpec{len})
  })
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
        /*(Dtype::Fp32, Dtype::Fp16) |
        (Dtype::Fp16, Dtype::Fp32) => {
          let op = CastFutThunkSpec{org_dtype, new_dtype};
          assert!(ctx_clean_arg());
          ctx_push_cell_arg(x);
          ctx_pop_thunk(op)
        }*/
        (Dtype::Fp32, Dtype::Bfloat16) => {
          let op = CastF32Bf16FutThunkSpec;
          assert!(ctx_clean_arg());
          ctx_push_cell_arg(x);
          ctx_pop_thunk(op)
        }
        (Dtype::Bfloat16, Dtype::Fp16) => {
          let op = CastBf16F16FutThunkSpec;
          assert!(ctx_clean_arg());
          ctx_push_cell_arg(x);
          ctx_pop_thunk(op)
        }
        (Dtype::Bfloat16, Dtype::Fp32) => {
          let op = CastBf16F32FutThunkSpec;
          assert!(ctx_clean_arg());
          ctx_push_cell_arg(x);
          ctx_pop_thunk(op)
        }
        _ => {
          // FIXME: other bf16 special cases.
          let op = CastFutThunkSpec{new_dtype};
          assert!(ctx_clean_arg());
          ctx_push_cell_arg(x);
          ctx_pop_thunk(op)
        }
        //_ => unimplemented!()
      }
    })
  }
}

impl<L: Borrow<CellPtr>> CastOps for L {}

/*pub trait GradOps<R: Borrow<CellPtr>>: Borrow<CellPtr> {
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

impl<L: Borrow<CellPtr>, R: Borrow<CellPtr>> GradOps<R> for L {}*/

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
  fn _unpack(&self) -> Option<(CellViewType, CellPtr)> {
    // FIXME
    unimplemented!();
  }

  #[track_caller]
  fn _unview(&self) -> (CellViewType, CellPtr, /*Vec<CellVOp>*/) {
    // FIXME
    unimplemented!();
  }
}

impl<L: Borrow<CellPtr> + Sized> ArrayOps for L {}

pub trait Ops_: Borrow<CellPtr> + Sized {
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

impl<L: Borrow<CellPtr> + Sized> Ops_ for L {}

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

  /*#[track_caller]
  fn apply_futhark(&self, lam_src: Cow<'static, str>) -> CellPtr {
    apply_futhark(lam_src, &[self.borrow()])
  }*/

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

  /*#[track_caller]
  fn set<X: AsRef<CellPtr>>(&self, _x: X) where Self: Sized {
    unimplemented!();
  }*/

  /*#[track_caller]
  fn unsafe_unseal(self) -> Self {
    unimplemented!();
  }*/

  /*#[track_caller]
  fn unsafe_unseal_init(self) -> Self {
    panick_wrap(|| TL_CTX.with(|ctx| {
      let mut spine = ctx.spine.borrow_mut();
      spine.unseal_mux(*self.borrow());
      self
    }))
  }*/

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
  fn snapshot(&self) -> CellPtr {
    panick_wrap(|| ctx_snapshot(*self.borrow()))
  }

  /*#[track_caller]
  fn checkpoint(self) -> CellPtr {
    unimplemented!();
  }*/
}

impl<L: Borrow<CellPtr> + Sized> Ops for L {}

pub trait CtlOps: Borrow<CellPtr> + Sized {
  #[track_caller]
  fn version(&self) -> Clock {
    panick_wrap(|| ctx_lookup_clk(*self.borrow()))
  }

  #[track_caller]
  fn _get_mem(&self) -> MemReg {
    panick_wrap(|| TL_CTX.with(|ctx| {
      let x = *self.borrow();
      let xclk = ctx_lookup_clk(x);
      match ctx.env.borrow_mut().pread_ref(x, xclk) {
        None => panic!("bug"),
        Some(e) => {
          match e.cel_ {
            &mut Cell_::Phy(.., ref mut cel_) => {
              // FIXME FIXME
              let pm = PMach::NvGpu;
              let addr = cel_.get(x, xclk, &e.ty, Locus::Mem, pm);
              TL_PCTX.with(|pctx| {
                let (_, icel) = pctx.lookup_pm(pm, addr).unwrap();
                icel.as_mem_reg().unwrap()
              })
            }
            _ => panic!("bug")
          }
        }
      }
    }))
  }
}

impl<L: Borrow<CellPtr> + Sized> CtlOps for L {}

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
