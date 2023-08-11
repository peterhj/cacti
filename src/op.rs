//use crate::cell::{CellPtr, StableCell, CellType, Dtype, ScalarVal_, IntoScalarValExt, CellState};
use crate::cell::*;
use crate::clock::{Clock};
use crate::ctx::*;
use crate::panick::*;
use crate::pctx::{TL_PCTX, Locus, PMach, MemReg};
use crate::spine::{SpineRet, SpineEntry, SpineEntryName};
use crate::thunk::{FutharkNdBroadcastMap2MonomorphicSpec};
use crate::thunk::op::*;
use cacti_cfg_env::*;

//use futhark_syntax::*;

use std::borrow::{Borrow};
use std::convert::{TryInto};
use std::iter::{repeat};
use std::ops::{AddAssign, BitXor, Index, IndexMut, Add, Sub, Mul, Div, Neg};
use std::rc::{Rc};

impl<R: CellDeref> AddAssign<R> for CellPtr {
  #[track_caller]
  fn add_assign(&mut self, rhs: R) {
    panick_wrap(|| {
      // FIXME: alias semantics.
      let this = *self;
      let rhs = rhs._deref();
      if this.is_nil() {
        // FIXME: depending on set flags, clobber the rhs entry to a noop.
        return;
      }
      if TL_CTX.with(|ctx| {
        let mut success = false;
        if ctx.ctlstate.borrow().accumulate_in_place.get() {
          //println!("DEBUG: AddAssign::add_assign: try: enter: this={:?} rhs={:?}", this, rhs);
          let spine = ctx.spine.borrow();
          let (this_root, rhs_root) = {
            let cur_env = spine.cur_env.borrow();
            let this_root = cur_env._deref(this);
            let rhs_root = cur_env._deref(rhs);
            (this_root, rhs_root)
          };
          //println!("DEBUG: AddAssign::add_assign: try:   this root={:?} rhs root={:?}", this_root, rhs_root);
          let base_clk = spine._counter();
          let mut this_clk = spine._version(this).unwrap_or_else(|| Clock::default());
          let rhs_clk = spine._version(rhs).unwrap();
          //println!("DEBUG: AddAssign::add_assign: try:   this clk ={:?} rhs clk ={:?} init once? {:?}", this_clk, rhs_clk, rhs_clk.is_init_once());
          if rhs_clk.is_init_once() {
            if ctx.ctlstate.assume_uninit_zero.get() {
              this_clk = base_clk.max(this_clk).init_or_update();
            } else {
              this_clk = base_clk.max(this_clk).update();
            }
            let bp = spine.curp.get();
            assert_eq!(bp, spine.log.borrow().len() as _);
            let mut state: u8 = 0;
            for sp in (0 .. bp).rev() {
              let e_sp = spine.log.borrow()[sp as usize];
              match (state, e_sp) {
                (0, SpineEntry::Cache(y, ..)) |
                (0, SpineEntry::YieldSet(y, ..)) |
                (0, SpineEntry::YieldInit(y, ..)) |
                (0, SpineEntry::Initialize(y, ..)) => {
                  let yroot = spine.cur_env.borrow()._deref(y);
                  if yroot == rhs_root {
                    //println!("DEBUG: AddAssign::add_assign: try:   e={:?} y == rhs", e_sp.name());
                    break;
                  }
                }
                (0, SpineEntry::Apply(y, yclk, th)) => {
                  let yroot = spine.cur_env.borrow()._deref(y);
                  if yroot == rhs_root {
                    //println!("DEBUG: AddAssign::add_assign: try:   e={:?} y == rhs", e_sp.name());
                    assert_eq!(yclk, rhs_clk);
                    // FIXME FIXME: the rewriting below could use some fixing up.
                    spine.log.borrow_mut()[sp as usize] = match e_sp.name() {
                      SpineEntryName::Initialize => {
                        SpineEntry::Initialize(this, this_clk, th)
                      }
                      SpineEntryName::Apply => {
                        if this_clk.is_update() {
                          if cfg_debug() { println!("DEBUG: AddAssign::add_assign: rewrite Apply({:?}, {:?}, {:?}) -> Accumulate({:?}, {:?}, {:?})", y, yclk, th, this, this_clk, th); }
                          SpineEntry::Accumulate(this, this_clk, th)
                        } else if this_clk.is_init_once() {
                          if cfg_debug() { println!("DEBUG: AddAssign::add_assign: rewrite Apply({:?}, {:?}, {:?}) -> Apply({:?}, {:?}, {:?})", y, yclk, th, this, this_clk, th); }
                          SpineEntry::Apply(this, this_clk, th)
                        } else {
                          unreachable!();
                        }
                      }
                      _ => unreachable!()
                    };
                    let mut cur_env = spine.cur_env.borrow_mut();
                    match cur_env._lookup_mut(rhs) {
                      None => panic!("bug"),
                      Some((_, state)) => {
                        // FIXME: sealing here is kinda hacky.
                        //state.flag.set_seal();
                        //state.flag.unset_intro();
                        assert!(!state.clk.is_uninit());
                        state.clk = state.clk.uninit();
                      }
                    }
                    match cur_env._lookup_mut(this) {
                      None => {
                        //let this_root = cur_env._deref(this);
                        let mut state = CellState::default();
                        //state.flag.set_intro();
                        assert!(this_clk.is_init_once());
                        assert!(state.clk < this_clk);
                        state.clk = this_clk;
                        assert!(cur_env.state.insert(this_root, state.into()).is_none());
                      }
                      Some((_, state)) => {
                        //state.flag.set_intro();
                        //assert!(state.flag.intro());
                        //assert!(!state.flag.seal());
                        //state.flag.unset_seal();
                        /*if !this_clk.is_update() {
                          println!("DEBUG: AddAssign::add_assign: this={:?} this_clk={:?} state.clk={:?}",
                              this, this_clk, state.clk);
                        }
                        assert!(this_clk.is_update());*/
                        assert!(this_clk.is_init_once() || this_clk.is_update());
                        assert!(state.clk < this_clk);
                        state.clk = this_clk;
                      }
                    }
                    //let this_root = cur_env._deref(this);
                    //let rhs_root = cur_env._deref(rhs);
                    let (orhs, arg) = cur_env.update.remove(&(rhs_root, rhs_clk)).unwrap();
                    //assert_eq!(orhs, rhs);
                    assert!(cur_env.update.insert((this_root, this_clk), (this, arg)).is_none());
                    drop(cur_env);
                    // FIXME
                    let mut thunkenv = ctx.thunkenv.borrow_mut();
                    let mut tclo = thunkenv.update.remove(&(rhs, rhs_clk)).unwrap();
                    //assert_eq!(tclo.out, rhs);
                    tclo.out = this;
                    assert!(thunkenv.update.insert((this_root, this_clk), tclo).is_none());
                    drop(thunkenv);
                    state = 1;
                    break;
                  }
                }
                (0, SpineEntry::Accumulate(y, ..)) => {
                  let yroot = spine.cur_env.borrow()._deref(y);
                  if yroot == rhs_root {
                    panic!("bug");
                  }
                }
                (0, SpineEntry::UnsafeWrite(y, ..)) => {
                  let yroot = spine.cur_env.borrow()._deref(y);
                  if yroot == rhs_root {
                    unimplemented!();
                  }
                }
                _ => {}
              }
            }
            match state {
              0 => {}
              //1 => panic!("bug"),
              1 => {
                // FIXME: should seal rhs.
                /*
                spine.seal(rhs);
                */
                success = true;
              }
              _ => unreachable!()
            }
          }
        }
        success
      }) {
        return;
      }
      //println!("DEBUG: AddAssign::add_assign: fallback: this={:?} rhs={:?}", this, rhs);
      TL_CTX.with(|ctx| {
      /*println!("DEBUG: AddAssign::add_assign: fallback:   {:?}",
          ctx.ctlstate.accumulate_in_place.get());
      println!("DEBUG: AddAssign::add_assign: fallback:   {:?}",
        ctx.ctlstate.assume_uninit_zero.get());*/
        let spine = ctx.spine.borrow();
        let base_clk = spine._counter();
        let mut this_clk = spine._version(this).unwrap_or_else(|| Clock::default());
        this_clk = base_clk.max(this_clk).init_or_update();
        if this_clk.is_update() {
          let op = IdentityFutThunkSpec;
          assert!(ctx_clean_arg());
          ctx_push_cell_arg(rhs);
          ctx_pop_accumulate_thunk(op, this)
        } else if this_clk.is_init_once() {
          if ctx.ctlstate.assume_uninit_zero.get() {
            // FIXME: could just snapshot.
            //let op = IdentityFutThunkSpec;
            let op = MemcpyThunkSpec;
            assert!(ctx_clean_arg());
            ctx_push_cell_arg(rhs);
            ctx_pop_apply_thunk(op, this)
          } else {
            unimplemented!();
          }
        } else {
          unreachable!();
        }
      });
    })
  }
}

impl<'l, R: CellDeref> AddAssign<R> for &'l mut CellPtr {
  #[track_caller]
  fn add_assign(&mut self, rhs: R) {
    panick_wrap(|| (*self).add_assign(rhs))
  }
}

impl<R: CellDeref> AddAssign<R> for StableCell {
  #[track_caller]
  fn add_assign(&mut self, rhs: R) {
    panick_wrap(|| self.as_ptr_mut().add_assign(rhs))
  }
}

impl<'l, R: CellDeref> AddAssign<R> for &'l mut StableCell {
  #[track_caller]
  fn add_assign(&mut self, rhs: R) {
    panick_wrap(|| self.as_ptr_mut().add_assign(rhs))
  }
}

impl<R: CellDeref> AddAssign<R> for CellViewHandle_ {
  #[track_caller]
  fn add_assign(&mut self, rhs: R) {
    panick_wrap(|| self._deref().add_assign(rhs))
  }
}

impl<'l, R: CellDeref> AddAssign<R> for &'l mut CellViewHandle_ {
  #[track_caller]
  fn add_assign(&mut self, rhs: R) {
    panick_wrap(|| self._deref().add_assign(rhs))
  }
}

#[derive(Clone, Copy, Debug)]
pub struct T;

#[derive(Clone, Copy, Debug)]
pub struct T_(pub i8, pub i8);

impl<'l> BitXor<T> for &'l CellPtr {
  type Output = CellPtr;

  #[track_caller]
  fn bitxor(self, _: T) -> CellPtr {
    panick_wrap(|| {
      let this = self._deref();
      let view = TL_CTX.with(|ctx| {
        ctx.alias_view_swap(this, -2, -1)
      });
      view
    })
  }
}

impl<'l> BitXor<T_> for &'l CellPtr {
  type Output = CellPtr;

  #[track_caller]
  fn bitxor(self, t: T_) -> CellPtr {
    panick_wrap(|| {
      let this = self._deref();
      let view = TL_CTX.with(|ctx| {
        ctx.alias_view_swap(this, t.0, t.1)
      });
      view
    })
  }
}

impl BitXor<T> for CellPtr {
  type Output = CellPtr;

  #[track_caller]
  fn bitxor(self, _: T) -> CellPtr {
    panick_wrap(|| {
      let this = self._deref();
      let view = TL_CTX.with(|ctx| {
        ctx.alias_view_swap(this, -2, -1)
      });
      view
    })
  }
}

impl BitXor<T_> for CellPtr {
  type Output = CellPtr;

  #[track_caller]
  fn bitxor(self, t: T_) -> CellPtr {
    panick_wrap(|| {
      let this = self._deref();
      let view = TL_CTX.with(|ctx| {
        ctx.alias_view_swap(this, t.0, t.1)
      });
      view
    })
  }
}

impl<'l> BitXor<T> for &'l StableCell {
  type Output = CellPtr;

  #[track_caller]
  fn bitxor(self, _: T) -> CellPtr {
    panick_wrap(|| self.as_ptr_ref().bitxor(T))
  }
}

impl<'l> BitXor<T_> for &'l StableCell {
  type Output = CellPtr;

  #[track_caller]
  fn bitxor(self, t: T_) -> CellPtr {
    panick_wrap(|| self.as_ptr_ref().bitxor(t))
  }
}

impl BitXor<T> for StableCell {
  type Output = CellPtr;

  #[track_caller]
  fn bitxor(self, _: T) -> CellPtr {
    panick_wrap(|| {
      let this = self._deref();
      let view = TL_CTX.with(|ctx| {
        ctx.alias_view_swap(this, -2, -1)
      });
      view
    })
  }
}

impl BitXor<T_> for StableCell {
  type Output = CellPtr;

  #[track_caller]
  fn bitxor(self, t: T_) -> CellPtr {
    panick_wrap(|| {
      let this = self._deref();
      let view = TL_CTX.with(|ctx| {
        ctx.alias_view_swap(this, t.0, t.1)
      });
      view
    })
  }
}

impl<'l> BitXor<T> for &'l CellViewHandle {
  type Output = CellPtr;

  #[track_caller]
  fn bitxor(mut self, _: T) -> CellPtr {
    panick_wrap(|| {
      let this = self._deref();
      let view = TL_CTX.with(|ctx| {
        ctx.alias_view_swap(this, -2, -1)
      });
      view
    })
  }
}

impl<'l> BitXor<T_> for &'l CellViewHandle {
  type Output = CellPtr;

  #[track_caller]
  fn bitxor(self, t: T_) -> CellPtr {
    panick_wrap(|| {
      let this = self._deref();
      let view = TL_CTX.with(|ctx| {
        ctx.alias_view_swap(this, t.0, t.1)
      });
      view
    })
  }
}

impl<'l> Add<ScalarVal_> for &'l CellPtr {
  type Output = CellPtr;

  #[track_caller]
  fn add(self, rhs: ScalarVal_) -> CellPtr {
    panick_wrap(|| {
      let op = AddScalarFutThunkSpec{val: rhs};
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(*self);
      ctx_pop_thunk(op)
    })
  }
}

impl Add<ScalarVal_> for CellPtr {
  type Output = CellPtr;

  #[track_caller]
  fn add(self, rhs: ScalarVal_) -> CellPtr {
    panick_wrap(|| (&self).add(rhs))
  }
}

impl<'l> Add<ScalarVal_> for &'l StableCell {
  type Output = CellPtr;

  #[track_caller]
  fn add(self, rhs: ScalarVal_) -> CellPtr {
    panick_wrap(|| self.as_ptr_ref().add(rhs))
  }
}

impl Add<ScalarVal_> for StableCell {
  type Output = CellPtr;

  #[track_caller]
  fn add(self, rhs: ScalarVal_) -> CellPtr {
    panick_wrap(|| self.as_ptr_ref().add(rhs))
  }
}

impl<'l> Add<f32> for &'l CellPtr {
  type Output = CellPtr;

  #[track_caller]
  fn add(self, rhs: f32) -> CellPtr {
    panick_wrap(|| {
      let op = AddScalarFutThunkSpec{val: rhs.into_scalar_val_()};
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

impl<'l, R: CellDeref> Add<R> for &'l CellPtr {
  type Output = CellPtr;

  #[track_caller]
  fn add(self, rhs: R) -> CellPtr {
    panick_wrap(|| {
      let x0 = *self;
      let x1 = rhs._deref();
      let x0_ty = x0.type_();
      let x1_ty = x1.type_();
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(x0);
      ctx_push_cell_arg(x1);
      if &x0_ty.shape != &x1_ty.shape || x0_ty.dtype != x1_ty.dtype {
        let mono = FutharkNdBroadcastMap2MonomorphicSpec::from2(&x0_ty, &x1_ty);
        ctx_pop_thunk(BroadcastAddFutThunkSpec{mono})
      } else {
        ctx_pop_thunk(AddFutThunkSpec)
      }
    })
  }
}

impl<R: CellDeref> Add<R> for CellPtr {
  type Output = CellPtr;

  #[track_caller]
  fn add(self, rhs: R) -> CellPtr {
    panick_wrap(|| (&self).add(rhs))
  }
}

impl<'l, R: CellDeref> Add<R> for &'l StableCell {
  type Output = CellPtr;

  #[track_caller]
  fn add(self, rhs: R) -> CellPtr {
    panick_wrap(|| self.as_ptr_ref().add(rhs))
  }
}

impl<R: CellDeref> Add<R> for StableCell {
  type Output = CellPtr;

  #[track_caller]
  fn add(self, rhs: R) -> CellPtr {
    panick_wrap(|| self.as_ptr_ref().add(rhs))
  }
}

impl<'l, R: CellDeref> Sub<R> for &'l ScalarVal_ {
  type Output = CellPtr;

  #[track_caller]
  fn sub(self, rhs: R) -> CellPtr {
    panick_wrap(|| {
      let op = LSubScalarFutThunkSpec{val: self.into_scalar_val_()};
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(rhs._deref());
      ctx_pop_thunk(op)
    })
  }
}

impl<R: CellDeref> Sub<R> for ScalarVal_ {
  type Output = CellPtr;

  #[track_caller]
  fn sub(self, rhs: R) -> CellPtr {
    panick_wrap(|| (&self).sub(rhs))
  }
}

impl<'l> Sub<CellPtr> for &'l f32 {
  type Output = CellPtr;

  #[track_caller]
  fn sub(self, rhs: CellPtr) -> CellPtr {
    panick_wrap(|| {
      let op = LSubScalarFutThunkSpec{val: self.into_scalar_val_()};
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
      let op = RSubScalarFutThunkSpec{val: rhs.into_scalar_val_()};
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

impl<'l, R: CellDeref> Sub<R> for &'l CellPtr {
  type Output = CellPtr;

  #[track_caller]
  fn sub(self, rhs: R) -> CellPtr {
    panick_wrap(|| {
      let x0 = *self.borrow();
      let x1 = rhs._deref();
      let x0_ty = x0.type_();
      let x1_ty = x1.type_();
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(x0);
      ctx_push_cell_arg(x1);
      if &x0_ty.shape != &x1_ty.shape || x0_ty.dtype != x1_ty.dtype {
        let mono = FutharkNdBroadcastMap2MonomorphicSpec::from2(&x0_ty, &x1_ty);
        ctx_pop_thunk(BroadcastSubFutThunkSpec{mono})
      } else {
        ctx_pop_thunk(SubFutThunkSpec)
      }
    })
  }
}

impl<R: CellDeref> Sub<R> for CellPtr {
  type Output = CellPtr;

  #[track_caller]
  fn sub(self, rhs: R) -> CellPtr {
    panick_wrap(|| (&self).sub(rhs))
  }
}

impl<'l, R: CellDeref> Sub<R> for &'l StableCell {
  type Output = CellPtr;

  #[track_caller]
  fn sub(self, rhs: R) -> CellPtr {
    panick_wrap(|| self.as_ptr_ref().sub(rhs))
  }
}

impl<R: CellDeref> Sub<R> for StableCell {
  type Output = CellPtr;

  #[track_caller]
  fn sub(self, rhs: R) -> CellPtr {
    panick_wrap(|| self.as_ptr_ref().sub(rhs))
  }
}

impl<'l> Mul<ScalarVal_> for &'l CellPtr {
  type Output = CellPtr;

  #[track_caller]
  fn mul(self, rhs: ScalarVal_) -> CellPtr {
    panick_wrap(|| {
      let op = MulScalarFutThunkSpec{val: rhs};
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(*self);
      ctx_pop_thunk(op)
    })
  }
}

impl Mul<ScalarVal_> for CellPtr {
  type Output = CellPtr;

  #[track_caller]
  fn mul(self, rhs: ScalarVal_) -> CellPtr {
    panick_wrap(|| (&self).mul(rhs))
  }
}

impl<'l> Mul<f32> for &'l CellPtr {
  type Output = CellPtr;

  #[track_caller]
  fn mul(self, rhs: f32) -> CellPtr {
    panick_wrap(|| {
      let op = MulScalarFutThunkSpec{val: rhs.into_scalar_val_()};
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(*self);
      ctx_pop_thunk(op)
    })
  }
}

impl Mul<f32> for CellPtr {
  type Output = CellPtr;

  #[track_caller]
  fn mul(self, rhs: f32) -> CellPtr {
    panick_wrap(|| (&self).mul(rhs))
  }
}

impl<'l, R: CellDeref> Mul<R> for &'l CellPtr {
  type Output = CellPtr;

  #[track_caller]
  fn mul(self, rhs: R) -> CellPtr {
    panick_wrap(|| {
      let x0 = *self.borrow();
      let x1 = rhs._deref();
      let x0_ty = x0.type_();
      let x1_ty = x1.type_();
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(x0);
      ctx_push_cell_arg(x1);
      if &x0_ty.shape != &x1_ty.shape || x0_ty.dtype != x1_ty.dtype {
        let mono = FutharkNdBroadcastMap2MonomorphicSpec::from2(&x0_ty, &x1_ty);
        ctx_pop_thunk(BroadcastMulFutThunkSpec{mono})
      } else {
        ctx_pop_thunk(MulFutThunkSpec)
      }
    })
  }
}

impl<R: CellDeref> Mul<R> for CellPtr {
  type Output = CellPtr;

  #[track_caller]
  fn mul(self, rhs: R) -> CellPtr {
    panick_wrap(|| (&self).mul(rhs))
  }
}

impl<'l, R: CellDeref> Mul<R> for &'l StableCell {
  type Output = CellPtr;

  #[track_caller]
  fn mul(self, rhs: R) -> CellPtr {
    panick_wrap(|| self.as_ptr_ref().mul(rhs))
  }
}

impl<R: CellDeref> Mul<R> for StableCell {
  type Output = CellPtr;

  #[track_caller]
  fn mul(self, rhs: R) -> CellPtr {
    panick_wrap(|| self.as_ptr_ref().mul(rhs))
  }
}

impl<'l, R: CellDeref> Div<R> for &'l ScalarVal_ {
  type Output = CellPtr;

  #[track_caller]
  fn div(self, rhs: R) -> CellPtr {
    panick_wrap(|| {
      let op = LDivScalarFutThunkSpec{val: self.into_scalar_val_()};
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(rhs._deref());
      ctx_pop_thunk(op)
    })
  }
}

impl<R: CellDeref> Div<R> for ScalarVal_ {
  type Output = CellPtr;

  #[track_caller]
  fn div(self, rhs: R) -> CellPtr {
    panick_wrap(|| (&self).div(rhs))
  }
}

impl<'l> Div<CellPtr> for &'l f32 {
  type Output = CellPtr;

  #[track_caller]
  fn div(self, rhs: CellPtr) -> CellPtr {
    panick_wrap(|| {
      let op = LDivScalarFutThunkSpec{val: self.into_scalar_val_()};
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(rhs);
      ctx_pop_thunk(op)
    })
  }
}

impl Div<CellPtr> for f32 {
  type Output = CellPtr;

  #[track_caller]
  fn div(self, rhs: CellPtr) -> CellPtr {
    panick_wrap(|| (&self).div(rhs))
  }
}

impl<'l> Div<ScalarVal_> for &'l CellPtr {
  type Output = CellPtr;

  #[track_caller]
  fn div(self, rhs: ScalarVal_) -> CellPtr {
    panick_wrap(|| {
      let op = RDivScalarFutThunkSpec{val: rhs.into_scalar_val_()};
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(*self.borrow());
      ctx_pop_thunk(op)
    })
  }
}

impl Div<ScalarVal_> for CellPtr {
  type Output = CellPtr;

  #[track_caller]
  fn div(self, rhs: ScalarVal_) -> CellPtr {
    panick_wrap(|| (&self).div(rhs))
  }
}

impl<'l> Div<f32> for &'l CellPtr {
  type Output = CellPtr;

  #[track_caller]
  fn div(self, rhs: f32) -> CellPtr {
    panick_wrap(|| {
      let op = RDivScalarFutThunkSpec{val: rhs.into_scalar_val_()};
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

impl<'l, R: CellDeref> Div<R> for &'l CellPtr {
  type Output = CellPtr;

  #[track_caller]
  fn div(self, rhs: R) -> CellPtr {
    panick_wrap(|| {
      let x0 = *self;
      let x1 = rhs._deref();
      let x0_ty = x0.type_();
      let x1_ty = x1.type_();
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(x0);
      ctx_push_cell_arg(x1);
      if &x0_ty.shape != &x1_ty.shape || x0_ty.dtype != x1_ty.dtype {
        let mono = FutharkNdBroadcastMap2MonomorphicSpec::from2(&x0_ty, &x1_ty);
        ctx_pop_thunk(BroadcastDivFutThunkSpec{mono})
      } else {
        ctx_pop_thunk(DivFutThunkSpec)
      }
    })
  }
}

impl<R: CellDeref> Div<R> for CellPtr {
  type Output = CellPtr;

  #[track_caller]
  fn div(self, rhs: R) -> CellPtr {
    panick_wrap(|| (&self).div(rhs))
  }
}

impl<'l, R: CellDeref> Div<R> for &'l StableCell {
  type Output = CellPtr;

  #[track_caller]
  fn div(self, rhs: R) -> CellPtr {
    panick_wrap(|| self.as_ptr_ref().div(rhs))
  }
}

impl<R: CellDeref> Div<R> for StableCell {
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
        Dtype::F16 => {
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

pub trait MathInitOps: CellDeref {
  #[track_caller]
  fn init_online_add_scale2<V: CellDeref, T: IntoScalarValExt>(&self, val: V, src_scale: T, dst_scale: T) {
    panick_wrap(|| {
      let this = self._deref();
      let val = val._deref();
      let src_scale = src_scale.into_scalar_val_();
      let dst_scale = dst_scale.into_scalar_val_();
      let ty = ctx_lookup_type(this);
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(val);
      ctx_pop_initialize_thunk_(OnlineAddScale2InitFutThunkSpec{src_scale, dst_scale}, this, ty)
    })
  }

  #[track_caller]
  fn init_online_average_scale<V: CellDeref, T: IntoScalarValExt>(&self, val: V, src_scale: T, rate: T) {
    panick_wrap(|| {
      let this = self._deref();
      let val = val._deref();
      let src_scale = src_scale.into_scalar_val_();
      let rate = rate.into_scalar_val_();
      let ty = ctx_lookup_type(this);
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(val);
      ctx_pop_initialize_thunk_(OnlineAverageScaleInitFutThunkSpec{src_scale, rate}, this, ty)
    })
  }

  #[track_caller]
  fn init_online_average_square_scale<V: CellDeref, T: IntoScalarValExt>(&self, val: V, src_scale: T, rate: T) {
    panick_wrap(|| {
      let this = self._deref();
      let val = val._deref();
      let src_scale = src_scale.into_scalar_val_();
      let rate = rate.into_scalar_val_();
      let ty = ctx_lookup_type(this);
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(val);
      ctx_pop_initialize_thunk_(OnlineAverageSquareScaleInitFutThunkSpec{src_scale, rate}, this, ty)
    })
  }

  #[track_caller]
  fn init_online_adamw_update32<V: CellDeref>(&self, grad1_avg: V, grad2_avg: V, iter_nr: i32, lr: f32, wd: f32, a1: f32, a2: f32, eps: f32) {
    let unbias1 = if iter_nr <= 0 {
      println!("ERROR: init_online_adamw_update32: invalid iter_nr={} (should be positive)", iter_nr);
      panic!();
    } else if iter_nr == 1 {
      a1
    } else {
      1.0 - (1.0 - a1).powi(iter_nr)
    };
    let unbias2 = if iter_nr <= 0 {
      println!("ERROR: init_online_adamw_update32: invalid iter_nr={} (should be positive)", iter_nr);
      panic!();
    } else if iter_nr == 1 {
      a2.sqrt()
    } else {
      (1.0 - (1.0 - a2).powi(iter_nr)).sqrt()
    };
    panick_wrap(|| {
      let this = self._deref();
      let grad1_avg = grad1_avg._deref();
      let grad2_avg = grad2_avg._deref();
      let signed_lr = (-lr).into_scalar_val_();
      let lr_unbias = (unbias2 / unbias1).into_scalar_val_();
      let lamda = (wd).into_scalar_val_();
      let eps = (eps * unbias2).into_scalar_val_();
      if cfg_debug() {
      println!("DEBUG: init_online_adamw_update32: signed lr={:?} lamda={:?} eps={:?}",
          signed_lr, lamda, eps);
      }
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(grad1_avg);
      ctx_push_cell_arg(grad2_avg);
      ctx_pop_initialize_thunk(OnlineAdamWUpdateInitFutThunkSpec{signed_lr, lr_unbias, lamda, eps}, this)
    })
  }
}

impl<L: CellDeref + ?Sized> MathInitOps for L {}

pub trait MathSetOps: CellDeref {
  #[track_caller]
  fn set<R: CellDeref>(&self, rhs: R) {
    panick_wrap(|| {
      let this = self._deref();
      let rhs = rhs._deref();
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(rhs);
      ctx_pop_apply_thunk(IdentityFutThunkSpec, this)
    })
  }

  #[track_caller]
  fn set_cast<R: CellDeref>(&self, rhs: R) {
    panick_wrap(|| {
      let this = self._deref();
      let rhs = rhs._deref();
      let ty = ctx_lookup_type(this);
      let new_dtype = ty.dtype;
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(rhs);
      ctx_pop_apply_thunk(CastFutThunkSpec{new_dtype}, this)
    })
  }

  #[track_caller]
  fn set_zeros(&self) {
    panick_wrap(|| {
      let this = self._deref();
      let ty = ctx_lookup_type(this);
      assert!(ctx_clean_arg());
      let val = ScalarVal_::zero(ty.dtype);
      ctx_pop_apply_thunk_(SetScalarFutThunkSpec{val}, this, ty)
    })
  }

  #[track_caller]
  fn set_ones(&self) {
    panick_wrap(|| {
      let this = self._deref();
      let ty = ctx_lookup_type(this);
      assert!(ctx_clean_arg());
      let val = ScalarVal_::one(ty.dtype);
      ctx_pop_apply_thunk_(SetScalarFutThunkSpec{val}, this, ty)
    })
  }
}

impl<L: CellDeref + ?Sized> MathSetOps for L {}

pub trait MathBinaryOps<R: CellDeref>: CellDeref {
  /*#[track_caller]
  fn pow(&self, rhs: R) -> CellPtr {
    panick_wrap(|| {
      unimplemented!();
    })
  }*/

  #[track_caller]
  fn inner_concat(&self, rhs: R) -> CellPtr {
    panick_wrap(|| {
      let op = InnerConcatFutThunkSpec;
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(self._deref());
      ctx_push_cell_arg(rhs._deref());
      ctx_pop_thunk(op)
    })
  }

  #[track_caller]
  fn inner_select(&self, rank: R) -> CellPtr {
    panick_wrap(|| {
      let rank = rank._deref();
      let rank_dtype = ctx_lookup_dtype(rank);
      if !rank_dtype.is_uint() {
        panic!("ERROR: inner_select: invalid argument: expected dtype uint, actual {:?}", rank_dtype);
      }
      let op = InnerSelectFutThunkSpec;
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(self._deref());
      ctx_push_cell_arg(rank);
      ctx_pop_thunk(op)
    })
  }

  #[track_caller]
  fn inner_inv_select(&self, rank: R, inner_len: i64) -> CellPtr {
    panick_wrap(|| {
      let rank = rank._deref();
      let rank_dtype = ctx_lookup_dtype(rank);
      if !rank_dtype.is_uint() {
        panic!("ERROR: inner_inv_select: invalid argument: expected dtype uint, actual {:?}", rank_dtype);
      }
      let op = InnerInvSelectFutThunkSpec{inner_len};
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(self._deref());
      ctx_push_cell_arg(rank);
      ctx_pop_thunk(op)
    })
  }

  #[track_caller]
  fn outer_select(&self, rank: R) -> CellPtr {
    panick_wrap(|| {
      let rank = rank._deref();
      let rank_dtype = ctx_lookup_dtype(rank);
      if !rank_dtype.is_uint() {
        panic!("ERROR: outer_select: invalid argument: expected dtype uint, actual {:?}", rank_dtype);
      }
      let op = OuterSelectFutThunkSpec;
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(self._deref());
      ctx_push_cell_arg(rank);
      ctx_pop_thunk(op)
    })
  }

  #[track_caller]
  fn inner_softmax_categorical_nll(&self, rank: R) -> CellPtr {
    panick_wrap(|| {
      let op = InnerSoftmaxCategoricalNLLFutThunkSpec;
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(self._deref());
      ctx_push_cell_arg(rank._deref());
      ctx_pop_thunk(op)
    })
  }

  #[track_caller]
  fn outer_mul(&self, rhs: R) -> CellPtr {
    panick_wrap(|| {
      let op = OuterMulFutThunkSpec;
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(self._deref());
      ctx_push_cell_arg(rhs._deref());
      ctx_pop_thunk(op)
    })
  }

  #[track_caller]
  fn matmul(&self, l_t: bool, rhs: R, r_t: bool) -> CellPtr {
    panick_wrap(|| {
      // FIXME: scalar dtype.
      self.matmul_scale(l_t, rhs, r_t, 1.0_f32)
    })
  }

  #[track_caller]
  fn matmul_scale<T: IntoScalarValExt>(&self, l_t: bool, rhs: R, r_t: bool, scale: T) -> CellPtr {
    panick_wrap(|| {
      let p = self._deref();
      let q = rhs._deref();
      let p_ty = ctx_lookup_type(p);
      let q_ty = ctx_lookup_type(q);
      assert_eq!(p_ty.ndim(), 2);
      assert_eq!(q_ty.ndim(), 2);
      let l_block = [p_ty.shape[0], p_ty.shape[1]];
      let r_block = [q_ty.shape[0], q_ty.shape[1]];
      if cfg_debug() {
      println!("DEBUG: matmul_scale: {:?}{} x {:?}{}",
          &p_ty.shape, if l_t { "^T" } else { "" },
          &q_ty.shape, if r_t { "^T" } else { "" },
      );
      }
      let l_blk_inner = if l_t { l_block[0] } else { l_block[1] };
      let r_blk_inner = if r_t { r_block[1] } else { r_block[0] };
      if l_blk_inner != r_blk_inner {
        println!("ERROR: matmul_scale: incompatible shapes:");
        println!("ERROR: matmul_scale:   {:?}{} x {:?}{}",
            l_block, if l_t { "^T" } else { "" },
            r_block, if r_t { "^T" } else { "" },
        );
        panic!();
      }
      assert!((16 % p_ty.dtype.size_bytes() == 0) ||
              (p_ty.dtype.size_bytes() % 16 == 0));
      let l_dty_isz = p_ty.dtype.size_bytes() as i64;
      let l_pad = if (l_block[1] * l_dty_isz) % 16 != 0 {
        let l_pad = ((l_block[1] * l_dty_isz) + 16 - 1) / 16 * 16 / l_dty_isz;
        if cfg_debug() {
        println!("DEBUG: matmul_scale: left argument requires padding:");
        println!("DEBUG: matmul_scale:   {:?}{} -> {:?}{}",
            l_block, if l_t { "^T" } else { "" },
            [l_block[0], l_pad], if l_t { "^T" } else { "" },
        );
        }
        l_pad
      } else {
        l_block[1]
      };
      assert!((16 % q_ty.dtype.size_bytes() == 0) ||
              (q_ty.dtype.size_bytes() % 16 == 0));
      let r_dty_isz = q_ty.dtype.size_bytes() as i64;
      let r_pad = if (r_block[1] * r_dty_isz) % 16 != 0 {
        let r_pad = ((r_block[1] * r_dty_isz) + 16 - 1) / 16 * 16 / r_dty_isz;
        if cfg_debug() {
        println!("DEBUG: matmul_scale: right argument requires padding:");
        println!("DEBUG: matmul_scale:   {:?}{} -> {:?}{}",
            r_block, if r_t { "^T" } else { "" },
            [r_block[0], r_pad], if r_t { "^T" } else { "" },
        );
        }
        r_pad
      } else {
        r_block[1]
      };
      let mut l_block_pad = [l_block[0], l_pad];
      let mut r_block_pad = [r_block[0], r_pad];
      let l_blk_pad_inner = if l_t { l_block_pad[0] } else { l_block_pad[1] };
      let r_blk_pad_inner = if r_t { r_block_pad[1] } else { r_block_pad[0] };
      if l_blk_pad_inner < r_blk_pad_inner {
        if l_t {
          l_block_pad[0] = r_blk_pad_inner;
        } else {
          l_block_pad[1] = r_blk_pad_inner;
        }
      } else if l_blk_pad_inner > r_blk_pad_inner {
        if r_t {
          r_block_pad[1] = l_blk_pad_inner;
        } else {
          r_block_pad[0] = l_blk_pad_inner;
        }
      }
      let l_blk_pad_inner = if l_t { l_block_pad[0] } else { l_block_pad[1] };
      let r_blk_pad_inner = if r_t { r_block_pad[1] } else { r_block_pad[0] };
      if l_blk_pad_inner != r_blk_pad_inner {
        println!("BUG: matmul_scale: incompatible blocks after padding:");
        println!("BUG: matmul_scale:   {:?}{} x {:?}{}",
            l_block_pad, if l_t { "^T" } else { "" },
            r_block_pad, if r_t { "^T" } else { "" },
        );
        panic!();
      }
      let l_blk_outer = if l_t { l_block[1] } else { l_block[0] };
      let r_blk_outer = if r_t { r_block[0] } else { r_block[1] };
      let l_blk_pad_outer = if l_t { l_block_pad[1] } else { l_block_pad[0] };
      let r_blk_pad_outer = if r_t { r_block_pad[0] } else { r_block_pad[1] };
      let o_block = [l_blk_outer, r_blk_outer];
      let o_block_pad = [l_blk_pad_outer, r_blk_pad_outer];
      if l_block != l_block_pad ||
         r_block != r_block_pad ||
         o_block != o_block_pad
      {
        if cfg_debug() {
        println!("DEBUG: matmul_scale: after padding:");
        println!("DEBUG: matmul_scale:   {:?}{} x {:?}{} = {:?}",
            l_block_pad, if l_t { "^T" } else { "" },
            r_block_pad, if r_t { "^T" } else { "" },
            o_block_pad,
        );
        }
      }
      let o_dtype = match p_ty.dtype.max(q_ty.dtype) {
        None => {
          println!("ERROR: matmul_scale: incompatible dtypes: {:?} x {:?}", p_ty.dtype, q_ty.dtype);
          panic!();
        }
        Some(dty) => dty
      };
      let o_scale = scale.into_scalar_val_();
      let o_scale_dty = o_scale.dtype();
      match o_scale_dty {
        Dtype::F32 => {}
        _ => {
          println!("ERROR: matmul_scale: unsupported scale dtype: {:?}", o_scale_dty);
          panic!();
        }
      }
      let mut p = p;
      let mut q = q;
      if l_block != l_block_pad {
        p = p.block_pad(l_block_pad, ScalarVal_::zero(q_ty.dtype));
      }
      if r_block != r_block_pad {
        q = q.block_pad(r_block_pad, ScalarVal_::zero(q_ty.dtype));
      }
      let op = MatrixMulThunkSpec{
        //l_block: l_block_pad,
        //r_block: r_block_pad,
        l_t,
        r_t,
        //l_dtype: p_ty.dtype,
        //r_dtype: q_ty.dtype,
        o_dtype,
        o_scale,
      };
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(p);
      ctx_push_cell_arg(q);
      let mut out = ctx_pop_thunk(op);
      if o_block != o_block_pad {
        out = out.block_unpad(o_block);
      }
      out
    })
  }

  #[track_caller]
  fn block_matmul(&self, l_blk_t: bool, rhs: R, r_blk_t: bool) -> CellPtr {
    panick_wrap(|| {
      // FIXME: scalar dtype.
      self.block_matmul_scale(l_blk_t, rhs, r_blk_t, 1.0_f32)
    })
  }

  #[track_caller]
  fn block_matmul_scale<T: IntoScalarValExt>(&self, l_blk_t: bool, rhs: R, r_blk_t: bool, scale: T) -> CellPtr {
    panick_wrap(|| {
      let p = self._deref();
      let q = rhs._deref();
      let p_ty = ctx_lookup_type(p);
      let q_ty = ctx_lookup_type(q);
      assert_eq!(p_ty.ndim(), 4);
      assert_eq!(q_ty.ndim(), 4);
      let l_block = [p_ty.shape[1], p_ty.shape[3]];
      let r_block = [q_ty.shape[1], q_ty.shape[3]];
      if cfg_debug() {
      println!("DEBUG: block_matmul_scale: ({:?} / {:?}{}) x ({:?} / {:?}{})",
          &p_ty.shape, l_block, if l_blk_t { "^T" } else { "" },
          &q_ty.shape, r_block, if r_blk_t { "^T" } else { "" },
      );
      }
      let l_nrow = p_ty.shape[0];
      let l_ncol = p_ty.shape[2];
      let r_nrow = q_ty.shape[0];
      let r_ncol = q_ty.shape[2];
      if !((l_nrow == r_nrow || l_nrow == 1 || r_nrow == 1) &&
           (l_ncol == r_ncol || l_ncol == 1 || r_ncol == 1))
      {
        println!("ERROR: block_matmul_scale: incompatible shapes:");
        println!("ERROR: block_matmul_scale:   ({:?} / {:?}{}) x ({:?} / {:?}{})",
            &p_ty.shape, l_block, if l_blk_t { "^T" } else { "" },
            &q_ty.shape, r_block, if r_blk_t { "^T" } else { "" },
        );
        panic!();
      }
      let l_blk_inner = if l_blk_t { l_block[0] } else { l_block[1] };
      let r_blk_inner = if r_blk_t { r_block[1] } else { r_block[0] };
      if l_blk_inner != r_blk_inner {
        println!("ERROR: block_matmul_scale: incompatible block shapes:");
        println!("ERROR: block_matmul_scale:   {:?}{} x {:?}{}",
            l_block, if l_blk_t { "^T" } else { "" },
            r_block, if r_blk_t { "^T" } else { "" },
        );
        panic!();
      }
      assert!((16 % p_ty.dtype.size_bytes() == 0) ||
              (p_ty.dtype.size_bytes() % 16 == 0));
      let l_dty_isz = p_ty.dtype.size_bytes() as i64;
      let l_pad = if (l_block[1] * l_dty_isz) % 16 != 0 {
        let l_pad = ((l_block[1] * l_dty_isz) + 16 - 1) / 16 * 16 / l_dty_isz;
        if cfg_debug() {
        println!("DEBUG: block_matmul_scale: left argument requires padding:");
        println!("DEBUG: block_matmul_scale:   {:?}{} -> {:?}{}",
            l_block, if l_blk_t { "^T" } else { "" },
            [l_block[0], l_pad], if l_blk_t { "^T" } else { "" },
        );
        }
        l_pad
      } else {
        l_block[1]
      };
      assert!((16 % q_ty.dtype.size_bytes() == 0) ||
              (q_ty.dtype.size_bytes() % 16 == 0));
      let r_dty_isz = q_ty.dtype.size_bytes() as i64;
      let r_pad = if (r_block[1] * r_dty_isz) % 16 != 0 {
        let r_pad = ((r_block[1] * r_dty_isz) + 16 - 1) / 16 * 16 / r_dty_isz;
        if cfg_debug() {
        println!("DEBUG: block_matmul_scale: right argument requires padding:");
        println!("DEBUG: block_matmul_scale:   {:?}{} -> {:?}{}",
            r_block, if r_blk_t { "^T" } else { "" },
            [r_block[0], r_pad], if r_blk_t { "^T" } else { "" },
        );
        }
        r_pad
      } else {
        r_block[1]
      };
      let mut l_block_pad = [l_block[0], l_pad];
      let mut r_block_pad = [r_block[0], r_pad];
      let l_blk_pad_inner = if l_blk_t { l_block_pad[0] } else { l_block_pad[1] };
      let r_blk_pad_inner = if r_blk_t { r_block_pad[1] } else { r_block_pad[0] };
      if l_blk_pad_inner < r_blk_pad_inner {
        if l_blk_t {
          l_block_pad[0] = r_blk_pad_inner;
        } else {
          l_block_pad[1] = r_blk_pad_inner;
        }
      } else if l_blk_pad_inner > r_blk_pad_inner {
        if r_blk_t {
          r_block_pad[1] = l_blk_pad_inner;
        } else {
          r_block_pad[0] = l_blk_pad_inner;
        }
      }
      let l_blk_pad_inner = if l_blk_t { l_block_pad[0] } else { l_block_pad[1] };
      let r_blk_pad_inner = if r_blk_t { r_block_pad[1] } else { r_block_pad[0] };
      if l_blk_pad_inner != r_blk_pad_inner {
        println!("BUG: block_matmul_scale: incompatible blocks after padding:");
        println!("BUG: block_matmul_scale:   {:?}{} x {:?}{}",
            l_block_pad, if l_blk_t { "^T" } else { "" },
            r_block_pad, if r_blk_t { "^T" } else { "" },
        );
        panic!();
      }
      let l_blk_outer = if l_blk_t { l_block[1] } else { l_block[0] };
      let r_blk_outer = if r_blk_t { r_block[0] } else { r_block[1] };
      let l_blk_pad_outer = if l_blk_t { l_block_pad[1] } else { l_block_pad[0] };
      let r_blk_pad_outer = if r_blk_t { r_block_pad[0] } else { r_block_pad[1] };
      let o_block = [l_blk_outer, r_blk_outer];
      let o_block_pad = [l_blk_pad_outer, r_blk_pad_outer];
      if l_block != l_block_pad ||
         r_block != r_block_pad ||
         o_block != o_block_pad
      {
        if cfg_debug() {
        println!("DEBUG: block_matmul_scale: after padding:");
        println!("DEBUG: block_matmul_scale:   {:?}{} x {:?}{} = {:?}",
            l_block_pad, if l_blk_t { "^T" } else { "" },
            r_block_pad, if r_blk_t { "^T" } else { "" },
            o_block_pad,
        );
        }
      }
      let o_dtype = match p_ty.dtype.max(q_ty.dtype) {
        None => {
          println!("ERROR: block_matmul_scale: incompatible dtypes: {:?} x {:?}", p_ty.dtype, q_ty.dtype);
          panic!();
        }
        Some(dty) => dty
      };
      let o_scale = scale.into_scalar_val_();
      let o_scale_dty = o_scale.dtype();
      match o_scale_dty {
        Dtype::F32 => {}
        _ => {
          println!("ERROR: block_matmul_scale: unsupported scale dtype: {:?}", o_scale_dty);
          panic!();
        }
      }
      let mut p = p;
      let mut q = q;
      if l_block != l_block_pad {
        p = p.block_pad(l_block_pad, ScalarVal_::zero(q_ty.dtype));
      }
      if r_block != r_block_pad {
        q = q.block_pad(r_block_pad, ScalarVal_::zero(q_ty.dtype));
      }
      let op = BlockMatrixMulThunkSpec{
        l_block: l_block_pad,
        r_block: r_block_pad,
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
      let mut out = ctx_pop_thunk(op);
      if o_block != o_block_pad {
        out = out.block_unpad(o_block);
      }
      out
    })
  }

  /*#[track_caller]
  fn block_mm(&self, l_block: [i64; 2], l_blk_t: bool, rhs: R, r_block: [i64; 2], r_blk_t: bool) -> CellPtr {
    panick_wrap(|| {
      self.block_mm_scale(l_block, l_blk_t, rhs, r_block, r_blk_t, 1.0_f32)
    })
  }

  #[track_caller]
  fn block_mm_scale<T: IntoScalarValExt>(&self, l_block: [i64; 2], l_blk_t: bool, rhs: R, r_block: [i64; 2], r_blk_t: bool, scale: T) -> CellPtr {
    panick_wrap(|| {
      let p = self._deref();
      let q = rhs._deref();
      let p_ty = ctx_lookup_type(p);
      let q_ty = ctx_lookup_type(q);
      // FIXME: can relax the ndim requirement, with well documented semantics.
      assert_eq!(p_ty.ndim(), 2);
      assert_eq!(q_ty.ndim(), 2);
      //println!("DEBUG: block_mm_scale: l_ty={:?} r_ty={:?}", p_ty, q_ty);
      //println!("DEBUG: block_mm_scale: lblk={:?} rblk={:?}", l_block, r_block);
      if cfg_debug() {
      println!("DEBUG: block_mm_scale: ({:?} / {:?}{}) x ({:?} / {:?}{})",
          &p_ty.shape, l_block, if l_blk_t { "^T" } else { "" },
          &q_ty.shape, r_block, if r_blk_t { "^T" } else { "" },
      );
      }
      let l_nrow = p_ty.shape[0] / l_block[0];
      let l_ncol = p_ty.shape[1] / l_block[1];
      let r_nrow = q_ty.shape[0] / r_block[0];
      let r_ncol = q_ty.shape[1] / r_block[1];
      assert_eq!(p_ty.shape[0] % l_block[0], 0);
      assert_eq!(p_ty.shape[1] % l_block[1], 0);
      assert_eq!(q_ty.shape[0] % r_block[0], 0);
      assert_eq!(q_ty.shape[1] % r_block[1], 0);
      // FIXME FIXME: modulo/round robin blocking in the thunk impl.
      //let (l_nrow_t, l_ncol_t) = if lt { (l_ncol, l_nrow) } else { (l_nrow, l_ncol) };
      //let (r_nrow_t, r_ncol_t) = if rt { (r_ncol, r_nrow) } else { (r_nrow, r_ncol) };
      if !(l_nrow == r_nrow || l_nrow == 1 || r_nrow == 1) ||
         !(l_ncol == r_ncol || l_ncol == 1 || r_ncol == 1)
      {
        println!("ERROR: block_mm_scale: incompatible shapes:");
        println!("ERROR: block_mm_scale:   ({:?} / {:?}{}) x ({:?} / {:?}{})",
            &p_ty.shape, l_block, if l_blk_t { "^T" } else { "" },
            &q_ty.shape, r_block, if r_blk_t { "^T" } else { "" },
        );
        panic!();
      }
      let l_blk_inner = if l_blk_t { l_block[0] } else { l_block[1] };
      let r_blk_inner = if r_blk_t { r_block[1] } else { r_block[0] };
      if l_blk_inner != r_blk_inner {
        println!("ERROR: block_mm_scale: incompatible blocks:");
        println!("ERROR: block_mm_scale:   {:?}{} x {:?}{}",
            l_block, if l_blk_t { "^T" } else { "" },
            r_block, if r_blk_t { "^T" } else { "" },
        );
        panic!();
      }
      assert_eq!(16 % p_ty.dtype.size_bytes(), 0);
      let l_dty_isz = p_ty.dtype.size_bytes() as i64;
      if (l_block[1] * l_dty_isz) % 16 != 0 {
        let l_pad = ((l_block[1] * l_dty_isz) + 16 - 1) / 16 * 16 / l_dty_isz;
        println!("WARNING: block_mm_scale: left argument requires padding:");
        println!("WARNING: block_mm_scale:   {:?}{} -> {:?}{}",
            l_block, if l_blk_t { "^T" } else { "" },
            [l_block[0], l_pad], if l_blk_t { "^T" } else { "" },
        );
        // TODO
      }
      assert_eq!(16 % q_ty.dtype.size_bytes(), 0);
      let r_dty_isz = q_ty.dtype.size_bytes() as i64;
      if (r_block[1] * r_dty_isz) % 16 != 0 {
        let r_pad = ((r_block[1] * r_dty_isz) + 16 - 1) / 16 * 16 / r_dty_isz;
        println!("WARNING: block_mm_scale: right argument requires padding:");
        println!("WARNING: block_mm_scale:   {:?}{} -> {:?}{}",
            r_block, if r_blk_t { "^T" } else { "" },
            [r_block[0], r_pad], if r_blk_t { "^T" } else { "" },
        );
        // TODO
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
        Dtype::F32 => {}
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
  }*/
}

impl<L: CellDeref + ?Sized, R: CellDeref> MathBinaryOps<R> for L {}

pub trait MathUnaryOps: CellDeref {
  #[track_caller]
  fn nan_count(&self) -> CellPtr {
    panick_wrap(|| {
      let x = self._deref();
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(x);
      ctx_pop_thunk(NanCountFutThunkSpec)
    })
  }

  #[track_caller]
  fn abs_log2_hist8(&self) -> CellPtr {
    panick_wrap(|| {
      let x = self._deref();
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(x);
      ctx_pop_thunk(AbsLog2Hist8FutThunkSpec)
    })
  }

  #[track_caller]
  fn abs_log2_hist16(&self) -> CellPtr {
    panick_wrap(|| {
      let x = self._deref();
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(x);
      ctx_pop_thunk(AbsLog2Hist16FutThunkSpec)
    })
  }

  #[track_caller]
  fn square(&self) -> CellPtr {
    panick_wrap(|| {
      let op = SquareFutThunkSpec;
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(self._deref());
      ctx_pop_thunk(op)
    })
  }

  #[track_caller]
  fn recip(&self) -> CellPtr {
    panick_wrap(|| {
      let op = RecipFutThunkSpec;
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(self._deref());
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
      ctx_push_cell_arg(self._deref());
      ctx_pop_thunk(op)
    })
  }

  #[track_caller]
  fn rsqrt(&self) -> CellPtr {
    panick_wrap(|| {
      let op = RsqrtFutThunkSpec;
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(self._deref());
      ctx_pop_thunk(op)
    })
  }

  #[track_caller]
  fn cos(&self) -> CellPtr {
    panick_wrap(|| {
      let op = CosFutThunkSpec;
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(self._deref());
      ctx_pop_thunk(op)
    })
  }

  #[track_caller]
  fn sin(&self) -> CellPtr {
    panick_wrap(|| {
      let op = SinFutThunkSpec;
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(self._deref());
      ctx_pop_thunk(op)
    })
  }

  #[track_caller]
  fn exp(&self) -> CellPtr {
    panick_wrap(|| {
      let op = ExpFutThunkSpec;
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(self._deref());
      ctx_pop_thunk(op)
    })
  }

  #[track_caller]
  fn logistic(&self) -> CellPtr {
    panick_wrap(|| {
      let op = LogisticFutThunkSpec;
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(self._deref());
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
      ctx_push_cell_arg(self._deref());
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
      ctx_push_cell_arg(self._deref());
      ctx_pop_thunk(op)
    })
  }

  #[track_caller]
  fn powi(&self, exp: i64) -> CellPtr {
    panick_wrap(|| {
      let p = self._deref();
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(p);
      match ctx_lookup_dtype(p) {
        // FIXME FIXME
        Dtype::F32 => {
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
      ctx_push_cell_arg(self._deref());
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
      ctx_push_cell_arg(self._deref());
      ctx_pop_thunk(op)*/
    })
  }

  #[track_caller]
  fn inner_sum(&self) -> CellPtr {
    panick_wrap(|| {
      unimplemented!();
      /*let op = InnerSumFutThunkSpec;
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(self._deref());
      ctx_pop_thunk(op)*/
    })
  }

  #[track_caller]
  fn inner_mean(&self) -> CellPtr {
    panick_wrap(|| {
      let op = InnerMeanFutThunkSpec;
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(self._deref());
      ctx_pop_thunk(op)
    })
  }

  #[track_caller]
  fn inner_softmax(&self) -> CellPtr {
    panick_wrap(|| {
      let op = InnerSoftmaxFutThunkSpec;
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(self._deref());
      ctx_pop_thunk(op)
    })
  }

  #[track_caller]
  fn inner_arg_max(&self, /*new_dtype: Dtype*/) -> CellPtr {
    panick_wrap(|| {
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(self._deref());
      ctx_pop_thunk(InnerArgMaxFutThunkSpec)
    })
  }

  #[track_caller]
  fn inner_one_hot(&self, inner_len: i64, new_dtype: Dtype) -> CellPtr {
    panick_wrap(|| {
      if !(inner_len > 0) {
        panic!("ERROR: inner_one_hot: invalid parameter: expected inner_len > 0, actual {:?}", inner_len);
      }
      let x = self._deref();
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

  /*#[track_caller]
  fn inner_transpose(&self) -> CellPtr {
    panick_wrap(|| {
      let op = InnerTransposeFutThunkSpec;
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(self._deref());
      ctx_pop_thunk(op)
    })
  }*/

  #[track_caller]
  fn flat_sum(&self) -> CellPtr {
    panick_wrap(|| {
      let op = FlatSumFutThunkSpec;
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(self._deref());
      ctx_pop_thunk(op)
    })
  }

  #[track_caller]
  fn block_pad<T: IntoScalarValExt>(&self, new_block: [i64; 2], pad_val: T) -> CellPtr where Self: Sized {
    panick_wrap(|| {
      let x = self._deref();
      let x_ty = ctx_lookup_type(x);
      let x_nd = x_ty.ndim() as usize;
      assert!(x_nd >= 3);
      let org_block = [x_ty.shape[x_nd - 3], x_ty.shape[x_nd - 1]];
      //println!("DEBUG: block_pad: org block={:?} new_block={:?}", org_block, new_block);
      if org_block == new_block {
        //println!("DEBUG: block_unpad:   snapshot");
        //return x.snapshot();
        return x;
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
      let x = self._deref();
      let x_ty = ctx_lookup_type(x);
      let x_nd = x_ty.ndim() as usize;
      assert!(x_nd >= 3);
      let org_block = [x_ty.shape[x_nd - 3], x_ty.shape[x_nd - 1]];
      //println!("DEBUG: block_unpad: org block={:?} new_block={:?}", org_block, new_block);
      if org_block == new_block {
        //println!("DEBUG: block_unpad:   snapshot");
        //return x.snapshot();
        return x;
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
      ctx_push_cell_arg(self._deref());
      ctx_pop_thunk(op)
    })
  }
}

impl<P: CellDeref + ?Sized> MathUnaryOps for P {}

#[track_caller]
pub fn inner_softmax_post_adj<Y: CellDeref, Dy: CellDeref>(y: Y, dy: Dy) -> CellPtr {
  panick_wrap(|| {
    assert!(ctx_clean_arg());
    ctx_push_cell_arg(y._deref());
    ctx_push_cell_arg(dy._deref());
    ctx_pop_thunk(InnerSoftmaxPostAdjFutThunkSpec)
  })
}

pub fn zeros<S: Into<Vec<i64>>, D: Into<Dtype>>(shape: S, dtype: D) -> CellPtr {
  panick_wrap(|| {
    let dtype = dtype.into();
    let ty = CellType{shape: shape.into(), dtype};
    assert!(ctx_clean_arg());
    let val = ScalarVal_::zero(dtype);
    ctx_pop_thunk_(SetScalarFutThunkSpec{val}, ty)
  })
}

pub fn ones<S: Into<Vec<i64>>, D: Into<Dtype>>(shape: S, dtype: D) -> CellPtr {
  panick_wrap(|| {
    let dtype = dtype.into();
    let ty = CellType{shape: shape.into(), dtype};
    assert!(ctx_clean_arg());
    let val = ScalarVal_::one(dtype);
    ctx_pop_thunk_(SetScalarFutThunkSpec{val}, ty)
  })
}

pub fn iota(len: i64) -> CellPtr {
  panick_wrap(|| {
    assert!(ctx_clean_arg());
    ctx_pop_thunk(IotaFutThunkSpec{len})
  })
}

pub trait CastOps: CellDeref {
  #[track_caller]
  fn cast(&self, new_dtype: Dtype) -> CellPtr {
    panick_wrap(|| {
      let x = self._deref();
      let org_dtype = ctx_lookup_dtype(x);
      if org_dtype == new_dtype {
        return x;
      }
      match (org_dtype, new_dtype) {
        (Dtype::F32, Dtype::Bf16) => {
          let op = CastF32Bf16FutThunkSpec;
          assert!(ctx_clean_arg());
          ctx_push_cell_arg(x);
          ctx_pop_thunk(op)
        }
        (Dtype::Bf16, Dtype::F16) => {
          let op = CastBf16F16FutThunkSpec;
          assert!(ctx_clean_arg());
          ctx_push_cell_arg(x);
          ctx_pop_thunk(op)
        }
        (Dtype::Bf16, Dtype::F32) => {
          let op = CastBf16F32FutThunkSpec;
          assert!(ctx_clean_arg());
          ctx_push_cell_arg(x);
          ctx_pop_thunk(op)
        }
        // FIXME: other bf16 special cases.
        _ => {
          let op = CastFutThunkSpec{new_dtype};
          assert!(ctx_clean_arg());
          ctx_push_cell_arg(x);
          ctx_pop_thunk(op)
        }
        //_ => unimplemented!()
      }
    })
  }

  #[track_caller]
  fn lossy_cast(&self, new_dtype: Dtype) -> CellPtr {
    panick_wrap(|| {
      // TODO
      unimplemented!();
    })
  }
}

impl<L: CellDeref + ?Sized> CastOps for L {}

pub trait ArrayOps: CellDeref {
  #[track_caller]
  fn type_(&self) -> CellType {
    panick_wrap(|| {
      ctx_lookup_type(self._deref())
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
      ctx_lookup_dtype(self._deref())
    })
  }

  #[track_caller]
  fn bit_alias(&self, new_dtype: Dtype) -> CellPtr {
    panick_wrap(|| {
      ctx_alias_bits(self._deref(), new_dtype)
    })
  }

  #[track_caller]
  fn new_shape<S: Into<Vec<i64>>>(&self, new_shape: S) -> CellPtr {
    panick_wrap(|| {
      ctx_alias_new_shape(self._deref(), new_shape.into())
    })
  }

  #[track_caller]
  fn reshape<S: Into<Vec<i64>>>(&self, new_shape: S) -> CellPtr { self.new_shape(new_shape) }
}

impl<L: CellDeref + ?Sized> ArrayOps for L {}

pub trait Ops: CellDeref {
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

  /*#[track_caller]
  fn unset(&self) {
    unimplemented!();
  }*/

  #[track_caller]
  fn cache(&self) /*-> Self */{
    panick_wrap(|| TL_CTX.with(|ctx| {
      let mut spine = ctx.spine.borrow();
      spine.cache(self._deref());
      //self
    }))
  }

  /*#[track_caller]
  fn cache_init(&self) /*-> Self */{
    panick_wrap(|| TL_CTX.with(|ctx| {
      let mut spine = ctx.spine.borrow();
      spine.init_cache_mux(self._deref());
      //self
    }))
  }

  #[track_caller]
  fn init_cache(&self) /*-> Self */{ self.cache_init() }*/

  /*fn cache_init_futhark(self, fut_str: &str) -> Self {
    // TODO: ???
    unimplemented!();
  }*/

  #[track_caller]
  fn keep(&self) -> StableCell {
    panick_wrap(|| StableCell::from(self._deref()))
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
      let mut spine = ctx.spine.borrow();
      spine.unseal_mux(self._deref());
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
      let mut spine = ctx.spine.borrow();
      spine.yield_set(self._deref(), Locus::Mem);
    }))
  }

  #[track_caller]
  fn mem_set_yield_with(&self, _: ()) {
    panick_wrap(|| self.mem_set_yield_())
  }

  /*#[track_caller]
  fn mem_init_yield_(&self) {
    panick_wrap(|| TL_CTX.with(|ctx| {
      let mut spine = ctx.spine.borrow();
      spine.yield_init(self._deref(), Locus::Mem);
    }))
  }

  #[track_caller]
  fn mem_init_yield_with(&self, _: ()) {
    panick_wrap(|| self.mem_init_yield_())
  }*/

  /*#[track_caller]
  fn eval(self) -> Self {
    panick_wrap(|| {
      let ret = eval(self._deref());
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
  fn _memcpy(&self) -> CellPtr {
    panick_wrap(|| {
      assert!(ctx_clean_arg());
      ctx_push_cell_arg(self._deref());
      ctx_pop_thunk(MemcpyThunkSpec)
    })
  }

  #[track_caller]
  fn const_(&self) -> CellPtr {
    panick_wrap(|| TL_CTX.with(|ctx| {
      ctx.const_(self._deref())
    }))
  }

  /*#[track_caller]
  fn snapshot(&self) -> CellPtr {
    panick_wrap(|| ctx_snapshot(self._deref()))
  }*/

  /*#[track_caller]
  fn checkpoint(self) -> CellPtr {
    unimplemented!();
  }*/
}

impl<L: CellDeref + ?Sized> Ops for L {}

pub trait CtlOps: CellDeref {
  /*#[track_caller]
  fn resident(&self, loc: Locus) -> bool {
    panick_wrap(|| TL_CTX.with(|ctx| {
      let this = self._deref();
      let env = ctx.env.borrow();
      match env.plookup_mut_view(this) {
        Err(_) => {
          println!("ERROR: CtlOps::resident: failed to dereference {:?} to a physical cell", this);
          panic!("");
        }
        Ok(e) => {
          match e.cel_ {
            &mut Cell_::Phy(.., ref mut pcel) => {
              pcel.lookup_loc(loc).is_some()
            }
            _ => panic!("bug")
          }
        }
      }
    }))
  }*/

  #[track_caller]
  fn spine_version(&self) -> Clock {
    panick_wrap(|| TL_CTX.with(|ctx| {
      let spine = ctx.spine.borrow();
      spine._version(self._deref()).unwrap_or_else(|| Clock::default())
    }))
  }

  #[track_caller]
  fn version(&self) -> Clock {
    panick_wrap(|| ctx_lookup_clk(self._deref()))
  }

  #[track_caller]
  fn _get_mem(&self) -> MemReg {
    panick_wrap(|| TL_CTX.with(|ctx| {
      let x = self._deref();
      let xclk = ctx_lookup_clk(x);
      match ctx.env.borrow_mut().pread_ref_(x, xclk, Locus::Mem) {
        Err(CellDerefErr::View) => panic!("bug"),
        Err(_) => panic!("bug"),
        Ok(e) => {
          let mut cel_ = e.cel_.borrow_mut();
          match &mut *cel_ {
            &mut Cell_::Phy(.., ref mut pcel) => {
              let (pm, addr) = match pcel.lookup_loc(Locus::Mem) {
                None => panic!("bug"),
                Some((_, pm, addr)) => (pm, addr)
              };
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

impl<L: CellDeref + ?Sized> CtlOps for L {}

/*#[track_caller]
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

#[track_caller]
pub fn apply_futhark_unverified(lam_src: &str, arg: &[&CellPtr]) -> CellPtr {
  // FIXME FIXME
  unimplemented!();
}*/

pub trait TypeOps: Borrow<CellType> {
  #[track_caller]
  fn zeros(&self) -> CellPtr {
    let ty = self.borrow();
    zeros(&ty.shape as &[_], ty.dtype)
  }

  #[track_caller]
  fn ones(&self) -> CellPtr {
    let ty = self.borrow();
    ones(&ty.shape as &[_], ty.dtype)
  }
}

impl TypeOps for CellType {}

pub trait MMapOps: Borrow<MCellPtr> {
  #[track_caller]
  fn vjp(&self) -> CellMap {
    panick_wrap(|| TL_CTX.with(|ctx| {
      let allsrc = CellMap::new();
      let sink = *self.borrow();
      let spine = ctx.spine.borrow();
      spine.adj_map(allsrc.as_ptr(), sink, &ctx.ctr, &ctx.thunkenv);
      allsrc
    }))
  }

  #[track_caller]
  fn jvp(&self) -> CellMap {
    panick_wrap(|| TL_CTX.with(|ctx| {
      let allsink = CellMap::new();
      let src = *self.borrow();
      let spine = ctx.spine.borrow();
      spine.dual_map(allsink.as_ptr(), src, &ctx.ctr, &ctx.thunkenv);
      allsink
    }))
  }
}

impl MMapOps for CellMap {}

pub fn vjp(allsrc: &CellMap, sink: &CellMap) {
  panick_wrap(|| TL_CTX.with(|ctx| {
    let spine = ctx.spine.borrow();
    spine.adj_map(allsrc.as_ptr(), sink.as_ptr(), &ctx.ctr, &ctx.thunkenv);
  }))
}

pub fn jvp(allsink: &CellMap, src: &CellMap) {
  panick_wrap(|| TL_CTX.with(|ctx| {
    let spine = ctx.spine.borrow();
    spine.dual_map(allsink.as_ptr(), src.as_ptr(), &ctx.ctr, &ctx.thunkenv);
  }))
}
