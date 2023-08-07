use crate::algo::{HashMap, HashSet};
use crate::cell::*;
use crate::clock::*;
use crate::nd::{IRange};
use crate::panick::{panick_wrap};
use crate::pctx::{TL_PCTX, Locus, PMach, MemReg};
use crate::spine::*;
use crate::thunk::*;
use crate::thunk::op::{SetScalarFutThunkSpec};
use crate::util::stat::*;
use cacti_cfg_env::*;

use futhark_syntax::re::{ReTrie};
use futhark_syntax::tokenizing::{Token as FutToken};

use std::any::{Any};
use std::borrow::{Borrow};
use std::cell::{Cell, RefCell, Ref, RefMut};
use std::mem::{swap};
use std::rc::{Rc};

thread_local! {
  pub static TL_CTX_CFG: CtxCfg = CtxCfg::default();
  pub static TL_CTX: Ctx = {
    let mut ctx = Ctx::new();
    TL_CTX_CFG.with(|cfg| cfg._seal.set(true));
    /*ctx.thunkenv.borrow_mut()._set_accumulate_in_place(false);*/
    ctx
  };
}

pub struct CtxCfg {
  pub swapfile_cap:     Cell<usize>,
  pub gpu_reserve:      Cell<u16>,
  pub gpu_workspace:    Cell<u16>,
  pub _seal:            Cell<bool>,
}

impl Default for CtxCfg {
  fn default() -> CtxCfg {
    if cfg_debug() { println!("DEBUG: CtxCfg::default"); }
    CtxCfg{
      swapfile_cap:     Cell::new(0),
      gpu_reserve:      Cell::new(9001),
      gpu_workspace:    Cell::new(111),
      _seal:            Cell::new(false),
    }
  }
}

pub fn ctx_cfg_get_swapfile_max_bytes() -> usize {
  TL_CTX_CFG.with(|ctx_cfg| ctx_cfg.swapfile_cap.get())
}

pub fn ctx_cfg_set_swapfile_max_bytes(sz: usize) {
  TL_CTX_CFG.with(|ctx_cfg| {
    if ctx_cfg._seal.get() {
      panic!("bug: cannot set context configuration after context initialization");
    }
    ctx_cfg.swapfile_cap.set(sz)
  })
}

pub fn ctx_cfg_get_gpu_reserve_mem_per_10k() -> u16 {
  TL_CTX_CFG.with(|ctx_cfg| ctx_cfg.gpu_reserve.get())
}

pub fn ctx_cfg_set_gpu_reserve_mem_per_10k(m: u16) {
  if m <= 0 {
    panic!("bug: gpu reserve too small: {}/10000", m);
  }
  if m >= 10000 {
    panic!("bug: gpu reserve too big: {}/10000", m);
  }
  TL_CTX_CFG.with(|ctx_cfg| {
    if ctx_cfg._seal.get() {
      panic!("bug: cannot set context configuration after context initialization");
    }
    ctx_cfg.gpu_reserve.set(m)
  })
}

pub fn ctx_cfg_get_gpu_workspace_mem_per_10k() -> u16 {
  TL_CTX_CFG.with(|ctx_cfg| ctx_cfg.gpu_workspace.get())
}

pub fn ctx_cfg_set_gpu_workspace_mem_per_10k(m: u16) {
  if m <= 0 {
    panic!("bug: gpu workspace too small: {}/10000", m);
  }
  if m >= 10000 {
    panic!("bug: gpu workspace too big: {}/10000", m);
  }
  TL_CTX_CFG.with(|ctx_cfg| {
    if ctx_cfg._seal.get() {
      panic!("bug: cannot set context configuration after context initialization");
    }
    ctx_cfg.gpu_workspace.set(m)
  })
}

pub struct Ctx {
  pub ctr:      CtxCtr,
  pub ctlstate: CtxCtlState,
  pub env:      RefCell<CtxEnv>,
  pub thunkenv: RefCell<CtxThunkEnv>,
  pub spine:    RefCell<Spine>,
  pub futhark:  RefCell<FutharkCtx>,
  pub timing:   TimingCtx,
  pub debugctr: DebugCtrs,
}

impl Drop for Ctx {
  fn drop(&mut self) {
    if cfg_info() {
    let digest = self.timing.digest();
    println!("INFO:   Ctx::drop: timing digest: pregemm1: {:?}", digest.pregemm1);
    println!("INFO:   Ctx::drop: timing digest: gemm1:    {:?}", digest.gemm1);
    println!("INFO:   Ctx::drop: timing digest: pregemm:  {:?}", digest.pregemm);
    println!("INFO:   Ctx::drop: timing digest: gemm:     {:?}", digest.gemm);
    println!("INFO:   Ctx::drop: timing digest: f_build1: {:?}", digest.f_build1);
    println!("INFO:   Ctx::drop: timing digest: f_setup1: {:?}", digest.f_setup1);
    println!("INFO:   Ctx::drop: timing digest: futhark1: {:?}", digest.futhark1);
    println!("INFO:   Ctx::drop: timing digest: f_build:  {:?}", digest.f_build);
    println!("INFO:   Ctx::drop: timing digest: f_setup:  {:?}", digest.f_setup);
    println!("INFO:   Ctx::drop: timing digest: futhark:  {:?}", digest.futhark);
    println!("INFO:   Ctx::drop: debug counter: accumulate hashes:       {:?}", self.debugctr.accumulate_hashes.borrow());
    println!("INFO:   Ctx::drop: debug counter: accumulate in place:     {:?}", self.debugctr.accumulate_in_place.get());
    println!("INFO:   Ctx::drop: debug counter: accumulate not in place: {:?}", self.debugctr.accumulate_not_in_place.get());
    }
  }
}

impl Ctx {
  pub fn new() -> Ctx {
    if cfg_debug() { println!("DEBUG: Ctx::new"); }
    Ctx{
      ctr:      CtxCtr::new(),
      ctlstate: CtxCtlState::default(),
      env:      RefCell::new(CtxEnv::default()),
      thunkenv: RefCell::new(CtxThunkEnv::default()),
      spine:    RefCell::new(Spine::default()),
      futhark:  RefCell::new(FutharkCtx::default()),
      timing:   TimingCtx::default(),
      debugctr: DebugCtrs::default(),
    }
  }

  pub fn reset(&self) {
    if cfg_info() { println!("INFO:   Ctx::reset: gc: scan {} cells", self.ctr.celfront.borrow().len()); }
    let mut env = self.env.borrow_mut();
    let mut next_celfront = Vec::new();
    let mut free_ct = 0;
    TL_PCTX.with(|pctx| {
      if let Some(gpu) = pctx.nvgpu.as_ref() {
        gpu._dump_info();
      }
      for &x in self.ctr.celfront.borrow().iter() {
        let mut f = false;
        match env._lookup_ref_(x) {
          Err(_) => {
            let _ = env.celtab.remove(&x);
          }
          Ok(e) => {
            let xroot = e.root();
            if e.stablect.get() > 0 {
              next_celfront.push(x);
            } else {
              match e.cel_ {
                &Cell_::Phy(.., ref pcel) => {
                  for (_, rep) in pcel.replicas.iter() {
                    match pctx.release(rep.addr.get()) {
                      None => {}
                      Some(icel) => {
                        //assert_eq!(icel.root(), Some(xroot));
                        f = true;
                      }
                    }
                  }
                }
                _ => {}
              }
              drop(e);
              assert!(env.celtab.remove(&x).is_some());
            }
          }
        }
        if f {
          free_ct += 1;
        }
      }
      if let Some(gpu) = pctx.nvgpu.as_ref() {
        gpu._dump_info();
      }
    });
    if cfg_info() { println!("INFO:   Ctx::reset: gc:   free {} cells", free_ct); }
    if cfg_info() { println!("INFO:   Ctx::reset: gc:   next {} cells", next_celfront.len()); }
    // TODO
    // FIXME
    //unimplemented!();
    swap(&mut *self.ctr.celfront.borrow_mut(), &mut next_celfront);
    // FIXME FIXME: reset all.
    //ctx.env.borrow_mut().reset();
    //ctx.thunkenv.borrow_mut().reset();
    self.spine.borrow_mut()._reset();
  }
}

#[derive(Default)]
pub struct CtxCtlState {
  // TODO
  pub primary:  Cell<Option<PMach>>,
  pub accumulate_in_place: Cell<bool>,
  pub assume_uninit_zero: Cell<bool>,
}

impl CtxCtlState {
  pub fn _set_primary(&self, pmach: PMach) -> Option<PMach> {
    let prev = self.primary.get();
    self.primary.set(Some(pmach));
    prev
  }

  pub fn _unset_primary(&self) -> Option<PMach> {
    let prev = self.primary.get();
    self.primary.set(None);
    prev
  }

  pub fn _set_accumulate_in_place(&self, flag: bool) {
    self.accumulate_in_place.set(flag);
  }

  pub fn _set_assume_uninit_zero(&self, flag: bool) {
    self.assume_uninit_zero.set(flag);
  }
}

#[derive(Default)]
pub struct FutharkCtx {
  pub trie: Option<Rc<ReTrie<FutToken>>>,
}

#[derive(Default)]
pub struct TimingCtx {
  pub pregemm1: RefCell<Vec<f64>>,
  pub gemm1:    RefCell<Vec<f64>>,
  pub pregemm:  RefCell<Vec<f64>>,
  pub gemm:     RefCell<Vec<f64>>,
  pub f_build1: RefCell<Vec<f64>>,
  pub f_setup1: RefCell<Vec<f64>>,
  pub futhark1: RefCell<Vec<f64>>,
  pub f_build:  RefCell<Vec<f64>>,
  pub f_setup:  RefCell<Vec<f64>>,
  pub futhark:  RefCell<Vec<f64>>,
}

#[derive(Debug)]
pub struct TimingDigest {
  pub pregemm1: StatDigest,
  pub gemm1:    StatDigest,
  pub pregemm:  StatDigest,
  pub gemm:     StatDigest,
  pub f_build1: StatDigest,
  pub f_setup1: StatDigest,
  pub futhark1: StatDigest,
  pub f_build:  StatDigest,
  pub f_setup:  StatDigest,
  pub futhark:  StatDigest,
}

impl TimingCtx {
  pub fn digest(&self) -> TimingDigest {
    TimingDigest{
      pregemm1: StatDigest::from(&*self.pregemm1.borrow()),
      gemm1:    StatDigest::from(&*self.gemm1.borrow()),
      pregemm:  StatDigest::from(&*self.pregemm.borrow()),
      gemm:     StatDigest::from(&*self.gemm.borrow()),
      f_build1: StatDigest::from(&*self.f_build1.borrow()),
      f_setup1: StatDigest::from(&*self.f_setup1.borrow()),
      futhark1: StatDigest::from(&*self.futhark1.borrow()),
      f_build:  StatDigest::from(&*self.f_build.borrow()),
      f_setup:  StatDigest::from(&*self.f_setup.borrow()),
      futhark:  StatDigest::from(&*self.futhark.borrow()),
    }
  }
}

#[derive(Default)]
pub struct DebugCtrs {
  pub accumulate_hashes: RefCell<HashMap<String, i32>>,
  pub accumulate_in_place: Cell<i64>,
  pub accumulate_not_in_place: Cell<i64>,
}

#[track_caller]
pub fn reset() {
  panick_wrap(|| TL_CTX.with(|ctx| ctx.reset()))
}

#[track_caller]
pub fn compile() {
  panick_wrap(|| TL_CTX.with(|ctx| {
    ctx.spine.borrow_mut()._compile();
  }))
}

#[track_caller]
pub fn resume() -> SpineRet {
  panick_wrap(|| TL_CTX.with(|ctx| {
    let mut env = ctx.env.borrow_mut();
    let mut thunkenv = ctx.thunkenv.borrow_mut();
    let mut spine = ctx.spine.borrow_mut();
    spine._resume(&ctx.ctr, &mut *env, &mut *thunkenv, /*CellPtr::nil(), Clock::default(),*/ SpineResume::_Top)
  }))
}

/*#[track_caller]
pub fn resume_put_mem_val<K: Borrow<CellPtr>>(key: K, val: &dyn Any) -> SpineRet {
  panick_wrap(|| TL_CTX.with(|ctx| {
    let mut env = ctx.env.borrow_mut();
    let mut thunkenv = ctx.thunkenv.borrow_mut();
    let mut spine = ctx.spine.borrow_mut();
    spine._resume(&ctx.ctr, &mut *env, &mut *thunkenv, /*CellPtr::nil(), Clock::default(),*/ SpineResume::PutMemV(*key.borrow(), val))
  }))
}*/

#[track_caller]
pub fn resume_put_mem_with<K: Borrow<CellPtr>, F: Fn(CellType, MemReg)>(key: K, fun: F) -> SpineRet {
  panick_wrap(|| TL_CTX.with(|ctx| {
    let mut env = ctx.env.borrow_mut();
    let mut thunkenv = ctx.thunkenv.borrow_mut();
    let mut spine = ctx.spine.borrow_mut();
    // FIXME FIXME
    spine._resume(&ctx.ctr, &mut *env, &mut *thunkenv, /*CellPtr::nil(), Clock::default(),*/ SpineResume::PutMemF(*key.borrow(), &fun as _))
  }))
}

/*#[track_caller]
pub fn eval(x: CellPtr) -> SpineRet {
  panick_wrap(|| TL_CTX.with(|ctx| {
    let mut env = ctx.env.borrow_mut();
    let mut thunkenv = ctx.thunkenv.borrow_mut();
    let mut spine = ctx.spine.borrow_mut();
    // FIXME FIXME: observe the arg clock.
    spine._resume(&ctx.ctr, &mut *env, &mut *thunkenv, /*x, _,*/ SpineResume::_Top)
  }))
}*/

/*#[track_caller]
pub fn yield_() {
  panick_wrap(|| TL_CTX.with(|ctx| {
    let mut spine = ctx.spine.borrow_mut();
    // FIXME FIXME
    unimplemented!();
    /*spine.curp += 1;
    //spine.env._;
    spine.log.push(SpineEntry::Yield_);*/
  }))
}

#[track_caller]
pub fn break_() {
  panick_wrap(|| TL_CTX.with(|ctx| {
    let mut spine = ctx.spine.borrow_mut();
    // FIXME FIXME
    unimplemented!();
    /*spine.curp += 1;
    //spine.env._;
    spine.log.push(SpineEntry::Break_);*/
  }))
}*/

pub struct PMachScope {
  pub prev: Option<PMach>,
}

impl Drop for PMachScope {
  fn drop(&mut self) {
    TL_CTX.with(|ctx| {
      match self.prev {
        None => {
          ctx.ctlstate._unset_primary();
        }
        Some(pm) => {
          ctx.ctlstate._set_primary(pm);
        }
      }
    })
  }
}

impl PMachScope {
  #[track_caller]
  pub fn new(pmach: PMach) -> PMachScope {
    TL_CTX.with(|ctx| {
      let prev = ctx.ctlstate._set_primary(pmach);
      PMachScope{prev}
    })
  }

  pub fn with<F: FnMut(&PMachScope)>(&self, mut f: F) {
    (f)(self);
  }
}

#[track_caller]
pub fn smp_scope() -> PMachScope {
  panick_wrap(|| PMachScope::new(PMach::Smp))
}

#[cfg(feature = "nvgpu")]
#[track_caller]
pub fn gpu_scope() -> PMachScope {
  nvgpu_scope()
}

#[cfg(feature = "nvgpu")]
#[track_caller]
pub fn nvgpu_scope() -> PMachScope {
  panick_wrap(|| PMachScope::new(PMach::NvGpu))
}

pub fn ctx_unwrap<F: FnMut(&Ctx) -> X, X>(f: &mut F) -> X {
  TL_CTX.with(f)
}

pub fn ctx_release(x: CellPtr) {
  TL_CTX.try_with(|ctx| {
    ctx.env.borrow().release(x);
  }).unwrap_or(())
}

pub fn ctx_retain(x: CellPtr) {
  TL_CTX.with(|ctx| {
    ctx.env.borrow().retain(x);
  })
}

pub fn ctx_lookup_type(x: CellPtr) -> CellType {
  TL_CTX.with(|ctx| {
    /*match ctx.env.borrow().lookup_ref(x) {
      None => panic!("bug"),
      Some(e) => e.ty.clone()
    }*/
    match ctx.env.borrow()._lookup_ref_(x) {
      Err(_) => panic!("bug"),
      Ok(e) => e.ty.clone()
    }
  })
}

pub fn ctx_lookup_dtype(x: CellPtr) -> Dtype {
  TL_CTX.with(|ctx| {
    /*match ctx.env.borrow().lookup_ref(x) {
      None => panic!("bug"),
      Some(e) => e.ty.dtype
    }*/
    match ctx.env.borrow()._lookup_ref_(x) {
      Err(_) => panic!("bug"),
      Ok(e) => e.ty.dtype
    }
  })
}

pub fn ctx_lookup_clk(x: CellPtr) -> Clock {
  TL_CTX.with(|ctx| {
    /*match ctx.env.borrow().lookup_ref(x) {
      None => panic!("bug"),
      Some(e) => {
        match e.cel_ {
          &Cell_::Top(ref state, ..) |
          &Cell_::Phy(ref state, ..) |
          &Cell_::Cow(ref state, ..) => {
            state.borrow().clk
          }
          &Cell_::Bot => panic!("bug"),
          _ => panic!("bug")
        }
      }
    }*/
    match ctx.env.borrow()._lookup_ref_(x) {
      Err(_) => panic!("bug"),
      Ok(e) => {
        match e.cel_ {
          &Cell_::Top(ref state, ..) |
          &Cell_::Phy(ref state, ..) |
          &Cell_::Cow(ref state, ..) => {
            state.borrow().clk
          }
          &Cell_::Bot => panic!("bug"),
          _ => panic!("bug")
        }
      }
    }
  })
}

pub fn ctx_fresh_mset() -> MCellPtr {
  TL_CTX.with(|ctx| {
    let x = ctx.ctr.fresh_cel()._into_mcel_ptr();
    ctx.env.borrow_mut().mceltab.insert(x, MCellEnvEntry{mcel_: MCell_::Set(MCellSet::default())});
    x
  })
}

pub fn ctx_fresh_mmap() -> MCellPtr {
  TL_CTX.with(|ctx| {
    let x = ctx.ctr.fresh_cel()._into_mcel_ptr();
    ctx.env.borrow_mut().mceltab.insert(x, MCellEnvEntry{mcel_: MCell_::Map(MCellMap::default())});
    x
  })
}

pub fn ctx_insert(ty: CellType) -> CellPtr {
  TL_CTX.with(|ctx| {
    let x = ctx.ctr.fresh_cel();
    ctx.env.borrow_mut().insert_top(x, ty);
    x
  })
}

/*pub fn ctx_alias_new_type(og: CellPtr, new_ty: CellType) -> CellPtr {
  TL_CTX.with(|ctx| {
    let mut env = ctx.env.borrow_mut();
    match env.lookup_ref(og) {
      None => panic!("bug"),
      Some(_e) => {
        let x = ctx.ctr.fresh_cel();
        env.insert_alias(x, new_ty, og);
        let spine = ctx.spine.borrow();
        spine.alias(x, og);
        x
      }
    }
  })
}*/

pub fn ctx_alias_new_shape(og: CellPtr, new_shape: Vec<i64>) -> CellPtr {
  TL_CTX.with(|ctx| {
    let mut env = ctx.env.borrow_mut();
    /*match env.lookup_ref(og) {
      None => panic!("bug"),
      Some(e) => {
        let new_ty = CellType{shape: new_shape, dtype: e.ty.dtype};
        let cmp = new_ty.shape_compat(&e.ty);
        if !(cmp == ShapeCompat::Equal || cmp == ShapeCompat::NewShape) {
          println!("ERROR: ctx_alias_new_shape: shape mismatch: og={:?} old shape={:?} new shape={:?} compat={:?}", og, &e.ty.shape, &new_ty.shape, cmp);
          panic!();
        }
        if cmp == ShapeCompat::Equal {
          return og;
        }
        let x = ctx.ctr.fresh_cel();
        //println!("DEBUG: ctx_alias_new_shape: og={:?} old shape={:?} x={:?} new shape={:?} compat={:?}", og, &e.ty.shape, x, &new_ty.shape, cmp);
        env.insert_alias(x, CellAlias::NewShape, new_ty, og);
        let spine = ctx.spine.borrow();
        spine.alias(x, og);
        x
      }
    }*/
    match env._lookup_ref_(og) {
      Err(_) => panic!("bug"),
      Ok(e) => {
        let new_ty = CellType{shape: new_shape, dtype: e.ty.dtype};
        let cmp = new_ty.shape_compat(&e.ty);
        if !(cmp == ShapeCompat::Equal || cmp == ShapeCompat::NewShape) {
          println!("ERROR: ctx_alias_new_shape: shape mismatch: og={:?} old shape={:?} new shape={:?} compat={:?}", og, &e.ty.shape, &new_ty.shape, cmp);
          panic!();
        }
        if cmp == ShapeCompat::Equal {
          return og;
        }
        let x = ctx.ctr.fresh_cel();
        env.insert_alias(x, CellAlias::NewShape, new_ty, og);
        let spine = ctx.spine.borrow();
        spine.alias(x, og);
        x
      }
    }
  })
}

pub fn ctx_alias_bits(og: CellPtr, new_dtype: Dtype) -> CellPtr {
  TL_CTX.with(|ctx| {
    let mut env = ctx.env.borrow_mut();
    /*match env.lookup_ref(og) {
      None => panic!("bug"),
      Some(e) => {
        if new_dtype.size_bytes() != e.ty.dtype.size_bytes() {
          println!("ERROR: ctx_alias_bits: og={:?} old dtype={:?} new dtype={:?}", og, e.ty.dtype, new_dtype);
          panic!();
        }
        if new_dtype == e.ty.dtype {
          return og;
        }
        let new_ty = CellType{dtype: new_dtype, shape: e.ty.shape.clone()};
        let x = ctx.ctr.fresh_cel();
        //println!("DEBUG: ctx_alias_bits: og={:?} old dtype={:?} x={:?} new dtype={:?}", og, e.ty.dtype, x, new_dtype);
        env.insert_alias(x, CellAlias::BitAlias, new_ty, og);
        let spine = ctx.spine.borrow();
        spine.alias(x, og);
        x
      }
    }*/
    match env._lookup_ref_(og) {
      Err(_) => panic!("bug"),
      Ok(e) => {
        if new_dtype.size_bytes() != e.ty.dtype.size_bytes() {
          println!("ERROR: ctx_alias_bits: og={:?} old dtype={:?} new dtype={:?}", og, e.ty.dtype, new_dtype);
          panic!();
        }
        if new_dtype == e.ty.dtype {
          return og;
        }
        let new_ty = CellType{dtype: new_dtype, shape: e.ty.shape.clone()};
        let x = ctx.ctr.fresh_cel();
        env.insert_alias(x, CellAlias::BitAlias, new_ty, og);
        let spine = ctx.spine.borrow();
        spine.alias(x, og);
        x
      }
    }
  })
}

impl Ctx {
  pub fn alias_view_slice(&self, og: CellPtr, idx: &[IRange]) -> CellPtr {
    let mut env = self.env.borrow_mut();
    match env._lookup_view(og) {
      Err(_) => panic!("bug"),
      Ok(mut e) => {
        let vop = CellViewOp::slice(idx);
        e.view_mut().vlog.push(vop.clone());
        let new_ty = match e.view().type_eval(&e.root_ty) {
          Err(_) => unimplemented!(),
          Ok(ty) => ty
        };
        let x = self.ctr.fresh_cel();
        env.insert_alias(x, CellAlias::View(vop), new_ty, og);
        let spine = self.spine.borrow();
        spine.alias(x, og);
        x
      }
    }
  }

  pub fn alias_view_proj(&self, og: CellPtr, mask: &[bool]) -> CellPtr {
    let mut env = self.env.borrow_mut();
    match env._lookup_view(og) {
      Err(_) => panic!("bug"),
      Ok(mut e) => {
        let vop = CellViewOp::proj(mask);
        e.view_mut().vlog.push(vop.clone());
        let new_ty = match e.view().type_eval(&e.root_ty) {
          Err(_) => unimplemented!(),
          Ok(ty) => ty
        };
        let x = self.ctr.fresh_cel();
        env.insert_alias(x, CellAlias::View(vop), new_ty, og);
        let spine = self.spine.borrow();
        spine.alias(x, og);
        x
      }
    }
  }

  pub fn alias_view_swap(&self, og: CellPtr, ld: i8, rd: i8) -> CellPtr {
    let mut env = self.env.borrow_mut();
    match env._lookup_view(og) {
      Err(_) => panic!("bug"),
      Ok(mut e) => {
        let vop = CellViewOp::swap(ld, rd);
        e.view_mut().vlog.push(vop.clone());
        let new_ty = match e.view().type_eval(&e.root_ty) {
          Err(_) => unimplemented!(),
          Ok(ty) => ty
        };
        let x = self.ctr.fresh_cel();
        env.insert_alias(x, CellAlias::View(vop), new_ty, og);
        let spine = self.spine.borrow();
        spine.alias(x, og);
        x
      }
    }
  }

  pub fn const_(&self, og: CellPtr) -> CellPtr {
    let x = self.ctr.fresh_cel();
    let mut env = self.env.borrow_mut();
    env.insert_const_(x, og);
    let spine = self.spine.borrow();
    spine.const_(x, og);
    x
  }
}

/*pub fn ctx_snapshot(og: CellPtr) -> CellPtr {
  TL_CTX.with(|ctx| {
    let mut env = ctx.env.borrow_mut();
    let x = env.snapshot(&ctx.ctr, og);
    let spine = ctx.spine.borrow();
    spine.snapshot(x, og);
    x
  })
}*/

pub fn ctx_clean_arg() -> bool {
  TL_CTX.with(|ctx| {
    let thunkenv = ctx.thunkenv.borrow();
    thunkenv.arg.is_empty()
    && thunkenv.param.is_empty()
    /*&& thunkenv.out.is_empty()*/
  })
}

pub fn ctx_push_cell_arg(x: CellPtr) {
  TL_CTX.with(|ctx| {
    match ctx.env.borrow()._lookup_ref_(x) {
      Err(_) => panic!("bug"),
      Ok(e) => {
        let spine = ctx.spine.borrow();
        let xclk = match spine._version(x) {
          None => {
            println!("DEBUG: ctx_push_cell_arg: no spine version: x={:?}", x);
            let cur_env = spine.cur_env.borrow();
            let xroot = cur_env._deref(x);
            println!("DEBUG: ctx_push_cell_arg:   xroot={:?} state={:?}",
                xroot, cur_env.state.get(&xroot));
            /*let query = CellPtr::from_unchecked(2401);
            /*println!("DEBUG: ctx_push_cell_arg:   query={:?} state={:?} alias={:?}",
                query, cur_env.state.get(&query), cur_env.alias.get(&query));*/
            println!("DEBUG: ctx_push_cell_arg:   query={:?} state={:?}",
                query, cur_env.state.get(&query));*/
            panic!("bug");
          }
          Some(xclk) => xclk
        };
        drop(spine);
        if xclk.is_nil() {
          println!("ERROR: ctx_push_cell_arg: tried to push an uninitialized thunk argument: {:?}", x);
          panic!();
        }
        //println!("DEBUG: ctx_push_cell_arg: x={:?} xclk={:?}", x, xclk);
        ctx.thunkenv.borrow_mut().arg.push((x, xclk))
      }
    }
  })
}

pub fn ctx_push_scalar_param<T: IntoScalarValExt>(x: T) -> ScalarVal_ {
  TL_CTX.with(|ctx| {
    let val = x.into_scalar_val_();
    ctx.thunkenv.borrow_mut().param.push(val);
    val
  })
}

pub fn ctx_pop_thunk<Th: ThunkSpec_ + 'static>(th: Th) -> CellPtr {
  TL_CTX.with(|ctx| {
    let mut dims = Vec::with_capacity(ctx.thunkenv.borrow().arg.len());
    let mut tys_ = Vec::with_capacity(ctx.thunkenv.borrow().arg.len());
    for &(arg, _) in ctx.thunkenv.borrow().arg.iter() {
      let ty_ = match ctx.env.borrow()._lookup_ref_(arg) {
        Err(_) => panic!("bug"),
        Ok(e) => e.ty.clone()
      };
      dims.push(ty_.to_dim());
      tys_.push(ty_);
    }
    // FIXME FIXME: multiple arity out.
    let odim = match th.out_dim(&dims) {
      Err(_) => {
        println!("ERROR: thunk apply dim error: name={:?}", th.debug_name());
        println!("ERROR: thunk apply dim error: dims={:?}", &dims);
        println!("ERROR: thunk apply dim error: tys_={:?}", &tys_);
        panic!();
      }
      Ok(dim) => dim
    };
    let oty_ = match th.out_ty_(&tys_) {
      Err(_) => {
        println!("ERROR: thunk apply type error: name={:?}", th.debug_name());
        println!("ERROR: thunk apply type error: dims={:?}", &dims);
        println!("ERROR: thunk apply type error: tys_={:?}", &tys_);
        panic!();
      }
      Ok(ty_) => ty_
    };
    assert_eq!(odim, oty_.to_dim());
    ctx_pop_thunk_(th, oty_)
  })
}

pub fn ctx_pop_thunk_<Th: ThunkSpec_ + 'static>(th: Th, out_ty: CellType) -> CellPtr {
  TL_CTX.with(|ctx| {
    let mut dims = Vec::with_capacity(ctx.thunkenv.borrow().arg.len());
    for &(arg, _) in ctx.thunkenv.borrow().arg.iter() {
      let ty_ = match ctx.env.borrow()._lookup_ref_(arg) {
        Err(_) => panic!("bug"),
        Ok(e) => e.ty.clone()
      };
      dims.push(ty_.to_dim());
      let spine = ctx.spine.borrow();
      spine.push_seal(arg);
    }
    // FIXME FIXME: multiple arity out.
    let odim = out_ty.to_dim();
    let oty_ = out_ty;
    dims.push(odim);
    let (lar, rar) = match th.arity() {
      None => unimplemented!(),
      Some(ar) => ar
    };
    // FIXME FIXME
    //let tp = ctx.thunkenv.borrow_mut().insert_(ctx, ar_in, ar_out, dims, th);
    let tp = ctx.thunkenv.borrow_mut().lookup_or_insert(&ctx.ctr, lar, rar, dims, th);
    let y = ctx.ctr.fresh_cel();
    ctx.env.borrow_mut().insert_top(y, oty_);
    let spine = ctx.spine.borrow();
    //spine.intro_aff(y);
    spine.apply(y, tp);
    let yclk = spine._version(y).unwrap();
    let mut arg = Vec::new();
    swap(&mut arg, &mut ctx.thunkenv.borrow_mut().arg);
    let mut param = Vec::new();
    swap(&mut param, &mut ctx.thunkenv.borrow_mut().param);
    ctx.thunkenv.borrow_mut().update(y, y, yclk, tp, arg, param);
    y
  })
}

pub fn ctx_pop_apply_thunk<Th: ThunkSpec_ + 'static>(th: Th, out: CellPtr) {
  TL_CTX.with(|ctx| {
    let mut dims = Vec::with_capacity(ctx.thunkenv.borrow().arg.len());
    let mut tys_ = Vec::with_capacity(ctx.thunkenv.borrow().arg.len());
    for &(arg, _) in ctx.thunkenv.borrow().arg.iter() {
      let ty_ = match ctx.env.borrow()._lookup_ref_(arg) {
        Err(_) => panic!("bug"),
        Ok(e) => e.ty.clone()
      };
      dims.push(ty_.to_dim());
      tys_.push(ty_);
    }
    // FIXME FIXME: multiple arity out.
    let odim = match th.out_dim(&dims) {
      Err(_) => {
        println!("ERROR: thunk apply dim error: name={:?}", th.debug_name());
        println!("ERROR: thunk apply dim error: dims={:?}", &dims);
        println!("ERROR: thunk apply dim error: tys_={:?}", &tys_);
        panic!();
      }
      Ok(dim) => dim
    };
    let oty_ = match th.out_ty_(&tys_) {
      Err(_) => {
        println!("ERROR: thunk apply type error: name={:?}", th.debug_name());
        println!("ERROR: thunk apply type error: dims={:?}", &dims);
        println!("ERROR: thunk apply type error: tys_={:?}", &tys_);
        panic!();
      }
      Ok(ty_) => ty_
    };
    assert_eq!(odim, oty_.to_dim());
    ctx_pop_apply_thunk_(th, out, oty_)
  })
}

pub fn ctx_pop_apply_thunk_<Th: ThunkSpec_ + 'static>(th: Th, out: CellPtr, out_ty: CellType) {
  TL_CTX.with(|ctx| {
    let mut dims = Vec::with_capacity(ctx.thunkenv.borrow().arg.len());
    for &(arg, _) in ctx.thunkenv.borrow().arg.iter() {
      let ty_ = match ctx.env.borrow()._lookup_ref_(arg) {
        Err(_) => panic!("bug"),
        Ok(e) => e.ty.clone()
      };
      dims.push(ty_.to_dim());
      let spine = ctx.spine.borrow();
      spine.push_seal(arg);
    }
    // FIXME FIXME: multiple arity out.
    let odim = out_ty.to_dim();
    let oty_ = out_ty;
    dims.push(odim);
    let (lar, rar) = match th.arity() {
      None => unimplemented!(),
      Some(ar) => ar
    };
    // FIXME FIXME
    //let tp = ctx.thunkenv.borrow_mut().insert_(ctx, ar_in, ar_out, dims, th);
    let tp = ctx.thunkenv.borrow_mut().lookup_or_insert(&ctx.ctr, lar, rar, dims, th);
    let y = out;
    let yroot = match ctx.env.borrow()._lookup_ref_(y) {
      Err(_) => panic!("bug"),
      Ok(e) => {
        assert_eq!(e.ty, &oty_);
        e.root()
      }
    };
    let spine = ctx.spine.borrow();
    assert_eq!(yroot, spine._deref(y));
    spine.apply(y, tp);
    let yclk = spine._version(y).unwrap();
    let mut arg = Vec::new();
    swap(&mut arg, &mut ctx.thunkenv.borrow_mut().arg);
    let mut param = Vec::new();
    swap(&mut param, &mut ctx.thunkenv.borrow_mut().param);
    ctx.thunkenv.borrow_mut().update(yroot, y, yclk, tp, arg, param);
  })
}

pub fn ctx_pop_initialize_thunk<Th: ThunkSpec_ + 'static>(th: Th, out: CellPtr) {
  TL_CTX.with(|ctx| {
    let mut dims = Vec::with_capacity(ctx.thunkenv.borrow().arg.len());
    let mut tys_ = Vec::with_capacity(ctx.thunkenv.borrow().arg.len());
    for &(arg, _) in ctx.thunkenv.borrow().arg.iter() {
      let ty_ = match ctx.env.borrow()._lookup_ref_(arg) {
        Err(_) => panic!("bug"),
        Ok(e) => e.ty.clone()
      };
      dims.push(ty_.to_dim());
      tys_.push(ty_);
    }
    let odim = match th.out_dim(&dims) {
      Err(_) => {
        println!("ERROR: thunk apply dim error: {:?}", &dims);
        panic!();
      }
      Ok(dim) => dim
    };
    let oty_ = match th.out_ty_(&tys_) {
      Err(_) => {
        println!("ERROR: thunk apply type error: {:?}", &tys_);
        panic!();
      }
      Ok(ty_) => ty_
    };
    assert_eq!(odim, oty_.to_dim());
    ctx_pop_initialize_thunk_(th, out, oty_)
  })
}

pub fn ctx_pop_initialize_thunk_<Th: ThunkSpec_ + 'static>(th: Th, out: CellPtr, out_ty: CellType) {
  TL_CTX.with(|ctx| {
    let mut dims = Vec::with_capacity(ctx.thunkenv.borrow().arg.len());
    for &(arg, _) in ctx.thunkenv.borrow().arg.iter() {
      let ty_ = match ctx.env.borrow()._lookup_ref_(arg) {
        Err(_) => panic!("bug"),
        Ok(e) => e.ty.clone()
      };
      dims.push(ty_.to_dim());
      let spine = ctx.spine.borrow();
      spine.push_seal(arg);
    }
    let odim = out_ty.to_dim();
    let oty_ = out_ty;
    dims.push(odim);
    let (lar, rar) = match th.arity() {
      None => unimplemented!(),
      Some(ar) => ar
    };
    assert_eq!(rar, 1);
    //let tp = ctx.thunkenv.borrow_mut().insert_(ctx, ar_in, ar_out, dims, th);
    let tp = ctx.thunkenv.borrow_mut().lookup_or_insert(&ctx.ctr, lar, rar, dims, th);
    //let y = ctx.ctr.fresh_cel();
    let y = out;
    let yroot = match ctx.env.borrow()._lookup_ref_(y) {
      Err(_) => panic!("bug"),
      Ok(e) => {
        assert_eq!(e.ty, &oty_);
        e.root()
      }
    };
    let spine = ctx.spine.borrow();
    assert_eq!(yroot, spine._deref(y));
    //spine.uninit(y);
    spine.initialize(y, tp);
    let yclk = spine._version(y).unwrap();
    let mut arg = Vec::new();
    swap(&mut arg, &mut ctx.thunkenv.borrow_mut().arg);
    let mut param = Vec::new();
    swap(&mut param, &mut ctx.thunkenv.borrow_mut().param);
    ctx.thunkenv.borrow_mut().update(yroot, y, yclk, tp, arg, param);
  })
}

pub fn ctx_pop_accumulate_thunk<Th: ThunkSpec_ + 'static>(th: Th, out: CellPtr) {
  TL_CTX.with(|ctx| {
    let mut dims = Vec::with_capacity(ctx.thunkenv.borrow().arg.len());
    let mut tys_ = Vec::with_capacity(ctx.thunkenv.borrow().arg.len());
    for &(arg, _) in ctx.thunkenv.borrow().arg.iter() {
      let ty_ = match ctx.env.borrow()._lookup_ref_(arg) {
        Err(_) => panic!("bug"),
        Ok(e) => e.ty.clone()
      };
      dims.push(ty_.to_dim());
      tys_.push(ty_);
      let spine = ctx.spine.borrow();
      spine.push_seal(arg);
    }
    let odim = match th.out_dim(&dims) {
      Err(_) => {
        println!("ERROR: thunk apply dim error: {:?}", &dims);
        panic!();
      }
      Ok(dim) => dim
    };
    let oty_ = match th.out_ty_(&tys_) {
      Err(_) => {
        println!("ERROR: thunk apply type error: {:?}", &tys_);
        panic!();
      }
      Ok(ty_) => ty_
    };
    assert_eq!(odim, oty_.to_dim());
    dims.push(odim);
    let (lar, rar) = match th.arity() {
      None => unimplemented!(),
      Some(ar) => ar
    };
    assert_eq!(rar, 1);
    //let tp = ctx.thunkenv.borrow_mut().insert_(ctx, ar_in, ar_out, dims, th);
    let tp = ctx.thunkenv.borrow_mut().lookup_or_insert(&ctx.ctr, lar, rar, dims, th);
    let y = out;
    let yroot = match ctx.env.borrow()._lookup_ref_(y) {
      Err(_) => panic!("bug"),
      Ok(e) => {
        assert_eq!(e.ty, &oty_);
        e.root()
      }
    };
    let spine = ctx.spine.borrow();
    assert_eq!(yroot, spine._deref(y));
    spine.accumulate(y, tp);
    let yclk = spine._version(y).unwrap();
    let mut arg = Vec::new();
    swap(&mut arg, &mut ctx.thunkenv.borrow_mut().arg);
    let mut param = Vec::new();
    swap(&mut param, &mut ctx.thunkenv.borrow_mut().param);
    ctx.thunkenv.borrow_mut().update(yroot, y, yclk, tp, arg, param);
  })
}

/*pub fn ctx_bar() {
  unimplemented!();
}*/

/*pub fn ctx_gc() {
  TL_CTX.with(|ctx| {
    // FIXME FIXME
    let mut gc_list = Vec::new();
    ctx.env.borrow().gc_prepare(&mut gc_list);
    ctx.env.borrow_mut().gc(&gc_list);
    //ctx.thunkenv.borrow_mut().gc(&gc_list);
  })
}*/

pub struct CtxCtr {
  pub ptr_ctr:  Cell<i64>,
  pub celfront: RefCell<Vec<CellPtr>>,
}

impl CtxCtr {
  pub fn new() -> CtxCtr {
    CtxCtr{
      ptr_ctr:  Cell::new(0),
      celfront: RefCell::new(Vec::new()),
    }
  }
}

impl CtxCtr {
  pub fn fresh_cel(&self) -> CellPtr {
    let next = self._fresh();
    let x = CellPtr::from_unchecked(next);
    self.celfront.borrow_mut().push(x);
    x
  }

  pub fn fresh_mcel(&self) -> MCellPtr {
    let next = self._fresh();
    MCellPtr::from_unchecked(next)
  }

  pub fn fresh_thunk(&self) -> ThunkPtr {
    let next = self._fresh();
    ThunkPtr::from_unchecked(next)
  }

  pub fn _fresh(&self) -> i64 {
    let next = self.ptr_ctr.get() + 1;
    assert!(next > 0);
    assert!(next < i64::max_value());
    self.ptr_ctr.set(next);
    next
  }
}

pub struct ThunkEnvEntry {
  pub pthunk:   Rc<PThunk>,
}

#[derive(Debug)]
pub struct ThunkClosure {
  pub pthunk:   ThunkPtr,
  pub arg:      Vec<(CellPtr, Clock)>,
  pub param:    Vec<ScalarVal_>,
  pub out:      CellPtr,
  pub pmach:    PMach,
}

#[derive(Default)]
pub struct CtxThunkEnv {
  pub thunktab: HashMap<ThunkPtr, ThunkEnvEntry>,
  pub thunkidx: HashMap<(u16, u16, Vec<Dim>, ThunkKey, ), ThunkPtr>,
  pub update:   HashMap<(CellPtr, Clock), ThunkClosure>,
  pub arg:      Vec<(CellPtr, Clock)>,
  pub param:    Vec<ScalarVal_>,
  //pub out:      Vec<(CellPtr, Clock)>,
}

impl CtxThunkEnv {
  pub fn reset(&mut self) {
    // FIXME FIXME
    //self.update.clear();
  }

  pub fn update(&mut self, yroot: CellPtr, y: CellPtr, yclk: Clock, tp: ThunkPtr, arg: Vec<(CellPtr, Clock)>, param: Vec<ScalarVal_>) {
    match self.thunktab.get(&tp) {
      None => panic!("bug"),
      Some(te) => {
        // FIXME: where to typecheck?
        assert_eq!(arg.len(), te.pthunk.lar as usize);
        assert_eq!(1, te.pthunk.rar);
        let pmach = TL_CTX.with(|ctx| ctx.ctlstate.primary.get())
            .unwrap_or_else(|| TL_PCTX.with(|pctx| pctx.fastest_pmach()));
        let tclo = ThunkClosure{pthunk: tp, arg, param, out: y, pmach};
        self.update.insert((yroot, yclk), tclo);
      }
    }
  }

  pub fn lookup_or_insert<Th: ThunkSpec_ + 'static>(&mut self, ctr: &CtxCtr, ar_in: u16, ar_out: u16, spec_dim: Vec<Dim>, th: Th) -> ThunkPtr {
    let mut tp_ = None;
    let tk = ThunkKey(Rc::new(th));
    let key = (ar_in, ar_out, spec_dim, tk, );
    match self.thunkidx.get(&key) {
      None => {}
      Some(&tp) => {
        match self.thunktab.get(&tp) {
          None => {
            // NB: this might happen due to thunk gc.
          }
          Some(te) => {
            assert!(((key.3).0).thunk_eq(&*te.pthunk.spec_).unwrap_or(false));
            tp_ = Some(tp);
          }
        }
      }
    }
    let (lar, rar, spec_dim, tk, ) = key;
    if tp_.is_none() {
      let tp = ctr.fresh_thunk();
      let pthunk = Rc::new(PThunk::new(tp, spec_dim.clone(), (tk.0).clone()));
      let te = ThunkEnvEntry{pthunk};
      self.thunkidx.insert((lar, rar, spec_dim, tk, ), tp);
      self.thunktab.insert(tp, te);
      tp_ = Some(tp);
    }
    tp_.unwrap()
  }

  pub fn gc(&mut self, gc_list: &[CellPtr]) {
    // FIXME FIXME: remove updating thunks.
  }
}

#[derive(Clone, Debug)]
pub struct CellClosure {
  pub ctr:      Counter,
  pub thunk:    Vec<ThunkPtr>,
}

impl Default for CellClosure {
  fn default() -> CellClosure {
    CellClosure{
      ctr:      Counter::default(),
      thunk:    Vec::new(),
    }
  }
}

impl CellClosure {
  pub fn init_once(&mut self, clk: Clock, th: ThunkPtr) {
    assert!(clk.is_init_once());
    if self.ctr > clk.ctr() {
      panic!("bug");
    } else if self.ctr < clk.ctr() {
      self.ctr = clk.ctr();
      self.thunk.clear();
    }
    assert_eq!(clk.up as usize, self.thunk.len());
    self.thunk.push(th);
  }

  pub fn update(&mut self, clk: Clock, th: ThunkPtr) {
    assert!(clk.is_update());
    if self.ctr > clk.ctr() {
      panic!("bug");
    } else if self.ctr < clk.ctr() {
      panic!("bug");
    }
    if !(clk.up as usize == self.thunk.len()) {
      println!("DEBUG: CellClosure::update: clk={:?} th={:?} self.ctr={:?} self.thunk={:?}",
          clk, th, &self.ctr, &self.thunk);
    }
    assert_eq!(clk.up as usize, self.thunk.len());
    self.thunk.push(th);
  }
}

pub struct CowCell {
  pub optr: CellPtr,
  pub pcel: CellPtr,
  pub pclk: Clock,
}

pub enum CellAlias {
  View(CellVOp),
  //NewType,
  NewShape,
  BitAlias,
  Opaque,
  Const_,
}

#[derive(Clone, Copy, Debug)]
pub enum CellName {
  Top,
  Phy,
  Cow,
  Alias,
  //VAlias,
  Bot,
}

pub enum Cell_ {
  Top(RefCell<CellState>, CellPtr),
  Phy(RefCell<CellState>, RefCell<CellClosure>, PCell),
  Cow(RefCell<CellState>, RefCell<CellClosure>, CowCell),
  Alias(CellAlias, CellPtr),
  Bot,
}

impl Cell_ {
  pub fn name(&self) -> CellName {
    match self {
      &Cell_::Top(..) => CellName::Top,
      &Cell_::Phy(..) => CellName::Phy,
      &Cell_::Cow(..) => CellName::Cow,
      &Cell_::Alias(..) => CellName::Alias,
      //&Cell_::VAlias(..) => CellName::VAlias,
      &Cell_::Bot => CellName::Bot,
    }
  }

  pub fn state_ref(&self) -> Ref<CellState> {
    match self {
      &Cell_::Top(ref state, ..) => state.borrow(),
      &Cell_::Phy(ref state, ..) => state.borrow(),
      &Cell_::Cow(ref state, ..) => state.borrow(),
      _ => panic!("bug")
    }
  }

  pub fn state_mut(&self) -> RefMut<CellState> {
    match self {
      &Cell_::Top(ref state, ..) => state.borrow_mut(),
      &Cell_::Phy(ref state, ..) => state.borrow_mut(),
      &Cell_::Cow(ref state, ..) => state.borrow_mut(),
      _ => panic!("bug")
    }
  }

  /*pub fn swap_in(&mut self, state: RefCell<CellState>, p: PCell) {
    let mut ret = Cell_::Phy(state, p);
    swap(self, &mut ret);
    match ret {
      Cell_::Bot => {}
      _ => panic!("bug")
    }
  }

  pub fn swap_out(&mut self) -> (RefCell<CellState>, PCell) {
    let mut ret = Cell_::Bot;
    swap(self, &mut ret);
    match ret {
      Cell_::Phy(state, p) => (state, p),
      _ => panic!("bug")
    }
  }*/
}

pub enum MCell_ {
  // FIXME
  //Tup(_),
  Set(MCellSet),
  Map(MCellMap),
  //Tup(RefCell<CellState>, MCellTup),
  //Set(RefCell<CellState>, MCellSet),
  //Map(RefCell<CellState>, MCellMap),
}

/*#[derive(Clone, Copy, Default)]
#[repr(transparent)]
pub struct CellEFlag {
  bits: u8,
}

impl CellEFlag {
  /*pub fn reset(&mut self) {
    self.bits = 0;
  }*/

  pub fn mutex(&self) -> bool {
    (self.bits & 1) != 0
  }

  pub fn set_mutex(&mut self) {
    self.bits |= 1;
  }

  pub fn rwlock(&self) -> bool {
    (self.bits & 2) != 0
  }

  pub fn set_rwlock(&mut self) {
    self.bits |= 2;
  }

  pub fn read(&self) -> bool {
    (self.bits & 4) != 0
  }

  pub fn set_read(&mut self) {
    self.bits |= 4;
  }

  /*pub fn opaque(&self) -> bool {
    (self.bits & 0x10) != 0
  }

  pub fn set_opaque(&mut self) {
    self.bits |= 0x10;
  }

  pub fn profile(&self) -> bool {
    (self.bits & 0x20) != 0
  }

  pub fn set_profile(&mut self) {
    self.bits |= 0x20;
  }

  pub fn trace(&self) -> bool {
    (self.bits & 0x40) != 0
  }

  pub fn set_trace(&mut self) {
    self.bits |= 0x40;
  }

  pub fn break_(&self) -> bool {
    (self.bits & 0x80) != 0
  }

  pub fn set_break(&mut self) {
    self.bits |= 0x80;
  }*/
}*/

pub struct CellEnvEntry {
  // FIXME
  pub ty:       CellType,
  pub stablect: Cell<u32>,
  //pub snapshot: Cell<u32>,
  //pub eflag:    CellEFlag,
  pub cel_:     Cell_,
}

impl CellEnvEntry {
  pub fn state_ref(&self) -> Ref<CellState> {
    self.cel_.state_ref()
  }
}

/*pub struct CellEnvEntryRef<'a> {
  pub root:     CellPtr,
  pub stablect: &'a Cell<u32>,
  //pub snapshot: &'a Cell<u32>,
  pub ty:       &'a CellType,
  //pub eflag:    CellEFlag,
  pub cel_:     &'a Cell_,
}

impl<'a> CellEnvEntryRef<'a> {
  pub fn state_ref(&self) -> Ref<CellState> {
    self.cel_.state_ref()
  }

  pub fn state(&self) -> RefMut<CellState> {
    self.cel_.state_mut()
  }

  pub fn clock_sync(self, prev_clk: Clock, next_clk: Clock, env: &CtxEnv) {
    let mut cursor = self;
    loop {
      assert_eq!(cursor.cel_.state_ref().clk, prev_clk);
      cursor.cel_.state_mut().clk = next_clk;
      break;
      /*match cursor.cel_ {
        &Cell_::Top(..) => {
          break;
        }
        &Cell_::Phy(.., ref pcel) => {
          for (_, rep) in pcel.replicas.iter() {
            if rep.clk.get() == prev_clk {
              rep.clk.set(next_clk);
            }
          }
          break;
        }
        &Cell_::Cow(.., ref cow) => {
          // FIXME FIXME
          unimplemented!();
          /*match env.lookup_ref(cow.pcel) {
            None => panic!("bug"),
            Some(e) => {
              cursor = e;
            }
          }*/
        }
        _ => panic!("bug")
      }*/
    }
  }

  pub fn clock_sync_rec(self, prev_clk: Clock, next_clk: Clock, env: &CtxEnv) {
    let mut cursor = self;
    loop {
      assert_eq!(cursor.cel_.state_ref().clk, prev_clk);
      cursor.cel_.state_mut().clk = next_clk;
      match cursor.cel_ {
        &Cell_::Top(..) => {
          break;
        }
        &Cell_::Phy(.., ref pcel) => {
          for (_, rep) in pcel.replicas.iter() {
            if rep.clk.get() == prev_clk {
              rep.clk.set(next_clk);
            }
          }
          break;
        }
        &Cell_::Cow(.., ref cow) => {
          // FIXME FIXME
          unimplemented!();
          /*match env.lookup_ref(cow.pcel) {
            None => panic!("bug"),
            Some(e) => {
              cursor = e;
            }
          }*/
        }
        _ => panic!("bug")
      }
    }
  }

  pub fn clock_sync_loc(self, loc: Locus, prev_clk: Clock, next_clk: Clock, env: &CtxEnv) {
    let mut cursor = self;
    loop {
      assert_eq!(cursor.cel_.state_ref().clk, prev_clk);
      cursor.cel_.state_mut().clk = next_clk;
      match cursor.cel_ {
        &Cell_::Top(..) => {
          break;
        }
        &Cell_::Phy(.., ref pcel) => {
          for (key, rep) in pcel.replicas.iter() {
            if key.as_ref().0 == loc && rep.clk.get() == prev_clk {
              rep.clk.set(next_clk);
            }
          }
          break;
        }
        &Cell_::Cow(.., ref cow) => {
          // FIXME FIXME
          unimplemented!();
          /*match env.lookup_ref(cow.pcel) {
            None => panic!("bug"),
            Some(e) => {
              cursor = e;
            }
          }*/
        }
        _ => panic!("bug")
      }
    }
  }
}

pub struct CellEnvEntryMutRef<'a> {
  pub root:     CellPtr,
  pub stablect: &'a Cell<u32>,
  //pub snapshot: &'a Cell<u32>,
  pub ty:       CellType,
  //pub eflag:    &'a mut CellEFlag,
  pub cel_:     &'a mut Cell_,
}

impl<'a> CellEnvEntryMutRef<'a> {
  pub fn state(&self) -> RefMut<CellState> {
    self.cel_.state_mut()
  }
}*/

pub struct MCellEnvEntry {
  // FIXME
  pub mcel_:    MCell_,
}

pub struct CellRef_<'a, R> {
  pub ref_:     R,
  pub root_ty:  &'a CellType,
  pub ty:       &'a CellType,
  pub stablect: &'a Cell<u32>,
  pub cel_:     &'a Cell_,
}

impl<'a> CellRef_<'a, CellPtr> {
  pub fn root(&self) -> CellPtr {
    self.ref_
  }

  pub fn state(&self) -> RefMut<CellState> {
    self.cel_.state_mut()
  }

  pub fn clock_sync(self, prev_clk: Clock, next_clk: Clock, env: &CtxEnv) {
    let cursor = self;
    assert_eq!(cursor.cel_.state_ref().clk, prev_clk);
    cursor.cel_.state_mut().clk = next_clk;
  }

  pub fn clock_sync_rec(self, prev_clk: Clock, next_clk: Clock, env: &CtxEnv) {
    let mut cursor = self;
    loop {
      assert_eq!(cursor.cel_.state_ref().clk, prev_clk);
      cursor.cel_.state_mut().clk = next_clk;
      match cursor.cel_ {
        &Cell_::Top(..) => {
          break;
        }
        &Cell_::Phy(.., ref pcel) => {
          for (_, rep) in pcel.replicas.iter() {
            if rep.clk.get() > prev_clk {
              panic!("bug");
            } else if rep.clk.get() == prev_clk {
              rep.clk.set(next_clk);
            }
          }
          break;
        }
        &Cell_::Cow(.., ref cow) => {
          // FIXME FIXME
          unimplemented!();
          /*match env.lookup_ref(cow.pcel) {
            None => panic!("bug"),
            Some(e) => {
              cursor = e;
            }
          }*/
        }
        _ => panic!("bug")
      }
    }
  }
}

impl<'a> CellRef_<'a, CellView> {
  pub fn root(&self) -> CellPtr {
    self.ref_.root
  }

  pub fn view(&self) -> &CellView {
    &self.ref_
  }

  pub fn view_mut(&mut self) -> &mut CellView {
    &mut self.ref_
  }
}

pub struct CellMutRef_<'a, R> {
  pub ref_:     R,
  pub root_ty:  &'a CellType,
  pub ty:       CellType,
  pub stablect: &'a Cell<u32>,
  pub cel_:     &'a mut Cell_,
}

impl<'a> CellMutRef_<'a, CellPtr> {
  pub fn root(&self) -> CellPtr {
    self.ref_
  }

  pub fn state(&self) -> RefMut<CellState> {
    self.cel_.state_mut()
  }
}

impl<'a> CellMutRef_<'a, CellView> {
  pub fn root(&self) -> CellPtr {
    self.ref_.root
  }

  pub fn view(&self) -> &CellView {
    &self.ref_
  }
}

pub type CellDerefResult<T=CellPtr> = Result<T, CellDerefErr>;

pub type CellProbePtr = CellDerefResult<CellPtr>;
pub type CellProbeView = CellDerefResult<CellView>;

pub type CellDerefPtr<'a> = CellDerefResult<CellRef_<'a, CellPtr>>;
pub type CellDerefView<'a> = CellDerefResult<CellRef_<'a, CellView>>;

pub type CellMutDerefPtr<'a> = CellDerefResult<CellMutRef_<'a, CellPtr>>;
pub type CellMutDerefView<'a> = CellDerefResult<CellMutRef_<'a, CellView>>;

#[derive(Clone, Copy, Debug)]
pub enum CellDerefErr {
  MissingRoot,
  Missing,
  Read,
  Write,
  View,
  Bot,
}

//#[derive(Default)]
pub struct CtxEnv {
  // FIXME
  //pub alias_root:   RefCell<HashMap<CellPtr, CellPtr>>,
  pub cow_root:     RefCell<HashMap<CellPtr, (CellPtr, Clock)>>,
  //pub stable:   HashSet<CellPtr>,
  pub celtab:   HashMap<CellPtr, CellEnvEntry>,
  pub snapshot: HashMap<(CellPtr, Clock), Vec<CellPtr>>,
  /*pub atomtab:  HashMap<Atom, ()>,*/
  pub mceltab:  HashMap<MCellPtr, MCellEnvEntry>,
  //pub tag:      HashMap<CellPtr, HashSet<String>>,
  //pub tag:      HashMap<CellPtr, Vec<String>>,
}

impl Default for CtxEnv {
  fn default() -> CtxEnv {
    CtxEnv{
      //alias_root:   RefCell::new(HashMap::new()),
      cow_root:     RefCell::new(HashMap::new()),
      celtab:   HashMap::new(),
      snapshot: HashMap::new(),
      /*atomtab:  HashMap::new(),*/
      mceltab:  HashMap::new(),
      //tag:      HashMap::new(),
    }
  }
}

impl CtxEnv {
  pub fn reset(&mut self) {
    // FIXME FIXME
    self.celtab.clear();
    //self.alias_root.borrow_mut().clear();
    //self.tag.clear();
  }

  pub fn _probe_ref(&self, query: CellPtr) -> CellProbePtr {
    let mut cursor = query;
    loop {
      match self.celtab.get(&cursor) {
        None => {
          return Err(CellDerefErr::Missing);
        }
        Some(e) => {
          match &e.cel_ {
            &Cell_::Top(..) |
            &Cell_::Phy(..) |
            &Cell_::Cow(..) |
            &Cell_::Bot => {
              return Ok(cursor);
            }
            &Cell_::Alias(ref alias, next) => {
              match alias {
                &CellAlias::View(_) => {
                  return Err(CellDerefErr::View);
                }
                //&CellAlias::NewType |
                &CellAlias::NewShape |
                &CellAlias::BitAlias |
                &CellAlias::Opaque |
                &CellAlias::Const_ => {}
              }
              cursor = next;
            }
            _ => unimplemented!()
          }
        }
      }
    }
  }

  pub fn _probe_view(&self, query: CellPtr) -> CellProbeView {
    let mut cursor = query;
    let mut view = CellView::default();
    loop {
      match self.celtab.get(&cursor) {
        None => {
          return Err(CellDerefErr::Missing);
        }
        Some(e) => {
          match &e.cel_ {
            &Cell_::Top(..) |
            &Cell_::Phy(..) |
            &Cell_::Cow(..) => {
              view.root = cursor;
              //view.r_ty = e.ty.clone();
              view.vlog.reverse();
              return Ok(view);
            }
            &Cell_::Alias(ref alias, next) => {
              match alias {
                &CellAlias::View(ref vop) => {
                  view.vlog.push(vop.clone());
                }
                &CellAlias::NewShape => {
                  view.vlog.push(CellViewOp::new_shape(&e.ty.shape));
                }
                &CellAlias::BitAlias |
                &CellAlias::Opaque |
                &CellAlias::Const_ => {}
              }
              cursor = next;
            }
            &Cell_::Bot => {
              return Err(CellDerefErr::Bot);
            }
            _ => unimplemented!()
          }
        }
      }
    }
  }

  pub fn _lookup_ref_(&self, query: CellPtr) -> CellDerefPtr {
    let ty = match self.celtab.get(&query) {
      None => return Err(CellDerefErr::Missing),
      Some(e) => {
        match &e.cel_ {
          &Cell_::Top(.., optr) => {
            assert_eq!(query, optr);
          }
          &Cell_::Phy(.., ref cel) => {
            assert_eq!(query, cel.optr);
          }
          &Cell_::Cow(.., ref cel) => {
            assert_eq!(query, cel.optr);
          }
          &Cell_::Alias(..) => {}
          &Cell_::Bot => {
            return Err(CellDerefErr::Bot);
          }
          _ => unimplemented!()
        }
        match &e.cel_ {
          &Cell_::Top(..) |
          &Cell_::Phy(..) |
          &Cell_::Cow(..) => {
            return Ok(CellRef_{
              ref_: query,
              root_ty: &e.ty,
              ty: &e.ty,
              stablect: &e.stablect,
              cel_: &e.cel_,
            });
          }
          &Cell_::Alias(..) => {}
          _ => unreachable!()
        }
        &e.ty
      }
    };
    let root = self._probe_ref(query)?;
    match self.celtab.get(&root) {
      None => return Err(CellDerefErr::MissingRoot),
      Some(e) => {
        //assert_eq!(root, e.root());
        match &e.cel_ {
          &Cell_::Top(.., optr) => {
            assert_eq!(root, optr);
          }
          &Cell_::Phy(.., ref cel) => {
            assert_eq!(root, cel.optr);
          }
          &Cell_::Cow(.., ref cel) => {
            assert_eq!(root, cel.optr);
          }
          &Cell_::Alias(..) => {
            panic!("bug");
          }
          &Cell_::Bot => {
            return Err(CellDerefErr::Bot);
          }
          _ => unimplemented!()
        }
        // FIXME: type compat.
        if e.ty.dtype != Dtype::_Top {
          assert_eq!(ty.dtype.size_bytes(), e.ty.dtype.size_bytes());
          assert!(ty.shape_compat(&e.ty) != ShapeCompat::Incompat);
        }
        Ok(CellRef_{
          ref_: root,
          root_ty: &e.ty,
          ty,
          stablect: &e.stablect,
          cel_: &e.cel_,
        })
      }
    }
  }

  pub fn _lookup_view(&self, query: CellPtr) -> CellDerefView {
    let ty = match self.celtab.get(&query) {
      None => return Err(CellDerefErr::Missing),
      Some(e) => {
        match &e.cel_ {
          &Cell_::Top(.., optr) => {
            assert_eq!(query, optr);
          }
          &Cell_::Phy(.., ref cel) => {
            assert_eq!(query, cel.optr);
          }
          &Cell_::Cow(.., ref cel) => {
            assert_eq!(query, cel.optr);
          }
          &Cell_::Alias(..) => {}
          &Cell_::Bot => {
            return Err(CellDerefErr::Bot);
          }
          _ => unimplemented!()
        }
        match &e.cel_ {
          &Cell_::Top(..) |
          &Cell_::Phy(..) |
          &Cell_::Cow(..) => {
            return Ok(CellRef_{
              ref_: query.into(),
              //ref_: CellView::new(query, e.ty.clone()),
              root_ty: &e.ty,
              ty: &e.ty,
              stablect: &e.stablect,
              cel_: &e.cel_,
            });
          }
          &Cell_::Alias(..) => {}
          _ => unreachable!()
        }
        &e.ty
      }
    };
    let view = self._probe_view(query)?;
    match self.celtab.get(&view.root) {
      None => return Err(CellDerefErr::MissingRoot),
      Some(e) => {
        //assert_eq!(view.root, e.root());
        match &e.cel_ {
          &Cell_::Top(.., optr) => {
            assert_eq!(view.root, optr);
          }
          &Cell_::Phy(.., ref cel) => {
            assert_eq!(view.root, cel.optr);
          }
          &Cell_::Cow(.., ref cel) => {
            assert_eq!(view.root, cel.optr);
          }
          &Cell_::Alias(..) => {
            panic!("bug");
          }
          &Cell_::Bot => {
            return Err(CellDerefErr::Bot);
          }
          _ => unimplemented!()
        }
        Ok(CellRef_{
          ref_: view,
          root_ty: &e.ty,
          ty,
          stablect: &e.stablect,
          cel_: &e.cel_,
        })
      }
    }
  }

  pub fn _pre_probe(&self, query: CellPtr) -> CellDerefResult<(CellType, bool)> {
    let mut noalias = false;
    let ty = match self.celtab.get(&query) {
      None => return Err(CellDerefErr::Missing),
      Some(e) => {
        match &e.cel_ {
          &Cell_::Top(.., optr) => {
            assert_eq!(query, optr);
          }
          &Cell_::Phy(.., ref cel) => {
            assert_eq!(query, cel.optr);
          }
          &Cell_::Cow(.., ref cel) => {
            assert_eq!(query, cel.optr);
          }
          &Cell_::Alias(..) => {}
          &Cell_::Bot => {
            return Err(CellDerefErr::Bot);
          }
          _ => unimplemented!()
        }
        match &e.cel_ {
          &Cell_::Top(..) |
          &Cell_::Phy(..) |
          &Cell_::Cow(..) => {
            noalias = true;
          }
          &Cell_::Alias(..) => {}
          _ => unreachable!()
        }
        e.ty.clone()
      }
    };
    Ok((ty, noalias))
  }

  pub fn _lookup_mut_ref_(&mut self, query: CellPtr) -> CellMutDerefPtr {
    let (ty, noalias) = self._pre_probe(query)?;
    let root = if !noalias { self._probe_ref(query)? } else { query };
    match self.celtab.get_mut(&root) {
      None => return Err(CellDerefErr::MissingRoot),
      Some(e) => {
        //assert_eq!(root, e.root());
        match &mut e.cel_ {
          &mut Cell_::Top(.., optr) => {
            assert_eq!(root, optr);
            return Ok(CellMutRef_{
              ref_: root,
              root_ty: &e.ty,
              ty,
              stablect: &e.stablect,
              cel_: &mut e.cel_,
            });
          }
          &mut Cell_::Phy(ref state, .., ref mut pcel) => {
            assert_eq!(root, pcel.optr);
            assert_eq!(&e.ty, &pcel.ogty);
            return Ok(CellMutRef_{
              ref_: root,
              root_ty: &e.ty,
              ty,
              stablect: &e.stablect,
              cel_: &mut e.cel_,
            });
          }
          &mut Cell_::Cow(.., ref cel) => {
            assert_eq!(root, cel.optr);
            unimplemented!();
          }
          &mut Cell_::Alias(..) => {
            panic!("bug");
          }
          &mut Cell_::Bot => {
            return Err(CellDerefErr::Bot);
          }
          _ => unimplemented!()
        }
      }
    }
  }

  pub fn _lookup_mut_view(&mut self, query: CellPtr) -> CellMutDerefView {
    let (ty, noalias) = self._pre_probe(query)?;
    let view = if !noalias { self._probe_view(query)? } else { query.into() };
    let root = view.root();
    match self.celtab.get_mut(&root) {
      None => return Err(CellDerefErr::MissingRoot),
      Some(e) => {
        //assert_eq!(root, e.root());
        match &mut e.cel_ {
          &mut Cell_::Top(.., optr) => {
            assert_eq!(root, optr);
            return Ok(CellMutRef_{
              ref_: view,
              root_ty: &e.ty,
              ty,
              stablect: &e.stablect,
              cel_: &mut e.cel_,
            });
          }
          &mut Cell_::Phy(ref state, .., ref mut pcel) => {
            assert_eq!(root, pcel.optr);
            assert_eq!(&e.ty, &pcel.ogty);
            return Ok(CellMutRef_{
              ref_: view,
              root_ty: &e.ty,
              ty,
              stablect: &e.stablect,
              cel_: &mut e.cel_,
            });
          }
          &mut Cell_::Cow(.., ref cel) => {
            assert_eq!(root, cel.optr);
            unimplemented!();
          }
          &mut Cell_::Alias(..) => {
            panic!("bug");
          }
          &mut Cell_::Bot => {
            return Err(CellDerefErr::Bot);
          }
          _ => unimplemented!()
        }
      }
    }
  }

  pub fn pread_ref_(&mut self, query: CellPtr, clk: Clock, loc: Locus) -> CellMutDerefPtr {
    let (ty, noalias) = self._pre_probe(query)?;
    let root = if !noalias { self._probe_ref(query)? } else { query };
    match self.celtab.get(&root) {
      None => return Err(CellDerefErr::MissingRoot),
      Some(e) => {
        //assert_eq!(root, e.root());
        match &e.cel_ {
          &Cell_::Top(.., optr) => {
            assert_eq!(root, optr);
            panic!("bug");
            //return Err(CellDerefErr::Read);
          }
          &Cell_::Phy(ref state, .., ref pcel) => {
            assert_eq!(root, pcel.optr);
            if &e.ty != &pcel.ogty {
              println!("DEBUG: CellEnv::pread_ref_: root={:?} query={:?} clk={:?} loc={:?} ty={:?} root ty={:?} ogty={:?}",
                  root, query, clk, loc, &ty, &e.ty, &pcel.ogty);
            }
            assert_eq!(&e.ty, &pcel.ogty);
            assert_eq!(clk, state.borrow().clk);
            match self.celtab.get_mut(&root) {
              None => panic!("bug"),
              Some(e) => {
                match &mut e.cel_ {
                  &mut Cell_::Phy(.., ref mut pcel) => {
                    pcel.read_loc(root, clk, &e.ty, loc);
                    return Ok(CellMutRef_{
                      ref_: root,
                      root_ty: &e.ty,
                      ty,
                      stablect: &e.stablect,
                      cel_: &mut e.cel_,
                    });
                  }
                  _ => unreachable!()
                }
              }
            }
          }
          &Cell_::Cow(.., ref cel) => {
            assert_eq!(root, cel.optr);
            assert!(!cel.pcel.is_nil());
            let pcel = cel.pcel;
            match self.celtab.get_mut(&pcel) {
              None => panic!("bug"),
              Some(e) => {
                match &mut e.cel_ {
                  &mut Cell_::Phy(.., ref mut pcel) => {
                    pcel.read_loc(root, clk, &e.ty, loc);
                    // FIXME: type compat.
                    return Ok(CellMutRef_{
                      ref_: root,
                      root_ty: &e.ty,
                      ty,
                      // FIXME: probably should be the cow stablect.
                      stablect: &e.stablect,
                      cel_: &mut e.cel_,
                    });
                  }
                  &mut Cell_::Cow(..) => {
                    // FIXME FIXME
                    unimplemented!();
                  }
                  _ => panic!("bug")
                }
              }
            }
          }
          &Cell_::Alias(..) => {
            panic!("bug");
          }
          &Cell_::Bot => {
            return Err(CellDerefErr::Bot);
          }
          _ => unimplemented!()
        }
      }
    }
  }

  pub fn pread_view(&mut self, query: CellPtr, clk: Clock, loc: Locus) -> CellMutDerefView {
    let (ty, noalias) = self._pre_probe(query)?;
    let view = if !noalias { self._probe_view(query)? } else { query.into() };
    let root = view.root();
    match self.celtab.get(&root) {
      None => return Err(CellDerefErr::MissingRoot),
      Some(e) => {
        //assert_eq!(root, e.root());
        match &e.cel_ {
          &Cell_::Top(.., optr) => {
            assert_eq!(root, optr);
            panic!("bug");
            //return Err(CellDerefErr::Read);
          }
          &Cell_::Phy(ref state, .., ref pcel) => {
            assert_eq!(root, pcel.optr);
            assert_eq!(&e.ty, &pcel.ogty);
            assert_eq!(clk, state.borrow().clk);
            match self.celtab.get_mut(&root) {
              None => panic!("bug"),
              Some(e) => {
                match &mut e.cel_ {
                  &mut Cell_::Phy(.., ref mut pcel) => {
                    pcel.read_loc(root, clk, &e.ty, loc);
                    return Ok(CellMutRef_{
                      ref_: view,
                      root_ty: &e.ty,
                      ty,
                      stablect: &e.stablect,
                      cel_: &mut e.cel_,
                    });
                  }
                  _ => unreachable!()
                }
              }
            }
          }
          &Cell_::Cow(.., ref cel) => {
            assert_eq!(root, cel.optr);
            assert!(!cel.pcel.is_nil());
            let pcel = cel.pcel;
            match self.celtab.get_mut(&pcel) {
              None => panic!("bug"),
              Some(e) => {
                match &mut e.cel_ {
                  &mut Cell_::Phy(.., ref mut pcel) => {
                    pcel.read_loc(root, clk, &e.ty, loc);
                    // FIXME: type compat.
                    return Ok(CellMutRef_{
                      ref_: view,
                      root_ty: &e.ty,
                      ty,
                      // FIXME: probably should be the cow stablect.
                      stablect: &e.stablect,
                      cel_: &mut e.cel_,
                    });
                  }
                  &mut Cell_::Cow(..) => {
                    // FIXME FIXME
                    unimplemented!();
                  }
                  _ => panic!("bug")
                }
              }
            }
          }
          &Cell_::Alias(..) => {
            panic!("bug");
          }
          &Cell_::Bot => {
            return Err(CellDerefErr::Bot);
          }
          _ => unimplemented!()
        }
      }
    }
  }

  pub fn pwrite_ref_(&mut self, query: CellPtr, next_clk: Clock, loc: Locus) -> CellMutDerefPtr {
    let (ty, noalias) = self._pre_probe(query)?;
    let root = if !noalias { self._probe_ref(query)? } else { query };
    match self.celtab.get_mut(&root) {
      None => return Err(CellDerefErr::MissingRoot),
      Some(e) => {
        //assert_eq!(root, e.root());
        match &mut e.cel_ {
          &mut Cell_::Top(ref state, optr) => {
            assert_eq!(root, optr);
            assert_eq!(next_clk, state.borrow().clk);
            let state = state.clone();
            let clo = RefCell::new(CellClosure::default());
            let mut pcel = PCell::new(optr, e.ty.clone());
            pcel.write_loc(root, next_clk, &e.ty, loc);
            e.cel_ = Cell_::Phy(state, clo, pcel);
            return Ok(CellMutRef_{
              ref_: root,
              root_ty: &e.ty,
              ty,
              stablect: &e.stablect,
              cel_: &mut e.cel_,
            });
          }
          &mut Cell_::Phy(ref state, .., ref mut pcel) => {
            assert_eq!(root, pcel.optr);
            assert_eq!(&e.ty, &pcel.ogty);
            assert_eq!(next_clk, state.borrow().clk);
            pcel.write_loc(root, next_clk, &e.ty, loc);
            return Ok(CellMutRef_{
              ref_: root,
              root_ty: &e.ty,
              ty,
              stablect: &e.stablect,
              cel_: &mut e.cel_,
            });
          }
          &mut Cell_::Cow(.., ref cel) => {
            assert_eq!(root, cel.optr);
            unimplemented!();
          }
          &mut Cell_::Alias(..) => {
            panic!("bug");
          }
          &mut Cell_::Bot => {
            return Err(CellDerefErr::Bot);
          }
          _ => unimplemented!()
        }
      }
    }
  }

  pub fn pwrite_view(&mut self, query: CellPtr, next_clk: Clock, loc: Locus) -> CellMutDerefView {
    let (ty, noalias) = self._pre_probe(query)?;
    let view = if !noalias { self._probe_view(query)? } else { query.into() };
    let root = view.root();
    match self.celtab.get_mut(&root) {
      None => return Err(CellDerefErr::MissingRoot),
      Some(e) => {
        //assert_eq!(root, e.root());
        match &mut e.cel_ {
          &mut Cell_::Top(ref state, optr) => {
            assert_eq!(root, optr);
            assert_eq!(next_clk, state.borrow().clk);
            let state = state.clone();
            let clo = RefCell::new(CellClosure::default());
            let mut pcel = PCell::new(optr, e.ty.clone());
            pcel.write_loc(root, next_clk, &e.ty, loc);
            e.cel_ = Cell_::Phy(state, clo, pcel);
            return Ok(CellMutRef_{
              ref_: view,
              root_ty: &e.ty,
              ty,
              stablect: &e.stablect,
              cel_: &mut e.cel_,
            });
          }
          &mut Cell_::Phy(ref state, .., ref mut pcel) => {
            assert_eq!(root, pcel.optr);
            assert_eq!(&e.ty, &pcel.ogty);
            assert_eq!(next_clk, state.borrow().clk);
            pcel.write_loc(root, next_clk, &e.ty, loc);
            return Ok(CellMutRef_{
              ref_: view,
              root_ty: &e.ty,
              ty,
              stablect: &e.stablect,
              cel_: &mut e.cel_,
            });
          }
          &mut Cell_::Cow(.., ref cel) => {
            assert_eq!(root, cel.optr);
            unimplemented!();
          }
          &mut Cell_::Alias(..) => {
            panic!("bug");
          }
          &mut Cell_::Bot => {
            return Err(CellDerefErr::Bot);
          }
          _ => unimplemented!()
        }
      }
    }
  }

  pub fn prewrite_ref_(&mut self, query: CellPtr, prev_clk: Clock, next_clk: Clock, loc: Locus) -> CellMutDerefPtr {
    let (ty, noalias) = self._pre_probe(query)?;
    let root = if !noalias { self._probe_ref(query)? } else { query };
    match self.celtab.get_mut(&root) {
      None => return Err(CellDerefErr::MissingRoot),
      Some(e) => {
        //assert_eq!(root, e.root());
        match &mut e.cel_ {
          &mut Cell_::Top(.., optr) => {
            assert_eq!(root, optr);
            panic!("bug");
          }
          &mut Cell_::Phy(ref state, .., ref mut pcel) => {
            assert_eq!(root, pcel.optr);
            assert_eq!(&e.ty, &pcel.ogty);
            assert_eq!(next_clk, state.borrow().clk);
            pcel.read_loc(root, prev_clk, &e.ty, loc);
            pcel.write_loc(root, next_clk, &e.ty, loc);
            return Ok(CellMutRef_{
              ref_: root,
              root_ty: &e.ty,
              ty,
              stablect: &e.stablect,
              cel_: &mut e.cel_,
            });
          }
          &mut Cell_::Cow(.., ref cel) => {
            assert_eq!(root, cel.optr);
            unimplemented!();
          }
          &mut Cell_::Alias(..) => {
            panic!("bug");
          }
          &mut Cell_::Bot => {
            return Err(CellDerefErr::Bot);
          }
          _ => unimplemented!()
        }
      }
    }
  }

  pub fn prewrite_view(&mut self, query: CellPtr, prev_clk: Clock, next_clk: Clock, loc: Locus) -> CellMutDerefView {
    let (ty, noalias) = self._pre_probe(query)?;
    let view = if !noalias { self._probe_view(query)? } else { query.into() };
    let root = view.root();
    match self.celtab.get_mut(&root) {
      None => return Err(CellDerefErr::MissingRoot),
      Some(e) => {
        //assert_eq!(root, e.root());
        match &mut e.cel_ {
          &mut Cell_::Top(.., optr) => {
            assert_eq!(root, optr);
            panic!("bug");
          }
          &mut Cell_::Phy(ref state, .., ref mut pcel) => {
            assert_eq!(root, pcel.optr);
            assert_eq!(&e.ty, &pcel.ogty);
            assert_eq!(next_clk, state.borrow().clk);
            pcel.read_loc(root, prev_clk, &e.ty, loc);
            pcel.write_loc(root, next_clk, &e.ty, loc);
            return Ok(CellMutRef_{
              ref_: view,
              root_ty: &e.ty,
              ty,
              stablect: &e.stablect,
              cel_: &mut e.cel_,
            });
          }
          &mut Cell_::Cow(.., ref cel) => {
            assert_eq!(root, cel.optr);
            unimplemented!();
          }
          &mut Cell_::Alias(..) => {
            panic!("bug");
          }
          &mut Cell_::Bot => {
            return Err(CellDerefErr::Bot);
          }
          _ => unimplemented!()
        }
      }
    }
  }

  /*pub fn probe(&self, x: CellPtr) -> CellPtr {
    let mut p = x;
    let mut root = self.alias_root.borrow_mut();
    loop {
      let p2;
      match root.get(&p) {
        None => {
          break;
        }
        Some(&q) => {
          p2 = q;
        }
      }
      match root.get(&p2) {
        None => {
          p = p2;
          break;
        }
        Some(&q) => {
          root.insert(p, q);
          p = q;
        }
      }
    }
    drop(root);
    p
  }

  pub fn _lookup_ref(&self, query: CellPtr) -> Option<CellEnvEntryRef> {
    let ty = match self.celtab.get(&query) {
      None => return None,
      Some(e) => {
        match &e.cel_ {
          &Cell_::Top(.., optr) => {
            assert_eq!(query, optr);
          }
          &Cell_::Phy(.., ref cel) => {
            assert_eq!(query, cel.optr);
          }
          &Cell_::Cow(.., ref cel) => {
            assert_eq!(query, cel.optr);
          }
          &Cell_::Alias(..) => {}
          &Cell_::Bot => {
            panic!("bug");
          }
          _ => unimplemented!()
        }
        match &e.cel_ {
          &Cell_::Top(..) |
          &Cell_::Phy(..) |
          &Cell_::Cow(..) => {
            return Some(CellEnvEntryRef{
              root: query,
              stablect: &e.stablect,
              //snapshot: &e.snapshot,
              ty: &e.ty,
              //eflag: e.eflag,
              cel_: &e.cel_,
            });
          }
          &Cell_::Alias(..) => {}
          _ => unreachable!()
        }
        &e.ty
      }
    };
    let root = self.probe(query);
    match self.celtab.get(&root) {
      None => return None,
      Some(e) => {
        match &e.cel_ {
          &Cell_::Top(.., optr) => {
            assert_eq!(root, optr);
          }
          &Cell_::Phy(.., ref cel) => {
            assert_eq!(root, cel.optr);
          }
          &Cell_::Cow(.., ref cel) => {
            assert_eq!(root, cel.optr);
          }
          &Cell_::Bot => {}
          &Cell_::Alias(..) => {
            panic!("bug");
          }
          _ => unimplemented!()
        }
        // FIXME: type compat.
        if e.ty.dtype != Dtype::_Top {
          assert_eq!(ty.dtype.size_bytes(), e.ty.dtype.size_bytes());
          assert!(ty.shape_compat(&e.ty) != ShapeCompat::Incompat);
        }
        Some(CellEnvEntryRef{
          root,
          stablect: &e.stablect,
          //snapshot: &e.snapshot,
          ty,
          //eflag: e.eflag,
          cel_: &e.cel_,
        })
      }
    }
  }

  pub fn lookup_ref(&self, query: CellPtr) -> Option<CellEnvEntryRef> {
    match self._lookup_ref(query) {
      None => panic!("bug: CtxEnv::lookup_ref: missing query={:?}", query),
      Some(e) => Some(e)
    }
  }

  pub fn lookup_mut_ref(&mut self, query: CellPtr) -> Option<CellEnvEntryMutRef> {
    let mut noalias = false;
    let ty = match self.celtab.get(&query) {
      None => panic!("bug: CtxEnv::lookup_mut_ref: missing query={:?}", query),
      Some(e) => {
        match &e.cel_ {
          &Cell_::Top(..) |
          &Cell_::Phy(..) |
          &Cell_::Cow(..) |
          &Cell_::Bot => {
            noalias = true;
          }
          &Cell_::Alias(..) => {}
          _ => unimplemented!()
        }
        e.ty.clone()
      }
    };
    let root = if !noalias { self.probe(query) } else { query };
    match self.celtab.get_mut(&root) {
      None => panic!("bug"),
      Some(e) => {
        match &e.cel_ {
          &Cell_::Top(.., optr) => {
            assert_eq!(root, optr);
          }
          &Cell_::Phy(.., ref cel) => {
            assert_eq!(root, cel.optr);
          }
          &Cell_::Cow(.., ref cel) => {
            assert_eq!(root, cel.optr);
          }
          &Cell_::Bot => {}
          &Cell_::Alias(..) => {
            panic!("bug");
          }
          _ => unimplemented!()
        }
        // FIXME: type compat.
        if e.ty.dtype != Dtype::_Top {
          assert_eq!(ty.dtype.size_bytes(), e.ty.dtype.size_bytes());
          assert!(ty.shape_compat(&e.ty) != ShapeCompat::Incompat);
        }
        return Some(CellEnvEntryMutRef{
          root,
          stablect: &e.stablect,
          //snapshot: &e.snapshot,
          ty,
          //eflag: &mut e.eflag,
          cel_: &mut e.cel_,
        });
      }
    }
  }

  /*pub fn pread_view(&mut self, x: CellPtr, xclk: Clock, loc: Locus) -> () {
    // TODO
    unimplemented!();
  }*/

  pub fn pread_ref(&mut self, x: CellPtr, xclk: Clock, loc: Locus) -> Option<CellEnvEntryMutRef> {
    let mut query = x;
    let mut ty = None;
    loop {
      let mut noalias = false;
      match self.celtab.get(&query) {
        None => panic!("bug"),
        Some(e) => {
          match &e.cel_ {
            &Cell_::Top(..) |
            &Cell_::Phy(..) |
            &Cell_::Cow(..) |
            &Cell_::Bot => {
              noalias = true;
            }
            &Cell_::Alias(..) => {}
            _ => unimplemented!()
          }
          if ty.is_none() {
            ty = Some(e.ty.clone());
          }
        }
      }
      let root = if !noalias { self.probe(query) } else { query };
      match self.celtab.get(&root) {
        None => panic!("bug"),
        Some(e) => {
          match &e.cel_ {
            &Cell_::Top(.., optr) => {
              assert_eq!(root, optr);
              println!("ERROR: runtime error: attempted to read an uninitialized cell");
              panic!();
            }
            &Cell_::Phy(ref state, .., ref cel) => {
              assert_eq!(root, cel.optr);
              assert_eq!(xclk, state.borrow().clk);
              /*let mut reps = Vec::new();
              let mut clks = Vec::new();
              for (&key, replica) in cel.replicas.iter() {
                let &(loc, pm) = key.as_ref();
                reps.push((loc, pm, replica.addr.get(), replica.clk.get()));
                clks.push(replica.clk.get());
              }
              //clks.sort();
              if clks.len() > 1 {
                let mut f = false;
                if xclk != e.state_ref().clk {
                  f = true;
                }
                if !f {
                  for &c in clks.iter() {
                    if c != xclk || c != e.state_ref().clk {
                      f = true;
                      break;
                    }
                  }
                }
                if f {
                  println!("DEBUG: CtxEnv::pread_ref: x={:?} xroot={:?} xclk={:?} e.clk={:?} replicas={:?}",
                      x, root, xclk, e.state_ref().clk, &reps);
                  panic!("BREAK");
                }
              }*/
              match self.celtab.get_mut(&root) {
                None => panic!("bug"),
                Some(e) => {
                  match &mut e.cel_ {
                    &mut Cell_::Phy(.., ref mut cel) => {
                      cel.get_loc(x, xclk, &e.ty, loc);
                    }
                    _ => unreachable!()
                  }
                  // FIXME: type compat.
                  return Some(CellEnvEntryMutRef{
                    root,
                    stablect: &e.stablect,
                    //snapshot: &e.snapshot,
                    ty: ty.unwrap(),
                    //eflag: &mut e.eflag,
                    cel_: &mut e.cel_,
                  });
                }
              }
            }
            &Cell_::Cow(.., ref cel) => {
              assert_eq!(root, cel.optr);
              let pcel = cel.pcel;
              match self.celtab.get(&pcel) {
                None => panic!("bug"),
                Some(e) => {
                  match &e.cel_ {
                    &Cell_::Phy(..) => {}
                    &Cell_::Cow(.., ref cow_cel) => {
                      query = cow_cel.pcel;
                      // FIXME FIXME
                      /*let cow_root = self.cow_root.borrow();
                      match cow_root.get(&pcel) {
                        None => panic!("bug"),
                        Some(&(root_cel, root_clk)) => {
                          match self.snapshot.get(&(root_cel, root_clk)) {
                            None => panic!("bug"),
                            Some(list) => {
                              assert!(list.len() > 0);
                              query = list[0];
                            }
                          }
                        }
                      }*/
                      continue;
                    }
                    _ => panic!("bug")
                  }
                }
              }
              match self.celtab.get_mut(&pcel) {
                None => panic!("bug"),
                Some(e) => {
                  // FIXME: type compat.
                  return Some(CellEnvEntryMutRef{
                    root,
                    stablect: &e.stablect,
                    //snapshot: &e.snapshot,
                    ty: ty.unwrap(),
                    //eflag: &mut e.eflag,
                    cel_: &mut e.cel_,
                  });
                }
              }
            }
            &Cell_::Bot => {
              panic!("bug");
            }
            &Cell_::Alias(..) => {
              panic!("bug");
            }
            _ => unimplemented!()
          }
        }
      }
    }
  }

  pub fn pfresh_ref(&mut self, x: CellPtr, next_xclk: Clock, loc: Locus) -> Option<CellEnvEntryMutRef> {
    let query = x;
    let mut noalias = false;
    let ty = match self.celtab.get(&query) {
      None => panic!("bug"),
      Some(e) => {
        match &e.cel_ {
          &Cell_::Top(..) |
          &Cell_::Phy(..) |
          &Cell_::Cow(..) |
          &Cell_::Bot => {
            noalias = true;
          }
          &Cell_::Alias(..) => {}
          _ => unimplemented!()
        }
        e.ty.clone()
      }
    };
    let root = if !noalias { self.probe(query) } else { query };
    match self.celtab.get(&root) {
      None => panic!("bug"),
      Some(e) => {
        match &e.cel_ {
          &Cell_::Top(ref state, optr) => {
            assert_eq!(root, optr);
            assert_eq!(next_xclk, state.borrow().clk);
            // FIXME: create a fresh PCell.
            let ty = e.ty.clone();
            if cfg_debug() { println!("DEBUG: CtxEnv::pfresh_ref: fresh pcel: ty={:?}", &ty); }
            let state = state.clone();
            match self.celtab.get_mut(&root) {
              None => panic!("bug"),
              Some(e) => {
                //state.borrow_mut().clk = xclk;
                let clo = RefCell::new(CellClosure::default());
                let pcel = PCell::new(optr, ty.clone());
                e.cel_ = Cell_::Phy(state, clo, pcel);
                return Some(CellEnvEntryMutRef{
                  root,
                  stablect: &e.stablect,
                  //snapshot: &e.snapshot,
                  ty,
                  //eflag: &mut e.eflag,
                  cel_: &mut e.cel_,
                });
              }
            }
          }
          &Cell_::Phy(ref state, _, ref cel) => {
            assert_eq!(root, cel.optr);
            assert_eq!(next_xclk, state.borrow().clk);
            match self.celtab.get_mut(&root) {
              None => panic!("bug"),
              Some(e) => {
                // FIXME: need to `get` here.
                match &mut e.cel_ {
                  &mut Cell_::Phy(.., ref mut cel) => {
                    cel.fresh_loc(x, next_xclk, &e.ty, loc);
                  }
                  _ => unreachable!()
                }
                // FIXME: type compat.
                return Some(CellEnvEntryMutRef{
                  root,
                  stablect: &e.stablect,
                  //snapshot: &e.snapshot,
                  ty,
                  //eflag: &mut e.eflag,
                  cel_: &mut e.cel_,
                });
              }
            }
          }
          &Cell_::Cow(ref state, _, ref cel) => {
            assert_eq!(root, cel.optr);
            assert_eq!(next_xclk, state.borrow().clk);
            unimplemented!();
          }
          &Cell_::Bot => {
            panic!("bug");
          }
          &Cell_::Alias(..) => {
            panic!("bug");
          }
          _ => unimplemented!()
        }
      }
    }
  }

  pub fn pwrite_ref(&mut self, x: CellPtr, prev_xclk: Clock, next_xclk: Clock, loc: Locus) -> Option<CellEnvEntryMutRef> {
    assert!(prev_xclk < next_xclk);
    let query = x;
    let mut noalias = false;
    let ty = match self.celtab.get(&query) {
      None => panic!("bug"),
      Some(e) => {
        match &e.cel_ {
          &Cell_::Top(..) |
          &Cell_::Phy(..) |
          &Cell_::Cow(..) |
          &Cell_::Bot => {
            noalias = true;
          }
          &Cell_::Alias(..) => {}
          _ => unimplemented!()
        }
        e.ty.clone()
      }
    };
    let root = if !noalias { self.probe(query) } else { query };
    match self.celtab.get(&root) {
      None => panic!("bug"),
      Some(e) => {
        match &e.cel_ {
          &Cell_::Top(ref state, optr) => {
            assert_eq!(root, optr);
            assert_eq!(next_xclk, state.borrow().clk);
            // FIXME: create a fresh PCell.
            let ty = e.ty.clone();
            if cfg_debug() { println!("DEBUG: CtxEnv::pwrite_ref: fresh pcel: ty={:?}", &ty); }
            let state = state.clone();
            match self.celtab.get_mut(&root) {
              None => panic!("bug"),
              Some(e) => {
                //state.borrow_mut().clk = xclk;
                let clo = RefCell::new(CellClosure::default());
                let pcel = PCell::new(optr, ty.clone());
                e.cel_ = Cell_::Phy(state, clo, pcel);
                return Some(CellEnvEntryMutRef{
                  root,
                  stablect: &e.stablect,
                  //snapshot: &e.snapshot,
                  ty,
                  //eflag: &mut e.eflag,
                  cel_: &mut e.cel_,
                });
              }
            }
          }
          &Cell_::Phy(ref state, _, ref cel) => {
            assert_eq!(root, cel.optr);
            assert_eq!(next_xclk, state.borrow().clk);
            match self.celtab.get_mut(&root) {
              None => panic!("bug"),
              Some(e) => {
                // FIXME: need to `get` here.
                match &mut e.cel_ {
                  &mut Cell_::Phy(.., ref mut cel) => {
                    cel.get_loc2(x, prev_xclk, next_xclk, &e.ty, loc);
                  }
                  _ => unreachable!()
                }
                // FIXME: type compat.
                return Some(CellEnvEntryMutRef{
                  root,
                  stablect: &e.stablect,
                  //snapshot: &e.snapshot,
                  ty,
                  //eflag: &mut e.eflag,
                  cel_: &mut e.cel_,
                });
              }
            }
          }
          &Cell_::Cow(ref state, _, ref cel) => {
            assert_eq!(root, cel.optr);
            assert_eq!(next_xclk, state.borrow().clk);
            let pcel = cel.pcel;
            match self.celtab.get(&pcel) {
              None => panic!("bug"),
              Some(pe) => {
                let (clo2, pcel2) = match &pe.cel_ {
                  &Cell_::Phy(.., ref clo, ref pcel) => {
                    (clo.clone(), pcel.hardcopy())
                  }
                  _ => panic!("bug")
                };
                let state2 = state.clone();
                match self.celtab.get_mut(&root) {
                  None => panic!("bug"),
                  Some(e) => {
                    e.cel_ = Cell_::Phy(state2, clo2, pcel2);
                    /*let mut cow_root = self.cow_root.borrow_mut();
                    match cow_root.remove(&root) {
                      None => panic!("bug"),
                      Some((root_cel, root_clk)) => {
                        // FIXME FIXME: after cow upgrades to phy,
                        // we can gc the snapshot.
                        /*match self.snapshot.get(&(root_cel, root_clk)) {
                          None => panic!("bug"),
                          Some(list) => {
                            assert!(list.len() > 0);
                            match (&list[1 .. ]).remove(&root) {
                              None => panic!("bug"),
                              Some(_) => {}
                            }
                            // FIXME
                          }
                        }*/
                      }
                    }*/
                    // FIXME: type compat.
                    return Some(CellEnvEntryMutRef{
                      root,
                      stablect: &e.stablect,
                      //snapshot: &e.snapshot,
                      ty,
                      //eflag: &mut e.eflag,
                      cel_: &mut e.cel_,
                    });
                  }
                }
              }
            }
          }
          &Cell_::Bot => {
            panic!("bug");
          }
          &Cell_::Alias(..) => {
            panic!("bug");
          }
          _ => unimplemented!()
        }
      }
    }
  }*/

  pub fn insert_top(&mut self, x: CellPtr, ty: CellType) {
    match self.celtab.get(&x) {
      None => {}
      Some(_) => panic!("bug")
    }
    let e = CellEnvEntry{
      stablect: Cell::new(0),
      //snapshot: Cell::new(0),
      ty,
      //eflag:    CellEFlag::default(),
      cel_:     Cell_::Top(RefCell::new(CellState::default()), x),
    };
    self.celtab.insert(x, e);
  }

  pub fn insert_phy(&mut self, x: CellPtr, ty: CellType, pcel: PCell) {
    match self.celtab.get(&x) {
      None => {}
      Some(_) => panic!("bug")
    }
    assert_eq!(x, pcel.optr);
    let e = CellEnvEntry{
      stablect: Cell::new(0),
      //snapshot: Cell::new(0),
      ty,
      //eflag:    CellEFlag::default(),
      cel_:     Cell_::Phy(
                    RefCell::new(CellState::default()),
                    RefCell::new(CellClosure::default()),
                    pcel,
                ),
    };
    self.celtab.insert(x, e);
  }

  pub fn insert_cow(&mut self, x: CellPtr, ty: CellType, pcel: CellPtr, pclk: Clock) {
    match self.celtab.get(&x) {
      None => {}
      Some(_) => panic!("bug")
    }
    let (state, clo) = match self._lookup_ref_(pcel) {
      Err(_) => panic!("bug"),
      Ok(e) => {
        match &e.cel_ {
          &Cell_::Top(ref state, ..) => {
            assert_eq!(state.borrow().clk, pclk);
            (state.clone(), RefCell::new(CellClosure::default()))
          }
          &Cell_::Phy(ref state, ref clo, ..) |
          &Cell_::Cow(ref state, ref clo, ..) => {
            assert_eq!(state.borrow().clk, pclk);
            (state.clone(), clo.clone())
          }
          _ => panic!("bug")
        }
      }
    };
    let e = CellEnvEntry{
      stablect: Cell::new(0),
      //snapshot: Cell::new(0),
      ty,
      //eflag:    CellEFlag::default(),
      cel_:     Cell_::Cow(
                    state,
                    clo,
                    CowCell{optr: x, pcel, pclk},
                ),
    };
    let mut root = self.cow_root.borrow_mut();
    match root.get(&x) {
      None => {}
      Some(_) => panic!("bug")
    }
    root.insert(x, (pcel, pclk));
    self.celtab.insert(x, e);
  }

  pub fn insert_alias(&mut self, x: CellPtr, alias: CellAlias, ty: CellType, og: CellPtr) {
    match self.celtab.get(&x) {
      None => {}
      Some(_) => panic!("bug")
    }
    let e = CellEnvEntry{
      stablect: Cell::new(u32::max_value()),
      //snapshot: Cell::new(u32::max_value()),
      ty,
      //eflag:    CellEFlag::default(),
      cel_:     Cell_::Alias(alias, og),
    };
    self.celtab.insert(x, e);
    /*let mut root = self.alias_root.borrow_mut();
    match root.get(&x) {
      None => {}
      Some(_) => panic!("bug")
    }
    root.insert(x, og);*/
  }

  pub fn insert_const_(&mut self, x: CellPtr, og: CellPtr) {
    match self.celtab.get(&x) {
      None => {}
      Some(_) => panic!("bug")
    }
    let ty = match self._lookup_ref_(og) {
      Err(_) => panic!("bug"),
      Ok(oe) => oe.ty.clone()
    };
    let e = CellEnvEntry{
      stablect: Cell::new(u32::max_value()),
      ty,
      //eflag:    CellEFlag::default(),
      cel_:     Cell_::Alias(CellAlias::Const_, og),
    };
    self.celtab.insert(x, e);
    /*let mut root = self.alias_root.borrow_mut();
    match root.get(&x) {
      None => {}
      Some(_) => panic!("bug")
    }
    root.insert(x, og);*/
  }

  pub fn retain(&self, x: CellPtr) {
    if x.is_nil() {
      return;
    }
    match self._lookup_ref_(x) {
      Err(_) => panic!("bug"),
      Ok(e) => {
        let cur = e.stablect.get();
        assert!(cur != u32::max_value());
        let next = cur + 1;
        e.stablect.set(next);
      }
    }
  }

  pub fn release(&self, x: CellPtr) {
    if x.is_nil() {
      return;
    }
    match self._lookup_ref_(x) {
      Err(_) => panic!("bug"),
      Ok(e) => {
        let cur = e.stablect.get();
        assert!(cur != u32::max_value());
        let next = cur - 1;
        e.stablect.set(next);
      }
    }
  }

  pub fn snapshot(&mut self, ctr: &CtxCtr, pcel: CellPtr) -> CellPtr {
    let (ty, pclk) = match self._lookup_ref_(pcel) {
      Err(_) => panic!("bug"),
      Ok(e) => {
        (e.ty.clone(), match &e.cel_ {
          &Cell_::Top(..) => {
            Clock::default()
          }
          &Cell_::Phy(ref state, ..) |
          &Cell_::Cow(ref state, ..) => {
            state.borrow().clk
          }
          _ => panic!("bug")
        })
      }
    };
    let x0 = match self.snapshot.get(&(pcel, pclk)) {
      None => {
        let x0 = ctr.fresh_cel();
        self.insert_cow(x0, ty.clone(), pcel, pclk);
        match self.snapshot.get_mut(&(pcel, pclk)) {
          None => {
            let mut list = Vec::new();
            list.push(x0);
            self.snapshot.insert((pcel, pclk), list);
          }
          Some(_) => panic!("bug")
        }
        x0
      }
      Some(list) => {
        assert!(list.len() > 0);
        list[0]
      }
    };
    let x = ctr.fresh_cel();
    self.insert_cow(x, ty, x0, pclk);
    match self.snapshot.get_mut(&(pcel, pclk)) {
      None => panic!("bug"),
      Some(list) => {
        assert!(list.len() > 0);
        list.push(x);
      }
    }
    x
  }

  /*pub fn gc_prepare(&self, gc_list: &mut Vec<CellPtr>) {
    // FIXME FIXME: temporarily commented out for debugging.
    /*for (&k, _) in self.celtab.iter() {
      match self.lookup(k) {
        None => panic!("bug"),
        Some(e) => {
          if e.stablect.get() <= 0 {
            gc_list.push(k);
          }
        }
      }
    }*/
  }

  pub fn gc(&mut self, gc_list: &[CellPtr]) {
    let mut root = self.alias_root.borrow_mut();
    for &k in gc_list.iter() {
      // FIXME FIXME: remove from other indexes.
      root.remove(&k);
      self.celtab.remove(&k);
    }
  }*/
}
