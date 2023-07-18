use crate::algo::{HashMap, HashSet};
use crate::cell::*;
use crate::clock::*;
use crate::panick::{panick_wrap};
use crate::pctx::{Locus, MemReg};
use crate::spine::*;
use crate::thunk::*;
use crate::thunk::op::{SetScalarFutThunkSpec};
use crate::util::stat::*;

use futhark_syntax::{Token as FutToken};
use futhark_syntax::re::{ReTrie};

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
    println!("DEBUG: CtxCfg::default");
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
  pub env:      RefCell<CtxEnv>,
  pub thunkenv: RefCell<CtxThunkEnv>,
  pub spine:    RefCell<Spine>,
  pub futhark:  RefCell<FutharkCtx>,
  pub timing:   TimingCtx,
}

impl Drop for Ctx {
  fn drop(&mut self) {
    let digest = self.timing.digest();
    println!("DEBUG: Ctx::drop: timing digest: pregemm1: {:?}", digest.pregemm1);
    println!("DEBUG: Ctx::drop: timing digest: pregemm:  {:?}", digest.pregemm);
    println!("DEBUG: Ctx::drop: timing digest: gemm1:    {:?}", digest.gemm1);
    println!("DEBUG: Ctx::drop: timing digest: gemm:     {:?}", digest.gemm);
    println!("DEBUG: Ctx::drop: timing digest: futhark1: {:?}", digest.futhark1);
    println!("DEBUG: Ctx::drop: timing digest: futhark:  {:?}", digest.futhark);
  }
}

impl Ctx {
  pub fn new() -> Ctx {
    println!("DEBUG: Ctx::new");
    Ctx{
      ctr:      CtxCtr::new(),
      env:      RefCell::new(CtxEnv::default()),
      thunkenv: RefCell::new(CtxThunkEnv::default()),
      spine:    RefCell::new(Spine::default()),
      futhark:  RefCell::new(FutharkCtx::default()),
      timing:   TimingCtx::default(),
    }
  }
}

#[derive(Default)]
pub struct FutharkCtx {
  pub trie: Option<Rc<ReTrie<FutToken>>>,
}

#[derive(Default)]
pub struct TimingCtx {
  pub pregemm1: RefCell<Vec<f64>>,
  pub pregemm:  RefCell<Vec<f64>>,
  pub gemm1:    RefCell<Vec<f64>>,
  pub gemm:     RefCell<Vec<f64>>,
  pub futhark1: RefCell<Vec<f64>>,
  pub futhark:  RefCell<Vec<f64>>,
}

#[derive(Debug)]
pub struct TimingDigest {
  pub pregemm1: StatDigest,
  pub pregemm:  StatDigest,
  pub gemm1:    StatDigest,
  pub gemm:     StatDigest,
  pub futhark1: StatDigest,
  pub futhark:  StatDigest,
}

impl TimingCtx {
  pub fn digest(&self) -> TimingDigest {
    TimingDigest{
      pregemm1: StatDigest::from(&*self.pregemm1.borrow()),
      pregemm:  StatDigest::from(&*self.pregemm.borrow()),
      gemm1:    StatDigest::from(&*self.gemm1.borrow()),
      gemm:     StatDigest::from(&*self.gemm.borrow()),
      futhark1: StatDigest::from(&*self.futhark1.borrow()),
      futhark:  StatDigest::from(&*self.futhark.borrow()),
    }
  }
}

#[track_caller]
pub fn reset() {
  panick_wrap(|| TL_CTX.with(|ctx| {
    // FIXME FIXME: reset all.
    //ctx.env.borrow_mut().reset();
    //ctx.thunkenv.borrow_mut().reset();
    ctx.spine.borrow_mut()._reset();
  }))
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

#[track_caller]
pub fn resume_put_mem_val<K: Borrow<CellPtr>>(key: K, val: &dyn Any) -> SpineRet {
  panick_wrap(|| TL_CTX.with(|ctx| {
    let mut env = ctx.env.borrow_mut();
    let mut thunkenv = ctx.thunkenv.borrow_mut();
    let mut spine = ctx.spine.borrow_mut();
    spine._resume(&ctx.ctr, &mut *env, &mut *thunkenv, /*CellPtr::nil(), Clock::default(),*/ SpineResume::PutMemV(*key.borrow(), val))
  }))
}

#[track_caller]
pub fn resume_put_mem_fun<K: Borrow<CellPtr>, F: Fn(CellType, MemReg)>(key: K, fun: F) -> SpineRet {
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

pub fn ctx_unwrap<F: FnMut(&Ctx) -> X, X>(f: &mut F) -> X {
  TL_CTX.with(f)
}

/*#[cfg(not(feature = "gpu"))]
pub fn ctx_init_gpu(_dev: i32) {
}

#[cfg(feature = "gpu")]
pub fn ctx_init_gpu(dev: i32) {
  TL_CTX.with(|ctx| {
    // FIXME FIXME
    ctx.gpu.dev.set(dev);
  })
}*/

/*pub fn ctx_fresh() -> CellPtr {
  TL_CTX.with(|ctx| {
    ctx.ctr.fresh_cel()
  })
}*/

/*pub fn ctx_tmp_fresh() -> CellPtr {
  TL_CTX.with(|ctx| {
    ctx.ctr.fresh_tmp()
  })
}

pub fn ctx_reset_tmp() {
  TL_CTX.with(|ctx| {
    ctx.env.borrow_mut().reset_tmp();
    ctx.ctr.reset_tmp();
  })
}

pub fn ctx_fresh_tmp() -> CellPtr {
  TL_CTX.with(|ctx| {
    ctx.ctr.fresh_tmp()
  })
}

pub fn ctx_peek_tmp() -> CellPtr {
  TL_CTX.with(|ctx| {
    ctx.ctr.peek_tmp()
  })
}

pub fn ctx_unify(x: CellPtr, uty: Option<CellType>) -> CellPtr {
  TL_CTX.with(|ctx| {
    ctx.env.borrow_mut().unify(&ctx.ctr, x, uty)
  })
}*/

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
    match ctx.env.borrow().lookup_ref(x) {
      None => panic!("bug"),
      Some(e) => e.ty.clone()
    }
  })
}

pub fn ctx_lookup_dtype(x: CellPtr) -> Dtype {
  TL_CTX.with(|ctx| {
    match ctx.env.borrow().lookup_ref(x) {
      None => panic!("bug"),
      Some(e) => e.ty.dtype
    }
  })
}

pub fn ctx_lookup_mode(x: CellPtr) -> CellMode {
  TL_CTX.with(|ctx| {
    match ctx.env.borrow().lookup_ref(x) {
      None => panic!("bug"),
      Some(e) => e.state().mode
    }
  })
}

pub fn ctx_lookup_flag(x: CellPtr) -> CellFlag {
  TL_CTX.with(|ctx| {
    match ctx.env.borrow().lookup_ref(x) {
      None => panic!("bug"),
      Some(e) => {
        e.state().flag
        //*e.cel.flag.borrow()
      }
    }
  })
}

pub fn ctx_lookup_eflag(x: CellPtr) -> CellEFlag {
  TL_CTX.with(|ctx| {
    match ctx.env.borrow().lookup_ref(x) {
      None => panic!("bug"),
      Some(e) => {
        e.eflag
      }
    }
  })
}

pub fn ctx_lookup_clk(x: CellPtr) -> Clock {
  TL_CTX.with(|ctx| {
    match ctx.env.borrow().lookup_ref(x) {
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

pub fn ctx_alias(og: CellPtr, new_ty: CellType) -> CellPtr {
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
}

pub fn ctx_alias_bits(og: CellPtr, new_dtype: Dtype) -> CellPtr {
  TL_CTX.with(|ctx| {
    let mut env = ctx.env.borrow_mut();
    match env.lookup_ref(og) {
      None => panic!("bug"),
      Some(e) => {
        if new_dtype.size_bytes() != e.ty.dtype.size_bytes() {
          println!("ERROR: ctx_alias_bits: og={:?} old dtype={:?} new dtype={:?}", og, e.ty.dtype, new_dtype);
          panic!();
        }
        let new_ty = CellType{dtype: new_dtype, shape: e.ty.shape.clone()};
        let x = ctx.ctr.fresh_cel();
        //println!("DEBUG: ctx_alias_bits: og={:?} old dtype={:?} x={:?} new dtype={:?}", og, e.ty.dtype, x, new_dtype);
        env.insert_alias(x, new_ty, og);
        let spine = ctx.spine.borrow();
        spine.alias(x, og);
        x
      }
    }
  })
}

pub fn ctx_alias_new_shape(og: CellPtr, new_shape: Vec<i64>) -> CellPtr {
  TL_CTX.with(|ctx| {
    let mut env = ctx.env.borrow_mut();
    match env.lookup_ref(og) {
      None => panic!("bug"),
      Some(e) => {
        let new_ty = CellType{shape: new_shape, dtype: e.ty.dtype};
        let cmp = new_ty.shape_compat(&e.ty);
        if !(cmp == ShapeCompat::Equal || cmp == ShapeCompat::NewShape) {
          println!("ERROR: ctx_alias_new_shape: shape mismatch: og={:?} old shape={:?} new shape={:?} compat={:?}", og, &e.ty.shape, &new_ty.shape, cmp);
          panic!();
        }
        let x = ctx.ctr.fresh_cel();
        //println!("DEBUG: ctx_alias_new_shape: og={:?} old shape={:?} x={:?} new shape={:?} compat={:?}", og, &e.ty.shape, x, &new_ty.shape, cmp);
        env.insert_alias(x, new_ty, og);
        let spine = ctx.spine.borrow();
        spine.alias(x, og);
        x
      }
    }
  })
}

pub fn ctx_snapshot(og: CellPtr) -> CellPtr {
  TL_CTX.with(|ctx| {
    let mut env = ctx.env.borrow_mut();
    let x = env.snapshot(&ctx.ctr, og);
    let spine = ctx.spine.borrow();
    spine.snapshot(x, og);
    x
  })
}

/*pub fn ctx_lookup_or_insert_gradl(x: CellPtr, tg: CellPtr) -> CellPtr {
  TL_CTX.with(|ctx| {
    ctx.env.borrow_mut().lookup_or_insert_gradr(&ctx.ctr, tg, x)
  })
}

pub fn ctx_lookup_or_insert_gradr(tg: CellPtr, x: CellPtr) -> CellPtr {
  TL_CTX.with(|ctx| {
    ctx.env.borrow_mut().lookup_or_insert_gradr(&ctx.ctr, tg, x)
  })
}

pub fn ctx_accumulate_gradr(tg: CellPtr, x: CellPtr, dx: CellPtr) {
  TL_CTX.with(|ctx| {
    // FIXME FIXME
    unimplemented!();
    //ctx.env.borrow_mut().accumulate_gradr(&ctx.ctr, tg, x, dx)
  })
}*/

/*pub fn ctx_trace_val(og: CellPtr) -> CellPtr {
  TL_CTX.with(|ctx| {
    // FIXME FIXME
    let mut env = ctx.env.borrow_mut();
    match env.lookup_ref(og) {
      None => panic!("bug"),
      Some(e) => {
        let ty = e.ty.clone();
        let x = ctx.ctr.fresh_cel();
        env.insert_alias(x, ty, og);
        // FIXME
        let mut spine = ctx.spine.borrow_mut();
        let sp = spine.curp;
        spine.curp += 1;
        spine.log.push(SpineEntry::TraceV(x, og));
        spine.env.cache.insert(x, sp);
        spine.env.aff.insert(x, sp);
        x
      }
    }
  })
}

pub fn ctx_profile_val(og: CellPtr) -> CellPtr {
  TL_CTX.with(|ctx| {
    // FIXME FIXME
    let mut env = ctx.env.borrow_mut();
    match env.lookup_ref(og) {
      None => panic!("bug"),
      Some(e) => {
        let ty = e.ty.clone();
        let x = ctx.ctr.fresh_cel();
        env.insert_alias(x, ty, og);
        // FIXME
        let mut spine = ctx.spine.borrow_mut();
        let sp = spine.curp;
        spine.curp += 1;
        spine.log.push(SpineEntry::Profile(x, og));
        spine.env.cache.insert(x, sp);
        spine.env.aff.insert(x, sp);
        x
      }
    }
  })
}*/

pub fn ctx_opaque(og: CellPtr) -> CellPtr {
  TL_CTX.with(|ctx| {
    // FIXME FIXME
    let mut env = ctx.env.borrow_mut();
    match env.lookup_ref(og) {
      None => panic!("bug"),
      Some(e) => {
        let ty = e.ty.clone();
        let x = ctx.ctr.fresh_cel();
        env.insert_alias(x, ty, og);
        // FIXME
        let spine = ctx.spine.borrow();
        spine.opaque(x, og);
        x
      }
    }
  })
}

/*pub fn ctx_init_zeros(ty: CellType) -> CellPtr {
  match ty.dtype {
    Dtype::Fp32 => {
      let value = 0.0_f32;
      //let x = ctx_fresh();
      match ty.ndim() {
        0 => {
          ctx_pop_init_thunk_(SetScalarFutThunkSpec{val: value.into_scalar_val_()}, ty)
        }
        1 => {
          unimplemented!();
          //ctx_pop_init_thunk_(SetScalar1dFutThunkSpec{val: value.into_scalar_val()}, ty)
        }
        _ => unimplemented!()
      }
    }
    _ => unimplemented!()
  }
}

pub fn ctx_set_ones(ty: CellType) -> CellPtr {
  match ty.dtype {
    Dtype::Fp32 => {
      let value = 1.0_f32;
      match ty.ndim() {
        0 => {
          ctx_pop_thunk_(SetScalarFutThunkSpec{val: value.into_scalar_val_()}, ty)
        }
        1 => {
          unimplemented!();
          //ctx_pop_thunk_(SetScalar1dFutThunkSpec{val: value.into_scalar_val()}, ty)
        }
        _ => unimplemented!()
      }
    }
    _ => unimplemented!()
  }
}*/

pub fn ctx_clean_arg() -> bool {
  TL_CTX.with(|ctx| {
    ctx.thunkenv.borrow().arg.is_empty()/* &&
    ctx.out.borrow().is_empty()*/
  })
}

pub fn ctx_push_cell_arg(x: CellPtr) {
  TL_CTX.with(|ctx| {
    match ctx.env.borrow().lookup_ref(x) {
      None => panic!("bug"),
      Some(e) => {
        let spine = ctx.spine.borrow();
        let xclk = match spine._version(x) {
          None => {
            println!("DEBUG: ctx_push_cell_arg: no spine version: x={:?}", x);
            let cur_env = spine.cur_env.borrow();
            let xroot = cur_env._deref(x);
            println!("DEBUG: ctx_push_cell_arg:   xroot={:?} state={:?}",
                xroot, cur_env.state.get(&xroot));
            let query = CellPtr::from_unchecked(2401);
            /*println!("DEBUG: ctx_push_cell_arg:   query={:?} state={:?} alias={:?}",
                query, cur_env.state.get(&query), cur_env.alias.get(&query));*/
            println!("DEBUG: ctx_push_cell_arg:   query={:?} state={:?}",
                query, cur_env.state.get(&query));
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

pub fn ctx_pop_thunk<Th: ThunkSpec_ + 'static>(th: Th) -> CellPtr {
  TL_CTX.with(|ctx| {
    let mut dims = Vec::with_capacity(ctx.thunkenv.borrow().arg.len());
    let mut tys_ = Vec::with_capacity(ctx.thunkenv.borrow().arg.len());
    for &(arg, _) in ctx.thunkenv.borrow().arg.iter() {
      let ty_ = match ctx.env.borrow().lookup_ref(arg) {
        None => panic!("bug"),
        Some(e) => e.ty.clone()
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
      let ty_ = match ctx.env.borrow().lookup_ref(arg) {
        None => panic!("bug"),
        Some(e) => e.ty.clone()
      };
      dims.push(ty_.to_dim());
      let spine = ctx.spine.borrow();
      spine.push_seal(arg);
    }
    // FIXME FIXME: multiple arity out.
    let odim = out_ty.to_dim();
    let oty_ = out_ty;
    dims.push(odim);
    let (ar_in, ar_out) = th.arity();
    // FIXME FIXME
    //let tp = ctx.thunkenv.borrow_mut().insert_(ctx, ar_in, ar_out, dims, th);
    let tp = ctx.thunkenv.borrow_mut().lookup_or_insert(&ctx.ctr, ar_in, ar_out, dims, th);
    let y = ctx.ctr.fresh_cel();
    ctx.env.borrow_mut().insert_top(y, oty_);
    let spine = ctx.spine.borrow();
    spine.intro_aff(y);
    spine.apply(y, tp);
    let yclk = spine._version(y).unwrap();
    let mut arg = Vec::new();
    swap(&mut arg, &mut ctx.thunkenv.borrow_mut().arg);
    ctx.thunkenv.borrow_mut().update(y, yclk, tp, arg);
    y
  })
}

pub fn ctx_pop_init_thunk<Th: ThunkSpec_ + 'static>(th: Th, /*out: CellPtr*/) -> CellPtr {
  TL_CTX.with(|ctx| {
    let mut dims = Vec::with_capacity(ctx.thunkenv.borrow().arg.len());
    let mut tys_ = Vec::with_capacity(ctx.thunkenv.borrow().arg.len());
    for &(arg, _) in ctx.thunkenv.borrow().arg.iter() {
      let ty_ = match ctx.env.borrow().lookup_ref(arg) {
        None => panic!("bug"),
        Some(e) => e.ty.clone()
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
    ctx_pop_init_thunk_(th, oty_)
  })
}

pub fn ctx_pop_init_thunk_<Th: ThunkSpec_ + 'static>(th: Th, out_ty: CellType, /*out: CellPtr*/) -> CellPtr {
  TL_CTX.with(|ctx| {
    let mut dims = Vec::with_capacity(ctx.thunkenv.borrow().arg.len());
    for &(arg, _) in ctx.thunkenv.borrow().arg.iter() {
      let ty_ = match ctx.env.borrow().lookup_ref(arg) {
        None => panic!("bug"),
        Some(e) => e.ty.clone()
      };
      dims.push(ty_.to_dim());
      let spine = ctx.spine.borrow();
      spine.push_seal(arg);
    }
    let odim = out_ty.to_dim();
    let oty_ = out_ty;
    dims.push(odim);
    let (ar_in, ar_out) = th.arity();
    assert_eq!(ar_out, 1);
    //let tp = ctx.thunkenv.borrow_mut().insert_(ctx, ar_in, ar_out, dims, th);
    let tp = ctx.thunkenv.borrow_mut().lookup_or_insert(&ctx.ctr, ar_in, ar_out, dims, th);
    let y = ctx.ctr.fresh_cel();
    ctx.env.borrow_mut().insert_top(y, oty_);
    let spine = ctx.spine.borrow();
    spine.uninit(y);
    spine.initialize(y, tp);
    let yclk = spine._version(y).unwrap();
    let mut arg = Vec::new();
    swap(&mut arg, &mut ctx.thunkenv.borrow_mut().arg);
    ctx.thunkenv.borrow_mut().update(y, yclk, tp, arg);
    y
  })
}

pub fn ctx_pop_accumulate_thunk<Th: ThunkSpec_ + 'static>(th: Th, out: CellPtr) {
  TL_CTX.with(|ctx| {
    let mut dims = Vec::with_capacity(ctx.thunkenv.borrow().arg.len());
    let mut tys_ = Vec::with_capacity(ctx.thunkenv.borrow().arg.len());
    for &(arg, _) in ctx.thunkenv.borrow().arg.iter() {
      let ty_ = match ctx.env.borrow().lookup_ref(arg) {
        None => panic!("bug"),
        Some(e) => e.ty.clone()
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
    let (ar_in, ar_out) = th.arity();
    assert_eq!(ar_out, 1);
    //let tp = ctx.thunkenv.borrow_mut().insert_(ctx, ar_in, ar_out, dims, th);
    let tp = ctx.thunkenv.borrow_mut().lookup_or_insert(&ctx.ctr, ar_in, ar_out, dims, th);
    let y = out;
    match ctx.env.borrow().lookup_ref(y) {
      None => panic!("bug"),
      Some(e) => {
        assert_eq!(e.ty, &oty_);
      }
    }
    let spine = ctx.spine.borrow();
    spine.accumulate(y, tp);
    let yclk = spine._version(y).unwrap();
    let mut arg = Vec::new();
    swap(&mut arg, &mut ctx.thunkenv.borrow_mut().arg);
    ctx.thunkenv.borrow_mut().update(y, yclk, tp, arg);
  })
}

/*pub fn ctx_bar() {
  unimplemented!();
}*/

pub fn ctx_gc() {
  TL_CTX.with(|ctx| {
    // FIXME FIXME
    let mut gc_list = Vec::new();
    ctx.env.borrow().gc_prepare(&mut gc_list);
    ctx.env.borrow_mut().gc(&gc_list);
    //ctx.thunkenv.borrow_mut().gc(&gc_list);
  })
}

pub struct CtxCtr {
  pub ptr_ctr:  Cell<i32>,
  //pub tmp_ctr:  Cell<i32>,
}

impl CtxCtr {
  pub fn new() -> CtxCtr {
    CtxCtr{
      ptr_ctr:  Cell::new(0),
      //tmp_ctr:  Cell::new(0),
    }
  }
}

impl CtxCtr {
  pub fn fresh_cel(&self) -> CellPtr {
    let next = self._fresh();
    CellPtr::from_unchecked(next)
  }

  pub fn fresh_mcel(&self) -> MCellPtr {
    let next = self._fresh();
    MCellPtr::from_unchecked(next)
  }

  pub fn fresh_thunk(&self) -> ThunkPtr {
    let next = self._fresh();
    ThunkPtr::from_unchecked(next)
  }

  pub fn _fresh(&self) -> i32 {
    let next = self.ptr_ctr.get() + 1;
    assert!(next > 0);
    assert!(next < i32::max_value());
    self.ptr_ctr.set(next);
    next
  }

  /*pub fn reset_tmp(&self) {
    self.tmp_ctr.set(0);
  }

  pub fn fresh_tmp(&self) -> CellPtr {
    let next = self.tmp_ctr.get() - 1;
    assert!(next < 0);
    assert!(next > i32::min_value());
    self.tmp_ctr.set(next);
    CellPtr::from_unchecked(next)
  }

  pub fn peek_tmp(&self) -> CellPtr {
    CellPtr::from_unchecked(self.tmp_ctr.get())
  }*/
}

pub struct ThunkEnvEntry {
  pub pthunk:   Rc<PThunk>,
}

#[derive(Debug)]
pub struct ThunkClosure {
  pub pthunk:   ThunkPtr,
  pub arg:      Vec<(CellPtr, Clock)>,
}

#[derive(Default)]
pub struct CtxThunkEnv {
  pub thunktab: HashMap<ThunkPtr, ThunkEnvEntry>,
  pub thunkidx: HashMap<(u16, u16, Vec<Dim>, ThunkKey, ), ThunkPtr>,
  pub update:   HashMap<(CellPtr, Clock), ThunkClosure>,
  pub arg:      Vec<(CellPtr, Clock)>,
  pub accumulate_in_place: Cell<bool>,
  pub assume_uninit_zero: Cell<bool>,
}

impl CtxThunkEnv {
  pub fn reset(&mut self) {
    // FIXME FIXME
    //self.update.clear();
  }

  pub fn _set_accumulate_in_place(&self, flag: bool) {
    self.accumulate_in_place.set(flag);
  }

  pub fn _set_assume_uninit_zero(&self, flag: bool) {
    self.assume_uninit_zero.set(flag);
  }

  pub fn update(&mut self, y: CellPtr, yclk: Clock, tp: ThunkPtr, arg: Vec<(CellPtr, Clock)>) {
    match self.thunktab.get(&tp) {
      None => panic!("bug"),
      Some(te) => {
        // FIXME: where to typecheck?
        assert_eq!(arg.len(), te.pthunk.arityin as usize);
        assert_eq!(1, te.pthunk.arityout);
        let tclo = ThunkClosure{pthunk: tp, arg};
        self.update.insert((y, yclk), tclo);
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
    let (ar_in, ar_out, spec_dim, tk, ) = key;
    if tp_.is_none() {
      let tp = ctr.fresh_thunk();
      /*let mut arg = Vec::new();
      swap(&mut arg, &mut *ctx.arg.borrow_mut());
      assert_eq!(arg.len(), ar_in as usize);*/
      let pthunk = Rc::new(PThunk::new(tp, spec_dim.clone(), (tk.0).clone()));
      let te = ThunkEnvEntry{/*arg,*/ pthunk};
      self.thunkidx.insert((ar_in, ar_out, spec_dim, tk, ), tp);
      self.thunktab.insert(tp, te);
      tp_ = Some(tp);
    }
    tp_.unwrap()
  }

  pub fn gc(&mut self, gc_list: &[CellPtr]) {
    // FIXME FIXME: remove updating thunks.
  }
}

/*#[derive(Clone, Debug)]
pub struct CellClosure {
  pub ithunk:   Option<(Counter, ThunkPtr)>,
  pub thunk_:   Vec<(Counter, ThunkPtr)>,
}

impl Default for CellClosure {
  fn default() -> CellClosure {
    CellClosure{
      ithunk:   None,
      thunk_:   Vec::new(),
    }
  }
}*/

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
    /*} else if self.ctr < clk.ctr() {
      self.ctr = clk.ctr();
      self.thunk.clear();
      self.thunk.push(ThunkPtr::nil());*/
    } else if self.ctr < clk.ctr() {
      panic!("bug");
    }
    if clk.up as usize != self.thunk.len() {
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

#[derive(Clone, Copy, Debug)]
pub enum CellName {
  Top,
  Phy,
  Cow,
  Alias,
  VAlias,
  Bot,
}

pub enum Cell_ {
  Top(RefCell<CellState>, CellPtr),
  Phy(RefCell<CellState>, RefCell<CellClosure>, PCell),
  Cow(RefCell<CellState>, RefCell<CellClosure>, CowCell),
  Alias(CellPtr),
  VAlias(CellVOp, CellPtr),
  Bot,
}

impl Cell_ {
  pub fn name(&self) -> CellName {
    match self {
      &Cell_::Top(..) => CellName::Top,
      &Cell_::Phy(..) => CellName::Phy,
      &Cell_::Cow(..) => CellName::Cow,
      &Cell_::Alias(..) => CellName::Alias,
      &Cell_::VAlias(..) => CellName::VAlias,
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

#[derive(Clone, Copy, Default)]
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
}

/*#[derive(Clone, Copy, Debug)]
#[repr(u8)]
pub enum CellEMode {
  Read,
  Rwlock,
  Mutex,
}*/

pub struct CellEnvEntry {
  // FIXME
  pub stablect: Cell<u32>,
  //pub snapshot: Cell<u32>,
  pub ty:       CellType,
  pub eflag:    CellEFlag,
  pub cel_:     Cell_,
}

pub struct CellEnvEntryRef<'a> {
  pub root:     CellPtr,
  pub stablect: &'a Cell<u32>,
  //pub snapshot: &'a Cell<u32>,
  pub ty:       &'a CellType,
  pub eflag:    CellEFlag,
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
  pub stablect: &'a Cell<u32>,
  //pub snapshot: &'a Cell<u32>,
  pub ty:       CellType,
  pub eflag:    &'a mut CellEFlag,
  pub cel_:     &'a mut Cell_,
}

impl<'a> CellEnvEntryMutRef<'a> {
  pub fn state(&self) -> RefMut<CellState> {
    self.cel_.state_mut()
  }
}

pub struct MCellEnvEntry {
  // FIXME
  pub mcel_:    MCell_,
}

//#[derive(Default)]
pub struct CtxEnv {
  // FIXME
  //pub tmptab:   HashMap<CellPtr, CellPtr>,
  pub alias_root:   RefCell<HashMap<CellPtr, CellPtr>>,
  pub cow_root:     RefCell<HashMap<CellPtr, (CellPtr, Clock)>>,
  //pub stable:   HashSet<CellPtr>,
  pub celtab:   HashMap<CellPtr, CellEnvEntry>,
  pub snapshot: HashMap<(CellPtr, Clock), Vec<CellPtr>>,
  /*pub atomtab:  HashMap<Atom, ()>,*/
  pub mceltab:  HashMap<MCellPtr, MCellEnvEntry>,
  //pub gradr:    HashMap<[CellPtr; 2], CellPtr>,
  //pub ungradr:  HashMap<[CellPtr; 2], CellPtr>,
  //pub bwd:      HashMap<CellPtr, Clock>,
  //pub gradr:    HashMap<(CellPtr, CellPtr, Clock), CellPtr>,
  pub tag:      HashMap<CellPtr, HashSet<String>>,
  //pub tag:      HashMap<CellPtr, Vec<String>>,
}

impl Default for CtxEnv {
  fn default() -> CtxEnv {
    CtxEnv{
      //tmptab:   HashMap::new(),
      alias_root:   RefCell::new(HashMap::new()),
      cow_root:     RefCell::new(HashMap::new()),
      celtab:   HashMap::new(),
      snapshot: HashMap::new(),
      /*atomtab:  HashMap::new(),*/
      mceltab:  HashMap::new(),
      //gradr:    HashMap::new(),
      //ungradr:  HashMap::new(),
      //bwd:      HashMap::new(),
      tag:      HashMap::new(),
    }
  }
}

impl CtxEnv {
  pub fn reset(&mut self) {
    // FIXME FIXME
    /*self.tmptab.clear();*/
    self.celtab.clear();
    self.alias_root.borrow_mut().clear();
    //self.gradr.clear();
    //self.ungradr.clear();
    //self.bwd.clear();
    self.tag.clear();
  }

  pub fn probe(&self, x: CellPtr) -> CellPtr {
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

  pub fn lookup_ref(&self, x: CellPtr) -> Option<CellEnvEntryRef> {
    let query = x;
    let ty = match self.celtab.get(&query) {
      None => panic!("bug: CtxEnv::lookup_ref: missing query={:?}", query),
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
              eflag: e.eflag,
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
        Some(CellEnvEntryRef{
          root,
          stablect: &e.stablect,
          //snapshot: &e.snapshot,
          ty,
          eflag: e.eflag,
          cel_: &e.cel_,
        })
      }
    }
  }

  pub fn lookup_mut_ref(&mut self, x: CellPtr) -> Option<CellEnvEntryMutRef> {
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
          stablect: &e.stablect,
          //snapshot: &e.snapshot,
          ty,
          eflag: &mut e.eflag,
          cel_: &mut e.cel_,
        });
      }
    }
  }

  pub fn pread_ref(&mut self, x: CellPtr, xclk: Clock, /*emode: CellEMode,*/ /*pmach: PMach,*/) -> Option<CellEnvEntryMutRef> {
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
            &Cell_::Phy(.., ref cel) => {
              assert_eq!(root, cel.optr);
              match self.celtab.get_mut(&root) {
                None => panic!("bug"),
                Some(e) => {
                  // FIXME: type compat.
                  /*// FIXME FIXME
                  match emode {
                    CellEMode::Read => {
                      assert!(!e.eflag.mutex());
                      assert!(!e.eflag.rwlock());
                      e.eflag.set_read();
                    }
                    CellEMode::Rwlock => {
                      assert!(!e.eflag.mutex());
                      assert!(!e.eflag.read());
                      e.eflag.set_rwlock();
                    }
                    _ => panic!("bug")
                  }*/
                  return Some(CellEnvEntryMutRef{
                    stablect: &e.stablect,
                    //snapshot: &e.snapshot,
                    ty: ty.unwrap(),
                    eflag: &mut e.eflag,
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
                    stablect: &e.stablect,
                    //snapshot: &e.snapshot,
                    ty: ty.unwrap(),
                    eflag: &mut e.eflag,
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

  pub fn pwrite_ref(&mut self, x: CellPtr, /*old_xclk: Clock,*/ new_xclk: Clock, /*emode: CellEMode,*/ /*pmach: PMach,*/) -> Option<CellEnvEntryMutRef> {
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
            // FIXME: create a fresh PCell.
            let ty = e.ty.clone();
            println!("DEBUG: CtxEnv::pwrite_ref: fresh pcel: ty={:?}", &ty);
            let state = state.clone();
            match self.celtab.get_mut(&root) {
              None => panic!("bug"),
              Some(e) => {
                let clo = RefCell::new(CellClosure::default());
                let pcel = PCell::new(optr, ty.clone());
                e.cel_ = Cell_::Phy(state, clo, pcel);
                return Some(CellEnvEntryMutRef{
                  stablect: &e.stablect,
                  //snapshot: &e.snapshot,
                  ty,
                  eflag: &mut e.eflag,
                  cel_: &mut e.cel_,
                });
              }
            }
          }
          &Cell_::Phy(ref state, _, ref cel) => {
            assert_eq!(root, cel.optr);
            /*assert_eq!(state.borrow().clk.update(), new_xclk);*/
            assert_eq!(state.borrow().clk, new_xclk);
            match self.celtab.get_mut(&root) {
              None => panic!("bug"),
              Some(e) => {
                // FIXME: type compat.
                return Some(CellEnvEntryMutRef{
                  stablect: &e.stablect,
                  //snapshot: &e.snapshot,
                  ty,
                  eflag: &mut e.eflag,
                  cel_: &mut e.cel_,
                });
              }
            }
          }
          &Cell_::Cow(ref state, _, ref cel) => {
            assert_eq!(root, cel.optr);
            /*assert_eq!(state.borrow().clk.update(), new_xclk);*/
            assert_eq!(state.borrow().clk, new_xclk);
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
                      stablect: &e.stablect,
                      //snapshot: &e.snapshot,
                      ty,
                      eflag: &mut e.eflag,
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
  }

  pub fn insert_top(&mut self, x: CellPtr, ty: CellType) {
    match self.celtab.get(&x) {
      None => {}
      Some(_) => panic!("bug")
    }
    let e = CellEnvEntry{
      stablect: Cell::new(0),
      //snapshot: Cell::new(0),
      ty,
      eflag:    CellEFlag::default(),
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
      eflag:    CellEFlag::default(),
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
    let (state, clo) = match self.lookup_ref(pcel) {
      None => panic!("bug"),
      Some(e) => {
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
      eflag:    CellEFlag::default(),
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

  pub fn insert_alias(&mut self, x: CellPtr, ty: CellType, og: CellPtr) {
    match self.celtab.get(&x) {
      None => {}
      Some(_) => panic!("bug")
    }
    let e = CellEnvEntry{
      stablect: Cell::new(u32::max_value()),
      //snapshot: Cell::new(u32::max_value()),
      ty,
      eflag:    CellEFlag::default(),
      cel_:     Cell_::Alias(og),
    };
    self.celtab.insert(x, e);
    let mut root = self.alias_root.borrow_mut();
    match root.get(&x) {
      None => {}
      Some(_) => panic!("bug")
    }
    root.insert(x, og);
  }

  pub fn retain(&self, x: CellPtr) {
    if x.is_nil() {
      return;
    }
    match self.lookup_ref(x) {
      None => panic!("bug"),
      Some(e) => {
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
    match self.lookup_ref(x) {
      None => panic!("bug"),
      Some(e) => {
        let cur = e.stablect.get();
        assert!(cur != u32::max_value());
        let next = cur - 1;
        e.stablect.set(next);
      }
    }
  }

  pub fn snapshot(&mut self, ctr: &CtxCtr, pcel: CellPtr) -> CellPtr {
    let (ty, pclk) = match self.lookup_ref(pcel) {
      None => panic!("bug"),
      Some(e) => {
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

  /*pub fn lookup_gradr(&mut self, ctr: &CtxCtr, tg: CellPtr, x: CellPtr) -> CellPtr {
    match self.gradr.get(&[tg, x]) {
      None => {
        panic!("bug");
      }
      Some(&dx) => dx
    }
  }

  pub fn lookup_or_insert_gradr(&mut self, ctr: &CtxCtr, tg: CellPtr, x: CellPtr) -> CellPtr {
    match self.gradr.get(&[tg, x]) {
      None => {
        // FIXME FIXME
        let dx = ctr.fresh_cel();
        let ty = match self.lookup_ref(x) {
          None => panic!("bug"),
          Some(e) => e.ty.clone()
        };
        let cel = PCell::new(dx, ty.clone());
        self.insert_phy(dx, ty, cel);
        self.gradr.insert([tg, x], dx);
        self.ungradr.insert([tg, dx], x);
        dx
      }
      Some(&dx) => dx
    }
  }

  pub fn accumulate_gradr(&mut self, ctr: &CtxCtr, tg: CellPtr, x: CellPtr, dx: CellPtr) -> CellPtr {
    // FIXME FIXME
    match self.gradr.get(&[tg, x]) {
      None => {
        // FIXME FIXME
        let ty = match self.lookup_ref(x) {
          None => panic!("bug"),
          Some(e) => e.ty.clone()
        };
        //let dx0 = self.zeros(ty);
        //assert!(self.gradr.insert([tg, x], dx0).is_none());
      }
      Some(_) => {}
    }
    match self.gradr.get(&[tg, x]) {
      None => unreachable!(),
      Some(&dx0) => {
        //dx0 += dx;
      }
    }
    unimplemented!();
  }*/

  /*pub fn reset_tmp(&mut self) {
    self.tmptab.clear();
  }

  pub fn unify(&mut self, ctr: &CtxCtr, x: CellPtr, uty: Option<CellType>) -> CellPtr {
    assert!(x.to_unchecked() < 0);
    //assert!(y.to_unchecked() > 0);
    match self.lookup_mut_ref(x) {
      None => panic!("bug"),
      Some(e) => {
        // FIXME FIXME: cases.
        match e.cel_ {
          &mut Cell_::Phy(ref _state, _, ref mut pcel) => {
            if pcel.optr == x {
              // FIXME FIXME: fixup Top celltype.
              let y = ctr.fresh_cel();
              pcel.optr = y;
              if let Some(ty) = uty.as_ref() {
                pcel.ogty = ty.clone();
                pcel.olay = CellLayout::new_packed(&ty);
              }
              let mut e = self.celtab.remove(&x).unwrap();
              if let Some(ty) = uty {
                e.ty = ty;
              }
              assert!(self.celtab.insert(y, e).is_none());
              let mut root = self.alias_root.borrow_mut();
              root.insert(x, y);
              self.tmptab.insert(x, y);
              y
            } else if pcel.optr.to_unchecked() <= 0 {
              panic!("bug")
            } else {
              // FIXME FIXME
              pcel.optr
            }
          }
          // FIXME FIXME
          _ => unimplemented!()
        }
      }
    }
  }*/

  pub fn gc_prepare(&self, gc_list: &mut Vec<CellPtr>) {
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
  }
}
