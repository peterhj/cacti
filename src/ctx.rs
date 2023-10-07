use crate::algo::{HashMap, BTreeSet};
use crate::cell::*;
use crate::clock::*;
use crate::nd::{IRange};
use crate::panick::{panick_wrap};
use crate::pctx::{TL_PCTX, Locus, PMach, MemReg};
use crate::spine::*;
use crate::thunk::*;
use crate::util::stat::*;
use cacti_cfg_env::*;

//use futhark_syntax::re::{ReTrie};
//use futhark_syntax::tokenizing::{Token as FutToken};

use std::any::{Any};
use std::borrow::{Borrow};
use std::cell::{Cell, RefCell, Ref, RefMut};
use std::mem::{swap};
use std::rc::{Rc};

thread_local! {
  pub static TL_CTX_CFG: CtxCfg = CtxCfg::default();
  pub static TL_CTX: Ctx = {
    let ctx = Ctx::new();
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
      //gpu_reserve:      Cell::new(901),
      //gpu_workspace:    Cell::new(1),
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
  pub ctlstate: CtlState,
  pub spine:    Spine,
  pub ctr:      CtxCtr,
  pub env:      RefCell<CtxEnv>,
  pub thunkenv: RefCell<CtxThunkEnv>,
  //pub futhark:  RefCell<FutharkCtx>,
  pub timing:   TimingCtx,
  pub debugctr: DebugCtrs,
}

impl Drop for Ctx {
  fn drop(&mut self) {
    if cfg_debug_timing() {
    let digest = self.timing.digest();
    println!("DEBUG:  Ctx::drop: timing digest: pregemm1: {:?}", digest.pregemm1);
    println!("DEBUG:  Ctx::drop: timing digest: gemm1:    {:?}", digest.gemm1);
    println!("DEBUG:  Ctx::drop: timing digest: pregemm:  {:?}", digest.pregemm);
    println!("DEBUG:  Ctx::drop: timing digest: gemm:     {:?}", digest.gemm);
    println!("DEBUG:  Ctx::drop: timing digest: f_build1: {:?}", digest.f_build1);
    println!("DEBUG:  Ctx::drop: timing digest: f_setup1: {:?}", digest.f_setup1);
    println!("DEBUG:  Ctx::drop: timing digest: futhark1: {:?}", digest.futhark1);
    println!("DEBUG:  Ctx::drop: timing digest: f_build:  {:?}", digest.f_build);
    println!("DEBUG:  Ctx::drop: timing digest: f_setup:  {:?}", digest.f_setup);
    println!("DEBUG:  Ctx::drop: timing digest: futhark:  {:?}", digest.futhark);
    println!("DEBUG:  Ctx::drop: debug counter: accumulate hashes:       {:?}", self.debugctr.accumulate_hashes.borrow());
    println!("DEBUG:  Ctx::drop: debug counter: accumulate in place:     {:?}", self.debugctr.accumulate_in_place.get());
    println!("DEBUG:  Ctx::drop: debug counter: accumulate not in place: {:?}", self.debugctr.accumulate_not_in_place.get());
    }
  }
}

impl Ctx {
  pub fn new() -> Ctx {
    if cfg_debug() { println!("DEBUG: Ctx::new"); }
    Ctx{
      ctlstate: CtlState::default(),
      spine:    Spine::default(),
      ctr:      CtxCtr::new(),
      env:      RefCell::new(CtxEnv::default()),
      thunkenv: RefCell::new(CtxThunkEnv::default()),
      //futhark:  RefCell::new(FutharkCtx::default()),
      timing:   TimingCtx::default(),
      debugctr: DebugCtrs::default(),
    }
  }

  pub fn reset(&self) -> Counter {
    if cfg_verbose_info() {
      println!("INFO:   Ctx::reset: gc: scan {} cells", self.ctr.celfront.borrow().len());
    }
    let mut env = self.env.borrow_mut();
    let mut spine_env = self.spine.cur_env.borrow_mut();
    let mut next_celfront = Vec::new();
    let mut free_ct = 0;
    TL_PCTX.with(|pctx| {
      if cfg_verbose_info() {
      pctx.swap._dump_usage();
      if let Some(gpu) = pctx.nvgpu.as_ref() {
        gpu._dump_usage();
        if cfg_verbose_report() {
        gpu._dump_sizes();
        gpu._dump_free();
        }
      }
      }
      for &x in self.ctr.celfront.borrow().iter() {
        let mut f = false;
        match env._lookup_view(x) {
          Err(_) => {
            let _ = env.celtab.remove(&x);
            let _ = env.celroot.borrow_mut().remove(&x);
          }
          Ok(e) => {
            let xroot = e.root();
            if e.stablect > 0 {
              next_celfront.push(x);
            } else {
              if x == xroot {
                let cel_ = e.cel_.borrow();
                match &*cel_ {
                  &Cell_::Phy(.., ref pcel) => {
                    for (_, rep) in pcel.replicas.iter() {
                      let addr = rep.addr.get();
                      match pctx.yeet(addr) {
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
              }
              if x.to_unchecked() == 726 {
                println!("WARNING:Ctx::reset: gc: x={:?}", x);
              }
              assert!(env.celtab.remove(&x).is_some());
              let _ = env.celroot.borrow_mut().remove(&x);
              let _ = spine_env.state.remove(&x);
            }
          }
        }
        if f {
          free_ct += 1;
        }
      }
      if let Some(gpu) = pctx.nvgpu.as_ref() {
        gpu.mem_pool.reset_peak_size();
      }
      if cfg_verbose_info() {
      pctx.swap._dump_usage();
      if let Some(gpu) = pctx.nvgpu.as_ref() {
        gpu._dump_usage();
        if cfg_verbose_report() {
        gpu._dump_sizes();
        gpu._dump_free();
        }
      }
      }
    });
    drop(spine_env);
    if cfg_verbose_info() {
      println!("INFO:   Ctx::reset: gc:   free {} cells", free_ct);
      println!("INFO:   Ctx::reset: gc:   next {} cells", next_celfront.len());
    }
    swap(&mut *self.ctr.celfront.borrow_mut(), &mut next_celfront);
    env._reset();
    drop(env);
    self.thunkenv.borrow_mut()._reset();
    let rst = self.spine._reset();
    rst
  }
}

#[derive(Default)]
pub struct CtlState {
  // TODO
  pub primary:  Cell<Option<PMach>>,
  pub accumulate_in_place: Cell<bool>,
  pub assume_uninit_zero: Cell<bool>,
}

impl CtlState {
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
  //pub trie: Option<Rc<ReTrie<FutToken>>>,
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
pub fn reset() -> Counter {
  panick_wrap(|| TL_CTX.with(|ctx| ctx.reset()))
}

#[track_caller]
pub fn compile() {
  panick_wrap(|| TL_CTX.with(|ctx| ctx.spine._compile()))
}

#[track_caller]
pub fn resume() -> SpineRet {
  panick_wrap(|| TL_CTX.with(|ctx| {
    let mut thunkenv = ctx.thunkenv.borrow_mut();
    ctx.spine._resume(&ctx.ctr, /*&mut *env,*/ &mut *thunkenv, /*CellPtr::nil(), Clock::default(),*/ SpineResume::_Top)
  }))
}

/*#[track_caller]
pub fn _resume_put_mem_with<K: CellDeref, F: Fn(CellType, MemReg)>(key: K, fun: F) -> SpineRet {
  panick_wrap(|| TL_CTX.with(|ctx| {
    let mut thunkenv = ctx.thunkenv.borrow_mut();
    ctx.spine._resume(&ctx.ctr, &mut *thunkenv, SpineResume::PutMemF(key._deref(), &fun as _))
  }))
}*/

#[track_caller]
pub fn resume_put_mem_with<K: CellDeref, F: Fn(TypedMemMut)>(key: K, fun: F) -> SpineRet {
  panick_wrap(|| TL_CTX.with(|ctx| {
    let mut thunkenv = ctx.thunkenv.borrow_mut();
    ctx.spine._resume(&ctx.ctr, &mut *thunkenv, SpineResume::PutMemMutFun(key._deref(), &fun as _))
  }))
}

#[track_caller]
pub fn resume_put<K: CellDeref, V: CellStoreTo>(key: K, ty: &CellType, val: &V) -> SpineRet {
  panick_wrap(|| TL_CTX.with(|ctx| {
    let mut thunkenv = ctx.thunkenv.borrow_mut();
    ctx.spine._resume(&ctx.ctr, &mut *thunkenv, SpineResume::Put(key._deref(), ty, val as _))
  }))
}

#[track_caller]
pub fn yield_() {
  panick_wrap(|| TL_CTX.with(|ctx| {
    let spine = ctx.spine.borrow();
    spine.yield_();
  }))
}

/*#[track_caller]
pub fn break_() {
  panick_wrap(|| TL_CTX.with(|ctx| {
    unimplemented!();
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

impl Default for PMachScope {
  #[track_caller]
  fn default() -> PMachScope {
    TL_CTX.with(|ctx| {
      let prev = ctx.ctlstate._unset_primary();
      PMachScope{prev}
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

  pub fn with<F: FnMut(&PMachScope)>(self, mut f: F) {
    (f)(&self);
  }
}

#[track_caller]
pub fn default_scope() -> PMachScope {
  panick_wrap(|| PMachScope::default())
}

#[track_caller]
pub fn no_scope() -> PMachScope {
  panick_wrap(|| PMachScope::default())
}

#[track_caller]
pub fn smp_scope() -> PMachScope {
  panick_wrap(|| PMachScope::new(PMach::Smp))
}

#[cfg(feature = "nvgpu")]
#[track_caller]
pub fn gpu_scope() -> PMachScope {
  panick_wrap(|| PMachScope::new(PMach::NvGpu))
}

#[cfg(feature = "nvgpu")]
#[track_caller]
pub fn nvgpu_scope() -> PMachScope {
  panick_wrap(|| PMachScope::new(PMach::NvGpu))
}

pub fn ctx_release(x: CellPtr) {
  TL_CTX.try_with(|ctx| {
    ctx.env.borrow()._release_probe(x);
  }).unwrap_or(())
}

pub fn ctx_retain(x: CellPtr) {
  TL_CTX.with(|ctx| {
    ctx.env.borrow()._retain_probe(x);
  })
}

pub fn ctx_lookup_type(x: CellPtr) -> CellType {
  TL_CTX.with(|ctx| {
    match ctx.env.borrow()._lookup_view(x) {
      Err(_) => panic!("bug"),
      Ok(e) => e.ty.clone()
    }
  })
}

pub fn ctx_lookup_dtype(x: CellPtr) -> Dtype {
  TL_CTX.with(|ctx| {
    match ctx.env.borrow()._lookup_view(x) {
      Err(_) => panic!("bug"),
      Ok(e) => e.ty.dtype
    }
  })
}

pub fn ctx_lookup_clk(x: CellPtr) -> Clock {
  TL_CTX.with(|ctx| {
    match ctx.env.borrow()._lookup_view(x) {
      Err(_) => panic!("bug"),
      Ok(e) => {
        let cel_ = e.cel_.borrow();
        match &*cel_ {
          &Cell_::Top(ref state, ..) => state.borrow().clk,
          &Cell_::Phy(ref state, ..) => state.borrow().clk,
          &Cell_::Bot => panic!("bug"),
          _ => panic!("bug")
        }
      }
    }
  })
}

/*pub fn ctx_fresh_mset() -> MCellPtr {
  TL_CTX.with(|ctx| {
    let x = ctx.ctr.fresh_mcel();
    //ctx.env.borrow_mut().mceltab.insert(x, MCellEnvEntry{mcel_: MCell_::Set(MCellSet::default())});
    x
  })
}

pub fn ctx_fresh_mmap() -> MCellPtr {
  TL_CTX.with(|ctx| {
    let x = ctx.ctr.fresh_mcel();
    //ctx.env.borrow_mut().mceltab.insert(x, MCellEnvEntry{mcel_: MCell_::Map(MCellMap::default())});
    x
  })
}*/

pub fn ctx_alias_new_shape(og: CellPtr, new_shape: Box<[i64]>) -> CellPtr {
  TL_CTX.with(|ctx| {
    let mut env = ctx.env.borrow_mut();
    match env._lookup_view(og) {
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
        //println!("DEBUG: ctx_alias_new_shape: og={:?} old shape={:?} x={:?} new shape={:?} compat={:?}", og, &e.ty.shape, x, &new_ty.shape, cmp);
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
    match env._lookup_view(og) {
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
        //println!("DEBUG: ctx_alias_bits: og={:?} old dtype={:?} x={:?} new dtype={:?}", og, e.ty.dtype, x, new_dtype);
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

pub fn ctx_clean_arg() -> bool {
  TL_CTX.with(|ctx| {
    let thunkenv = ctx.thunkenv.borrow();
    thunkenv.param.is_empty()
    && thunkenv.arg.is_empty()
    /*&& thunkenv.out.is_empty()*/
  })
}

pub fn ctx_push_cell_arg(x: CellPtr) {
  TL_CTX.with(|ctx| {
    match ctx.env.borrow()._lookup_view(x) {
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
      let ty_ = match ctx.env.borrow()._lookup_view(arg) {
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
      let ty_ = match ctx.env.borrow()._lookup_view(arg) {
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
      let ty_ = match ctx.env.borrow()._lookup_view(arg) {
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
      let ty_ = match ctx.env.borrow()._lookup_view(arg) {
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
    let yroot = match ctx.env.borrow()._lookup_view(y) {
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
      let ty_ = match ctx.env.borrow()._lookup_view(arg) {
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
      let ty_ = match ctx.env.borrow()._lookup_view(arg) {
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
    let yroot = match ctx.env.borrow()._lookup_view(y) {
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
      let ty_ = match ctx.env.borrow()._lookup_view(arg) {
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
    let yroot = match ctx.env.borrow()._lookup_view(y) {
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

pub struct CtxCtr {
  //pub reset:    Cell<Counter>,
  pub ptr_ctr:  Cell<i64>,
  pub celfront: RefCell<Vec<CellPtr>>,
}

impl CtxCtr {
  pub fn new() -> CtxCtr {
    CtxCtr{
      //reset:    Cell::new(Counter::default()),
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
  pub param:    Vec<ScalarVal_>,
  pub arg:      Vec<(CellPtr, Clock)>,
  pub out:      CellPtr,
  pub pmach:    PMach,
}

#[derive(Default)]
pub struct CtxThunkEnv {
  pub thunktab: HashMap<ThunkPtr, ThunkEnvEntry>,
  pub thunkidx: HashMap<(u16, u16, Vec<Dim>, ThunkKey, ), ThunkPtr>,
  pub update:   HashMap<(CellPtr, Clock), ThunkClosure>,
  pub param:    Vec<ScalarVal_>,
  pub arg:      Vec<(CellPtr, Clock)>,
  //pub out:      Vec<(CellPtr, Clock)>,
}

impl CtxThunkEnv {
  pub fn _reset(&mut self) {
    self.update.clear();
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

  /*pub fn gc(&mut self, gc_list: &[CellPtr]) {
    // FIXME FIXME: remove updating thunks.
  }*/
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
      println!("DEBUG: CellClosure::update: clk={:?} th={:?} self.ctr={:?} self.thunk={:?}",
          clk, th, &self.ctr, &self.thunk);
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

pub enum CellAlias {
  View(CellVOp),
  NewShape,
  BitAlias,
  Const_,
  Opaque,
}

#[derive(Clone, Copy, Debug)]
pub enum CellName {
  Top,
  Phy,
  Alias,
  Bot,
}

pub enum Cell_ {
  Top(CellState, CellPtr),
  Phy(CellState, CellClosure, PCell),
  Alias(CellAlias, CellPtr),
  Bot,
}

impl Cell_ {
  pub fn name(&self) -> CellName {
    match self {
      &Cell_::Top(..) => CellName::Top,
      &Cell_::Phy(..) => CellName::Phy,
      &Cell_::Alias(..) => CellName::Alias,
      &Cell_::Bot => CellName::Bot,
    }
  }

  pub fn state_ref(&self) -> &CellState {
    match self {
      &Cell_::Top(ref state, ..) => state,
      &Cell_::Phy(ref state, ..) => state,
      _ => panic!("bug")
    }
  }

  pub fn state_mut(&mut self) -> &mut CellState {
    match self {
      &mut Cell_::Top(ref mut state, ..) => state,
      &mut Cell_::Phy(ref mut state, ..) => state,
      _ => panic!("bug")
    }
  }
}

pub struct CellEnvEntry {
  pub ty:       CellType,
  pub stablect: Cell<u32>,
  pub cel_:     RefCell<Cell_>,
}

impl CellEnvEntry {
  pub fn state_ref(&self) -> Ref<CellState> {
    Ref::map(self.cel_.borrow(), |cel| cel.state_ref())
  }
}

pub struct CellRef_<'a, R> {
  pub ref_:     R,
  pub root_ty:  &'a CellType,
  pub ty:       &'a CellType,
  pub stablect: u32,
  pub cel_:     &'a RefCell<Cell_>,
}

impl<'a> CellRef_<'a, CellPtr> {
  pub fn root(&self) -> CellPtr {
    self.ref_
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

impl<'a, R> CellRef_<'a, R> {
  pub fn state(&self) -> RefMut<CellState> {
    RefMut::map(self.cel_.borrow_mut(), |cel| cel.state_mut())
  }

  pub fn clock_sync(&self, prev_clk: Clock, next_clk: Clock, env: &CtxEnv) {
    assert_eq!(self.cel_.borrow().state_ref().clk, prev_clk);
    self.cel_.borrow_mut().state_mut().clk = next_clk;
  }

  pub fn clock_sync_rec(&self, prev_clk: Clock, next_clk: Clock, env: &CtxEnv) {
    assert_eq!(self.cel_.borrow().state_ref().clk, prev_clk);
    self.cel_.borrow_mut().state_mut().clk = next_clk;
    match &*self.cel_.borrow() {
      &Cell_::Top(..) => {}
      &Cell_::Phy(.., ref pcel) => {
        for (_, rep) in pcel.replicas.iter() {
          if rep.clk.get() > prev_clk {
            panic!("bug");
          } else if rep.clk.get() == prev_clk {
            rep.clk.set(next_clk);
          }
        }
      }
      _ => panic!("bug")
    }
  }
}

pub type CellDerefResult<T=CellPtr> = Result<T, CellDerefErr>;

pub type CellProbePtr = CellDerefResult<CellPtr>;
pub type CellProbeView = CellDerefResult<CellView>;

pub type CellDerefPtr<'a> = CellDerefResult<CellRef_<'a, CellPtr>>;
pub type CellDerefView<'a> = CellDerefResult<CellRef_<'a, CellView>>;

#[derive(Clone, Copy, Debug)]
pub enum CellDerefErr {
  Nil,
  MissingRoot,
  Missing,
  Reentrant,
  //Read,
  //Write,
  View,
  Bot,
}

//#[derive(Default)]
pub struct CtxEnv {
  pub celtab:   HashMap<CellPtr, CellEnvEntry>,
  pub celroot:  RefCell<HashMap<CellPtr, (CellPtr, Option<CellView>)>>,
  pub unlive:   RefCell<BTreeSet<CellPtr>>,
  /*pub atomtab:  HashMap<Atom, ()>,
  pub mceltab:  HashMap<MCellPtr, MCellEnvEntry>,*/
  //pub tag:      HashMap<CellPtr, HashSet<String>>,
  //pub tag:      HashMap<CellPtr, Vec<String>>,
}

impl Default for CtxEnv {
  fn default() -> CtxEnv {
    CtxEnv{
      celtab:   HashMap::default(),
      celroot:  RefCell::new(HashMap::default()),
      unlive:   RefCell::new(BTreeSet::new()),
      /*atomtab:  HashMap::new(),
      mceltab:  HashMap::new(),*/
      //tag:      HashMap::new(),
    }
  }
}

impl CtxEnv {
  pub fn _reset(&self) {
    /*self.celtab.clear();*/
    self.unlive.borrow_mut().clear();
  }

  pub fn _retain_probe(&self, query: CellPtr) {
    if query.is_nil() {
      return;
    }
    let mut cursor = query;
    loop {
      match self.celtab.get(&cursor) {
        None => {
          panic!("bug");
        }
        Some(e) => {
          let ref_ct = e.stablect.get();
          assert!(ref_ct < u32::max_value() - 1);
          e.stablect.set(ref_ct + 1);
          let cel_ = e.cel_.borrow();
          match &*cel_ {
            &Cell_::Top(..) |
            &Cell_::Phy(..) => {
              return;
            }
            &Cell_::Alias(_, next) => {
              cursor = next;
            }
            &Cell_::Bot => {
              panic!("bug");
            }
            _ => unimplemented!()
          }
        }
      }
    }
  }

  pub fn _release_probe(&self, query: CellPtr) {
    if query.is_nil() {
      return;
    }
    let mut cursor = query;
    loop {
      match self.celtab.get(&cursor) {
        None => {
          panic!("bug");
        }
        Some(e) => {
          let ref_ct = e.stablect.get();
          assert!(ref_ct > 0);
          e.stablect.set(ref_ct - 1);
          let cel_ = e.cel_.borrow();
          match &*cel_ {
            &Cell_::Top(..) |
            &Cell_::Phy(..) => {
              return;
            }
            &Cell_::Alias(_, next) => {
              cursor = next;
            }
            &Cell_::Bot => {
              panic!("bug");
            }
            _ => unimplemented!()
          }
        }
      }
    }
  }

  pub fn _try_probe_ref(&self, query: CellPtr) -> Option<CellProbePtr> {
    if query.is_nil() {
      return Some(Err(CellDerefErr::Nil));
    }
    match self.celroot.borrow().get(&query) {
      None => panic!("bug"),
      Some(&(root, _)) => {
        return Some(Ok(root));
      }
    }
    /*let mut cursor = query;
    loop {
      match self.celtab.get(&cursor) {
        None => {
          return Some(Err(CellDerefErr::Missing));
        }
        Some(e) => {
          let cel_ = match e.cel_.try_borrow() {
            Err(_) => return None,
            Ok(cel_) => cel_
          };
          match &*cel_ {
            &Cell_::Top(..) |
            &Cell_::Phy(..) => {
              return Some(Ok(cursor));
            }
            &Cell_::Alias(ref alias, next) => {
              match alias {
                &CellAlias::View(_) => {
                  return Some(Err(CellDerefErr::View));
                }
                //&CellAlias::NewType |
                &CellAlias::NewShape |
                &CellAlias::BitAlias |
                &CellAlias::Opaque |
                &CellAlias::Const_ => {}
              }
              cursor = next;
            }
            &Cell_::Bot => {
              return Some(Err(CellDerefErr::Bot));
            }
            _ => unimplemented!()
          }
        }
      }
    }*/
  }

  pub fn _probe_ref(&self, query: CellPtr) -> CellProbePtr {
    if query.is_nil() {
      return Err(CellDerefErr::Nil);
    }
    match self.celroot.borrow().get(&query) {
      None => panic!("bug"),
      Some(&(root, _)) => {
        return Ok(root);
      }
    }
    /*let mut cursor = query;
    loop {
      match self.celtab.get(&cursor) {
        None => {
          return Err(CellDerefErr::Missing);
        }
        Some(e) => {
          let cel_ = e.cel_.borrow();
          match &*cel_ {
            &Cell_::Top(..) |
            &Cell_::Phy(..) => {
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
            &Cell_::Bot => {
              return Err(CellDerefErr::Bot);
            }
            _ => unimplemented!()
          }
        }
      }
    }*/
  }

  pub fn _probe_view(&self, query: CellPtr) -> CellProbeView {
    if query.is_nil() {
      return Err(CellDerefErr::Nil);
    }
    match self.celroot.borrow().get(&query) {
      None => panic!("bug"),
      Some(&(_, None)) => {}
      Some(&(root, Some(ref view))) => {
        assert_eq!(root, view.root);
        let view = view.clone();
        return Ok(view);
      }
    }
    let mut cursor = query;
    let mut view = CellView::default();
    loop {
      match self.celtab.get(&cursor) {
        None => {
          return Err(CellDerefErr::Missing);
        }
        Some(e) => {
          let cel_ = e.cel_.borrow();
          match &*cel_ {
            &Cell_::Top(..) |
            &Cell_::Phy(..) => {
              view.root = cursor;
              //view.r_ty = e.ty.clone();
              view.vlog.reverse();
              match self.celroot.borrow_mut().get_mut(&query) {
                None => panic!("bug"),
                Some(&mut (root, ref mut view_)) => {
                  assert_eq!(root, cursor);
                  assert!(view_.is_none());
                  *view_ = Some(view.clone());
                }
              }
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
                &CellAlias::BitAlias => {
                  view.vlog.push(CellViewOp::bit_alias(e.ty.dtype));
                }
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

  pub fn _probe_root(&self, query: CellPtr) -> CellProbePtr {
    if query.is_nil() {
      return Err(CellDerefErr::Nil);
    }
    let mut cursor = query;
    loop {
      match self.celtab.get(&cursor) {
        None => {
          return Err(CellDerefErr::Missing);
        }
        Some(e) => {
          let cel_ = e.cel_.borrow();
          match &*cel_ {
            &Cell_::Top(..) |
            &Cell_::Phy(..) => {
              return Ok(cursor);
            }
            &Cell_::Alias(_, next) => {
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

  pub fn _try_lookup_ref_(&self, query: CellPtr) -> Option<CellDerefPtr> {
    let (ty, stablect) = match self.celtab.get(&query) {
      None => return Some(Err(CellDerefErr::Missing)),
      Some(e) => {
        let cel_ = match e.cel_.try_borrow() {
          Err(_) => return None,
          Ok(cel_) => cel_
        };
        match &*cel_ {
          &Cell_::Top(.., optr) => {
            assert_eq!(query, optr);
          }
          &Cell_::Phy(.., ref cel) => {
            assert_eq!(query, cel.optr);
          }
          &Cell_::Alias(..) => {}
          &Cell_::Bot => {
            return Some(Err(CellDerefErr::Bot));
          }
          _ => unimplemented!()
        }
        match &*cel_ {
          &Cell_::Top(..) |
          &Cell_::Phy(..) => {
            return Some(Ok(CellRef_{
              ref_: query,
              root_ty: &e.ty,
              ty: &e.ty,
              stablect: e.stablect.get(),
              cel_: &e.cel_,
            }));
          }
          &Cell_::Alias(..) => {}
          _ => unreachable!()
        }
        (&e.ty, e.stablect.get())
      }
    };
    let root = match self._try_probe_ref(query) {
      None => return None,
      Some(Err(e)) => return Some(Err(e)),
      Some(Ok(root)) => root
    };
    match self.celtab.get(&root) {
      None => return Some(Err(CellDerefErr::MissingRoot)),
      Some(e) => {
        let cel_ = match e.cel_.try_borrow() {
          Err(_) => return None,
          Ok(cel_) => cel_
        };
        match &*cel_ {
          &Cell_::Top(.., optr) => {
            assert_eq!(root, optr);
          }
          &Cell_::Phy(.., ref cel) => {
            assert_eq!(root, cel.optr);
          }
          &Cell_::Alias(..) => {
            panic!("bug");
          }
          &Cell_::Bot => {
            return Some(Err(CellDerefErr::Bot));
          }
          _ => unimplemented!()
        }
        // FIXME: type compat.
        if e.ty.dtype != Dtype::_Top {
          assert_eq!(ty.dtype.size_bytes(), e.ty.dtype.size_bytes());
          assert!(ty.shape_compat(&e.ty) != ShapeCompat::Incompat);
        }
        Some(Ok(CellRef_{
          ref_: root,
          root_ty: &e.ty,
          ty,
          stablect,
          cel_: &e.cel_,
        }))
      }
    }
  }

  pub fn _lookup_ref_(&self, query: CellPtr) -> CellDerefPtr {
    let (ty, stablect) = match self.celtab.get(&query) {
      None => return Err(CellDerefErr::Missing),
      Some(e) => {
        let cel_ = e.cel_.borrow();
        match &*cel_ {
          &Cell_::Top(.., optr) => {
            assert_eq!(query, optr);
          }
          &Cell_::Phy(.., ref cel) => {
            assert_eq!(query, cel.optr);
          }
          &Cell_::Alias(..) => {}
          &Cell_::Bot => {
            return Err(CellDerefErr::Bot);
          }
          _ => unimplemented!()
        }
        match &*cel_ {
          &Cell_::Top(..) |
          &Cell_::Phy(..) => {
            return Ok(CellRef_{
              ref_: query,
              root_ty: &e.ty,
              ty: &e.ty,
              stablect: e.stablect.get(),
              cel_: &e.cel_,
            });
          }
          &Cell_::Alias(..) => {}
          _ => unreachable!()
        }
        (&e.ty, e.stablect.get())
      }
    };
    let root = self._probe_ref(query)?;
    match self.celtab.get(&root) {
      None => return Err(CellDerefErr::MissingRoot),
      Some(e) => {
        let cel_ = e.cel_.borrow();
        match &*cel_ {
          &Cell_::Top(.., optr) => {
            assert_eq!(root, optr);
          }
          &Cell_::Phy(.., ref cel) => {
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
          stablect,
          cel_: &e.cel_,
        })
      }
    }
  }

  pub fn _lookup_view(&self, query: CellPtr) -> CellDerefView {
    let (ty, stablect) = match self.celtab.get(&query) {
      None => return Err(CellDerefErr::Missing),
      Some(e) => {
        let cel_ = e.cel_.borrow();
        match &*cel_ {
          &Cell_::Top(.., optr) => {
            assert_eq!(query, optr);
          }
          &Cell_::Phy(.., ref cel) => {
            assert_eq!(query, cel.optr);
          }
          &Cell_::Alias(..) => {}
          &Cell_::Bot => {
            return Err(CellDerefErr::Bot);
          }
          _ => unimplemented!()
        }
        match &*cel_ {
          &Cell_::Top(..) |
          &Cell_::Phy(..) => {
            return Ok(CellRef_{
              ref_: query.into(),
              root_ty: &e.ty,
              ty: &e.ty,
              stablect: e.stablect.get(),
              cel_: &e.cel_,
            });
          }
          &Cell_::Alias(..) => {}
          _ => unreachable!()
        }
        (&e.ty, e.stablect.get())
      }
    };
    let view = self._probe_view(query)?;
    match self.celtab.get(&view.root) {
      None => return Err(CellDerefErr::MissingRoot),
      Some(e) => {
        let cel_ = e.cel_.borrow();
        match &*cel_ {
          &Cell_::Top(.., optr) => {
            assert_eq!(view.root, optr);
          }
          &Cell_::Phy(.., ref cel) => {
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
          stablect,
          cel_: &e.cel_,
        })
      }
    }
  }

  pub fn pread_view(&self, query: CellPtr, clk: Clock, loc: Locus) -> CellDerefView {
    let mut noalias = false;
    let (ty, stablect) = match self.celtab.get(&query) {
      None => return Err(CellDerefErr::Missing),
      Some(e) => {
        let cel_ = e.cel_.borrow();
        match &*cel_ {
          &Cell_::Top(.., optr) => {
            assert_eq!(query, optr);
            noalias = true;
          }
          &Cell_::Phy(.., ref pcel) => {
            assert_eq!(query, pcel.optr);
            noalias = true;
          }
          &Cell_::Alias(..) => {}
          &Cell_::Bot => {
            return Err(CellDerefErr::Bot);
          }
          _ => unreachable!()
        }
        (&e.ty, e.stablect.get())
      }
    };
    let view = if !noalias { self._probe_view(query)? } else { query.into() };
    let root = view.root();
    match self.celtab.get(&root) {
      None => return Err(CellDerefErr::MissingRoot),
      Some(e) => {
        let mut cel_ = e.cel_.borrow_mut();
        match &mut *cel_ {
          &mut Cell_::Top(.., optr) => {
            assert_eq!(root, optr);
            panic!("bug");
            //return Err(CellDerefErr::Read);
          }
          &mut Cell_::Phy(ref state, .., ref mut pcel) => {
            assert_eq!(root, pcel.optr);
            assert_eq!(&e.ty, &pcel.ogty);
            assert_eq!(clk, state.clk);
            pcel.read_loc(root, clk, &e.ty, loc);
            drop(cel_);
            return Ok(CellRef_{
              ref_: view,
              root_ty: &e.ty,
              ty,
              stablect,
              cel_: &e.cel_,
            });
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

  pub fn pwrite_view(&self, query: CellPtr, next_clk: Clock, loc: Locus) -> CellDerefView {
    let mut noalias = false;
    let (ty, stablect) = match self.celtab.get(&query) {
      None => return Err(CellDerefErr::Missing),
      Some(e) => {
        let cel_ = e.cel_.borrow();
        match &*cel_ {
          &Cell_::Top(.., optr) => {
            assert_eq!(query, optr);
            noalias = true;
          }
          &Cell_::Phy(.., ref pcel) => {
            assert_eq!(query, pcel.optr);
            noalias = true;
          }
          &Cell_::Alias(..) => {}
          &Cell_::Bot => {
            return Err(CellDerefErr::Bot);
          }
          _ => unreachable!()
        }
        (&e.ty, e.stablect.get())
      }
    };
    let view = if !noalias { self._probe_view(query)? } else { query.into() };
    let root = view.root();
    match self.celtab.get(&root) {
      None => return Err(CellDerefErr::MissingRoot),
      Some(e) => {
        let mut cel_ = e.cel_.borrow_mut();
        match &mut *cel_ {
          &mut Cell_::Top(ref state, optr) => {
            assert_eq!(root, optr);
            assert_eq!(next_clk, state.clk);
            let state = state.clone();
            let clo = CellClosure::default();
            let mut pcel = PCell::new(optr, e.ty.clone());
            pcel.write_loc(root, next_clk, &e.ty, loc);
            *cel_ = Cell_::Phy(state, clo, pcel);
            drop(cel_);
            return Ok(CellRef_{
              ref_: view,
              root_ty: &e.ty,
              ty,
              stablect,
              cel_: &e.cel_,
            });
          }
          &mut Cell_::Phy(ref state, .., ref mut pcel) => {
            assert_eq!(root, pcel.optr);
            assert_eq!(&e.ty, &pcel.ogty);
            assert_eq!(next_clk, state.clk);
            pcel.write_loc(root, next_clk, &e.ty, loc);
            drop(cel_);
            return Ok(CellRef_{
              ref_: view,
              root_ty: &e.ty,
              ty,
              stablect,
              cel_: &e.cel_,
            });
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

  pub fn prewrite_view(&self, query: CellPtr, prev_clk: Clock, next_clk: Clock, loc: Locus) -> CellDerefView {
    let mut noalias = false;
    let (ty, stablect) = match self.celtab.get(&query) {
      None => return Err(CellDerefErr::Missing),
      Some(e) => {
        let cel_ = e.cel_.borrow();
        match &*cel_ {
          &Cell_::Top(.., optr) => {
            assert_eq!(query, optr);
            noalias = true;
          }
          &Cell_::Phy(.., ref pcel) => {
            assert_eq!(query, pcel.optr);
            noalias = true;
          }
          &Cell_::Alias(..) => {}
          &Cell_::Bot => {
            return Err(CellDerefErr::Bot);
          }
          _ => unreachable!()
        }
        (&e.ty, e.stablect.get())
      }
    };
    let view = if !noalias { self._probe_view(query)? } else { query.into() };
    let root = view.root();
    match self.celtab.get(&root) {
      None => return Err(CellDerefErr::MissingRoot),
      Some(e) => {
        let mut cel_ = e.cel_.borrow_mut();
        match &mut *cel_ {
          &mut Cell_::Top(ref state, optr) => {
            assert_eq!(root, optr);
            panic!("bug");
          }
          &mut Cell_::Phy(ref state, .., ref mut pcel) => {
            assert_eq!(root, pcel.optr);
            assert_eq!(&e.ty, &pcel.ogty);
            assert_eq!(next_clk, state.clk);
            pcel.read_loc(root, prev_clk, &e.ty, loc);
            pcel.write_loc(root, next_clk, &e.ty, loc);
            drop(cel_);
            return Ok(CellRef_{
              ref_: view,
              root_ty: &e.ty,
              ty,
              stablect,
              cel_: &e.cel_,
            });
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

  pub fn insert_top(&mut self, x: CellPtr, ty: CellType) {
    let e = CellEnvEntry{
      stablect: Cell::new(0),
      ty,
      cel_: RefCell::new(Cell_::Top(CellState::default(), x)),
    };
    assert!(self.celtab.insert(x, e).is_none());
    assert!(self.celroot.borrow_mut().insert(x, (x, None)).is_none());
  }

  pub fn insert_phy(&mut self, x: CellPtr, ty: CellType, pcel: PCell) {
    assert_eq!(x, pcel.optr);
    let e = CellEnvEntry{
      stablect: Cell::new(0),
      ty,
      cel_: RefCell::new(Cell_::Phy(
                CellState::default(),
                CellClosure::default(),
                pcel,
            )),
    };
    assert!(self.celtab.insert(x, e).is_none());
    assert!(self.celroot.borrow_mut().insert(x, (x, None)).is_none());
  }

  pub fn insert_alias(&mut self, x: CellPtr, alias: CellAlias, ty: CellType, og: CellPtr) {
    let e = CellEnvEntry{
      //stablect: Cell::new(u32::max_value()),
      stablect: Cell::new(0),
      ty,
      cel_: RefCell::new(Cell_::Alias(alias, og)),
    };
    assert!(self.celtab.insert(x, e).is_none());
    let root = match self._probe_root(og) {
      Err(_) => panic!("bug"),
      Ok(root) => root
    };
    assert!(self.celroot.borrow_mut().insert(x, (root, None)).is_none());
  }

  pub fn insert_const_(&mut self, x: CellPtr, og: CellPtr) {
    let ty = match self._lookup_view(og) {
      Err(_) => panic!("bug"),
      Ok(oe) => oe.ty.clone()
    };
    let e = CellEnvEntry{
      stablect: Cell::new(0),
      ty,
      cel_: RefCell::new(Cell_::Alias(CellAlias::Const_, og)),
    };
    assert!(self.celtab.insert(x, e).is_none());
    let root = match self._probe_root(og) {
      Err(_) => panic!("bug"),
      Ok(root) => root
    };
    assert!(self.celroot.borrow_mut().insert(x, (root, None)).is_none());
  }
}
