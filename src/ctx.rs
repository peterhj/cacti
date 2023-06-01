use crate::cell::*;
use crate::clock::*;
use crate::panick::*;
use crate::spine::*;
use crate::thunk::*;
use crate::thunk::op::*;

//use std::any::{Any, TypeId};
use std::cell::{Cell, RefCell, RefMut};
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::mem::{swap};
use std::rc::{Rc};

thread_local! {
  pub static TL_CTX_CFG: CtxCfg = CtxCfg::default();
  pub static TL_CTX: Ctx = Ctx::new();
}

pub struct CtxCfg {
  //pub default_primary:  Cell<PMachSpec>,
  //pub default_compute:  Cell<PMachSpec>,
  pub swapfile_cap:     Cell<usize>,
  pub gpu_reserve:      Cell<u16>,
  pub gpu_workspace:    Cell<u16>,
  pub _seal:            Cell<bool>,
}

impl Default for CtxCfg {
  fn default() -> CtxCfg {
    println!("DEBUG: CtxCfg::default");
    CtxCfg{
      //default_primary:  Cell::new(PMachSpec::Smp),
      //default_compute:  Cell::new(PMachSpec::Gpu),
      swapfile_cap:     Cell::new(0),
      gpu_reserve:      Cell::new(9001),
      gpu_workspace:    Cell::new(111),
      _seal:            Cell::new(false),
    }
  }
}

/*pub fn ctx_cfg_get_default_primary() -> PMachSpec {
  TL_CTX_CFG.with(|ctx_cfg| ctx_cfg.default_primary.get())
}

pub fn ctx_cfg_set_default_primary(spec: PMachSpec) {
  TL_CTX_CFG.with(|ctx_cfg| {
    if ctx_cfg._seal.get() {
      panic!("bug: cannot set context configuration after context initialization");
    }
    ctx_cfg.default_primary.set(spec)
  })
}

pub fn ctx_cfg_get_default_compute() -> PMachSpec {
  TL_CTX_CFG.with(|ctx_cfg| ctx_cfg.default_compute.get())
}

pub fn ctx_cfg_set_default_compute(spec: PMachSpec) {
  TL_CTX_CFG.with(|ctx_cfg| {
    if ctx_cfg._seal.get() {
      panic!("bug: cannot set context configuration after context initialization");
    }
    ctx_cfg.default_compute.set(spec)
  })
}*/

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
  //pub panick:   PanickCtx,
  pub ctr:      CtxCtr,
  // FIXME
  pub arg:      RefCell<Vec<CellPtr>>,
  //pub out:      Cell<Option<CellPtr>>,
  pub out:      RefCell<Vec<CellPtr>>,
  pub env:      RefCell<CtxEnv>,
  pub thunkenv: RefCell<CtxThunkEnv>,
  pub spine:    RefCell<Spine>,
}

impl Ctx {
  pub fn new() -> Ctx {
    println!("DEBUG: Ctx::new");
    let ctx = Ctx{
      //panick:   PanickCtx::new(),
      ctr:      CtxCtr::new(),
      // FIXME
      arg:      RefCell::new(Vec::new()),
      //out:      Cell::new(None),
      out:      RefCell::new(Vec::new()),
      env:      RefCell::new(CtxEnv::default()),
      thunkenv: RefCell::new(CtxThunkEnv::default()),
      spine:    RefCell::new(Spine::default()),
    };
    TL_CTX_CFG.with(|cfg| cfg._seal.set(true));
    ctx
  }
}

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

pub fn ctx_fresh() -> CellPtr {
  TL_CTX.with(|ctx| {
    ctx.ctr.fresh_cel()
  })
}

pub fn ctx_tmp_fresh() -> CellPtr {
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
}

pub fn ctx_retain(x: CellPtr) {
  TL_CTX.with(|ctx| {
    ctx.env.borrow().retain(x);
  })
}

pub fn ctx_release(x: CellPtr) {
  TL_CTX.try_with(|ctx| {
    ctx.env.borrow().release(x);
  }).unwrap_or(())
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

pub fn ctx_insert(ty: CellType) -> CellPtr {
  TL_CTX.with(|ctx| {
    let x = ctx.ctr.fresh_cel();
    let cel = PCell::new(x, ty.clone());
    ctx.env.borrow_mut().insert_phy(x, ty, cel);
    x
  })
}

/*pub fn ctx_insert_pmach(ty: CellType, primary: Option<PMachSpec>, compute: Option<PMachSpec>) -> CellPtr {
  TL_CTX.with(|ctx| {
    let x = ctx.ctr.fresh_cel();
    let cel = PCell::new_pmach(x, ty.clone(), primary, compute);
    ctx.env.borrow_mut().insert_phy(x, ty, cel);
    x
  })
}*/

pub fn ctx_alias_bits(og: CellPtr, new_dtype: Dtype) -> CellPtr {
  TL_CTX.with(|ctx| {
    let mut env = ctx.env.borrow_mut();
    match env.lookup_ref(og) {
      None => panic!("bug"),
      Some(e) => {
        assert_eq!(new_dtype.size_bytes(), e.ty.dtype.size_bytes());
        let new_ty = CellType{dtype: new_dtype, shape: e.ty.shape.clone()};
        let x = ctx.ctr.fresh_cel();
        env.insert_alias(x, new_ty, og);
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
        let new_ty = CellType{dtype: e.ty.dtype, shape: new_shape};
        let cmp = new_ty.shape_compat(&e.ty);
        assert!(cmp == ShapeCompat::Equal || cmp == ShapeCompat::NewShape);
        let x = ctx.ctr.fresh_cel();
        env.insert_alias(x, new_ty, og);
        x
      }
    }
  })
}

pub fn ctx_lookup_or_insert_gradl(x: CellPtr, tg: CellPtr) -> CellPtr {
  TL_CTX.with(|ctx| {
    ctx.env.borrow_mut().lookup_or_insert_gradr(&ctx.ctr, tg, x)
  })
}

pub fn ctx_lookup_or_insert_gradr(tg: CellPtr, x: CellPtr) -> CellPtr {
  TL_CTX.with(|ctx| {
    ctx.env.borrow_mut().lookup_or_insert_gradr(&ctx.ctr, tg, x)
  })
}

/*pub fn ctx_set_copy_scalar_value<T: ThunkValExt>(x: CellPtr, value: T) {
  TL_CTX.with(|ctx| {
    // FIXME FIXME
    let mut env = ctx.env.borrow_mut();
    let mut thunkenv = ctx.thunkenv.borrow_mut();
    match env.lookup(x) {
      None => panic!("bug"),
      Some(e) => {
        // FIXME
        let mut spine = ctx.spine.borrow_mut();
        spine.intro_aff(x);
        // FIXME FIXME: this should use the pop thunk api.
        let th = ctx.ctr.fresh_thunk();
        let tspec = CopyScalarFutThunkSpec{val: value.into_thunk_val()};
        let tys = vec![ThunkSpec_::out_dim(&tspec, &[]).unwrap()];
        let pthunk = PThunk::new(th, tys, Rc::new(tspec));
        thunkenv.insert0(th, pthunk);
        spine.apply_aff(th, x);
      }
    }
  })
}*/

/*//#[track_caller]
pub fn ctx_set_mem<T: DtypeExt>(x: CellPtr, mem: &[T]) {
  // FIXME FIXME
  unimplemented!();
}*/

/*//#[track_caller]
pub fn ctx_set_profile(x: CellPtr) {
  TL_CTX.with(|ctx| {
    match ctx.env.borrow_mut().lookup_mut(x) {
      None => panic!("bug"),
      Some(e) => {
        e.eflag.set_profile();
      }
    }
  })
}

//#[track_caller]
pub fn ctx_set_trace(x: CellPtr) {
  TL_CTX.with(|ctx| {
    match ctx.env.borrow_mut().lookup_mut(x) {
      None => panic!("bug"),
      Some(e) => {
        e.eflag.set_trace();
      }
    }
  })
}

//#[track_caller]
pub fn ctx_set_break(x: CellPtr) {
  TL_CTX.with(|ctx| {
    match ctx.env.borrow_mut().lookup_mut(x) {
      None => panic!("bug"),
      Some(e) => {
        e.eflag.set_break();
      }
    }
  })
}

//#[track_caller]
pub fn ctx_set_opaque(x: CellPtr) {
  TL_CTX.with(|ctx| {
    match ctx.env.borrow_mut().lookup_mut(x) {
      None => panic!("bug"),
      Some(e) => {
        e.eflag.set_opaque();
      }
    }
  })
}*/

/*
pub fn ctx_yield_val(og: CellPtr) -> CellPtr {
  unimplemented!();
}

pub fn ctx_break_val(og: CellPtr) -> CellPtr {
  TL_CTX.with(|ctx| {
    // FIXME FIXME
    let mut env = ctx.env.borrow_mut();
    match env.lookup(og) {
      None => panic!("bug"),
      Some(e) => {
        let ty = e.ty.clone();
        let x = ctx.ctr.fresh_cel();
        env.insert_alias(x, ty, og);
        // FIXME
        let mut spine = ctx.spine.borrow_mut();
        let sp = spine.curp;
        spine.curp += 1;
        spine.log.push(SpineEntry::BreakV(x, og));
        spine.env.cache.insert(x, sp);
        spine.env.aff.insert(x, sp);
        x
      }
    }
  })
}
*/

pub fn ctx_trace_val(og: CellPtr) -> CellPtr {
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
}

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
        let mut spine = ctx.spine.borrow_mut();
        spine.opaque(x, og);
        x
      }
    }
  })
}

/*pub fn ctx_set_cache(x: CellPtr) {
  TL_CTX.with(|ctx| {
    //match ctx.env.borrow().lookup(x) {}
    let mut env = ctx.env.borrow_mut();
    match env.lookup_mut(x) {
      None => panic!("bug"),
      Some(e) => {
        // FIXME
        let mut spine = ctx.spine.borrow_mut();
        let sp = spine.curp;
        spine.curp += 1;
        spine.log.push(SpineEntry::CacheAff(x));
        spine.env.cache.insert(x, sp);
        spine.env.aff.insert(x, sp);
      }
    }
  })
}*/

/*pub fn ctx_init_cache(x: CellPtr) {
  TL_CTX.with(|ctx| {
    //match ctx.env.borrow().lookup(x) {}
    match ctx.env.borrow_mut().lookup_mut(x) {
      None => panic!("bug"),
      Some(e) => {
        //e.cel.flag.borrow_mut().set_cache();
        let flag = !e.cel.flag.set_cache();
        let mode = match e.cel.mode.set_mux() {
          Err(_) => panic!("bug"),
          Ok(prev) => !prev
        };
        // FIXME
        if !flag & !mode {
        } else if flag & mode {
          // FIXME
          let mut spine = ctx.spine.borrow_mut();
          let sp = spine.curp;
          spine.curp += 1;
          spine.log.push(SpineEntry::ICacheMux(x));
          spine.env.cache.insert(x, sp);
          spine.env.mux.insert(x, sp);
        } else {
          panic!("bug");
        }
      }
    }
  })
}*/

/*pub fn ctx_set_seal(x: CellPtr) {
  TL_CTX.with(|ctx| {
    //match ctx.env.borrow().lookup(x) {}
    match ctx.env.borrow_mut().lookup_mut(x) {
      None => panic!("bug"),
      Some(e) => {
        //e.cel.flag.borrow_mut().set_eval();
        if !e.cel.flag.set_seal() {
          let mut spine = ctx.spine.borrow_mut();
          let sp = spine.curp;
          spine.curp += 1;
          spine.log.push(SpineEntry::SealMux(x));
          spine.env.seal.insert(x, sp);
        }
      }
    }
  })
}*/

/*pub fn ctx_set_unseal(x: CellPtr) {
  TL_CTX.with(|ctx| {
    //match ctx.env.borrow().lookup(x) {}
    match ctx.env.borrow_mut().lookup_mut(x) {
      None => panic!("bug"),
      Some(e) => {
        //e.cel.flag.borrow_mut().set_eval();
        if !e.cel.flag.unset_seal() {
          let mut spine = ctx.spine.borrow_mut();
          match spine.env.seal.get(&x) {
            None => {}
            Some(&old_sp) => {
              let sp = spine.curp;
              spine.curp += 1;
              assert!(old_sp < sp);
              spine.log.push(SpineEntry::UnsealMux(x));
              spine.env.seal.insert(x, sp);
            }
          }
        }
      }
    }
  })
}*/

/*pub fn ctx_set_eval(x: CellPtr) {
  TL_CTX.with(|ctx| {
    //match ctx.env.borrow().lookup(x) {}
    match ctx.env.borrow_mut().lookup_mut(x) {
      None => panic!("bug"),
      Some(e) => {
        //e.cel.flag.borrow_mut().set_eval();
        if !e.cel.flag.set_eval() {
          let mut spine = ctx.spine.borrow_mut();
          let sp = spine.curp;
          spine.curp += 1;
          spine.log.push(SpineEntry::Eval(x));
          spine.env.eval.insert(x, sp);
        }
      }
    }
  })
}*/

pub fn ctx_clean_arg() -> bool {
  TL_CTX.with(|ctx| {
    ctx.arg.borrow().is_empty() &&
    ctx.out.borrow().is_empty()
  })
}

pub fn ctx_push_cell_arg(x: CellPtr) {
  TL_CTX.with(|ctx| {
    ctx.arg.borrow_mut().push(x)
  })
}

pub fn ctx_push_cell_out(x: CellPtr) {
  TL_CTX.with(|ctx| {
    /*assert!(ctx.out.get().is_none());
    ctx.out.set(Some(x));*/
    ctx.out.borrow_mut().push(x)
  })
}

pub fn ctx_push_cell_tmp_out() {
  TL_CTX.with(|ctx| {
    //assert!(ctx.out.get().is_none());
    let x = ctx.ctr.fresh_tmp();
    //ctx.out.set(Some(x));
    ctx.out.borrow_mut().push(x)
  })
}

pub fn ctx_pop_thunk<Th: ThunkSpec_ + 'static>(th: Th) -> CellPtr {
  TL_CTX.with(|ctx| {
    let mut tp_ = None;
    let mut thunkenv = ctx.thunkenv.borrow_mut();
    // FIXME FIXME: we also need out dim.
    let mut tys = Vec::with_capacity(ctx.arg.borrow().len());
    let mut tys_ = Vec::with_capacity(ctx.arg.borrow().len());
    for &arg in ctx.arg.borrow().iter() {
      /*let ty_ = ctx_lookup_type(arg);*/
      let ty_ = match ctx.env.borrow().lookup_ref(arg) {
        None => panic!("bug"),
        Some(e) => e.ty.clone()
      };
      tys.push(ty_.to_dim());
      tys_.push(ty_);
    }
    let (ar_in, ar_out) = th.arity();
    let oty = match th.out_dim(&tys) {
      Err(_) => panic!("BUG: type error"),
      Ok(ty) => ty
    };
    let oty_ = match th.out_ty_(&tys_) {
      Err(_) => panic!("BUG: type error"),
      Ok(ty_) => ty_
    };
    tys.push(oty);
    let tk = ThunkKey(Rc::new(th));
    let key = (tk, ar_in, ar_out, tys);
    match thunkenv.thunkidx.get(&key) {
      None => {}
      Some(&tp) => {
        match thunkenv.thunktab.get(&tp) {
          None => {
            // NB: this might happen due to thunk gc.
          }
          Some(te) => {
            assert!(((key.0).0).thunk_eq(&*te.thunk.spec_).unwrap_or(false));
            tp_ = Some(tp);
          }
        }
      }
    }
    let (tk, ar_in, ar_out, tys) = key;
    if tp_.is_none() {
      let tp = ctx.ctr.fresh_thunk();
      let mut args = Vec::new();
      swap(&mut args, &mut *ctx.arg.borrow_mut());
      let pt = PThunk::new(tp, tys.clone(), (tk.0).clone());
      let te = ThunkEnvEntry{args, thunk: pt};
      thunkenv.thunkidx.insert((tk, ar_in, ar_out, tys), tp);
      thunkenv.thunktab.insert(tp, te);
      tp_ = Some(tp);
    }
    drop(thunkenv);
    let tp = tp_.unwrap();
    let x = ctx.ctr.fresh_cel();
    // FIXME FIXME: actually, prefer to insert a Top ref.
    /*let cel = PCell::new(x, oty_.clone());
    ctx.env.borrow_mut().insert_phy(x, oty_, cel);*/
    ctx.env.borrow_mut().insert_top(x, oty_);
    let mut spine = ctx.spine.borrow_mut();
    spine.intro_aff(x);
    spine.apply_aff(tp, x);
    x
  })
}

pub fn ctx_pop_thunk_mux<Th: ThunkSpec_ + 'static>(th: Th, out: CellPtr) {
  // FIXME FIXME
  unimplemented!();
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
  pub tmp_ctr:  Cell<i32>,
}

impl CtxCtr {
  pub fn new() -> CtxCtr {
    CtxCtr{
      ptr_ctr:  Cell::new(0),
      tmp_ctr:  Cell::new(0),
    }
  }
}

impl CtxCtr {
  pub fn fresh_cel(&self) -> CellPtr {
    let next = self.ptr_ctr.get() + 1;
    assert!(next > 0);
    assert!(next <= i32::max_value());
    self.ptr_ctr.set(next);
    CellPtr::from_unchecked(next)
  }

  pub fn fresh_thunk(&self) -> ThunkPtr {
    let next = self.ptr_ctr.get() + 1;
    assert!(next > 0);
    assert!(next <= i32::max_value());
    self.ptr_ctr.set(next);
    ThunkPtr::from_unchecked(next)
  }

  pub fn reset_tmp(&self) {
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
  }
}

pub struct ThunkEnvEntry {
  // FIXME
  pub args:     Vec<CellPtr>,
  pub thunk:    PThunk,
}

#[derive(Default)]
pub struct CtxThunkEnv {
  // FIXME
  pub thunktab: HashMap<ThunkPtr, ThunkEnvEntry>,
  pub thunkidx: HashMap<(ThunkKey, u16, u16, Vec<Dim>), ThunkPtr>,
}

impl CtxThunkEnv {
  pub fn insert0(&mut self, th: ThunkPtr, thunk: PThunk) {
    // FIXME FIXME
    match self.thunktab.get(&th) {
      None => {}
      Some(_) => panic!("bug")
    }
    let e = ThunkEnvEntry{args: Vec::new(), thunk};
    self.thunktab.insert(th, e);
  }

  pub fn insert(&mut self, th: ThunkPtr, args: Vec<CellPtr>, thunk: PThunk) {
    // FIXME FIXME
    match self.thunktab.get(&th) {
      None => {}
      Some(_) => panic!("bug")
    }
    let e = ThunkEnvEntry{args, thunk};
    self.thunktab.insert(th, e);
  }

  pub fn gc(&mut self, gc_list: &[CellPtr]) {
    // FIXME FIXME: remove updating thunks.
  }
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

  pub fn set_opaque(&mut self) {
    self.bits |= 1;
  }

  pub fn opaque(&self) -> bool {
    (self.bits & 1) != 0
  }

  pub fn set_profile(&mut self) {
    self.bits |= 0x20;
  }

  pub fn profile(&self) -> bool {
    (self.bits & 0x20) != 0
  }

  pub fn set_trace(&mut self) {
    self.bits |= 0x40;
  }

  pub fn trace(&self) -> bool {
    (self.bits & 0x40) != 0
  }

  pub fn set_break(&mut self) {
    self.bits |= 0x80;
  }

  pub fn break_(&self) -> bool {
    (self.bits & 0x80) != 0
  }
}

pub enum Cell_ {
  // FIXME FIXME
  Top(RefCell<CellState>, CellPtr),
  Phy(RefCell<CellState>, PCell),
  Cow(RefCell<CellState>, CowCell),
  Alias(CellPtr),
  Bot,
}

impl Cell_ {
  pub fn state_mut(&self) -> RefMut<CellState> {
    match self {
      &Cell_::Top(ref state, _) => state.borrow_mut(),
      &Cell_::Phy(ref state, _) => state.borrow_mut(),
      &Cell_::Cow(ref state, _) => state.borrow_mut(),
      _ => panic!("bug")
    }
  }

  pub fn swap_in(&mut self, state: RefCell<CellState>, p: PCell) {
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
  }
}

pub struct CellEnvEntry {
  pub stablect: Cell<usize>,
  pub ty:       CellType,
  // FIXME FIXME
  /*pub lay:      CellLayout,
  pub mode:     CellMode,
  pub flag:     CellFlag,*/
  //pub clk:      Clock,
  //pub ithunk:   Option<PThunk>,
  //pub thunk:    Vec<PThunk>,
  pub ithunk:   Option<ThunkPtr>,
  pub thunk:    Vec<ThunkPtr>,
  pub eflag:    CellEFlag,
  //pub cel:      PCell,
  pub cel_:     Cell_,
}

pub struct CellEnvEntryRef<'a> {
  pub stablect: &'a Cell<usize>,
  pub ty:       &'a CellType,
  pub ithunk:   Option<ThunkPtr>,
  pub thunk:    &'a [ThunkPtr],
  pub eflag:    CellEFlag,
  pub cel:      &'a PCell,
}

pub struct CellEnvEntryRef_<'a> {
  pub stablect: &'a Cell<usize>,
  pub ty:       &'a CellType,
  pub ithunk:   Option<ThunkPtr>,
  pub thunk:    &'a [ThunkPtr],
  pub eflag:    CellEFlag,
  pub cel_:     &'a Cell_,
}

impl<'a> CellEnvEntryRef_<'a> {
  pub fn state(&self) -> RefMut<CellState> {
    self.cel_.state_mut()
  }
}

pub struct CellEnvEntryMut<'a> {
  pub stablect: &'a Cell<usize>,
  pub ty:       CellType,
  pub ithunk:   &'a mut Option<ThunkPtr>,
  pub thunk:    &'a mut Vec<ThunkPtr>,
  pub eflag:    &'a mut CellEFlag,
  pub cel:      &'a mut PCell,
}

pub struct CellEnvEntryMutRef<'a> {
  pub stablect: &'a Cell<usize>,
  pub ty:       CellType,
  // FIXME FIXME
  //pub mode:     &'a mut CellMode,
  //pub flag:     &'a mut CellFlag,
  //pub clk:      &'a mut CellFlag,
  pub ithunk:   &'a mut Option<ThunkPtr>,
  pub thunk:    &'a mut Vec<ThunkPtr>,
  pub eflag:    &'a mut CellEFlag,
  pub cel_:     &'a mut Cell_,
}

impl<'a> CellEnvEntryMutRef<'a> {
  pub fn state(&self) -> RefMut<CellState> {
    self.cel_.state_mut()
  }
}

#[derive(Default)]
pub struct CtxEnv {
  // FIXME
  pub tmptab:   HashMap<CellPtr, CellPtr>,
  pub celtab:   HashMap<CellPtr, CellEnvEntry>,
  pub root:     RefCell<HashMap<CellPtr, CellPtr>>,
  pub gradr:    HashMap<[CellPtr; 2], CellPtr>,
  pub ungradr:  HashMap<[CellPtr; 2], CellPtr>,
  pub tag:      HashMap<CellPtr, HashSet<String>>,
  //pub tag:      HashMap<CellPtr, Vec<String>>,
}

impl CtxEnv {
  pub fn retain(&self, x: CellPtr) {
    match self.lookup_ref(x) {
      None => panic!("bug"),
      Some(e) => {
        let cur = e.stablect.get();
        assert!(cur != usize::max_value());
        let next = cur + 1;
        e.stablect.set(next);
      }
    }
  }

  pub fn release(&self, x: CellPtr) {
    match self.lookup_ref(x) {
      None => panic!("bug"),
      Some(e) => {
        let cur = e.stablect.get();
        assert!(cur != usize::max_value());
        let next = cur - 1;
        e.stablect.set(next);
      }
    }
  }

  /*//pub fn lookup(&self, x: CellPtr) -> (CellType, Option<&PCell>) {}
  pub fn lookup(&self, x: CellPtr) -> Option<CellEnvEntryRef> {
    let ty = match self.celtab.get(&x) {
      None => panic!("bug"),
      Some(e) => {
        match &e.cel_ {
          &Cell_::Top => {
            unimplemented!();
          }
          &Cell_::Phy(ref cel) => {
            assert_eq!(x, cel.optr);
            return Some(CellEnvEntryRef{
              stablect: &e.stablect,
              ty: &e.ty,
              ithunk: e.ithunk,
              thunk: &e.thunk,
              eflag: e.eflag,
              cel,
            });
          }
          &Cell_::Cow(_) => {
            // FIXME FIXME
            unimplemented!();
          }
          &Cell_::Alias(_) => {}
          &Cell_::Bot => {
            // FIXME FIXME
            panic!("bug");
            //return None;
          }
        }
        &e.ty
      }
    };
    let mut p = x;
    let mut root = self.root.borrow_mut();
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
    match self.celtab.get(&p) {
      None => panic!("bug"),
      Some(e) => {
        match &e.cel_ {
          &Cell_::Top => {
            unimplemented!();
          }
          &Cell_::Phy(ref cel) => {
            // FIXME FIXME
            if e.ty.dtype != Dtype::_Top {
              assert_eq!(ty.dtype.size_bytes(), e.ty.dtype.size_bytes());
              assert!(ty.shape_compat(&e.ty) != ShapeCompat::Incompat);
            }
            assert_eq!(p, cel.optr);
            return Some(CellEnvEntryRef{
              stablect: &e.stablect,
              ty,
              ithunk: e.ithunk,
              thunk: &e.thunk,
              eflag: e.eflag,
              cel,
            });
          }
          &Cell_::Cow(_) => {
            // FIXME FIXME
            unimplemented!();
          }
          &Cell_::Alias(_) => {
            panic!("bug");
          }
          &Cell_::Bot => {
            // FIXME FIXME
            panic!("bug");
            //return None;
          }
        }
      }
    }
  }*/

  pub fn lookup_ref(&self, x: CellPtr) -> Option<CellEnvEntryRef_> {
    let ty = match self.celtab.get(&x) {
      None => panic!("bug"),
      Some(e) => {
        match &e.cel_ {
          &Cell_::Phy(_, ref cel) => {
            assert_eq!(x, cel.optr);
          }
          _ => {}
        }
        match &e.cel_ {
          &Cell_::Top(..) |
          &Cell_::Phy(..) |
          &Cell_::Cow(..) |
          &Cell_::Bot => {
            return Some(CellEnvEntryRef_{
              stablect: &e.stablect,
              ty: &e.ty,
              ithunk: e.ithunk,
              thunk: &e.thunk,
              eflag: e.eflag,
              cel_: &e.cel_,
            });
          }
          &Cell_::Alias(_) => {}
        }
        &e.ty
      }
    };
    let mut p = x;
    let mut root = self.root.borrow_mut();
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
    match self.celtab.get(&p) {
      None => panic!("bug"),
      Some(e) => {
        match &e.cel_ {
          &Cell_::Top(_, optr) => {
            assert_eq!(p, optr);
          }
          &Cell_::Phy(_, ref cel) => {
            assert_eq!(p, cel.optr);
          }
          &Cell_::Cow(_, ref cel) => {
            assert_eq!(p, cel.optr);
          }
          &Cell_::Bot => {}
          &Cell_::Alias(_) => {
            panic!("bug");
          }
        }
        // FIXME FIXME
        if e.ty.dtype != Dtype::_Top {
          assert_eq!(ty.dtype.size_bytes(), e.ty.dtype.size_bytes());
          assert!(ty.shape_compat(&e.ty) != ShapeCompat::Incompat);
        }
        Some(CellEnvEntryRef_{
          stablect: &e.stablect,
          ty,
          ithunk: e.ithunk,
          thunk: &e.thunk,
          eflag: e.eflag,
          cel_: &e.cel_,
        })
      }
    }
  }

  /*pub fn lookup_mut(&mut self, x: CellPtr) -> Option<CellEnvEntryMut> {
    let mut noalias = false;
    let ty = match self.celtab.get(&x) {
      None => panic!("bug"),
      Some(e) => {
        match &e.cel_ {
          &Cell_::Top => {
            unimplemented!();
          }
          &Cell_::Phy(ref cel) => {
            noalias = true;
          }
          &Cell_::Cow(_) => {
            // FIXME FIXME
            unimplemented!();
          }
          &Cell_::Alias(_) => {}
          &Cell_::Bot => {
            // FIXME FIXME
            panic!("bug");
            //return None;
          }
        }
        e.ty.clone()
      }
    };
    let mut p = x;
    if !noalias {
      let mut root = self.root.borrow_mut();
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
    }
    match self.celtab.get_mut(&p) {
      None => panic!("bug"),
      Some(e) => {
        match &mut e.cel_ {
          &mut Cell_::Top => {
            unimplemented!();
          }
          &mut Cell_::Phy(ref mut cel) => {
            // FIXME FIXME
            if e.ty.dtype != Dtype::_Top {
              assert_eq!(ty.dtype.size_bytes(), e.ty.dtype.size_bytes());
              assert!(ty.shape_compat(&e.ty) != ShapeCompat::Incompat);
            }
            assert_eq!(p, cel.optr);
            return Some(CellEnvEntryMut{
              stablect: &e.stablect,
              ty,
              ithunk: &mut e.ithunk,
              thunk: &mut e.thunk,
              eflag: &mut e.eflag,
              cel,
            });
          }
          &mut Cell_::Cow(_) => {
            // FIXME FIXME
            unimplemented!();
          }
          &mut Cell_::Alias(_) => {
            panic!("bug");
          }
          &mut Cell_::Bot => {
            // FIXME FIXME
            panic!("bug");
            //return None;
          }
        }
      }
    }
  }*/

  pub fn lookup_mut_ref(&mut self, x: CellPtr) -> Option<CellEnvEntryMutRef> {
    let mut noalias = false;
    let ty = match self.celtab.get(&x) {
      None => panic!("bug"),
      Some(e) => {
        match &e.cel_ {
          &Cell_::Top(..) |
          &Cell_::Phy(..) |
          &Cell_::Cow(..) |
          &Cell_::Bot => {
            noalias = true;
          }
          &Cell_::Alias(_) => {}
        }
        e.ty.clone()
      }
    };
    let mut p = x;
    if !noalias {
      let mut root = self.root.borrow_mut();
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
    }
    match self.celtab.get_mut(&p) {
      None => panic!("bug"),
      Some(e) => {
        match &e.cel_ {
          &Cell_::Top(_, optr) => {
            assert_eq!(p, optr);
          }
          &Cell_::Phy(_, ref cel) => {
            assert_eq!(p, cel.optr);
          }
          &Cell_::Cow(_, ref cel) => {
            assert_eq!(p, cel.optr);
          }
          &Cell_::Bot => {}
          &Cell_::Alias(_) => {
            panic!("bug");
          }
        }
        if e.ty.dtype != Dtype::_Top {
          assert_eq!(ty.dtype.size_bytes(), e.ty.dtype.size_bytes());
          assert!(ty.shape_compat(&e.ty) != ShapeCompat::Incompat);
        }
        return Some(CellEnvEntryMutRef{
          stablect: &e.stablect,
          ty,
          ithunk: &mut e.ithunk,
          thunk: &mut e.thunk,
          eflag: &mut e.eflag,
          cel_: &mut e.cel_,
        });
      }
    }
  }

  pub fn insert_top(&mut self, x: CellPtr, ty: CellType) {
    match self.celtab.get(&x) {
      None => {}
      Some(_) => panic!("bug")
    }
    //let lay = CellLayout::new_packed(&ty);
    let e = CellEnvEntry{
      stablect: Cell::new(0),
      ty,
      /*lay,
      mode:     CellMode::default(),
      flag:     CellFlag::default(),*/
      //clk:      Clock::default(),
      ithunk:   None,
      thunk:    Vec::new(),
      eflag:    CellEFlag::default(),
      cel_:     Cell_::Top(RefCell::new(CellState::default()), x),
    };
    self.celtab.insert(x, e);
  }

  pub fn insert_phy(&mut self, x: CellPtr, ty: CellType, cel: PCell) {
    match self.celtab.get(&x) {
      None => {}
      Some(_) => panic!("bug")
    }
    //let lay = CellLayout::new_packed(&ty);
    let e = CellEnvEntry{
      stablect: Cell::new(0),
      ty,
      /*lay,
      mode:     CellMode::default(),
      flag:     CellFlag::default(),*/
      //clk:      Clock::default(),
      ithunk:   None,
      thunk:    Vec::new(),
      eflag:    CellEFlag::default(),
      cel_:     Cell_::Phy(RefCell::new(CellState::default()), cel),
    };
    self.celtab.insert(x, e);
  }

  pub fn insert_cow(&mut self, x: CellPtr, ty: CellType, og: CellPtr) {
    match self.celtab.get(&x) {
      None => {}
      Some(_) => panic!("bug")
    }
    //let lay = CellLayout::new_packed(&ty);
    let e = CellEnvEntry{
      stablect: Cell::new(0),
      ty,
      /*lay,
      mode:     CellMode::default(),
      flag:     CellFlag::default(),*/
      //clk:      Clock::default(),
      ithunk:   None,
      thunk:    Vec::new(),
      eflag:    CellEFlag::default(),
      cel_:     Cell_::Cow(RefCell::new(CellState::default()), CowCell{optr: x, pcel: og}),
    };
    self.celtab.insert(x, e);
  }

  pub fn insert_alias(&mut self, x: CellPtr, ty: CellType, og: CellPtr) {
    match self.celtab.get(&x) {
      None => {}
      Some(_) => panic!("bug")
    }
    //let lay = CellLayout::new_packed(&ty);
    let e = CellEnvEntry{
      stablect: Cell::new(usize::max_value()),
      ty,
      /*lay,
      mode:     CellMode::default(),
      flag:     CellFlag::default(),*/
      eflag:    CellEFlag::default(),
      //clk:      Clock::default(),
      ithunk:   None,
      thunk:    Vec::new(),
      cel_:     Cell_::Alias(og),
    };
    self.celtab.insert(x, e);
    let mut root = self.root.borrow_mut();
    match root.get(&x) {
      None => {}
      Some(_) => panic!("bug")
    }
    root.insert(x, og);
  }

  pub fn lookup_or_insert_gradr(&mut self, ctr: &CtxCtr, tg: CellPtr, x: CellPtr) -> CellPtr {
    match self.gradr.get(&[tg, x]) {
      None => {
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

  pub fn reset_tmp(&mut self) {
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
          &mut Cell_::Phy(ref state, ref mut pcel) => {
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
              let mut root = self.root.borrow_mut();
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
  }

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
    let mut root = self.root.borrow_mut();
    for &k in gc_list.iter() {
      // FIXME FIXME: remove from other indexes.
      root.remove(&k);
      self.celtab.remove(&k);
    }
  }
}
