use crate::cell::*;
use crate::ptr::*;
use crate::spine::*;
use crate::thunk::*;

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};

thread_local! {
  static TL_CTX: Ctx = Ctx::default();
}

pub struct Ctx {
  default_primary:  Cell<PMachSpec>,
  default_compute:  Cell<PMachSpec>,
  swap_cap:         Cell<usize>,
  gpu_reserve:      Cell<u16>,
  gpu_workspace:    Cell<u16>,
  ptr_ctr:  Cell<i32>,
  tmp_ctr:  Cell<i32>,
  // FIXME
  arg:      RefCell<Vec<CellPtr>>,
  out:      Cell<Option<CellPtr>>,
  env:      RefCell<CtxEnv>,
  spine:    RefCell<Spine>,
}

impl Default for Ctx {
  fn default() -> Ctx {
    Ctx{
      default_primary:  Cell::new(PMachSpec::Cpu),
      default_compute:  Cell::new(PMachSpec::Cpu),
      swap_cap:         Cell::new(0),
      gpu_reserve:      Cell::new(9001),
      gpu_workspace:    Cell::new(111),
      ptr_ctr:  Cell::new(0),
      tmp_ctr:  Cell::new(0),
      // FIXME
      arg:      RefCell::new(Vec::new()),
      out:      Cell::new(None),
      env:      RefCell::new(CtxEnv::default()),
      spine:    RefCell::new(Spine::default()),
    }
  }
}

pub fn tl_ctx_get_default_primary() -> PMachSpec {
  TL_CTX.with(|ctx| ctx.default_primary.get())
}

pub fn tl_ctx_set_default_primary(spec: PMachSpec) {
  TL_CTX.with(|ctx| ctx.default_primary.set(spec))
}

pub fn tl_ctx_get_default_compute() -> PMachSpec {
  TL_CTX.with(|ctx| ctx.default_compute.get())
}

pub fn tl_ctx_set_default_compute(spec: PMachSpec) {
  TL_CTX.with(|ctx| ctx.default_compute.set(spec))
}

pub fn tl_ctx_get_swapfile_max_bytes() -> usize {
  TL_CTX.with(|ctx| ctx.swap_cap.get())
}

pub fn tl_ctx_set_swapfile_max_bytes(sz: usize) {
  TL_CTX.with(|ctx| ctx.swap_cap.set(sz))
}

pub fn tl_ctx_get_gpu_reserve_mem_per_10k() -> u16 {
  TL_CTX.with(|ctx| ctx.gpu_reserve.get())
}

pub fn tl_ctx_set_gpu_reserve_mem_per_10k(m: u16) {
  if m <= 0 {
    panic!("bug: gpu reserve too small: {}/10000", m);
  }
  if m >= 10000 {
    panic!("bug: gpu reserve too big: {}/10000", m);
  }
  TL_CTX.with(|ctx| ctx.gpu_reserve.set(m))
}

pub fn tl_ctx_get_gpu_workspace_mem_per_10k() -> u16 {
  TL_CTX.with(|ctx| ctx.gpu_workspace.get())
}

pub fn tl_ctx_set_gpu_workspace_mem_per_10k(m: u16) {
  if m <= 0 {
    panic!("bug: gpu workspace too small: {}/10000", m);
  }
  if m >= 10000 {
    panic!("bug: gpu workspace too big: {}/10000", m);
  }
  TL_CTX.with(|ctx| ctx.gpu_workspace.set(m))
}

pub fn tl_ctx_fresh() -> UnstablePtr {
  TL_CTX.with(|ctx| {
    let next = ctx.ptr_ctr.get() + 1;
    assert!(next > 0);
    assert!(next <= i32::max_value());
    ctx.ptr_ctr.set(next);
    UnstablePtr::from_unchecked(next)
  })
}

pub fn tl_ctx_fresh_tmp() -> UnstablePtr {
  TL_CTX.with(|ctx| {
    let next = ctx.tmp_ctr.get() - 1;
    assert!(next < 0);
    assert!(next > i32::min_value());
    ctx.tmp_ctr.set(next);
    UnstablePtr::from_unchecked(next)
  })
}

pub fn tl_ctx_reset_tmp() {
  unimplemented!();
}

pub fn tl_ctx_alias(x: CellPtr, y: CellPtr) {
  unimplemented!();
}

pub fn tl_ctx_lookup_type(x: CellPtr) -> CellType {
  TL_CTX.with(|ctx| {
    match ctx.env.borrow().celtab.get(&x) {
      None => panic!("bug"),
      Some(e) => e.cel.ty.clone()
    }
  })
}

pub fn tl_ctx_lookup_grad(tg: CellPtr, x: CellPtr) -> CellPtr {
  unimplemented!();
}

pub fn tl_ctx_clean_arg(x: CellPtr) -> bool {
  TL_CTX.with(|ctx| {
    ctx.arg.borrow().is_empty() &&
    ctx.out.get().is_none()
  })
}

pub fn tl_ctx_push_cell_arg(x: CellPtr) {
  TL_CTX.with(|ctx| ctx.arg.borrow_mut().push(x))
}

pub fn tl_ctx_push_cell_out(x: CellPtr) {
  TL_CTX.with(|ctx| {
    assert!(ctx.out.get().is_none());
    ctx.out.set(Some(x));
  })
}

pub fn tl_ctx_pop_thunk(th: ()) -> UnstablePtr {
  unimplemented!();
  /*
  TL_CTX.with(|ctx| {
    let mut arg = ctx.arg.borrow_mut();
    // TODO
    arg.clear();
    ctx.out.set(None);
    // FIXME
  })
  */
}

pub fn tl_ctx_reset() {
  unimplemented!();
}

pub fn tl_ctx_reduce() {
  unimplemented!();
}

pub fn tl_ctx_resume() {
  unimplemented!();
}

pub fn tl_ctx_wait() {
  unimplemented!();
}

/*pub fn tl_ctx_fresh_ptr() -> StablePtr {
  TL_CTX.with(|ctx| {
    let next = ctx.ptr_ctr.get() + 1;
    assert!(next >= 0);
    assert!(next < u32::max_value());
    ctx.ptr_ctr.set(next);
    StablePtr(next)
  })
}*/

/*pub struct ThunkTabEntry {
}*/

pub enum CellTabRef {
  Top,
  P(PCell),
  Alias(CellPtr),
  Bot,
}

pub struct CellTabEntry {
  //pub state:    CellState,
  pub ty:       CellType,
  //pub ithunk:   Option<PThunk>,
  //pub thunk:    Vec<PThunk>,
  pub ithunk:   Option<ThunkPtr>,
  pub thunk:    Vec<ThunkPtr>,
  pub cel:      PCell,
  //pub cel:      CellTabRef,
}

#[derive(Default)]
pub struct CtxEnv {
  // FIXME
  pub thunktab: HashMap<ThunkPtr, PThunk>,
  pub celtab:   HashMap<CellPtr, CellTabEntry>,
  // FIXME FIXME
  pub root:     HashMap<CellPtr, Cell<CellPtr>>,
  //pub root:     RefCell<HashMap<CellPtr, CellPtr>>,
  pub stable:   HashSet<CellPtr>,
  pub unstable: HashSet<CellPtr>,
  pub cache:    HashSet<CellPtr>,
  pub grad:     HashMap<[CellPtr; 2], CellPtr>,
  pub ungrad:   HashMap<[CellPtr; 2], CellPtr>,
  pub tag:      HashMap<CellPtr, HashSet<String>>,
  //pub tag:      HashMap<CellPtr, Vec<String>>,
}

impl CtxEnv {
  //pub fn lookup(&self, x: CellPtr) -> (CellType, Option<&PCell>) {}
  pub fn lookup(&self, x: CellPtr) -> Option<(&CellType, &PCell)> {
    unimplemented!();
    /*
    let ty = match self.celtab.get(&x) {
      None => panic!("bug"),
      Some(e) => {
        match &e.ref_ {
          &CellTabRef::Top => {
            unimplemented!();
          }
          &CellTabRef::P(ref cel) => {
            assert_eq!(p, cel.ptr.get());
            return Some((&e.ty, cel));
          }
          &CellTabRef::Alias(_) => {}
          &CellTabRef::Bot => {
            return None;
          }
        }
        &e.ty
      }
    };
    let mut p = x;
    loop {
      match self.root.get(&p) {
        None => {
          break;
        }
        Some(&q) => {
          p = q.get();
        }
      }
    }
    match self.celtab.get(&p) {
      None => panic!("bug"),
      Some(e) => {
        match &e.ref_ {
          &CellTabRef::Top => {
            unimplemented!();
          }
          &CellTabRef::P(ref cel) => {
            //assert!(ty.alias_compat(&e.ty));
            assert_eq!(p, cel.ptr.get());
            return Some((ty, cel));
          }
          &CellTabRef::Alias(_) => {
            panic!("bug");
          }
          &CellTabRef::Bot => {
            return None;
          }
        }
      }
    }
    */
  }
}
