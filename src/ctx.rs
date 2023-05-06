use crate::cell::*;
use crate::ptr::*;
use crate::spine::*;
use crate::thunk::*;

use futhark_ffi::blake2s::{Blake2s};

use std::any::{Any, TypeId};
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::hash::{Hash};

thread_local! {
  static TL_CTX: Ctx = Ctx::default();
}

pub struct Ctx {
  pub default_primary:  Cell<PMachSpec>,
  pub default_compute:  Cell<PMachSpec>,
  pub swapfile_cap:     Cell<usize>,
  pub gpu_reserve:      Cell<u16>,
  pub gpu_workspace:    Cell<u16>,
  pub dev:      Cell<i32>,
  pub ptr_ctr:  Cell<i32>,
  pub tmp_ctr:  Cell<i32>,
  // FIXME
  pub arg:      RefCell<Vec<CellPtr>>,
  pub out:      Cell<Option<CellPtr>>,
  pub env:      RefCell<CtxEnv>,
  pub thunkenv: RefCell<CtxThunkEnv>,
  pub spine:    RefCell<Spine>,
}

impl Default for Ctx {
  fn default() -> Ctx {
    Ctx{
      default_primary:  Cell::new(PMachSpec::Smp),
      default_compute:  Cell::new(PMachSpec::Gpu),
      swapfile_cap:     Cell::new(0),
      gpu_reserve:      Cell::new(9001),
      gpu_workspace:    Cell::new(111),
      dev:      Cell::new(0),
      ptr_ctr:  Cell::new(0),
      tmp_ctr:  Cell::new(0),
      // FIXME
      arg:      RefCell::new(Vec::new()),
      out:      Cell::new(None),
      env:      RefCell::new(CtxEnv::default()),
      thunkenv: RefCell::new(CtxThunkEnv::default()),
      spine:    RefCell::new(Spine::default()),
    }
  }
}

impl Ctx {
  pub fn fresh(&self) -> CellPtr {
    let next = self.ptr_ctr.get() + 1;
    assert!(next > 0);
    assert!(next <= i32::max_value());
    self.ptr_ctr.set(next);
    CellPtr::from_unchecked(next)
  }

  pub fn fresh_tmp(&self) -> CellPtr {
    let next = self.tmp_ctr.get() - 1;
    assert!(next < 0);
    assert!(next > i32::min_value());
    self.tmp_ctr.set(next);
    CellPtr::from_unchecked(next)
  }
}

pub fn ctx_unwrap<F: FnMut(&Ctx) -> X, X>(f: &mut F) -> X {
  TL_CTX.with(f)
}

pub fn ctx_get_default_primary() -> PMachSpec {
  TL_CTX.with(|ctx| ctx.default_primary.get())
}

pub fn ctx_set_default_primary(spec: PMachSpec) {
  TL_CTX.with(|ctx| ctx.default_primary.set(spec))
}

pub fn ctx_get_default_compute() -> PMachSpec {
  TL_CTX.with(|ctx| ctx.default_compute.get())
}

pub fn ctx_set_default_compute(spec: PMachSpec) {
  TL_CTX.with(|ctx| ctx.default_compute.set(spec))
}

pub fn ctx_get_swapfile_max_bytes() -> usize {
  TL_CTX.with(|ctx| ctx.swapfile_cap.get())
}

pub fn ctx_set_swapfile_max_bytes(sz: usize) {
  TL_CTX.with(|ctx| ctx.swapfile_cap.set(sz))
}

pub fn ctx_get_gpu_reserve_mem_per_10k() -> u16 {
  TL_CTX.with(|ctx| ctx.gpu_reserve.get())
}

pub fn ctx_set_gpu_reserve_mem_per_10k(m: u16) {
  if m <= 0 {
    panic!("bug: gpu reserve too small: {}/10000", m);
  }
  if m >= 10000 {
    panic!("bug: gpu reserve too big: {}/10000", m);
  }
  TL_CTX.with(|ctx| ctx.gpu_reserve.set(m))
}

pub fn ctx_get_gpu_workspace_mem_per_10k() -> u16 {
  TL_CTX.with(|ctx| ctx.gpu_workspace.get())
}

pub fn ctx_set_gpu_workspace_mem_per_10k(m: u16) {
  if m <= 0 {
    panic!("bug: gpu workspace too small: {}/10000", m);
  }
  if m >= 10000 {
    panic!("bug: gpu workspace too big: {}/10000", m);
  }
  TL_CTX.with(|ctx| ctx.gpu_workspace.set(m))
}

pub fn ctx_init_gpu(dev: i32) {
  TL_CTX.with(|ctx| {
    // FIXME FIXME
    ctx.dev.set(dev);
  })
}

pub fn ctx_fresh() -> CellPtr {
  TL_CTX.with(|ctx| {
    ctx.fresh()
  })
}

pub fn ctx_fresh_tmp() -> CellPtr {
  TL_CTX.with(|ctx| {
    ctx.fresh_tmp()
  })
}

pub fn ctx_reset_tmp_unchecked() {
  TL_CTX.with(|ctx| {
    // FIXME FIXME
    ctx.tmp_ctr.set(0);
  })
}

pub fn ctx_retain(x: CellPtr) {
  TL_CTX.with(|ctx| {
    match ctx.env.borrow().lookup(x) {
      None => panic!("bug"),
      Some(e) => {
        let cur = e.stablect.get();
        assert!(cur != usize::max_value());
        let next = cur + 1;
        e.stablect.set(next);
      }
    }
  })
}

pub fn ctx_release(x: CellPtr) {
  TL_CTX.with(|ctx| {
    match ctx.env.borrow().lookup(x) {
      None => panic!("bug"),
      Some(e) => {
        let cur = e.stablect.get();
        assert!(cur != usize::max_value());
        let next = cur - 1;
        e.stablect.set(next);
      }
    }
  })
}

pub fn ctx_alias_bits(og: CellPtr, new_dtype: Dtype) -> CellPtr {
  TL_CTX.with(|ctx| {
    let mut env = ctx.env.borrow_mut();
    match env.lookup(og) {
      None => panic!("bug"),
      Some(e) => {
        assert_eq!(new_dtype.size_bytes(), e.ty.dtype.size_bytes());
        let new_ty = CellType{dtype: new_dtype, shape: e.ty.shape.clone()};
        let x = ctx.fresh();
        env.insert_alias(x, new_ty, og);
        x
      }
    }
  })
}

pub fn ctx_alias_new_shape(og: CellPtr, new_shape: Vec<i64>) -> CellPtr {
  TL_CTX.with(|ctx| {
    let mut env = ctx.env.borrow_mut();
    match env.lookup(og) {
      None => panic!("bug"),
      Some(e) => {
        let new_ty = CellType{dtype: e.ty.dtype, shape: new_shape};
        let cmp = new_ty.shape_compat(&e.ty);
        assert!(cmp == ShapeCompat::Equal || cmp == ShapeCompat::NewShape);
        let x = ctx.fresh();
        env.insert_alias(x, new_ty, og);
        x
      }
    }
  })
}

pub fn ctx_lookup_type(x: CellPtr) -> CellType {
  TL_CTX.with(|ctx| {
    match ctx.env.borrow().lookup(x) {
      None => panic!("bug"),
      Some(e) => e.ty.clone()
    }
  })
}

pub fn ctx_lookup_mode(x: CellPtr) -> CellMode {
  TL_CTX.with(|ctx| {
    match ctx.env.borrow().lookup(x) {
      None => panic!("bug"),
      Some(e) => e.cel.mode
    }
  })
}

pub fn ctx_lookup_flag(x: CellPtr) -> CellFlag {
  TL_CTX.with(|ctx| {
    match ctx.env.borrow().lookup(x) {
      None => panic!("bug"),
      Some(e) => {
        e.cel.flag
        //*e.cel.flag.borrow()
      }
    }
  })
}

pub fn ctx_set_cache(x: CellPtr) {
  TL_CTX.with(|ctx| {
    match ctx.env.borrow_mut().lookup_mut(x) {
    //match ctx.env.borrow().lookup(x) {}
      None => panic!("bug"),
      Some(e) => {
        e.cel.flag.set_cache();
        //e.cel.flag.borrow_mut().set_cache();
      }
    }
  })
}

pub fn ctx_set_eval(x: CellPtr) {
  TL_CTX.with(|ctx| {
    match ctx.env.borrow_mut().lookup_mut(x) {
    //match ctx.env.borrow().lookup(x) {}
      None => panic!("bug"),
      Some(e) => {
        e.cel.flag.set_eval();
        //e.cel.flag.borrow_mut().set_eval();
      }
    }
  })
}

pub fn ctx_lookup_gradl(x: CellPtr, tg: CellPtr) -> CellPtr {
  TL_CTX.with(|ctx| {
    ctx.env.borrow_mut().lookup_or_insert_gradr(tg, x)
  })
}

pub fn ctx_lookup_gradr(tg: CellPtr, x: CellPtr) -> CellPtr {
  TL_CTX.with(|ctx| {
    ctx.env.borrow_mut().lookup_or_insert_gradr(tg, x)
  })
}

pub fn ctx_clean_arg() -> bool {
  TL_CTX.with(|ctx| {
    ctx.arg.borrow().is_empty() &&
    ctx.out.get().is_none()
  })
}

pub fn ctx_push_cell_arg(x: CellPtr) {
  TL_CTX.with(|ctx| {
    ctx.arg.borrow_mut().push(x)
  })
}

pub fn ctx_push_cell_out(x: CellPtr) {
  TL_CTX.with(|ctx| {
    assert!(ctx.out.get().is_none());
    ctx.out.set(Some(x));
  })
}

pub fn ctx_push_cell_tmp_out() {
  TL_CTX.with(|ctx| {
    assert!(ctx.out.get().is_none());
    let x = ctx.fresh_tmp();
    ctx.out.set(Some(x));
  })
}

pub fn ctx_pop_thunk_<Th: Thunk>(th: Th) -> CellPtr {
  TL_CTX.with(|ctx| {
    let mut h = Blake2s::new_hash();
    for a in ctx.arg.borrow().iter() {
      h.hash_bytes(a.as_bytes_repr());
    }
    h.hash_bytes(th.as_bytes_repr());
    let th_hash = h.finalize();
    let thunkenv = ctx.thunkenv.borrow();
    let mut tp = match thunkenv.thunkidx.get(&th_hash) {
      None => {
        unimplemented!();
      }
      Some(&tp) => {
        match thunkenv.thunktab.get(&tp) {
          None => {
            // FIXME: this might happen due to thunk gc.
            unimplemented!();
          }
          Some(_) => {
            // FIXME FIXME: thunk comparison here.
            Some(tp)
          }
        }
      }
    };
    drop(thunkenv);
    if tp.is_none() {
      // TODO
      unimplemented!();
    }
    unimplemented!();
  })
}

//pub fn ctx_pop_thunk<Th: Any + Hash>(th: Th) -> CellPtr {}
pub fn ctx_pop_thunk(th: ThunkPtr) -> CellPtr {
  TL_CTX.with(|ctx| {
    // FIXME FIXME
    let mut thunkenv = ctx.thunkenv.borrow_mut();
    //let th_tyid = TypeId::new::<Th>();
    // FIXME
    //let th_hash = [0; 32];
    //thunkenv.thunkidx.insert((th_tyid, th_hash), ());
    unimplemented!();
    /*
    let mut arg = ctx.arg.borrow_mut();
    // TODO
    arg.clear();
    ctx.out.set(None);
    // FIXME
    */
  })
}

pub fn ctx_reset() {
  unimplemented!();
  /*TL_CTX.with(|ctx| {
    // FIXME FIXME
    ctx.spine.borrow_mut().reset();
  })*/
}

pub fn ctx_compile() {
  unimplemented!();
  /*TL_CTX.with(|ctx| {
    // FIXME FIXME
    ctx.spine.borrow_mut().reduce();
  })*/
}

pub fn ctx_resume() {
  unimplemented!();
  /*TL_CTX.with(|ctx| {
    // FIXME FIXME
    ctx.spine.borrow_mut().resume();
  })*/
}

pub fn ctx_bar() {
  unimplemented!();
}

pub fn ctx_gc() {
  TL_CTX.with(|ctx| {
    // FIXME FIXME
    let mut gc_list = Vec::new();
    ctx.env.borrow().gc_prepare(&mut gc_list);
    ctx.env.borrow_mut().gc(&gc_list);
    //ctx.thunkenv.borrow_mut().gc(&gc_list);
  })
}

/*pub struct ThunkEnvEntry {
}*/

#[derive(Default)]
pub struct CtxThunkEnv {
  // FIXME
  pub thunktab: HashMap<ThunkPtr, PThunk>,
  //pub thunkidx: HashMap<Box<dyn (Any + Eq + Hash + 'static)>, ThunkPtr>,
  //pub thunkidx: HashMap<(TypeId, [u8; 32]), ThunkPtr>,
  pub thunkidx: HashMap<[u8; 32], ThunkPtr>,
}

impl CtxThunkEnv {
  pub fn gc(&mut self, gc_list: &[CellPtr]) {
    // FIXME FIXME: remove updating thunks.
  }
}

#[derive(Clone, Copy, Default)]
#[repr(transparent)]
pub struct CellRefFlag {
  bits: u8,
}

impl CellRefFlag {
  pub fn reset(&mut self) {
    self.bits = 0;
  }

  pub fn set_opaque(&mut self) {
    self.bits |= 1;
  }

  pub fn opaque(&self) -> bool {
    (self.bits & 1) != 0
  }

  pub fn set_trace(&mut self) {
    self.bits |= 0x40;
  }

  pub fn trace(&self) -> bool {
    (self.bits & 0x40) != 0
  }

  pub fn set_break_(&mut self) {
    self.bits |= 0x80;
  }

  pub fn break_(&self) -> bool {
    (self.bits & 0x80) != 0
  }
}

pub enum CellRef {
  Top,
  Place(PCell),
  Alias(CellPtr),
  Bot,
}

pub struct CellEnvEntry {
  pub stablect: Cell<usize>,
  pub ty:       CellType,
  // FIXME FIXME
  //pub lay:      CellLayout,
  //pub ithunk:   Option<PThunk>,
  //pub thunk:    Vec<PThunk>,
  pub ithunk:   Option<ThunkPtr>,
  pub thunk:    Vec<ThunkPtr>,
  pub rflag:    CellRefFlag,
  pub ref_:     CellRef,
  //pub cel:      PCell,
}

pub struct CellEnvEntryRef<'a> {
  pub stablect: &'a Cell<usize>,
  pub ty:       &'a CellType,
  pub ithunk:   Option<ThunkPtr>,
  pub thunk:    &'a [ThunkPtr],
  pub rflag:    CellRefFlag,
  pub cel:      &'a PCell,
}

pub struct CellEnvEntryMut<'a> {
  pub stablect: &'a Cell<usize>,
  pub ty:       CellType,
  pub ithunk:   Option<ThunkPtr>,
  pub thunk:    &'a [ThunkPtr],
  pub rflag:    &'a mut CellRefFlag,
  pub cel:      &'a mut PCell,
}

#[derive(Default)]
pub struct CtxEnv {
  // FIXME
  pub celtab:   HashMap<CellPtr, CellEnvEntry>,
  pub root:     RefCell<HashMap<CellPtr, CellPtr>>,
  pub gradr:    HashMap<[CellPtr; 2], CellPtr>,
  pub ungradr:  HashMap<[CellPtr; 2], CellPtr>,
  pub tag:      HashMap<CellPtr, HashSet<String>>,
  //pub tag:      HashMap<CellPtr, Vec<String>>,
}

impl CtxEnv {
  //pub fn lookup(&self, x: CellPtr) -> (CellType, Option<&PCell>) {}
  pub fn lookup(&self, x: CellPtr) -> Option<CellEnvEntryRef> {
    let ty = match self.celtab.get(&x) {
      None => panic!("bug"),
      Some(e) => {
        match &e.ref_ {
          &CellRef::Top => {
            unimplemented!();
          }
          &CellRef::Place(ref cel) => {
            assert_eq!(x, cel.ptr);
            return Some(CellEnvEntryRef{
              stablect: &e.stablect,
              ty: &e.ty,
              ithunk: e.ithunk,
              thunk: &e.thunk,
              rflag: e.rflag,
              cel,
            });
          }
          &CellRef::Alias(_) => {}
          &CellRef::Bot => {
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
        match &e.ref_ {
          &CellRef::Top => {
            unimplemented!();
          }
          &CellRef::Place(ref cel) => {
            // FIXME FIXME
            assert_eq!(ty.dtype.size_bytes(), e.ty.dtype.size_bytes());
            assert!(ty.shape_compat(&e.ty) != ShapeCompat::Incompat);
            assert_eq!(p, cel.ptr);
            return Some(CellEnvEntryRef{
              stablect: &e.stablect,
              ty,
              ithunk: e.ithunk,
              thunk: &e.thunk,
              rflag: e.rflag,
              cel,
            });
          }
          &CellRef::Alias(_) => {
            panic!("bug");
          }
          &CellRef::Bot => {
            // FIXME FIXME
            panic!("bug");
            //return None;
          }
        }
      }
    }
  }

  pub fn lookup_mut(&mut self, x: CellPtr) -> Option<CellEnvEntryMut> {
    let mut noalias = false;
    let ty = match self.celtab.get(&x) {
      None => panic!("bug"),
      Some(e) => {
        match &e.ref_ {
          &CellRef::Top => {
            unimplemented!();
          }
          &CellRef::Place(ref cel) => {
            noalias = true;
          }
          &CellRef::Alias(_) => {}
          &CellRef::Bot => {
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
        match &mut e.ref_ {
          &mut CellRef::Top => {
            unimplemented!();
          }
          &mut CellRef::Place(ref mut cel) => {
            // FIXME FIXME
            assert_eq!(ty.dtype.size_bytes(), e.ty.dtype.size_bytes());
            assert!(ty.shape_compat(&e.ty) != ShapeCompat::Incompat);
            assert_eq!(p, cel.ptr);
            return Some(CellEnvEntryMut{
              stablect: &e.stablect,
              ty,
              ithunk: e.ithunk,
              thunk: &e.thunk,
              rflag: &mut e.rflag,
              cel,
            });
          }
          &mut CellRef::Alias(_) => {
            panic!("bug");
          }
          &mut CellRef::Bot => {
            // FIXME FIXME
            panic!("bug");
            //return None;
          }
        }
      }
    }
  }

  pub fn insert(&mut self, x: CellPtr, ty: CellType, cel: PCell) {
    unimplemented!();
  }

  pub fn insert_alias(&mut self, x: CellPtr, ty: CellType, og: CellPtr) {
    let e = CellEnvEntry{
      stablect: Cell::new(usize::max_value()),
      ty,
      ithunk:   None,
      thunk:    Vec::new(),
      rflag:    CellRefFlag::default(),
      ref_:     CellRef::Alias(og),
    };
    self.celtab.insert(x, e);
  }

  pub fn lookup_or_insert_gradr(&mut self, tg: CellPtr, x: CellPtr) -> CellPtr {
    unimplemented!();
  }

  pub fn gc_prepare(&self, gc_list: &mut Vec<CellPtr>) {
    for (&k, _) in self.celtab.iter() {
      match self.lookup(k) {
        None => panic!("bug"),
        Some(e) => {
          if e.stablect.get() <= 0 {
            gc_list.push(k);
          }
        }
      }
    }
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
