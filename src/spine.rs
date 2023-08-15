use crate::algo::{HashMap, HashSet, BTreeMap, BTreeSet, StdCellExt};
use crate::cell::*;
use crate::clock::{Counter, Clock, TotalClock};
use crate::ctx::*;
use crate::panick::{panick_wrap};
use crate::pctx::{TL_PCTX, Locus, MemReg, BorrowRefMut};
use crate::thunk::*;
use crate::thunk::op::{SetScalarFutThunkSpec};
use crate::util::time::{Stopwatch};
use cacti_cfg_env::*;

use std::any::{Any};
use std::cell::{Cell, RefCell};
use std::cmp::{Ordering, max, min};
use std::convert::{TryInto};
use std::mem::{swap};

/*pub const ADD_CODE:             u8 = b'A';
pub const INTRO_CODE:           u8 = b':';
pub const CACHE_CODE:           u8 = b'C';
pub const CACHE_INIT_CODE:      u8 = b'I';
pub const YIELD_SET_CODE:       u8 = b'Y';
pub const YIELD_INIT_CODE:      u8 = b'J';
pub const ALIAS_CODE:           u8 = b'@';
//pub const KEEP_CODE:            u8 = _;
//pub const SNAPSHOT_CODE:        u8 = _;
pub const PUSH_SEAL_CODE:       u8 = b',';
pub const INITIALIZE_CODE:      u8 = b'0';
pub const APPLY_CODE:           u8 = b'=';
pub const ACCUMULATE_CODE:      u8 = b'+';
pub const UNSAFE_WRITE_CODE:    u8 = b'!';*/

pub enum SpineResume<'a> {
  _Top,
  PutMemV(CellPtr, &'a dyn Any),
  PutMemF(CellPtr, &'a dyn Fn(CellType, MemReg, )),
  PutMemFun(CellPtr, &'a dyn Fn(&CellType, &[u8])),
  PutMemMutFun(CellPtr, &'a dyn Fn(&CellType, &mut [u8])),
  Put(CellPtr, &'a CellType, &'a dyn CellStoreTo),
}

impl<'a> SpineResume<'a> {
  pub fn key(&self) -> Option<CellPtr> {
    let ret = match self {
      &SpineResume::_Top => {
        None
      }
      &SpineResume::PutMemV(key, _) => {
        Some(key)
      }
      &SpineResume::PutMemF(key, _) => {
        Some(key)
      }
      &SpineResume::PutMemFun(key, ..) |
      &SpineResume::PutMemMutFun(key, ..) |
      &SpineResume::Put(key, ..) => {
        Some(key)
      }
      _ => unimplemented!()
    };
    ret
  }

  pub fn take(&mut self) -> SpineResume<'a> {
    let ret = match &*self {
      &SpineResume::_Top => {
        SpineResume::_Top
      }
      &SpineResume::PutMemV(key, val) => {
        SpineResume::PutMemV(key, val)
      }
      &SpineResume::PutMemF(key, fun) => {
        SpineResume::PutMemF(key, fun)
      }
      &SpineResume::PutMemFun(key, fun) => {
        SpineResume::PutMemFun(key, fun)
      }
      &SpineResume::PutMemMutFun(key, fun) => {
        SpineResume::PutMemMutFun(key, fun)
      }
      &SpineResume::Put(key, ty, val) => {
        SpineResume::Put(key, ty, val)
      }
      _ => unimplemented!()
    };
    *self = SpineResume::_Top;
    ret
  }
}

//#[derive(Clone, Debug)]
#[derive(Clone, Copy, Debug)]
//#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SpineEntry {
  _Top,
  Yield_,
  YieldV(CellPtr, CellPtr),
  Break_,
  BreakV(CellPtr, CellPtr),
  TraceV(CellPtr, CellPtr),
  Profile(CellPtr, CellPtr),
  AdjMap(MCellPtr, MCellPtr, ),
  DualMap(MCellPtr, MCellPtr, ),
  Add(MCellPtr, CellPtr, Clock),
  Add2(MCellPtr, CellPtr, Clock, CellPtr, Clock),
  Cache(CellPtr, Clock),
  //ICacheMux(CellPtr),
  //OIntro(CellPtr, CellPtr),
  //Intro(CellPtr, Clock),
  //Uninit(CellPtr, Clock),
  //IntroFin(CellPtr),
  YieldSet(CellPtr, Clock, Locus),
  YieldInit(CellPtr, Clock, Locus),
  Alias(CellPtr, CellPtr),
  CAlias(CellPtr, CellPtr),
  //Snapshot(CellPtr, CellPtr),
  //Snapshot(CellPtr, CellPtr, CellPtr, /*Clock*/),
  //Keep(CellPtr),
  Push(CellPtr, Clock),
  //PushOut(CellPtr, Clock),
  Initialize(CellPtr, Clock, ThunkPtr),
  Apply(CellPtr, Clock, ThunkPtr),
  //Apply2([(CellPtr, Clock); 2], ThunkPtr),
  Accumulate(CellPtr, Clock, ThunkPtr),
  UnsafeWrite(CellPtr, Clock, ThunkPtr),
  Seal(CellPtr),
  Unseal(CellPtr),
  //Eval(CellPtr),
  //Uneval(CellPtr),
  Live(CellPtr),
  Unlive(CellPtr),
  Unsync(CellPtr),
  //Fetch(_),
  //Unfetch(_),
  // TODO
  Bot,
}

impl SpineEntry {
  pub fn name(&self) -> SpineEntryName {
    match self {
      &SpineEntry::_Top           => SpineEntryName::_Top,
      &SpineEntry::Yield_         => SpineEntryName::Yield_,
      &SpineEntry::YieldV(..)     => SpineEntryName::YieldV,
      &SpineEntry::Break_         => SpineEntryName::Break_,
      &SpineEntry::BreakV(..)     => SpineEntryName::BreakV,
      &SpineEntry::TraceV(..)     => SpineEntryName::TraceV,
      &SpineEntry::Profile(..)    => SpineEntryName::Profile,
      &SpineEntry::AdjMap(..)     => SpineEntryName::AdjMap,
      &SpineEntry::DualMap(..)    => SpineEntryName::DualMap,
      &SpineEntry::Add(..)        => SpineEntryName::Add,
      &SpineEntry::Add2(..)       => SpineEntryName::Add2,
      &SpineEntry::Cache(..)      => SpineEntryName::Cache,
      //&SpineEntry::ICacheMux(..)  => SpineEntryName::ICacheMux,
      //&SpineEntry::OIntro(..)     => SpineEntryName::OIntro,
      //&SpineEntry::Intro(..)      => SpineEntryName::Intro,
      //&SpineEntry::Uninit(..)     => SpineEntryName::Uninit,
      &SpineEntry::YieldSet(..)   => SpineEntryName::YieldSet,
      &SpineEntry::YieldInit(..)  => SpineEntryName::YieldInit,
      &SpineEntry::Alias(..)      => SpineEntryName::Alias,
      &SpineEntry::CAlias(..)     => SpineEntryName::CAlias,
      //&SpineEntry::Snapshot(..)   => SpineEntryName::Snapshot,
      &SpineEntry::Push(..)       => SpineEntryName::Push,
      &SpineEntry::Initialize(..) => SpineEntryName::Initialize,
      &SpineEntry::Apply(..)      => SpineEntryName::Apply,
      &SpineEntry::Accumulate(..) => SpineEntryName::Accumulate,
      &SpineEntry::UnsafeWrite(..) => SpineEntryName::UnsafeWrite,
      &SpineEntry::Seal(..)       => SpineEntryName::Seal,
      &SpineEntry::Unseal(..)     => SpineEntryName::Unseal,
      //&SpineEntry::Eval(..)       => SpineEntryName::Eval,
      &SpineEntry::Live(..)       => SpineEntryName::Live,
      &SpineEntry::Unlive(..)     => SpineEntryName::Unlive,
      &SpineEntry::Unsync(..)     => SpineEntryName::Unsync,
      &SpineEntry::Bot            => SpineEntryName::Bot,
    }
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SpineEntryName {
  _Top,
  Yield_,
  YieldV,
  Break_,
  BreakV,
  TraceV,
  Profile,
  AdjMap,
  DualMap,
  Add,
  Add2,
  Cache,
  //ICacheMux,
  //OIntro,
  //Intro,
  //Uninit,
  YieldSet,
  YieldInit,
  Alias,
  CAlias,
  //Snapshot,
  Push,
  Initialize,
  Apply,
  Accumulate,
  UnsafeWrite,
  Seal,
  Unseal,
  //Eval,
  //Uneval,
  Live,
  Unlive,
  Unsync,
  //Fetch,
  //Unfetch,
  // TODO
  Bot,
}

#[derive(Clone, Copy, Debug)]
#[repr(u8)]
pub enum SpineRet {
  _Top,
  Halt,
  Pause,
  Yield_(()),
  Break_,
  Bot,
}

pub type SpineCellState = CellState;
//pub type SpineCellSet = MCellSet;
//pub type SpineCellMap = MCellMap;

/*pub struct SpineCellState {
  // FIXME FIXME
  //pub alias_ct: u32,
  pub hasalias: bool,
  pub is_alias: bool,
  pub mode:     CellMode,
  pub flag:     CellFlag,
  pub clk:      Clock,
}*/

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct SpineCellAlias {
  pub const_: bool,
  //pub bits: u8,
}

impl Default for SpineCellAlias {
  fn default() -> SpineCellAlias {
    SpineCellAlias{const_: false}
    //SpineCellAlias{bits: 0}
  }
}

impl SpineCellAlias {
  pub fn merge(self, rhs: SpineCellAlias) -> SpineCellAlias {
    SpineCellAlias{const_: self.const_ || rhs.const_}
  }
}

#[derive(Clone, Debug)]
pub enum SpineEnvEntry {
  State(SpineCellState),
  Alias(SpineCellAlias, CellPtr),
}

impl From<SpineCellState> for SpineEnvEntry {
  fn from(state: SpineCellState) -> SpineEnvEntry {
    SpineEnvEntry::State(state)
  }
}

#[derive(Clone, Copy, Debug)]
pub enum SpineEnvMapEntry {
  Value(CellPtr, Clock),
  Alias(CellPtr),
}

#[derive(Clone, Debug)]
pub enum SpineEnvMapEntry2 {
  Value(CellPtr, Vec<(Clock, Clock)>),
  Alias(CellPtr),
}

#[derive(Clone, Default, Debug)]
pub struct SpineFlowEnv {
  pub live_cur: u32,
  pub liveness: HashMap<CellPtr, SpineFlowLivenessEntry>,
  pub liveinit: BTreeSet<(u32, CellPtr)>,
  pub live_fin: BTreeSet<(u32, CellPtr)>,
}

impl SpineFlowEnv {
  pub fn reset(&mut self) {
    self.live_cur = 0;
    self.liveness.clear();
    self.liveinit.clear();
    self.live_fin.clear();
  }
}

#[derive(Clone, Copy, Debug)]
pub struct SpineFlowLivenessEntry {
  pub init: u32,
  pub last: u32,
  //pub lup:  u32,
}

impl Default for SpineFlowLivenessEntry {
  fn default() -> SpineFlowLivenessEntry {
    SpineFlowLivenessEntry{
      init: u32::max_value(),
      last: u32::max_value(),
    }
  }
}

#[derive(Clone, Copy, Debug)]
pub enum SpineEnvGet {
  Value(CellPtr, Clock),
  ReqAlias(CellPtr, CellPtr),
}

#[derive(Clone, Default, Debug)]
pub struct SpineEnv {
  pub ctr:      Counter,
  pub state:    HashMap<CellPtr, SpineEnvEntry>,
  //pub set:      BTreeSet<(MCellPtr, CellPtr)>,
  pub map:      BTreeMap<(MCellPtr, CellPtr), SpineEnvMapEntry2>,
  pub arg:      Vec<(CellPtr, Clock)>,
  //pub out:      Vec<(CellPtr, Clock)>,
  pub update:   HashMap<(CellPtr, Clock), (CellPtr, Vec<(CellPtr, Clock)>)>,
  pub flow:     SpineFlowEnv,
}

impl SpineEnv {
  pub fn reset(&mut self, ctr: Counter) {
    if cfg_debug() { println!("DEBUG: SpineEnv::reset: ctr={:?}", ctr); }
    self.ctr = ctr;
    // FIXME: during ctx reset, need to manually clear gc'd state.
    /*self.state.clear();*/
    //self.set.clear();
    self.map.clear();
    self.arg.clear();
    //self.out.clear();
    self.update.clear();
    self.flow.reset();
  }

  pub fn _deref(&self, query: CellPtr) -> CellPtr {
    let mut cursor = query;
    loop {
      match self.state.get(&cursor) {
        Some(&SpineEnvEntry::Alias(_, next)) => {
          cursor = next;
        }
        _ => return cursor
      }
    }
  }

  pub fn _deref_alias(&self, query: CellPtr) -> (SpineCellAlias, CellPtr) {
    let mut alias = SpineCellAlias::default();
    let mut cursor = query;
    loop {
      match self.state.get(&cursor) {
        Some(&SpineEnvEntry::Alias(a, next)) => {
          alias = alias.merge(a);
          cursor = next;
        }
        _ => return (alias, cursor)
      }
    }
  }

  pub fn _lookup(&self, x: CellPtr) -> Option<(CellPtr, &SpineCellState)> {
    let root = self._deref(x);
    //self.state.get(&root).map(|state| (root, state))
    match self.state.get(&root) {
      None => None,
      Some(&SpineEnvEntry::State(ref state)) => Some((root, state)),
      _ => panic!("bug")
    }
  }

  pub fn _lookup_mut(&mut self, x: CellPtr) -> Option<(CellPtr, &mut SpineCellState)> {
    let root = self._deref(x);
    //self.state.get_mut(&root).map(|state| (root, state))
    match self.state.get_mut(&root) {
      None => None,
      Some(&mut SpineEnvEntry::State(ref mut state)) => Some((root, state)),
      _ => panic!("bug")
    }
  }

  pub fn _get(&self, mx: MCellPtr, k: CellPtr, kclk: Clock) -> Option<SpineEnvGet> {
    let kroot = self._deref(k);
    if k != kroot {
      match (self.map.get(&(mx, k)), self.map.get(&(mx, kroot))) {
        (None, None) => None,
        (None, Some(&SpineEnvMapEntry2::Value(vroot, _))) => {
          Some(SpineEnvGet::ReqAlias(kroot, vroot))
        }
        (Some(&SpineEnvMapEntry2::Alias(v)), Some(&SpineEnvMapEntry2::Value(vroot, ref kvclk))) => {
          assert_eq!(self._deref(v), vroot);
          // FIXME: binary search.
          for &(okclk, vclk) in kvclk.iter().rev() {
            if okclk == kclk {
              return Some(SpineEnvGet::Value(v, vclk));
            }
          }
          unreachable!();
        }
        _ => panic!("bug")
      }
    } else {
      match self.map.get(&(mx, k)) {
        None => None,
        Some(&SpineEnvMapEntry2::Value(v, ref kvclk)) => {
          // FIXME: binary search.
          for &(okclk, vclk) in kvclk.iter().rev() {
            if okclk == kclk {
              return Some(SpineEnvGet::Value(v, vclk));
            }
          }
          unreachable!();
        }
        _ => panic!("bug")
      }
    }
  }

  pub fn step(this: &RefCell<SpineEnv>, sp: u32, curp: &Cell<u32>, log: &RefCell<Vec<SpineEntry>>, ctr: Option<&CtxCtr>, thunkenv: Option<&RefCell<CtxThunkEnv>>) {
    let e_sp = log.borrow()[sp as usize];
    if cfg_debug() { println!("DEBUG: SpineEnv::step: idx={} e={:?}", sp, &e_sp); }
    match e_sp {
      SpineEntry::_Top => {}
      SpineEntry::AdjMap(allsrc, sink) => {
        if cfg_debug() {
        println!("DEBUG: SpineEnv::step: AdjMap (allsrc={:?} sink={:?})",
            allsrc, sink);
        }
        let self_ = this.borrow();
        match self_.map.range(&(sink, CellPtr::nil()) .. ).next() {
          None => {
            return;
          }
          Some((&(mx, _), _)) => {
            if mx > sink {
              return;
            }
          }
        }
        drop(self_);
        let bp = sp;
        'for_sp: for sp in (0 .. bp).rev() {
          let e = log.borrow()[sp as usize];
          match e {
            SpineEntry::AdjMap(..) |
            SpineEntry::Add(..) |
            SpineEntry::Add2(..) |
            SpineEntry::Cache(..) |
            //SpineEntry::Intro(..) |
            //SpineEntry::Uninit(..) |
            SpineEntry::YieldSet(..) |
            SpineEntry::YieldInit(..) |
            SpineEntry::Alias(..) |
            SpineEntry::CAlias(..) |
            //SpineEntry::Snapshot(..) |
            SpineEntry::Push(..) => {}
            SpineEntry::Initialize(y, yclk, th) |
            SpineEntry::Apply(y, yclk, th) => {
              if cfg_debug() {
                let thunkenv = thunkenv.map(|env| env.borrow()).unwrap();
                match thunkenv.thunktab.get(&th) {
                  None => panic!("bug"),
                  Some(te) => {
                    println!("DEBUG: SpineEnv::step: AdjMap: {:?} y={:?} yclk={:?} th={:?} {:?}",
                        e.name(), y, yclk, th, te.pthunk.spec_.debug_name());
                  }
                }
              }
              let self_ = this.borrow();
              let (yalias, yroot) = self_._deref_alias(y);
              if yalias.const_ {
                if cfg_debug() { println!("DEBUG: SpineEnv::step: AdjMap:   const_ yroot={:?} y={:?}", yroot, y); }
                continue 'for_sp;
              }
              let yctr = yclk.ctr();
              match self_._get(sink, y, yclk)
                    .or_else(|| self_._get(allsrc, y, yclk))
              {
                None => {}
                Some(SpineEnvGet::ReqAlias(..)) => {
                  // FIXME: helpful error desc.
                  panic!("bug");
                }
                Some(SpineEnvGet::Value(dy, _dyclk)) => {
                  match self_.update.get(&(yroot, yclk.into())) {
                    None => panic!("bug"),
                    Some(&(y_ref, ref arg)) => {
                      if cfg_debug() { println!("DEBUG: SpineEnv::step: AdjMap:   yroot={:?} y_ref={:?} arg={:?}", yroot, y_ref, arg); }
                      let arg = arg.clone();
                      let mut arg_adj: Vec<CellPtr> = Vec::with_capacity(arg.len());
                      drop(self_);
                      {
                        // NB: do not allocate fresh dx for repeated x in arg.
                        'for_x: for (idx, &(x, xclk)) in arg.iter().rev().enumerate() {
                          let self_ = this.borrow();
                          for (&(x2, _), &dx2) in arg.iter().rev().zip(arg_adj.iter()).take(idx) {
                            if x == x2 {
                              arg_adj.push(dx2);
                              continue 'for_x;
                            }
                          }
                          let (xalias, xroot) = self_._deref_alias(x);
                          if xalias.const_ {
                            if cfg_debug() { println!("DEBUG: SpineEnv::step: AdjMap:   const_ xroot={:?} x={:?}", xroot, x); }
                            arg_adj.push(CellPtr::nil());
                            continue 'for_x;
                          }
                          let xctr = xclk.ctr();
                          if xctr > yctr {
                            panic!("bug");
                          } else if xctr < yctr {
                            if cfg_debug() { println!("DEBUG: SpineEnv::step: AdjMap:   stale xroot={:?} x={:?} xclk={:?}", xroot, x, xclk); }
                            arg_adj.push(CellPtr::nil());
                            continue 'for_x;
                          } else if xctr == yctr {
                          } else {
                            panic!("bug");
                          }
                          let dx =
                              self_._get(sink, x, xclk.into())
                              .or_else(|| self_._get(allsrc, x, xclk.into()))
                              //.map(|&(dx, _)| dx)
                              .and_then(|get| match get {
                                SpineEnvGet::Value(dx, _) => Some(dx),
                                SpineEnvGet::ReqAlias(..) => None
                              })
                              .unwrap_or_else(|| TL_CTX.with(|ctx| {
                                drop(self_);
                                let env = ctx.env.borrow();
                                match env._lookup_view(x) {
                                  Err(_) => panic!("bug"),
                                  Ok(e) => {
                                    let x_ty = e.ty.clone();
                                    assert_eq!(xroot, e.root());
                                    if x != xroot {
                                      let xroot_clk = xclk;
                                      drop(e);
                                      drop(env);
                                      let self_ = this.borrow();
                                      let dxroot =
                                          self_._get(sink, xroot, xroot_clk.into())
                                          .or_else(|| self_._get(allsrc, xroot, xroot_clk.into()))
                                          //.map(|&(dx, _)| dx)
                                          .and_then(|get| match get {
                                            SpineEnvGet::Value(dx, _) => Some(dx),
                                            SpineEnvGet::ReqAlias(..) => None
                                          })
                                          .unwrap_or_else(|| {
                                            drop(self_);
                                            let xroot_ty = match ctx.env.borrow()._lookup_ref_(xroot) {
                                              Err(_) => panic!("bug"),
                                              Ok(e) => e.ty.clone()
                                            };
                                            let dxroot = ctr.unwrap().fresh_cel();
                                            if cfg_debug() { println!("DEBUG: SpineEnv::step: AdjMap:   fresh: xroot={:?} dxroot={:?} ty={:?}", xroot, dxroot, xroot_ty); }
                                            ctx.env.borrow_mut().insert_top(dxroot, xroot_ty);
                                            dxroot
                                          });
                                      let dx = ctr.unwrap().fresh_cel();
                                      if cfg_debug() { println!("DEBUG: SpineEnv::step: AdjMap:   fresh: x={:?} dx={:?} ty={:?}", x, dx, x_ty); }
                                      if cfg_debug() { println!("DEBUG: SpineEnv::step: AdjMap:          xroot={:?} dxroot={:?}", xroot, dxroot); }
                                      // FIXME: double check if NewShape here is correct.
                                      ctx.env.borrow_mut().insert_alias(dx, CellAlias::NewShape, x_ty, dxroot);
                                      {
                                        let sp = {
                                          let sp = curp.get();
                                          curp.set(sp + 1);
                                          let e = SpineEntry::Alias(dx, dxroot);
                                          assert_eq!(log.borrow().len(), sp as usize);
                                          log.borrow_mut().push(e);
                                          sp
                                        };
                                        SpineEnv::step(this, sp, curp, log, ctr, thunkenv);
                                      }
                                      dx
                                    } else {
                                      drop(e);
                                      drop(env);
                                      let dx = ctr.unwrap().fresh_cel();
                                      if cfg_debug() { println!("DEBUG: SpineEnv::step: AdjMap:   fresh: x={:?} dx={:?} ty={:?}", x, dx, x_ty); }
                                      ctx.env.borrow_mut().insert_top(dx, x_ty);
                                      dx
                                    }
                                  }
                                }
                              }));
                          arg_adj.push(dx);
                        }
                        arg_adj.reverse();
                      }
                      TL_CTX.with(|ctx| {
                        // FIXME: this is really a shortcut for quickly
                        // computing first order derivatives.
                        ctx.ctlstate._set_accumulate_in_place(true);
                        ctx.ctlstate._set_assume_uninit_zero(true);
                      });
                      {
                        let thunkenv = thunkenv.map(|env| env.borrow()).unwrap();
                        match thunkenv.thunktab.get(&th) {
                          None => panic!("bug"),
                          Some(te) => {
                            let pthunk = te.pthunk.clone();
                            drop(thunkenv);
                            if cfg_debug() {
                            println!("DEBUG: SpineEnv::step: AdjMap:   pop adj: th={:?} {:?} x={:?} y={:?} yclk={:?} dy={:?} dx={:?}",
                                th, pthunk.spec_.debug_name(), &arg, y, yclk, dy, &arg_adj);
                            }
                            match pthunk.spec_.pop_adj(&arg, y, yclk, ThunkMode::Apply, dy, &mut arg_adj) {
                              Err(_) => {
                                println!("ERROR: SpineEnv::step: adj failure ({:?} th={:?} x={:?} y={:?} yclk={:?} dy={:?} dx={:?})",
                                    e.name(), th, &arg, y, yclk, dy, &arg_adj);
                                panic!();
                              }
                              Ok(_) => {}
                            }
                          }
                        }
                      }
                      TL_CTX.with(|ctx| {
                        ctx.ctlstate._set_accumulate_in_place(false);
                        ctx.ctlstate._set_assume_uninit_zero(false);
                      });
                      for (&(x, xclk), &dx) in arg.iter().zip(arg_adj.iter()).rev() {
                        if dx.is_nil() {
                          if cfg_debug() { println!("DEBUG: SpineEnv::step: AdjMap:   skip: x={:?} dx={:?}", x, dx); }
                          continue;
                        }
                        if cfg_debug() { println!("DEBUG: SpineEnv::step: AdjMap:   add2: allsrc={:?} x={:?} dx={:?}", allsrc, x, dx); }
                        let sp = {
                          let sp = curp.get();
                          curp.set(sp + 1);
                          let e = SpineEntry::Add2(allsrc, x, xclk, dx, Clock::default());
                          assert_eq!(log.borrow().len(), sp as usize);
                          log.borrow_mut().push(e);
                          sp
                        };
                        SpineEnv::step(this, sp, curp, log, ctr, thunkenv);
                      }
                    }
                  }
                }
              }
            }
            SpineEntry::Accumulate(y, yclk, th) => {
              println!("ERROR: SpineEnv::step: AdjMap: unimplemented: autodiff through Accumulate (y={:?} yclk={:?} th={:?})",
                  y, yclk, th);
              panic!();
            }
            SpineEntry::UnsafeWrite(y, yclk, th) => {
              println!("ERROR: SpineEnv::step: AdjMap: cannot autodiff through UnsafeWrite (y={:?} yclk={:?} th={:?})",
                  y, yclk, th);
              panic!();
            }
            _ => {
              println!("DEBUG: SpineEnv::step: AdjMap: unimplemented: bp={} e={:?}", bp, e_sp);
              let min_sp = if sp < 10 { 0 } else { sp - 10 };
              for idx in (min_sp ..= sp).rev() {
                let e = log.borrow()[idx as usize];
                println!("DEBUG: SpineEnv::step: AdjMap: unimplemented:   idx={} e={:?}", idx, e);
              }
              panic!("bug");
            }
          }
        }
        if cfg_debug() { println!("DEBUG: SpineEnv::step: AdjMap: done"); }
      }
      SpineEntry::DualMap(allsink, src) => {
        println!("ERROR: SpineEnv::step: unimplemented: jacobian-vector product via DualMap (allsink={:?} src={:?})",
            allsink, src);
        panic!();
      }
      SpineEntry::Add(mx, x, _xclk) => {
        // FIXME FIXME
        unimplemented!();
        /*let mut self_ = this.borrow_mut();
        let xclk = match self_._lookup(x) {
          //None => panic!("bug"),
          None => Clock::default(),
          Some((_, state)) => state.clk
        };
        let _ = self_.set.insert((mx, x, xclk.into()));
        log.borrow_mut()[sp as usize] = SpineEntry::Add(mx, x, xclk);*/
      }
      SpineEntry::Add2(mx, k, _kclk, v, _vclk) => {
        let mut self_ = this.borrow_mut();
        let kroot = self_._deref(k);
        let kclk = match self_._lookup(k) {
          //None => panic!("bug"),
          None => Clock::default(),
          Some((_, state)) => state.clk
        };
        if v.is_nil() {
          println!("ERROR: SpineEnv::step: Add2: trying to add a nil value: map={:?} k={:?} kroot={:?} v={:?}",
              mx, k, kroot, v);
          unimplemented!();
        }
        let vroot = self_._deref(v);
        let vclk = match self_._lookup(v) {
          //None => panic!("bug"),
          None => Clock::default(),
          Some((_, state)) => state.clk
        };
        if k != kroot {
          if !(v != vroot) {
            println!("DEBUG: SpineEnv::step: Add2 mx={:?} k={:?} v={:?}", mx, k, v);
            println!("DEBUG: SpineEnv::step:   kroot={:?} kclk={:?}", kroot, kclk);
            println!("DEBUG: SpineEnv::step:   vroot={:?} vclk={:?}", vroot, vclk);
          }
          assert!(v != vroot);
          match self_.map.get_mut(&(mx, kroot)) {
            None => {
              let mut kvclk = Vec::new();
              kvclk.push((kclk, vclk));
              self_.map.insert((mx, kroot), SpineEnvMapEntry2::Value(vroot, kvclk));
            }
            Some(&mut SpineEnvMapEntry2::Value(ovroot, ref mut kvclk)) => {
              assert_eq!(ovroot, vroot);
              let fin = kvclk.len() - 1;
              let (okclk, ovclk) = kvclk[fin];
              // FIXME: descriptive error.
              assert!(okclk <= kclk);
              assert!(ovclk <= vclk);
              if okclk == kclk {
                kvclk[fin] = (kclk, vclk);
              } else {
                kvclk.push((kclk, vclk));
              }
            }
            _ => panic!("bug")
          }
          match self_.map.insert((mx, k), SpineEnvMapEntry2::Alias(v)) {
            None => {}
            Some(SpineEnvMapEntry2::Alias(ov)) => {
              assert_eq!(ov, v);
            }
            _ => panic!("bug")
          }
        } else {
          assert_eq!(v, vroot);
          match self_.map.get_mut(&(mx, k)) {
            None => {
              let mut kvclk = Vec::new();
              kvclk.push((kclk, vclk));
              self_.map.insert((mx, k), SpineEnvMapEntry2::Value(v, kvclk));
            }
            Some(&mut SpineEnvMapEntry2::Value(ov, ref mut kvclk)) => {
              assert_eq!(ov, v);
              let fin = kvclk.len() - 1;
              let (okclk, ovclk) = kvclk[fin];
              // FIXME: descriptive error.
              assert!(okclk <= kclk);
              assert!(ovclk <= vclk);
              if okclk == kclk {
                kvclk[fin] = (kclk, vclk);
              } else {
                kvclk.push((kclk, vclk));
              }
            }
            _ => panic!("bug")
          }
        }
        log.borrow_mut()[sp as usize] = SpineEntry::Add2(mx, k, kclk, v, vclk);
      }
      SpineEntry::Cache(x, _xclk) => {
        let mut self_ = this.borrow_mut();
        let xroot = self_._deref(x);
        match self_._lookup(x) {
          None => {
            self_.state.insert(xroot, CellState::default().into());
          }
          _ => {}
        }
        let base_clk: Clock = self_.ctr.into();
        match self_._lookup_mut(x) {
          None => panic!("bug"),
          Some((_, state)) => {
            //assert!(!state.flag.seal());
            /*assert_eq!(state.clk.up, 0);*/
            //assert!(state.clk.is_uninit());
            //state.flag.unset_seal();
            let next_clk = base_clk.init_once();
            assert!(state.clk < next_clk);
            state.clk = next_clk;
            drop(state);
            log.borrow_mut()[sp as usize] = SpineEntry::Cache(x, next_clk);
          }
        }
      }
      /*SpineEntry::ICacheMux(x) => {
        // FIXME FIXME
        unimplemented!();
      }*/
      SpineEntry::YieldSet(x, _xclk, _loc) => {
        let mut self_ = this.borrow_mut();
        let xroot = self_._deref(x);
        match self_._lookup(x) {
          None => {
            self_.state.insert(xroot, CellState::default().into());
          }
          _ => {}
        }
        let base_clk: Clock = self_.ctr.into();
        match self_._lookup_mut(x) {
          None => panic!("bug"),
          Some((_, state)) => {
            //assert!(!state.flag.seal());
            /*assert_eq!(state.clk.up, 0);*/
            //assert!(state.clk.is_uninit());
            //state.flag.unset_seal();
            let next_clk = base_clk.init_once();
            assert!(state.clk < next_clk);
            state.clk = next_clk;
            drop(state);
            log.borrow_mut()[sp as usize] = SpineEntry::YieldSet(x, next_clk, _loc);
          }
        }
      }
      SpineEntry::YieldInit(x, _xclk, _loc) => {
        let mut self_ = this.borrow_mut();
        let xroot = self_._deref(x);
        match self_._lookup(x) {
          None => {
            self_.state.insert(xroot, CellState::default().into());
          }
          _ => {}
        }
        let base_clk: Clock = self_.ctr.into();
        match self_._lookup_mut(x) {
          None => panic!("bug"),
          Some((_, state)) => {
            //assert!(!state.flag.seal());
            /*assert_eq!(state.clk.up, 0);*/
            // FIXME
            /*assert!(state.clk.is_uninit());*/
            let next_clk = base_clk.init_once();
            assert!(state.clk < next_clk);
            state.clk = next_clk;
            drop(state);
            log.borrow_mut()[sp as usize] = SpineEntry::YieldInit(x, next_clk, _loc);
          }
        }
      }
      SpineEntry::Alias(x, og) => {
        let mut self_ = this.borrow_mut();
        match self_.state.get(&x) {
          None => {
            self_.state.insert(x, SpineEnvEntry::Alias(SpineCellAlias::default(), og));
          }
          Some(_) => {
            panic!("bug");
          }
        }
      }
      SpineEntry::CAlias(x, og) => {
        let mut self_ = this.borrow_mut();
        match self_.state.get(&x) {
          None => {
            let mut alias = SpineCellAlias::default();
            alias.const_ = true;
            self_.state.insert(x, SpineEnvEntry::Alias(alias, og));
          }
          Some(_) => {
            panic!("bug");
          }
        }
      }
      /*SpineEntry::Snapshot(x, og) => {
        let mut self_ = this.borrow_mut();
        // FIXME FIXME
        match self_._lookup(og) {
          None => panic!("bug"),
          Some((_, state)) => {
            let state = state.clone();
            self_.state.insert(x, state.into());
          }
        }
      }*/
      SpineEntry::Push(x, _xclk) => {
        let mut self_ = this.borrow_mut();
        match self_._lookup_mut(x) {
          None => {
            //println!("DEBUG: Push: x={:?} spine env={:?}", x, self);
            panic!("bug");
          }
          Some((_, state)) => {
            // NB: late/lazy idempotent seal.
            /*assert!(!state.flag.seal());*/
            //state.flag.set_seal();
            let next_clk = state.clk;
            drop(state);
            self_.arg.push((x, next_clk));
            log.borrow_mut()[sp as usize] = SpineEntry::Push(x, next_clk);
          }
        }
      }
      SpineEntry::Initialize(x, _xclk, ith) => {
        let mut self_ = this.borrow_mut();
        let base_clk: Clock = self_.ctr.into();
        let xroot = self_._deref(x);
        match self_._lookup(x) {
          None => {
            // FIXME FIXME
            self_.state.insert(xroot, CellState::default().into());
          }
          Some(_) => {}
        }
        match self_._lookup_mut(x) {
          None => panic!("bug"),
          Some((xroot, state)) => {
            //assert!(!state.flag.seal());
            /*assert_eq!(state.clk.up, 0);*/
            // FIXME
            /*assert!(state.clk.is_uninit());*/
            //state.flag.unset_seal();
            let next_clk = base_clk.max(state.clk).init_once();
            assert!(state.clk < next_clk);
            state.clk = next_clk;
            drop(state);
            let mut arg = Vec::new();
            swap(&mut self_.arg, &mut arg);
            assert!(self_.update.insert((xroot, next_clk), (x, arg)).is_none());
            log.borrow_mut()[sp as usize] = SpineEntry::Initialize(x, next_clk, ith);
          }
        }
      }
      SpineEntry::Apply(x, _xclk, th) => {
        let mut self_ = this.borrow_mut();
        let base_clk: Clock = self_.ctr.into();
        let xroot = self_._deref(x);
        match self_._lookup(x) {
          None => {
            // FIXME FIXME
            self_.state.insert(xroot, CellState::default().into());
          }
          Some(_) => {}
        }
        match self_._lookup_mut(x) {
          None => panic!("bug"),
          Some((xroot, state)) => {
            //assert!(!state.flag.seal());
            /*assert_eq!(state.clk.up, 0);*/
            //assert!(state.clk.is_uninit());
            //state.flag.unset_seal();
            let next_clk = base_clk.max(state.clk).init_once();
            assert!(state.clk < next_clk);
            state.clk = next_clk;
            drop(state);
            let mut arg = Vec::new();
            swap(&mut self_.arg, &mut arg);
            assert!(self_.update.insert((xroot, next_clk), (x, arg)).is_none());
            log.borrow_mut()[sp as usize] = SpineEntry::Apply(x, next_clk, th);
          }
        }
      }
      SpineEntry::Accumulate(x, _xclk, th) => {
        let mut self_ = this.borrow_mut();
        let base_clk: Clock = self_.ctr.into();
        let xroot = self_._deref(x);
        //println!("DEBUG: SpineEnv::step: Accumulate: x={:?} xroot={:?} th={:?}", x, xroot, th);
        match self_._lookup(x) {
          None => {
            // FIXME FIXME
            self_.state.insert(xroot, CellState::default().into());
          }
          Some(_) => {}
        }
        match self_._lookup_mut(x) {
          None => panic!("bug"),
          Some((_, state)) => {
            //assert!(!state.flag.seal());
            //state.flag.unset_seal();
            let next_clk = base_clk.max(state.clk).update();
            assert!(state.clk < next_clk);
            state.clk = next_clk;
            drop(state);
            let mut arg = Vec::new();
            swap(&mut self_.arg, &mut arg);
            assert!(self_.update.insert((xroot, next_clk), (x, arg)).is_none());
            log.borrow_mut()[sp as usize] = SpineEntry::Accumulate(x, next_clk, th);
          }
        }
      }
      SpineEntry::UnsafeWrite(x, _xclk, th) => {
        unimplemented!();
      }
      SpineEntry::Seal(x) => {
        let mut self_ = this.borrow_mut();
        match self_._lookup_mut(x) {
          None => panic!("bug"),
          Some((_, state)) => {
            // NB: idempotent seal.
            /*assert!(!state.flag.seal());*/
            //state.flag.set_seal();
          }
        }
      }
      SpineEntry::Unseal(x) => {
        let mut self_ = this.borrow_mut();
        match self_._lookup_mut(x) {
          None => panic!("bug"),
          Some((_, state)) => {
            // NB: idempotent unseal.
            /*assert!(state.flag.seal());*/
            //state.flag.unset_seal();
          }
        }
      }
      SpineEntry::Unsync(x) => {
        // FIXME FIXME
        unimplemented!();
      }
      // TODO TODO
      _ => unimplemented!()
    }
  }

  /*pub fn unstep(&mut self, /*sp: u32,*/ e: &SpineEntry) {
    // FIXME FIXME
  }*/

  pub fn _live_flow(&mut self, curp: u32, log: &[SpineEntry]) {
    let ctr = self.ctr;
    let bp = self.flow.live_cur;
    // FIXME: incremental analysis.
    assert_eq!(bp, 0);
    let mut sp = curp;
    loop {
      if sp == bp {
        break;
      }
      sp -= 1;
      match log[sp as usize] {
        SpineEntry::Cache(y, yclk) |
        SpineEntry::YieldSet(y, yclk, _) |
        SpineEntry::Initialize(y, yclk, _) |
        SpineEntry::Apply(y, yclk, _) => {
          assert!(yclk.is_init_once());
          assert_eq!(ctr, yclk.ctr());
          let y = self._deref(y);
          if cfg_debug() {
            println!("DEBUG:  SpineEnv::_live_flow: sp={} yroot={:?} e={:?}",
                sp, y, log[sp as usize]);
          }
          match self.flow.liveness.get_mut(&y) {
            None => {
              let mut e = SpineFlowLivenessEntry::default();
              e.init = sp;
              self.flow.liveness.insert(y, e);
            }
            Some(e) => {
              assert_eq!(e.init, u32::max_value());
              assert!(sp <= e.last);
              e.init = sp;
            }
          }
          let mut sp2 = sp;
          loop {
            if sp2 == bp {
              sp = sp2;
              break;
            }
            sp2 -= 1;
            match log[sp2 as usize] {
              SpineEntry::Push(x, xclk) => {
                let x = self._deref(x);
                if cfg_debug() {
                  println!("DEBUG:  SpineEnv::_live_flow:   sp2={} xroot={:?} e={:?}",
                      sp2, x, log[sp2 as usize]);
                }
                match self.flow.liveness.get(&x) {
                  None => {
                    let mut e = SpineFlowLivenessEntry::default();
                    e.last = sp;
                    self.flow.liveness.insert(x, e);
                  }
                  Some(e) => {
                    if ctr == xclk.ctr() {
                      if !(e.init == u32::max_value()) {
                        println!("DEBUG:  SpineEnv::_live_flow:     sp2={} xroot={:?} live={:?}",
                            sp2, x, e);
                      }
                      assert_eq!(e.init, u32::max_value());
                    }
                    assert!(sp <= e.last);
                  }
                }
              }
              _ => {
                sp = sp2 + 1;
                break;
              }
            }
          }
        }
        SpineEntry::Accumulate(y, yclk, _) => {
          assert!(yclk.is_update());
          assert_eq!(ctr, yclk.ctr());
          let y = self._deref(y);
          if cfg_debug() {
            println!("DEBUG:  SpineEnv::_live_flow: sp={} yroot={:?} e={:?}",
                sp, y, log[sp as usize]);
          }
          match self.flow.liveness.get_mut(&y) {
            None => {
              let mut e = SpineFlowLivenessEntry::default();
              e.last = sp;
              self.flow.liveness.insert(y, e);
            }
            Some(e) => {
              assert_eq!(e.init, u32::max_value());
              assert!(sp <= e.last);
            }
          }
          let mut sp2 = sp;
          loop {
            if sp2 == bp {
              sp = sp2;
              break;
            }
            sp2 -= 1;
            match log[sp2 as usize] {
              SpineEntry::Push(x, xclk) => {
                let x = self._deref(x);
                if cfg_debug() {
                  println!("DEBUG:  SpineEnv::_live_flow:   sp2={} xroot={:?} e={:?}",
                      sp2, x, log[sp2 as usize]);
                }
                match self.flow.liveness.get(&x) {
                  None => {
                    let mut e = SpineFlowLivenessEntry::default();
                    e.last = sp;
                    self.flow.liveness.insert(x, e);
                  }
                  Some(e) => {
                    if ctr == xclk.ctr() {
                      assert_eq!(e.init, u32::max_value());
                    }
                    assert!(sp <= e.last);
                  }
                }
              }
              _ => {
                sp = sp2 + 1;
                break;
              }
            }
          }
        }
        _ => {}
      }
    }
    for (&x, e) in self.flow.liveness.iter() {
      if cfg_debug() {
        if e.init == u32::max_value() {
          println!("WARNING:SpineEnv::_live_flow: invalid: x={:?} e={:?}", x, e);
        } else {
          println!("DEBUG:  SpineEnv::_live_flow: valid: x={:?} e={:?}", x, e);
        }
      }
      //assert!(!(e.init == u32::max_value()));
      self.flow.liveinit.insert((min(e.init, e.last), x));
      if e.last == u32::max_value() {
        self.flow.live_fin.insert((e.init, x));
      } else {
        self.flow.live_fin.insert((max(e.init, e.last), x));
      }
    }
    self.flow.live_cur = curp;
  }
}

pub struct Spine {
  pub ctr:  Cell<Counter>,
  pub ctlp: Cell<u32>,
  pub hltp: Cell<u32>,
  pub curp: Cell<u32>,
  pub retp: Cell<u32>,
  pub log:  RefCell<Vec<SpineEntry>>,
  pub cur_env:  RefCell<SpineEnv>,
}

impl Default for Spine {
  fn default() -> Spine {
    Spine{
      ctr:  Cell::new(Counter::default()),
      ctlp: Cell::new(0),
      hltp: Cell::new(0),
      curp: Cell::new(0),
      retp: Cell::new(u32::max_value()),
      log:  RefCell::new(Vec::new()),
      cur_env:  RefCell::new(SpineEnv::default()),
    }
  }
}

impl Spine {
  pub fn _reset(&self) -> Counter {
    if cfg_debug() {
    let retp = if self.retp.get() == u32::max_value() { None } else { Some(self.retp.get()) };
    println!("DEBUG: Spine::_reset: pre:  ctr={:?} ctlp={} hltp={} curp={} retp={:?}",
        self.ctr.get(), self.ctlp.get(), self.hltp.get(), self.curp.get(), retp);
    }
    self.ctr.set(self.ctr.get().advance());
    self.ctlp.set(0);
    self.hltp.set(0);
    self.curp.set(0);
    self.retp.set(u32::max_value());
    self.log.borrow_mut().clear();
    self.cur_env.borrow_mut().reset(self.ctr.get());
    if cfg_debug() {
    let retp = if self.retp.get() == u32::max_value() { None } else { Some(self.retp.get()) };
    println!("DEBUG: Spine::_reset: post: ctr={:?} ctlp={} hltp={} curp={} retp={:?}",
        self.ctr.get(), self.ctlp.get(), self.hltp.get(), self.curp.get(), retp);
    }
    self.ctr.get()
  }

  pub fn adj_map(&self, allsrc: MCellPtr, sink: MCellPtr, ctr: &CtxCtr, thunkenv: &RefCell<CtxThunkEnv>) {
    let sp = self.curp.fetch_add(1);
    let e = SpineEntry::AdjMap(allsrc, sink);
    self.log.borrow_mut().push(e);
    SpineEnv::step(&self.cur_env, sp, &self.curp, &self.log, Some(ctr), Some(thunkenv));
    assert_eq!(self.curp.get(), self.log.borrow().len() as _);
  }

  pub fn dual_map(&self, allsink: MCellPtr, src: MCellPtr, ctr: &CtxCtr, thunkenv: &RefCell<CtxThunkEnv>) {
    let sp = self.curp.fetch_add(1);
    let e = SpineEntry::DualMap(allsink, src);
    self.log.borrow_mut().push(e);
    SpineEnv::step(&self.cur_env, sp, &self.curp, &self.log, Some(ctr), Some(thunkenv));
    assert_eq!(self.curp.get(), self.log.borrow().len() as _);
  }

  pub fn add(&self, mx: MCellPtr, x: CellPtr) {
    let sp = self.curp.fetch_add(1);
    let e = SpineEntry::Add(mx, x, Clock::default());
    self.log.borrow_mut().push(e);
    SpineEnv::step(&self.cur_env, sp, &self.curp, &self.log, None, None);
    assert_eq!(self.curp.get(), self.log.borrow().len() as _);
  }

  pub fn add2(&self, mx: MCellPtr, k: CellPtr, v: CellPtr) {
    let sp = self.curp.fetch_add(1);
    let e = SpineEntry::Add2(mx, k, Clock::default(), v, Clock::default());
    self.log.borrow_mut().push(e);
    SpineEnv::step(&self.cur_env, sp, &self.curp, &self.log, None, None);
    assert_eq!(self.curp.get(), self.log.borrow().len() as _);
  }

  pub fn cache_aff(&self, x: CellPtr) {
    self.cache(x)
  }

  pub fn cache(&self, x: CellPtr) {
    let sp = self.curp.fetch_add(1);
    let e = SpineEntry::Cache(x, Clock::default());
    self.log.borrow_mut().push(e);
    SpineEnv::step(&self.cur_env, sp, &self.curp, &self.log, None, None);
    assert_eq!(self.curp.get(), self.log.borrow().len() as _);
  }

  pub fn yield_set(&self, x: CellPtr, loc: Locus) {
    let sp = self.curp.fetch_add(1);
    let e = SpineEntry::YieldSet(x, Clock::default(), loc);
    self.log.borrow_mut().push(e);
    SpineEnv::step(&self.cur_env, sp, &self.curp, &self.log, None, None);
    assert_eq!(self.curp.get(), self.log.borrow().len() as _);
  }

  pub fn yield_init(&self, x: CellPtr, loc: Locus) {
    let sp = self.curp.fetch_add(1);
    let e = SpineEntry::YieldInit(x, Clock::default(), loc);
    self.log.borrow_mut().push(e);
    SpineEnv::step(&self.cur_env, sp, &self.curp, &self.log, None, None);
    assert_eq!(self.curp.get(), self.log.borrow().len() as _);
  }

  pub fn alias(&self, x: CellPtr, og: CellPtr) {
    let sp = self.curp.fetch_add(1);
    let e = SpineEntry::Alias(x, og);
    self.log.borrow_mut().push(e);
    SpineEnv::step(&self.cur_env, sp, &self.curp, &self.log, None, None);
    assert_eq!(self.curp.get(), self.log.borrow().len() as _);
  }

  pub fn const_(&self, x: CellPtr, og: CellPtr) {
    let sp = self.curp.fetch_add(1);
    let e = SpineEntry::CAlias(x, og);
    self.log.borrow_mut().push(e);
    SpineEnv::step(&self.cur_env, sp, &self.curp, &self.log, None, None);
    assert_eq!(self.curp.get(), self.log.borrow().len() as _);
  }

  /*pub fn opaque(&self, x: CellPtr, og: CellPtr) {
    let sp = self.curp.get();
    self.curp.set(sp + 1);
    let e = SpineEntry::OAlias(x, og);
    self.log.borrow_mut().push(e);
    SpineEnv::step(&self.cur_env, sp, &self.curp, &self.log, None, None);
    assert_eq!(self.curp.get(), self.log.borrow().len() as _);
  }*/

  /*pub fn snapshot(&self, x: CellPtr, og: CellPtr) {
    let sp = self.curp.get();
    self.curp.set(sp + 1);
    let e = SpineEntry::Snapshot(x, og);
    self.log.borrow_mut().push(e);
    SpineEnv::step(&self.cur_env, sp, &self.curp, &self.log, None, None);
    assert_eq!(self.curp.get(), self.log.borrow().len() as _);
  }*/

  pub fn push_seal(&self, x: CellPtr) {
    self.push(x)
  }

  pub fn push(&self, x: CellPtr) {
    let sp = self.curp.fetch_add(1);
    let e = SpineEntry::Push(x, Clock::default());
    self.log.borrow_mut().push(e);
    SpineEnv::step(&self.cur_env, sp, &self.curp, &self.log, None, None);
    assert_eq!(self.curp.get(), self.log.borrow().len() as _);
  }

  pub fn initialize(&self, y: CellPtr, ith: ThunkPtr) {
    let sp = self.curp.fetch_add(1);
    let e = SpineEntry::Initialize(y, Clock::default(), ith);
    self.log.borrow_mut().push(e);
    SpineEnv::step(&self.cur_env, sp, &self.curp, &self.log, None, None);
    assert_eq!(self.curp.get(), self.log.borrow().len() as _);
  }

  pub fn apply(&self, y: CellPtr, th: ThunkPtr) {
    let sp = self.curp.fetch_add(1);
    let e = SpineEntry::Apply(y, Clock::default(), th);
    self.log.borrow_mut().push(e);
    SpineEnv::step(&self.cur_env, sp, &self.curp, &self.log, None, None);
    assert_eq!(self.curp.get(), self.log.borrow().len() as _);
  }

  pub fn accumulate(&self, y: CellPtr, th: ThunkPtr) {
    let sp = self.curp.fetch_add(1);
    let e = SpineEntry::Accumulate(y, Clock::default(), th);
    self.log.borrow_mut().push(e);
    SpineEnv::step(&self.cur_env, sp, &self.curp, &self.log, None, None);
    assert_eq!(self.curp.get(), self.log.borrow().len() as _);
  }

  /*pub fn seal(&self, x: CellPtr) {
    let sp = self.curp.fetch_add(1);
    let e = SpineEntry::Seal(x);
    self.log.borrow_mut().push(e);
    SpineEnv::step(&self.cur_env, sp, &self.curp, &self.log, None, None);
    assert_eq!(self.curp.get(), self.log.borrow().len() as _);
  }

  pub fn unseal_mux(&self, x: CellPtr) {
    self.unseal(x)
  }

  pub fn unseal(&self, x: CellPtr) {
    let sp = self.curp.fetch_add(1);
    let e = SpineEntry::Unseal(x);
    self.log.borrow_mut().push(e);
    SpineEnv::step(&self.cur_env, sp, &self.curp, &self.log, None, None);
    assert_eq!(self.curp.get(), self.log.borrow().len() as _);
  }

  pub fn unsync(&self, x: CellPtr) {
    unimplemented!();
  }*/

  pub fn _counter(&self) -> Clock {
    let ctr = self.cur_env.borrow().ctr;
    assert_eq!(ctr, self.ctr.get());
    ctr.into()
  }

  pub fn _deref(&self, query: CellPtr) -> CellPtr {
    self.cur_env.borrow()._deref(query)
  }

  pub fn _version(&self, query: CellPtr) -> Option<Clock> {
    let cur_env = self.cur_env.borrow();
    match cur_env._lookup(query) {
      None => None,
      Some((_, state)) => Some(state.clk)
    }
  }

  pub fn _get(&self, mm: MCellPtr, k: CellPtr, kclk: Clock) -> Option<(CellPtr, Clock)> {
    let cur_env = self.cur_env.borrow();
    match cur_env.map.get(&(mm, k)) {
      None => None,
      Some(&SpineEnvMapEntry2::Value(v, ref kvclk)) => {
        // FIXME: binary search.
        for &(okclk, vclk) in kvclk.iter().rev() {
          if okclk == kclk {
            return Some((v, vclk));
          }
        }
        unreachable!();
      }
      Some(&SpineEnvMapEntry2::Alias(v)) => {
        let kroot = cur_env._deref(k);
        assert!(k != kroot);
        let vroot = cur_env._deref(v);
        assert!(v != vroot);
        match cur_env.map.get(&(mm, kroot)) {
          None => panic!("bug"),
          Some(&SpineEnvMapEntry2::Value(ovroot, ref kvclk)) => {
            assert_eq!(ovroot, vroot);
            // FIXME: binary search.
            for &(okclk, vclk) in kvclk.iter().rev() {
              if okclk == kclk {
                return Some((v, vclk));
              }
            }
            unreachable!();
          }
          _ => panic!("bug"),
        }
      }
    }
  }

  pub fn _compile(&self) {
    if self.hltp.get() != 0 {
      println!("ERROR:  Spine::_compile: incremental compilation is not yet supported;");
      println!("ERROR:  Spine::_compile: for now, please call `compile` once per `reset");
      panic!();
    }
    assert_eq!(self.ctlp.get(), 0);
    assert_eq!(self.retp.get(), u32::max_value());
    let prev_curp = self.curp.get();
    {
      let mut cur_env = self.cur_env.borrow_mut();
      let log = self.log.borrow();
      let prev_len: usize = prev_curp.try_into().unwrap();
      assert_eq!(prev_len, log.len());
      cur_env._live_flow(prev_curp, &*log);
    }
    {
      let cur_env = self.cur_env.borrow();
      let mut prev_log = self.log.borrow_mut();
      let mut next_log = Vec::new();
      for prev_sp in 0 .. prev_curp {
        for &(sp, x) in cur_env.flow.liveinit.range((prev_sp, CellPtr::nil()) .. ) {
          if prev_sp < sp {
            break;
          }
          assert!(!x.is_nil());
          next_log.push(SpineEntry::Live(x));
        }
        next_log.push(prev_log[prev_sp as usize]);
        for &(sp, x) in cur_env.flow.live_fin.range((prev_sp, CellPtr::nil()) .. ) {
          if prev_sp < sp {
            break;
          }
          assert!(!x.is_nil());
          next_log.push(SpineEntry::Unlive(x));
        }
      }
      //cur_env.flow.reset();
      let next_curp: u32 = next_log.len().try_into().unwrap();
      // FIXME: we can save rather than drop the prev log.
      swap(&mut *prev_log, &mut next_log);
      self.curp.set(next_curp);
    }
  }

  /*pub fn _start(&mut self) {
    self.hltp = self.curp;
  }*/

  pub fn _resume(&self, ctr: &CtxCtr, thunkenv: &mut CtxThunkEnv, mut item: SpineResume) -> SpineRet {
    //self._start();
    if cfg_debug() {
    let retp = if self.retp.get() == u32::max_value() { None } else { Some(self.retp.get()) };
    println!("DEBUG: Spine::_resume: ctr={:?} ctlp={} hltp={} curp={} retp={:?} item={:?}",
        self.ctr.get(), self.ctlp.get(), self.hltp.get(), self.curp.get(), retp, item.key());
    }
    self.hltp.set(self.curp.get());
    loop {
      let state = self._step(ctr, /*env,*/ thunkenv, item.take());
      match state {
        SpineRet::Yield_(_) |
        SpineRet::Bot => {
          self.retp.set(self.ctlp.get());
          return state;
        }
        _ => {}
      }
      self.ctlp.fetch_add(1);
      self.retp.set(u32::max_value());
      match state {
        SpineRet::Halt   |
        SpineRet::Pause  |
        SpineRet::Break_ => {
          return state;
        }
        _ => {}
      }
    }
  }

  pub fn _step(&self, ctr: &CtxCtr, thunkenv: &mut CtxThunkEnv, item: SpineResume) -> SpineRet {
    if self.ctlp.get() >= self.hltp.get() {
      if cfg_debug() {
      let retp = if self.retp.get() == u32::max_value() { None } else { Some(self.retp.get()) };
      println!("DEBUG: Spine::_step: ctr={:?} ctlp={} hltp={} curp={} retp={:?} halt",
          self.ctr.get(), self.ctlp.get(), self.hltp.get(), self.curp.get(), retp);
      }
      return SpineRet::Halt;
    }
    let mut ret = SpineRet::_Top;
    let entry = self.log.borrow()[self.ctlp.get() as usize];
    if cfg_debug() {
    let retp = if self.retp.get() == u32::max_value() { None } else { Some(self.retp.get()) };
    println!("DEBUG: Spine::_step: ctr={:?} ctlp={} hltp={} curp={} retp={:?} entry={:?}",
        self.ctr.get(), self.ctlp.get(), self.hltp.get(), self.curp.get(), retp, entry.name());
    }
    let t0 = Stopwatch::tl_stamp();
    match entry {
      // TODO
      SpineEntry::_Top => {}
      SpineEntry::Yield_ => {
        ret = SpineRet::Yield_(());
      }
      SpineEntry::YieldV(_, _) => {
        unimplemented!();
      }
      SpineEntry::Break_ => {
        ret = SpineRet::Break_;
      }
      SpineEntry::BreakV(_, _) => {
        unimplemented!();
      }
      SpineEntry::TraceV(_, _) => {
        unimplemented!();
      }
      SpineEntry::Profile(_, _) => {
        unimplemented!();
      }
      SpineEntry::AdjMap(..) => {}
      SpineEntry::Add(_mx, _x, _xclk) => {}
      SpineEntry::Add2(_mx, _k, _kclk, _v, _vclk) => {}
      SpineEntry::Cache(x, _xclk) => {
        TL_CTX.with(|ctx| {
          let env = ctx.env.borrow();
        match env._lookup_view(x) {
          Err(_) => panic!("bug"),
          Ok(e) => {
            let base_clk: Clock = self.ctr.get().into();
            let prev_clk = e.state().clk;
            let next_clk = base_clk.init_once();
            if prev_clk >= next_clk {
              panic!("bug");
            } else if prev_clk < next_clk {
              e.clock_sync_rec(prev_clk, next_clk, &*env);
            }
            let cel_ = e.cel_.borrow();
            match &*cel_ {
              &Cell_::Phy(_, ref clo, _) => {
                clo.borrow_mut().init_once(next_clk, ThunkPtr::opaque());
              }
              _ => panic!("bug")
            }
          }
        }
        });
      }
      SpineEntry::YieldSet(x, _xclk, loc) => {
        TL_CTX.with(|ctx| {
          let env = ctx.env.borrow();
        let ctlp = self.ctlp.get();
        let retp = if self.retp.get() == u32::max_value() { None } else { Some(self.retp.get()) };
        if cfg_debug() {
        println!("DEBUG: Spine::_step: YieldSet: ctlp={:?} retp={:?} x={:?} loc={:?} key={:?}",
            ctlp, retp, x, loc, item.key());
        match env._lookup_view(x) {
          Err(_) => panic!("bug"),
          Ok(e) => {
            println!("DEBUG: Spine::_step: YieldSet:   expected dtype {:?}", e.ty.dtype);
            match e.ty.dtype {
              Dtype::F32 => {
                //println!("DEBUG: Spine::_step: YieldSet:   expected dtype {:?}", e.ty.dtype);
                match &item {
                  &SpineResume::_Top => {
                    println!("DEBUG: Spine::_step: YieldSet:   no value");
                  }
                  &SpineResume::PutMemV(k, v) => {
                    match v.downcast_ref::<f32>() {
                      None => {
                        println!("DEBUG: Spine::_step: YieldSet:   wrong type");
                        panic!("bug");
                      }
                      Some(v) => {
                        println!("DEBUG: Spine::_step: YieldSet:   PutMemV key={:?} value={:?}", k, v);
                      }
                    }
                  }
                  &SpineResume::PutMemF(k, _f) => {
                    println!("DEBUG: Spine::_step: YieldSet:   PutMemF key={:?}", k);
                  }
                  &SpineResume::PutMemFun(k, ..) => {
                    println!("DEBUG: Spine::_step: YieldSet:   PutMemFun key={:?}", k);
                  }
                  &SpineResume::PutMemMutFun(k, ..) => {
                    println!("DEBUG: Spine::_step: YieldSet:   PutMemMutFun key={:?}", k);
                  }
                  &SpineResume::Put(k, ty, _v) => {
                    println!("DEBUG: Spine::_step: YieldSet:   Put key={:?} ty={:?}", k, ty);
                  }
                  _ => unimplemented!()
                }
              }
              _ => {
              }
            }
          }
        }
        }
        if Some(ctlp) == retp {
          if cfg_debug() { println!("DEBUG: Spine::_step: YieldSet:   ...resume"); }
          let (prev_xclk, xclk) = match env._lookup_view(x) {
            Err(_) => panic!("bug"),
            Ok(e) => {
              let base_clk: Clock = self.ctr.get().into();
              let prev_clk = e.state().clk;
              let next_clk = base_clk.init_once();
              if cfg_debug() {
              println!("DEBUG: Spine::_step: YieldSet:   prev clk={:?}", prev_clk);
              println!("DEBUG: Spine::_step: YieldSet:   base clk={:?}", base_clk);
              println!("DEBUG: Spine::_step: YieldSet:   next clk={:?}", next_clk);
              }
              if prev_clk >= next_clk {
                panic!("bug");
              } else if prev_clk < next_clk {
                e.clock_sync(prev_clk, next_clk, &*env);
              }
              (prev_clk, next_clk)
            }
          };
          match (loc, &item) {
            (Locus::Mem, &SpineResume::PutMemMutFun(key, fun)) => {
              if key != x {
                println!("ERROR:  Spine::_step: YieldSet: resume_put_mem_mut_with on mismatched keys: dst={:?} src={:?}", x, key);
                panic!();
              }
              match env.pwrite_view(x, xclk, loc) {
                Err(_) => panic!("bug"),
                Ok(e) => {
                  let root = e.root();
                  let v_ty = match e.view().eval_contiguous(e.root_ty) {
                    Err(_) => {
                      println!("ERROR:  Spine::_step: YieldSet: destination is not a zero-copy (contiguous) view");
                      panic!();
                    }
                    Ok(ty) => ty
                  };
                  let v_ptroff = v_ty.pointer_offset();
                  assert_eq!(e.ty, v_ty.as_ref());
                  let mut cel_ = e.cel_.borrow_mut();
                  match &mut *cel_ {
                    &mut Cell_::Phy(_, ref clo, ref mut pcel) => {
                      let optr = pcel.optr;
                      assert_eq!(root, optr);
                      let (pm, addr) = match pcel.find_loc_nocow(xclk, loc) {
                        None => panic!("bug"),
                        Some((pm, addr)) => (pm, addr)
                      };
                      TL_PCTX.with(|pctx| {
                        let (_, icel) = pctx.lookup_pm(pm, addr).unwrap();
                        match icel.mem_borrow_mut() {
                          Err(_) => {
                            println!("ERROR:  Spine::_step: YieldSet: resume_put_mem_mut_with failed to borrow memory for {:?} (replica: pmach={:?} addr={:?})", x, pm, addr);
                            panic!();
                          }
                          Ok(mut mem) => {
                            let offset: usize = v_ptroff.try_into().unwrap();
                            let sz: usize = e.ty.packed_span_bytes().try_into().unwrap();
                            if e.ty != e.root_ty {
                              let start = offset;
                              let end = offset + sz;
                              BorrowRefMut::map_mut(&mut mem, |mem: &mut [u8]| {
                                &mut mem[start .. end]
                              });
                            } else {
                              assert_eq!(offset, 0);
                              assert_eq!(sz, mem.len());
                            }
                            (fun)(&e.ty, &mut *mem);
                          }
                        };
                      });
                      clo.borrow_mut().init_once(xclk, ThunkPtr::opaque());
                    }
                    _ => unreachable!()
                  }
                }
                _ => panic!("bug")
              }
            }
            (_, &SpineResume::Put(key, ty, val)) => {
              if key != x {
                println!("ERROR:  Spine::_step: YieldSet: resume_put on mismatched keys: dst={:?} src={:?}", x, key);
                panic!();
              }
              // FIXME: view.
              let root = match env._lookup_ref_(x) {
                Err(CellDerefErr::View) => panic!("bug"),
                Err(_) => panic!("bug"),
                Ok(e) => {
                  if ty != e.ty {
                    println!("ERROR:  Spine::_step: YieldSet: resume_put on mismatched types: dst={:?} src={:?}", e.ty, ty);
                    panic!();
                  }
                  e.root()
                }
              };
              val._store_to(root, xclk, &*env);
              match env._lookup_ref_(x) {
                Err(CellDerefErr::View) => panic!("bug"),
                Err(_) => panic!("bug"),
                Ok(e) => {
                  let cel_ = e.cel_.borrow();
                  match &*cel_ {
                    &Cell_::Phy(_, ref clo, _) => {
                      clo.borrow_mut().init_once(xclk, ThunkPtr::opaque());
                    }
                    _ => panic!("bug")
                  }
                }
              }
            }
            _ => unimplemented!()
          }
        } else {
          if cfg_debug() { println!("DEBUG: Spine::_step: YieldSet:   yield..."); }
          ret = SpineRet::Yield_(());
        }
        });
      }
      SpineEntry::YieldInit(x, _xclk, loc) => {
        if cfg_debug() { println!("DEBUG: Spine::_step: YieldInit: x={:?} loc={:?} key={:?}", x, loc, item.key()); }
        unimplemented!();
      }
      SpineEntry::Alias(_x, _og) => {}
      SpineEntry::CAlias(_x, _og) => {}
      /*SpineEntry::Snapshot(x, og) => {
        // FIXME FIXME
        //unimplemented!();
      }*/
      SpineEntry::Push(x, _xclk) => {
        TL_CTX.with(|ctx| {
          let env = ctx.env.borrow();
        match env._lookup_view(x) {
          Err(_) => panic!("bug"),
          Ok(e) => {
            let xclk = e.state().clk;
            assert!(!xclk.ctr().is_nil());
          }
        }
        });
      }
      SpineEntry::Initialize(x, _xclk, th) => {
        TL_CTX.with(|ctx| {
        let (xroot, prev_xclk, xclk) = match ctx.env.borrow()._lookup_view(x) {
          Err(_) => panic!("bug"),
          Ok(e) => {
            let root = e.root();
            let base_clk: Clock = self.ctr.get().into();
            let prev_clk = e.state().clk;
            let next_clk = base_clk.max(prev_clk).init_once();
            assert!(prev_clk < next_clk);
            e.clock_sync(prev_clk, next_clk, &*ctx.env.borrow());
            (root, prev_clk, next_clk)
          }
        };
        {
          let tclo = match thunkenv.update.get(&(xroot, xclk)) {
            None => panic!("bug"),
            Some(tclo) => tclo
          };
          assert_eq!(tclo.pthunk, th);
          let te = match thunkenv.thunktab.get(&th) {
            None => panic!("bug"),
            Some(te) => te
          };
          let ret = te.pthunk.initialize(ctr, &tclo.param, &tclo.arg, th, x, prev_xclk, xclk, tclo.pmach);
          match ret {
            Err(ThunkErr::NotImpl) => {
              println!("ERROR: Spine::_step: Initialize: thunk not implemented");
              panic!();
            }
            Err(ThunkErr::Failure) => {
              println!("ERROR: Spine::_step: Initialize: unrecoverable thunk failure");
              panic!();
            }
            Ok(_) => {}
          }
          match ctx.env.borrow()._lookup_view(x) {
            Err(_) => panic!("bug"),
            Ok(e) => {
              let cel_ = e.cel_.borrow();
              match &*cel_ {
                &Cell_::Phy(_, ref clo, _) => {
                  clo.borrow_mut().init_once(xclk, th);
                }
                _ => panic!("bug")
              }
            }
          }
        }
        });
      }
      SpineEntry::Apply(x, _xclk, th) => {
        TL_CTX.with(|ctx| {
        let (xroot, prev_xclk, xclk) = match ctx.env.borrow()._lookup_view(x) {
          Err(_) => panic!("bug"),
          Ok(e) => {
            let root = e.root();
            let base_clk: Clock = self.ctr.get().into();
            let prev_clk = e.state().clk;
            let next_clk = base_clk.max(prev_clk).init_once();
            assert!(prev_clk < next_clk);
            e.clock_sync(prev_clk, next_clk, &*ctx.env.borrow());
            (root, prev_clk, next_clk)
          }
        };
        {
          let tclo = match thunkenv.update.get(&(xroot, xclk)) {
            None => panic!("bug"),
            Some(tclo) => tclo
          };
          if cfg_debug() {
          println!("DEBUG: Spine::_step: Apply: x={:?} xclk={:?} th={:?} tclo={:?}",
              x, xclk, th, tclo);
          }
          assert_eq!(tclo.pthunk, th);
          let te = match thunkenv.thunktab.get(&th) {
            None => panic!("bug"),
            Some(te) => te
          };
          let ret = te.pthunk.apply(ctr, &tclo.param, &tclo.arg, th, x, prev_xclk, xclk, tclo.pmach);
          match ret {
            Err(ThunkErr::NotImpl) => {
              println!("ERROR: Spine::_step: Apply: thunk not implemented");
              panic!();
            }
            Err(ThunkErr::Failure) => {
              println!("ERROR: Spine::_step: Apply: unrecoverable thunk failure");
              panic!();
            }
            Ok(_) => {}
          }
          match ctx.env.borrow()._lookup_view(x) {
            Err(_) => panic!("bug"),
            Ok(e) => {
              let cel_ = e.cel_.borrow();
              match &*cel_ {
                &Cell_::Phy(_, ref clo, _) => {
                  clo.borrow_mut().init_once(xclk, th);
                }
                _ => panic!("bug: Spine::_step: Apply: cel={:?}", cel_.name())
              }
            }
          }
        }
        });
      }
      SpineEntry::Accumulate(x, _xclk, th) => {
        TL_CTX.with(|ctx| {
        let (xroot, prev_xclk, xclk) = match ctx.env.borrow()._lookup_view(x) {
          Err(_) => panic!("bug"),
          Ok(e) => {
            let root = e.root();
            let base_clk: Clock = self.ctr.get().into();
            let prev_clk = e.state().clk;
            let next_clk = base_clk.max(prev_clk).update();
            assert!(prev_clk < next_clk);
            e.clock_sync(prev_clk, next_clk, &*ctx.env.borrow());
            (root, prev_clk, next_clk)
          }
        };
        {
          let tclo = match thunkenv.update.get(&(xroot, xclk)) {
            None => panic!("bug"),
            Some(tclo) => tclo
          };
          assert_eq!(tclo.pthunk, th);
          let te = match thunkenv.thunktab.get(&th) {
            None => panic!("bug"),
            Some(te) => te
          };
          let ret = te.pthunk.accumulate(ctr, &tclo.param, &tclo.arg, th, x, prev_xclk, xclk, tclo.pmach);
          match ret {
            Err(ThunkErr::NotImpl) => {
              println!("ERROR: Spine::_step: Accumulate: thunk not implemented");
              panic!();
            }
            Err(ThunkErr::Failure) => {
              println!("ERROR: Spine::_step: Accumulate: unrecoverable thunk failure");
              panic!();
            }
            Ok(_) => {}
          }
          match ctx.env.borrow()._lookup_view(x) {
            Err(_) => panic!("bug"),
            Ok(e) => {
              let cel_ = e.cel_.borrow();
              match &*cel_ {
                &Cell_::Phy(_, ref clo, _) => {
                  clo.borrow_mut().update(xclk, th);
                }
                _ => panic!("bug")
              }
            }
          }
        }
        });
      }
      SpineEntry::UnsafeWrite(x, _xclk, th) => {
        unimplemented!();
      }
      /*SpineEntry::Seal(x) => {
        TL_CTX.with(|ctx| {
          let env = ctx.env.borrow();
        match env._lookup_view(x) {
          Err(_) => panic!("bug"),
          Ok(e) => {
            assert_eq!(e.state().clk.ctr(), self.ctr.get());
            assert!(!e.state().clk.is_uninit());
            //assert!(!e.state().flag.seal());
            //e.state().flag.set_seal();
          }
        }
        });
      }
      SpineEntry::Unseal(x) => {
        unimplemented!();
      }
      SpineEntry::Unsync(x) => {
        TL_CTX.with(|ctx| {
          let env = ctx.env.borrow();
        match env._lookup_view(x) {
          Err(_) => panic!("bug"),
          Ok(e) => {
            assert_eq!(e.state().clk.ctr(), self.ctr.get());
            //assert!(e.state().flag.seal());
            // FIXME FIXME
            unimplemented!()
          }
        }
        });
      }*/
      SpineEntry::Live(x) => {
        if cfg_debug() { println!("DEBUG: Spine::_step: Live: x={:?}", x); }
        TL_CTX.with(|ctx| {
          let env = ctx.env.borrow();
          assert!(!env.unlive.borrow().contains(&x));
        });
      }
      SpineEntry::Unlive(x) => {
        if cfg_debug() { println!("DEBUG: Spine::_step: Unlive: x={:?}", x); }
        TL_CTX.with(|ctx| {
          let env = ctx.env.borrow();
          env.unlive.borrow_mut().insert(x);
        });
      }
      SpineEntry::Bot => {
        ret = SpineRet::Bot;
      }
      e => panic!("bug: Spine::_step: unimplemented: {:?}", e)
    }
    let t1 = Stopwatch::tl_stamp();
    let d = t1 - t0;
    //println!("DEBUG: Spine::_step:   t1={}.{:09} s", t1.s(), t1.sub_ns());
    if cfg_debug() { println!("DEBUG: Spine::_step:   d={:.09} s", d); }
    ret
  }

  pub fn _debug_dump(&self) {
    println!("DEBUG: Spine::_debug_dump: ctr={:?} ctlp={} hltp={} curp={}",
        self.ctr.get(), self.ctlp.get(), self.hltp.get(), self.curp.get());
    for (i, e) in self.log.borrow().iter().enumerate() {
      println!("DEBUG: Spine::_debug_dump: log[{}]={:?}", i, e);
    }
  }
}
