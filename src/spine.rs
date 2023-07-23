use crate::algo::{HashMap, HashSet, BTreeMap, BTreeSet};
use crate::cell::*;
use crate::clock::{Counter, Clock, TotalClock};
use crate::ctx::*;
use crate::panick::{panick_wrap};
use crate::pctx::{TL_PCTX, Locus, MemReg};
use crate::thunk::*;
use crate::thunk::op::{SetScalarFutThunkSpec};
use crate::util::time::{Stopwatch};

use std::any::{Any};
use std::cell::{Cell, RefCell};
use std::cmp::{Ordering};
use std::mem::{swap};

pub const ADD_CODE:             u8 = b'A';
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
pub const UNSAFE_WRITE_CODE:    u8 = b'!';

pub enum SpineResume<'a> {
  _Top,
  PutMemV(CellPtr, &'a dyn Any),
  PutMemF(CellPtr, &'a dyn Fn(CellType, MemReg, )),
}

impl<'a> SpineResume<'a> {
  pub fn key(&self) -> Option<CellPtr> {
    let ret = match self {
      &SpineResume::_Top => {
        None
      }
      &SpineResume::PutMemV(key, val) => {
        Some(key)
      }
      &SpineResume::PutMemF(key, fun) => {
        Some(key)
      }
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
  ICacheMux(CellPtr),
  OIntro(CellPtr, CellPtr),
  Intro(CellPtr, Clock),
  Uninit(CellPtr, Clock),
  //IntroFin(CellPtr),
  YieldSet(CellPtr, Clock, Locus),
  YieldInit(CellPtr, Clock, Locus),
  Alias(CellPtr, CellPtr),
  CAlias(CellPtr, CellPtr),
  Snapshot(CellPtr, CellPtr),
  //Snapshot(CellPtr, CellPtr, CellPtr, /*Clock*/),
  //Keep(CellPtr),
  PushSeal(CellPtr, Clock),
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
  Unlive(CellPtr),
  Unsync(CellPtr),
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
      &SpineEntry::ICacheMux(..)  => SpineEntryName::ICacheMux,
      &SpineEntry::OIntro(..)     => SpineEntryName::OIntro,
      &SpineEntry::Intro(..)      => SpineEntryName::Intro,
      &SpineEntry::Uninit(..)     => SpineEntryName::Uninit,
      &SpineEntry::YieldSet(..)   => SpineEntryName::YieldSet,
      &SpineEntry::YieldInit(..)  => SpineEntryName::YieldInit,
      &SpineEntry::Alias(..)      => SpineEntryName::Alias,
      &SpineEntry::CAlias(..)     => SpineEntryName::CAlias,
      &SpineEntry::Snapshot(..)   => SpineEntryName::Snapshot,
      &SpineEntry::PushSeal(..)   => SpineEntryName::PushSeal,
      &SpineEntry::Initialize(..) => SpineEntryName::Initialize,
      &SpineEntry::Apply(..)      => SpineEntryName::Apply,
      &SpineEntry::Accumulate(..) => SpineEntryName::Accumulate,
      &SpineEntry::UnsafeWrite(..) => SpineEntryName::UnsafeWrite,
      &SpineEntry::Seal(..)       => SpineEntryName::Seal,
      &SpineEntry::Unseal(..)     => SpineEntryName::Unseal,
      //&SpineEntry::Eval(..)       => SpineEntryName::Eval,
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
  ICacheMux,
  OIntro,
  Intro,
  Uninit,
  YieldSet,
  YieldInit,
  Alias,
  CAlias,
  Snapshot,
  PushSeal,
  Initialize,
  Apply,
  Accumulate,
  UnsafeWrite,
  Seal,
  Unseal,
  //Eval,
  //Uneval,
  Unlive,
  Unsync,
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

#[derive(Clone, Copy, Debug)]
pub enum SpineEnvGet {
  Value(CellPtr, Clock),
  ReqAlias(CellPtr, CellPtr),
}

#[derive(Clone, Default, Debug)]
pub struct SpineEnv {
  // TODO TODO
  pub ctr:      Counter,
  //pub alias:    HashMap<CellPtr, CellPtr>,
  //pub aroot:    RefCell<HashMap<CellPtr, CellPtr>>,
  //pub state:    HashMap<CellPtr, SpineCellState>,
  pub state:    HashMap<CellPtr, SpineEnvEntry>,
  // FIXME
  //pub set:      BTreeSet<(MCellPtr, CellPtr, TotalClock)>,
  //pub map:      BTreeMap<(MCellPtr, CellPtr, TotalClock), (CellPtr, Clock)>,
  // FIXME: alias-consistent map.
  //pub map:      BTreeMap<(MCellPtr, CellPtr, TotalClock), SpineEnvMapEntry>,
  pub map:      BTreeMap<(MCellPtr, CellPtr), SpineEnvMapEntry2>,
  pub arg:      Vec<(CellPtr, Clock)>,
  //pub out:      Vec<(CellPtr, Clock)>,
  pub update:   HashMap<(CellPtr, Clock), (CellPtr, Vec<(CellPtr, Clock)>)>,
  /*pub iapply:   HashMap<CellPtr, (u32, ThunkPtr, Clock)>,
  pub apply_:   HashMap<CellPtr, Vec<(u32, ThunkPtr, Clock)>>,
  pub bwd:      HashMap<CellPtr, Clock>,
  //pub bwd:      HashSet<(CellPtr, Clock)>,
  pub gradr:    HashMap<(CellPtr, CellPtr, Clock), CellPtr>,*/
}

impl SpineEnv {
  pub fn reset(&mut self, ctr: Counter) {
    println!("DEBUG: SpineEnv::reset: ctr={:?}", ctr);
    self.ctr = ctr;
    //self.alias.clear();
    self.state.clear();
    //self.set.clear();
    self.map.clear();
    self.arg.clear();
    //self.out.clear();
    self.update.clear();
    /*self.iapply.clear();
    self.apply_.clear();
    self.bwd.clear();
    self.gradr.clear();*/
  }

  pub fn _deref(&self, query: CellPtr) -> CellPtr {
    let mut cursor = query;
    loop {
      /*match self.alias.get(&cursor) {
        None => return cursor,
        Some(&next) => {
          cursor = next;
        }
      }*/
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

  /*//pub fn _get(&self, mx: MCellPtr, k: CellPtr, kclk: Clock) -> Option<(CellPtr, Clock)> {}
  pub fn _get(&self, mx: MCellPtr, k: CellPtr, kclk: Clock) -> Option<SpineEnvGet> {
    let kroot = self._deref(k);
    if k != kroot {
      match (self.map.get(&(mx, k, kclk.into())), self.map.get(&(mx, kroot, kclk.into()))) {
        (None, None) => None,
        (None, Some(&SpineEnvMapEntry::Value(vroot, _))) => {
          Some(SpineEnvGet::ReqAlias(kroot, vroot))
        }
        (Some(&SpineEnvMapEntry::Alias(v)), Some(&SpineEnvMapEntry::Value(vroot, vclk))) => {
          assert_eq!(self._deref(v), vroot);
          Some(SpineEnvGet::Value(v, vclk.into()))
        }
        _ => panic!("bug")
      }
    } else {
      match self.map.get(&(mx, k, kclk.into())) {
        None => None,
        Some(&SpineEnvMapEntry::Value(v, vclk)) => {
          Some(SpineEnvGet::Value(v, vclk.into()))
        }
        _ => panic!("bug")
      }
    }
  }*/

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
    // FIXME FIXME
    let e_sp = log.borrow()[sp as usize];
    println!("DEBUG: SpineEnv::step: idx={} e={:?}", sp, &e_sp);
    match e_sp {
      SpineEntry::_Top => {}
      SpineEntry::OIntro(x, _) => {
        /*self.cache.insert(x, sp);*/
        // FIXME FIXME
        unimplemented!();
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
        /*if k != kroot {
          assert!(v != vroot);
          match self_.map.insert((mx, kroot, kclk.into()), SpineEnvMapEntry::Value(vroot, vclk)) {
            None => {}
            Some(SpineEnvMapEntry::Value(ovroot, ovclk)) => {
              assert_eq!(ovroot, vroot);
              if !(ovclk <= vclk) {
                println!("DEBUG: SpineEnv::step: Add2 mx={:?} k={:?} v={:?}", mx, k, v);
                println!("DEBUG: SpineEnv::step:   kroot={:?} kclk={:?}", kroot, kclk);
                println!("DEBUG: SpineEnv::step:   vroot={:?} vclk={:?} ovclk={:?}", vroot, vclk, ovclk);
              }
              assert!(ovclk <= vclk);
            }
            _ => panic!("bug")
          }
          match self_.map.insert((mx, k, kclk.into()), SpineEnvMapEntry::Alias(v)) {
            None => {}
            Some(SpineEnvMapEntry::Alias(ov)) => {
              assert_eq!(ov, v);
            }
            _ => panic!("bug")
          }
        } else {
          match self_.map.insert((mx, k, kclk.into()), SpineEnvMapEntry::Value(v, vclk)) {
            None => {}
            Some(SpineEnvMapEntry::Value(ov, ovclk)) => {
              assert_eq!(ov, v);
              assert!(ovclk <= vclk);
            }
            _ => panic!("bug")
          }
        }*/
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
      SpineEntry::AdjMap(allsrc, sink) => {
        let self_ = this.borrow();
        //match self_.map.range(&(sink, CellPtr::nil(), TotalClock::default()) .. ).next() {}
        match self_.map.range(&(sink, CellPtr::nil()) .. ).next() {
          None => {
            return;
          }
          //Some((&(mx, _, _), _)) => {}
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
            SpineEntry::Add(..) |
            SpineEntry::Add2(..) |
            SpineEntry::Cache(..) |
            SpineEntry::Intro(..) |
            SpineEntry::Uninit(..) |
            SpineEntry::YieldSet(..) |
            SpineEntry::YieldInit(..) |
            SpineEntry::Alias(..) |
            SpineEntry::CAlias(..) |
            //SpineEntry::Snapshot(..) |
            SpineEntry::PushSeal(..) => {}
            /*SpineEntry::Alias(y, yclk, x, xclk) => {
              let self_ = this.borrow();
              match self_.map.get(&(sink, y, yclk.into()))
                    .or_else(|| self_.map.get(&(allsrc, y, yclk.into())))
              {
                None => {}
                Some(&(dy, _dyclk)) => {
                  // FIXME
                  //let x_ty = ctx_lookup_type(x);
                  /*
                  let dx =
                      self_.map.get(&(sink, x, xclk.into()))
                      .or_else(|| self_.map.get(&(allsrc, x, xclk.into())))
                      .map(|&(dx, _)| dx)
                      .unwrap_or_else(|| {
                        let x_ty = ctx_lookup_type(x);
                        ctx_insert(x_ty)
                      });
                  */
                  unimplemented!();
                }
              }
            }*/
            SpineEntry::Initialize(y, yclk, th) |
            SpineEntry::Apply(y, yclk, th) => {
              {
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
                println!("DEBUG: SpineEnv::step: AdjMap:   const_ yroot={:?} y={:?}", yroot, y);
                continue 'for_sp;
              }
              // FIXME
              match self_._get(sink, y, yclk)
                    .or_else(|| self_._get(allsrc, y, yclk))
              {
                None => {}
                Some(SpineEnvGet::ReqAlias(..)) => {
                  // FIXME
                  panic!("bug");
                }
                Some(SpineEnvGet::Value(dy, _dyclk)) => {
                  match self_.update.get(&(yroot, yclk.into())) {
                    None => panic!("bug"),
                    Some(&(y_ref, ref arg)) => {
                      println!("DEBUG: SpineEnv::step: AdjMap:   yroot={:?} y_ref={:?} arg={:?}", yroot, y_ref, arg);
                      let arg = arg.clone();
                      let mut arg_adj: Vec<CellPtr> = Vec::with_capacity(arg.len());
                      drop(self_);
                      {
                        //let ctr = ctr.unwrap();
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
                            println!("DEBUG: SpineEnv::step: AdjMap:   const_ xroot={:?} x={:?}", xroot, x);
                            arg_adj.push(CellPtr::nil());
                            continue 'for_x;
                          }
                          let dx =
                              self_._get(sink, x, xclk.into())
                              .or_else(|| self_._get(allsrc, x, xclk.into()))
                              //.map(|&(dx, _)| dx)
                              .and_then(|get| match get {
                                SpineEnvGet::Value(dx, _) => Some(dx),
                                SpineEnvGet::ReqAlias(..) => None
                              })
                              /*.unwrap_or_else(|| {
                                let x_ty = ctx_lookup_type(x);
                                ctx_insert(x_ty)
                              });*/
                              .unwrap_or_else(|| TL_CTX.with(|ctx| {
                                drop(self_);
                                //let x_ty = ctx.env.lookup_ref(x).map(|e| e.ty.clone()).unwrap();
                                let env = ctx.env.borrow();
                                match env.lookup_ref(x) {
                                  None => panic!("bug"),
                                  Some(e) => {
                                    let x_ty = e.ty.clone();
                                    assert_eq!(xroot, e.root);
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
                                            let xroot_ty = match ctx.env.borrow().lookup_ref(xroot) {
                                              None => panic!("bug"),
                                              Some(e) => e.ty.clone()
                                            };
                                            //ctx_insert(xroot_ty)
                                            let dxroot = ctr.unwrap().fresh_cel();
                                            println!("DEBUG: SpineEnv::step: AdjMap:   fresh: xroot={:?} dxroot={:?} ty={:?}", xroot, dxroot, xroot_ty);
                                            ctx.env.borrow_mut().insert_top(dxroot, xroot_ty);
                                            dxroot
                                          });
                                      /*{
                                        let sp = {
                                          let sp = curp.get();
                                          curp.set(sp + 1);
                                          let e = SpineEntry::Add2(allsrc, xroot, Clock::default(), dxroot, Clock::default());
                                          log.borrow_mut().push(e);
                                          sp
                                        };
                                        SpineEnv::step(this, sp, curp, log, ctr, thunkenv);
                                      }*/
                                      //ctx_alias(dxroot, x_ty)
                                      let dx = ctr.unwrap().fresh_cel();
                                      println!("DEBUG: SpineEnv::step: AdjMap:   fresh: x={:?} dx={:?} ty={:?}", x, dx, x_ty);
                                      println!("DEBUG: SpineEnv::step: AdjMap:          xroot={:?} dxroot={:?}", xroot, dxroot);
                                      ctx.env.borrow_mut().insert_alias(dx, x_ty, dxroot);
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
                                      //ctx_insert(x_ty)
                                      let dx = ctr.unwrap().fresh_cel();
                                      println!("DEBUG: SpineEnv::step: AdjMap:   fresh: x={:?} dx={:?} ty={:?}", x, dx, x_ty);
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
                      {
                        let thunkenv = thunkenv.map(|env| env.borrow()).unwrap();
                        thunkenv._set_accumulate_in_place(true);
                        thunkenv._set_assume_uninit_zero(true);
                        match thunkenv.thunktab.get(&th) {
                          None => panic!("bug"),
                          Some(te) => {
                            let pthunk = te.pthunk.clone();
                            drop(thunkenv);
                            println!("DEBUG: SpineEnv::step: AdjMap:   pop adj: th={:?} {:?} x={:?} y={:?} yclk={:?} dy={:?} dx={:?}",
                                th, pthunk.spec_.debug_name(), &arg, y, yclk, dy, &arg_adj);
                            match pthunk.spec_.pop_adj(&arg, y, yclk, ThunkMode::Apply0, dy, &mut arg_adj) {
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
                      {
                        let thunkenv = thunkenv.map(|env| env.borrow()).unwrap();
                        thunkenv._set_accumulate_in_place(false);
                        thunkenv._set_assume_uninit_zero(false);
                      }
                      for (&(x, xclk), &dx) in arg.iter().zip(arg_adj.iter()).rev() {
                        if dx.is_nil() {
                          println!("DEBUG: SpineEnv::step: AdjMap:   skip: x={:?} dx={:?}", x, dx);
                          continue;
                        }
                        println!("DEBUG: SpineEnv::step: AdjMap:   add2: allsrc={:?} x={:?} dx={:?}", allsrc, x, dx);
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
              // TODO TODO
              unimplemented!();
              /*
              let self_ = this.borrow();
              match self_.map.get(&(sink, y, yclk.into()))
                    .or_else(|| self_.map.get(&(allsrc, y, yclk.into())))
              {
                None => {}
                Some(&(dy, _dyclk)) => {
                  drop(self_);
                  let mut yclk = yclk;
                  while yclk.up >= 0 {
                    let self_ = this.borrow();
                    match self_.update.get(&(y, yclk.into())) {
                      None => panic!("bug"),
                      Some(&(_, ref arg)) => {
                        let arg = arg.clone();
                        let mut arg_adj = Vec::with_capacity(arg.len());
                        {
                          let ctr = ctr.unwrap();
                          for &(x, xclk) in arg.iter().rev() {
                            // FIXME: alias x.
                            let dx =
                                self_.map.get(&(sink, x, xclk.into()))
                                .or_else(|| self_.map.get(&(allsrc, x, xclk.into())))
                                .map(|&(dx, _)| dx)
                                .unwrap_or_else(|| {
                                  let x_ty = ctx_lookup_type(x);
                                  ctx_insert(x_ty)
                                });
                            arg_adj.push(dx);
                          }
                          arg_adj.reverse();
                        }
                        drop(self_);
                        {
                          let thunkenv = thunkenv.map(|env| env.borrow()).unwrap();
                          thunkenv._set_accumulate_in_place(true);
                          thunkenv._set_assume_uninit_zero(true);
                          match thunkenv.thunktab.get(&th) {
                            None => panic!("bug"),
                            Some(te) => {
                            let pthunk = te.pthunk.clone();
                            drop(thunkenv);
                              match pthunk.spec_.pop_adj(&arg, y, yclk, dy, &mut arg_adj) {
                                Err(_) => {
                                  println!("ERROR: SpineEnv::step: adj failure ({:?} th={:?} x={:?} y={:?} yclk={:?} dy={:?} dx={:?})",
                                      e_sp.name(), th, &arg, y, yclk, dy, &arg_adj);
                                  panic!();
                                }
                                Ok(_) => {}
                              }
                            }
                          }
                        }
                        {
                          let thunkenv = thunkenv.map(|env| env.borrow()).unwrap();
                          thunkenv._set_accumulate_in_place(false);
                          thunkenv._set_assume_uninit_zero(false);
                        }
                        for (&(x, xclk), &dx) in arg.iter().zip(arg_adj.iter()).rev() {
                          let sp = {
                            let sp = curp.get();
                            curp.set(sp + 1);
                            let e = SpineEntry::Add2(allsrc, x, xclk, dx, Clock::default());
                            log.borrow_mut().push(e);
                            sp
                          };
                          SpineEnv::step(this, sp, curp, log, ctr, thunkenv);
                        }
                      }
                    }
                    yclk.up -= 1;
                  }
                }
              }
              */
            }
            SpineEntry::UnsafeWrite(y, yclk, th) => {
              println!("ERROR: SpineEnv::step: cannot differentiate through UnsafeWrite (y={:?} yclk={:?} th={:?})",
                  y, yclk, th);
              panic!();
            }
            _ => {
              println!("DEBUG: SpineEnv::step: unimplemented: bp={} e={:?}", bp, e_sp);
              let min_sp = if sp < 10 { 0 } else { sp - 10 };
              for idx in (min_sp ..= sp).rev() {
                let e = log.borrow()[idx as usize];
                println!("DEBUG: SpineEnv::step: unimplemented:   idx={} e={:?}", idx, e);
              }
              panic!("bug");
            }
          }
          //rev_env.unstep(&log[sp as usize]);
        }
        println!("DEBUG: SpineEnv::step: AdjMap: done");
      }
      /*SpineEntry::AdjMap2(some_src, src_mask, sink) => {
        // FIXME FIXME
        unimplemented!();
      }*/
      SpineEntry::DualMap(allsink, src) => {
        // FIXME FIXME
        unimplemented!();
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
            //assert_eq!(state.mode, CellMode::_Top);
            //assert!(!state.flag.intro());
            assert!(!state.flag.seal());
            /*assert_eq!(state.clk.up, 0);*/
            //assert!(state.clk.is_uninit());
            //state.mode = CellMode::Aff;
            //state.flag.set_intro();
            let next_clk = base_clk.init_once();
            assert!(state.clk < next_clk);
            state.clk = next_clk;
            drop(state);
            /*match self_.apply_.get(&x) {
              None => {
                self_.apply_.insert(x, Vec::new());
              }
              _ => {}
            }
            match self_.apply_.get_mut(&x) {
              None => panic!("bug"),
              Some(thlist) => {
                assert_eq!(thlist.len(), next_clk.up as _);
                thlist.push((sp, ThunkPtr::opaque(), next_clk));
              }
            }*/
            log.borrow_mut()[sp as usize] = SpineEntry::Cache(x, next_clk);
          }
        }
      }
      SpineEntry::ICacheMux(x) => {
        // FIXME FIXME
        unimplemented!();
      }
      SpineEntry::Intro(x, _xclk) => {
        let mut self_ = this.borrow_mut();
        match self_._lookup(x) {
          None => {
            self_.state.insert(x, CellState::default().into());
          }
          _ => {}
        }
        let base_clk: Clock = self_.ctr.into();
        match self_._lookup_mut(x) {
          None => panic!("bug"),
          Some((_, state)) => {
            assert_eq!(state.mode, CellMode::_Top);
            assert!(!state.flag.intro());
            assert!(!state.flag.seal());
            /*assert_eq!(state.clk.up, 0);*/
            assert!(state.clk.is_uninit());
            state.mode = CellMode::Aff;
            state.flag.set_intro();
            let next_clk = base_clk;
            state.clk = next_clk;
            log.borrow_mut()[sp as usize] = SpineEntry::Intro(x, next_clk);
          }
        }
      }
      SpineEntry::Uninit(x, _xclk) => {
        let mut self_ = this.borrow_mut();
        let base_clk: Clock = self_.ctr.into();
        match self_._lookup_mut(x) {
          None => panic!("bug"),
          Some((_, state)) => {
            assert_eq!(state.mode, CellMode::_Top);
            assert!(!state.flag.intro());
            assert!(!state.flag.seal());
            /*assert_eq!(state.clk.up, 0);*/
            // FIXME
            /*assert!(state.clk.is_uninit());*/
            state.mode = CellMode::Init;
            state.flag.set_intro();
            let next_clk = base_clk;
            state.clk = next_clk;
            log.borrow_mut()[sp as usize] = SpineEntry::Uninit(x, next_clk);
          }
        }
      }
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
            //assert_eq!(state.mode, CellMode::_Top);
            //assert!(!state.flag.intro());
            assert!(!state.flag.seal());
            /*assert_eq!(state.clk.up, 0);*/
            assert!(state.clk.is_uninit());
            //state.mode = CellMode::Aff;
            //state.flag.set_intro();
            let next_clk = base_clk.init_once();
            assert!(state.clk < next_clk);
            state.clk = next_clk;
            drop(state);
            /*match self_.apply_.get(&x) {
              None => {
                self_.apply_.insert(x, Vec::new());
              }
              _ => {}
            }
            match self_.apply_.get_mut(&x) {
              None => panic!("bug"),
              Some(thlist) => {
                if thlist.len() != next_clk.up as _ {
                  println!("DEBUG: SpineEnv::step: thlist.len={} next clk up={}", thlist.len(), next_clk.up);
                }
                assert_eq!(thlist.len(), next_clk.up as _);
                thlist.push((sp, ThunkPtr::opaque(), next_clk));
              }
            }*/
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
            //assert_eq!(state.mode, CellMode::_Top);
            //assert!(!state.flag.intro());
            assert!(!state.flag.seal());
            /*assert_eq!(state.clk.up, 0);*/
            // FIXME
            /*assert!(state.clk.is_uninit());*/
            //state.mode = CellMode::Init;
            //state.flag.set_intro();
            let next_clk = base_clk.init_once();
            assert!(state.clk < next_clk);
            state.clk = next_clk;
            drop(state);
            // FIXME FIXME
            //assert!(self_.iapply.insert(x, (sp, ThunkPtr::opaque(), next_clk)).is_none());
            log.borrow_mut()[sp as usize] = SpineEntry::YieldInit(x, next_clk, _loc);
          }
        }
      }
      SpineEntry::Alias(x, og) => {
        let mut self_ = this.borrow_mut();
        // FIXME FIXME: prefer unification.
        /*match self_.state.get(&og) {
          None => panic!("bug"),
          Some(state) => {
            let state = state.clone();
            self_.state.insert(x, state);
          }
        }*/
        //assert!(self_.alias.insert(x, og).is_none());
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
      SpineEntry::Snapshot(x, og) => {
        let mut self_ = this.borrow_mut();
        // FIXME FIXME
        match self_._lookup(og) {
          None => panic!("bug"),
          Some((_, state)) => {
            let state = state.clone();
            self_.state.insert(x, state.into());
          }
        }
      }
      SpineEntry::PushSeal(x, _xclk) => {
        let mut self_ = this.borrow_mut();
        match self_._lookup_mut(x) {
          None => {
            //println!("DEBUG: PushSeal: x={:?} spine env={:?}", x, self);
            panic!("bug");
          }
          Some((_, state)) => {
            //assert!(state.flag.intro());
            // NB: late/lazy idempotent seal.
            /*assert!(!state.flag.seal());*/
            let next_clk = state.clk;
            state.flag.set_seal();
            drop(state);
            self_.arg.push((x, next_clk));
            log.borrow_mut()[sp as usize] = SpineEntry::PushSeal(x, next_clk);
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
            //assert_eq!(state.mode, CellMode::_Top);
            //assert!(!state.flag.intro());
            assert!(!state.flag.seal());
            /*assert_eq!(state.clk.up, 0);*/
            // FIXME
            /*assert!(state.clk.is_uninit());*/
            //state.mode = CellMode::Init;
            //state.flag.set_intro();
            //let next_clk = state.clk.init_once();
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
            //assert_eq!(state.mode, CellMode::Aff);
            //assert!(state.flag.intro());
            assert!(!state.flag.seal());
            /*assert_eq!(state.clk.up, 0);*/
            //assert!(state.clk.is_uninit());
            //let next_clk = state.clk.init_once();
            let next_clk = base_clk.max(state.clk).init_once();
            assert!(state.clk < next_clk);
            state.clk = next_clk;
            drop(state);
            /*match self_.apply_.get(&x) {
              None => {
                self_.apply_.insert(x, Vec::new());
              }
              _ => {}
            }
            match self_.apply_.get_mut(&x) {
              None => panic!("bug"),
              Some(thlist) => {
                assert_eq!(thlist.len(), next_clk.up as _);
                thlist.push((sp, th, next_clk));
              }
            }*/
            let mut arg = Vec::new();
            swap(&mut self_.arg, &mut arg);
            assert!(self_.update.insert((xroot, next_clk), (x, arg)).is_none());
            /*let arg_ = arg.clone();
            if let Some(oarg) = self_.update.insert(th, arg) {
              println!("DEBUG: SpineEnv::step: Apply: x={:?} xclk={:?} th={:?} arg={:?} oarg={:?}",
                  x, next_clk, th, arg_, oarg);
              panic!("bug");
            }*/
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
            //assert_eq!(state.mode, CellMode::Init);
            //assert!(state.mode != CellMode::Aff);
            //assert!(state.flag.intro());
            assert!(!state.flag.seal());
            //state.mode = CellMode::Init;
            //state.flag.set_intro();
            //let next_clk = state.clk.update();
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
            /*assert!(state.mode != CellMode::_Top);*/
            assert!(state.flag.intro());
            // NB: idempotent seal.
            /*assert!(!state.flag.seal());*/
            state.flag.set_seal();
          }
        }
      }
      SpineEntry::Unseal(x) => {
        let mut self_ = this.borrow_mut();
        match self_._lookup_mut(x) {
          None => panic!("bug"),
          Some((_, state)) => {
            /*assert!(state.mode != CellMode::_Top);*/
            // NB: idempotent unseal.
            /*assert!(state.flag.seal());*/
            state.flag.unset_seal();
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
}

pub struct Spine {
  pub ctr:  Counter,
  pub ctlp: u32,
  pub hltp: u32,
  //pub curp: u32,
  pub curp: Cell<u32>,
  pub retp: u32,
  //pub log:  Vec<SpineEntry>,
  pub log:  RefCell<Vec<SpineEntry>>,
  //pub cur_env:  SpineEnv,
  pub cur_env:  RefCell<SpineEnv>,
}

impl Default for Spine {
  fn default() -> Spine {
    Spine{
      ctr:  Counter::default(),
      ctlp: 0,
      hltp: 0,
      //curp: 0,
      curp: Cell::new(0),
      retp: u32::max_value(),
      //log:  Vec::new(),
      log:  RefCell::new(Vec::new()),
      //cur_env:  SpineEnv::default(),
      cur_env:  RefCell::new(SpineEnv::default()),
    }
  }
}

impl Spine {
  pub fn _reset(&mut self) {
    println!("DEBUG: Spine::_reset: ctr={:?} ctlp={} hltp={} curp={}",
        self.ctr, self.ctlp, self.hltp, self.curp.get());
    self.ctr = self.ctr.advance();
    self.ctlp = 0;
    self.hltp = 0;
    self.curp.set(0);
    self.retp = u32::max_value();
    self.log.borrow_mut().clear();
    self.cur_env.borrow_mut().reset(self.ctr);
  }

  /*pub fn _fastfwd(&mut self, sp: u32) {
    // FIXME: for adj, dual.
    let start_sp = sp + 1;
    let end_sp = self.log.len() as _;
    self.curp = end_sp;
    for sp in start_sp .. end_sp {
      self.cur_env.step(sp, &self.log);
    }
  }*/

  pub fn adj_map(&self, allsrc: MCellPtr, sink: MCellPtr, ctr: &CtxCtr, thunkenv: &RefCell<CtxThunkEnv>) {
    //let sp = self.curp;
    //self.curp += 1;
    let sp = self.curp.get();
    self.curp.set(sp + 1);
    let e = SpineEntry::AdjMap(allsrc, sink);
    self.log.borrow_mut().push(e);
    SpineEnv::step(&self.cur_env, sp, &self.curp, &self.log, Some(ctr), Some(thunkenv));
    //self.curp = self.log.borrow().len() as _;
    //self.curp.set(self.log.borrow().len() as _);
    assert_eq!(self.curp.get(), self.log.borrow().len() as _);
  }

  pub fn dual_map(&self, allsink: MCellPtr, src: MCellPtr, ctr: &CtxCtr, thunkenv: &RefCell<CtxThunkEnv>) {
    //let sp = self.curp;
    //self.curp += 1;
    let sp = self.curp.get();
    self.curp.set(sp + 1);
    let e = SpineEntry::DualMap(allsink, src);
    self.log.borrow_mut().push(e);
    SpineEnv::step(&self.cur_env, sp, &self.curp, &self.log, Some(ctr), Some(thunkenv));
    //self.curp = self.log.borrow().len() as _;
    //self.curp.set(self.log.borrow().len() as _);
    assert_eq!(self.curp.get(), self.log.borrow().len() as _);
  }

  pub fn add(&self, mx: MCellPtr, x: CellPtr) {
    //let sp = self.curp;
    //self.curp += 1;
    let sp = self.curp.get();
    self.curp.set(sp + 1);
    let e = SpineEntry::Add(mx, x, Clock::default());
    self.log.borrow_mut().push(e);
    SpineEnv::step(&self.cur_env, sp, &self.curp, &self.log, None, None);
  }

  pub fn add2(&self, mx: MCellPtr, k: CellPtr, v: CellPtr) {
    //let sp = self.curp;
    //self.curp += 1;
    let sp = self.curp.get();
    self.curp.set(sp + 1);
    let e = SpineEntry::Add2(mx, k, Clock::default(), v, Clock::default());
    self.log.borrow_mut().push(e);
    SpineEnv::step(&self.cur_env, sp, &self.curp, &self.log, None, None);
  }

  pub fn cache_aff(&self, x: CellPtr) {
    //let sp = self.curp;
    //self.curp += 1;
    let sp = self.curp.get();
    self.curp.set(sp + 1);
    let e = SpineEntry::Cache(x, Clock::default());
    self.log.borrow_mut().push(e);
    SpineEnv::step(&self.cur_env, sp, &self.curp, &self.log, None, None);
  }

  pub fn opaque(&self, x: CellPtr, og: CellPtr) {
    //let sp = self.curp;
    //self.curp += 1;
    let sp = self.curp.get();
    self.curp.set(sp + 1);
    let e = SpineEntry::OIntro(x, og);
    self.log.borrow_mut().push(e);
    SpineEnv::step(&self.cur_env, sp, &self.curp, &self.log, None, None);
  }

  /*pub fn intro_aff(&self, x: CellPtr) {
    //let sp = self.curp;
    //self.curp += 1;
    let sp = self.curp.get();
    self.curp.set(sp + 1);
    let e = SpineEntry::Intro(x, Clock::default());
    self.log.borrow_mut().push(e);
    SpineEnv::step(&self.cur_env, sp, &self.curp, &self.log, None, None);
  }

  pub fn uninit(&self, x: CellPtr) {
    let sp = self.curp.get();
    self.curp.set(sp + 1);
    let e = SpineEntry::Uninit(x, Clock::default());
    self.log.borrow_mut().push(e);
    SpineEnv::step(&self.cur_env, sp, &self.curp, &self.log, None, None);
  }*/

  pub fn init_cache_mux(&self, x: CellPtr) {
    //let sp = self.curp;
    //self.curp += 1;
    let sp = self.curp.get();
    self.curp.set(sp + 1);
    let e = SpineEntry::ICacheMux(x);
    self.log.borrow_mut().push(e);
    SpineEnv::step(&self.cur_env, sp, &self.curp, &self.log, None, None);
  }

  pub fn yield_set(&self, x: CellPtr, loc: Locus) {
    //let sp = self.curp;
    //self.curp += 1;
    let sp = self.curp.get();
    self.curp.set(sp + 1);
    let e = SpineEntry::YieldSet(x, Clock::default(), loc);
    self.log.borrow_mut().push(e);
    SpineEnv::step(&self.cur_env, sp, &self.curp, &self.log, None, None);
  }

  pub fn yield_init(&self, x: CellPtr, loc: Locus) {
    //let sp = self.curp;
    //self.curp += 1;
    let sp = self.curp.get();
    self.curp.set(sp + 1);
    let e = SpineEntry::YieldInit(x, Clock::default(), loc);
    self.log.borrow_mut().push(e);
    SpineEnv::step(&self.cur_env, sp, &self.curp, &self.log, None, None);
  }

  pub fn alias(&self, x: CellPtr, og: CellPtr) {
    //let sp = self.curp;
    //self.curp += 1;
    let sp = self.curp.get();
    self.curp.set(sp + 1);
    let e = SpineEntry::Alias(x, og);
    self.log.borrow_mut().push(e);
    SpineEnv::step(&self.cur_env, sp, &self.curp, &self.log, None, None);
  }

  pub fn const_(&self, x: CellPtr, og: CellPtr) {
    let sp = self.curp.get();
    self.curp.set(sp + 1);
    let e = SpineEntry::CAlias(x, og);
    self.log.borrow_mut().push(e);
    SpineEnv::step(&self.cur_env, sp, &self.curp, &self.log, None, None);
  }

  pub fn snapshot(&self, x: CellPtr, og: CellPtr) {
    //let sp = self.curp;
    //self.curp += 1;
    let sp = self.curp.get();
    self.curp.set(sp + 1);
    let e = SpineEntry::Snapshot(x, og);
    self.log.borrow_mut().push(e);
    SpineEnv::step(&self.cur_env, sp, &self.curp, &self.log, None, None);
  }

  pub fn push_seal(&self, x: CellPtr) {
    //let sp = self.curp;
    //self.curp += 1;
    let sp = self.curp.get();
    self.curp.set(sp + 1);
    let e = SpineEntry::PushSeal(x, Clock::default());
    self.log.borrow_mut().push(e);
    SpineEnv::step(&self.cur_env, sp, &self.curp, &self.log, None, None);
  }

  pub fn initialize(&self, y: CellPtr, ith: ThunkPtr) {
    //let sp = self.curp;
    //self.curp += 1;
    let sp = self.curp.get();
    self.curp.set(sp + 1);
    let e = SpineEntry::Initialize(y, Clock::default(), ith);
    self.log.borrow_mut().push(e);
    SpineEnv::step(&self.cur_env, sp, &self.curp, &self.log, None, None);
  }

  pub fn apply(&self, y: CellPtr, th: ThunkPtr) {
    //let sp = self.curp;
    //self.curp += 1;
    let sp = self.curp.get();
    self.curp.set(sp + 1);
    let e = SpineEntry::Apply(y, Clock::default(), th);
    self.log.borrow_mut().push(e);
    SpineEnv::step(&self.cur_env, sp, &self.curp, &self.log, None, None);
  }

  pub fn accumulate(&self, y: CellPtr, th: ThunkPtr) {
    //let sp = self.curp;
    //self.curp += 1;
    let sp = self.curp.get();
    self.curp.set(sp + 1);
    let e = SpineEntry::Accumulate(y, Clock::default(), th);
    self.log.borrow_mut().push(e);
    SpineEnv::step(&self.cur_env, sp, &self.curp, &self.log, None, None);
  }

  pub fn seal(&self, x: CellPtr) {
    //let sp = self.curp;
    //self.curp += 1;
    let sp = self.curp.get();
    self.curp.set(sp + 1);
    let e = SpineEntry::Seal(x);
    self.log.borrow_mut().push(e);
    SpineEnv::step(&self.cur_env, sp, &self.curp, &self.log, None, None);
  }

  pub fn unseal_mux(&self, x: CellPtr) {
    //let sp = self.curp;
    //self.curp += 1;
    let sp = self.curp.get();
    self.curp.set(sp + 1);
    let e = SpineEntry::Unseal(x);
    self.log.borrow_mut().push(e);
    SpineEnv::step(&self.cur_env, sp, &self.curp, &self.log, None, None);
  }

  pub fn unsync(&self, x: CellPtr) {
    unimplemented!();
  }

  pub fn _counter(&self) -> Clock {
    self.cur_env.borrow().ctr.into()
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

  /*pub fn _get(&self, mm: MCellPtr, k: CellPtr, kclk: Clock) -> Option<(CellPtr, Clock)> {
    let cur_env = self.cur_env.borrow();
    match cur_env.map.get(&(mm, k, kclk.into())) {
      None => None,
      //Some(&(okclk, v, vclk)) => {}
      /*Some(&(v, vclk)) => {
        /*if kclk != okclk {
          println!("DEBUG: Spine::_get: mm={:?} k={:?} kclk={:?} okclk={:?} v={:?} vclk={:?}",
              mm, k, kclk, okclk, v, vclk);
          panic!("bug");
        }*/
        Some((v, vclk.into()))
      }*/
      Some(&SpineEnvMapEntry::Value(v, vclk)) => {
        Some((v, vclk.into()))
      }
      Some(&SpineEnvMapEntry::Alias(v)) => {
        let kroot = cur_env._deref(k);
        assert!(k != kroot);
        let vroot = cur_env._deref(v);
        assert!(v != vroot);
        match cur_env.map.get(&(mm, kroot, kclk.into())) {
          None => panic!("bug"),
          Some(&SpineEnvMapEntry::Value(ovroot, vclk)) => {
            assert_eq!(ovroot, vroot);
            Some((v, vclk.into()))
          }
          _ => panic!("bug"),
        }
      }
    }
  }*/

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

  pub fn _compile(&mut self, ) {
    // FIXME FIXME
    //unimplemented!();
    /*
    let mut dry = DrySpine::new(self);
    dry._interp();
    dry._reorder();
    dry._fuse();
    dry._reduce();
    */
  }

  /*pub fn _start(&mut self) {
    self.hltp = self.curp;
  }*/

  pub fn _resume(&mut self, ctr: &CtxCtr, env: &mut CtxEnv, thunkenv: &mut CtxThunkEnv, /*target: CellPtr, tg_clk: Clock,*/ mut item: SpineResume) -> SpineRet {
    //self._start();
    let retp = if self.retp == u32::max_value() { None } else { Some(self.retp) };
    println!("DEBUG: Spine::_resume: ctr={:?} ctlp={} hltp={} curp={} retp={:?} item={:?}",
        self.ctr, self.ctlp, self.hltp, self.curp.get(), retp, item.key());
    self.hltp = self.curp.get();
    loop {
      let state = self._step(ctr, env, thunkenv, item.take());
      match state {
        SpineRet::Yield_(_) |
        SpineRet::Bot => {
          self.retp = self.ctlp;
          return state;
        }
        _ => {}
      }
      self.ctlp += 1;
      self.retp = u32::max_value();
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

  pub fn _step(&self, ctr: &CtxCtr, env: &mut CtxEnv, thunkenv: &mut CtxThunkEnv, item: SpineResume) -> SpineRet {
    let retp = if self.retp == u32::max_value() { None } else { Some(self.retp) };
    if self.ctlp >= self.hltp {
      println!("DEBUG: Spine::_step: ctr={:?} ctlp={} hltp={} curp={} retp={:?} halt",
          self.ctr, self.ctlp, self.hltp, self.curp.get(), retp);
      return SpineRet::Halt;
    }
    let mut ret = SpineRet::_Top;
    let entry = self.log.borrow()[self.ctlp as usize];
    println!("DEBUG: Spine::_step: ctr={:?} ctlp={} hltp={} curp={} retp={:?} entry={:?}",
        self.ctr, self.ctlp, self.hltp, self.curp.get(), retp, entry.name());
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
      SpineEntry::AdjMap(..) => {
        // FIXME FIXME
        //unimplemented!();
      }
      SpineEntry::Add(mx, x, _xclk) => {
        // FIXME FIXME
        unimplemented!();
      }
      SpineEntry::Add2(mx, k, _kclk, v, _vclk) => {
        // FIXME FIXME
        //unimplemented!();
      }
      SpineEntry::Cache(x, _xclk) => {
        match env.lookup_ref(x) {
          None => panic!("bug"),
          Some(e) => {
            let base_clk: Clock = self.ctr.into();
            let prev_clk = e.state().clk;
            let next_clk = base_clk.init_once();
            if prev_clk >= next_clk {
              panic!("bug");
            } else if prev_clk < next_clk {
              //e.state().mode = CellMode::Aff;
              e.state().flag.set_intro();
              e.state().flag.unset_seal();
              // FIXME FIXME
              //e.state().clk = next_clk;
              e.clock_sync(prev_clk, next_clk, env);
            }
          }
        }
      }
      SpineEntry::ICacheMux(x) => {
        match env.lookup_ref(x) {
          None => panic!("bug"),
          Some(e) => {
            let flag = !e.state().flag.set_cache();
            let mode = match e.state().mode.set_init() {
              Err(_) => panic!("bug"),
              Ok(prev) => !prev
            };
            assert!(flag);
            assert!(mode);
            //e.cel.clk.
          }
        }
      }
      SpineEntry::OIntro(_x, _og) => {
        // FIXME FIXME
        //unimplemented!();
      }
      SpineEntry::Intro(x, _xclk) => {
        match env.lookup_ref(x) {
          None => panic!("bug"),
          Some(e) => {
            /*match e.state().mode {
              CellMode::_Top => {}
              CellMode::Aff => {
                // FIXME FIXME: should warn here?
              }
              _ => panic!("bug")
            }*/
            e.state().mode = CellMode::Aff;
            //if !e.state().clk.happens_before(self.ctr).unwrap() {}
            match e.state().clk.partial_cmp_(self.ctr) {
              Some(Ordering::Less) => {}
              _ => panic!("bug")
            }
            assert!(!e.state().flag.intro());
            assert!(!e.state().flag.seal());
            // FIXME
            e.state().clk = self.ctr.into();
            /*e.state().flag.reset();*/
            e.state().flag.set_intro();
          }
        }
      }
      SpineEntry::Uninit(x, _xclk) => {
        // FIXME FIXME
        unimplemented!();
      }
      SpineEntry::YieldSet(x, _xclk, loc) => {
        let ctlp = self.ctlp;
        let retp = if self.retp == u32::max_value() { None } else { Some(self.retp) };
        println!("DEBUG: Spine::_step: YieldSet: ctlp={:?} retp={:?} x={:?} loc={:?} key={:?}",
            ctlp, retp, x, loc, item.key());
        match env.lookup_ref(x) {
          None => panic!("bug"),
          Some(e) => {
            println!("DEBUG: Spine::_step: YieldSet:   expected dtype {:?}", e.ty.dtype);
            match e.ty.dtype {
              Dtype::Fp32 => {
                //println!("DEBUG: Spine::_step: YieldSet:   expected dtype {:?}", e.ty.dtype);
                match &item {
                  SpineResume::_Top => {
                    println!("DEBUG: Spine::_step: YieldSet:   no value");
                  }
                  SpineResume::PutMemV(k, v) => {
                    match v.downcast_ref::<f32>() {
                      None => {
                        println!("DEBUG: Spine::_step: YieldSet:   wrong type");
                      }
                      Some(v) => {
                        println!("DEBUG: Spine::_step: YieldSet:   key={:?} value={:?}", k, v);
                      }
                    }
                  }
                  SpineResume::PutMemF(k, _f) => {
                    println!("DEBUG: Spine::_step: YieldSet:   key={:?} fun", k);
                    // TODO
                  }
                }
              }
              _ => {
              }
            }
          }
        }
        if Some(ctlp) == retp {
          println!("DEBUG: Spine::_step: YieldSet:   ...resume");
          let xclk = match env.lookup_ref(x) {
            None => panic!("bug"),
            Some(e) => {
              let base_clk: Clock = self.ctr.into();
              let prev_clk = e.state().clk;
              let next_clk = base_clk.init_once();
              println!("DEBUG: Spine::_step: YieldSet:   prev clk={:?}", prev_clk);
              println!("DEBUG: Spine::_step: YieldSet:   base clk={:?}", base_clk);
              println!("DEBUG: Spine::_step: YieldSet:   next clk={:?}", next_clk);
              if prev_clk >= next_clk {
                panic!("bug");
              } else if prev_clk < next_clk {
                /*//assert!(!e.state().flag.intro());
                //assert!(!e.state().flag.seal());
                // FIXME: concise condition.
                assert_eq!(e.state().clk.ctr(), self.ctr);
                assert!(prev_clk.ctr().is_nil() || prev_clk.ctr() == self.ctr);
                assert_eq!(prev_clk.up, 0);*/
                //e.state().mode = CellMode::Aff;
                e.state().flag.set_intro();
                e.state().flag.unset_seal();
                e.clock_sync_loc(Locus::Mem, prev_clk, next_clk, env);
              }
              next_clk
            }
          };
          match env.pwrite_ref(x, xclk) {
            None => panic!("bug"),
            Some(e) => {
              match e.cel_ {
                &mut Cell_::Phy(_, ref clo, ref mut cel_) => {
                  match (loc, &item) {
                    (Locus::Mem, SpineResume::PutMemV(_, _val)) => {
                      // FIXME FIXME
                      unimplemented!();
                    }
                    (Locus::Mem, SpineResume::PutMemF(_, fun)) => {
                      let (pm, addr) = cel_.get_loc_nosync(x, xclk, &e.ty, Locus::Mem);
                      TL_PCTX.with(|pctx| {
                        let (_, icel) = pctx.lookup_pm(pm, addr).unwrap();
                        (fun)(e.ty.clone(), icel.as_mem_reg().unwrap());
                      });
                    }
                    _ => {
                      unimplemented!();
                    }
                  }
                  clo.borrow_mut().init_once(xclk, ThunkPtr::opaque());
                }
                _ => panic!("bug")
              }
            }
          }
        } else {
          println!("DEBUG: Spine::_step: YieldSet:   yield...");
          ret = SpineRet::Yield_(());
        }
      }
      SpineEntry::YieldInit(x, _xclk, loc) => {
        println!("DEBUG: Spine::_step: YieldInit: x={:?} loc={:?} key={:?}", x, loc, item.key());
        unimplemented!();
      }
      SpineEntry::Alias(x, og) => {
        // FIXME FIXME
        //unimplemented!();
      }
      SpineEntry::CAlias(x, og) => {
        // FIXME FIXME
        //unimplemented!();
      }
      SpineEntry::Snapshot(x, og) => {
        // FIXME FIXME
        //unimplemented!();
      }
      SpineEntry::PushSeal(x, _xclk) => {
        match env.lookup_ref(x) {
          None => panic!("bug"),
          Some(e) => {
            if !e.state().flag.intro() {
              let xroot = e.root;
              let xclk = e.state().clk;
              println!("DEBUG: Spine::_step: PushSeal: xroot={:?} x={:?} xclk={:?}", xroot, x, xclk);
            }
            assert!(e.state().flag.intro());
            let xclk = e.state().clk;
            /*if xclk.ctr().is_nil() {
              println!("DEBUG: Spine::_step: PushSeal: x={:?} xclk={:?}", x, xclk);
            }*/
            assert!(!xclk.ctr().is_nil());
            e.state().flag.set_seal();
          }
        }
      }
      SpineEntry::Initialize(x, _xclk, th) => {
        // FIXME FIXME
        let (xroot, xclk) = match env.lookup_ref(x) {
          None => panic!("bug"),
          Some(e) => {
            let root = e.root;
            /*match e.state().mode {
              CellMode::Init => {}
              _ => panic!("bug")
            }*/
            assert_eq!(e.state().clk.ctr(), self.ctr);
            //assert!(e.state().flag.intro());
            assert!(!e.state().flag.seal());
            e.state().flag.set_intro();
            // FIXME FIXME
            let base_clk: Clock = self.ctr.into();
            let prev_clk = e.state().clk;
            let next_clk = base_clk.max(prev_clk).init_once();
            assert!(prev_clk < next_clk);
            //e.state().clk = next_clk;
            e.clock_sync(prev_clk, next_clk, env);
            (root, next_clk)
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
          let ret = te.pthunk.initialize(ctr, env, &tclo.arg, th, x, xclk);
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
          match env.lookup_ref(x) {
            None => panic!("bug"),
            Some(e) => {
              match e.cel_ {
                &Cell_::Phy(_, ref clo, _) |
                &Cell_::Cow(_, ref clo, _) => {
                  clo.borrow_mut().init_once(xclk, th);
                }
                _ => panic!("bug")
              }
            }
          }
        }
      }
      SpineEntry::Apply(x, _xclk, th) => {
        let (xroot, xclk) = match env.lookup_ref(x) {
          None => panic!("bug"),
          Some(e) => {
            let root = e.root;
            /*match e.state().mode {
              CellMode::Aff => {}
              _ => panic!("bug")
            }*/
            //assert_eq!(e.state().clk.ctr(), self.ctr);
            /*assert_eq!(e.state().clk.up, 0);*/
            //assert!(e.state().clk.is_uninit());
            //assert!(e.state().flag.intro());
            assert!(!e.state().flag.seal());
            e.state().flag.set_intro();
            // FIXME
            let base_clk: Clock = self.ctr.into();
            let prev_clk = e.state().clk;
            let next_clk = base_clk.max(prev_clk).init_once();
            assert!(prev_clk < next_clk);
            //e.state().clk = next_clk;
            e.clock_sync(prev_clk, next_clk, env);
            (root, next_clk)
          }
        };
        {
          let tclo = match thunkenv.update.get(&(xroot, xclk)) {
            None => panic!("bug"),
            Some(tclo) => tclo
          };
          println!("DEBUG: Spine::_step: Apply: x={:?} xclk={:?} th={:?} tclo={:?}",
              x, xclk, th, tclo);
          assert_eq!(tclo.pthunk, th);
          let te = match thunkenv.thunktab.get(&th) {
            None => panic!("bug"),
            Some(te) => te
          };
          let ret = te.pthunk.apply(ctr, env, &tclo.arg, th, x, xclk);
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
          match env.lookup_ref(x) {
            None => panic!("bug"),
            Some(e) => {
              match e.cel_ {
                &Cell_::Phy(_, ref clo, _) |
                &Cell_::Cow(_, ref clo, _) => {
                  clo.borrow_mut().init_once(xclk, th);
                }
                _ => panic!("bug: Spine::_step: Apply: cel={:?}", e.cel_.name())
              }
            }
          }
        }
      }
      SpineEntry::Accumulate(x, _xclk, th) => {
        let (xroot, xclk) = match env.lookup_ref(x) {
          None => panic!("bug"),
          Some(e) => {
            let root = e.root;
            /*match e.state().mode {
              CellMode::Init => {}
              _ => panic!("bug")
            }*/
            //assert_eq!(e.state().clk.ctr(), self.ctr);
            //assert!(e.state().clk.is_uninit());
            //assert!(e.state().flag.intro());
            assert!(!e.state().flag.seal());
            e.state().flag.set_intro();
            // FIXME
            let base_clk: Clock = self.ctr.into();
            let prev_clk = e.state().clk;
            let next_clk = base_clk.max(prev_clk).update();
            assert!(prev_clk < next_clk);
            //e.state().clk = next_clk;
            e.clock_sync(prev_clk, next_clk, env);
            (root, next_clk)
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
          let ret = te.pthunk.accumulate(ctr, env, &tclo.arg, th, x, xclk);
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
          match env.lookup_ref(x) {
            None => panic!("bug"),
            Some(e) => {
              match e.cel_ {
                &Cell_::Phy(_, ref clo, _) |
                &Cell_::Cow(_, ref clo, _) => {
                  clo.borrow_mut().update(xclk, th);
                }
                _ => panic!("bug")
              }
            }
          }
        }
      }
      SpineEntry::UnsafeWrite(x, _xclk, th) => {
        unimplemented!();
      }
      SpineEntry::Seal(x) => {
        match env.lookup_ref(x) {
          None => panic!("bug"),
          Some(e) => {
            match e.state().mode {
              CellMode::_Top => panic!("bug"),
              _ => {}
            }
            assert_eq!(e.state().clk.ctr(), self.ctr);
            // FIXME: clock validity conditions.
            /*assert!(e.state().clk.up > 0);*/
            // FIXME: is this a vestige of final?
            /*assert!(e.state().clk.up != i32::max_value());*/
            assert!(!e.state().clk.is_uninit());
            assert!(e.state().flag.intro());
            assert!(!e.state().flag.seal());
            /*e.state().flag.reset();*/
            e.state().flag.set_seal();
          }
        }
      }
      SpineEntry::Unseal(x) => {
        unimplemented!();
      }
      SpineEntry::Unsync(x) => {
        match env.lookup_ref(x) {
          None => panic!("bug"),
          Some(e) => {
            assert_eq!(e.state().clk.ctr(), self.ctr);
            assert!(e.state().flag.intro());
            assert!(e.state().flag.seal());
            // FIXME FIXME
            unimplemented!()
          }
        }
      }
      SpineEntry::Bot => {
        ret = SpineRet::Bot;
      }
      //_ => unimplemented!()
      e => panic!("bug: Spine::_step: unimplemented: {:?}", e)
    }
    let t1 = Stopwatch::tl_stamp();
    let d = t1 - t0;
    //println!("DEBUG: Spine::_step:   t1={}.{:09} s", t1.s(), t1.sub_ns());
    println!("DEBUG: Spine::_step:   d={:.09} s", d);
    ret
  }

  pub fn _debug_dump(&self) {
    println!("DEBUG: Spine::_debug_dump: ctr={:?} ctlp={} hltp={} curp={}",
        self.ctr, self.ctlp, self.hltp, self.curp.get());
    for (i, e) in self.log.borrow().iter().enumerate() {
      println!("DEBUG: Spine::_debug_dump: log[{}]={:?}", i, e);
    }
  }
}

/*#[derive(Default)]
pub struct DryEnv {
  intro:    HashMap<CellPtr, u32>,
  fin:      HashSet<CellPtr>,
  aff:      HashSet<CellPtr>,
  semi:     HashSet<CellPtr>,
  seal:     HashMap<CellPtr, u32>,
  apply:    HashMap<CellPtr, Vec<(u32, ThunkPtr)>>,
}

impl DryEnv {
  pub fn reset(&mut self) {
    // FIXME
  }
}

pub struct DrySpine {
  // TODO
  ctr:  Counter,
  ctlp: u32,
  hltp: u32,
  curp: u32,
  env:  [DryEnv; 2],
  log:  [Vec<SpineEntry>; 2],
}

impl DrySpine {
  pub fn new(sp: &Spine) -> DrySpine {
    assert!(!sp.ctr.is_nil());
    if sp.ctlp != 0 {
      panic!("bug");
    }
    DrySpine{
      ctr:  sp.ctr,
      ctlp: 0,
      hltp: sp.hltp,
      curp: sp.curp,
      env:  [DryEnv::default(), DryEnv::default()],
      log:  [sp.log.clone(), Vec::new()],
    }
  }

  pub fn _interp(&mut self) {
    self.ctlp = 0;
    // FIXME FIXME
    //self.env[0].reset();
    /*self.log[0].clear();*/
    while self.ctlp < self.hltp {
      // FIXME
      match &self.log[0][self.ctlp as usize] {
        &SpineEntry::Cache(x) => {
          unimplemented!();
        }
        &SpineEntry::ICacheMux(x) => {
          unimplemented!();
        }
        /*&SpineEntry::IntroFin(x) => {
          self.env[0].intro.insert(x, self.ctlp);
          self.env[0].fin.insert(x);
        }*/
        &SpineEntry::Intro(x) => {
          self.env[0].intro.insert(x, self.ctlp);
          self.env[0].aff.insert(x);
        }
        &SpineEntry::Initialize(x, _ith) => {
          // FIXME FIXME
          self.env[0].intro.insert(x, self.ctlp);
          self.env[0].semi.insert(x);
        }
        &SpineEntry::Apply(x, th) => {
          assert!(self.env[0].intro.contains_key(&x));
          assert!(self.env[0].aff.contains(&x));
          match self.env[0].apply.get_mut(&x) {
            None => {
              let mut ap = Vec::new();
              ap.push((self.ctlp, th));
              self.env[0].apply.insert(x, ap);
            }
            Some(_) => panic!("bug")
          }
        }
        &SpineEntry::Accumulate(x, th) => {
          assert!(self.env[0].intro.contains_key(&x));
          assert!(self.env[0].semi.contains(&x));
          match self.env[0].apply.get_mut(&x) {
            None => {
              let mut ap = Vec::new();
              ap.push((self.ctlp, th));
              self.env[0].apply.insert(x, ap);
            }
            Some(ap) => {
              ap.push((self.ctlp, th));
            }
          }
        }
        &SpineEntry::Seal(x) => {
          assert!(self.env[0].semi.contains(&x));
          self.env[0].seal.insert(x, self.ctlp);
        }
        _ => unimplemented!()
      }
      self.ctlp += 1;
    }
  }

  pub fn _reorder(&mut self) {
    self.ctlp = 0;
    self.env.swap(0, 1);
    self.log.swap(0, 1);
    self.env[0].reset();
    self.log[0].clear();
    unimplemented!();
  }

  pub fn _fuse(&mut self) {
  }

  pub fn _reduce(&mut self) {
  }
}
*/
