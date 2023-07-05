use crate::algo::{HashMap, HashSet};
use crate::cell::*;
use crate::clock::{Counter, Clock};
use crate::ctx::*;
use crate::panick::{panick_wrap};
use crate::pctx::{TL_PCTX, Locus, MemReg};
use crate::thunk::*;
use crate::thunk::op::{SetScalarFutThunkSpec};
use crate::util::time::{Stopwatch};

use std::any::{Any};
use std::cmp::{Ordering};
//use std::mem::{swap};

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

#[derive(Clone, Debug)]
//#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SpineEntry {
  _Top,
  Yield_,
  YieldV(CellPtr, CellPtr),
  Break_,
  BreakV(CellPtr, CellPtr),
  TraceV(CellPtr, CellPtr),
  Profile(CellPtr, CellPtr),
  Cache(CellPtr),
  ICacheMux(CellPtr),
  OIntro(CellPtr, CellPtr),
  Intro(CellPtr),
  Uninit(CellPtr),
  //IntroFin(CellPtr),
  YieldSet(CellPtr, Locus),
  YieldInit(CellPtr, Locus),
  Alias(CellPtr, CellPtr),
  Snapshot(CellPtr, CellPtr),
  //Snapshot(CellPtr, CellPtr, CellPtr, /*Clock*/),
  //Keep(CellPtr),
  PushSeal(CellPtr),
  Initialize(CellPtr, ThunkPtr),
  Apply(CellPtr, ThunkPtr),
  Accumulate(CellPtr, ThunkPtr),
  UnsafeWrite(CellPtr, ThunkPtr),
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
      &SpineEntry::Cache(..)      => SpineEntryName::Cache,
      &SpineEntry::ICacheMux(..)  => SpineEntryName::ICacheMux,
      &SpineEntry::OIntro(..)     => SpineEntryName::OIntro,
      &SpineEntry::Intro(..)      => SpineEntryName::Intro,
      &SpineEntry::Uninit(..)     => SpineEntryName::Uninit,
      &SpineEntry::YieldSet(..)   => SpineEntryName::YieldSet,
      &SpineEntry::YieldInit(..)  => SpineEntryName::YieldInit,
      &SpineEntry::Alias(..)      => SpineEntryName::Alias,
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
  Cache,
  ICacheMux,
  OIntro,
  Intro,
  Uninit,
  YieldSet,
  YieldInit,
  Alias,
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
pub type SpineCellSet = MCellSet;
pub type SpineCellMap = MCellMap;

/*pub struct SpineCellState {
  // FIXME FIXME
  //pub alias_ct: u32,
  pub hasalias: bool,
  pub is_alias: bool,
  pub mode:     CellMode,
  pub flag:     CellFlag,
  pub clk:      Clock,
}*/

#[derive(Clone, Default, Debug)]
pub struct SpineEnv {
  // FIXME FIXME
  /*pub aff:      HashMap<CellPtr, u32>,
  pub init:     HashMap<CellPtr, (u32, ThunkPtr)>,
  pub cache:    HashMap<CellPtr, u32>,
  pub intro:    HashMap<CellPtr, u32>,
  //pub fin:      HashSet<CellPtr>,
  pub seal:     HashMap<CellPtr, u32>,
  pub apply:    HashMap<CellPtr, Vec<(u32, ThunkPtr)>>,
  //pub eval:     HashMap<CellPtr, u32>,*/
  // TODO TODO
  pub ctr:      Counter,
  pub state:    HashMap<CellPtr, SpineCellState>,
  pub set:      HashMap<CellPtr, SpineCellSet>,
  pub map:      HashMap<CellPtr, SpineCellMap>,
  //pub aliases:  HashMap<(CellPtr, Clock), HashSet<CellPtr>>,
  //pub alias_of: HashMap<CellPtr, (CellPtr, Clock)>,
  pub arg:      Vec<(CellPtr, Clock)>,
  pub iapply:   HashMap<CellPtr, (u32, ThunkPtr, Clock)>,
  pub apply_:   HashMap<CellPtr, Vec<(u32, ThunkPtr, Clock)>>,
  pub bwd:      HashMap<CellPtr, Clock>,
  //pub bwd:      HashSet<(CellPtr, Clock)>,
  pub gradr:    HashMap<(CellPtr, CellPtr, Clock), CellPtr>,
}

impl SpineEnv {
  pub fn reset(&mut self, ctr: Counter) {
    println!("DEBUG: SpineEnv::reset: ctr={:?}", ctr);
    /*self.aff.clear();
    self.init.clear();
    self.cache.clear();
    self.intro.clear();
    self.seal.clear();
    self.apply.clear();
    //self.eval.clear();*/
    self.ctr = ctr;
    self.state.clear();
    self.set.clear();
    self.map.clear();
    self.arg.clear();
    self.iapply.clear();
    self.apply_.clear();
    self.bwd.clear();
    self.gradr.clear();
  }

  pub fn step(&mut self, sp: u32, e: &SpineEntry) {
    // FIXME FIXME
    println!("DEBUG: SpineEnv::step: idx={} e={:?}", sp, e);
    match e {
      &SpineEntry::_Top => {}
      &SpineEntry::OIntro(x, _) => {
        /*self.cache.insert(x, sp);*/
        // FIXME FIXME
        unimplemented!();
      }
      &SpineEntry::Cache(x) => {
        /*assert!(!self.intro.contains_key(&x));
        assert!(!self.seal.contains_key(&x));
        self.intro.insert(x, sp);
        self.seal.insert(x, sp);
        self.aff.insert(x, sp);*/
        // FIXME FIXME
        //unimplemented!();
        match self.state.get(&x) {
          None => {
            self.state.insert(x, CellState::default());
          }
          _ => {}
        }
        match self.state.get_mut(&x) {
          None => panic!("bug"),
          Some(state) => {
            assert_eq!(state.mode, CellMode::_Top);
            assert!(!state.flag.intro());
            assert!(!state.flag.seal());
            assert_eq!(state.clk.up, 0);
            state.mode = CellMode::Aff;
            state.flag.set_intro();
            let clk0: Clock = self.ctr.into();
            let next_clk = clk0.update();
            match self.apply_.get(&x) {
              None => {
                self.apply_.insert(x, Vec::new());
              }
              _ => {}
            }
            match self.apply_.get_mut(&x) {
              None => panic!("bug"),
              Some(thlist) => {
                thlist.push((sp, ThunkPtr::opaque(), next_clk));
                assert_eq!(thlist.len(), next_clk.up as _);
              }
            }
            state.clk = next_clk;
          }
        }
      }
      &SpineEntry::ICacheMux(x) => {
        /*assert!(!self.seal.contains_key(&x));
        self.intro.insert(x, sp);
        self.init.insert(x, (sp, ThunkPtr::nil()));*/
        // FIXME FIXME
        unimplemented!();
        /*match self.state.get_mut(&x) {
          None => panic!("bug"),
          Some(state) => {
            assert_eq!(state.mode, CellMode::Init);
            assert!(!state.flag.intro());
            assert!(!state.flag.seal());
            assert_eq!(state.clk.rst, self.ctr.previous());
            state.mode.set_init();
            state.flag.set_intro();
            let next_clk = self.ctr.into();
            state.clk = next_clk;
          }
        }*/
      }
      &SpineEntry::Intro(x) => {
        /*assert!(!self.intro.contains_key(&x));
        assert!(!self.seal.contains_key(&x));
        self.intro.insert(x, sp);
        self.aff.insert(x, sp);*/
        match self.state.get(&x) {
          None => {
            self.state.insert(x, CellState::default());
          }
          _ => {}
        }
        match self.state.get_mut(&x) {
          None => panic!("bug"),
          Some(state) => {
            assert_eq!(state.mode, CellMode::_Top);
            assert!(!state.flag.intro());
            assert!(!state.flag.seal());
            assert_eq!(state.clk.up, 0);
            state.mode = CellMode::Aff;
            state.flag.set_intro();
            state.clk = self.ctr.into();
          }
        }
      }
      &SpineEntry::Uninit(x) => {
        match self.state.get_mut(&x) {
          None => panic!("bug"),
          Some(state) => {
            assert_eq!(state.mode, CellMode::_Top);
            assert!(!state.flag.intro());
            assert!(!state.flag.seal());
            /*assert_eq!(state.clk.up, 0);*/
            state.mode = CellMode::Init;
            state.flag.set_intro();
            let next_clk = self.ctr.into();
            state.clk = next_clk;
          }
        }
      }
      &SpineEntry::YieldSet(x, _loc) => {
        match self.state.get(&x) {
          None => {
            self.state.insert(x, CellState::default());
          }
          _ => {}
        }
        match self.state.get_mut(&x) {
          None => panic!("bug"),
          Some(state) => {
            assert_eq!(state.mode, CellMode::_Top);
            assert!(!state.flag.intro());
            assert!(!state.flag.seal());
            assert_eq!(state.clk.up, 0);
            state.mode = CellMode::Aff;
            state.flag.set_intro();
            let clk0: Clock = self.ctr.into();
            let next_clk = clk0.update();
            match self.apply_.get(&x) {
              None => {
                self.apply_.insert(x, Vec::new());
              }
              _ => {}
            }
            match self.apply_.get_mut(&x) {
              None => panic!("bug"),
              Some(thlist) => {
                thlist.push((sp, ThunkPtr::opaque(), next_clk));
                assert_eq!(thlist.len(), next_clk.up as _);
              }
            }
            state.clk = next_clk;
          }
        }
      }
      &SpineEntry::YieldInit(x, _loc) => {
        match self.state.get(&x) {
          None => {
            self.state.insert(x, CellState::default());
          }
          _ => {}
        }
        match self.state.get_mut(&x) {
          None => panic!("bug"),
          Some(state) => {
            assert_eq!(state.mode, CellMode::_Top);
            assert!(!state.flag.intro());
            assert!(!state.flag.seal());
            /*assert_eq!(state.clk.up, 0);*/
            state.mode = CellMode::Init;
            state.flag.set_intro();
            let next_clk = self.ctr.into();
            assert!(self.iapply.insert(x, (sp, ThunkPtr::opaque(), next_clk)).is_none());
            state.clk = next_clk;
          }
        }
      }
      &SpineEntry::Alias(x, og) => {
        // FIXME FIXME: prefer unification?
        /*match self.state.get(&x) {
          None => {
            self.state.insert(x, CellState::default());
          }
          _ => {}
        }*/
        match self.state.get(&og) {
          None => panic!("bug"),
          Some(state) => {
            self.state.insert(x, state.clone());
          }
        }
      }
      &SpineEntry::Snapshot(x, og) => {
        // FIXME FIXME
        /*match self.state.get(&x) {
          None => {
            self.state.insert(x, CellState::default());
          }
          _ => {}
        }*/
        match self.state.get(&og) {
          None => panic!("bug"),
          Some(state) => {
            self.state.insert(x, state.clone());
          }
        }
      }
      &SpineEntry::PushSeal(x) => {
        match self.state.get_mut(&x) {
          None => {
            println!("DEBUG: PushSeal: x={:?} spine env={:?}", x, self);
            panic!("bug");
          }
          Some(state) => {
            assert!(state.flag.intro());
            // NB: late/lazy idempotent seal.
            /*assert!(!state.flag.seal());*/
            self.arg.push((x, state.clk));
            state.flag.set_seal();
          }
        }
      }
      &SpineEntry::Initialize(x, ith) => {
        match self.state.get_mut(&x) {
          None => panic!("bug"),
          Some(state) => {
            assert_eq!(state.mode, CellMode::_Top);
            assert!(!state.flag.intro());
            assert!(!state.flag.seal());
            /*assert_eq!(state.clk.up, 0);*/
            state.mode = CellMode::Init;
            state.flag.set_intro();
            let next_clk = self.ctr.into();
            assert!(self.iapply.insert(x, (sp, ith, next_clk)).is_none());
            state.clk = next_clk;
            self.arg.clear();
          }
        }
      }
      &SpineEntry::Apply(x, th) => {
        match self.state.get_mut(&x) {
          None => panic!("bug"),
          Some(state) => {
            assert_eq!(state.mode, CellMode::Aff);
            assert!(state.flag.intro());
            assert!(!state.flag.seal());
            assert_eq!(state.clk.up, 0);
            let next_clk = state.clk.update();
            match self.apply_.get(&x) {
              None => {
                self.apply_.insert(x, Vec::new());
              }
              _ => {}
            }
            match self.apply_.get_mut(&x) {
              None => panic!("bug"),
              Some(thlist) => {
                thlist.push((sp, th, next_clk));
                assert_eq!(thlist.len(), next_clk.up as _);
              }
            }
            state.clk = next_clk;
            self.arg.clear();
          }
        }
      }
      &SpineEntry::Accumulate(x, th) => {
        match self.state.get_mut(&x) {
          None => panic!("bug"),
          Some(state) => {
            assert_eq!(state.mode, CellMode::Init);
            assert!(state.flag.intro());
            assert!(!state.flag.seal());
            let next_clk = state.clk.update();
            match self.apply_.get(&x) {
              None => {
                self.apply_.insert(x, Vec::new());
              }
              _ => {}
            }
            match self.apply_.get_mut(&x) {
              None => panic!("bug"),
              Some(thlist) => {
                thlist.push((sp, th, next_clk));
                assert_eq!(thlist.len(), next_clk.up as _);
              }
            }
            state.clk = next_clk;
            self.arg.clear();
          }
        }
      }
      &SpineEntry::UnsafeWrite(x, th) => {
        unimplemented!();
      }
      &SpineEntry::Seal(x) => {
        match self.state.get_mut(&x) {
          None => panic!("bug"),
          Some(state) => {
            /*assert!(state.mode != CellMode::_Top);*/
            assert!(state.flag.intro());
            // NB: idempotent seal.
            /*assert!(!state.flag.seal());*/
            state.flag.set_seal();
          }
        }
      }
      &SpineEntry::Unseal(x) => {
        match self.state.get_mut(&x) {
          None => panic!("bug"),
          Some(state) => {
            /*assert!(state.mode != CellMode::_Top);*/
            // NB: idempotent unseal.
            /*assert!(state.flag.seal());*/
            state.flag.unset_seal();
          }
        }
      }
      &SpineEntry::Unsync(x) => {
        // FIXME FIXME
        unimplemented!();
      }
      // TODO TODO
      _ => unimplemented!()
    }
  }

  pub fn unstep(&mut self, /*sp: u32,*/ e: &SpineEntry) {
    // FIXME FIXME
  }
}

pub struct Spine {
  pub ctr:  Counter,
  pub ctlp: u32,
  pub hltp: u32,
  pub curp: u32,
  pub retp: u32,
  pub cur_env:  SpineEnv,
  pub log:  Vec<SpineEntry>,
}

impl Default for Spine {
  fn default() -> Spine {
    Spine{
      ctr:  Counter::default(),
      ctlp: 0,
      hltp: 0,
      curp: 0,
      retp: u32::max_value(),
      cur_env:  SpineEnv::default(),
      log:  Vec::new(),
    }
  }
}

impl Spine {
  pub fn _reset(&mut self) {
    println!("DEBUG: Spine::_reset: ctr={:?} ctlp={} hltp={} curp={}",
        self.ctr, self.ctlp, self.hltp, self.curp);
    self.ctr = self.ctr.advance();
    self.ctlp = 0;
    self.hltp = 0;
    self.curp = 0;
    self.retp = u32::max_value();
    self.cur_env.reset(self.ctr);
    self.log.clear();
  }

  pub fn opaque(&mut self, x: CellPtr, og: CellPtr) {
    let sp = self.curp;
    self.curp += 1;
    let e = SpineEntry::OIntro(x, og);
    self.cur_env.step(sp, &e);
    self.log.push(e);
  }

  pub fn cache_aff(&mut self, x: CellPtr) {
    let sp = self.curp;
    self.curp += 1;
    let e = SpineEntry::Cache(x);
    self.cur_env.step(sp, &e);
    self.log.push(e);
  }

  pub fn intro_aff(&mut self, x: CellPtr) {
    let sp = self.curp;
    self.curp += 1;
    let e = SpineEntry::Intro(x);
    self.cur_env.step(sp, &e);
    self.log.push(e);
  }

  pub fn init_cache_mux(&mut self, x: CellPtr) {
    let sp = self.curp;
    self.curp += 1;
    let e = SpineEntry::ICacheMux(x);
    self.cur_env.step(sp, &e);
    self.log.push(e);
  }

  pub fn yield_set(&mut self, x: CellPtr, loc: Locus) {
    let sp = self.curp;
    self.curp += 1;
    let e = SpineEntry::YieldSet(x, loc);
    self.cur_env.step(sp, &e);
    self.log.push(e);
  }

  pub fn yield_init(&mut self, x: CellPtr, loc: Locus) {
    let sp = self.curp;
    self.curp += 1;
    let e = SpineEntry::YieldInit(x, loc);
    self.cur_env.step(sp, &e);
    self.log.push(e);
  }

  pub fn alias(&mut self, x: CellPtr, og: CellPtr) {
    let sp = self.curp;
    self.curp += 1;
    let e = SpineEntry::Alias(x, og);
    self.cur_env.step(sp, &e);
    self.log.push(e);
  }

  pub fn snapshot(&mut self, x: CellPtr, og: CellPtr) {
    let sp = self.curp;
    self.curp += 1;
    let e = SpineEntry::Snapshot(x, og);
    self.cur_env.step(sp, &e);
    self.log.push(e);
  }

  pub fn push_seal(&mut self, x: CellPtr) {
    let sp = self.curp;
    self.curp += 1;
    let e = SpineEntry::PushSeal(x);
    self.cur_env.step(sp, &e);
    self.log.push(e);
  }

  pub fn initialize(&mut self, y: CellPtr, ith: ThunkPtr) {
    let sp = self.curp;
    self.curp += 1;
    let e = SpineEntry::Initialize(y, ith);
    self.cur_env.step(sp, &e);
    self.log.push(e);
  }

  pub fn apply(&mut self, y: CellPtr, th: ThunkPtr) {
    let sp = self.curp;
    self.curp += 1;
    let e = SpineEntry::Apply(y, th);
    self.cur_env.step(sp, &e);
    self.log.push(e);
  }

  pub fn accumulate(&mut self, y: CellPtr, th: ThunkPtr) {
    let sp = self.curp;
    self.curp += 1;
    let e = SpineEntry::Accumulate(y, th);
    self.cur_env.step(sp, &e);
    self.log.push(e);
  }

  pub fn seal(&mut self, x: CellPtr) {
    let sp = self.curp;
    self.curp += 1;
    let e = SpineEntry::Seal(x);
    self.cur_env.step(sp, &e);
    self.log.push(e);
  }

  pub fn unseal_mux(&mut self, x: CellPtr) {
    let sp = self.curp;
    self.curp += 1;
    let e = SpineEntry::Unseal(x);
    self.cur_env.step(sp, &e);
    self.log.push(e);
  }

  pub fn unsync(&mut self, x: CellPtr) {
    unimplemented!();
  }

  pub fn _version(&self, x: CellPtr) -> Option<Clock> {
    match self.cur_env.state.get(&x) {
      None => None,
      Some(state) => Some(state.clk)
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
        self.ctr, self.ctlp, self.hltp, self.curp, retp, item.key());
    self.hltp = self.curp;
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
          self.ctr, self.ctlp, self.hltp, self.curp, retp);
      return SpineRet::Halt;
    }
    let t0 = Stopwatch::tl_stamp();
    let mut ret = SpineRet::_Top;
    let entry = &self.log[self.ctlp as usize];
    println!("DEBUG: Spine::_step: ctr={:?} ctlp={} hltp={} curp={} retp={:?} entry={:?}",
        self.ctr, self.ctlp, self.hltp, self.curp, retp, entry.name());
    match entry {
      // TODO
      &SpineEntry::_Top => {}
      &SpineEntry::Yield_ => {
        ret = SpineRet::Yield_(());
      }
      &SpineEntry::YieldV(_, _) => {
        unimplemented!();
      }
      &SpineEntry::Break_ => {
        ret = SpineRet::Break_;
      }
      &SpineEntry::BreakV(_, _) => {
        unimplemented!();
      }
      &SpineEntry::TraceV(_, _) => {
        unimplemented!();
      }
      &SpineEntry::Profile(_, _) => {
        unimplemented!();
      }
      &SpineEntry::OIntro(_x, _og) => {
        // FIXME FIXME
      }
      &SpineEntry::Cache(x) => {
        match env.lookup_ref(x) {
          None => panic!("bug"),
          Some(e) => {
            /*match e.cel.mode {
              CellMode::_Top => {
                e.cel.mode = CellMode::Aff;
              }
              CellMode::Aff => {}
              _ => panic!("bug")
            }*/
            /*// FIXME FIXME
            //unimplemented!();
            let flag = !e.state().flag.set_cache();
            let mode = match e.state().mode.set_aff() {
              Err(_) => panic!("bug"),
              Ok(prev) => !prev
            };
            assert!(flag);
            //assert!(mode);
            assert!(self.ctr.succeeds(e.state().clk));*/
            let prev_clk = e.state().clk;
            let base_clk: Clock = self.ctr.into();
            let next_clk = base_clk.update();
            if prev_clk >= next_clk {
              panic!("bug");
            } else if prev_clk < next_clk {
              e.state().mode = CellMode::Aff;
              e.state().flag.set_intro();
              e.state().flag.unset_seal();
              // FIXME FIXME
              //e.state().clk = next_clk;
              e.clock_sync(prev_clk, next_clk, env);
            /*} else {
              assert!(e.state().flag.intro());
              assert_eq!(e.state().mode, CellMode::Aff);*/
            }
          }
        }
      }
      &SpineEntry::ICacheMux(x) => {
        match env.lookup_ref(x) {
          None => panic!("bug"),
          Some(e) => {
            /*match e.cel.mode {
              CellMode::_Top => {
                e.cel.mode = CellMode::Init;
              }
              CellMode::Init => {}
              _ => panic!("bug")
            }*/
            // FIXME FIXME
            //unimplemented!();
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
      &SpineEntry::Intro(x) => {
        match env.lookup_ref(x) {
          None => panic!("bug"),
          Some(e) => {
            match e.state().mode {
              CellMode::_Top => {}
              CellMode::Aff => {
                // FIXME FIXME: should warn here?
              }
              _ => panic!("bug")
            }
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
      &SpineEntry::Uninit(x) => {
        unimplemented!();
      }
      &SpineEntry::YieldSet(x, loc) => {
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
                  &SpineResume::PutMemV(k, v) => {
                    match v.downcast_ref::<f32>() {
                      None => {
                        println!("DEBUG: Spine::_step: YieldSet:   wrong type");
                      }
                      Some(v) => {
                        println!("DEBUG: Spine::_step: YieldSet:   key={:?} value={:?}", k, v);
                      }
                    }
                  }
                  &SpineResume::PutMemF(k, _f) => {
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
              let prev_clk = e.state().clk;
              let base_clk: Clock = self.ctr.into();
              let next_clk = base_clk.update();
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
                e.state().mode = CellMode::Aff;
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
                    (Locus::Mem, &SpineResume::PutMemV(_, _val)) => {
                      // FIXME FIXME
                      unimplemented!();
                    }
                    (Locus::Mem, &SpineResume::PutMemF(_, fun)) => {
                      let (pm, addr) = cel_.get_loc(xclk, &e.ty, Locus::Mem);
                      TL_PCTX.with(|pctx| {
                        let (_, icel) = pctx.lookup_pm(pm, addr).unwrap();
                        (fun)(e.ty.clone(), icel.as_mem_reg().unwrap());
                      });
                    }
                    _ => {
                      unimplemented!();
                    }
                  }
                  /*let mut clo = clo.borrow_mut();
                  let mut tgc = clo.thunk_.len();
                  for (i, &(tctr, th)) in clo.thunk_.iter().enumerate() {
                    if tctr >= xclk.ctr() {
                      tgc = i;
                      break;
                    }
                  }
                  let len_gc = clo.thunk_.len() - tgc;
                  clo.thunk_.copy_within(tgc .., 0);
                  clo.thunk_.resize_with(len_gc, || unreachable!());
                  clo.thunk_.push((xclk.ctr(), ThunkPtr::opaque()));
                  if clo.thunk_.len() != xclk.up as usize {
                    println!("DEBUG: Spine::_step: YieldSet:   x={:?} xclk={:?} clo={:?}",
                        x, xclk, clo);
                  }
                  assert_eq!(clo.thunk_.len(), xclk.up as usize);*/
                  clo.borrow_mut().update(xclk, ThunkPtr::opaque());
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
      &SpineEntry::YieldInit(x, loc) => {
        println!("DEBUG: Spine::_step: YieldInit: x={:?} loc={:?} key={:?}", x, loc, item.key());
        unimplemented!();
      }
      &SpineEntry::Alias(x, og) => {
        // FIXME FIXME
        //unimplemented!();
      }
      &SpineEntry::Snapshot(x, og) => {
        // FIXME FIXME
        //unimplemented!();
      }
      &SpineEntry::PushSeal(x) => {
        match env.lookup_ref(x) {
          None => panic!("bug"),
          Some(e) => {
            assert!(e.state().flag.intro());
            let xclk = e.state().clk;
            if xclk.ctr().is_nil() {
              println!("DEBUG: Spine::_step: PushSeal: x={:?} xclk={:?}", x, xclk);
            }
            assert!(!xclk.ctr().is_nil());
            /*println!("DEBUG: Spine::_step: PushSeal: thunkenv.arg.push: x={:?} xclk={:?}", x, xclk);
            thunkenv.arg.push((x, xclk));*/
            e.state().flag.set_seal();
          }
        }
      }
      &SpineEntry::Initialize(x, th) => {
        /*match env.lookup_ref(x) {
          None => panic!("bug"),
          Some(e) => {
            match e.state().mode {
              CellMode::_Top => {
                e.state().mode = CellMode::Init;
              }
              CellMode::Init => {}
              _ => panic!("bug")
            }
            //if !e.state().clk.happens_before(self.ctr).unwrap() {}
            match e.state().clk.partial_cmp_(self.ctr) {
              Some(Ordering::Less) => {}
              _ => panic!("bug")
            }
            assert!(!e.state().flag.intro());
            assert!(!e.state().flag.seal());
            // FIXME
            e.state().clk = self.ctr.into();
            // FIXME FIXME: set ithunk.
            //match (env.cache.contains(&x), e.ithunk) {}
            match (e.state().flag.cache(), e.ithunk) {
              (true, None) => {}
              (false, Some(thunk)) => {
                let te = match thunkenv.thunktab.get(&thunk) {
                  None => panic!("bug"),
                  Some(thunk) => thunk
                };
                /*match &e.cel.compute {
                  &InnerCell::Uninit => panic!("bug"),
                  &InnerCell::Primary => {
                    e.cel.primary.sync_thunk(x, &e.ty, &te.args, &te.thunk, e.cel.clk);
                  }
                  _ => {
                    e.cel.compute.sync_thunk(x, &e.ty, &te.args, &te.thunk, e.cel.clk);
                  }
                }*/
                // FIXME FIXME: the above api is not very ergonomic.
                unimplemented!();
                //te.thunk.apply(/*env,*/ &te.args, x, &e.ty, &mut e.cel);
              }
              _ => panic!("bug")
            }
            /*e.state().flag.reset();*/
            e.state().flag.set_intro();
          }
        }*/
        // FIXME FIXME
        let xclk = match env.lookup_mut_ref(x) {
          None => panic!("bug"),
          Some(e) => {
            match e.state().mode {
              CellMode::Init => {}
              _ => panic!("bug")
            }
            assert_eq!(e.state().clk.ctr(), self.ctr);
            assert!(e.state().flag.intro());
            assert!(!e.state().flag.seal());
            // FIXME FIXME
            let next_clk = e.state().clk.update();
            e.state().clk = next_clk;
            next_clk
          }
        };
        {
          let tclo = match thunkenv.update.get(&(x, xclk)) {
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
            ThunkRet::NotImpl => {
              println!("ERROR: Spine::_step: Initialize: thunk not implemented");
              panic!();
            }
            ThunkRet::Failure => {
              println!("ERROR: Spine::_step: Initialize: unrecoverable thunk failure");
              panic!();
            }
            ThunkRet::Success => {}
          }
          match env.lookup_ref(x) {
            None => panic!("bug"),
            Some(e) => {
              match e.cel_ {
                &Cell_::Phy(_, ref clo, _) |
                &Cell_::Cow(_, ref clo, _) => {
                  clo.borrow_mut().init(xclk, th);
                }
                _ => panic!("bug")
              }
            }
          }
        }
      }
      &SpineEntry::Apply(x, th) => {
        let xclk = match env.lookup_mut_ref(x) {
          None => panic!("bug"),
          Some(e) => {
            match e.state().mode {
              CellMode::Aff => {}
              _ => panic!("bug")
            }
            assert_eq!(e.state().clk.ctr(), self.ctr);
            assert_eq!(e.state().clk.up, 0);
            assert!(e.state().flag.intro());
            assert!(!e.state().flag.seal());
            // FIXME
            /*let tup = e.state().clk.up;
            if (tup as usize) != e.thunk.len() {
              println!("DEBUG: Spine::_step: tup={} e.thunk.len={}", tup, e.thunk.len());
              self._debug_dump();
              panic!("bug");
            }
            //assert_eq!((tup as usize), e.thunk.len());*/
            let next_clk = e.state().clk.update();
            /*e.thunk.push(th);*/
            e.state().clk = next_clk;
            next_clk
          }
        };
        {
          let tclo = match thunkenv.update.get(&(x, xclk)) {
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
            ThunkRet::NotImpl => {
              println!("ERROR: Spine::_step: Apply: thunk not implemented");
              panic!();
            }
            ThunkRet::Failure => {
              println!("ERROR: Spine::_step: Apply: unrecoverable thunk failure");
              panic!();
            }
            ThunkRet::Success => {}
          }
          match env.lookup_ref(x) {
            None => panic!("bug"),
            Some(e) => {
              match e.cel_ {
                &Cell_::Phy(_, ref clo, _) |
                &Cell_::Cow(_, ref clo, _) => {
                  /*clo.borrow_mut().thunk_.push((xclk.ctr(), th));
                  assert_eq!(clo.borrow().thunk_.len(), xclk.up as usize);*/
                  clo.borrow_mut().update(xclk, th);
                }
                _ => panic!("bug: Spine::_step: Apply: cel={:?}", e.cel_.name())
              }
            }
          }
        }
      }
      &SpineEntry::Accumulate(x, th) => {
        let xclk = match env.lookup_mut_ref(x) {
          None => panic!("bug"),
          Some(e) => {
            match e.state().mode {
              CellMode::Init => {}
              _ => panic!("bug")
            }
            assert_eq!(e.state().clk.ctr(), self.ctr);
            assert!(e.state().flag.intro());
            assert!(!e.state().flag.seal());
            // FIXME
            /*let tup = e.state().clk.up;
            if (tup as usize) != e.thunk.len() {
              println!("DEBUG: Spine::_step: tup={} e.thunk.len={}", tup, e.thunk.len());
              self._debug_dump();
              panic!("bug");
            }
            //assert_eq!((tup as usize), e.thunk.len());*/
            let next_clk = e.state().clk.update();
            /*e.thunk.push(th);*/
            e.state().clk = next_clk;
            next_clk
          }
        };
        {
          let tclo = match thunkenv.update.get(&(x, xclk)) {
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
            ThunkRet::NotImpl => {
              println!("ERROR: Spine::_step: Accumulate: thunk not implemented");
              panic!();
            }
            ThunkRet::Failure => {
              println!("ERROR: Spine::_step: Accumulate: unrecoverable thunk failure");
              panic!();
            }
            ThunkRet::Success => {}
          }
          match env.lookup_ref(x) {
            None => panic!("bug"),
            Some(e) => {
              match e.cel_ {
                &Cell_::Phy(_, ref clo, _) |
                &Cell_::Cow(_, ref clo, _) => {
                  /*clo.borrow_mut().thunk_.push((xclk.ctr(), th));
                  assert_eq!(clo.borrow().thunk_.len(), xclk.up as usize);*/
                  clo.borrow_mut().update(xclk, th);
                }
                _ => panic!("bug")
              }
            }
          }
        }
      }
      &SpineEntry::UnsafeWrite(x, th) => {
        unimplemented!();
      }
      &SpineEntry::Seal(x) => {
        match env.lookup_ref(x) {
          None => panic!("bug"),
          Some(e) => {
            match e.state().mode {
              CellMode::_Top => panic!("bug"),
              _ => {}
            }
            assert_eq!(e.state().clk.ctr(), self.ctr);
            /*assert!(e.state().clk.up > 0);*/
            assert!(e.state().clk.up != u32::max_value());
            assert!(e.state().flag.intro());
            assert!(!e.state().flag.seal());
            /*e.state().flag.reset();*/
            e.state().flag.set_seal();
          }
        }
      }
      &SpineEntry::Unseal(x) => {
        unimplemented!();
      }
      /*&SpineEntry::Eval(x) => {
        match env.lookup_ref(x) {
          None => panic!("bug"),
          Some(e) => {
            assert_eq!(e.state().clk.ctr(), self.ctr);
            assert!(e.state().flag.intro());
            assert!(e.state().flag.seal());
            // FIXME FIXME
            unimplemented!()
            /*
            match &e.state().compute {
              &InnerCell::Uninit => panic!("bug"),
              &InnerCell::Primary => {
                assert!(e.state().primary.synced(&e.ty, e.state().clk));
              }
              _ => {
                match &e.state().primary {
                  &InnerCell::Uninit => {}
                  &InnerCell::Primary => panic!("bug"),
                  _ => {
                    e.state().primary.sync_cell(x, &e.ty, &e.state().compute, e.state().clk);
                  }
                }
              }
            }
            */
          }
        }
      }*/
      &SpineEntry::Unsync(x) => {
        match env.lookup_ref(x) {
          None => panic!("bug"),
          Some(e) => {
            assert_eq!(e.state().clk.ctr(), self.ctr);
            assert!(e.state().flag.intro());
            assert!(e.state().flag.seal());
            // FIXME FIXME
            unimplemented!()
            /*
            match &e.state().compute {
              &InnerCell::Uninit => panic!("bug"),
              &InnerCell::Primary => {
                assert!(e.state().primary.synced(&e.ty, e.state().clk));
              }
              _ => {
                match &e.state().primary {
                  &InnerCell::Uninit => {}
                  &InnerCell::Primary => panic!("bug"),
                  _ => {
                    e.state().primary.sync_cell(x, &e.ty, &e.state().compute, e.state().clk);
                  }
                }
                e.state().compute.unsync(e.state().clk);
              }
            }
            */
          }
        }
      }
      &SpineEntry::Bot => {
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
        self.ctr, self.ctlp, self.hltp, self.curp);
    for (i, e) in self.log.iter().enumerate() {
      println!("DEBUG: Spine::_debug_dump: log[{}]={:?}", i, e);
    }
  }
}

pub struct Backward {
  pub env: SpineEnv,
  //pub frontier: Vec<CellPtr>,
  pub frontier_set: HashSet<CellPtr>,
  pub complete_set: HashSet<CellPtr>,
}

impl Backward {
  pub fn _backward_rec(&mut self, ctx: &Ctx, tg_idx: usize, tg_clk: Clock, tg: CellPtr) {
    for idx in (0 .. tg_idx).rev() {
      let spine = ctx.spine.borrow();
      let e = &spine.log[idx];
      match e {
        // FIXME FIXME: other cases.
        &SpineEntry::Cache(..) => {
          unimplemented!();
        }
        &SpineEntry::ICacheMux(..) => {
          unimplemented!();
        }
        &SpineEntry::OIntro(y, _) |
        &SpineEntry::Intro(y) => {
          assert!(!self.complete_set.contains(&y));
          if !self.frontier_set.contains(&y) {
            self.env.unstep(e);
            continue;
          }
          self.frontier_set.remove(&y);
          self.complete_set.insert(y);
        }
        &SpineEntry::Initialize(y, ith) => {
          assert!(!self.complete_set.contains(&y));
          if !self.frontier_set.contains(&y) {
            self.env.unstep(e);
            continue;
          }
          unimplemented!();
        }
        &SpineEntry::Apply(y, th) => {
          assert!(!self.complete_set.contains(&y));
          if !self.frontier_set.contains(&y) {
            self.env.unstep(e);
            continue;
          }
          drop(e);
          let yclk = spine._version(y).unwrap();
          let dy = match spine.cur_env.gradr.get(&(tg, y, tg_clk)) {
            None => panic!("bug"),
            //None => CellPtr::nil(),
            Some(&dy) => dy
          };
          drop(spine);
          let thunkenv = ctx.thunkenv.borrow();
          match thunkenv.update.get(&(y, yclk)) {
            None => panic!("bug"),
            Some(tclo) => {
              let arg = tclo.arg.clone();
              let spec_ = match thunkenv.thunktab.get(&tclo.pthunk) {
                None => panic!("bug"),
                Some(te) => te.pthunk.spec_.clone()
              };
              drop(tclo);
              drop(thunkenv);
              // FIXME FIXME: clocks.
              let mut arg_adj = Vec::with_capacity(arg.len());
              for &(x, _) in arg.iter() {
                assert!(!self.complete_set.contains(&x));
                self.frontier_set.insert(x);
                let ty_ = ctx_lookup_type(x);
                arg_adj.push(ctx_init_zeros(ty_));
              }
              let mut spine = ctx.spine.borrow_mut();
              for (&(x, _), &dx) in arg.iter().zip(arg_adj.iter()) {
                match spine.cur_env.gradr.insert((tg, x, tg_clk), dx) {
                  None => {}
                  Some(o_dx) => {
                    assert_eq!(o_dx, dx);
                  }
                }
              }
              drop(spine);
              match spec_.pop_adj(&arg, y, yclk, dy, &mut arg_adj) {
                Err(_) => unimplemented!(),
                Ok(_) => {}
              }
            }
          }
        }
        &SpineEntry::Accumulate(y, th) => {
          assert!(!self.complete_set.contains(&y));
          if !self.frontier_set.contains(&y) {
            self.env.unstep(e);
            continue;
          }
          drop(e);
          let dy = match spine.cur_env.gradr.get(&(tg, y, tg_clk)) {
            None => panic!("bug"),
            //None => CellPtr::nil(),
            Some(&dy) => dy
          };
          drop(spine);
          unimplemented!();
        }
        &SpineEntry::Seal(..) => {
          unimplemented!();
        }
        &SpineEntry::Unseal(..) => {
          unimplemented!();
        }
        _ => {}
      }
      let spine = ctx.spine.borrow();
      self.env.unstep(&spine.log[idx]);
    }
    let mut spine = ctx.spine.borrow_mut();
    match spine.cur_env.bwd.get_mut(&tg) {
      None => {}
      Some(bwd_clk) => {
        *bwd_clk = tg_clk;
      }
    }
  }
}

#[track_caller]
pub fn backward(tg: CellPtr) {
  panick_wrap(|| TL_CTX.with(|ctx| {
    // TODO TODO
    let spine = ctx.spine.borrow();
    let mut env = spine.cur_env.clone();
    for (idx, e) in spine.log.iter().enumerate().rev() {
      match e {
        // FIXME FIXME: other cases.
        &SpineEntry::OIntro(y, _) => {
          if y == tg {
            return;
          }
        }
        &SpineEntry::Apply(y, th) => {
          if y == tg {
            drop(e);
            let yclk = spine._version(y).unwrap();
            let tg_clk = match spine.cur_env.state.get(&tg) {
              None => panic!("bug"),
              Some(state) => state.clk
            };
            match spine.cur_env.bwd.get(&tg) {
              None => {}
              Some(&bwd_clk) => {
                match bwd_clk.partial_cmp(&tg_clk) {
                  None => panic!("bug"),
                  Some(Ordering::Greater) => panic!("bug"),
                  Some(Ordering::Less) => {}
                  Some(Ordering::Equal) => {
                    return;
                  }
                }
              }
            }
            drop(spine);
            let tg_idx = idx;
            let tg_ty_ = ctx_lookup_type(tg);
            // FIXME: make this a special cow constant.
            /*let sink = match tg_ty_.dtype {
              Dtype::Fp32 => {
                let value = 1.0_f32;
                ctx_pop_thunk(SetScalarFutThunkSpec{val: value.into_scalar_val()})
              }
              _ => unimplemented!()
            };*/
            let sink = if !tg_ty_.is_scalar() {
              // FIXME FIXME: this deserves a more informative error message.
              panic!("ERROR");
            } else {
              ctx_set_ones(tg_ty_)
            };
            let thunkenv = ctx.thunkenv.borrow();
            match thunkenv.update.get(&(y, yclk)) {
              None => panic!("bug"),
              Some(tclo) => {
                let arg = tclo.arg.clone();
                let spec_ = match thunkenv.thunktab.get(&tclo.pthunk) {
                  None => panic!("bug"),
                  Some(te) => te.pthunk.spec_.clone()
                };
                drop(tclo);
                drop(thunkenv);
                // FIXME FIXME: clocks.
                let mut frontier_set = HashSet::new();
                let mut arg_adj = Vec::with_capacity(arg.len());
                for &(x, _) in arg.iter() {
                  frontier_set.insert(x);
                  let ty_ = ctx_lookup_type(x);
                  arg_adj.push(ctx_init_zeros(ty_));
                }
                let mut spine = ctx.spine.borrow_mut();
                for (&(x, _), &dx) in arg.iter().zip(arg_adj.iter()) {
                  assert!(spine.cur_env.gradr.insert((tg, x, tg_clk), dx).is_none());
                }
                drop(spine);
                match spec_.pop_adj(&arg, y, yclk, sink, &mut arg_adj) {
                  Err(_) => unimplemented!(),
                  Ok(_) => {}
                }
                let spine = ctx.spine.borrow();
                env.unstep(&spine.log[idx]);
                drop(spine);
                let mut bwd = Backward{
                  env,
                  frontier_set,
                  complete_set: HashSet::new(),
                };
                return bwd._backward_rec(ctx, tg_idx, tg_clk, tg);
              }
            }
          }
        }
        &SpineEntry::Accumulate(y, th) => {
          if y == tg {
            unimplemented!();
          }
        }
        _ => {}
      }
      env.unstep(e);
    }
  }))
}

#[derive(Default)]
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
