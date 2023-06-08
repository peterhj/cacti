use crate::cell::*;
use crate::clock::{Counter, Clock};
use crate::ctx::{TL_CTX, Ctx, CtxCtr, CtxEnv, CtxThunkEnv, ctx_lookup_type, ctx_lookup_or_insert_gradr, ctx_accumulate_gradr, ctx_init_zeros, ctx_set_ones, ctx_pop_thunk};
use crate::panick::{panick_wrap};
use crate::thunk::*;
use crate::thunk::op::{SetScalarFutThunkSpec};

use std::cmp::{Ordering};
use std::collections::{HashMap, HashSet};
//use std::mem::{swap};

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
  CacheAff(CellPtr),
  ICacheMux(CellPtr),
  OIntro(CellPtr, CellPtr),
  Intro(CellPtr),
  Uninit(CellPtr),
  //IntroFin(CellPtr),
  Initialize(CellPtr, ThunkPtr),
  Apply(CellPtr, ThunkPtr),
  Accumulate(CellPtr, ThunkPtr),
  Seal(CellPtr),
  Unseal(CellPtr),
  Eval(CellPtr),
  //Uneval(CellPtr),
  Unsync(CellPtr),
  //Alias(CellPtr, CellPtr),
  //Cache(CellPtr),
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
      &SpineEntry::CacheAff(..)   => SpineEntryName::CacheAff,
      &SpineEntry::ICacheMux(..)  => SpineEntryName::ICacheMux,
      &SpineEntry::OIntro(..)     => SpineEntryName::OIntro,
      &SpineEntry::Intro(..)      => SpineEntryName::Intro,
      &SpineEntry::Uninit(..)     => SpineEntryName::Uninit,
      &SpineEntry::Initialize(..) => SpineEntryName::Initialize,
      &SpineEntry::Apply(..)      => SpineEntryName::Apply,
      &SpineEntry::Accumulate(..) => SpineEntryName::Accumulate,
      &SpineEntry::Seal(..)       => SpineEntryName::Seal,
      &SpineEntry::Unseal(..)     => SpineEntryName::Unseal,
      &SpineEntry::Eval(..)       => SpineEntryName::Eval,
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
  CacheAff,
  ICacheMux,
  OIntro,
  Intro,
  Uninit,
  Initialize,
  Apply,
  Accumulate,
  Seal,
  Unseal,
  Eval,
  //Uneval,
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
  Yield_,
  Break_,
  Bot,
}

pub type SpineCellState = CellState;

/*pub struct SpineCellState {
  // FIXME FIXME
  //pub alias_ct: u32,
  pub hasalias: bool,
  pub is_alias: bool,
  pub mode:     CellMode,
  pub flag:     CellFlag,
  pub clk:      Clock,
}*/

#[derive(Clone, Default)]
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
  //pub aliases:  HashMap<(CellPtr, Clock), HashSet<CellPtr>>,
  //pub alias_of: HashMap<CellPtr, (CellPtr, Clock)>,
  pub iapply:   HashMap<CellPtr, (u32, ThunkPtr, Clock)>,
  pub apply_:   HashMap<CellPtr, Vec<(u32, ThunkPtr, Clock)>>,
  pub bwd:      HashMap<CellPtr, Clock>,
  //pub bwd:      HashSet<(CellPtr, Clock)>,
  pub gradr:    HashMap<(CellPtr, CellPtr, Clock), CellPtr>,
}

impl SpineEnv {
  pub fn reset(&mut self, ctr: Counter) {
    /*self.aff.clear();
    self.init.clear();
    self.cache.clear();
    self.intro.clear();
    self.seal.clear();
    self.apply.clear();
    //self.eval.clear();*/
    self.ctr = ctr;
    self.state.clear();
    self.iapply.clear();
    self.apply_.clear();
    self.bwd.clear();
    self.gradr.clear();
  }

  pub fn step(&mut self, sp: u32, e: &SpineEntry) {
    // FIXME FIXME
    match e {
      &SpineEntry::OIntro(x, _) => {
        /*self.cache.insert(x, sp);*/
        // FIXME FIXME
        unimplemented!();
      }
      &SpineEntry::CacheAff(x) => {
        /*assert!(!self.intro.contains_key(&x));
        assert!(!self.seal.contains_key(&x));
        self.intro.insert(x, sp);
        self.seal.insert(x, sp);
        self.aff.insert(x, sp);*/
        // FIXME FIXME
        unimplemented!();
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
            assert_eq!(state.clk.tup, 0);
            state.mode.set_aff();
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
            /*assert_eq!(state.clk.tup, 0);*/
            state.mode.set_init();
            state.flag.set_intro();
            let next_clk = self.ctr.into();
            state.clk = next_clk;
          }
        }
      }
      &SpineEntry::Initialize(x, ith) => {
        /*// FIXME FIXME
        self.intro.insert(x, sp);
        self.init.insert(x, (sp, ith));*/
        match self.state.get_mut(&x) {
          None => panic!("bug"),
          Some(state) => {
            assert_eq!(state.mode, CellMode::_Top);
            assert!(!state.flag.intro());
            assert!(!state.flag.seal());
            /*assert_eq!(state.clk.tup, 0);*/
            state.mode.set_init();
            state.flag.set_intro();
            let next_clk = self.ctr.into();
            assert!(self.iapply.insert(x, (sp, ith, next_clk)).is_none());
            state.clk = next_clk;
          }
        }
      }
      &SpineEntry::Apply(x, th) => {
        /*assert!(self.aff.contains_key(&x));
        match self.apply.get_mut(&x) {
          None => {
            let mut thlist = Vec::new();
            thlist.push((sp, th));
            self.apply.insert(x, thlist);
          }
          Some(thlist) => {
            thlist.push((sp, th));
          }
        }
        /*self.seal.insert(x, sp);*/*/
        match self.state.get_mut(&x) {
          None => panic!("bug"),
          Some(state) => {
            assert_eq!(state.mode, CellMode::Aff);
            assert!(state.flag.intro());
            assert!(!state.flag.seal());
            assert_eq!(state.clk.tup, 0);
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
                assert_eq!(thlist.len(), next_clk.tup as _);
              }
            }
            state.clk = next_clk;
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
                assert_eq!(thlist.len(), next_clk.tup as _);
              }
            }
            state.clk = next_clk;
          }
        }
      }
      &SpineEntry::Seal(x) => {
        /*// TODO: idempotent seal?
        /*assert!(!self.seal.contains_key(&x));*/
        assert!(self.init.contains_key(&x));
        self.seal.insert(x, sp);*/
        match self.state.get_mut(&x) {
          None => panic!("bug"),
          Some(state) => {
            assert!(state.mode != CellMode::_Top);
            assert!(state.flag.intro());
            assert!(!state.flag.seal());
            state.flag.set_seal();
          }
        }
      }
      &SpineEntry::Unseal(x) => {
        /*// TODO: idempotent seal?
        /*assert!(self.seal.contains_key(&x));*/
        assert!(self.init.contains_key(&x));
        self.seal.remove(&x);*/
        match self.state.get_mut(&x) {
          None => panic!("bug"),
          Some(state) => {
            assert!(state.mode != CellMode::_Top);
            assert!(state.flag.seal());
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
  pub env:  SpineEnv,
  pub log:  Vec<SpineEntry>,
}

impl Default for Spine {
  fn default() -> Spine {
    Spine{
      ctr:  Counter::default(),
      ctlp: 0,
      hltp: 0,
      curp: 0,
      env:  SpineEnv::default(),
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
    self.env.reset(self.ctr);
    self.log.clear();
  }

  pub fn opaque(&mut self, x: CellPtr, og: CellPtr) {
    let sp = self.curp;
    self.curp += 1;
    let e = SpineEntry::OIntro(x, og);
    self.env.step(sp, &e);
    self.log.push(e);
  }

  pub fn cache_aff(&mut self, x: CellPtr) {
    let sp = self.curp;
    self.curp += 1;
    let e = SpineEntry::CacheAff(x);
    self.env.step(sp, &e);
    self.log.push(e);
  }

  pub fn intro_aff(&mut self, x: CellPtr) {
    let sp = self.curp;
    self.curp += 1;
    let e = SpineEntry::Intro(x);
    self.env.step(sp, &e);
    self.log.push(e);
  }

  pub fn init_cache_mux(&mut self, x: CellPtr) {
    let sp = self.curp;
    self.curp += 1;
    let e = SpineEntry::ICacheMux(x);
    self.env.step(sp, &e);
    self.log.push(e);
  }

  pub fn initialize(&mut self, ith: ThunkPtr, x: CellPtr) {
    // FIXME FIXME
    unimplemented!();
    /*
    let sp = self.curp;
    self.curp += 1;
    let e = SpineEntry::Initialize(x, ith);
    self.env.step(sp, &e);
    self.log.push(e);
    */
  }

  pub fn apply(&mut self, th: ThunkPtr, y: CellPtr) {
    let sp = self.curp;
    self.curp += 1;
    let e = SpineEntry::Apply(y, th);
    self.env.step(sp, &e);
    self.log.push(e);
  }

  pub fn accumulate(&mut self, th: ThunkPtr, y: CellPtr) {
    let sp = self.curp;
    self.curp += 1;
    let e = SpineEntry::Accumulate(y, th);
    self.env.step(sp, &e);
    self.log.push(e);
  }

  pub fn seal(&mut self, x: CellPtr) {
    let sp = self.curp;
    self.curp += 1;
    let e = SpineEntry::Seal(x);
    self.env.step(sp, &e);
    self.log.push(e);
  }

  pub fn unseal_mux(&mut self, x: CellPtr) {
    let sp = self.curp;
    self.curp += 1;
    let e = SpineEntry::Unseal(x);
    self.env.step(sp, &e);
    self.log.push(e);
  }

  pub fn unsync(&mut self, x: CellPtr) {
    unimplemented!();
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

  pub fn _resume(&mut self, ctr: &CtxCtr, env: &mut CtxEnv, thunkenv: &mut CtxThunkEnv, /*target: CellPtr, tg_clk: Clock*/) -> SpineRet {
    //self._start();
    println!("DEBUG: Spine::_resume: ctr={:?} ctlp={} hltp={} curp={}",
        self.ctr, self.ctlp, self.hltp, self.curp);
    self.hltp = self.curp;
    loop {
      let state = self._step(ctr, env, thunkenv);
      match state {
        SpineRet::Bot => {
          return state;
        }
        _ => {}
      }
      self.ctlp += 1;
      match state {
        SpineRet::Halt   |
        SpineRet::Yield_ |
        SpineRet::Break_ => {
          return state;
        }
        _ => {}
      }
    }
  }

  pub fn _step(&self, ctr: &CtxCtr, env: &mut CtxEnv, thunkenv: &mut CtxThunkEnv) -> SpineRet {
    if self.ctlp >= self.hltp {
      println!("DEBUG: Spine::_step: ctr={:?} ctlp={} hltp={} curp={} halt",
          self.ctr, self.ctlp, self.hltp, self.curp);
      return SpineRet::Halt;
    }
    let mut ret = SpineRet::_Top;
    let entry = &self.log[self.ctlp as usize];
    println!("DEBUG: Spine::_step: ctr={:?} ctlp={} hltp={} curp={} entry={:?}",
        self.ctr, self.ctlp, self.hltp, self.curp, entry.name());
    match entry {
      // TODO
      &SpineEntry::Yield_ => {
        ret = SpineRet::Yield_;
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
      }
      &SpineEntry::CacheAff(x) => {
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
            // FIXME FIXME
            //unimplemented!();
            let flag = !e.state().flag.set_cache();
            let mode = match e.state().mode.set_aff() {
              Err(_) => panic!("bug"),
              Ok(prev) => !prev
            };
            assert!(flag);
            assert!(mode);
            assert!(self.ctr.succeeds(e.state().clk));
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
            let mode = match e.state().mode.set_mux() {
              Err(_) => panic!("bug"),
              Ok(prev) => !prev
            };
            assert!(flag);
            assert!(mode);
            //e.cel.clk.
          }
        }
      }
      /*
      &SpineEntry::IntroFin(x) => {
        match env.lookup_mut(x) {
          None => panic!("bug"),
          Some(e) => {
            match e.cel.mode {
              CellMode::_Top => {
                e.cel.mode = CellMode::Fin;
              }
              CellMode::Fin => {}
              _ => panic!("bug")
            }
            if !e.cel.clk.happens_before(self.ctr).unwrap() {
              panic!("bug");
            }
            assert!(!e.cel.flag.intro());
            assert!(!e.cel.flag.seal());
            // FIXME
            e.cel.clk = self.ctr.into();
            match &e.cel.compute {
              &InnerCell::Uninit => panic!("bug"),
              &InnerCell::Primary => {
                assert!(e.cel.primary.synced(&e.ty, e.cel.clk));
              }
              _ => {
                e.cel.compute.sync_cell(&e.ty, &e.cel.primary, e.cel.clk);
              }
            }
            // FIXME: first, wait for primary sync.
            //e.cel._;
            e.cel.flag.reset();
            e.cel.flag.set_intro();
          }
        }
      }
      */
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
            if !e.state().clk.happens_before(self.ctr).unwrap() {
              panic!("bug");
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
      &SpineEntry::Initialize(x, ith) => {
        match env.lookup_ref(x) {
          None => panic!("bug"),
          Some(e) => {
            match e.state().mode {
              CellMode::_Top => {
                e.state().mode = CellMode::Init;
              }
              CellMode::Init => {}
              _ => panic!("bug")
            }
            if !e.state().clk.happens_before(self.ctr).unwrap() {
              panic!("bug");
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
        }
      }
      &SpineEntry::Apply(x, th) => {
        match env.lookup_mut_ref(x) {
          None => panic!("bug"),
          Some(e) => {
            match e.state().mode {
              CellMode::Aff => {}
              _ => panic!("bug")
            }
            assert_eq!(e.state().clk.ctr(), self.ctr);
            assert_eq!(e.state().clk.tup, 0);
            assert!(e.state().flag.intro());
            assert!(!e.state().flag.seal());
            // FIXME
            let tup = e.state().clk.tup;
            if (tup as usize) != e.thunk.len() {
              println!("DEBUG: Spine::_step: tup={} e.thunk.len={}", tup, e.thunk.len());
              self._debug_dump();
              panic!("bug");
            }
            //assert_eq!((tup as usize), e.thunk.len());
            let next_clk = e.state().clk.update();
            e.thunk.push(th);
            e.state().clk = next_clk;
          }
        }
        {
          let te = match thunkenv.thunktab.get(&th) {
            None => panic!("bug"),
            Some(thunk) => thunk
          };
          te.thunk.apply(ctr, env, &te.arg, x);
        }
        /*match env.lookup_ref(x) {
          None => panic!("bug"),
          Some(e) => {
            e.state().flag.set_seal();
          }
        }*/
      }
      &SpineEntry::Accumulate(x, th) => {
        match env.lookup_mut_ref(x) {
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
            let tup = e.state().clk.tup;
            if (tup as usize) != e.thunk.len() {
              println!("DEBUG: Spine::_step: tup={} e.thunk.len={}", tup, e.thunk.len());
              self._debug_dump();
              panic!("bug");
            }
            //assert_eq!((tup as usize), e.thunk.len());
            let next_clk = e.state().clk.update();
            e.thunk.push(th);
            e.state().clk = next_clk;
          }
        }
        {
          let te = match thunkenv.thunktab.get(&th) {
            None => panic!("bug"),
            Some(thunk) => thunk
          };
          te.thunk.accumulate(ctr, env, &te.arg, x);
        }
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
            /*assert!(e.state().clk.tup > 0);*/
            assert!(e.state().clk.tup != u32::max_value());
            assert!(e.state().flag.intro());
            assert!(!e.state().flag.seal());
            /*e.state().flag.reset();*/
            e.state().flag.set_seal();
          }
        }
      }
      &SpineEntry::Eval(x) => {
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
      }
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
      /*&SpineEntry::Alias(x, y) => {
        unimplemented!();
      }*/
      &SpineEntry::Bot => {
        ret = SpineRet::Bot;
      }
      _ => unimplemented!()
    }
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
      self.env.unstep(e);
      match e {
        // FIXME FIXME: other cases.
        &SpineEntry::CacheAff(..) => {
          unimplemented!();
        }
        &SpineEntry::ICacheMux(..) => {
          unimplemented!();
        }
        &SpineEntry::OIntro(y, _) |
        &SpineEntry::Intro(y) => {
          assert!(!self.complete_set.contains(&y));
          if !self.frontier_set.contains(&y) {
            continue;
          }
          self.frontier_set.remove(&y);
          self.complete_set.insert(y);
        }
        &SpineEntry::Initialize(y, ith) => {
          assert!(!self.complete_set.contains(&y));
          if !self.frontier_set.contains(&y) {
            continue;
          }
          unimplemented!();
        }
        &SpineEntry::Apply(y, th) => {
          drop(e);
          assert!(!self.complete_set.contains(&y));
          if !self.frontier_set.contains(&y) {
            continue;
          }
          let dy = match spine.env.gradr.get(&(tg, y, tg_clk)) {
            None => panic!("bug"),
            //None => CellPtr::nil(),
            Some(&dy) => dy
          };
          drop(spine);
          let thunkenv = ctx.thunkenv.borrow();
          match thunkenv.thunktab.get(&th) {
            None => panic!("bug"),
            Some(te) => {
              let arg = te.arg.clone();
              let spec_ = te.thunk.spec_.clone();
              drop(te);
              drop(thunkenv);
              let mut arg_adj = Vec::with_capacity(arg.len());
              for &x in arg.iter() {
                assert!(!self.complete_set.contains(&x));
                self.frontier_set.insert(x);
                let ty_ = ctx_lookup_type(x);
                arg_adj.push(ctx_init_zeros(ty_));
              }
              let mut spine = ctx.spine.borrow_mut();
              for (&x, &dx) in arg.iter().zip(arg_adj.iter()) {
                match spine.env.gradr.insert((tg, x, tg_clk), dx) {
                  None => {}
                  Some(o_dx) => {
                    assert_eq!(o_dx, dx);
                  }
                }
              }
              drop(spine);
              match spec_.pop_adj(&arg, dy, &arg_adj) {
                Err(_) => unimplemented!(),
                Ok(_) => {}
              }
            }
          }
        }
        &SpineEntry::Accumulate(y, th) => {
          drop(e);
          assert!(!self.complete_set.contains(&y));
          if !self.frontier_set.contains(&y) {
            continue;
          }
          let dy = match spine.env.gradr.get(&(tg, y, tg_clk)) {
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
    }
    let mut spine = ctx.spine.borrow_mut();
    match spine.env.bwd.get_mut(&tg) {
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
    let mut env = spine.env.clone();
    for (idx, e) in spine.log.iter().enumerate().rev() {
      env.unstep(e);
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
            let tg_clk = match spine.env.state.get(&tg) {
              None => panic!("bug"),
              Some(state) => state.clk
            };
            match spine.env.bwd.get(&tg) {
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
              Dtype::Float32 => {
                let value = 1.0_f32;
                ctx_pop_thunk(SetScalarFutThunkSpec{val: value.into_thunk_val()})
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
            match thunkenv.thunktab.get(&th) {
              None => panic!("bug"),
              Some(te) => {
                let arg = te.arg.clone();
                let spec_ = te.thunk.spec_.clone();
                drop(te);
                drop(thunkenv);
                let mut frontier_set = HashSet::new();
                let mut arg_adj = Vec::with_capacity(arg.len());
                for &x in arg.iter() {
                  frontier_set.insert(x);
                  let ty_ = ctx_lookup_type(x);
                  arg_adj.push(ctx_init_zeros(ty_));
                }
                let mut spine = ctx.spine.borrow_mut();
                for (&x, &dx) in arg.iter().zip(arg_adj.iter()) {
                  assert!(spine.env.gradr.insert((tg, x, tg_clk), dx).is_none());
                }
                drop(spine);
                match spec_.pop_adj(&arg, sink, &arg_adj) {
                  Err(_) => unimplemented!(),
                  Ok(_) => {}
                }
                let complete_set = HashSet::new();
                let mut bwd = Backward{
                  env,
                  frontier_set,
                  complete_set,
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
        &SpineEntry::CacheAff(x) => {
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
