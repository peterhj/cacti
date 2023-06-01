use crate::cell::*;
use crate::clock::{Counter};
use crate::ctx::{TL_CTX, CtxCtr, CtxEnv, CtxThunkEnv};
use crate::panick::{panick_wrap};
use crate::thunk::*;

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
  Opaque(CellPtr, CellPtr),
  CacheAff(CellPtr),
  IntroAff(CellPtr),
  ICacheMux(CellPtr),
  InitMux(ThunkPtr, CellPtr),
  //IntroFin(CellPtr),
  SealMux(CellPtr),
  UnsealMux(CellPtr),
  ApplyAff(ThunkPtr, CellPtr),
  ApplyMux(ThunkPtr, CellPtr),
  Eval(CellPtr),
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
      &SpineEntry::Opaque(..)     => SpineEntryName::Opaque,
      &SpineEntry::CacheAff(..)   => SpineEntryName::CacheAff,
      &SpineEntry::ICacheMux(..)  => SpineEntryName::ICacheMux,
      &SpineEntry::IntroAff(..)   => SpineEntryName::IntroAff,
      &SpineEntry::InitMux(..)    => SpineEntryName::InitMux,
      &SpineEntry::SealMux(..)    => SpineEntryName::SealMux,
      &SpineEntry::UnsealMux(..)  => SpineEntryName::UnsealMux,
      &SpineEntry::ApplyAff(..)   => SpineEntryName::ApplyAff,
      &SpineEntry::ApplyMux(..)   => SpineEntryName::ApplyMux,
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
  Opaque,
  CacheAff,
  ICacheMux,
  IntroAff,
  InitMux,
  SealMux,
  UnsealMux,
  ApplyAff,
  ApplyMux,
  Eval,
  Unsync,
  // TODO
  Bot,
}

#[derive(Clone, Copy, Debug)]
#[repr(u8)]
pub enum SpineState {
  _Top,
  Halt,
  Yield_,
  Break_,
  Bot,
}

#[derive(Default)]
pub struct SpineEnv {
  pub aff:      HashMap<CellPtr, u32>,
  pub mux:      HashMap<CellPtr, u32>,
  pub cache:    HashMap<CellPtr, u32>,
  pub intro:    HashMap<CellPtr, u32>,
  //pub fin:      HashSet<CellPtr>,
  pub seal:     HashMap<CellPtr, u32>,
  pub apply:    HashMap<CellPtr, Vec<(u32, ThunkPtr)>>,
  pub eval:     HashMap<CellPtr, u32>,
}

impl SpineEnv {
  pub fn reset(&mut self) {
    self.aff.clear();
    self.mux.clear();
    self.cache.clear();
    self.intro.clear();
    self.seal.clear();
    self.apply.clear();
    self.eval.clear();
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
    self.env.reset();
    self.log.clear();
  }

  pub fn opaque(&mut self, x: CellPtr, og: CellPtr) {
    let sp = self.curp;
    self.curp += 1;
    self.env.cache.insert(x, sp);
    self.log.push(SpineEntry::Opaque(x, og));
  }

  pub fn cache_aff(&mut self, x: CellPtr) {
    assert!(!self.env.intro.contains_key(&x));
    assert!(!self.env.seal.contains_key(&x));
    let sp = self.curp;
    self.curp += 1;
    self.env.intro.insert(x, sp);
    self.env.seal.insert(x, sp);
    self.env.aff.insert(x, sp);
    self.log.push(SpineEntry::CacheAff(x));
  }

  pub fn intro_aff(&mut self, x: CellPtr) {
    assert!(!self.env.intro.contains_key(&x));
    assert!(!self.env.seal.contains_key(&x));
    let sp = self.curp;
    self.curp += 1;
    self.env.intro.insert(x, sp);
    self.env.aff.insert(x, sp);
    self.log.push(SpineEntry::IntroAff(x));
  }

  pub fn init_cache_mux(&mut self, x: CellPtr) {
    assert!(!self.env.seal.contains_key(&x));
    let sp = self.curp;
    self.curp += 1;
    self.env.intro.insert(x, sp);
    self.env.mux.insert(x, sp);
    self.log.push(SpineEntry::ICacheMux(x));
  }

  pub fn init_mux(&mut self, x: CellPtr) {
    // FIXME FIXME
    unimplemented!();
    /*
    let sp = self.curp;
    self.curp += 1;
    self.env.intro.insert(x, sp);
    self.env.mux.insert(x, sp);
    self.log.push(SpineEntry::InitMux(x));
    */
  }

  pub fn seal_mux(&mut self, x: CellPtr) {
    // TODO: idempotent seal?
    /*assert!(!self.env.seal.contains_key(&x));*/
    assert!(self.env.mux.contains_key(&x));
    let sp = self.curp;
    self.curp += 1;
    self.env.seal.insert(x, sp);
    self.log.push(SpineEntry::SealMux(x));
  }

  pub fn unseal_mux(&mut self, x: CellPtr) {
    // TODO: idempotent seal?
    /*assert!(self.env.seal.contains_key(&x));*/
    assert!(self.env.mux.contains_key(&x));
    //let sp = self.curp;
    self.curp += 1;
    self.env.seal.remove(&x);
    self.log.push(SpineEntry::UnsealMux(x));
  }

  pub fn apply_aff(&mut self, th: ThunkPtr, x: CellPtr) {
    assert!(self.env.aff.contains_key(&x));
    let sp = self.curp;
    self.curp += 1;
    match self.env.apply.get_mut(&x) {
      None => {
        let mut thlist = Vec::new();
        thlist.push((sp, th));
        self.env.apply.insert(x, thlist);
      }
      Some(thlist) => {
        thlist.push((sp, th));
      }
    }
    self.env.seal.insert(x, sp);
    self.log.push(SpineEntry::ApplyAff(th, x));
  }

  pub fn apply_mux(&mut self, th: ThunkPtr, x: CellPtr) {
    assert!(self.env.mux.contains_key(&x));
    unimplemented!();
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

  pub fn _resume(&mut self, ctr: &CtxCtr, env: &mut CtxEnv, thunkenv: &mut CtxThunkEnv) -> SpineState {
    //self._start();
    println!("DEBUG: Spine::_resume: ctr={:?} ctlp={} hltp={} curp={}",
        self.ctr, self.ctlp, self.hltp, self.curp);
    self.hltp = self.curp;
    loop {
      let state = self._step(ctr, env, thunkenv);
      match state {
        SpineState::Bot => {
          return state;
        }
        _ => {}
      }
      self.ctlp += 1;
      match state {
        SpineState::Halt   |
        SpineState::Yield_ |
        SpineState::Break_ => {
          return state;
        }
        _ => {}
      }
    }
    unreachable!();
  }

  pub fn _step(&self, ctr: &CtxCtr, env: &mut CtxEnv, thunkenv: &mut CtxThunkEnv) -> SpineState {
    if self.ctlp >= self.hltp {
      println!("DEBUG: Spine::_step: ctr={:?} ctlp={} hltp={} curp={} halt",
          self.ctr, self.ctlp, self.hltp, self.curp);
      return SpineState::Halt;
    }
    let mut state = SpineState::_Top;
    let entry = &self.log[self.ctlp as usize];
    println!("DEBUG: Spine::_step: ctr={:?} ctlp={} hltp={} curp={} entry={:?}",
        self.ctr, self.ctlp, self.hltp, self.curp, entry.name());
    match entry {
      // TODO
      &SpineEntry::Yield_ => {
        state = SpineState::Yield_;
      }
      &SpineEntry::YieldV(_, _) => {
        unimplemented!();
      }
      &SpineEntry::Break_ => {
        state = SpineState::Break_;
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
      &SpineEntry::Opaque(_x, _og) => {
      }
      &SpineEntry::CacheAff(x) => {
        match env.lookup_ref(x) {
          None => panic!("bug"),
          Some(mut e) => {
            /*match e.cel.mode {
              CellMode::Top => {
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
          Some(mut e) => {
            /*match e.cel.mode {
              CellMode::Top => {
                e.cel.mode = CellMode::Mux;
              }
              CellMode::Mux => {}
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
              CellMode::Top => {
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
      &SpineEntry::IntroAff(x) => {
        match env.lookup_ref(x) {
          None => panic!("bug"),
          Some(mut e) => {
            match e.state().mode {
              CellMode::Top => {}
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
            e.state().flag.reset();
            e.state().flag.set_intro();
          }
        }
      }
      &SpineEntry::InitMux(ith, x) => {
        match env.lookup_ref(x) {
          None => panic!("bug"),
          Some(mut e) => {
            match e.state().mode {
              CellMode::Top => {
                e.state().mode = CellMode::Mux;
              }
              CellMode::Mux => {}
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
            e.state().flag.reset();
            e.state().flag.set_intro();
          }
        }
      }
      &SpineEntry::SealMux(x) => {
        match env.lookup_ref(x) {
          None => panic!("bug"),
          Some(mut e) => {
            match e.state().mode {
              CellMode::Top => {
                e.state().mode = CellMode::Mux;
              }
              CellMode::Mux => {}
              _ => panic!("bug")
            }
            assert_eq!(e.state().clk.ctr(), self.ctr);
            assert!(e.state().clk.tup > 0);
            assert!(e.state().clk.tup != u16::max_value());
            assert!(e.state().flag.intro());
            assert!(!e.state().flag.seal());
            e.state().flag.reset();
            e.state().flag.set_seal();
          }
        }
      }
      &SpineEntry::ApplyAff(th, x) => {
        let tup = match env.lookup_mut_ref(x) {
          None => panic!("bug"),
          Some(mut e) => {
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
            assert_eq!((tup as usize), e.thunk.len());
            e.thunk.push(th);
            tup
            /*let te = match thunkenv.thunktab.get(&e.thunk[tup as usize]) {
              None => panic!("bug"),
              Some(thunk) => thunk
            };
            e.cel.clk = e.cel.clk.update();
            te.thunk.apply(/*env,*/ &te.args, x, &e.ty, &mut e.cel);*/
          }
        };
        /*let mut cel = match env.lookup_ref(x) {
          None => panic!("bug"),
          Some(mut e) => {
            let te = match thunkenv.thunktab.get(&e.thunk[tup as usize]) {
              None => panic!("bug"),
              Some(thunk) => thunk
            };
            //e.cel.clk = e.cel.clk.update();
            let mut cel = e.ref_.swap_out();
            cel.clk = cel.clk.update();
            let ty = e.ty;
            te.thunk.apply(ctr, env, &te.args, x, &ty, &mut cel);
            cel
          }
        };
        match env.lookup_ref(x) {
          None => panic!("bug"),
          Some(mut e) => {
            //e.cel.flag.set_seal();
            cel.flag.set_seal();
            e.ref_.swap_in(cel);
          }
        }*/
        {
          let te = match thunkenv.thunktab.get(&th) {
            None => panic!("bug"),
            Some(thunk) => thunk
          };
          te.thunk.apply(ctr, env, &te.args, x);
        }
        match env.lookup_ref(x) {
          None => panic!("bug"),
          Some(mut e) => {
            e.state().flag.set_seal();
          }
        }
      }
      &SpineEntry::ApplyMux(th, x) => {
        match env.lookup_ref(x) {
          None => panic!("bug"),
          Some(mut e) => {
            match e.state().mode {
              CellMode::Top => {
                e.state().mode = CellMode::Mux;
              }
              CellMode::Mux => {}
              _ => panic!("bug")
            }
            assert_eq!(e.state().clk.ctr(), self.ctr);
            /*assert!(e.state().clk.tup >= 0);*/
            assert!(e.state().clk.tup != u16::max_value());
            assert!(e.state().flag.intro());
            assert!(!e.state().flag.seal());
            // FIXME
            let tup = e.state().clk.tup;
            assert!((tup as usize) < e.thunk.len());
            let te = match thunkenv.thunktab.get(&e.thunk[tup as usize]) {
              None => panic!("bug"),
              Some(thunk) => thunk
            };
            e.state().clk = e.state().clk.update();
            /*match &e.state().compute {
              &InnerCell::Uninit => panic!("bug"),
              &InnerCell::Primary => {
                e.state().primary.sync_thunk(x, &e.ty, &te.args, &te.thunk, e.state().clk);
              }
              _ => {
                e.state().compute.sync_thunk(x, &e.ty, &te.args, &te.thunk, e.state().clk);
              }
            }*/
            // FIXME FIXME: the above api is not very ergonomic.
            unimplemented!();
            //te.thunk.apply(/*env,*/ &te.args, x, &e.ty, &mut e.state());
            //e.state().flag.set_seal();
          }
        }
      }
      &SpineEntry::Eval(x) => {
        match env.lookup_ref(x) {
          None => panic!("bug"),
          Some(mut e) => {
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
          Some(mut e) => {
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
        state = SpineState::Bot;
      }
      _ => unimplemented!()
    }
    state
  }

  pub fn _debug_dump(&self) {
    println!("DEBUG: Spine::_debug_dump: ctr={:?} ctlp={} hltp={} curp={}",
        self.ctr, self.ctlp, self.hltp, self.curp);
    for (i, e) in self.log.iter().enumerate() {
      println!("DEBUG: Spine::_debug_dump: log[{}]={:?}", i, e);
    }
  }
}

#[track_caller]
pub fn reset() {
  panick_wrap(|| TL_CTX.with(|ctx| {
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
pub fn resume() {
  panick_wrap(|| TL_CTX.with(|ctx| {
    let mut env = ctx.env.borrow_mut();
    let mut thunkenv = ctx.thunkenv.borrow_mut();
    let mut spine = ctx.spine.borrow_mut();
    spine._resume(&ctx.ctr, &mut *env, &mut *thunkenv);
  }))
}

#[track_caller]
pub fn yield_() {
  panick_wrap(|| TL_CTX.with(|ctx| {
    let mut spine = ctx.spine.borrow_mut();
    spine.curp += 1;
    spine.log.push(SpineEntry::Yield_);
  }))
}

#[track_caller]
pub fn break_() {
  panick_wrap(|| TL_CTX.with(|ctx| {
    let mut spine = ctx.spine.borrow_mut();
    spine.curp += 1;
    spine.log.push(SpineEntry::Break_);
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
        &SpineEntry::IntroAff(x) => {
          self.env[0].intro.insert(x, self.ctlp);
          self.env[0].aff.insert(x);
        }
        &SpineEntry::InitMux(_th, x) => {
          // FIXME FIXME
          self.env[0].intro.insert(x, self.ctlp);
          self.env[0].semi.insert(x);
        }
        &SpineEntry::SealMux(x) => {
          assert!(self.env[0].semi.contains(&x));
          self.env[0].seal.insert(x, self.ctlp);
        }
        &SpineEntry::ApplyAff(th, x) => {
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
        &SpineEntry::ApplyMux(th, x) => {
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
