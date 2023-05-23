use crate::cell::*;
use crate::clock::{Counter};
use crate::ctx::{TL_CTX, CtxEnv, CtxThunkEnv};
use crate::panick::{panick_wrap};
use crate::thunk::*;

use std::collections::{HashMap, HashSet};
//use std::mem::{swap};

#[derive(Clone, Copy, Debug)]
//#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SpineEntry {
  _Top,
  Yield_,
  Break_,
  YieldV(CellPtr, CellPtr),
  BreakV(CellPtr, CellPtr),
  TraceV(CellPtr, CellPtr),
  Profile(CellPtr, CellPtr),
  Opaque(CellPtr, CellPtr),
  CacheAff(CellPtr),
  CacheMux(CellPtr),
  //IntroFin(CellPtr),
  IntroAff(CellPtr),
  InitMux(ThunkPtr, CellPtr),
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
  pub log:  Vec<SpineEntry>,
  pub env:  SpineEnv,
}

impl Default for Spine {
  fn default() -> Spine {
    Spine{
      ctr:  Counter::default(),
      ctlp: 0,
      hltp: 0,
      curp: 0,
      log:  Vec::new(),
      env:  SpineEnv::default(),
    }
  }
}

impl Spine {
  pub fn reset(&mut self) {
    self.ctr = self.ctr.advance();
    self.ctlp = 0;
    self.hltp = 0;
    self.curp = 0;
    self.log.clear();
    self.env.reset();
  }

  pub fn compile(&mut self, ) {
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

  pub fn resume(&mut self, /*ctr: &CtxCtr,*/ env: &mut CtxEnv, thunkenv: &mut CtxThunkEnv) -> SpineState {
    //self._start();
    println!("DEBUG: Spine::resume: ctr={:?} ctlp={} hltp={} curp={}",
        self.ctr, self.ctlp, self.hltp, self.curp);
    self.hltp = self.curp;
    loop {
      let state = self._step(/*ctr,*/ env, thunkenv);
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

  pub fn _step(&self, /*ctr: &CtxCtr,*/ env: &mut CtxEnv, thunkenv: &mut CtxThunkEnv) -> SpineState {
    if self.ctlp >= self.hltp {
      return SpineState::Halt;
    }
    let mut state = SpineState::_Top;
    let entry = &self.log[self.ctlp as usize];
    match entry {
      // TODO
      &SpineEntry::Yield_ => {
        state = SpineState::Yield_;
      }
      &SpineEntry::Break_ => {
        state = SpineState::Break_;
      }
      &SpineEntry::YieldV(_, _) => {
        unimplemented!();
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
      &SpineEntry::Opaque(x, og) => {
        unimplemented!();
      }
      &SpineEntry::CacheAff(x) => {
        match env.lookup_mut(x) {
          None => panic!("bug"),
          Some(e) => {
            /*match e.cel.mode {
              CellMode::Top => {
                e.cel.mode = CellMode::Aff;
              }
              CellMode::Aff => {}
              _ => panic!("bug")
            }*/
            // FIXME FIXME
            //unimplemented!();
            let flag = !e.cel.flag.set_cache();
            let mode = match e.cel.mode.set_aff() {
              Err(_) => panic!("bug"),
              Ok(prev) => !prev
            };
            assert!(flag);
            assert!(mode);
            assert!(self.ctr.succeeds(e.cel.clk));
          }
        }
      }
      &SpineEntry::CacheMux(x) => {
        match env.lookup_mut(x) {
          None => panic!("bug"),
          Some(e) => {
            /*match e.cel.mode {
              CellMode::Top => {
                e.cel.mode = CellMode::Mux;
              }
              CellMode::Mux => {}
              _ => panic!("bug")
            }*/
            // FIXME FIXME
            //unimplemented!();
            let flag = !e.cel.flag.set_cache();
            let mode = match e.cel.mode.set_mux() {
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
        //match env.celtab.get_mut(&x) {}
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
        //match env.celtab.get_mut(&x) {}
        match env.lookup_mut(x) {
          None => panic!("bug"),
          Some(e) => {
            match e.cel.mode {
              CellMode::Top => {
                e.cel.mode = CellMode::Aff;
              }
              CellMode::Aff => {}
              _ => panic!("bug")
            }
            if !e.cel.clk.happens_before(self.ctr).unwrap() {
              panic!("bug");
            }
            assert!(!e.cel.flag.intro());
            assert!(!e.cel.flag.seal());
            // FIXME
            e.cel.clk = self.ctr.into();
            e.cel.flag.reset();
            e.cel.flag.set_intro();
          }
        }
      }
      &SpineEntry::InitMux(ith, x) => {
        //match env.celtab.get_mut(&x) {}
        match env.lookup_mut(x) {
          None => panic!("bug"),
          Some(e) => {
            match e.cel.mode {
              CellMode::Top => {
                e.cel.mode = CellMode::Mux;
              }
              CellMode::Mux => {}
              _ => panic!("bug")
            }
            if !e.cel.clk.happens_before(self.ctr).unwrap() {
              panic!("bug");
            }
            assert!(!e.cel.flag.intro());
            assert!(!e.cel.flag.seal());
            // FIXME
            e.cel.clk = self.ctr.into();
            // FIXME FIXME: set ithunk.
            //match (env.cache.contains(&x), e.ithunk) {}
            match (e.cel.flag.cache(), e.ithunk) {
              (true, None) => {}
              (false, Some(thunk)) => {
                let thunk = match thunkenv.thunktab.get(&thunk) {
                  None => panic!("bug"),
                  Some(thunk) => thunk
                };
                match &e.cel.compute {
                  &InnerCell::Uninit => panic!("bug"),
                  &InnerCell::Primary => {
                    e.cel.primary.sync_thunk(x, &e.ty, thunk, e.cel.clk);
                  }
                  _ => {
                    e.cel.compute.sync_thunk(x, &e.ty, thunk, e.cel.clk);
                  }
                }
              }
              _ => panic!("bug")
            }
            e.cel.flag.reset();
            e.cel.flag.set_intro();
          }
        }
      }
      &SpineEntry::SealMux(x) => {
        //match env.celtab.get_mut(&x) {}
        match env.lookup_mut(x) {
          None => panic!("bug"),
          Some(e) => {
            match e.cel.mode {
              CellMode::Top => {
                e.cel.mode = CellMode::Mux;
              }
              CellMode::Mux => {}
              _ => panic!("bug")
            }
            assert_eq!(e.cel.clk.ctr(), self.ctr);
            assert!(e.cel.clk.tup > 0);
            assert!(e.cel.clk.tup != u16::max_value());
            assert!(e.cel.flag.intro());
            assert!(!e.cel.flag.seal());
            e.cel.flag.reset();
            e.cel.flag.set_seal();
          }
        }
      }
      &SpineEntry::ApplyAff(th, x) => {
        //match env.celtab.get_mut(&x) {}
        match env.lookup_mut(x) {
          None => panic!("bug"),
          Some(e) => {
            match e.cel.mode {
              CellMode::Top => {
                e.cel.mode = CellMode::Aff;
              }
              CellMode::Aff => {}
              _ => panic!("bug")
            }
            assert_eq!(e.cel.clk.ctr(), self.ctr);
            assert_eq!(e.cel.clk.tup, 0);
            assert!(e.cel.flag.intro());
            assert!(!e.cel.flag.seal());
            // FIXME
            let tup = e.cel.clk.tup;
            if (tup as usize) != e.thunk.len() {
              println!("DEBUG: Spine::_step: tup={} e.thunk.len={}", tup, e.thunk.len());
              self._debug_dump();
              panic!("bug");
            }
            assert_eq!((tup as usize), e.thunk.len());
            e.thunk.push(th);
            let thunk = match thunkenv.thunktab.get(&e.thunk[tup as usize]) {
              None => panic!("bug"),
              Some(thunk) => thunk
            };
            e.cel.clk = e.cel.clk.update();
            match &e.cel.compute {
              &InnerCell::Uninit => panic!("bug"),
              &InnerCell::Primary => {
                e.cel.primary.sync_thunk(x, &e.ty, thunk, e.cel.clk);
              }
              _ => {
                e.cel.compute.sync_thunk(x, &e.ty, thunk, e.cel.clk);
              }
            }
            e.cel.flag.set_seal();
          }
        }
      }
      &SpineEntry::ApplyMux(th, x) => {
        //match env.celtab.get_mut(&x) {}
        match env.lookup_mut(x) {
          None => panic!("bug"),
          Some(e) => {
            match e.cel.mode {
              CellMode::Top => {
                e.cel.mode = CellMode::Mux;
              }
              CellMode::Mux => {}
              _ => panic!("bug")
            }
            assert_eq!(e.cel.clk.ctr(), self.ctr);
            /*assert!(e.cel.clk.tup >= 0);*/
            assert!(e.cel.clk.tup != u16::max_value());
            assert!(e.cel.flag.intro());
            assert!(!e.cel.flag.seal());
            // FIXME
            let tup = e.cel.clk.tup;
            assert!((tup as usize) < e.thunk.len());
            let thunk = match thunkenv.thunktab.get(&e.thunk[tup as usize]) {
              None => panic!("bug"),
              Some(thunk) => thunk
            };
            e.cel.clk = e.cel.clk.update();
            match &e.cel.compute {
              &InnerCell::Uninit => panic!("bug"),
              &InnerCell::Primary => {
                e.cel.primary.sync_thunk(x, &e.ty, thunk, e.cel.clk);
              }
              _ => {
                e.cel.compute.sync_thunk(x, &e.ty, thunk, e.cel.clk);
              }
            }
            e.cel.flag.set_seal();
          }
        }
      }
      &SpineEntry::Eval(x) => {
        //match env.celtab.get_mut(&x) {}
        match env.lookup_mut(x) {
          None => panic!("bug"),
          Some(e) => {
            assert_eq!(e.cel.clk.ctr(), self.ctr);
            assert!(e.cel.flag.intro());
            assert!(e.cel.flag.seal());
            match &e.cel.compute {
              &InnerCell::Uninit => panic!("bug"),
              &InnerCell::Primary => {
                assert!(e.cel.primary.synced(&e.ty, e.cel.clk));
              }
              _ => {
                match &e.cel.primary {
                  &InnerCell::Uninit => {}
                  &InnerCell::Primary => panic!("bug"),
                  _ => {
                    e.cel.primary.sync_cell(x, &e.ty, &e.cel.compute, e.cel.clk);
                  }
                }
              }
            }
          }
        }
      }
      &SpineEntry::Unsync(x) => {
        //match env.celtab.get_mut(&x) {}
        match env.lookup_mut(x) {
          None => panic!("bug"),
          Some(e) => {
            assert_eq!(e.cel.clk.ctr(), self.ctr);
            assert!(e.cel.flag.intro());
            assert!(e.cel.flag.seal());
            match &e.cel.compute {
              &InnerCell::Uninit => panic!("bug"),
              &InnerCell::Primary => {
                assert!(e.cel.primary.synced(&e.ty, e.cel.clk));
              }
              _ => {
                match &e.cel.primary {
                  &InnerCell::Uninit => {}
                  &InnerCell::Primary => panic!("bug"),
                  _ => {
                    e.cel.primary.sync_cell(x, &e.ty, &e.cel.compute, e.cel.clk);
                  }
                }
                e.cel.compute.unsync(e.cel.clk);
              }
            }
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
    ctx.spine.borrow_mut().reset();
  }))
}

#[track_caller]
pub fn compile() {
  panick_wrap(|| TL_CTX.with(|ctx| {
    ctx.spine.borrow_mut().compile();
  }))
}

#[track_caller]
pub fn resume() {
  panick_wrap(|| TL_CTX.with(|ctx| {
    let mut env = ctx.env.borrow_mut();
    let mut thunkenv = ctx.thunkenv.borrow_mut();
    let mut spine = ctx.spine.borrow_mut();
    spine.resume(/*&ctx.ctr,*/ &mut *env, &mut *thunkenv);
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
        &SpineEntry::CacheMux(x) => {
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
