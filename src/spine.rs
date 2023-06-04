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
  Init(CellPtr, ThunkPtr),
  //IntroFin(CellPtr),
  Seal(CellPtr),
  Unseal(CellPtr),
  Apply(CellPtr, ThunkPtr),
  Accumulate(CellPtr, ThunkPtr),
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
      &SpineEntry::Init(..)       => SpineEntryName::Init,
      &SpineEntry::Seal(..)       => SpineEntryName::Seal,
      &SpineEntry::Unseal(..)     => SpineEntryName::Unseal,
      &SpineEntry::Apply(..)      => SpineEntryName::Apply,
      &SpineEntry::Accumulate(..) => SpineEntryName::Accumulate,
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
  Init,
  Seal,
  Unseal,
  Apply,
  Accumulate,
  Eval,
  //Uneval,
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

#[derive(Clone, Default)]
pub struct SpineEnv {
  pub aff:      HashMap<CellPtr, u32>,
  pub init:     HashMap<CellPtr, (u32, ThunkPtr)>,
  pub cache:    HashMap<CellPtr, u32>,
  pub intro:    HashMap<CellPtr, u32>,
  //pub fin:      HashSet<CellPtr>,
  pub seal:     HashMap<CellPtr, u32>,
  pub apply:    HashMap<CellPtr, Vec<(u32, ThunkPtr)>>,
  //pub eval:     HashMap<CellPtr, u32>,
}

impl SpineEnv {
  pub fn reset(&mut self) {
    self.aff.clear();
    self.init.clear();
    self.cache.clear();
    self.intro.clear();
    self.seal.clear();
    self.apply.clear();
    //self.eval.clear();
  }

  pub fn step(&mut self, sp: u32, e: &SpineEntry) {
    // FIXME FIXME
    match e {
      &SpineEntry::OIntro(x, _) => {
        // FIXME FIXME
        self.cache.insert(x, sp);
      }
      &SpineEntry::CacheAff(x) => {
        assert!(!self.intro.contains_key(&x));
        assert!(!self.seal.contains_key(&x));
        self.intro.insert(x, sp);
        self.seal.insert(x, sp);
        self.aff.insert(x, sp);
      }
      &SpineEntry::ICacheMux(x) => {
        assert!(!self.seal.contains_key(&x));
        self.intro.insert(x, sp);
        self.init.insert(x, (sp, ThunkPtr::nil()));
      }
      &SpineEntry::Intro(x) => {
        assert!(!self.intro.contains_key(&x));
        assert!(!self.seal.contains_key(&x));
        self.intro.insert(x, sp);
        self.aff.insert(x, sp);
      }
      &SpineEntry::Init(x, ith) => {
        // FIXME FIXME
        self.intro.insert(x, sp);
        self.init.insert(x, (sp, ith));
        // FIXME FIXME
      }
      &SpineEntry::Seal(x) => {
        // TODO: idempotent seal?
        /*assert!(!self.seal.contains_key(&x));*/
        assert!(self.init.contains_key(&x));
        self.seal.insert(x, sp);
      }
      &SpineEntry::Unseal(x) => {
        // TODO: idempotent seal?
        /*assert!(self.seal.contains_key(&x));*/
        assert!(self.init.contains_key(&x));
        self.seal.remove(&x);
      }
      &SpineEntry::Apply(x, th) => {
        assert!(self.aff.contains_key(&x));
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
        self.seal.insert(x, sp);
      }
      // TODO TODO
      _ => unimplemented!()
    }
  }

  pub fn unstep(&mut self, e: &SpineEntry) {
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
    self.env.reset();
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

  pub fn init_mux(&mut self, ith: ThunkPtr, x: CellPtr) {
    // FIXME FIXME
    unimplemented!();
    /*
    let sp = self.curp;
    self.curp += 1;
    let e = SpineEntry::Init(x, ith);
    self.env.step(sp, &e);
    self.log.push(e);
    */
  }

  pub fn seal_mux(&mut self, x: CellPtr) {
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

  pub fn apply_aff(&mut self, th: ThunkPtr, x: CellPtr) {
    let sp = self.curp;
    self.curp += 1;
    let e = SpineEntry::Apply(x, th);
    self.env.step(sp, &e);
    self.log.push(e);
  }

  pub fn apply_mux(&mut self, th: ThunkPtr, x: CellPtr) {
    assert!(self.env.init.contains_key(&x));
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
            e.state().flag.reset();
            e.state().flag.set_intro();
          }
        }
      }
      &SpineEntry::Init(x, ith) => {
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
            e.state().flag.reset();
            e.state().flag.set_intro();
          }
        }
      }
      &SpineEntry::Seal(x) => {
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
      &SpineEntry::Apply(x, th) => {
        let tup = match env.lookup_mut_ref(x) {
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
            assert_eq!((tup as usize), e.thunk.len());
            e.thunk.push(th);
            tup
          }
        };
        {
          let te = match thunkenv.thunktab.get(&th) {
            None => panic!("bug"),
            Some(thunk) => thunk
          };
          te.thunk.apply(ctr, env, &te.args, x);
        }
        match env.lookup_ref(x) {
          None => panic!("bug"),
          Some(e) => {
            e.state().flag.set_seal();
          }
        }
      }
      &SpineEntry::Accumulate(x, th) => {
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
        &SpineEntry::Init(y, ith) => {
          assert!(!self.complete_set.contains(&y));
          if !self.frontier_set.contains(&y) {
            continue;
          }
          unimplemented!();
        }
        &SpineEntry::Seal(..) => {
          unimplemented!();
        }
        &SpineEntry::Unseal(..) => {
          unimplemented!();
        }
        &SpineEntry::Apply(y, th) => {
          drop(e);
          drop(spine);
          assert!(!self.complete_set.contains(&y));
          if !self.frontier_set.contains(&y) {
            continue;
          }
          let dy = ctx_lookup_or_insert_gradr(tg, y);
          //let dy = ctx.lookup_gradr_or_nil(tg, y);
          let thunkenv = ctx.thunkenv.borrow();
          match thunkenv.thunktab.get(&th) {
            None => panic!("bug"),
            Some(te) => {
              let arg = te.args.clone();
              let spec_ = te.thunk.spec_.clone();
              drop(te);
              drop(thunkenv);
              let mut arg_adj = Vec::with_capacity(arg.len());
              for &x in arg.iter() {
                assert!(!self.complete_set.contains(&x));
                self.frontier_set.insert(x);
                let ty_ = ctx_lookup_type(x);
                arg_adj.push(ctx_init_zeros(&ty_));
              }
              match spec_.pop_adj(&arg, dy, &arg_adj) {
                Err(_) => unimplemented!(),
                Ok(_) => {}
              }
            }
          }
        }
        &SpineEntry::Accumulate(y, th) => {
          drop(e);
          drop(spine);
          assert!(!self.complete_set.contains(&y));
          if !self.frontier_set.contains(&y) {
            continue;
          }
          //let dy = ctx.lookup_gradr_or_nil(tg, y);
          unimplemented!();
        }
        _ => {}
      }
    }
    let mut env = ctx.env.borrow_mut();
    match env.bwd.get_mut(&tg) {
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
            drop(spine);
            let tg_clk = match ctx.env.borrow().lookup_ref(tg) {
              None => panic!("bug"),
              Some(e) => e.state().clk
            };
            match ctx.env.borrow().bwd.get(&tg) {
              None => {}
              Some(&bwd_clk) => {
                match bwd_clk.partial_cmp(tg_clk) {
                  None => panic!("bug"),
                  Some(Ordering::Greater) => panic!("bug"),
                  Some(Ordering::Less) => {}
                  Some(Ordering::Equal) => {
                    return;
                  }
                }
              }
            }
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
              ctx_set_ones(&tg_ty_)
            };
            let thunkenv = ctx.thunkenv.borrow();
            match thunkenv.thunktab.get(&th) {
              None => panic!("bug"),
              Some(te) => {
                let arg = te.args.clone();
                let spec_ = te.thunk.spec_.clone();
                drop(te);
                drop(thunkenv);
                let mut frontier_set = HashSet::new();
                let mut arg_adj = Vec::with_capacity(arg.len());
                for &x in arg.iter() {
                  frontier_set.insert(x);
                  let ty_ = ctx_lookup_type(x);
                  arg_adj.push(ctx_init_zeros(&ty_));
                }
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
        &SpineEntry::Init(x, _ith) => {
          // FIXME FIXME
          self.env[0].intro.insert(x, self.ctlp);
          self.env[0].semi.insert(x);
        }
        &SpineEntry::Seal(x) => {
          assert!(self.env[0].semi.contains(&x));
          self.env[0].seal.insert(x, self.ctlp);
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
