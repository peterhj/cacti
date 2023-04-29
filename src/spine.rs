use crate::cell::*;
use crate::clock::{Counter};
use crate::ctx::{CtxEnv};
use crate::ptr::*;
use crate::thunk::*;

use std::collections::{HashMap, HashSet};
//use std::mem::{swap};

#[derive(Clone, Copy)]
//#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SpineEntry {
  _Top,
  IntroFin(CellPtr),
  IntroAff(CellPtr),
  IntroSemi(Option<ThunkPtr>, CellPtr),
  SealSemi(CellPtr),
  ApplyAff(ThunkPtr, CellPtr),
  ApplySemi(ThunkPtr, CellPtr),
  Eval(CellPtr),
  Unsync(CellPtr),
  Alias(CellPtr, CellPtr),
  //Cache(CellPtr),
  // TODO
  Bot,
}

#[derive(Clone, Copy)]
#[repr(u8)]
pub enum SpineStatus {
  Top,
  Halt,
  EarlyHalt,
  Bot,
}

pub struct Spine {
  ctr:  Counter,
  ctlp: u32,
  hltp: u32,
  curp: u32,
  log:  Vec<SpineEntry>,
}

impl Default for Spine {
  fn default() -> Spine {
    Spine{
      ctr:  Counter::default(),
      ctlp: 0,
      hltp: 0,
      curp: 0,
      log:  Vec::new(),
    }
  }
}

impl Spine {
  pub fn _reset(&mut self) {
    self.ctr = self.ctr.advance();
    self.ctlp = 0;
    self.hltp = 0;
    self.curp = 0;
    self.log.clear();
  }

  pub fn _reduce(&mut self, ) {
    let mut dry = DrySpine::new(self);
    //dry._interp();
    //dry._reorder();
    //dry._fuse();
    //dry._reduce();
    unimplemented!();
  }

  /*pub fn _start(&mut self) {
    self.hltp = self.curp;
  }*/

  pub fn _resume(&mut self, env: &mut CtxEnv) -> SpineStatus {
    let mut status = SpineStatus::Top;
    //self._start();
    while self.ctlp < self.hltp {
      status = self._step(env);
      match status {
        SpineStatus::Bot => {
          return status;
        }
        _ => {}
      }
      self.ctlp += 1;
      match status {
        SpineStatus::Halt => {
          return SpineStatus::EarlyHalt;
        }
        _ => {}
      }
    }
    SpineStatus::Halt
  }

  pub fn _step(&mut self, env: &mut CtxEnv) -> SpineStatus {
    if self.ctlp >= self.hltp {
      return SpineStatus::Halt;
    }
    let mut status = SpineStatus::Top;
    let entry = &self.log[self.ctlp as usize];
    match entry {
      &SpineEntry::IntroFin(x) => {
        match env.celtab.get_mut(&x) {
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
                assert!(e.cel.primary.synced(&e.cel.ty, e.cel.clk));
              }
              _ => {
                e.cel.compute.sync_cell(&e.cel.ty, &e.cel.primary, e.cel.clk);
              }
            }
            // FIXME: first, wait for primary sync.
            //e.cel._;
            e.cel.flag.reset();
            e.cel.flag.set_intro();
          }
        }
      }
      &SpineEntry::IntroAff(x) => {
        match env.celtab.get_mut(&x) {
          None => panic!("bug"),
          Some(e) => {
            match e.cel.mode {
              CellMode::_Top => {
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
      &SpineEntry::IntroSemi(ith, x) => {
        match env.celtab.get_mut(&x) {
          None => panic!("bug"),
          Some(e) => {
            match e.cel.mode {
              CellMode::_Top => {
                e.cel.mode = CellMode::Semi;
              }
              CellMode::Semi => {}
              _ => panic!("bug")
            }
            if !e.cel.clk.happens_before(self.ctr).unwrap() {
              panic!("bug");
            }
            assert!(!e.cel.flag.intro());
            assert!(!e.cel.flag.seal());
            // FIXME
            e.cel.clk = self.ctr.into();
            match (env.cache.contains(&x), e.ithunk) {
              (true, None) => {}
              (false, Some(thunk)) => {
                let thunk = match env.thunktab.get(&thunk) {
                  None => panic!("bug"),
                  Some(thunk) => thunk
                };
                match &e.cel.compute {
                  &InnerCell::Uninit => panic!("bug"),
                  &InnerCell::Primary => {
                    e.cel.primary.sync_thunk(&e.cel.ty, thunk, e.cel.clk);
                  }
                  _ => {
                    e.cel.compute.sync_thunk(&e.cel.ty, thunk, e.cel.clk);
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
      &SpineEntry::SealSemi(x) => {
        match env.celtab.get_mut(&x) {
          None => panic!("bug"),
          Some(e) => {
            match e.cel.mode {
              CellMode::_Top => {
                e.cel.mode = CellMode::Semi;
              }
              CellMode::Semi => {}
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
        match env.celtab.get_mut(&x) {
          None => panic!("bug"),
          Some(e) => {
            match e.cel.mode {
              CellMode::_Top => {
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
            assert!((tup as usize) < e.thunk.len());
            let thunk = match env.thunktab.get(&e.thunk[tup as usize]) {
              None => panic!("bug"),
              Some(thunk) => thunk
            };
            e.cel.clk = e.cel.clk.update();
            match &e.cel.compute {
              &InnerCell::Uninit => panic!("bug"),
              &InnerCell::Primary => {
                e.cel.primary.sync_thunk(&e.cel.ty, thunk, e.cel.clk);
              }
              _ => {
                e.cel.compute.sync_thunk(&e.cel.ty, thunk, e.cel.clk);
              }
            }
            e.cel.flag.set_seal();
          }
        }
      }
      &SpineEntry::ApplySemi(th, x) => {
        match env.celtab.get_mut(&x) {
          None => panic!("bug"),
          Some(e) => {
            match e.cel.mode {
              CellMode::_Top => {
                e.cel.mode = CellMode::Semi;
              }
              CellMode::Semi => {}
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
            let thunk = match env.thunktab.get(&e.thunk[tup as usize]) {
              None => panic!("bug"),
              Some(thunk) => thunk
            };
            e.cel.clk = e.cel.clk.update();
            match &e.cel.compute {
              &InnerCell::Uninit => panic!("bug"),
              &InnerCell::Primary => {
                e.cel.primary.sync_thunk(&e.cel.ty, thunk, e.cel.clk);
              }
              _ => {
                e.cel.compute.sync_thunk(&e.cel.ty, thunk, e.cel.clk);
              }
            }
            e.cel.flag.set_seal();
          }
        }
      }
      &SpineEntry::Eval(x) => {
        match env.celtab.get_mut(&x) {
          None => panic!("bug"),
          Some(e) => {
            assert_eq!(e.cel.clk.ctr(), self.ctr);
            assert!(e.cel.flag.intro());
            assert!(e.cel.flag.seal());
            match &e.cel.compute {
              &InnerCell::Uninit => panic!("bug"),
              &InnerCell::Primary => {
                assert!(e.cel.primary.synced(&e.cel.ty, e.cel.clk));
              }
              _ => {
                match &e.cel.primary {
                  &InnerCell::Uninit => {}
                  &InnerCell::Primary => panic!("bug"),
                  _ => {
                    e.cel.primary.sync_cell(&e.cel.ty, &e.cel.compute, e.cel.clk);
                  }
                }
              }
            }
          }
        }
      }
      &SpineEntry::Unsync(x) => {
        match env.celtab.get_mut(&x) {
          None => panic!("bug"),
          Some(e) => {
            assert_eq!(e.cel.clk.ctr(), self.ctr);
            assert!(e.cel.flag.intro());
            assert!(e.cel.flag.seal());
            match &e.cel.compute {
              &InnerCell::Uninit => panic!("bug"),
              &InnerCell::Primary => {
                assert!(e.cel.primary.synced(&e.cel.ty, e.cel.clk));
              }
              _ => {
                match &e.cel.primary {
                  &InnerCell::Uninit => {}
                  &InnerCell::Primary => panic!("bug"),
                  _ => {
                    e.cel.primary.sync_cell(&e.cel.ty, &e.cel.compute, e.cel.clk);
                  }
                }
                e.cel.compute.unsync(e.cel.clk);
              }
            }
          }
        }
      }
      &SpineEntry::Alias(x, y) => {
        unimplemented!();
      }
      _ => unimplemented!()
    }
    status
  }
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
        &SpineEntry::IntroFin(x) => {
          self.env[0].intro.insert(x, self.ctlp);
          self.env[0].fin.insert(x);
        }
        &SpineEntry::IntroAff(x) => {
          self.env[0].intro.insert(x, self.ctlp);
          self.env[0].aff.insert(x);
        }
        &SpineEntry::IntroSemi(_th, x) => {
          // FIXME FIXME
          self.env[0].intro.insert(x, self.ctlp);
          self.env[0].semi.insert(x);
        }
        &SpineEntry::SealSemi(x) => {
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
        &SpineEntry::ApplySemi(th, x) => {
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
