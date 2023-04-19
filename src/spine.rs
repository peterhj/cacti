use crate::pmach::*;

use std::collections::{HashMap};
use std::mem::{swap};

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct StablePtr(u32);

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct StableRef {
  ptr:  StablePtr,
  rst:  u16,
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct DataPtr(usize);

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct CodePtr(usize);

#[derive(Clone, Copy)]
//#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct SpinePtr(u32);

impl SpinePtr {
  pub fn idx(self) -> usize {
    self.0 as _
  }
}

pub trait SpineThunk: Copy {
  fn _apply(&self, args: &[SpinePtr], spine: &Spine);
  fn _backward(&mut self, log: &mut Vec<SpineEntry>);
}

#[derive(Clone, Copy)]
//#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SpineEntry {
  _Top,
  Read(SpinePtr),
  Write(SpinePtr),
  Accumulate(SpinePtr),
  SyncGpu(SpinePtr),
  UnsyncGpu(SpinePtr),
  Apply(CodePtr),
  //Apply(Box<Any + SpineThunk>),
  Constant(DataPtr),
  Variable(DataPtr),
  Bot,
}

pub struct SpineCell {
  primary:  PCell,
  compute:  PCell,
}

impl SpineCell {
  pub fn fresh() -> SpineCell {
    let primary = PCell::fresh(tl_get_default_primary());
    let compute = PCell::primary_or_fresh(tl_get_default_primary(), tl_get_default_compute());
    SpineCell{
      primary,
      compute,
    }
  }
}

pub struct SpineEnv {
  cel:      HashMap<StablePtr, SpineCell>,
}

pub struct Spine {
  //ackp: SpinePtr,
  bp:   SpinePtr,
  ctlp: SpinePtr,
  hltp: SpinePtr,
  curp: SpinePtr,
  //bar:  Vec<SpinePtr>,
  bwd:  Vec<SpinePtr>,
  rst:  Vec<SpinePtr>,
  log:  Vec<SpineEntry>,
  env:  SpineEnv,
}

impl Spine {
  pub fn constant(&mut self, x: DataPtr) {
    self.log.push(SpineEntry::Constant(x));
    self.curp = SpinePtr(self.log.len() as _);
  }

  pub fn variable(&mut self, x: DataPtr) {
    self.log.push(SpineEntry::Variable(x));
    self.curp = SpinePtr(self.log.len() as _);
  }

  // FIXME FIXME: should probably not directly accept SpinePtr.
  pub fn push_arg(&mut self, p: SpinePtr) {
    self.log.push(SpineEntry::Read(p));
  }

  // FIXME FIXME: should probably not directly return SpinePtr.
  pub fn apply<T: SpineThunk>(&mut self, thunk: T) -> SpinePtr {
    //self.log.push(SpineEntry::Apply(_));
    unimplemented!();
    self.curp = SpinePtr(self.log.len() as _);
  }

  pub fn backward(&mut self) {
    self.bwd.push(self.curp);
    unimplemented!();
  }

  /*pub fn wait(&mut self) {
    unimplemented!();
    //self.ackp = self.curp;
  }*/

  pub fn minimize(&mut self) {
    if self.hltp.idx() >= self.curp.idx() {
      println!("WARNING: Spine::minimize: nothing to minimize (hltp={} curp={})",
          self.hltp.idx(), self.curp.idx());
      return;
    }
    unimplemented!();
  }

  pub fn resume(&mut self) {
    // NB: bump the halt ptr to the cursor.
    let mut swap_hltp = self.curp;
    swap(&mut swap_hltp, &mut self.hltp);
    // FIXME FIXME: hint to start any stalled work.
    unimplemented!();
    /*if swap_hltp < self.hltp {
    }*/
    /*while self.ctlp < self.hltp {
    }*/
  }

  pub fn reset(&mut self) {
    self.rst.push(self.curp);
    if self.rst.len() >= 2 {
      let init_rst = if self.rst.len() >= 3 {
        self.rst[self.rst.len() - 3]
      } else {
        SpinePtr(0)
      };
      let prev_rst = self.rst[self.rst.len() - 2];
      let last_rst = self.rst[self.rst.len() - 1];
      // FIXME: tail compression.
      // FIXME FIXME: SpinePtr should not be directly compared;
      // need to convert to relative addr.
      unimplemented!();
      /*if &self.log[init_rst.idx() .. prev_rst.idx()] ==
         &self.log[prev_rst.idx() .. last_rst.idx()]
      {
        // FIXME: self.bp.
        // FIXME: self.bwd.
        self.curp = prev_rst;
        self.rst.pop();
        self.log.resize(prev_rst.idx(), SpineEntry::_Top);
      }*/
    }
    unimplemented!();
  }
}

pub struct SpineIter {
}

impl SpineIter {
}

impl Iterator for SpineIter {
  type Item = ();

  fn next(&mut self) -> Option<()> {
    None
  }
}
