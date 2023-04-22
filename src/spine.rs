use crate::pmach::*;

use std::collections::{HashMap, HashSet};
use std::mem::{swap};

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

pub type SpineCell = PCell;

/*pub struct SpineCell {
  //primary:  PCell,
  //compute:  PCell,
  pcel: PCell_,
}

impl SpineCell {
  pub fn fresh(ptr: StablePtr, dtype: Dtype, /*shape: _*/) -> SpineCell {
    /*let primary = PCell::fresh(tl_get_default_primary());
    let compute = PCell::primary_or_fresh(tl_get_default_primary(), tl_get_default_compute());
    SpineCell{
      primary,
      compute,
    }*/
    SpineCell{
      pcel: PCell_::fresh(ptr, dtype),
    }
  }
}*/

pub struct SpineEnv {
  //ctr:      u32,
  cel:      HashMap<StablePtr, SpineCell>,
  tag:      HashMap<StablePtr, HashSet<String>>,
  grad:     HashMap<[StablePtr; 2], StablePtr>,
  ungrad:   HashMap<[StablePtr; 2], StablePtr>,
  retain:   HashSet<StablePtr>,
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
  // FIXME FIXME
  /*fn _fresh_ptr(&mut self) -> StablePtr {
    let next = self.env.ctr + 1;
    assert!(next >= 0);
    assert!(next < u32::max_value());
    self.env.ctr = next;
    let x = StablePtr(next);
    x
  }*/

  pub fn fresh(&mut self, dtype: Dtype) -> StablePtr {
    let x = tl_fresh_ptr();
    let cel = SpineCell::fresh(x, dtype);
    self.env.cel.insert(x, cel);
    x
  }

  pub fn add_tag<S: AsRef<str>>(&mut self, x: StablePtr, tag: S) {
    if !self.env.cel.contains_key(&x) {
      panic!("bug");
    }
    match self.env.tag.get_mut(&x) {
      None => {
        let mut tag_set = HashSet::default();
        tag_set.insert(tag.as_ref().to_owned());
        self.env.tag.insert(x, tag_set);
      }
      Some(tag_set) => {
        tag_set.insert(tag.as_ref().to_owned());
      }
    }
  }

  pub fn remove_tag<S: AsRef<str>>(&mut self, x: StablePtr, tag: S) {
    if !self.env.cel.contains_key(&x) {
      panic!("bug");
    }
    match self.env.tag.get_mut(&x) {
      None => {}
      Some(tag_set) => {
        tag_set.remove(tag.as_ref());
      }
    }
  }

  pub fn grad(&mut self, y: StablePtr, x: StablePtr) -> StablePtr {
    if !self.env.cel.contains_key(&x) {
      panic!("bug");
    }
    match self.env.grad.get(&[y, x]) {
      None => {
        let dtype = match self.env.cel.get(&x) {
          None => panic!("bug"),
          Some(cel) => {
            cel.dtype
          }
        };
        let dx = self.fresh(dtype);
        self.env.grad.insert([y, x], dx);
        self.env.ungrad.insert([y, dx], x);
        dx
      }
      Some(&dx) => dx
    }
  }

  pub fn ungrad(&mut self, y: StablePtr, dx: StablePtr) -> StablePtr {
    if !self.env.cel.contains_key(&dx) {
      panic!("bug");
    }
    match self.env.ungrad.get(&[y, dx]) {
      None => panic!("bug"),
      Some(&x) => x
    }
  }

  pub fn retain_all(&mut self, xs: &[StablePtr]) {
    for &x in xs.iter() {
      self.retain(x);
    }
  }

  pub fn retain(&mut self, x: StablePtr) {
    if !self.env.cel.contains_key(&x) {
      panic!("bug");
    }
    self.env.retain.insert(x);
  }

  pub fn unretain(&mut self, x: StablePtr) {
    if !self.env.cel.contains_key(&x) {
      panic!("bug");
    }
    self.env.retain.remove(&x);
  }

  /*
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
  */

  pub fn backward(&mut self) {
    self.bwd.push(self.curp);
    unimplemented!();
  }

  /*pub fn wait(&mut self) {
    unimplemented!();
    //self.ackp = self.curp;
  }*/

  pub fn reduce(&mut self) {
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
