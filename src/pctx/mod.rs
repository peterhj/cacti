use self::nvgpu::{NvGpuPCtx, *};
use self::smp::{SmpPCtx};
use crate::algo::{RevSortMap8};
use crate::cell::{CellPtr, InnerCell, InnerCell_};

use std::cmp::{max};
use std::rc::{Rc, Weak};

pub mod nvgpu;
pub mod smp;
pub mod swap;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[repr(u8)]
pub enum Locus {
  // TODO TODO
  _Top = 0,
  Swap = 31,
  Mem  = 63,
  VMem = 127,
  _Bot = 255,
}

impl Locus {
  pub fn fastest() -> Locus {
    TL_PCTX.with(|pctx| pctx.fastest_locus())
  }

  pub fn max(self, rhs: Locus) -> Locus {
    max(self, rhs)
  }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[repr(u8)]
pub enum PMach {
  // TODO TODO
  _Top = 0,
  Smp,
  NvGpu,
  _Bot = 255,
}

#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct PMachSet {
  bits: u8,
}

impl Default for PMachSet {
  fn default() -> PMachSet {
    PMachSet{bits: 0}
  }
}

impl PMachSet {
  pub fn insert(&mut self, pm: PMach) {
    match pm {
      PMach::_Top => {}
      PMach::Smp => {
        self.bits |= 1;
      }
      PMach::NvGpu => {
        self.bits |= 2;
      }
      PMach::_Bot => panic!("bug")
    }
  }

  pub fn contains(&self, pm: PMach) -> bool {
    match pm {
      PMach::_Top => true,
      PMach::Smp => {
        (self.bits & 1) != 0
      }
      PMach::NvGpu => {
        (self.bits & 2) != 0
      }
      PMach::_Bot => panic!("bug")
    }
  }
}

#[derive(Clone, Copy, Debug)]
#[repr(u8)]
pub enum PMemErr {
  Oom,
  Bot,
}

thread_local! {
  pub static TL_PCTX: PCtx = PCtx::new();
}

pub struct PCtx {
  // TODO TODO
  pub pmset:    PMachSet,
  //pub lpmatrix: Vec<(Locus, PMach)>,
  //pub plmatrix: Vec<(PMach, Locus)>,
  pub lpmatrix: RevSortMap8<(Locus, PMach), ()>,
  pub plmatrix: RevSortMap8<(PMach, Locus), ()>,
  pub smp:      SmpPCtx,
  #[cfg(feature = "gpu")]
  pub nvgpu_ct: i32,
  #[cfg(feature = "gpu")]
  pub nvgpu:    Option<NvGpuPCtx>,
  /*pub nvgpu:    Vec<NvGpuPCtx>,*/
}

impl PCtx {
  pub fn new() -> PCtx {
    println!("DEBUG: PCtx::new");
    let mut pctx = PCtx{
      pmset:    PMachSet::default(),
      //lpmatrix: Vec::new(),
      //plmatrix: Vec::new(),
      lpmatrix: RevSortMap8::default(),
      plmatrix: RevSortMap8::default(),
      smp:      SmpPCtx::new(),
      #[cfg(feature = "gpu")]
      nvgpu_ct: 0,
      #[cfg(feature = "gpu")]
      nvgpu:    None,
    };
    #[cfg(feature = "gpu")]
    {
      let gpu_ct = NvGpuPCtx::dev_count();
      pctx.nvgpu_ct = gpu_ct;
      for dev in 0 .. gpu_ct {
        pctx.nvgpu = NvGpuPCtx::new(dev);
        break;
      }
    }
    pctx.smp.append_matrix(&mut pctx.lpmatrix, &mut pctx.plmatrix);
    pctx.pmset.insert(PMach::Smp);
    #[cfg(feature = "gpu")]
    if let Some(gpu) = pctx.nvgpu.as_ref() {
      gpu.append_matrix(&mut pctx.lpmatrix, &mut pctx.plmatrix);
      pctx.pmset.insert(PMach::NvGpu);
    }
    //pctx.lpmatrix.sort_by(|lx, rx| rx.cmp(lx));
    //pctx.plmatrix.sort_by(|lx, rx| rx.cmp(lx));
    pctx
  }

  pub fn fastest_locus(&self) -> Locus {
    let mut max_locus = Locus::_Top;
    max_locus = max_locus.max(self.smp.fastest_locus());
    if let Some(locus) = self.nvgpu.as_ref().map(|gpu| gpu.fastest_locus()) {
      max_locus = max_locus.max(locus);
    }
    max_locus
  }

  //pub fn try_alloc(&self, x: CellPtr, sz: usize, locus: Locus) -> Result<Option<Weak<dyn InnerCell_>>, PMemErr> {}
  pub fn try_alloc(&self, x: CellPtr, sz: usize, locus: Locus) -> Result<Option<Rc<dyn InnerCell_>>, PMemErr> {
    match self.lpmatrix.find_lub((locus, PMach::_Top)) {
      None => Ok(None),
      Some((key, _)) => {
        let pmach = key.key.as_ref().1;
        //let ret: Result<Option<Weak<dyn InnerCell_>>, _> =
        let ret = match pmach {
          PMach::Smp => {
            self.smp.try_alloc(x, sz, self.pmset)
              //.map(|r| Some(Rc::downgrade(&r) as _))
              .map(|r| Some(r as _))
          }
          PMach::NvGpu => {
            self.nvgpu.as_ref().unwrap().try_alloc(x, sz, self.pmset)
              //.map(|r| Some(Rc::downgrade(&r) as _))
              .map(|r| Some(r as _))
          }
          _ => panic!("bug")
        };
        ret
      }
    }
  }
}

pub trait PCtxImpl {
  // FIXME: don't need this associated type.
  type ICel: InnerCell;

  fn pmach(&self) -> PMach;
  fn fastest_locus(&self) -> Locus;
  fn append_matrix(&self, lp: &mut RevSortMap8<(Locus, PMach), ()>, pl: &mut RevSortMap8<(PMach, Locus), ()>);
  fn try_alloc(&self, x: CellPtr, sz: usize, pmset: PMachSet) -> Result<Rc<Self::ICel>, PMemErr>;
  //fn try_alloc(&self, x: CellPtr, sz: usize, locus: Locus) -> Result<Rc<dyn InnerCell_>, PMemErr>;
}
