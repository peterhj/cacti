use self::nvgpu::{NvGpuPCtx};
use self::smp::{SmpPCtx};
use crate::algo::{RevSortMap8};
use crate::cell::{CellPtr, CellType, InnerCell, InnerCell_};

use std::cell::{Cell, RefCell};
use std::cmp::{max};
use std::fmt::{Debug, Formatter, Result as FmtResult};
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

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct PAddr {
  pub bits: i64,
}

impl Debug for PAddr {
  fn fmt(&self, f: &mut Formatter) -> FmtResult {
    write!(f, "PAddr({})", self.bits)
  }
}

impl PAddr {
  pub fn from_unchecked(bits: i64) -> PAddr {
    PAddr{bits}
  }

  pub fn to_unchecked(&self) -> i64 {
    self.bits
  }
}

/*#[derive(Clone, Copy)]
pub struct PAddrTabEntry {
  pub loc:  Locus,
  pub pm:   PMach,
  pub pin:  bool,
}*/

thread_local! {
  pub static TL_PCTX: PCtx = PCtx::new();
}

#[derive(Default)]
pub struct PCtxCtr {
  pub addr: Cell<i64>,
}

impl PCtxCtr {
  pub fn fresh_addr(&self) -> PAddr {
    let next = self.addr.get() + 1;
    assert!(next > 0);
    self.addr.set(next);
    PAddr{bits: next}
  }

  pub fn next_addr(&self) -> PAddr {
    let next = self.addr.get() + 1;
    assert!(next > 0);
    PAddr{bits: next}
  }

  pub fn peek_addr(&self) -> PAddr {
    let cur = self.addr.get();
    PAddr{bits: cur}
  }
}

pub struct PCtx {
  // TODO TODO
  pub ctr:      PCtxCtr,
  pub pmset:    PMachSet,
  pub lpmatrix: RevSortMap8<(Locus, PMach), ()>,
  pub plmatrix: RevSortMap8<(PMach, Locus), ()>,
  //pub addrtab:  RefCell<HashMap<PAddr, PAddrTabEntry>>,
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
      ctr:      PCtxCtr::default(),
      pmset:    PMachSet::default(),
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

  /*//pub fn try_alloc(&self, x: CellPtr, sz: usize, locus: Locus) -> Result<Option<Weak<dyn InnerCell_>>, PMemErr> {}
  pub fn try_alloc(&self, x: CellPtr, sz: usize, locus: Locus) -> Result<Option<Rc<dyn InnerCell_>>, PMemErr> {
    match self.lpmatrix.find_lub((locus, PMach::_Top)) {
      None => Ok(None),
      Some((key, _)) => {
        let pmach = key.key.as_ref().1;
        let ret = match pmach {
          PMach::Smp => {
            self.smp.try_alloc(x, sz, locus)
              //.map(|r| Some(Rc::downgrade(&r) as _))
              .map(|r| Some(r as _))
          }
          PMach::NvGpu => {
            self.nvgpu.as_ref().unwrap().try_alloc(x, sz, locus)
              //.map(|r| Some(Rc::downgrade(&r) as _))
              .map(|r| Some(r as _))
          }
          _ => panic!("bug")
        };
        ret
      }
    }
  }*/

  pub fn alloc_loc(&self, locus: Locus, ty: &CellType) -> (PMach, PAddr) {
    match self.lpmatrix.find_lub((locus, PMach::_Bot)) {
      None => {
        panic!("bug: PCtx::alloc_loc: failed to alloc: locus={:?}", locus);
      }
      Some((key, _)) => {
        let pmach = key.key.as_ref().1;
        match pmach {
          #[cfg(not(feature = "gpu"))]
          PMach::NvGpu => {
            unimplemented!();
          }
          #[cfg(feature = "gpu")]
          PMach::NvGpu => {
            let addr = match self.nvgpu.as_ref().unwrap().try_alloc(&self.ctr, locus, ty) {
              Err(e) => panic!("bug: PCtx::alloc_loc: unimplemented error: {:?}", e),
              Ok(addr) => addr
            };
            (PMach::NvGpu, addr)
          }
          _ => {
            println!("DEBUG: {:?}", &self.lpmatrix);
            println!("DEBUG: {:?}", &self.plmatrix);
            panic!("bug: PCtx::alloc_loc: unimplemented: locus={:?} pmach={:?}", locus, pmach);
          }
        }
      }
    }
  }

  pub fn lookup_pm(&self, pmach: PMach, addr: PAddr) -> Option<(Locus, Rc<dyn InnerCell_>)> {
    match pmach {
      #[cfg(not(feature = "gpu"))]
      PMach::NvGpu => {
        unimplemented!();
      }
      #[cfg(feature = "gpu")]
      PMach::NvGpu => {
        self.nvgpu.as_ref().unwrap().lookup2(addr)
      }
      _ => {
        unimplemented!();
      }
    }
  }

  pub fn lookup(&self, addr: PAddr) -> Option<(Locus, Rc<dyn InnerCell_>)> {
    // FIXME FIXME
    unimplemented!();
  }
}

pub trait PCtxImpl {
  // FIXME: don't need this associated type.
  //type ICel: InnerCell;

  fn pmach(&self) -> PMach;
  fn fastest_locus(&self) -> Locus;
  fn append_matrix(&self, lp: &mut RevSortMap8<(Locus, PMach), ()>, pl: &mut RevSortMap8<(PMach, Locus), ()>);
  //fn try_alloc(&self, x: CellPtr, sz: usize, pmset: PMachSet) -> Result<Rc<Self::ICel>, PMemErr>;
  //fn try_alloc(&self, x: CellPtr, sz: usize, locus: Locus) -> Result<Rc<dyn InnerCell_>, PMemErr>;
  fn try_alloc(&self, pctr: &PCtxCtr, locus: Locus, ty: &CellType) -> Result<PAddr, PMemErr>;
}
