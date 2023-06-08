use self::nvgpu::{NvGpuPCtx};
use self::smp::{SmpPCtx};

pub mod nvgpu;
pub mod smp;
pub mod swap;

/*#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct PMachFlag {
  bits: u8,
}

impl Default for PMachFlag {
  fn default() -> PMachFlag {
    PMachFlag{bits: 0}
  }
}

impl PMachFlag {
  pub fn set_primary(self) -> PMachFlag {
    PMachFlag{bits: self.bits | 1}
  }

  pub fn set_compute(self) -> PMachFlag {
    PMachFlag{bits: self.bits | 2}
  }

  pub fn primary(self) -> bool {
    (self.bits & 1) != 0
  }

  pub fn compute(self) -> bool {
    (self.bits & 2) != 0
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum PMachSpec {
  _Top = 0,
  //Cpu = 1,
  //Cpu(CpuSet),
  Smp = 1,
  //Smp(CpuSet),
  Swap = 2,
  //Swap(SwapSet),
  #[cfg(feature = "gpu")]
  Gpu = 3,
  //Gpu(GpuSet),
}

impl PMachSpec {
  pub fn flag(&self) -> PMachFlag {
    match self {
      &PMachSpec::Smp => {
        PMachFlag::default().set_primary().set_compute()
      }
      &PMachSpec::Swap => {
        PMachFlag::default().set_primary()
      }
      #[cfg(feature = "gpu")]
      &PMachSpec::Gpu => {
        PMachFlag::default().set_compute()
      }
      _ => panic!("bug: unimplemented")
    }
  }
}*/

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

thread_local! {
  pub static TL_PCTX: PCtx = PCtx::new();
}

pub struct PCtx {
  // TODO TODO
  pub lpmatrix: Vec<(Locus, PMach)>,
  pub plmatrix: Vec<(PMach, Locus)>,
  pub smp:      SmpPCtx,
  #[cfg(feature = "gpu")]
  pub nvgpu:    Option<NvGpuPCtx>,
}

impl PCtx {
  pub fn new() -> PCtx {
    println!("DEBUG: PCtx::new");
    let mut pctx = PCtx{
      lpmatrix: Vec::new(),
      plmatrix: Vec::new(),
      smp:      SmpPCtx::new(),
      #[cfg(feature = "gpu")]
      nvgpu:    NvGpuPCtx::new(0),
    };
    pctx.smp.append_matrix(&mut pctx.lpmatrix, &mut pctx.plmatrix);
    #[cfg(feature = "gpu")]
    if let Some(gpu) = pctx.nvgpu.as_ref() {
      gpu.append_matrix(&mut pctx.lpmatrix, &mut pctx.plmatrix);
    }
    pctx.lpmatrix.sort_by(|lx, rx| rx.cmp(lx));
    pctx.plmatrix.sort_by(|lx, rx| rx.cmp(lx));
    pctx
  }

  pub fn fastest_locus(&self) -> Locus {
    #[cfg(feature = "gpu")]
    if let Some(locus) = self.nvgpu.as_ref().map(|gpu| gpu.fastest_locus()) {
      return locus;
    }
    Locus::Mem
  }
}
