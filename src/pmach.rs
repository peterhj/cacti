use std::cell::{Cell};

pub struct TlsContext {
  default_primary:  Cell<PMachSpec>,
  default_compute:  Cell<PMachSpec>,
  gpu_reserve:      u16,
  gpu_workspace:    u16,
}

impl Default for TlsContext {
  fn default() -> TlsContext {
    TlsContext{
      default_primary:  Cell::new(PMachSpec::CpuDefault),
      default_compute:  Cell::new(PMachSpec::CpuDefault),
      gpu_reserve:      9001,
      gpu_workspace:    111,
    }
  }
}

thread_local! {
  static TL_CTX: TlsContext = TlsContext::default();
}

pub fn tl_get_default_primary() -> PMachSpec {
  TL_CTX.with(|ctx| ctx.default_primary.get())
}

pub fn tl_set_default_primary(spec: PMachSpec) {
  TL_CTX.with(|ctx| ctx.default_primary.set(spec))
}

pub fn tl_get_default_compute() -> PMachSpec {
  TL_CTX.with(|ctx| ctx.default_compute.get())
}

pub fn tl_set_default_compute(spec: PMachSpec) {
  TL_CTX.with(|ctx| ctx.default_compute.set(spec))
}

#[derive(Clone, Copy)]
pub struct PMachFlag {
  bits: u8,
}

impl Default for PMachFlag {
  fn default() -> PMachFlag {
    PMachFlag{bits: 0}
  }
}

impl PMachFlag {
  pub fn set_storage(self) -> PMachFlag {
    PMachFlag{bits: self.bits | 1}
  }

  pub fn set_compute(self) -> PMachFlag {
    PMachFlag{bits: self.bits | 2}
  }

  pub fn storage(self) -> bool {
    (self.bits & 1) != 0
  }

  pub fn compute(self) -> bool {
    (self.bits & 2) != 0
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PMachSpec {
  _Top,
  CpuDefault,
  //Cpu(CpuSet),
  GpuDefault,
  //Gpu(GpuSet),
  //SwapDefault,
  //Swap(SwapSet),
}

impl PMachSpec {
  pub fn flag(&self) -> PMachFlag {
    match self {
      &PMachSpec::CpuDefault => {
        PMachFlag::default().set_storage().set_compute()
      }
      &PMachSpec::GpuDefault => {
        PMachFlag::default().set_storage().set_compute()
      }
      _ => panic!("bug: unimplemented")
    }
  }
}

pub enum PCellState {
  Unalloc,
  Uninit,
  Init(u16),
}

pub enum PCell {
  Primary,
  Cpu(CpuPCell),
  Gpu(GpuPCell),
  //Swap(SwapPCell),
}

impl PCell {
  pub fn fresh(spec: PMachSpec) -> PCell {
    match spec {
      PMachSpec::CpuDefault => {
        PCell::Cpu(CpuPCell{
          state:  PCellState::Unalloc,
        })
      }
      PMachSpec::GpuDefault => {
        PCell::Gpu(GpuPCell{
          state:  PCellState::Unalloc,
        })
      }
      _ => unimplemented!()
    }
  }

  pub fn primary_or_fresh(primary: PMachSpec, spec: PMachSpec) -> PCell {
    if primary == spec {
      PCell::Primary
    } else {
      PCell::fresh(spec)
    }
  }
}

pub struct CpuPCell {
  state:    PCellState,
  // FIXME FIXME: just use malloc'd buffer.
}

pub struct GpuPCell {
  state:    PCellState,
  // FIXME FIXME: need offsets into a simple GPU bump allocator.
  //ptr:      u64,
  //sz:       u64,
  //pitch:    Vec<u64>,
  //shape:    Vec<usize>,
  //dtype:    ,
}
