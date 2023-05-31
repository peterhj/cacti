use self::nvgpu::{NvGpuPCtx};

pub mod nvgpu;
//pub mod smp;

thread_local! {
  pub static TL_PCTX: PCtx = PCtx::new();
}

pub struct PCtx {
  //pub smp:      SmpPCtx,
  #[cfg(feature = "gpu")]
  pub nvgpu:    NvGpuPCtx,
}

impl PCtx {
  pub fn new() -> PCtx {
    println!("DEBUG: PCtx::new");
    PCtx{
      #[cfg(feature = "gpu")]
      nvgpu:    NvGpuPCtx::new(0),
    }
  }
}
