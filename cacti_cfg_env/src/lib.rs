use once_cell::sync::{Lazy};

use std::env;
use std::path::{PathBuf};

pub static CFG_ENV: Lazy<CfgEnv> = Lazy::new(|| CfgEnv::get());

pub struct CfgEnv {
  pub cuda_home:  Vec<PathSpec>,
  pub virtualenv: bool,
}

impl CfgEnv {
  pub fn get() -> CfgEnv {
    let cuda_home = env::var("CACTI_CUDA_HOME").map(|s| {
      let mut ps = Vec::new();
      for s in s.split(":") {
        if s == "@cuda_home" {
          ps.push(PathSpec::BuiltinCudaHome);
        } else if !s.is_empty() {
          ps.push(PathSpec::Path(PathBuf::from(s)));
        }
      }
      ps
    }).unwrap_or_else(|_| env::var("CUDA_HOME").map(|s| {
      let mut ps = Vec::new();
      for s in s.split(":") {
        if !s.is_empty() {
          ps.push(PathSpec::Path(PathBuf::from(s)));
        }
      }
      ps
    }).unwrap_or_else(|_| vec![PathSpec::BuiltinCudaHome])
    );
    let virtualenv = env::var("VIRTUAL_ENV")
      .map(|_| true)
      .unwrap_or_else(|_| false);
    CfgEnv{
      cuda_home,
      virtualenv,
    }
  }
}

pub enum PathSpec {
  Path(PathBuf),
  BuiltinCudaHome,
}

impl PathSpec {
  pub fn to_path(&self) -> PathBuf {
    match self {
      &PathSpec::Path(ref p) => p.clone(),
      // FIXME FIXME: os-specific paths.
      &PathSpec::BuiltinCudaHome => PathBuf::from("/usr/local/cuda"),
    }
  }
}
