extern crate home;
extern crate once_cell;

use home::{home_dir};
use once_cell::sync::{Lazy};

use std::env::{var};
use std::path::{PathBuf};

pub static CFG_ENV: Lazy<CfgEnv> = Lazy::new(|| CfgEnv::get());
thread_local! {
  pub static TL_CFG_ENV: CfgEnv = CFG_ENV.clone();
}

#[derive(Clone)]
pub struct CfgEnv {
  pub cabalpath:  Vec<PathBuf>,
  pub cudaprefix: Vec<PathBuf>,
  pub virtualenv: bool,
  pub debug:      bool,
}

impl CfgEnv {
  pub fn get() -> CfgEnv {
    let cabalpath = var("CACTI_CABAL_BIN_PATH").map(|s| {
      let mut ps = Vec::new();
      for s in s.split(":") {
        if !s.is_empty() {
          ps.push(PathBuf::from(s));
        }
      }
      ps
    }).unwrap_or_else(|_| {
      home_dir().map(|p| vec![p.join(".cabal").join("bin")])
        .unwrap_or_else(|| Vec::new())
    });
    let cudaprefix = var("CACTI_CUDA_PREFIX").map(|s| {
      let mut ps = Vec::new();
      for s in s.split(":") {
        if s == "@cuda" {
          // FIXME FIXME: os-specific paths.
          ps.push(PathBuf::from("/usr/local/cuda"));
        } else if !s.is_empty() {
          ps.push(PathBuf::from(s));
        }
      }
      ps
    }).unwrap_or_else(|_| var("CUDA_HOME").map(|s| {
      let mut ps = Vec::new();
      if !s.is_empty() {
        ps.push(PathBuf::from(s));
      }
      ps
    }).unwrap_or_else(|_| var("CUDA_ROOT").map(|s| {
      let mut ps = Vec::new();
      if !s.is_empty() {
        ps.push(PathBuf::from(s));
      }
      ps
    }).unwrap_or_else(|_| var("CUDA_PATH").map(|s| {
      let mut ps = Vec::new();
      if !s.is_empty() {
        ps.push(PathBuf::from(s));
      }
      ps
    }).unwrap_or_else(|_| vec![PathBuf::from("/usr/local/cuda")])
    )));
    let virtualenv = var("VIRTUAL_ENV")
      .map(|_| true)
      .unwrap_or_else(|_| false);
    let debug = var("CACTI_DEBUG")
      .map(|_| true)
      .unwrap_or_else(|_| false);
    CfgEnv{
      cabalpath,
      cudaprefix,
      virtualenv,
      debug,
    }
  }
}
