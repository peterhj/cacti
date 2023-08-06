extern crate home;
extern crate once_cell;

use home::{home_dir};
use once_cell::sync::{Lazy};

use std::env::{var};
use std::path::{PathBuf};

pub static CFG_ENV: Lazy<CfgEnv> = Lazy::new(|| CfgEnv::get_once());
thread_local! {
  pub static TL_CFG_ENV: CfgEnv = CFG_ENV.clone();
}

#[derive(Clone)]
pub struct CfgEnv {
  pub cabalpath:  Vec<PathBuf>,
  pub cudaprefix: Vec<PathBuf>,
  pub virtualenv: bool,
  pub no_kcache:  bool,
  pub futhark_pedantic: bool,
  pub futhark_trace: bool,
  pub silent:     bool,
  pub debug_accumulate: i8,
  pub debug_apply: i8,
  pub debug:      i8,
  pub devel_dump: bool,
}

impl CfgEnv {
  pub fn get_once() -> CfgEnv {
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
    let no_kcache = var("CACTI_NO_KCACHE")
      .map(|_| true)
      .unwrap_or_else(|_| false);
    let futhark_pedantic = var("CACTI_FUTHARK_PEDANTIC")
      .map(|_| true)
      .unwrap_or_else(|_| false);
    let futhark_trace = var("CACTI_FUTHARK_TRACE")
      .map(|_| true)
      .unwrap_or_else(|_| false);
    let silent = var("CACTI_SILENT")
      .map(|_| true)
      .unwrap_or_else(|_| false);
    let debug_accumulate = var("CACTI_DEBUG_ACCUMULATE")
      .map(|s| match s.parse() {
        Ok(d) => d,
        Err(_) => 1
      })
      .unwrap_or_else(|_| 0);
    let debug_apply = var("CACTI_DEBUG_APPLY")
      .map(|s| match s.parse() {
        Ok(d) => d,
        Err(_) => 1
      })
      .unwrap_or_else(|_| 0);
    let debug = var("CACTI_DEBUG")
      .map(|s| match s.parse() {
        Ok(d) => d,
        Err(_) => 1
      })
      .unwrap_or_else(|_| 0);
    let devel_dump = var("CACTI_DEVEL_DUMP")
      .map(|_| true)
      .unwrap_or_else(|_| false);
    if !silent && debug >= 0 {
      for p in cabalpath.iter() {
        println!("INFO:  cacti_cfg_env: CACTI_CABAL_PATH={}", p.to_str().map(|s| _safe_ascii(s.as_bytes())).unwrap());
      }
      for p in cudaprefix.iter() {
        println!("INFO:  cacti_cfg_env: CACTI_CUDA_PREFIX={}", p.to_str().map(|s| _safe_ascii(s.as_bytes())).unwrap());
      }
    }
    CfgEnv{
      cabalpath,
      cudaprefix,
      virtualenv,
      no_kcache,
      futhark_pedantic,
      futhark_trace,
      silent,
      debug_accumulate,
      debug_apply,
      debug,
      devel_dump,
    }
  }
}

pub fn cfg_devel_dump() -> bool {
  TL_CFG_ENV.with(|cfg| {
    !cfg.silent && cfg.devel_dump
  })
}

pub fn cfg_info() -> bool {
  TL_CFG_ENV.with(|cfg| {
    !cfg.silent && cfg.debug >= 0
  })
}

pub fn cfg_debug() -> bool {
  TL_CFG_ENV.with(|cfg| {
    !cfg.silent && cfg.debug >= 1
  })
}

pub fn cfg_trace() -> bool {
  TL_CFG_ENV.with(|cfg| {
    !cfg.silent && cfg.debug >= 3
  })
}

pub fn cfg_debug_(level: i8) -> bool {
  TL_CFG_ENV.with(|cfg| {
    !cfg.silent && cfg.debug >= level
  })
}

pub fn _safe_ascii(s: &[u8]) -> String {
  let mut buf = Vec::new();
  for &u in s.iter() {
    if u >= b'0' && u <= b'9' {
      buf.push(u);
    } else if u >= b'A' && u <= b'Z' {
      buf.push(u);
    } else if u >= b'a' && u <= b'z' {
      buf.push(u);
    } else if u <= 0x20 {
      buf.push(b' ');
    } else {
      match u {
        b' ' |
        b'.' |
        b',' |
        b':' |
        b';' |
        b'/' |
        b'\\' |
        b'|' |
        b'-' |
        b'_' |
        b'<' |
        b'>' |
        b'[' |
        b']' |
        b'{' |
        b'}' |
        b'(' |
        b')' => {
          buf.push(u);
        }
        _ => {
          buf.push(b'?');
        }
      }
    }
  }
  String::from_utf8_lossy(&buf).into()
}
