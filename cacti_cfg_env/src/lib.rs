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

#[derive(Clone, Copy, Debug)]
pub enum OomPolicy {
  Soft,
  Hard,
}

#[derive(Clone)]
pub struct CfgEnv {
  pub cabalpath:  Vec<PathBuf>,
  pub cudaprefix: Vec<PathBuf>,
  pub mem_soft_limit: Option<()>,
  pub mem_oom:    OomPolicy,
  pub vmem_soft_limit: Option<()>,
  pub vmem_oom:   OomPolicy,
  pub no_kcache:  bool,
  pub futhark_pedantic: bool,
  pub futhark_trace: bool,
  pub verbose:    bool,
  //pub verbose:    i8,
  pub silent:     bool,
  pub report:     bool,
  pub debug_yeet: i8,
  pub debug_mem_pool: i8,
  pub debug_initialize: i8,
  pub debug_accumulate: i8,
  pub debug_apply: i8,
  pub debug:      i8,
  pub devel_dump: bool,
  pub rust_backtrace: bool,
  //pub cuda_visible_devices: (),
  //pub virtualenv: bool,
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
    let mem_soft_limit = var("CACTI_MEM_SOFT_LIMIT").map(|s| {
      // FIXME
      ()
    }).ok();
    let mem_oom = var("CACTI_MEM_OOM").ok().and_then(|s| {
      match s.as_str() {
        "soft" => Some(OomPolicy::Soft),
        "hard" => Some(OomPolicy::Hard),
        _ => None
      }
    }).unwrap_or_else(|| OomPolicy::Soft);
    let vmem_soft_limit = var("CACTI_VMEM_SOFT_LIMIT").map(|s| {
      // FIXME
      ()
    }).ok();
    let vmem_oom = var("CACTI_VMEM_OOM").ok().and_then(|s| {
      match s.as_str() {
        "soft" => Some(OomPolicy::Soft),
        "hard" => Some(OomPolicy::Hard),
        _ => None
      }
    }).unwrap_or_else(|| OomPolicy::Soft);
    let no_kcache = var("CACTI_NO_KCACHE")
      .map(|_| true)
      .unwrap_or_else(|_| false);
    let futhark_pedantic = var("CACTI_FUTHARK_PEDANTIC")
      .map(|_| true)
      .unwrap_or_else(|_| false);
    let futhark_trace = var("CACTI_FUTHARK_TRACE")
      .map(|_| true)
      .unwrap_or_else(|_| false);
    let verbose = var("CACTI_VERBOSE")
      .map(|_| true)
      .unwrap_or_else(|_| false);
    let silent = var("CACTI_SILENT")
      .map(|_| true)
      .unwrap_or_else(|_| false);
    let report = var("CACTI_REPORT")
      .map(|_| true)
      .unwrap_or_else(|_| false);
    let debug_yeet = var("CACTI_DEBUG_YEET")
      .map(|s| match s.parse() {
        Ok(d) => d,
        Err(_) => 1
      })
      .unwrap_or_else(|_| 0);
    let debug_mem_pool = var("CACTI_DEBUG_MEM_POOL")
      .map(|s| match s.parse() {
        Ok(d) => d,
        Err(_) => 1
      })
      .unwrap_or_else(|_| 0);
    let debug_initialize = var("CACTI_DEBUG_INITIALIZE")
      .map(|s| match s.parse() {
        Ok(d) => d,
        Err(_) => 1
      })
      .unwrap_or_else(|_| 0);
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
    let rust_backtrace = var("RUST_BACKTRACE").ok()
      .map(|s| match s.as_str() {
        "0" => false,
        _ => true
      })
      .unwrap_or_else(|| false);
    /*let virtualenv = var("VIRTUAL_ENV")
      .map(|_| true)
      .unwrap_or_else(|_| false);*/
    if !silent && debug >= 0 {
      for (i, p) in cabalpath.iter().enumerate() {
        println!("INFO:   cacti_cfg_env: CACTI_CABAL_PATH[{}]={}",
            i, p.to_str().map(|s| _safe_ascii(s.as_bytes())).unwrap());
      }
      for (i, p) in cudaprefix.iter().enumerate() {
        println!("INFO:   cacti_cfg_env: CACTI_CUDA_PREFIX[{}]={}",
            i, p.to_str().map(|s| _safe_ascii(s.as_bytes())).unwrap());
      }
      // FIXME: format.
      //println!("INFO:   cacti_cfg_env: CACTI_VMEM_LIMIT={}", _);
    }
    CfgEnv{
      cabalpath,
      cudaprefix,
      mem_soft_limit,
      mem_oom,
      vmem_soft_limit,
      vmem_oom,
      no_kcache,
      futhark_pedantic,
      futhark_trace,
      verbose,
      silent,
      report,
      debug_yeet,
      debug_mem_pool,
      debug_initialize,
      debug_accumulate,
      debug_apply,
      debug,
      devel_dump,
      rust_backtrace,
      //virtualenv,
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
    !cfg.silent
  })
}

pub fn cfg_verbose_info() -> bool {
  TL_CFG_ENV.with(|cfg| {
    !cfg.silent && cfg.verbose
  })
}

pub fn cfg_report() -> bool {
  TL_CFG_ENV.with(|cfg| {
    !cfg.silent && cfg.report
  })
}

pub fn cfg_verbose_report() -> bool {
  TL_CFG_ENV.with(|cfg| {
    !cfg.silent && cfg.report && cfg.verbose
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
