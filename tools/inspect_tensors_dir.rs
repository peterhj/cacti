extern crate cacti;

use cacti::algo::str::{safe_ascii};
use cacti::util::safetensor::{TensorsDir};

use std::env;
use std::path::{PathBuf};

fn main() {
  let argv: Vec<_> = env::args().collect();
  if argv.len() <= 1 {
    println!("usage: inspect_tensors_dir <path-to-safetensors-dir>");
    return;
  }
  let dir = PathBuf::from(&argv[1]);
  let tensdir = TensorsDir::open(dir).unwrap();
  for (i, k) in tensdir.serial_key.iter().enumerate() {
    let (ty, _) = tensdir.get(k);
    println!("INFO:   tensor i={} dtype={:?} shape={:?} name=\"{}\"",
        i,
        ty.dtype(),
        ty.shape(),
        safe_ascii(k.as_bytes()),
    );
  }
}
