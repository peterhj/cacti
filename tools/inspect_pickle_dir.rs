extern crate cacti;

use cacti::algo::str::{safe_ascii};
use cacti::util::pickle::{PickleDir};

use std::env;
use std::path::{PathBuf};

fn main() {
  let argv: Vec<_> = env::args().collect();
  if argv.len() <= 1 {
    println!("usage: inspect_pickle_dir <path-to-pickle-dir> [model-prefix]");
    return;
  }
  let dir = PathBuf::from(&argv[1]);
  let pickdir = if argv.len() > 2 {
    PickleDir::open_with_prefix(dir, &argv[2]).unwrap()
  } else {
    PickleDir::open(dir).unwrap()
  };
  for (i, k) in pickdir.serial_key.iter().enumerate() {
    let (ty, _) = pickdir.get(k);
    println!("INFO:   tensor i={} dtype={:?} shape={:?} name=\"{}\"",
        i,
        ty.dtype(),
        ty.shape(),
        safe_ascii(k.as_bytes()),
    );
  }
}
