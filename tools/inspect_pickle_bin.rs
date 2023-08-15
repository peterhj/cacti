extern crate cacti;

use cacti::algo::str::{safe_ascii};
use cacti::cell::Dtype;
use cacti::util::pickle::{PickleFile};

use std::convert::{TryFrom};
use std::env;
use std::path::{PathBuf};

fn main() {
  let argv: Vec<_> = env::args().collect();
  if argv.len() <= 1 {
    println!("usage: inspect_pth <path-to-pickle-bin-file>");
    return;
  }
  let p = PathBuf::from(&argv[1]);
  let ckpt = PickleFile::open(p).unwrap();
  for (i, t) in ckpt.tensors().iter().enumerate() {
    println!("INFO:   tensor i={} dtype={:?} shape={:?} stride={:?} name=\"{}\"",
        i,
        t.tensor_type.clone()
          .and_then(|t| Dtype::try_from(t))
          .map_err(|s| safe_ascii(s.as_str().as_bytes())),
        &t.shape,
        &t.stride,
        safe_ascii(t.name.as_bytes()),
    );
  }
}
