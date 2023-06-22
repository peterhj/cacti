extern crate cacti;

use cacti::cell::Dtype;
use cacti::util::pickle::{PickleFile};

use std::convert::{TryFrom};
use std::env;
use std::path::{PathBuf};

fn sane_ascii(s: &str) -> String {
  let mut buf = Vec::new();
  for &u in s.as_bytes().iter() {
    if u == b' ' || u == b'.' || u == b':' || u == b'/' || u == b'-' || u == b'_' {
      buf.push(u);
    } else if u >= b'0' && u <= b'9' {
      buf.push(u);
    } else if u >= b'A' && u <= b'Z' {
      buf.push(u);
    } else if u >= b'a' && u <= b'z' {
      buf.push(u);
    } else {
      buf.push(b'?');
    }
  }
  String::from_utf8_lossy(&buf).to_string()
}

fn main() {
  let argv: Vec<_> = env::args().collect();
  if argv.len() <= 1 {
    return;
  }
  let p = PathBuf::from(&argv[1]);
  let ckpt = PickleFile::open(p).unwrap();
  for (i, t) in ckpt.tensors().iter().enumerate() {
    println!("DEBUG: tensor i={} dtype={:?} shape={:?} stride={:?} name='{}'",
        i,
        t.tensor_type.clone()
          .and_then(|t| Dtype::try_from(t))
          .map_err(|s| sane_ascii(s.as_str())),
        &t.shape,
        &t.stride,
        sane_ascii(&t.name),
    );
  }
}
