extern crate cacti;
//extern crate repugnant_pickle;

use cacti::cell::Dtype;
//use repugnant_pickle::{RepugnantTorchFile, TensorType};
use cacti::util::torch::{TorchFile, TorchDtype};

//use std::borrow::{Cow};
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

fn dtype_from_tensor_type(t: &TorchDtype) -> Dtype {
  match t {
    &TorchDtype::Float64 => Dtype::Float64,
    &TorchDtype::Float32 => Dtype::Float32,
    &TorchDtype::Float16 => Dtype::Float16,
    &TorchDtype::BFloat16 => Dtype::BFloat16,
    &TorchDtype::Int64 => Dtype::Int64,
    &TorchDtype::Int32 => Dtype::Int32,
    &TorchDtype::Int16 => Dtype::Int16,
    &TorchDtype::Int8 => Dtype::Int8,
    &TorchDtype::UInt64 => Dtype::UInt64,
    &TorchDtype::UInt32 => Dtype::UInt32,
    &TorchDtype::UInt16 => Dtype::UInt16,
    &TorchDtype::UInt8 => Dtype::UInt8,
    //&TorchDtype::Unknown(_) => return None
  }
}

fn main() {
  let argv: Vec<_> = env::args().collect();
  if argv.len() <= 1 {
    return;
  }
  let p = PathBuf::from(&argv[1]);
  let th = TorchFile::open(p).unwrap();
  for (i, t) in th.tensors().iter().enumerate() {
    println!("DEBUG: tensor i={} dtype={:?} shape={:?} stride={:?} name='{}'",
        i, t.tensor_type.as_ref().ok().map(|t| dtype_from_tensor_type(t)), &t.shape, &t.stride, sane_ascii(&t.name));
  }
}
