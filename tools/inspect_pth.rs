extern crate cacti;
extern crate repugnant_pickle;

use cacti::cell::Dtype;
use repugnant_pickle::{RepugnantTorchFile, TensorType};

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

fn dtype_from_tensor_type(t: &TensorType) -> Option<Dtype> {
  Some(match t {
    &TensorType::Float64 => Dtype::Float64,
    &TensorType::Float32 => Dtype::Float32,
    &TensorType::Float16 => Dtype::Float16,
    &TensorType::BFloat16 => Dtype::BFloat16,
    &TensorType::Int64 => Dtype::Int64,
    &TensorType::Int32 => Dtype::Int32,
    &TensorType::Int16 => Dtype::Int16,
    &TensorType::Int8 => Dtype::Int8,
    &TensorType::UInt8 => Dtype::UInt8,
    &TensorType::Unknown(_) => return None
  })
}

fn main() {
  let argv: Vec<_> = env::args().collect();
  if argv.len() <= 1 {
    return;
  }
  let p = PathBuf::from(&argv[1]);
  let th = RepugnantTorchFile::new_from_file(p).unwrap();
  for (i, t) in th.tensors().iter().enumerate() {
    println!("DEBUG: tensor i={} dtype={:?} shape={:?} stride={:?} name='{}'",
        i, dtype_from_tensor_type(&t.tensor_type), &t.shape, &t.stride, sane_ascii(&t.name));
  }
}
