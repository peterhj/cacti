extern crate libc;
extern crate libloading;
extern crate once_cell;

use crate::bindings::*;
//use crate::types::*;

//use libc::{c_char, c_int, cpu_set_t};
use once_cell::sync::{Lazy};

pub mod bindings;
pub mod types;

pub static LIBCBLAS: Lazy<Libcblas> = Lazy::new(|| {
  unsafe {
    match Libcblas::open_default() {
      Err(_) => {
        panic!("bug: failed to dynamically link cblas");
      }
      Ok(lib) => lib
    }
  }
});
