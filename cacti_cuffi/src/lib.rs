extern crate libloading;

//use libc::{c_void, c_int};
use libloading::os::unix::{Library, Symbol};

use std::ffi::{c_void};

#[allow(non_snake_case)]
#[derive(Default)]
pub struct Libcuda {
  inner_library:    Option<Library>,
  pub cudaAlloc:    Option<Symbol<extern "C" fn (usize, *mut *mut c_void) -> i32>>,
  //pub cudaMemcpy:   _,
  // TODO
}

impl Libcuda {
  pub fn open_default() -> Result<Libcuda, ()> {
    let mut lib = Libcuda::default();
    unsafe {
      let library = Library::new("cuda").map_err(|_| ())?;
      lib.cudaAlloc = library.get(b"cudaAlloc").ok();
      // TODO
      lib.inner_library = library.into();
    }
    lib._check_required()?;
    Ok(lib)
  }

  fn _check_required(&self) -> Result<(), ()> {
    self.cudaAlloc.as_ref().ok_or(())?;
    // TODO
    Ok(())
  }
}

#[allow(non_snake_case)]
#[derive(Default)]
pub struct Libcublas {
  inner_library:    Option<Library>,
  // TODO
}

#[allow(non_snake_case)]
#[derive(Default)]
pub struct Libcudnn {
  inner_library:    Option<Library>,
  // TODO
}
