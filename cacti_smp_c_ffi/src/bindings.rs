use crate::types::*;

use libc::{c_char, c_int, cpu_set_t};
use libloading::nonsafe::{Library, Symbol};

#[derive(Default)]
pub struct Libopenblas {
  pub set_num_threads:  Option<Symbol<extern "C" fn (c_int)>>,
  pub get_num_threads:  Option<Symbol<extern "C" fn () -> c_int>>,
  pub get_config:       Option<Symbol<extern "C" fn () -> *mut c_char>>,
  pub get_corename:     Option<Symbol<extern "C" fn () -> *mut c_char>>,
  pub setaffinity:      Option<Symbol<extern "C" fn (c_int, usize, *mut cpu_set_t) -> c_int>>,
  pub getaffinity:      Option<Symbol<extern "C" fn (c_int, usize, *mut cpu_set_t) -> c_int>>,
  pub get_parallel:     Option<Symbol<extern "C" fn () -> c_int>>,
  // TODO
}

impl Libopenblas {
  pub unsafe fn load_symbols(&mut self, library: &Library) -> Result<(), ()> {
    self.set_num_threads = library.get(b"openblas_set_num_threads").ok();
    self.get_num_threads = library.get(b"openblas_get_num_threads").ok();
    self.get_config = library.get(b"openblas_get_config").ok();
    self.get_corename = library.get(b"openblas_get_corename").ok();
    self.setaffinity = library.get(b"openblas_setaffinity").ok();
    self.getaffinity = library.get(b"openblas_getaffinity").ok();
    self.get_parallel = library.get(b"openblas_get_parallel").ok();
    // TODO
    self._check_required()?;
    Ok(())
  }

  fn _check_required(&self) -> Result<(), ()> {
    //self._.as_ref().ok_or(())?;
    // TODO
    Ok(())
  }
}

#[derive(Default)]
pub struct Libcblas {
  inner_library:        Option<Library>,
  pub openblas:         Libopenblas,
  // TODO
  pub cblas_sdot:       Option<Symbol<extern "C" fn (
                            c_int,
                            *const f32, c_int,
                            *const f32, c_int,
                        ) -> f32>>,
  pub cblas_sasum:      Option<Symbol<extern "C" fn (
                            c_int,
                            *const f32, c_int,
                        ) -> f32>>,
  pub cblas_ssum:       Option<Symbol<extern "C" fn (
                            c_int,
                            *const f32, c_int,
                        ) -> f32>>,
  pub cblas_snrm2:      Option<Symbol<extern "C" fn (
                            c_int,
                            *const f32, c_int,
                        ) -> f32>>,
  pub cblas_isamax:     Option<Symbol<extern "C" fn (
                            c_int,
                            *const f32, c_int,
                        ) -> CBLAS_INDEX>>,
  pub cblas_isamin:     Option<Symbol<extern "C" fn (
                            c_int,
                            *const f32, c_int,
                        ) -> CBLAS_INDEX>>,
  pub cblas_ismax:      Option<Symbol<extern "C" fn (
                            c_int,
                            *const f32, c_int,
                        ) -> CBLAS_INDEX>>,
  pub cblas_ismin:      Option<Symbol<extern "C" fn (
                            c_int,
                            *const f32, c_int,
                        ) -> CBLAS_INDEX>>,
  pub cblas_saxpy:      Option<Symbol<extern "C" fn (
                            c_int,
                            f32,
                            *const f32, c_int,
                            *mut f32, c_int,
                        )>>,
  pub cblas_scopy:      Option<Symbol<extern "C" fn (
                            c_int,
                            *const f32, c_int,
                            *mut f32, c_int,
                        )>>,
  pub cblas_sswap:      Option<Symbol<extern "C" fn (
                            c_int,
                            *mut f32, c_int,
                            *mut f32, c_int,
                        )>>,
  pub cblas_sscal:      Option<Symbol<extern "C" fn (
                            c_int,
                            f32,
                            *mut f32, c_int,
                        )>>,
  pub cblas_sgemv:      Option<Symbol<extern "C" fn (
                            CBLAS_ORDER,
                            CBLAS_TRANSPOSE,
                            c_int, c_int,
                            f32,
                            *const f32, c_int,
                            *const f32, c_int,
                            f32,
                            *mut f32, c_int,
                        ) -> f32>>,
  pub cblas_sgemm:      Option<Symbol<extern "C" fn (
                            CBLAS_ORDER,
                            CBLAS_TRANSPOSE,
                            CBLAS_TRANSPOSE,
                            c_int, c_int, c_int,
                            f32,
                            *const f32, c_int,
                            *const f32, c_int,
                            f32,
                            *mut f32, c_int,
                        ) -> f32>>,
  // TODO
}

impl Drop for Libcblas {
  fn drop(&mut self) {
    let inner_library = self.inner_library.take();
    if let Some(inner) = inner_library {
      *self = Default::default();
      drop(inner);
    }
  }
}

impl Libcblas {
  pub unsafe fn open_default() -> Result<Libcblas, ()> {
    let library = Library::new("libcblas.so").map_err(|_| ())?;
    let mut this = Libcblas::default();
    this.inner_library = library.into();
    this.load_symbols()?;
    Ok(this)
  }

  pub unsafe fn load_symbols(&mut self) -> Result<(), ()> {
    let library = self.inner_library.as_ref().unwrap();
    self.openblas.load_symbols(library)?;
    self.cblas_sdot = library.get(b"cblas_sdot").ok();
    self.cblas_sasum = library.get(b"cblas_sasum").ok();
    self.cblas_ssum = library.get(b"cblas_ssum").ok();
    self.cblas_snrm2 = library.get(b"cblas_snrm2").ok();
    self.cblas_isamax = library.get(b"cblas_isamax").ok();
    self.cblas_isamin = library.get(b"cblas_isamin").ok();
    self.cblas_ismax = library.get(b"cblas_ismax").ok();
    self.cblas_ismin = library.get(b"cblas_ismin").ok();
    self.cblas_saxpy = library.get(b"cblas_saxpy").ok();
    self.cblas_scopy = library.get(b"cblas_scopy").ok();
    self.cblas_sswap = library.get(b"cblas_sswap").ok();
    self.cblas_sscal = library.get(b"cblas_sscal").ok();
    self.cblas_sgemv = library.get(b"cblas_sgemv").ok();
    self.cblas_sgemm = library.get(b"cblas_sgemm").ok();
    // TODO
    self._check_required()?;
    Ok(())
  }

  fn _check_required(&self) -> Result<(), ()> {
    //self._.as_ref().ok_or(())?;
    // TODO
    Ok(())
  }
}
