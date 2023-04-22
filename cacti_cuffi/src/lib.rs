extern crate libc;
extern crate libloading;
extern crate once_cell;

use crate::bindings::*;
use crate::types::*;

use libc::{c_void, c_int};
use once_cell::sync::{Lazy};

use std::ptr::{null_mut};

pub mod bindings;
#[cfg(feature = "experimental")]
pub mod experimental;
pub mod types;

static LIBCUDART: Lazy<Libcudart> = Lazy::new(|| {
  let mut lib = Libcudart::default();
  unsafe {
    if lib.load_default().is_err() {
      panic!("bug: failed to dynamically link cudart");
    }
  }
  lib
});

pub type CudartResult<T=()> = Result<T, i32>;

pub fn cudart_get_dev_count() -> CudartResult<i32> {
  let mut c = 0;
  let e = unsafe {
    (LIBCUDART.cudaGetDeviceCount.as_ref().unwrap())(&mut c as *mut _)
  };
  if e != cudaSuccess {
    return Err(e);
  }
  Ok(c)
}

pub fn cudart_get_cur_dev() -> CudartResult<i32> {
  let mut dev: c_int = -1;
  let e = unsafe {
    (LIBCUDART.cudaGetDevice.as_ref().unwrap())(&mut dev as *mut _)
  };
  if e != cudaSuccess {
    return Err(e);
  }
  Ok(dev)
}

pub fn cudart_set_cur_dev(dev: i32) -> CudartResult {
  let e = unsafe {
    (LIBCUDART.cudaSetDevice.as_ref().unwrap())(dev)
  };
  if e != cudaSuccess {
    return Err(e);
  }
  Ok(())
}

pub fn cudart_get_mem_info() -> CudartResult<(usize, usize)> {
  let mut free = 0;
  let mut total = 0;
  let e = unsafe {
    (LIBCUDART.cudaMemGetInfo.as_ref().unwrap())(&mut free as *mut _, &mut total as *mut _)
  };
  if e != cudaSuccess {
    return Err(e);
  }
  Ok((free, total))
}

pub fn cudart_sync() -> CudartResult {
  let e = unsafe {
    (LIBCUDART.cudaDeviceSynchronize.as_ref().unwrap())()
  };
  if e != cudaSuccess {
    return Err(e);
  }
  Ok(())
}

pub fn cudart_malloc(sz: usize) -> CudartResult<*mut c_void> {
  let mut ptr = null_mut();
  let e = unsafe {
    (LIBCUDART.cudaMalloc.as_ref().unwrap())(&mut ptr as *mut _, sz)
  };
  if e != cudaSuccess {
    return Err(e);
  }
  Ok(ptr)
}

pub fn cudart_free(ptr: *mut c_void) -> CudartResult {
  let e = unsafe {
    (LIBCUDART.cudaFree.as_ref().unwrap())(ptr)
  };
  if e != cudaSuccess {
    return Err(e);
  }
  Ok(())
}

pub fn cudart_memcpy(dst: *mut c_void, src: *const c_void, sz: usize, stream: &CudartStream) -> CudartResult {
  let e = unsafe {
    (LIBCUDART.cudaMemcpyAsync.as_ref().unwrap())(dst, src, sz, cudaMemcpyDefault, stream.raw)
  };
  if e != cudaSuccess {
    return Err(e);
  }
  Ok(())
}

pub struct CudartEvent {
  raw:  cudaEvent_t,
  dev:  i32,
}

impl Drop for CudartEvent {
  fn drop(&mut self) {
    assert!(!self.raw.is_null());
    match cudart_set_cur_dev(self.dev) {
      Ok(_) | Err(cudaCudartUnloading) => {}
      _ => panic!("bug")
    }
    let e = unsafe {
      (LIBCUDART.cudaEventDestroy.as_ref().unwrap())(self.raw)
    };
    match e {
      cudaSuccess | cudaCudartUnloading => {}
      _ => panic!("bug")
    }
  }
}

impl CudartEvent {
  pub fn create_fastest() -> CudartResult<CudartEvent> {
    let dev = cudart_get_cur_dev()?;
    let mut raw: cudaEvent_t = null_mut();
    let e = unsafe {
      (LIBCUDART.cudaEventCreateWithFlags.as_ref().unwrap())(&mut raw as *mut _, cudaEventDisableTiming)
    };
    if e != cudaSuccess {
      return Err(e);
    }
    assert!(!raw.is_null());
    Ok(CudartEvent{raw, dev})
  }

  pub fn record(&self, stream: &CudartStream) -> CudartResult {
    if stream.dev >= 0 {
      assert_eq!(self.dev, stream.dev);
    }
    let e = unsafe {
      (LIBCUDART.cudaEventRecordWithFlags.as_ref().unwrap())(self.raw, stream.raw, 0)
    };
    if e != cudaSuccess {
      return Err(e);
    }
    Ok(())
  }

  pub fn sync(&self) -> CudartResult {
    let e = unsafe {
      (LIBCUDART.cudaEventSynchronize.as_ref().unwrap())(self.raw)
    };
    if e != cudaSuccess {
      return Err(e);
    }
    Ok(())
  }
}

pub struct CudartStream {
  raw:  cudaStream_t,
  dev:  i32,
}

impl Drop for CudartStream {
  fn drop(&mut self) {
    if self.raw.is_null() {
      return;
    }
    assert!(self.dev >= 0);
    match cudart_set_cur_dev(self.dev) {
      Ok(_) | Err(cudaCudartUnloading) => {}
      _ => panic!("bug")
    }
    let e = unsafe {
      (LIBCUDART.cudaStreamDestroy.as_ref().unwrap())(self.raw)
    };
    match e {
      cudaSuccess | cudaCudartUnloading => {}
      _ => panic!("bug")
    }
  }
}

impl CudartStream {
  pub fn null() -> CudartStream {
    let raw: cudaStream_t = null_mut();
    let dev = -1;
    CudartStream{raw, dev}
  }

  pub fn create() -> CudartResult<CudartStream> {
    let dev = cudart_get_cur_dev()?;
    let mut raw: cudaStream_t = null_mut();
    let e = unsafe {
      (LIBCUDART.cudaStreamCreateWithFlags.as_ref().unwrap())(&mut raw as *mut _, cudaStreamDefault)
    };
    if e != cudaSuccess {
      return Err(e);
    }
    assert!(!raw.is_null());
    Ok(CudartStream{raw, dev})
  }

  pub fn create_nonblocking() -> CudartResult<CudartStream> {
    let dev = cudart_get_cur_dev()?;
    let mut raw: cudaStream_t = null_mut();
    let e = unsafe {
      (LIBCUDART.cudaStreamCreateWithFlags.as_ref().unwrap())(&mut raw as *mut _, cudaStreamNonblocking)
    };
    if e != cudaSuccess {
      return Err(e);
    }
    assert!(!raw.is_null());
    Ok(CudartStream{raw, dev})
  }

  pub fn sync(&self) -> CudartResult {
    let e = unsafe {
      (LIBCUDART.cudaStreamSynchronize.as_ref().unwrap())(self.raw)
    };
    if e != cudaSuccess {
      return Err(e);
    }
    Ok(())
  }

  pub fn wait_event(&self, event: &CudartEvent) -> CudartResult {
    let e = unsafe {
      (LIBCUDART.cudaStreamWaitEvent.as_ref().unwrap())(self.raw, event.raw, 0)
    };
    if e != cudaSuccess {
      return Err(e);
    }
    Ok(())
  }
}
