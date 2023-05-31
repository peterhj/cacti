#![allow(non_upper_case_globals)]

extern crate cacti_cfg_env;
extern crate libc;
extern crate libloading;
extern crate once_cell;

use crate::bindings::*;
use crate::types::*;

use libc::{c_void, c_int};
use once_cell::sync::{Lazy};

use std::ops::{Deref};
use std::ptr::{null_mut};

pub mod bindings;
#[cfg(feature = "experimental")]
pub mod experimental;
pub mod types;

pub static LIBCUDA: Lazy<Libcuda> = Lazy::new(|| {
  let mut lib = Libcuda::default();
  unsafe {
    if let Err(_) = lib.open_default() {
      panic!("bug: failed to dynamically link libcuda.so");
    }
    if let Err(code) = lib.load_symbols() {
      panic!("bug: failed to load symbols from libcuda.so: {}", code);
    }
    if let Err(e) = lib.try_init() {
      panic!("bug: cuda: init failed: {:?}", e);
    }
  }
  println!("DEBUG: libcuda loaded");
  lib
});

pub static LIBCUDART: Lazy<Libcudart> = Lazy::new(|| {
  let mut lib = Libcudart::default();
  unsafe {
    if let Err(_) = lib.open_default() {
      panic!("bug: failed to dynamically link libcudart.so");
    }
    if let Err(code) = lib.load_symbols() {
      panic!("bug: failed to load symbols from libcudart.so: {}", code);
    }
  }
  lib
});

pub static LIBNVRTC_BUILTINS: Lazy<Libnvrtc_builtins> = Lazy::new(|| {
  let mut lib = Libnvrtc_builtins::default();
  unsafe {
    if let Err(_) = lib.open_default() {
      panic!("bug: failed to dynamically link libnvrtc-builtins.so");
    }
  }
  lib
});

thread_local! {
  pub static TL_LIBNVRTC_BUILTINS_BARRIER: bool = LIBNVRTC_BUILTINS._inner.is_some();
}

pub static LIBNVRTC: Lazy<Libnvrtc> = Lazy::new(|| {
  let mut lib = Libnvrtc::default();
  unsafe {
    if let Err(_) = lib.open_default() {
      panic!("bug: failed to dynamically link libnvrtc.so");
    }
    if let Err(code) = lib.load_symbols() {
      panic!("bug: failed to load symbols from libnvrtc.so: {}", code);
    }
  }
  lib
});

pub static LIBCUBLAS: Lazy<Libcublas> = Lazy::new(|| {
  let mut lib = Libcublas::default();
  unsafe {
    if let Err(_) = lib.open_default() {
      panic!("bug: failed to dynamically link libcublas.so");
    }
    if let Err(code) = lib.load_symbols() {
      panic!("bug: failed to load symbols from libcublas.so: {}", code);
    }
  }
  lib
});

pub type CudaResult<T=()> = Result<T, CUresult>;

pub struct CudaPrimaryCtx {
  raw:  CUcontext,
  dev:  i32,
}

impl Drop for CudaPrimaryCtx {
  fn drop(&mut self) {
    let e = (LIBCUDA.cuDevicePrimaryCtxRelease.as_ref().unwrap())(self.dev);
    match e {
      CUDA_SUCCESS | CUDA_ERROR_DEINITIALIZED => {}
      _ => panic!("bug")
    }
  }
}

impl CudaPrimaryCtx {
  pub fn retain(dev: i32) -> CudaResult<CudaPrimaryCtx> {
    let mut raw = null_mut();
    let e = (LIBCUDA.cuDevicePrimaryCtxRetain.as_ref().unwrap())(&mut raw, dev);
    if e != CUDA_SUCCESS {
      return Err(e);
    }
    Ok(CudaPrimaryCtx{dev, raw})
  }

  pub fn set_flags(&self, flags: u32) -> CudaResult {
    let e = (LIBCUDA.cuDevicePrimaryCtxSetFlags.as_ref().unwrap())(self.dev, flags);
    if e != CUDA_SUCCESS {
      return Err(e);
    }
    Ok(())
  }

  pub fn device(&self) -> i32 {
    self.dev
  }
}

pub fn cuda_mem_alloc(sz: usize) -> CudaResult<u64> {
  let mut dptr = 0;
  let e = (LIBCUDA.cuMemAlloc.as_ref().unwrap())(&mut dptr, sz);
  if e != CUDA_SUCCESS {
    return Err(e);
  }
  Ok(dptr)
}

pub fn cuda_mem_alloc_host(sz: usize) -> CudaResult<*mut c_void> {
  let mut ptr = null_mut();
  let e = (LIBCUDA.cuMemAllocHost.as_ref().unwrap())(&mut ptr, sz);
  if e != CUDA_SUCCESS {
    return Err(e);
  }
  Ok(ptr)
}

pub fn cuda_mem_free(dptr: u64) -> CudaResult {
  let e = (LIBCUDA.cuMemFree.as_ref().unwrap())(dptr);
  if e != CUDA_SUCCESS {
    return Err(e);
  }
  Ok(())
}

pub fn cuda_mem_free_host(ptr: *mut c_void) -> CudaResult {
  let e = (LIBCUDA.cuMemFreeHost.as_ref().unwrap())(ptr);
  if e != CUDA_SUCCESS {
    return Err(e);
  }
  Ok(())
}

pub fn cuda_memcpy_async(dst: u64, src: u64, sz: usize, stream_raw: &CUstream) -> CudaResult {
  let e = (LIBCUDA.cuMemcpyAsync.as_ref().unwrap())(dst, src, sz, *stream_raw);
  if e != CUDA_SUCCESS {
    return Err(e);
  }
  Ok(())
}

pub fn cuda_memcpy_h2d_async(dst: u64, src: *const c_void, sz: usize, stream_raw: &CUstream) -> CudaResult {
  let e = (LIBCUDA.cuMemcpyHtoDAsync.as_ref().unwrap())(dst, src, sz, *stream_raw);
  if e != CUDA_SUCCESS {
    return Err(e);
  }
  Ok(())
}

pub fn cuda_memcpy_d2h_async(dst: *mut c_void, src: u64, sz: usize, stream_raw: &CUstream) -> CudaResult {
  let e = (LIBCUDA.cuMemcpyDtoHAsync.as_ref().unwrap())(dst, src, sz, *stream_raw);
  if e != CUDA_SUCCESS {
    return Err(e);
  }
  Ok(())
}

pub type CudartResult<T=()> = Result<T, cudaError_t>;

pub fn cudart_get_dev_count() -> CudartResult<i32> {
  let mut c = 0;
  let e = (LIBCUDART.cudaGetDeviceCount.as_ref().unwrap())(&mut c);
  if e != cudaSuccess {
    return Err(e);
  }
  Ok(c)
}

pub fn cudart_get_cur_dev() -> CudartResult<i32> {
  let mut dev: c_int = -1;
  let e = (LIBCUDART.cudaGetDevice.as_ref().unwrap())(&mut dev);
  if e != cudaSuccess {
    return Err(e);
  }
  Ok(dev)
}

pub fn cudart_set_cur_dev(dev: i32) -> CudartResult {
  let e = (LIBCUDART.cudaSetDevice.as_ref().unwrap())(dev);
  if e != cudaSuccess {
    return Err(e);
  }
  Ok(())
}

pub fn cudart_get_mem_info() -> CudartResult<(usize, usize)> {
  let mut free = 0;
  let mut total = 0;
  let e = (LIBCUDART.cudaMemGetInfo.as_ref().unwrap())(&mut free, &mut total);
  if e != cudaSuccess {
    return Err(e);
  }
  Ok((free, total))
}

pub fn cudart_sync() -> CudartResult {
  let e = (LIBCUDART.cudaDeviceSynchronize.as_ref().unwrap())();
  if e != cudaSuccess {
    return Err(e);
  }
  Ok(())
}

pub fn cudart_malloc(sz: usize) -> CudartResult<*mut c_void> {
  let mut ptr = null_mut();
  let e = (LIBCUDART.cudaMalloc.as_ref().unwrap())(&mut ptr as *mut _, sz);
  if e != cudaSuccess {
    return Err(e);
  }
  Ok(ptr)
}

pub fn cudart_free(ptr: *mut c_void) -> CudartResult {
  let e = (LIBCUDART.cudaFree.as_ref().unwrap())(ptr);
  if e != cudaSuccess {
    return Err(e);
  }
  Ok(())
}

pub fn cudart_memcpy(dst: *mut c_void, src: *const c_void, sz: usize, stream: &CudartStream) -> CudartResult {
  let e = (LIBCUDART.cudaMemcpyAsync.as_ref().unwrap())(dst, src, sz, cudaMemcpyDefault, stream.raw);
  if e != cudaSuccess {
    return Err(e);
  }
  Ok(())
}

pub struct CudartEvent {
  raw:  cudaEvent_t,
  dev:  i32,
}

impl Deref for CudartEvent {
  type Target = CUevent;

  #[inline]
  fn deref(&self) -> &CUevent {
    &self.raw
  }
}

impl Drop for CudartEvent {
  fn drop(&mut self) {
    assert!(!self.raw.is_null());
    match cudart_set_cur_dev(self.dev) {
      Ok(_) | Err(cudaErrorCudartUnloading) => {}
      _ => panic!("bug")
    }
    let e = (LIBCUDART.cudaEventDestroy.as_ref().unwrap())(self.raw);
    match e {
      cudaSuccess | cudaErrorCudartUnloading => {}
      _ => panic!("bug")
    }
  }
}

impl CudartEvent {
  pub fn create_fastest() -> CudartResult<CudartEvent> {
    let dev = cudart_get_cur_dev()?;
    let mut raw: cudaEvent_t = null_mut();
    let e = (LIBCUDART.cudaEventCreateWithFlags.as_ref().unwrap())(&mut raw, cudaEventDisableTiming);
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
    let e = (LIBCUDART.cudaEventRecordWithFlags.as_ref().unwrap())(self.raw, stream.raw, 0);
    if e != cudaSuccess {
      return Err(e);
    }
    Ok(())
  }

  pub fn query(&self) -> CudartResult {
    let e = (LIBCUDART.cudaEventQuery.as_ref().unwrap())(self.raw);
    if e != cudaSuccess {
      return Err(e);
    }
    Ok(())
  }

  pub fn sync(&self) -> CudartResult {
    let e = (LIBCUDART.cudaEventSynchronize.as_ref().unwrap())(self.raw);
    if e != cudaSuccess {
      return Err(e);
    }
    Ok(())
  }

  pub fn as_ptr(&self) -> CUevent {
    self.raw
  }

  pub fn device(&self) -> i32 {
    self.dev
  }
}

#[repr(C)]
pub struct CudartStream {
  raw:  cudaStream_t,
  dev:  i32,
}

impl Deref for CudartStream {
  type Target = CUstream;

  #[inline]
  fn deref(&self) -> &CUstream {
    &self.raw
  }
}

impl Drop for CudartStream {
  fn drop(&mut self) {
    if self.raw.is_null() {
      return;
    }
    assert!(self.dev >= 0);
    match cudart_set_cur_dev(self.dev) {
      Ok(_) | Err(cudaErrorCudartUnloading) => {}
      _ => panic!("bug")
    }
    let e = (LIBCUDART.cudaStreamDestroy.as_ref().unwrap())(self.raw);
    match e {
      cudaSuccess | cudaErrorCudartUnloading => {}
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
    let e = (LIBCUDART.cudaStreamCreateWithFlags.as_ref().unwrap())(&mut raw, cudaStreamDefault);
    if e != cudaSuccess {
      return Err(e);
    }
    assert!(!raw.is_null());
    Ok(CudartStream{raw, dev})
  }

  pub fn create_nonblocking() -> CudartResult<CudartStream> {
    let dev = cudart_get_cur_dev()?;
    let mut raw: cudaStream_t = null_mut();
    let e = (LIBCUDART.cudaStreamCreateWithFlags.as_ref().unwrap())(&mut raw, cudaStreamNonblocking);
    if e != cudaSuccess {
      return Err(e);
    }
    assert!(!raw.is_null());
    Ok(CudartStream{raw, dev})
  }

  pub fn sync(&self) -> CudartResult {
    let e = (LIBCUDART.cudaStreamSynchronize.as_ref().unwrap())(self.raw);
    if e != cudaSuccess {
      return Err(e);
    }
    Ok(())
  }

  pub fn wait_event(&self, event: &CudartEvent) -> CudartResult {
    let e = (LIBCUDART.cudaStreamWaitEvent.as_ref().unwrap())(self.raw, event.raw, 0);
    if e != cudaSuccess {
      return Err(e);
    }
    Ok(())
  }

  pub fn as_ptr(&self) -> CUstream {
    self.raw
  }

  pub fn device(&self) -> i32 {
    self.dev
  }
}
