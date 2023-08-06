#![allow(non_upper_case_globals)]

extern crate cacti_cfg_env;
extern crate libc;
extern crate libloading;
extern crate once_cell;

use crate::bindings::*;
use crate::types::*;
use cacti_cfg_env::*;

use libc::{c_void};
use once_cell::sync::{Lazy};

use std::cell::{UnsafeCell};
use std::convert::{TryFrom};
use std::ffi::{CStr};
use std::ops::{Deref};
use std::ptr::{null, null_mut};

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
  if cfg_debug() { println!("DEBUG: libcuda loaded"); }
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

pub fn cuda_device_get_count() -> CudaResult<i32> {
  let mut c = -1;
  let e = (LIBCUDA.cuDeviceGetCount.as_ref().unwrap())(&mut c);
  if e != CUDA_SUCCESS {
    return Err(e);
  }
  Ok(c)
}

pub fn cuda_device_get(rank: i32) -> CudaResult<i32> {
  let mut dev = -1;
  let e = (LIBCUDA.cuDeviceGet.as_ref().unwrap())(&mut dev, rank);
  if e != CUDA_SUCCESS {
    return Err(e);
  }
  Ok(dev)
}

pub fn cuda_device_attribute_get(dev: i32, key: i32) -> CudaResult<i32> {
  let mut val = -1;
  let e = (LIBCUDA.cuDeviceGetAttribute.as_ref().unwrap())(&mut val, key, dev);
  if e != CUDA_SUCCESS {
    return Err(e);
  }
  Ok(val)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(i32)]
pub enum CudaComputeMode {
  Default = CU_COMPUTEMODE_DEFAULT,
  Prohibited = CU_COMPUTEMODE_PROHIBITED,
  ExclusiveProcess = CU_COMPUTEMODE_EXCLUSIVE_PROCESS,
}

impl TryFrom<i32> for CudaComputeMode {
  type Error = i32;

  fn try_from(val: i32) -> Result<CudaComputeMode, i32> {
    Ok(match val {
      CU_COMPUTEMODE_DEFAULT => CudaComputeMode::Default,
      CU_COMPUTEMODE_PROHIBITED => CudaComputeMode::Prohibited,
      CU_COMPUTEMODE_EXCLUSIVE_PROCESS => CudaComputeMode::ExclusiveProcess,
      _ => return Err(val)
    })
  }
}

pub fn cudart_device_reset() -> CudartResult {
  let e = (LIBCUDART.cudaDeviceReset.as_ref().unwrap())();
  if e != cudaSuccess {
    return Err(e);
  }
  Ok(())
}

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
  pub fn reset(dev: i32) -> CudaResult {
    let e = (LIBCUDA.cuDevicePrimaryCtxReset.as_ref().unwrap())(dev);
    if e != CUDA_SUCCESS {
      return Err(e);
    }
    Ok(())
  }

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

  pub fn as_ptr(&self) -> CUcontext {
    self.raw
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

pub fn cuda_memcpy_d2h(dst: *mut c_void, src: u64, sz: usize) -> CudaResult {
  let e = (LIBCUDA.cuMemcpyDtoH.as_ref().unwrap())(dst, src, sz);
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

pub struct CudaModule {
  pub raw: CUmodule,
}

impl Drop for CudaModule {
  fn drop(&mut self) {
    let e = (LIBCUDA.cuModuleUnload.as_ref().unwrap())(self.raw);
    match e {
      CUDA_SUCCESS | CUDA_ERROR_DEINITIALIZED => {}
      _ => panic!("bug")
    }
  }
}

impl CudaModule {
  pub fn load_data(data: *const c_void) -> CudaResult<CudaModule> {
    let mut raw = null_mut();
    let e = (LIBCUDA.cuModuleLoadData.as_ref().unwrap())(&mut raw, data);
    if e != CUDA_SUCCESS {
      return Err(e);
    }
    Ok(CudaModule{raw})
  }

  pub fn get_function(&self, fname_buf: &[u8]) -> CudaResult<CudaFunction> {
    let fname_cstr = CStr::from_bytes_with_nul(fname_buf).unwrap();
    let mut func_raw = null_mut();
    let e = (LIBCUDA.cuModuleGetFunction.as_ref().unwrap())(&mut func_raw, self.raw, fname_cstr.as_ptr());
    if e != CUDA_SUCCESS {
      return Err(e);
    }
    assert!(!func_raw.is_null());
    Ok(CudaFunction{raw: func_raw})
  }
}

pub struct CudaFunction {
  pub raw: CUfunction,
}

impl CudaFunction {
  pub fn launch_kernel(&self, grid_dim: [u32; 3], block_dim: [u32; 3], shared_mem: u32, args: &UnsafeCell<[*mut c_void]>, stream_raw: &CUstream) -> CudaResult {
    let e = unsafe {
      (LIBCUDA.cuLaunchKernel.as_ref().unwrap())(
          self.raw,
          grid_dim[0], grid_dim[1], grid_dim[2],
          block_dim[0], block_dim[1], block_dim[2],
          shared_mem,
          *stream_raw,
          args.get() as *mut *mut c_void,
          null_mut()
      )
    };
    if e != CUDA_SUCCESS {
      return Err(e);
    }
    Ok(())
  }
}

pub type CudartResult<T=()> = Result<T, cudaError_t>;

pub fn cudart_get_dev_count() -> CudartResult<i32> {
  let mut c = -1;
  let e = (LIBCUDART.cudaGetDeviceCount.as_ref().unwrap())(&mut c);
  if e != cudaSuccess {
    return Err(e);
  }
  Ok(c)
}

pub fn cudart_get_cur_dev() -> CudartResult<i32> {
  let mut dev = -1;
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

pub type NvrtcResult<T=()> = Result<T, nvrtcResult>;

pub struct NvrtcProgram {
  pub raw:  nvrtcProgram,
}

impl Drop for NvrtcProgram {
  fn drop(&mut self) {
    assert!(!self.raw.is_null());
    let _e = (LIBNVRTC.nvrtcDestroyProgram.as_ref().unwrap())(&mut self.raw);
  }
}

impl NvrtcProgram {
  pub fn create(src_buf: &[u8]) -> NvrtcResult<NvrtcProgram> {
    let src_cstr = CStr::from_bytes_with_nul(src_buf).unwrap();
    let mut raw = null_mut();
    let e = (LIBNVRTC.nvrtcCreateProgram.as_ref().unwrap())(&mut raw, src_cstr.as_ptr(), null(), 0, null(), null());
    if e != NVRTC_SUCCESS {
      return Err(e);
    }
    assert!(!raw.is_null());
    Ok(NvrtcProgram{raw})
  }

  pub fn compile(&self, opt_bufs: &[&[u8]]) -> NvrtcResult {
    let mut opt_cstrs = Vec::with_capacity(opt_bufs.len());
    for buf in opt_bufs.iter() {
      opt_cstrs.push(CStr::from_bytes_with_nul(buf).unwrap().as_ptr());
    }
    assert!(opt_cstrs.len() <= i32::max_value() as usize);
    let e = (LIBNVRTC.nvrtcCompileProgram.as_ref().unwrap())(self.raw, opt_cstrs.len() as i32, opt_cstrs.as_ptr());
    if e != NVRTC_SUCCESS {
      return Err(e);
    }
    Ok(())
  }

  pub fn get_log_size(&self) -> NvrtcResult<usize> {
    let mut size = 0;
    let e = (LIBNVRTC.nvrtcGetProgramLogSize.as_ref().unwrap())(self.raw, &mut size);
    if e != NVRTC_SUCCESS {
      return Err(e);
    }
    Ok(size)
  }

  pub fn get_log(&self, buf: &mut [u8]) -> NvrtcResult {
    let e = (LIBNVRTC.nvrtcGetProgramLog.as_ref().unwrap())(self.raw, buf.as_mut_ptr() as *mut _);
    if e != NVRTC_SUCCESS {
      return Err(e);
    }
    Ok(())
  }

  pub fn get_ptx_size(&self) -> NvrtcResult<usize> {
    let mut size = 0;
    let e = (LIBNVRTC.nvrtcGetPTXSize.as_ref().unwrap())(self.raw, &mut size);
    if e != NVRTC_SUCCESS {
      return Err(e);
    }
    Ok(size)
  }

  pub fn get_ptx(&self, buf: &mut [u8]) -> NvrtcResult {
    let e = (LIBNVRTC.nvrtcGetPTX.as_ref().unwrap())(self.raw, buf.as_mut_ptr() as *mut _);
    if e != NVRTC_SUCCESS {
      return Err(e);
    }
    Ok(())
  }
}

pub type CublasResult<T=()> = Result<T, cublasStatus_t>;

#[derive(Clone, Copy, Debug)]
#[repr(i32)]
pub enum CublasPointerMode {
  Host = CUBLAS_POINTER_MODE_HOST,
  Device = CUBLAS_POINTER_MODE_DEVICE,
}

#[derive(Clone, Copy, Debug)]
#[repr(i32)]
pub enum CublasAtomicsMode {
  NotAllowed = CUBLAS_ATOMICS_NOT_ALLOWED,
  Allowed = CUBLAS_ATOMICS_ALLOWED,
}

#[repr(C)]
pub struct CublasContext {
  raw:  cublasHandle_t,
  //dev:  i32,
}

impl Drop for CublasContext {
  fn drop(&mut self) {
    if self.raw.is_null() {
      return;
    }
    /*assert!(self.dev >= 0);
    match cudart_set_cur_dev(self.dev) {
      Ok(_) | Err(cudaErrorCudartUnloading) => {}
      _ => panic!("bug")
    }*/
    let e = (LIBCUBLAS.cublasDestroy.as_ref().unwrap())(self.raw);
    match e {
      CUBLAS_STATUS_SUCCESS => {}
      _ => panic!("bug")
    }
  }
}

impl CublasContext {
  pub fn create() -> CublasResult<CublasContext> {
    //let dev = cudart_get_cur_dev()?;
    let mut raw: cublasHandle_t = null_mut();
    let e = (LIBCUBLAS.cublasCreate.as_ref().unwrap())(&mut raw);
    if e != CUBLAS_STATUS_SUCCESS {
      return Err(e);
    }
    assert!(!raw.is_null());
    Ok(CublasContext{raw, /*dev*/})
  }

  pub fn set_stream(&self, stream_raw: &CUstream) -> CublasResult {
    let e = (LIBCUBLAS.cublasSetStream.as_ref().unwrap())(self.raw, *stream_raw);
    if e != CUBLAS_STATUS_SUCCESS {
      return Err(e);
    }
    Ok(())
  }

  pub fn set_pointer_mode(&self, mode: CublasPointerMode) -> CublasResult {
    let e = (LIBCUBLAS.cublasSetPointerMode.as_ref().unwrap())(self.raw, mode as _);
    if e != CUBLAS_STATUS_SUCCESS {
      return Err(e);
    }
    Ok(())
  }

  pub fn set_atomics_mode(&self, mode: CublasAtomicsMode) -> CublasResult {
    let e = (LIBCUBLAS.cublasSetAtomicsMode.as_ref().unwrap())(self.raw, mode as _);
    if e != CUBLAS_STATUS_SUCCESS {
      return Err(e);
    }
    Ok(())
  }

  pub fn set_math_mode(&self, flags: u32) -> CublasResult {
    let e = (LIBCUBLAS.cublasSetMathMode.as_ref().unwrap())(self.raw, flags);
    if e != CUBLAS_STATUS_SUCCESS {
      return Err(e);
    }
    Ok(())
  }

  pub fn as_ptr(&self) -> cublasHandle_t {
    self.raw
  }
}

pub fn cublas_gemm(
    ctx: &CublasContext,
    tr_a: bool,
    tr_b: bool,
    m: i32,
    n: i32,
    k: i32,
    alpha: *const c_void,
    a: u64,
    a_ty: cudaDataType_t,
    lda: i32,
    b: u64,
    b_ty: cudaDataType_t,
    ldb: i32,
    beta: *const c_void,
    c: u64,
    c_ty: cudaDataType_t,
    ldc: i32,
    stream_raw: &CUstream,
) -> CublasResult {
  ctx.set_stream(stream_raw).unwrap();
  ctx.set_pointer_mode(CublasPointerMode::Host).unwrap();
  ctx.set_atomics_mode(CublasAtomicsMode::NotAllowed).unwrap();
  let mut flags = CUBLAS_DEFAULT_MATH;
  let compute_ty = match (a_ty, b_ty, c_ty) {
    (CUDA_R_64F, CUDA_R_64F, CUDA_R_64F) => {
      CUBLAS_COMPUTE_64F
    }
    (CUDA_R_32F, CUDA_R_32F, CUDA_R_32F) => {
      CUBLAS_COMPUTE_32F
    }
    (CUDA_R_16F, CUDA_R_16F, CUDA_R_16F) |
    (CUDA_R_16F, CUDA_R_16F, CUDA_R_32F) |
    (CUDA_R_16F, CUDA_R_32F, CUDA_R_16F) |
    (CUDA_R_32F, CUDA_R_16F, CUDA_R_16F) |
    (CUDA_R_16F, CUDA_R_32F, CUDA_R_32F) |
    (CUDA_R_32F, CUDA_R_16F, CUDA_R_32F) |
    (CUDA_R_32F, CUDA_R_32F, CUDA_R_16F) => {
      if cfg_debug() { println!("DEBUG: cublas_gemm: fp16/mixed-precision mode"); }
      flags |= CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION;
      CUBLAS_COMPUTE_32F
    }
    _ => {
      panic!("bug: cacti_gpu_cu_ffi::cublas_gemm: unimplemented: a={} b={} c={}", a_ty, b_ty, c_ty);
    }
  };
  ctx.set_math_mode(flags).unwrap();
  let e = (LIBCUBLAS.cublasGemmEx.as_ref().unwrap())(
    ctx.raw,
    if tr_a { CUBLAS_OP_T } else { CUBLAS_OP_N },
    if tr_b { CUBLAS_OP_T } else { CUBLAS_OP_N },
    m, n, k,
    alpha,
    a as usize as *const _, a_ty, lda,
    b as usize as *const _, b_ty, ldb,
    beta,
    c as usize as *mut _, c_ty, ldc,
    compute_ty,
    CUBLAS_GEMM_DEFAULT,
  );
  if e != CUBLAS_STATUS_SUCCESS {
    return Err(e);
  }
  Ok(())
}

pub fn cublas_gemm_batched(
    ctx: &CublasContext,
    tr_a: bool,
    tr_b: bool,
    m: i32,
    n: i32,
    k: i32,
    alpha: *const c_void,
    a: u64,
    a_ty: cudaDataType_t,
    lda: i32,
    b: u64,
    b_ty: cudaDataType_t,
    ldb: i32,
    beta: *const c_void,
    c: u64,
    c_ty: cudaDataType_t,
    ldc: i32,
    nblk: i32,
    stream_raw: &CUstream,
) -> CublasResult {
  //assert_eq!(a.len(), c.len());
  //assert_eq!(b.len(), c.len());
  //assert!(c.len() <= i32::max_value() as usize);
  ctx.set_stream(stream_raw).unwrap();
  ctx.set_pointer_mode(CublasPointerMode::Host).unwrap();
  ctx.set_atomics_mode(CublasAtomicsMode::NotAllowed).unwrap();
  let mut flags = CUBLAS_DEFAULT_MATH;
  let e = match (a_ty, b_ty, c_ty) {
    (CUDA_R_64F, CUDA_R_64F, CUDA_R_64F) => {
      ctx.set_math_mode(flags).unwrap();
      (LIBCUBLAS.cublasDgemmBatched.as_ref().unwrap())(
        ctx.raw,
        if tr_a { CUBLAS_OP_T } else { CUBLAS_OP_N },
        if tr_b { CUBLAS_OP_T } else { CUBLAS_OP_N },
        m, n, k,
        alpha as *const _,
        a as usize as *const *const _, lda,
        b as usize as *const *const _, ldb,
        beta as *const _,
        c as usize as *const *mut _, ldc,
        nblk,
      )
    }
    (CUDA_R_32F, CUDA_R_32F, CUDA_R_32F) => {
      ctx.set_math_mode(flags).unwrap();
      (LIBCUBLAS.cublasSgemmBatched.as_ref().unwrap())(
        ctx.raw,
        if tr_a { CUBLAS_OP_T } else { CUBLAS_OP_N },
        if tr_b { CUBLAS_OP_T } else { CUBLAS_OP_N },
        m, n, k,
        alpha as *const _,
        a as usize as *const *const _, lda,
        b as usize as *const *const _, ldb,
        beta as *const _,
        c as usize as *const *mut _, ldc,
        nblk,
      )
    }
    (CUDA_R_16F, CUDA_R_16F, CUDA_R_16F) |
    (CUDA_R_16F, CUDA_R_16F, CUDA_R_32F) |
    (CUDA_R_16F, CUDA_R_32F, CUDA_R_16F) |
    (CUDA_R_32F, CUDA_R_16F, CUDA_R_16F) |
    (CUDA_R_16F, CUDA_R_32F, CUDA_R_32F) |
    (CUDA_R_32F, CUDA_R_16F, CUDA_R_32F) |
    (CUDA_R_32F, CUDA_R_32F, CUDA_R_16F) => {
      if cfg_debug() { println!("DEBUG: cublas_gemm_batched: fp16/mixed-precision mode"); }
      flags |= CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION;
      ctx.set_math_mode(flags).unwrap();
      (LIBCUBLAS.cublasGemmBatchedEx.as_ref().unwrap())(
        ctx.raw,
        if tr_a { CUBLAS_OP_T } else { CUBLAS_OP_N },
        if tr_b { CUBLAS_OP_T } else { CUBLAS_OP_N },
        m, n, k,
        alpha as *const _,
        a as usize as *const *const _, a_ty, lda,
        b as usize as *const *const _, b_ty, ldb,
        beta as *const _,
        c as usize as *const *mut _, c_ty, ldc,
        nblk,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT,
      )
    }
    _ => {
      panic!("bug: cacti_gpu_cu_ffi::cublas_gemm_batched: unimplemented: a={} b={} c={}", a_ty, b_ty, c_ty);
    }
  };
  if e != CUBLAS_STATUS_SUCCESS {
    return Err(e);
  }
  Ok(())
}
