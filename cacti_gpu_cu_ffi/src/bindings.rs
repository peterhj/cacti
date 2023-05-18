use crate::types::*;

use libc::{c_void, c_char, c_int, c_uint};
use libloading::nonsafe::{Library, Symbol};

use std::env;

#[cfg(not(unix))]
pub unsafe fn open_default<S: AsRef<str>>(_lib_name: S) -> Result<Library, ()> {
  unimplemented!();
}

#[cfg(unix)]
pub unsafe fn open_default<S: AsRef<str>>(lib_name: S) -> Result<Library, ()> {
  match env::var("CUDA_HOME") {
    Ok(prefix) => {
      if !prefix.is_empty() {
        // FIXME FIXME
        Library::new(&format!("{}/lib64/lib{}.so", prefix, lib_name.as_ref()))
      } else {
        Library::new(&format!("lib{}.so", lib_name.as_ref()))
      }
    }
    _ => Library::new(&format!("lib{}.so", lib_name.as_ref()))
  }.map_err(|_| ())
}

#[allow(non_snake_case)]
#[derive(Default)]
pub struct Libcuda {
  pub _inner:                   Option<Library>,
  pub cuGetErrorString:         Option<Symbol<extern "C" fn (CUresult, *mut *const c_char) -> CUresult>>,
  pub cuGetErrorName:           Option<Symbol<extern "C" fn (CUresult, *mut *const c_char) -> CUresult>>,
  pub cuInit:                   Option<Symbol<extern "C" fn (c_uint) -> CUresult>>,
  pub cuDriverGetVersion:       Option<Symbol<extern "C" fn (*mut c_int) -> CUresult>>,
  pub cuDeviceGetCount:         Option<Symbol<extern "C" fn (*mut c_int) -> CUresult>>,
  pub cuDeviceGetName:          Option<Symbol<extern "C" fn (*mut c_char, c_int, CUdevice) -> CUresult>>,
  pub cuDeviceGet:              Option<Symbol<extern "C" fn (*mut CUdevice, c_int) -> CUresult>>,
  pub cuDeviceGetAttribute:     Option<Symbol<extern "C" fn (*mut c_int, CUdevice_attribute, CUdevice) -> CUresult>>,
  pub cuCtxCreate:              Option<Symbol<extern "C" fn (*mut CUcontext, c_uint, CUdevice) -> CUresult>>,
  pub cuCtxDestroy:             Option<Symbol<extern "C" fn (CUcontext) -> CUresult>>,
  pub cuCtxPopCurrent:          Option<Symbol<extern "C" fn (*mut CUcontext) -> CUresult>>,
  pub cuCtxPushCurrent:         Option<Symbol<extern "C" fn (CUcontext) -> CUresult>>,
  pub cuCtxSynchronize:         Option<Symbol<extern "C" fn () -> CUresult>>,
  pub cuMemAlloc:               Option<Symbol<extern "C" fn (*mut CUdeviceptr, usize) -> CUresult>>,
  pub cuMemFree:                Option<Symbol<extern "C" fn (CUdeviceptr) -> CUresult>>,
  pub cuMemcpy:                 Option<Symbol<extern "C" fn (CUdeviceptr, CUdeviceptr, usize) -> CUresult>>,
  pub cuMemcpyHtoD:             Option<Symbol<extern "C" fn (CUdeviceptr, *const c_void, usize) -> CUresult>>,
  pub cuMemcpyDtoH:             Option<Symbol<extern "C" fn (*mut c_void, CUdeviceptr, usize) -> CUresult>>,
  pub cuMemcpyAsync:            Option<Symbol<extern "C" fn (CUdeviceptr, CUdeviceptr, usize, CUstream) -> CUresult>>,
  pub cuMemcpyHtoDAsync:        Option<Symbol<extern "C" fn (CUdeviceptr, *const c_void, usize, CUstream) -> CUresult>>,
  pub cuMemcpyDtoHAsync:        Option<Symbol<extern "C" fn (*mut c_void, CUdeviceptr, usize, CUstream) -> CUresult>>,
  pub cuModuleLoadData:         Option<Symbol<extern "C" fn (*mut CUmodule, *const c_void) -> CUresult>>,
  pub cuModuleUnload:           Option<Symbol<extern "C" fn (CUmodule) -> CUresult>>,
  pub cuModuleGetFunction:      Option<Symbol<extern "C" fn (*mut CUfunction, CUmodule, *const c_char) -> CUresult>>,
  pub cuFuncGetAttribute:       Option<Symbol<extern "C" fn (*mut c_int, CUfunction_attribute, CUfunction) -> CUresult>>,
  pub cuLaunchKernel:           Option<Symbol<extern "C" fn (CUfunction, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, CUstream, *mut *mut c_void, *mut *mut c_void) -> CUresult>>,
  // TODO
}

impl Drop for Libcuda {
  fn drop(&mut self) {
    let inner_library = self._inner.take();
    *self = Default::default();
    if let Some(inner) = inner_library {
      drop(inner);
    }
  }
}

impl Libcuda {
  pub unsafe fn open_default(&mut self) -> Result<(), ()> {
    self._inner = Some(open_default("cuda")?);
    Ok(())
  }

  pub unsafe fn load_symbols(&mut self) -> Result<(), i32> {
    let library = self._inner.as_ref().unwrap();
    self.cuGetErrorString = library.get(b"cuGetErrorString").ok();
    self.cuGetErrorName = library.get(b"cuGetErrorName").ok();
    self.cuInit = library.get(b"cuInit").ok();
    self.cuDriverGetVersion = library.get(b"cuDriverGetVersion").ok();
    self.cuDeviceGetCount = library.get(b"cuDeviceGetCount").ok();
    self.cuDeviceGetName = library.get(b"cuDeviceGetName").ok();
    self.cuDeviceGet = library.get(b"cuDeviceGet").ok();
    self.cuDeviceGetAttribute = library.get(b"cuDeviceGetAttribute").ok();
    self.cuCtxCreate = library.get(b"cuCtxCreate").ok();
    self.cuCtxDestroy = library.get(b"cuCtxDestroy").ok();
    self.cuCtxPopCurrent = library.get(b"cuCtxPopCurrent").ok();
    self.cuCtxPushCurrent = library.get(b"cuCtxPushCurrent").ok();
    self.cuCtxSynchronize = library.get(b"cuCtxSynchronize").ok();
    self.cuMemAlloc = library.get(b"cuMemAlloc").ok();
    self.cuMemFree = library.get(b"cuMemFree").ok();
    self.cuMemcpy = library.get(b"cuMemcpy").ok();
    self.cuMemcpyHtoD = library.get(b"cuMemcpyHtoD").ok();
    self.cuMemcpyDtoH = library.get(b"cuMemcpyDtoH").ok();
    self.cuMemcpyAsync = library.get(b"cuMemcpyAsync").ok();
    self.cuMemcpyHtoDAsync = library.get(b"cuMemcpyHtoDAsync").ok();
    self.cuMemcpyDtoHAsync = library.get(b"cuMemcpyDtoHAsync").ok();
    self.cuModuleLoadData = library.get(b"cuModuleLoadData").ok();
    self.cuModuleUnload = library.get(b"cuModuleUnload").ok();
    self.cuModuleGetFunction = library.get(b"cuModuleGetFunction").ok();
    self.cuFuncGetAttribute = library.get(b"cuFuncGetAttribute").ok();
    self.cuLaunchKernel = library.get(b"cuLaunchKernel").ok();
    // TODO
    //self._check_required().map_err(|_| 2)?;
    Ok(())
  }

  pub unsafe fn try_init(&self) -> Result<(), Option<CUresult>> {
    match self.cuInit.as_ref() {
      Some(func) => {
        match (func)(0) {
          0 => Ok(()),
          e => Err(Some(e))
        }
      }
      None => Err(None),
    }
  }
}

#[allow(non_snake_case)]
#[derive(Default)]
pub struct Libcudart {
  pub _inner:                   Option<Library>,
  pub cudaGetErrorString:       Option<Symbol<extern "C" fn (cudaError_t) -> *const c_char>>,
  pub cudaDriverGetVersion:     Option<Symbol<extern "C" fn (*mut c_int) -> cudaError_t>>,
  pub cudaRuntimeGetVersion:    Option<Symbol<extern "C" fn (*mut c_int) -> cudaError_t>>,
  pub cudaGetDeviceCount:       Option<Symbol<extern "C" fn (*mut c_int) -> cudaError_t>>,
  pub cudaDeviceCanAccessPeer:  Option<Symbol<extern "C" fn (*mut c_int, c_int, c_int) -> cudaError_t>>,
  pub cudaGetDevice:            Option<Symbol<extern "C" fn (*mut c_int) -> cudaError_t>>,
  pub cudaSetDevice:            Option<Symbol<extern "C" fn (c_int) -> cudaError_t>>,
  pub cudaDeviceDisablePeerAccess: Option<Symbol<extern "C" fn (c_int, c_uint) -> cudaError_t>>,
  pub cudaDeviceEnablePeerAccess: Option<Symbol<extern "C" fn (c_int, c_uint) -> cudaError_t>>,
  pub cudaDeviceReset:          Option<Symbol<extern "C" fn () -> cudaError_t>>,
  pub cudaDeviceSynchronize:    Option<Symbol<extern "C" fn () -> cudaError_t>>,
  pub cudaMemGetInfo:           Option<Symbol<extern "C" fn (*mut usize, *mut usize) -> cudaError_t>>,
  pub cudaMalloc:               Option<Symbol<extern "C" fn (*mut *mut c_void, usize) -> cudaError_t>>,
  pub cudaMallocHost:           Option<Symbol<extern "C" fn (*mut *mut c_void, usize) -> cudaError_t>>,
  pub cudaFree:                 Option<Symbol<extern "C" fn (*mut c_void) -> cudaError_t>>,
  pub cudaFreeHost:             Option<Symbol<extern "C" fn (*mut c_void) -> cudaError_t>>,
  pub cudaMemcpyAsync:          Option<Symbol<extern "C" fn (*mut c_void, *const c_void, usize, c_int, cudaStream_t) -> cudaError_t>>,
  pub cudaMemcpyPeerAsync:      Option<Symbol<extern "C" fn (*mut c_void, c_int, *const c_void, c_int, usize, cudaStream_t) -> cudaError_t>>,
  pub cudaEventCreate:          Option<Symbol<extern "C" fn (*mut cudaEvent_t) -> cudaError_t>>,
  pub cudaEventCreateWithFlags: Option<Symbol<extern "C" fn (*mut cudaEvent_t, c_uint) -> cudaError_t>>,
  pub cudaEventDestroy:         Option<Symbol<extern "C" fn (cudaEvent_t) -> cudaError_t>>,
  pub cudaEventElapsedTime:     Option<Symbol<extern "C" fn (*mut f32, cudaEvent_t, cudaEvent_t) -> cudaError_t>>,
  pub cudaEventQuery:           Option<Symbol<extern "C" fn (cudaEvent_t) -> cudaError_t>>,
  pub cudaEventRecord:          Option<Symbol<extern "C" fn (cudaEvent_t, cudaStream_t) -> cudaError_t>>,
  pub cudaEventRecordWithFlags: Option<Symbol<extern "C" fn (cudaEvent_t, cudaStream_t, c_uint) -> cudaError_t>>,
  pub cudaEventSynchronize:     Option<Symbol<extern "C" fn (cudaEvent_t) -> cudaError_t>>,
  pub cudaStreamCreate:         Option<Symbol<extern "C" fn (*mut cudaStream_t) -> cudaError_t>>,
  pub cudaStreamCreateWithFlags: Option<Symbol<extern "C" fn (*mut cudaStream_t, c_uint) -> cudaError_t>>,
  pub cudaStreamDestroy:        Option<Symbol<extern "C" fn (cudaStream_t) -> cudaError_t>>,
  pub cudaStreamSynchronize:    Option<Symbol<extern "C" fn (cudaStream_t) -> cudaError_t>>,
  pub cudaStreamWaitEvent:      Option<Symbol<extern "C" fn (cudaStream_t, cudaEvent_t, c_uint) -> cudaError_t>>,
  // TODO
}

impl Drop for Libcudart {
  fn drop(&mut self) {
    let inner_library = self._inner.take();
    *self = Default::default();
    if let Some(inner) = inner_library {
      drop(inner);
    }
  }
}

impl Libcudart {
  pub unsafe fn open_default(&mut self) -> Result<(), ()> {
    self._inner = Some(open_default("cudart")?);
    Ok(())
  }

  pub unsafe fn load_symbols(&mut self) -> Result<(), i32> {
    let library = self._inner.as_ref().unwrap();
    self.cudaGetErrorString = library.get(b"cudaGetErrorString").ok();
    self.cudaDriverGetVersion = library.get(b"cudaDriverGetVersion").ok();
    self.cudaRuntimeGetVersion = library.get(b"cudaRuntimeGetVersion").ok();
    self.cudaGetDeviceCount = library.get(b"cudaGetDeviceCount").ok();
    self.cudaDeviceCanAccessPeer = library.get(b"cudaDeviceCanAccessPeer").ok();
    self.cudaGetDevice = library.get(b"cudaGetDevice").ok();
    self.cudaSetDevice = library.get(b"cudaSetDevice").ok();
    self.cudaDeviceDisablePeerAccess = library.get(b"cudaDeviceDisablePeerAccess").ok();
    self.cudaDeviceEnablePeerAccess = library.get(b"cudaDeviceEnablePeerAccess").ok();
    self.cudaDeviceReset = library.get(b"cudaDeviceReset").ok();
    self.cudaDeviceSynchronize = library.get(b"cudaDeviceSynchronize").ok();
    self.cudaMemGetInfo = library.get(b"cudaMemGetInfo").ok();
    self.cudaMalloc = library.get(b"cudaMalloc").ok();
    self.cudaMallocHost = library.get(b"cudaMallocHost").ok();
    self.cudaFree = library.get(b"cudaFree").ok();
    self.cudaFreeHost = library.get(b"cudaFreeHost").ok();
    self.cudaMemcpyAsync = library.get(b"cudaMemcpyAsync").ok();
    self.cudaMemcpyPeerAsync = library.get(b"cudaMemcpyPeerAsync").ok();
    self.cudaEventCreate = library.get(b"cudaEventCreate").ok();
    self.cudaEventCreateWithFlags = library.get(b"cudaEventCreateWithFlags").ok();
    self.cudaEventDestroy = library.get(b"cudaEventDestroy").ok();
    self.cudaEventElapsedTime = library.get(b"cudaEventElapsedTime").ok();
    self.cudaEventQuery = library.get(b"cudaEventQuery").ok();
    self.cudaEventRecord = library.get(b"cudaEventRecord").ok();
    self.cudaEventRecordWithFlags = library.get(b"cudaEventRecordWithFlags").ok();
    self.cudaEventSynchronize = library.get(b"cudaEventSynchronize").ok();
    self.cudaStreamCreate = library.get(b"cudaStreamCreate").ok();
    self.cudaStreamCreateWithFlags = library.get(b"cudaStreamCreateWithFlags").ok();
    self.cudaStreamDestroy = library.get(b"cudaStreamDestroy").ok();
    self.cudaStreamSynchronize = library.get(b"cudaStreamSynchronize").ok();
    self.cudaStreamWaitEvent = library.get(b"cudaStreamWaitEvent").ok();
    // TODO
    self._check_required().map_err(|_| 2)?;
    Ok(())
  }

  fn _check_required(&self) -> Result<(), ()> {
    self.cudaGetErrorString.as_ref().ok_or(())?;
    self.cudaDriverGetVersion.as_ref().ok_or(())?;
    self.cudaRuntimeGetVersion.as_ref().ok_or(())?;
    self.cudaGetDeviceCount.as_ref().ok_or(())?;
    self.cudaGetDevice.as_ref().ok_or(())?;
    self.cudaSetDevice.as_ref().ok_or(())?;
    self.cudaMemGetInfo.as_ref().ok_or(())?;
    self.cudaMalloc.as_ref().ok_or(())?;
    self.cudaMallocHost.as_ref().ok_or(())?;
    self.cudaFree.as_ref().ok_or(())?;
    self.cudaFreeHost.as_ref().ok_or(())?;
    self.cudaMemcpyAsync.as_ref().ok_or(())?;
    self.cudaEventCreateWithFlags.as_ref().ok_or(())?;
    self.cudaEventDestroy.as_ref().ok_or(())?;
    self.cudaEventRecordWithFlags.as_ref().ok_or(())?;
    self.cudaEventSynchronize.as_ref().ok_or(())?;
    self.cudaStreamCreateWithFlags.as_ref().ok_or(())?;
    self.cudaStreamDestroy.as_ref().ok_or(())?;
    self.cudaStreamSynchronize.as_ref().ok_or(())?;
    self.cudaStreamWaitEvent.as_ref().ok_or(())?;
    // TODO
    Ok(())
  }
}

#[allow(non_snake_case)]
#[derive(Default)]
pub struct Libnvrtc {
  pub _inner:                   Option<Library>,
  pub nvrtcGetErrorString:      Option<Symbol<extern "C" fn (nvrtcResult) -> *const c_char>>,
  pub nvrtcGetVersion:          Option<Symbol<extern "C" fn (*mut c_int, *mut c_int) -> nvrtcResult>>,
  pub nvrtcGetNumSupportedArchs: Option<Symbol<extern "C" fn (*mut c_int) -> nvrtcResult>>,
  pub nvrtcGetSupportedArchs:   Option<Symbol<extern "C" fn (*mut c_int) -> nvrtcResult>>,
  pub nvrtcCreateProgram:       Option<Symbol<extern "C" fn (*mut nvrtcProgram, *const c_char, *const c_char, c_int, *const *const c_char, *const *const c_char) -> nvrtcResult>>,
  pub nvrtcDestroyProgram:      Option<Symbol<extern "C" fn (*mut nvrtcProgram) -> nvrtcResult>>,
  pub nvrtcCompileProgram:      Option<Symbol<extern "C" fn (nvrtcProgram, c_int, *const *const c_char) -> nvrtcResult>>,
  pub nvrtcGetProgramLogSize:   Option<Symbol<extern "C" fn (nvrtcProgram, *mut usize) -> nvrtcResult>>,
  pub nvrtcGetProgramLog:       Option<Symbol<extern "C" fn (nvrtcProgram, *mut c_char) -> nvrtcResult>>,
  pub nvrtcGetPTXSize:          Option<Symbol<extern "C" fn (nvrtcProgram, *mut usize) -> nvrtcResult>>,
  pub nvrtcGetPTX:              Option<Symbol<extern "C" fn (nvrtcProgram, *mut c_char) -> nvrtcResult>>,
  // TODO
}

impl Drop for Libnvrtc {
  fn drop(&mut self) {
    let inner_library = self._inner.take();
    *self = Default::default();
    if let Some(inner) = inner_library {
      drop(inner);
    }
  }
}

impl Libnvrtc {
  pub unsafe fn open_default(&mut self) -> Result<(), ()> {
    self._inner = Some(open_default("nvrtc")?);
    Ok(())
  }

  pub unsafe fn load_symbols(&mut self) -> Result<(), i32> {
    let library = self._inner.as_ref().unwrap();
    self.nvrtcGetErrorString = library.get(b"nvrtcGetErrorString").ok();
    self.nvrtcCreateProgram = library.get(b"nvrtcCreateProgram").ok();
    self.nvrtcDestroyProgram = library.get(b"nvrtcDestroyProgram").ok();
    self.nvrtcCompileProgram = library.get(b"nvrtcCompileProgram").ok();
    self.nvrtcGetProgramLogSize = library.get(b"nvrtcGetProgramLogSize").ok();
    self.nvrtcGetProgramLog = library.get(b"nvrtcGetProgramLog").ok();
    self.nvrtcGetPTXSize = library.get(b"nvrtcGetPTXSize").ok();
    self.nvrtcGetPTX = library.get(b"nvrtcGetPTX").ok();
    // TODO
    //self._check_required().map_err(|_| 2)?;
    Ok(())
  }
}

#[allow(non_snake_case)]
#[derive(Default)]
pub struct Libcublas {
  pub _inner:                   Option<Library>,
  pub cublasCreate_v2:          Option<Symbol<extern "C" fn (*mut cublasHandle_t) -> cublasStatus_t>>,
  pub cublasDestroy_v2:         Option<Symbol<extern "C" fn (cublasHandle_t) -> cublasStatus_t>>,
  pub cublasGetVersion_v2:      Option<Symbol<extern "C" fn (cublasHandle_t, *mut c_int) -> cublasStatus_t>>,
  pub cublasGetStream_v2:       Option<Symbol<extern "C" fn (cublasHandle_t, *mut cudaStream_t) -> cublasStatus_t>>,
  pub cublasSetStream_v2:       Option<Symbol<extern "C" fn (cublasHandle_t, cudaStream_t) -> cublasStatus_t>>,
  pub cublasSetPointerMode_v2:  Option<Symbol<extern "C" fn (cublasHandle_t, cublasPointerMode_t) -> cublasStatus_t>>,
  pub cublasSetAtomicsMode_v2:  Option<Symbol<extern "C" fn (cublasHandle_t, cublasAtomicsMode_t) -> cublasStatus_t>>,
  pub cublasSetMathMode:        Option<Symbol<extern "C" fn (cublasHandle_t, cublasMath_t) -> cublasStatus_t>>,
  pub cublasGemmEx:             Option<Symbol<extern "C" fn (
                                    cublasHandle_t,
                                    cublasOperation_t,
                                    cublasOperation_t,
                                    c_int, c_int, c_int,
                                    *const c_void,
                                    *const c_void, cudaDataType_t, c_int,
                                    *const c_void, cudaDataType_t, c_int,
                                    *const c_void,
                                    *mut c_void, cudaDataType_t, c_int,
                                    cublasComputeType_t,
                                    cublasGemmAlgo_t,
                                ) -> cublasStatus_t>>,
  pub cublasSgemmEx:            Option<Symbol<extern "C" fn (
                                    cublasHandle_t,
                                    cublasOperation_t,
                                    cublasOperation_t,
                                    c_int, c_int, c_int,
                                    *const c_void,
                                    *const c_void, cudaDataType_t, c_int,
                                    *const c_void, cudaDataType_t, c_int,
                                    *const c_void,
                                    *mut c_void, cudaDataType_t, c_int,
                                ) -> cublasStatus_t>>,
  // TODO
}

impl Drop for Libcublas {
  fn drop(&mut self) {
    let inner_library = self._inner.take();
    *self = Default::default();
    if let Some(inner) = inner_library {
      drop(inner);
    }
  }
}

impl Libcublas {
  pub unsafe fn open_default(&mut self) -> Result<(), ()> {
    self._inner = Some(open_default("cublas")?);
    Ok(())
  }

  pub unsafe fn load_symbols(&mut self) -> Result<(), ()> {
    let library = self._inner.as_ref().unwrap();
    self.cublasCreate_v2 = library.get(b"cublasCreate_v2").ok();
    self.cublasDestroy_v2 = library.get(b"cublasDestroy_v2").ok();
    self.cublasGetVersion_v2 = library.get(b"cublasGetVersion_v2").ok();
    self.cublasSetStream_v2 = library.get(b"cublasSetStream_v2").ok();
    self.cublasSetPointerMode_v2 = library.get(b"cublasSetPointerMode_v2").ok();
    self.cublasSetAtomicsMode_v2 = library.get(b"cublasSetAtomicsMode_v2").ok();
    self.cublasSetMathMode = library.get(b"cublasSetMathMode").ok();
    self.cublasSgemmEx = library.get(b"cublasSgemmEx").ok();
    self.cublasGemmEx = library.get(b"cublasGemmEx").ok();
    // TODO
    self._check_required()?;
    Ok(())
  }

  fn _check_required(&self) -> Result<(), ()> {
    // FIXME
    Ok(())
  }
}

#[allow(non_snake_case)]
#[derive(Default)]
pub struct Libcusolver {
  pub _inner:                   Option<Library>,
  pub cusolverDnCreate:         Option<Symbol<extern "C" fn (*mut cusolverDnHandle_t) -> cusolverStatus_t>>,
  pub cusolverDnDestroy:        Option<Symbol<extern "C" fn (cusolverDnHandle_t) -> cusolverStatus_t>>,
  pub cusolverDnGetStream:      Option<Symbol<extern "C" fn (cusolverDnHandle_t, *mut cudaStream_t) -> cusolverStatus_t>>,
  pub cusolverDnSetStream:      Option<Symbol<extern "C" fn (cusolverDnHandle_t, cudaStream_t) -> cusolverStatus_t>>,
  pub cusolverDnSSgels_bufferSize:      Option<Symbol<extern "C" fn (
                                            cusolverDnHandle_t,
                                            c_int, c_int, c_int,
                                            *mut f32, c_int,
                                            *mut f32, c_int,
                                            *mut f32, c_int,
                                            *mut c_void, usize,
                                        ) -> cusolverStatus_t>>,
  pub cusolverDnSSgels:                 Option<Symbol<extern "C" fn (
                                            cusolverDnHandle_t,
                                            c_int, c_int, c_int,
                                            *mut f32, c_int,
                                            *mut f32, c_int,
                                            *mut f32, c_int,
                                            *mut c_void, usize,
                                            *mut c_int,
                                            *mut c_int,
                                        ) -> cusolverStatus_t>>,
  pub cusolverDnSHgels_bufferSize:      Option<Symbol<extern "C" fn (
                                            cusolverDnHandle_t,
                                            c_int, c_int, c_int,
                                            *mut f32, c_int,
                                            *mut f32, c_int,
                                            *mut f32, c_int,
                                            *mut c_void, usize,
                                        ) -> cusolverStatus_t>>,
  pub cusolverDnSHgels:                 Option<Symbol<extern "C" fn (
                                            cusolverDnHandle_t,
                                            c_int, c_int, c_int,
                                            *mut f32, c_int,
                                            *mut f32, c_int,
                                            *mut f32, c_int,
                                            *mut c_void, usize,
                                            *mut c_int,
                                            *mut c_int,
                                        ) -> cusolverStatus_t>>,
  pub cusolverDnSXgels_bufferSize:      Option<Symbol<extern "C" fn (
                                            cusolverDnHandle_t,
                                            c_int, c_int, c_int,
                                            *mut f32, c_int,
                                            *mut f32, c_int,
                                            *mut f32, c_int,
                                            *mut c_void, usize,
                                        ) -> cusolverStatus_t>>,
  pub cusolverDnSXgels:                 Option<Symbol<extern "C" fn (
                                            cusolverDnHandle_t,
                                            c_int, c_int, c_int,
                                            *mut f32, c_int,
                                            *mut f32, c_int,
                                            *mut f32, c_int,
                                            *mut c_void, usize,
                                            *mut c_int,
                                            *mut c_int,
                                        ) -> cusolverStatus_t>>,
}

impl Drop for Libcusolver {
  fn drop(&mut self) {
    let inner_library = self._inner.take();
    *self = Default::default();
    if let Some(inner) = inner_library {
      drop(inner);
    }
  }
}

impl Libcusolver {
  pub unsafe fn open_default(&mut self) -> Result<(), ()> {
    self._inner = Some(open_default("cusolver")?);
    Ok(())
  }

  pub unsafe fn load_symbols(&mut self) -> Result<(), ()> {
    let library = self._inner.as_ref().unwrap();
    self.cusolverDnCreate = library.get(b"cusolverDnCreate").ok();
    self.cusolverDnDestroy = library.get(b"cusolverDnDestroy").ok();
    self.cusolverDnGetStream = library.get(b"cusolverDnGetStream").ok();
    self.cusolverDnSetStream = library.get(b"cusolverDnSetStream").ok();
    self.cusolverDnSSgels_bufferSize = library.get(b"cusolverDnSSgels_bufferSize").ok();
    self.cusolverDnSSgels = library.get(b"cusolverDnSSgels").ok();
    self.cusolverDnSHgels_bufferSize = library.get(b"cusolverDnSHgels_bufferSize").ok();
    self.cusolverDnSHgels = library.get(b"cusolverDnSHgels").ok();
    self.cusolverDnSXgels_bufferSize = library.get(b"cusolverDnSXgels_bufferSize").ok();
    self.cusolverDnSXgels = library.get(b"cusolverDnSXgels").ok();
    self._check_required()?;
    Ok(())
  }

  fn _check_required(&self) -> Result<(), ()> {
    // FIXME
    Ok(())
  }
}
