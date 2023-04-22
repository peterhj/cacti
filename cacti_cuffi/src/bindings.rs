use crate::types::*;

use libc::{c_void, c_char, c_int, c_uint};
use libloading::os::unix::{Library, Symbol};

#[allow(non_snake_case)]
#[derive(Default)]
pub struct Libcudart {
  inner_library:                Option<Library>,
  pub cudaGetErrorString:       Option<Symbol<extern "C" fn (c_int) -> *const c_char>>,
  pub cudaDriverGetVersion:     Option<Symbol<extern "C" fn (*mut c_int) -> cudaError>>,
  pub cudaRuntimeGetVersion:    Option<Symbol<extern "C" fn (*mut c_int) -> cudaError>>,
  pub cudaGetDeviceCount:       Option<Symbol<extern "C" fn (*mut c_int) -> cudaError>>,
  pub cudaDeviceCanAccessPeer:  Option<Symbol<extern "C" fn (*mut c_int, c_int, c_int) -> cudaError>>,
  pub cudaGetDevice:            Option<Symbol<extern "C" fn (*mut c_int) -> cudaError>>,
  pub cudaSetDevice:            Option<Symbol<extern "C" fn (c_int) -> cudaError>>,
  pub cudaDeviceDisablePeerAccess:      Option<Symbol<extern "C" fn (c_int, c_uint) -> cudaError>>,
  pub cudaDeviceEnablePeerAccess:       Option<Symbol<extern "C" fn (c_int, c_uint) -> cudaError>>,
  pub cudaDeviceReset:          Option<Symbol<extern "C" fn () -> cudaError>>,
  pub cudaDeviceSynchronize:    Option<Symbol<extern "C" fn () -> cudaError>>,
  pub cudaMemGetInfo:           Option<Symbol<extern "C" fn (*mut usize, *mut usize) -> cudaError>>,
  pub cudaMalloc:               Option<Symbol<extern "C" fn (*mut *mut c_void, usize) -> cudaError>>,
  pub cudaMallocHost:           Option<Symbol<extern "C" fn (*mut *mut c_void, usize) -> cudaError>>,
  pub cudaFree:                 Option<Symbol<extern "C" fn (*mut c_void) -> cudaError>>,
  pub cudaFreeHost:             Option<Symbol<extern "C" fn (*mut c_void) -> cudaError>>,
  pub cudaMemcpyAsync:          Option<Symbol<extern "C" fn (*mut c_void, *const c_void, usize, c_int, cudaStream_t) -> cudaError>>,
  pub cudaMemcpyPeerAsync:      Option<Symbol<extern "C" fn (*mut c_void, c_int, *const c_void, c_int, usize, cudaStream_t) -> cudaError>>,
  pub cudaEventCreateWithFlags: Option<Symbol<extern "C" fn (*mut cudaEvent_t, c_uint) -> cudaError>>,
  pub cudaEventDestroy:         Option<Symbol<extern "C" fn (cudaEvent_t) -> cudaError>>,
  pub cudaEventQuery:           Option<Symbol<extern "C" fn (cudaEvent_t) -> cudaError>>,
  pub cudaEventRecord:          Option<Symbol<extern "C" fn (cudaEvent_t, cudaStream_t) -> cudaError>>,
  pub cudaEventRecordWithFlags: Option<Symbol<extern "C" fn (cudaEvent_t, cudaStream_t, c_uint) -> cudaError>>,
  pub cudaEventSynchronize:     Option<Symbol<extern "C" fn (cudaEvent_t) -> cudaError>>,
  pub cudaStreamCreate:         Option<Symbol<extern "C" fn (*mut cudaStream_t) -> cudaError>>,
  pub cudaStreamCreateWithFlags:        Option<Symbol<extern "C" fn (*mut cudaStream_t, c_uint) -> cudaError>>,
  pub cudaStreamDestroy:        Option<Symbol<extern "C" fn (cudaStream_t) -> cudaError>>,
  pub cudaStreamSynchronize:    Option<Symbol<extern "C" fn (cudaStream_t) -> cudaError>>,
  pub cudaStreamWaitEvent:      Option<Symbol<extern "C" fn (cudaStream_t, cudaEvent_t, c_uint) -> cudaError>>,
  // TODO
}

impl Drop for Libcudart {
  fn drop(&mut self) {
    let library = self.inner_library.take();
    assert!(library.is_some());
    *self = Default::default();
    drop(library.unwrap());
  }
}

impl Libcudart {
  //pub fn open_default() -> Result<Libcudart, ()> {}
  pub unsafe fn load_default(&mut self) -> Result<(), ()> {
    let library = Library::new("cudart").map_err(|_| ())?;
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
    self.cudaEventCreateWithFlags = library.get(b"cudaEventCreateWithFlags").ok();
    self.cudaEventDestroy = library.get(b"cudaEventDestroy").ok();
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
    self.inner_library = library.into();
    self._check_required()?;
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
    self.cudaFree.as_ref().ok_or(())?;
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
pub struct Libcublas {
  inner_library:            Option<Library>,
  pub cublasCreate_v2:      Option<Symbol<extern "C" fn (*mut cublasHandle_t) -> cublasStatus_t>>,
  pub cublasDestroy_v2:     Option<Symbol<extern "C" fn (cublasHandle_t) -> cublasStatus_t>>,
  pub cublasGetVersion_v2:  Option<Symbol<extern "C" fn (cublasHandle_t, *mut c_int) -> cublasStatus_t>>,
  pub cublasGetStream_v2:   Option<Symbol<extern "C" fn (cublasHandle_t, *mut cudaStream_t) -> cublasStatus_t>>,
  pub cublasSetStream_v2:   Option<Symbol<extern "C" fn (cublasHandle_t, cudaStream_t) -> cublasStatus_t>>,
  pub cublasSetPointerMode_v2:      Option<Symbol<extern "C" fn (cublasHandle_t, cublasPointerMode_t) -> cublasStatus_t>>,
  pub cublasSetAtomicsMode_v2:      Option<Symbol<extern "C" fn (cublasHandle_t, cublasAtomicsMode_t) -> cublasStatus_t>>,
  pub cublasSetMathMode:    Option<Symbol<extern "C" fn (cublasHandle_t, cublasMath_t) -> cublasStatus_t>>,
  pub cublasGemmEx:         Option<Symbol<extern "C" fn (
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
  pub cublasSgemmEx:        Option<Symbol<extern "C" fn (
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
    let library = self.inner_library.take();
    assert!(library.is_some());
    *self = Default::default();
    drop(library.unwrap());
  }
}

impl Libcublas {
  pub unsafe fn load_default(&mut self) -> Result<(), ()> {
    let library = Library::new("cublas").map_err(|_| ())?;
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
    self.inner_library = library.into();
    self._check_required()?;
    Ok(())
  }

  fn _check_required(&self) -> Result<(), ()> {
    // FIXME
    Ok(())
  }
}
