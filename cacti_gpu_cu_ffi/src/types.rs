#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]

use libc::{c_int, c_uint};

pub type CUresult = c_int;

pub const CUDA_SUCCESS: CUresult = 0;
pub const CUDA_ERROR_OUT_OF_MEMORY: CUresult = 2;
pub const CUDA_ERROR_DEINITIALIZED: CUresult = 4;
pub const CUDA_ERROR_INVALID_CONTEXT: CUresult = 201;
pub const CUDA_ERROR_ILLEGAL_ADDRESS: CUresult = 700;

pub type CUdevice = c_int;
pub type CUdevice_attribute = c_int;
pub type CUfunction_attribute = c_int;

#[cfg(target_pointer_width = "32")]
pub type CUdeviceptr = u32;
#[cfg(target_pointer_width = "64")]
pub type CUdeviceptr = u64;

pub type CUctx_flags = c_uint;

pub const CU_CTX_SCHED_AUTO: CUctx_flags = 0;
pub const CU_CTX_SCHED_SPIN: CUctx_flags = 1;
pub const CU_CTX_SCHED_YIELD: CUctx_flags = 2;
pub const CU_CTX_SCHED_BLOCKING_SYNC: CUctx_flags = 4;

pub const CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK: CUdevice_attribute = 1;
pub const CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X: CUdevice_attribute = 2;
pub const CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X: CUdevice_attribute = 5;
pub const CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK: CUdevice_attribute = 8;
pub const CU_DEVICE_ATTRIBUTE_WARP_SIZE: CUdevice_attribute = 10;
pub const CU_DEVICE_ATTRIBUTE_MAX_PITCH: CUdevice_attribute = 11;
pub const CU_DEVICE_ATTRIBUTE_CLOCK_RATE: CUdevice_attribute = 13;
pub const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: CUdevice_attribute = 16;
pub const CU_DEVICE_ATTRIBUTE_INTEGRATED: CUdevice_attribute = 18;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_MODE: CUdevice_attribute = 20;
pub const CU_DEVICE_ATTRIBUTE_PCI_BUS_ID: CUdevice_attribute = 33;
pub const CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID: CUdevice_attribute = 34;
pub const CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE: CUdevice_attribute = 36;
pub const CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH: CUdevice_attribute = 37;
pub const CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR: CUdevice_attribute = 39;
pub const CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING: CUdevice_attribute = 41;
pub const CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID: CUdevice_attribute = 50;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: CUdevice_attribute = 75;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: CUdevice_attribute = 76;
pub const CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY: CUdevice_attribute = 83;

pub const CU_COMPUTEMODE_DEFAULT: c_int = 0;
pub const CU_COMPUTEMODE_PROHIBITED: c_int = 2;
pub const CU_COMPUTEMODE_EXCLUSIVE_PROCESS: c_int = 3;

#[repr(C)]
pub struct CUctx_st([u8; 0]);

#[repr(C)]
pub struct CUmod_st([u8; 0]);

#[repr(C)]
pub struct CUfunc_st([u8; 0]);

#[repr(C)]
pub struct CUevent_st([u8; 0]);

#[repr(C)]
pub struct CUstream_st([u8; 0]);

pub type CUcontext = *mut CUctx_st;
pub type CUmodule = *mut CUmod_st;
pub type CUfunction = *mut CUfunc_st;
pub type CUevent = *mut CUevent_st;
pub type CUstream = *mut CUstream_st;

pub type cudaError_t = c_int;

pub const cudaSuccess: cudaError_t = 0;
pub const cudaErrorInvalidValue: cudaError_t = 1;
pub const cudaErrorMemoryAllocation: cudaError_t = 2;
pub const cudaErrorInitializationError: cudaError_t = 3;
pub const cudaErrorCudartUnloading: cudaError_t = 4;
pub const cudaErrorNotReady: cudaError_t = 600;

pub type cudaMemcpyKind = c_int;

pub const cudaMemcpyHostToHost: cudaMemcpyKind = 0;
pub const cudaMemcpyHostToDevice: cudaMemcpyKind = 1;
pub const cudaMemcpyDeviceToHost: cudaMemcpyKind = 2;
pub const cudaMemcpyDeviceToDevice: cudaMemcpyKind = 3;
pub const cudaMemcpyDefault: cudaMemcpyKind = 4;

pub type cudaDataType_t = c_int;

pub const CUDA_R_16F: cudaDataType_t = 2;
pub const CUDA_R_16BF: cudaDataType_t = 14;
pub const CUDA_R_32F: cudaDataType_t = 0;
pub const CUDA_R_64F: cudaDataType_t = 1;
// TODO

pub const cudaEventDefault: c_uint = 0;
pub const cudaEventBlocking: c_uint = 1;
pub const cudaEventDisableTiming: c_uint = 2;

pub const cudaStreamDefault: c_uint = 0;
pub const cudaStreamNonblocking: c_uint = 1;

pub type cudaEvent_t = CUevent;
pub type cudaStream_t = CUstream;

pub type cublasStatus_t = c_int;

pub type nvrtcResult = c_int;

#[repr(C)]
pub struct _nvrtcProgram([u8; 0]);

pub type nvrtcProgram = *mut _nvrtcProgram;

pub const CUBLAS_STATUS_SUCCESS: cublasStatus_t = 0;

pub type cublasOperation_t = c_int;

pub const CUBLAS_OP_N: cublasOperation_t = 0;
pub const CUBLAS_OP_T: cublasOperation_t = 1;

pub type cublasPointerMode_t = c_int;

pub const CUBLAS_POINTER_MODE_HOST: cublasPointerMode_t = 0;
pub const CUBLAS_POINTER_MODE_DEVICE: cublasPointerMode_t = 1;

pub type cublasAtomicsMode_t = c_int;

pub const CUBLAS_ATOMICS_NOT_ALLOWED: cublasAtomicsMode_t = 0;
pub const CUBLAS_ATOMICS_ALLOWED: cublasAtomicsMode_t = 1;

pub type cublasGemmAlgo_t = c_int;

pub const CUBLAS_GEMM_DEFAULT: cublasGemmAlgo_t = -1;

pub type cublasMath_t = c_uint;

pub const CUBLAS_DEFAULT_MATH: cublasMath_t = 0;
pub const CUBLAS_TENSOR_OP_MATH: cublasMath_t = 1;
pub const CUBLAS_PEDANTIC_MATH: cublasMath_t = 2;
pub const CUBLAS_TF32_TENSOR_OP_MATH: cublasMath_t = 3;
pub const CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION: cublasMath_t = 0x10;

pub type cublasComputeType_t = c_int;

pub const CUBLAS_COMPUTE_16F: cublasComputeType_t = 64;
pub const CUBLAS_COMPUTE_32F: cublasComputeType_t = 68;

#[repr(C)]
pub struct cublasContext([u8; 0]);

pub type cublasHandle_t = *mut cublasContext;

pub type cusolverStatus_t = c_int;

pub const CUSOLVER_STATUS_SUCCESS: cusolverStatus_t = 0;

#[repr(C)]
pub struct cusolverDnContext([u8; 0]);

pub type cusolverDnHandle_t = *mut cusolverDnContext;
