#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]

use libc::{c_int, c_uint};

pub type cudaError = c_int;

pub const cudaSuccess: cudaError = 0;
pub const cudaErrorInvalidValue: cudaError = 1;
pub const cudaErrorMemoryAllocation: cudaError = 2;
pub const cudaErrorInitializationError: cudaError = 3;
pub const cudaErrorCudartUnloading: cudaError = 4;
pub const cudaErrorNotReady: cudaError = 600;

pub type cudaMemcpyKind = c_int;

pub const cudaMemcpyHostToHost: cudaMemcpyKind = 0;
pub const cudaMemcpyHostToDevice: cudaMemcpyKind = 1;
pub const cudaMemcpyDeviceToHost: cudaMemcpyKind = 2;
pub const cudaMemcpyDeviceToDevice: cudaMemcpyKind = 3;
pub const cudaMemcpyDefault: cudaMemcpyKind = 4;

pub type cudaDataType_t = c_int;

pub const cudaEventDefault: c_uint = 0;
pub const cudaEventBlocking: c_uint = 1;
pub const cudaEventDisableTiming: c_uint = 2;

pub const cudaStreamDefault: c_uint = 0;
pub const cudaStreamNonblocking: c_uint = 1;

pub type CUdevice = c_int;

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

pub type cudaEvent_t = CUevent;
pub type cudaStream_t = CUstream;

pub type cublasStatus_t = c_int;

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
