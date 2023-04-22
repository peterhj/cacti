#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]

use libc::{c_int, c_uint};

pub type cudaError = c_int;

pub const cudaSuccess: cudaError = 0;
pub const cudaMemoryAllocation: cudaError = 2;
pub const cudaInitializationError: cudaError = 3;
pub const cudaCudartUnloading: cudaError = 4;

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

#[repr(C)]
pub struct CUevent_st([u8; 0]);

#[repr(C)]
pub struct CUstream_st([u8; 0]);

pub type cudaEvent_t = *mut CUevent_st;
pub type cudaStream_t = *mut CUstream_st;

pub type cublasStatus_t = c_int;

pub const CUBLAS_STATUS_SUCCESS: cublasStatus_t = 0;

pub type cublasPointerMode_t = c_int;

pub const CUBLAS_POINTER_MODE_HOST: cublasPointerMode_t = 0;
pub const CUBLAS_POINTER_MODE_DEVICE: cublasPointerMode_t = 1;

pub type cublasAtomicsMode_t = c_int;

pub const CUBLAS_ATOMICS_MODE_NOT_ALLOWED: cublasAtomicsMode_t = 0;
pub const CUBLAS_ATOMICS_MODE_ALLOWED: cublasAtomicsMode_t = 1;

pub type cublasGemmAlgo_t = c_int;

// FIXME
pub const CUBLAS_GEMM_DEFAULT: cublasGemmAlgo_t = 0;

pub type cublasMath_t = c_uint;

// FIXME
pub const CUBLAS_DEFAULT_MATH: cublasMath_t = 0;
//pub const CUBLAS_TENSOR_OP_MATH: cublasMath_t = 1;
//pub const CUBLAS_PEDANTIC_MATH: cublasMath_t = _;
//pub const CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION: cublasMath_t = _;

pub type cublasOperation_t = c_int;

// FIXME
pub const CUBLAS_OP_N: cublasOperation_t = 0;
pub const CUBLAS_OP_T: cublasOperation_t = 1;

pub type cublasComputeType_t = c_int;

// FIXME
//pub const CUBLAS_COMPUTE_16F: cublasComputeType_t = _;
//pub const CUBLAS_COMPUTE_32F: cublasComputeType_t = _;

// FIXME
#[repr(C)]
pub struct cublasHandle_st([u8; 0]);

// FIXME
pub type cublasHandle_t = *mut cublasHandle_st;
