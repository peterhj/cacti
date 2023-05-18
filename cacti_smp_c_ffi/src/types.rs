#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]

use libc::{c_int};

pub type CBLAS_INDEX = usize;

pub type CBLAS_ORDER = c_int;

pub const CblasRowMajor: CBLAS_ORDER = 101;
pub const CblasColMajor: CBLAS_ORDER = 102;

pub type CBLAS_TRANSPOSE = c_int;

pub const CblasNoTrans: CBLAS_TRANSPOSE = 111;
pub const CblasTrans: CBLAS_TRANSPOSE = 112;
pub const CblasConjTrans: CBLAS_TRANSPOSE = 113;
pub const CblasConjNoTrans: CBLAS_TRANSPOSE = 114;

pub type CBLAS_UPLO = c_int;

pub const CblasUpper: CBLAS_UPLO = 121;
pub const CblasLower: CBLAS_UPLO = 122;

pub type CBLAS_DIAG = c_int;

pub const CblasNonUnit: CBLAS_DIAG = 131;
pub const CblasUnit: CBLAS_DIAG = 132;

pub type CBLAS_SIDE = c_int;

pub const CblasLeft: CBLAS_DIAG = 141;
pub const CblasRight: CBLAS_DIAG = 142;

pub type CBLAS_LAYOUT = CBLAS_ORDER;
