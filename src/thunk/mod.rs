use crate::algo::fp::*;
use crate::cell::*;
use crate::clock::*;
use crate::ctx::{CtxCtr, CtxEnv, Cell_};
//use crate::op::*;
use crate::pctx::{TL_PCTX, Locus, PMach};
#[cfg(feature = "gpu")]
use crate::pctx::nvgpu::*;
//use crate::pctx::smp::*;
//use crate::spine::{Spine};
use cacti_cfg_env::*;
use cacti_gpu_cu_ffi::{LIBCUDA, LIBCUDART, LIBNVRTC, TL_LIBNVRTC_BUILTINS_BARRIER};
use cacti_gpu_cu_ffi::{cuda_memcpy_d2h_async, CudartStream, cudart_set_cur_dev};

use futhark_ffi::{
  Config as FutConfig,
  Object as FutObject,
  ObjectExt,
  Backend as FutBackend,
  MulticoreBackend,
  CudaBackend,
  ArrayDev as FutArrayDev,
  FutharkFloatFormatter,
};
//use futhark_ffi::types::*;
use home::{home_dir};
use libc::{c_void};
//use smol_str::{SmolStr};

use std::any::{Any};
use std::cell::{RefCell};
use std::cmp::{max};
//use std::convert::{TryFrom};
use std::fmt::{Debug, Formatter, Result as FmtResult, Write as FmtWrite};
use std::hash::{Hash, Hasher};
use std::mem::{size_of};
use std::ptr::{null_mut};
use std::rc::{Rc, Weak};
use std::slice::{from_raw_parts};

pub mod op;
//pub mod op_gpu;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct ThunkPtr(i32);

impl Debug for ThunkPtr {
  fn fmt(&self, f: &mut Formatter) -> FmtResult {
    write!(f, "ThunkPtr({})", self.0)
  }
}

impl ThunkPtr {
  pub fn nil() -> ThunkPtr {
    ThunkPtr(0)
  }

  pub fn from_unchecked(p: i32) -> ThunkPtr {
    ThunkPtr(p)
  }

  pub fn to_unchecked(&self) -> i32 {
    self.0
  }

  pub fn is_nil(&self) -> bool {
    self.0 == 0
  }

  pub fn as_bytes_repr(&self) -> &[u8] {
    // SAFETY: This should be safe as the type is `Copy`.
    let ptr = (self as *const ThunkPtr) as *const u8;
    let len = size_of::<ThunkPtr>();
    assert_eq!(len, 4);
    unsafe { from_raw_parts(ptr, len) }
  }
}

/*#[derive(Clone, Copy)]
#[repr(u8)]
pub enum ThunkArgMode {
  _Top,
  I,
  O,
  IO,
}

#[derive(Clone)]
pub struct ThunkArg {
  pub ptr:  CellPtr,
  //pub mode: CellMode,
  //pub ty_:  CellType,
  pub amod: ThunkArgMode,
  // TODO
}*/

pub trait ThunkValExt: DtypeExt {
  type Val: DtypeExt + Copy + Eq + Any;

  fn into_thunk_val(self) -> Self::Val;
}

impl ThunkValExt for f32 {
  type Val = TotalOrd<f32>;

  fn into_thunk_val(self) -> Self::Val {
    self.into()
  }
}

#[derive(Clone, Copy, Debug)]
#[repr(u8)]
pub enum ThunkRet {
  Success,
  Failure,
  NotImpl,
}

#[derive(Clone, Copy, Debug)]
pub enum ThunkDimErr {
  // FIXME FIXME
  _Bot,
}

impl Default for ThunkDimErr {
  fn default() -> ThunkDimErr {
    ThunkDimErr::_Bot
  }
}

#[derive(Clone, Copy, Debug)]
pub enum ThunkTypeErr {
  // FIXME FIXME
  Nondeterministic,
  _Bot,
}

impl Default for ThunkTypeErr {
  fn default() -> ThunkTypeErr {
    ThunkTypeErr::_Bot
  }
}

#[derive(Clone, Copy, Debug)]
pub enum ThunkAdjErr {
  // FIXME FIXME
  NotImpl,
}

#[derive(Clone)]
pub struct ThunkKey(pub Rc<dyn ThunkSpec_ + 'static>);

impl PartialEq for ThunkKey {
  fn eq(&self, rhs: &ThunkKey) -> bool {
    (self.0).thunk_eq(&*(rhs.0)).unwrap_or(false)
  }
}

impl Eq for ThunkKey {}

impl Hash for ThunkKey {
  fn hash<H: Hasher>(&self, hasher: &mut H) {
    (self.0).as_bytes_repr().hash(hasher)
  }
}

pub trait ThunkSpec {
  fn arity(&self) -> (u16, u16);
  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr>;
  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr>;
  fn scalar_val(&self) -> Option<&dyn DtypeExt> { None }
  //fn mode(&self) -> CellMode { CellMode::Aff }
  fn gen_impl_(&self, _spec_dim: Vec<Dim>, _pmach: PMach) -> Option<Rc<dyn ThunkImpl_>> { None }
  fn pop_adj(&self, /*_ctr: &CtxCtr, _env: &mut CtxEnv, _spine: &mut Spine,*/ _arg: &[CellPtr], /*_out: CellPtr,*/ _out_adj: CellPtr, _arg_adj: &[CellPtr]) -> Result<(), ThunkAdjErr> { Err(ThunkAdjErr::NotImpl) }
}

pub trait ThunkSpec_ {
  fn as_any(&self) -> &dyn Any;
  fn as_bytes_repr(&self) -> &[u8];
  fn thunk_eq(&self, other: &dyn ThunkSpec_) -> Option<bool>;
  fn arity(&self) -> (u16, u16);
  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr>;
  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr>;
  //fn scalar_val(&self) -> Option<&dyn DtypeExt>;
  //fn mode(&self) -> CellMode;
  fn gen_impl_(&self, spec_dim: Vec<Dim>, pmach: PMach) -> Option<Rc<dyn ThunkImpl_>>;
  fn pop_adj(&self, /*ctr: &CtxCtr, env: &mut CtxEnv, spine: &mut Spine,*/ arg: &[CellPtr], /*out: CellPtr,*/ out_adj: CellPtr, arg_adj: &[CellPtr]) -> Result<(), ThunkAdjErr>;
}

impl<T: ThunkSpec + Copy + Eq + Any> ThunkSpec_ for T {
  fn as_any(&self) -> &dyn Any {
    self
  }

  fn as_bytes_repr(&self) -> &[u8] {
    // SAFETY: This should be safe as the type is `Copy`.
    let ptr = (self as *const T) as *const u8;
    let len = size_of::<T>();
    unsafe { from_raw_parts(ptr, len) }
  }

  fn thunk_eq(&self, other: &dyn ThunkSpec_) -> Option<bool> {
    other.as_any().downcast_ref::<T>().map(|other| self == other)
  }

  fn arity(&self) -> (u16, u16) {
    ThunkSpec::arity(self)
  }

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    ThunkSpec::out_dim(self, arg)
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    ThunkSpec::out_ty_(self, arg)
  }

  /*fn mode(&self) -> CellMode {
    ThunkSpec::mode(self)
  }*/

  fn gen_impl_(&self, spec_dim: Vec<Dim>, pmach: PMach) -> Option<Rc<dyn ThunkImpl_>> {
    ThunkSpec::gen_impl_(self, spec_dim, pmach)
  }

  fn pop_adj(&self, /*ctr: &CtxCtr, env: &mut CtxEnv, spine: &mut Spine,*/ arg: &[CellPtr], out_adj: CellPtr, arg_adj: &[CellPtr]) -> Result<(), ThunkAdjErr> {
    ThunkSpec::pop_adj(self, /*ctr, env, spine,*/ arg, out_adj, arg_adj)
  }
}

//pub type ThunkSmpImpl_ = ThunkImpl_<Cel=SmpInnerCell>;
//pub type ThunkGpuImpl_ = ThunkImpl_<Cel=GpuInnerCell>;

pub trait ThunkImpl {
  //type Cel;

  //fn apply(&self, _ctr: &CtxCtr, _env: &mut CtxEnv, _spec_: &dyn ThunkSpec_, _args: &[CellPtr], _cel: &mut Weak<Self::Cel>) -> ThunkRet {}
  fn apply(&self, _ctr: &CtxCtr, _env: &mut CtxEnv, _spec_: &dyn ThunkSpec_, _args: &[CellPtr], _out: CellPtr, /*_cel: &mut Option<Weak<dyn InnerCell_>>*/) -> ThunkRet { ThunkRet::NotImpl }
  fn accumulate(&self, _ctr: &CtxCtr, _env: &mut CtxEnv, _spec_: &dyn ThunkSpec_, _arg: &[CellPtr], _out: CellPtr) -> ThunkRet { ThunkRet::NotImpl }
}

pub trait ThunkImpl_ {
  //type Cel;

  fn as_any(&self) -> &dyn Any;
  //fn thunk_eq(&self, other: &dyn ThunkImpl_) -> Option<bool>;
  //fn apply(&self, ctr: &CtxCtr, env: &mut CtxEnv, spec_: &dyn ThunkSpec_, args: &[CellPtr], cel: &mut Weak<Self::Cel>) -> ThunkRet;
  fn apply(&self, ctr: &CtxCtr, env: &mut CtxEnv, spec_: &dyn ThunkSpec_, args: &[CellPtr], out: CellPtr, /*cel: &mut Option<Weak<dyn InnerCell_>>*/) -> ThunkRet;
  fn accumulate(&self, ctr: &CtxCtr, env: &mut CtxEnv, spec_: &dyn ThunkSpec_, arg: &[CellPtr], out: CellPtr) -> ThunkRet;
}

impl<T: ThunkImpl + Any> ThunkImpl_ for T {
  //type Cel = <T as ThunkImpl>::Cel;

  fn as_any(&self) -> &dyn Any {
    self
  }

  //fn apply(&self, ctr: &CtxCtr, env: &mut CtxEnv, spec_: &dyn ThunkSpec_, arg: &[CellPtr], cel: &mut Weak<Self::Cel>) -> ThunkRet {}
  fn apply(&self, ctr: &CtxCtr, env: &mut CtxEnv, spec_: &dyn ThunkSpec_, arg: &[CellPtr], out: CellPtr, /*cel: &mut Option<Weak<dyn InnerCell_>>*/) -> ThunkRet {
    ThunkImpl::apply(self, ctr, env, spec_, arg, out, /*cel*/)
  }

  fn accumulate(&self, ctr: &CtxCtr, env: &mut CtxEnv, spec_: &dyn ThunkSpec_, arg: &[CellPtr], out: CellPtr) -> ThunkRet {
    ThunkImpl::accumulate(self, ctr, env, spec_, arg, out)
  }
}

pub trait FutharkThunkSpec {
  fn arity(&self) -> (u16, u16);
  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr>;
  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr>;
  fn scalar_val(&self) -> Option<&dyn DtypeExt> { None }
  //fn mode(&self) -> CellMode { CellMode::Aff }
  fn gen_futhark(&self) -> FutharkThunkCode;
  //fn gen_adj_futhark(&self, _gen_code: &FutharkThunkCode) -> Option<FutharkThunkCode> { None }
}

impl<T: FutharkThunkSpec> ThunkSpec for T {
  fn arity(&self) -> (u16, u16) {
    FutharkThunkSpec::arity(self)
  }

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    FutharkThunkSpec::out_dim(self, arg)
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    FutharkThunkSpec::out_ty_(self, arg)
  }

  /*fn mode(&self) -> CellMode {
    FutharkThunkSpec::mode(self)
  }*/

  fn gen_impl_(&self, spec_dim: Vec<Dim>, pmach: PMach) -> Option<Rc<dyn ThunkImpl_>> {
    let (arityin, arityout) = self.arity();
    let code = self.gen_futhark();
    Some(match pmach {
      PMach::Smp => {
        Rc::new(FutharkThunkImpl::<MulticoreBackend>{
          arityin,
          arityout,
          spec_dim,
          code,
          object: RefCell::new(None),
          consts: RefCell::new(Vec::new()),
        })
      }
      PMach::NvGpu => {
        Rc::new(FutharkThunkImpl::<CudaBackend>{
          arityin,
          arityout,
          spec_dim,
          code,
          object: RefCell::new(None),
          consts: RefCell::new(Vec::new()),
        })
      }
      _ => unimplemented!()
    })
  }
}

pub trait FutharkNumExt {
  fn as_any(&self) -> &dyn Any;
  fn dtype(&self) -> Dtype;
}

impl<T: DtypeExt + Eq + Any> FutharkNumExt for T {
  fn as_any(&self) -> &dyn Any {
    self
  }

  fn dtype(&self) -> Dtype {
    <T as DtypeExt>::dtype()
  }
}

#[derive(Default)]
pub struct FutharkNumFormatter {
  ffmt: FutharkFloatFormatter,
}

impl FutharkNumFormatter {
  pub fn format(&self, x: &dyn FutharkNumExt) -> String {
    match x.dtype() {
      Dtype::Float64    => unimplemented!(),
      Dtype::Float32    => {
        let x = x.as_any();
        self.ffmt.format_f32(*x.downcast_ref::<TotalOrd<f32>>().map(|x| x.as_ref()).or_else(||
                              x.downcast_ref::<NonNan<f32>>().map(|x| x.as_ref())).unwrap())
      }
      Dtype::Float16    => unimplemented!(),
      Dtype::BFloat16   => unimplemented!(),
      Dtype::Int64      => format!("{}i64", x.as_any().downcast_ref::<i64>().unwrap()),
      Dtype::Int32      => format!("{}i32", x.as_any().downcast_ref::<i32>().unwrap()),
      Dtype::Int16      => format!("{}i16", x.as_any().downcast_ref::<i16>().unwrap()),
      Dtype::Int8       => format!("{}i8", x.as_any().downcast_ref::<i8>().unwrap()),
      Dtype::UInt64     => format!("{}u64", x.as_any().downcast_ref::<u64>().unwrap()),
      Dtype::UInt32     => format!("{}u32", x.as_any().downcast_ref::<u32>().unwrap()),
      Dtype::UInt16     => format!("{}u16", x.as_any().downcast_ref::<u16>().unwrap()),
      Dtype::UInt8      => format!("{}u8", x.as_any().downcast_ref::<u8>().unwrap()),
      _ => panic!("bug")
    }
  }
}

pub struct FutharkThunkCode {
  // TODO
  //pub arityin:  u16,
  //pub arityout: u16,
  //pub decl:     String,
  //pub body:     String,
  pub body:     Vec<String>,
}

pub trait FutharkThunkImpl_<B: FutBackend> {
  fn _dropck(&mut self);
  unsafe fn _setup_object(obj: &mut FutObject<B>);
  fn _build_object(ctr: &CtxCtr, env: &mut CtxEnv, config: &FutConfig, source: &str) -> Option<(FutObject<B>, Vec<StableCell>)>;
}

pub struct FutharkThunkImpl<B: FutBackend> where FutharkThunkImpl<B>: FutharkThunkImpl_<B> {
  // TODO
  //pub f_decl:   Vec<u8>,
  //pub f_body:   Vec<u8>,
  //pub f_hash:   _,
  pub arityin:  u16,
  pub arityout: u16,
  pub spec_dim: Vec<Dim>,
  pub code:     FutharkThunkCode,
  pub object:   RefCell<Option<FutObject<B>>>,
  pub consts:   RefCell<Vec<StableCell>>,
}

impl FutharkThunkImpl_<MulticoreBackend> for FutharkThunkImpl<MulticoreBackend> {
  fn _dropck(&mut self) {
  }

  unsafe fn _setup_object(_obj: &mut FutObject<MulticoreBackend>) {
  }

  fn _build_object(_ctr: &CtxCtr, _env: &mut CtxEnv, _config: &FutConfig, _source: &str) -> Option<(FutObject<MulticoreBackend>, Vec<StableCell>)> {
    None
  }
}

impl FutharkThunkImpl_<CudaBackend> for FutharkThunkImpl<CudaBackend> {
  fn _dropck(&mut self) {
    assert!(LIBCUDA._inner.is_some());
    assert!(LIBCUDART._inner.is_some());
    assert!(LIBNVRTC._inner.is_some());
  }

  unsafe fn _setup_object(obj: &mut FutObject<CudaBackend>) {
    println!("DEBUG: FutharkThunkImpl::_setup_object: cfg...");
    obj.cfg = (obj.ffi.ctx_cfg_new.as_ref().unwrap())();
    assert!(!obj.cfg.is_null());
    // TODO TODO
    // FIXME FIXME
    //(obj.ffi.ctx_cfg_set_gpu_alloc.as_ref().unwrap())(obj.cfg, LIBCUDA.cuMemAlloc.as_ref().unwrap().as_ptr() as _);
    //(obj.ffi.ctx_cfg_set_gpu_free.as_ref().unwrap())(obj.cfg, LIBCUDA.cuMemFree.as_ref().unwrap().as_ptr() as _);
    (obj.ffi.ctx_cfg_set_gpu_alloc.as_ref().unwrap())(obj.cfg, tl_pctx_gpu_alloc_hook as *const c_void as _);
    (obj.ffi.ctx_cfg_set_gpu_free.as_ref().unwrap())(obj.cfg, tl_pctx_gpu_free_hook as *const c_void as _);
    (obj.ffi.ctx_cfg_set_gpu_back_alloc.as_ref().unwrap())(obj.cfg, tl_pctx_gpu_back_alloc_hook as *const c_void as _);
    (obj.ffi.ctx_cfg_set_gpu_back_free.as_ref().unwrap())(obj.cfg, tl_pctx_gpu_back_free_hook as *const c_void as _);
    // TODO TODO
    (obj.ffi.ctx_cfg_set_cuGetErrorString.as_ref().unwrap())(obj.cfg, LIBCUDA.cuGetErrorString.as_ref().unwrap().as_ptr() as _);
    (obj.ffi.ctx_cfg_set_cuInit.as_ref().unwrap())(obj.cfg, LIBCUDA.cuInit.as_ref().unwrap().as_ptr() as _);
    (obj.ffi.ctx_cfg_set_cuDeviceGetCount.as_ref().unwrap())(obj.cfg, LIBCUDA.cuDeviceGetCount.as_ref().unwrap().as_ptr() as _);
    (obj.ffi.ctx_cfg_set_cuDeviceGetName.as_ref().unwrap())(obj.cfg, LIBCUDA.cuDeviceGetName.as_ref().unwrap().as_ptr() as _);
    (obj.ffi.ctx_cfg_set_cuDeviceGet.as_ref().unwrap())(obj.cfg, LIBCUDA.cuDeviceGet.as_ref().unwrap().as_ptr() as _);
    (obj.ffi.ctx_cfg_set_cuDeviceGetAttribute.as_ref().unwrap())(obj.cfg, LIBCUDA.cuDeviceGetAttribute.as_ref().unwrap().as_ptr() as _);
    (obj.ffi.ctx_cfg_set_cuDevicePrimaryCtxRetain.as_ref().unwrap())(obj.cfg, LIBCUDA.cuDevicePrimaryCtxRetain.as_ref().unwrap().as_ptr() as _);
    (obj.ffi.ctx_cfg_set_cuDevicePrimaryCtxRelease.as_ref().unwrap())(obj.cfg, LIBCUDA.cuDevicePrimaryCtxRelease.as_ref().unwrap().as_ptr() as _);
    (obj.ffi.ctx_cfg_set_cuCtxCreate.as_ref().unwrap())(obj.cfg, LIBCUDA.cuCtxCreate.as_ref().unwrap().as_ptr() as _);
    (obj.ffi.ctx_cfg_set_cuCtxDestroy.as_ref().unwrap())(obj.cfg, LIBCUDA.cuCtxDestroy.as_ref().unwrap().as_ptr() as _);
    (obj.ffi.ctx_cfg_set_cuCtxPopCurrent.as_ref().unwrap())(obj.cfg, LIBCUDA.cuCtxPopCurrent.as_ref().unwrap().as_ptr() as _);
    (obj.ffi.ctx_cfg_set_cuCtxPushCurrent.as_ref().unwrap())(obj.cfg, LIBCUDA.cuCtxPushCurrent.as_ref().unwrap().as_ptr() as _);
    (obj.ffi.ctx_cfg_set_cuCtxSynchronize.as_ref().unwrap())(obj.cfg, LIBCUDA.cuCtxSynchronize.as_ref().unwrap().as_ptr() as _);
    (obj.ffi.ctx_cfg_set_cuMemAlloc.as_ref().unwrap())(obj.cfg, LIBCUDA.cuMemAlloc.as_ref().unwrap().as_ptr() as _);
    (obj.ffi.ctx_cfg_set_cuMemFree.as_ref().unwrap())(obj.cfg, LIBCUDA.cuMemFree.as_ref().unwrap().as_ptr() as _);
    (obj.ffi.ctx_cfg_set_cuMemcpy.as_ref().unwrap())(obj.cfg, LIBCUDA.cuMemcpy.as_ref().unwrap().as_ptr() as _);
    (obj.ffi.ctx_cfg_set_cuMemcpyHtoD.as_ref().unwrap())(obj.cfg, LIBCUDA.cuMemcpyHtoD.as_ref().unwrap().as_ptr() as _);
    (obj.ffi.ctx_cfg_set_cuMemcpyDtoH.as_ref().unwrap())(obj.cfg, LIBCUDA.cuMemcpyDtoH.as_ref().unwrap().as_ptr() as _);
    (obj.ffi.ctx_cfg_set_cuMemcpyAsync.as_ref().unwrap())(obj.cfg, LIBCUDA.cuMemcpyAsync.as_ref().unwrap().as_ptr() as _);
    (obj.ffi.ctx_cfg_set_cuMemcpyHtoDAsync.as_ref().unwrap())(obj.cfg, LIBCUDA.cuMemcpyHtoDAsync.as_ref().unwrap().as_ptr() as _);
    (obj.ffi.ctx_cfg_set_cuMemcpyDtoHAsync.as_ref().unwrap())(obj.cfg, LIBCUDA.cuMemcpyDtoHAsync.as_ref().unwrap().as_ptr() as _);
    (obj.ffi.ctx_cfg_set_cudaEventCreate.as_ref().unwrap())(obj.cfg, LIBCUDART.cudaEventCreate.as_ref().unwrap().as_ptr() as _);
    (obj.ffi.ctx_cfg_set_cudaEventDestroy.as_ref().unwrap())(obj.cfg, LIBCUDART.cudaEventDestroy.as_ref().unwrap().as_ptr() as _);
    (obj.ffi.ctx_cfg_set_cudaEventRecord.as_ref().unwrap())(obj.cfg, LIBCUDART.cudaEventRecord.as_ref().unwrap().as_ptr() as _);
    (obj.ffi.ctx_cfg_set_cudaEventElapsedTime.as_ref().unwrap())(obj.cfg, LIBCUDART.cudaEventElapsedTime.as_ref().unwrap().as_ptr() as _);
    (obj.ffi.ctx_cfg_set_nvrtcGetErrorString.as_ref().unwrap())(obj.cfg, LIBNVRTC.nvrtcGetErrorString.as_ref().unwrap().as_ptr() as _);
    (obj.ffi.ctx_cfg_set_nvrtcCreateProgram.as_ref().unwrap())(obj.cfg, LIBNVRTC.nvrtcCreateProgram.as_ref().unwrap().as_ptr() as _);
    (obj.ffi.ctx_cfg_set_nvrtcDestroyProgram.as_ref().unwrap())(obj.cfg, LIBNVRTC.nvrtcDestroyProgram.as_ref().unwrap().as_ptr() as _);
    (obj.ffi.ctx_cfg_set_nvrtcCompileProgram.as_ref().unwrap())(obj.cfg, LIBNVRTC.nvrtcCompileProgram.as_ref().unwrap().as_ptr() as _);
    (obj.ffi.ctx_cfg_set_nvrtcGetProgramLogSize.as_ref().unwrap())(obj.cfg, LIBNVRTC.nvrtcGetProgramLogSize.as_ref().unwrap().as_ptr() as _);
    (obj.ffi.ctx_cfg_set_nvrtcGetProgramLog.as_ref().unwrap())(obj.cfg, LIBNVRTC.nvrtcGetProgramLog.as_ref().unwrap().as_ptr() as _);
    (obj.ffi.ctx_cfg_set_nvrtcGetPTXSize.as_ref().unwrap())(obj.cfg, LIBNVRTC.nvrtcGetPTXSize.as_ref().unwrap().as_ptr() as _);
    (obj.ffi.ctx_cfg_set_nvrtcGetPTX.as_ref().unwrap())(obj.cfg, LIBNVRTC.nvrtcGetPTX.as_ref().unwrap().as_ptr() as _);
    (obj.ffi.ctx_cfg_set_cuModuleLoadData.as_ref().unwrap())(obj.cfg, LIBCUDA.cuModuleLoadData.as_ref().unwrap().as_ptr() as _);
    (obj.ffi.ctx_cfg_set_cuModuleUnload.as_ref().unwrap())(obj.cfg, LIBCUDA.cuModuleUnload.as_ref().unwrap().as_ptr() as _);
    (obj.ffi.ctx_cfg_set_cuModuleGetFunction.as_ref().unwrap())(obj.cfg, LIBCUDA.cuModuleGetFunction.as_ref().unwrap().as_ptr() as _);
    (obj.ffi.ctx_cfg_set_cuFuncGetAttribute.as_ref().unwrap())(obj.cfg, LIBCUDA.cuFuncGetAttribute.as_ref().unwrap().as_ptr() as _);
    (obj.ffi.ctx_cfg_set_cuLaunchKernel.as_ref().unwrap())(obj.cfg, LIBCUDA.cuLaunchKernel.as_ref().unwrap().as_ptr() as _);
    // TODO TODO
    println!("DEBUG: FutharkThunkImpl::_setup_object: cfg done");
    println!("DEBUG: FutharkThunkImpl::_setup_object: ctx...");
    obj.ctx = (obj.ffi.ctx_new.as_ref().unwrap())(obj.cfg);
    assert!(!obj.ctx.is_null());
    // FIXME FIXME: read out the main stream from gpu pctx.
    (obj.ffi.ctx_set_stream.as_ref().unwrap())(obj.ctx, CudartStream::null().as_ptr() as *mut _);
    println!("DEBUG: FutharkThunkImpl::_setup_object: ctx done");
    // FIXME FIXME
    //unimplemented!();
  }

  fn _build_object(ctr: &CtxCtr, env: &mut CtxEnv, config: &FutConfig, source: &str) -> Option<(FutObject<CudaBackend>, Vec<StableCell>)> {
    assert!(TL_LIBNVRTC_BUILTINS_BARRIER.with(|&bar| bar));
    // FIXME FIXME: get the correct dev from pctx.
    cudart_set_cur_dev(0).unwrap();
    match config.cached_or_new_object::<CudaBackend>(source.as_bytes()) {
      Err(e) => {
        println!("WARNING: FutharkThunkImpl::<CudaBackend>::_build_object: build error: {:?}", e);
        None
      }
      Ok(mut obj) => {
        // FIXME FIXME: object ctx may create constants that need to be tracked.
        let mut consts = Vec::new();
        env.reset_tmp();
        ctr.reset_tmp();
        unsafe { FutharkThunkImpl::<CudaBackend>::_setup_object(&mut obj); }
        let x0 = ctr.peek_tmp();
        /*let tmp_ct = -x0.to_unchecked();
        if tmp_ct > 0 {
          println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_build_object: alloc tmp: {}", tmp_ct);
        }*/
        for x in (x0.to_unchecked() .. 0).rev() {
          let x = CellPtr::from_unchecked(x);
          // FIXME FIXME
          TL_PCTX.with(|pctx| {
            match pctx.nvgpu.as_ref().unwrap().cel_map.borrow().get(&x) {
              None => panic!("bug"),
              Some(cel) => {
                let ty = CellType::top();
                let cel = Rc::downgrade(cel);
                let mut pcel = PCell::new(x, ty.clone());
                // FIXME FIXME
                /*pcel.compute = InnerCell::Gpu(cel);*/
                pcel.push_new_replica(Locus::VMem, PMach::NvGpu, Some(cel));
                env.insert_phy(x, ty, pcel);
              }
            }
          });
          let y = env.unify(ctr, x, None);
          TL_PCTX.with(|pctx| {
            pctx.nvgpu.as_ref().unwrap().unify(x, y);
          });
          consts.push(StableCell::retain(env, y));
          // FIXME FIXME
        }
        if consts.len() > 0 {
          println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_build_object: consts: {:?}", consts);
        }
        Some((obj, consts))
      }
    }
  }
}

impl<B: FutBackend> Drop for FutharkThunkImpl<B> where FutharkThunkImpl<B>: FutharkThunkImpl_<B> {
  fn drop(&mut self) {
    *self.object.borrow_mut() = None;
    /*// FIXME FIXME: in case of custom dropck that takes &mut CtxEnv.
    let mut consts = Vec::new();
    swap(&mut *self.consts.borrow_mut(), &mut consts);
    for x in consts.into_iter() {
      x.release(_);
    }*/
    self._dropck();
  }
}

impl<B: FutBackend> FutharkThunkImpl<B> where FutharkThunkImpl<B>: FutharkThunkImpl_<B> {
  pub fn _to_futhark_entry_type(dim: Dim) -> String {
    let mut s = String::new();
    // NB: futhark scalars in entry arg position are always emitted as
    // pointers to host memory, so we must coerce those to 1-d arrays.
    if dim.ndim == 0 {
      s.push_str("[1]");
    }
    for _ in 0 .. dim.ndim {
      s.push_str("[]");
    }
    s.push_str(dim.dtype.format_futhark());
    s
  }

  pub fn _try_build(&self, ctr: &CtxCtr, env: &mut CtxEnv, out_mode: CellMode) {
    let mut s = String::new();
    write!(&mut s, "entry kernel").unwrap();
    for k in 0 .. self.arityin {
      let dim = self.spec_dim[k as usize];
      write!(&mut s, " (x_{}: {})", k, Self::_to_futhark_entry_type(dim)).unwrap();
    }
    if self.arityout == 1 {
      match out_mode {
        CellMode::_Top => panic!("bug"),
        CellMode::Aff => {
          let dim = self.spec_dim[self.arityin as usize];
          write!(&mut s, " : {}", Self::_to_futhark_entry_type(dim)).unwrap();
        }
        CellMode::Init => {
          let dim = self.spec_dim[self.arityin as usize];
          let fty = Self::_to_futhark_entry_type(dim);
          write!(&mut s, " (oy_{}: *{}) : *{}", 0, fty, fty).unwrap();
        }
      }
    } else if out_mode == CellMode::Aff {
      write!(&mut s, " : (").unwrap();
      for k in 0 .. self.arityout {
        let dim = self.spec_dim[(self.arityin + k) as usize];
        write!(&mut s, "{}, ", Self::_to_futhark_entry_type(dim)).unwrap();
      }
      write!(&mut s, ")").unwrap();
    } else {
      panic!("bug");
    }
    write!(&mut s, " =\n").unwrap();
    for k in 0 .. self.arityin {
      let dim = self.spec_dim[k as usize];
      if dim.ndim == 0 {
        write!(&mut s, "\tlet [x_{}] = x_{} in\n", k, k).unwrap();
      }
    }
    if self.arityout == 1 {
      match out_mode {
        CellMode::_Top => panic!("bug"),
        CellMode::Aff => {}
        CellMode::Init => {
          let dim = self.spec_dim[self.arityin as usize];
          if dim.ndim == 0 {
            write!(&mut s, "\tlet [oy_{}] = oy_{} in\n", 0, 0).unwrap();
          }
        }
      }
    }
    for line in self.code.body.iter() {
      // FIXME FIXME: better pattern match/replace.
      let mut line = line.clone();
      for k in 0 .. self.arityin {
        line = line.replace(&format!("{{%{}}}", k), &format!("x_{}", k));
      }
      for k in 0 .. self.arityout {
        line = line.replace(&format!("{{%{}}}", self.arityin + k), &format!("y_{}", k));
      }
      write!(&mut s, "\t").unwrap();
      s.push_str(&line);
      write!(&mut s, "\n").unwrap();
    }
    if self.arityout == 1 {
      match out_mode {
        CellMode::_Top => panic!("bug"),
        CellMode::Aff => {}
        CellMode::Init => {
          let dim = self.spec_dim[self.arityin as usize];
          match dim.ndim {
            0 => {
              write!(&mut s, "\tlet y_{} = oy_{} + y_{} in\n", 0, 0, 0).unwrap();
            }
            1 => {
              write!(&mut s, "\tlet y_{} = map2 (+) oy_{} y_{} in\n", 0, 0, 0).unwrap();
            }
            2 => {
              write!(&mut s, "\tlet oy_{} = flatten oy_{} in\n", 0, 0).unwrap();
              write!(&mut s, "\tlet y_{} = flatten y_{} in\n", 0, 0).unwrap();
              write!(&mut s, "\tlet y_{} = map2 (+) oy_{} y_{} in\n", 0, 0, 0).unwrap();
              write!(&mut s, "\tlet y_{} = unflatten y_{}_s_0 y_{}_s_1 y_{} in\n", 0, 0, 0, 0).unwrap();
            }
            3 => {
              write!(&mut s, "\tlet oy_{} = flatten_3d oy_{} in\n", 0, 0).unwrap();
              write!(&mut s, "\tlet y_{} = flatten_3d y_{} in\n", 0, 0).unwrap();
              write!(&mut s, "\tlet y_{} = map2 (+) oy_{} y_{} in\n", 0, 0, 0).unwrap();
              write!(&mut s, "\tlet y_{} = unflatten_3d y_{}_s_0 y_{}_s_1 y_{}_s_2 y_{} in\n", 0, 0, 0, 0, 0).unwrap();
            }
            4 => {
              write!(&mut s, "\tlet oy_{} = flatten_4d oy_{} in\n", 0, 0).unwrap();
              write!(&mut s, "\tlet y_{} = flatten_4d y_{} in\n", 0, 0).unwrap();
              write!(&mut s, "\tlet y_{} = map2 (+) oy_{} y_{} in\n", 0, 0, 0).unwrap();
              write!(&mut s, "\tlet y_{} = unflatten_4d y_{}_s_0 y_{}_s_1 y_{}_s_2 y_{}_s_3 y_{} in\n", 0, 0, 0, 0, 0, 0).unwrap();
            }
            _ => unimplemented!()
          }
        }
      }
    }
    for k in 0 .. self.arityout {
      let dim = self.spec_dim[(self.arityin + k) as usize];
      if dim.ndim == 0 {
        write!(&mut s, "\tlet y_{} = [y_{}] in\n", k, k).unwrap();
      }
    }
    write!(&mut s, "\t").unwrap();
    if self.arityout == 1 {
      write!(&mut s, "y_{}", 0).unwrap();
    } else if out_mode == CellMode::Aff {
      write!(&mut s, "(").unwrap();
      for k in 0 .. self.arityout {
        write!(&mut s, "y_{}, ", k).unwrap();
      }
      write!(&mut s, " )").unwrap();
    } else {
      panic!("bug");
    }
    write!(&mut s, "\n").unwrap();
    let mut config = FutConfig::default();
    // FIXME FIXME: os-specific paths.
    config.cachedir = home_dir().unwrap().join(".cacti").join("cache");
    TL_CFG_ENV.with(|env| {
      if let Some(path) = env.cabalpath.first() {
        config.futhark = path.join("cacti-futhark");
      }
      if let Some(prefix) = env.cudaprefix.first() {
        config.include = prefix.join("include");
      }
    });
    if let Some((obj, consts)) = FutharkThunkImpl::<B>::_build_object(ctr, env, &config, &s) {
      *self.object.borrow_mut() = Some(obj);
      *self.consts.borrow_mut() = consts;
    }
  }
}

impl ThunkImpl for FutharkThunkImpl<MulticoreBackend> {
  //type Cel = SmpInnerCell;

  //fn apply(&self, ctr: &CtxCtr, env: &mut CtxEnv, spec_: &dyn ThunkSpec_, arg: &[CellPtr], cel: &mut Weak<SmpInnerCell>) -> ThunkRet {}
  fn apply(&self, ctr: &CtxCtr, env: &mut CtxEnv, spec_: &dyn ThunkSpec_, arg: &[CellPtr], out: CellPtr, /*cel: &mut Option<Weak<dyn InnerCell_>>*/) -> ThunkRet {
    // FIXME
    if self.object.borrow().is_none() {
      self._try_build(ctr, env, CellMode::Aff);
    }
    if self.object.borrow().is_none() {
      panic!("bug: FutharkThunkImpl::<MulticoreBackend>::apply: build error");
    }
    unimplemented!();
  }
}

impl ThunkImpl for FutharkThunkImpl<CudaBackend> {
  //type Cel = GpuInnerCell;

  fn apply(&self, ctr: &CtxCtr, env: &mut CtxEnv, spec_: &dyn ThunkSpec_, arg: &[CellPtr], out: CellPtr, /*cel: &mut Option<Weak<dyn InnerCell_>>*/) -> ThunkRet {
    if self.object.borrow().is_none() {
      self._try_build(ctr, env, CellMode::Aff);
    }
    if self.object.borrow().is_none() {
      panic!("bug: FutharkThunkImpl::<CudaBackend>::apply: build error");
    }
    assert_eq!(arg.len(), self.arityin as usize);
    let mut arg_ty_ = Vec::with_capacity(self.arityin as usize);
    let mut arg_arr = Vec::with_capacity(self.arityin as usize);
    for k in 0 .. self.arityin as usize {
      let ty_ = env.lookup_ref(arg[k]).unwrap().ty.clone();
      assert_eq!(self.spec_dim[k], ty_.to_dim());
      let a = match self.spec_dim[k].ndim {
        0 | 1 => FutArrayDev::alloc_1d(),
        2 => FutArrayDev::alloc_2d(),
        3 => FutArrayDev::alloc_3d(),
        4 => FutArrayDev::alloc_4d(),
        _ => unimplemented!()
      };
      // FIXME FIXME: actually init the array.
      if self.spec_dim[k].ndim == 0 {
        a.set_shape(&[1]);
      } else {
        a.set_shape(&ty_.shape);
      }
      arg_ty_.push(ty_);
      arg_arr.push(a);
    }
    let mut out_ty_ = Vec::with_capacity(self.arityout as usize);
    match spec_.out_ty_(&arg_ty_) {
      Err(_) => panic!("BUG: type error"),
      Ok(ty_) => {
        out_ty_.push(ty_);
      }
    }
    let mut out_raw_arr = Vec::with_capacity(self.arityout as usize);
    for k in 0 .. self.arityout as usize {
      let ty_ = &out_ty_[k];
      assert_eq!(self.spec_dim[self.arityin as usize + k], ty_.to_dim());
      // FIXME FIXME
      out_raw_arr.push(null_mut());
    }
    let mut obj = self.object.borrow_mut();
    let obj = obj.as_mut().unwrap();
    // FIXME FIXME: pre-entry setup.
    obj.reset();
    env.reset_tmp();
    ctr.reset_tmp();
    let o_ret = obj.enter_kernel(self.arityin, self.arityout, &arg_arr, &mut out_raw_arr);
    if o_ret.is_err() || (obj.may_fail() && obj.sync().is_err()) {
      // FIXME FIXME: error handling.
      panic!("bug: FutharkThunkImpl::<CudaBackend>::apply: runtime error");
    }
    println!("DEBUG: FutharkThunkImpl::<CudaBackend>::apply: out={:?}", out);
    // FIXME: at this point, the remaining memblocks are the outputs.
    // but, if any of the inputs were clobbered, then we have to unset those.
    // so, do some kind of unification here.
    let mut out_arr = Vec::with_capacity(self.arityout as usize);
    for (k, raw) in out_raw_arr.into_iter().enumerate() {
      out_arr.push(FutArrayDev::from_raw(raw, max(1, out_ty_[k].ndim())));
    }
    unsafe {
      println!("DEBUG: FutharkThunkImpl::<CudaBackend>::apply: out: shape={:?}", out_arr[0].shape().unwrap());
      if out_ty_[0].ndim() == 0 {
        assert_eq!(&[1], out_arr[0].shape().unwrap());
      } else {
        assert_eq!(&out_ty_[0].shape, out_arr[0].shape().unwrap());
      }
    }
    let (mem_dptr, mem_size) = out_arr[0].parts().unwrap();
    TL_PCTX.with(|pctx| {
      let gpu = pctx.nvgpu.as_ref().unwrap();
      match gpu.mem_pool.lookup_dptr(mem_dptr) {
        None => {
        }
        Some((region, p)) => {
          println!("DEBUG: FutharkThunkImpl::<CudaBackend>::apply: out: region={:?} p={:?}", region, p);
          if let Some(p) = p {
            for k in self.consts.borrow().iter() {
              if k.as_ref() == &p {
                println!("DEBUG: FutharkThunkImpl::<CudaBackend>::apply: out:   is const");
                match env.lookup_mut_ref(out) {
                  None => panic!("bug"),
                  Some(e) => {
                    match e.cel_ {
                      &mut Cell_::Top(ref state, optr) => {
                        let state = RefCell::new(state.borrow().clone());
                        *e.cel_ = Cell_::Cow(state, CowCell{optr, pcel: p});
                        println!("DEBUG: FutharkThunkImpl::<CudaBackend>::apply: out: cow {:?} -> {:?}", out, p);
                      }
                      // FIXME FIXME
                      _ => unimplemented!()
                    }
                  }
                }
                break;
              }
            }
          }
        }
      }
    });
    // TODO TODO
    println!("DEBUG: FutharkThunkImpl::<CudaBackend>::apply: out: rc={:?} dptr=0x{:016x} size={}", out_arr[0].refcount(), mem_dptr, mem_size);
    unsafe {
      let mut dst_buf = Vec::with_capacity(mem_size);
      dst_buf.set_len(mem_size);
      cuda_memcpy_d2h_async(dst_buf.as_mut_ptr(), mem_dptr, mem_size, &CudartStream::null()).unwrap();
      let out_val = *(dst_buf.as_ptr() as *const f32);
      println!("DEBUG: FutharkThunkImpl::<CudaBackend>::apply: out: val={:?}", out_val);
    }
    ThunkRet::Success
    //unimplemented!();
  }

  fn accumulate(&self, ctr: &CtxCtr, env: &mut CtxEnv, spec_: &dyn ThunkSpec_, arg: &[CellPtr], out: CellPtr) -> ThunkRet {
    if self.object.borrow().is_none() {
      self._try_build(ctr, env, CellMode::Init);
    }
    if self.object.borrow().is_none() {
      panic!("bug: FutharkThunkImpl::<CudaBackend>::accumulate: build error");
    }
    assert_eq!(arg.len(), self.arityin as usize);
    assert_eq!(1, self.arityout);
    let mut arg_ty_ = Vec::with_capacity((self.arityin + 1) as usize);
    let mut arg_arr = Vec::with_capacity((self.arityin + 1) as usize);
    for k in 0 .. self.arityin as usize {
      let ty_ = env.lookup_ref(arg[k]).unwrap().ty.clone();
      assert_eq!(self.spec_dim[k], ty_.to_dim());
      let a = match self.spec_dim[k].ndim {
        0 | 1 => FutArrayDev::alloc_1d(),
        2 => FutArrayDev::alloc_2d(),
        3 => FutArrayDev::alloc_3d(),
        4 => FutArrayDev::alloc_4d(),
        _ => unimplemented!()
      };
      // FIXME FIXME: actually init the array.
      if self.spec_dim[k].ndim == 0 {
        a.set_shape(&[1]);
      } else {
        a.set_shape(&ty_.shape);
      }
      arg_ty_.push(ty_);
      arg_arr.push(a);
    }
    let mut out_ty_ = Vec::with_capacity(1);
    match spec_.out_ty_(&arg_ty_) {
      Err(_) => panic!("BUG: type error"),
      Ok(ty_) => {
        let k = self.arityin as usize;
        assert_eq!(self.spec_dim[k], ty_.to_dim());
        let a = match self.spec_dim[k].ndim {
          0 | 1 => FutArrayDev::alloc_1d(),
          2 => FutArrayDev::alloc_2d(),
          3 => FutArrayDev::alloc_3d(),
          4 => FutArrayDev::alloc_4d(),
          _ => unimplemented!()
        };
        // FIXME FIXME: actually init the array.
        if self.spec_dim[k].ndim == 0 {
          a.set_shape(&[1]);
        } else {
          a.set_shape(&ty_.shape);
        }
        arg_ty_.push(ty_.clone());
        arg_arr.push(a);
        out_ty_.push(ty_);
      }
    }
    let mut out_raw_arr = Vec::with_capacity(1);
    for k in 0 .. 1 {
      let ty_ = &out_ty_[k];
      assert_eq!(self.spec_dim[self.arityin as usize + k], ty_.to_dim());
      // FIXME FIXME
      out_raw_arr.push(null_mut());
    }
    let mut obj = self.object.borrow_mut();
    let obj = obj.as_mut().unwrap();
    // FIXME FIXME: pre-entry setup.
    obj.reset();
    env.reset_tmp();
    ctr.reset_tmp();
    let o_ret = obj.enter_kernel(self.arityin + 1, 1, &arg_arr, &mut out_raw_arr);
    if o_ret.is_err() || (obj.may_fail() && obj.sync().is_err()) {
      // FIXME FIXME: error handling.
      panic!("bug: FutharkThunkImpl::<CudaBackend>::accumulate: runtime error");
    }
    println!("DEBUG: FutharkThunkImpl::<CudaBackend>::accumulate: out={:?}", out);
    // FIXME: because of uniqueness, the lone output should the same memblock as
    // the last input; so, make sure not to double free.
    let mut out_arr = Vec::with_capacity(1);
    for (k, raw) in out_raw_arr.into_iter().enumerate() {
      out_arr.push(FutArrayDev::from_raw(raw, max(1, out_ty_[k].ndim())));
    }
    /*assert_eq!(arg_arr[self.arityin as usize].as_ptr(), out_arr[0].as_ptr());*/
    let (out_ptr, out_ndim) = arg_arr.pop().unwrap().into_raw();
    assert_eq!(out_ptr, out_arr[0].as_ptr());
    assert_eq!(out_ndim, out_arr[0].ndim());
    // FIXME FIXME
    unimplemented!();
  }
}

// TODO

pub struct PThunk {
  // FIXME
  pub ptr:      ThunkPtr,
  pub clk:      Clock,
  pub arityin:  u16,
  pub arityout: u16,
  pub spec_dim: Vec<Dim>,
  pub spec_:    Rc<dyn ThunkSpec_>,
  //pub sub:      Vec<ThunkArg>,
  //pub sub:      RefCell<Vec<ThunkArg>>,
  // TODO TODO
  pub impl_:    RefCell<Vec<(PMach, Rc<dyn ThunkImpl_>)>>,
}

impl PThunk {
  //pub fn new<Th: ThunkSpec + Any + Copy + Eq>(ptr: ThunkPtr, thunk: Th) -> PThunk {}
  pub fn new(ptr: ThunkPtr, spec_dim: Vec<Dim>, spec_: Rc<dyn ThunkSpec_>) -> PThunk {
    let clk = Clock::default();
    let (arityin, arityout) = spec_.arity();
    assert_eq!(spec_dim.len(), (arityin + arityout) as usize);
    //let spec_ = Rc::new(thunk);
    //let sub = Vec::new();
    PThunk{
      ptr,
      clk,
      arityin,
      arityout,
      spec_dim,
      spec_,
      //sub,
      impl_: RefCell::new(Vec::new()),
    }
  }

  pub fn push_new_impl_(&self, pmach: PMach, thunk: Rc<dyn ThunkImpl_>) {
    for &(o_pmach, _) in self.impl_.borrow().iter() {
      if o_pmach == pmach {
        panic!("bug");
      }
    }
    let mut impl_ = self.impl_.borrow_mut();
    impl_.push((pmach, thunk));
    impl_.sort_by(|&(l_pmach, _), &(r_pmach, _)| r_pmach.cmp(&l_pmach));
  }

  pub fn lookup_impl_(&self, q_pmach: PMach) -> Option<Rc<dyn ThunkImpl_>> {
    for &(pmach, ref thimpl_) in self.impl_.borrow().iter() {
      if pmach == q_pmach {
        return Some(thimpl_.clone());
      }
    }
    None
  }

  // FIXME FIXME: think about this api.
  pub fn apply(&self, ctr: &CtxCtr, env: &mut CtxEnv, args: &[CellPtr], out: CellPtr, /*out_ty: &CellType, out_cel: &mut PCell*/) -> ThunkRet {
    // FIXME FIXME
    match self.lookup_impl_(PMach::NvGpu) {
      None => {
        match self.spec_.gen_impl_(self.spec_dim.clone(), PMach::NvGpu) {
          None => {
            // FIXME: fail stop here.
          }
          Some(thimpl_) => {
            self.push_new_impl_(PMach::NvGpu, thimpl_);
          }
        }
      }
      _ => {}
    }
    match self.lookup_impl_(PMach::NvGpu) {
      None => panic!("bug"),
      Some(thimpl_) => {
        thimpl_.apply(ctr, env, &*self.spec_, args, out, /*out_ty, cel*/)
      }
    }
  }
}
