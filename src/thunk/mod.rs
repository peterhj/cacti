use crate::algo::fp::*;
use crate::cell::*;
use crate::cell::smp::*;
use crate::clock::*;
use crate::ctx::{CtxCtr, CtxEnv};
use crate::op::*;
use crate::pctx::{TL_PCTX};
#[cfg(feature = "gpu")]
use crate::pctx::nvgpu::*;
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
use futhark_ffi::types::*;
use home::{home_dir};
use libc::{c_void};
//use smol_str::{SmolStr};

use std::any::{Any};
use std::cell::{RefCell};
use std::cmp::{max};
use std::convert::{TryFrom};
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
  pub fn from_unchecked(p: i32) -> ThunkPtr {
    ThunkPtr(p)
  }

  pub fn to_unchecked(&self) -> i32 {
    self.0
  }

  pub fn as_bytes_repr(&self) -> &[u8] {
    // SAFETY: This should be safe as the type is `Copy`.
    let ptr = (self as *const ThunkPtr) as *const u8;
    let len = size_of::<ThunkPtr>();
    assert_eq!(len, 4);
    unsafe { from_raw_parts(ptr, len) }
  }
}

/*#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ThunkKey {
  // FIXME
  //tyid: TypeId,
}*/

#[derive(Clone, Copy)]
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
}

pub trait ThunkValExt: DtypeExt {
  type Val: DtypeExt + Eq + Any;

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

#[derive(Clone, Copy, Debug)]
pub enum ThunkTypeErr {
  // FIXME FIXME
  _Bot,
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
  fn mode(&self) -> CellMode { CellMode::Aff }
  fn gen_impl_smp(&self, _spectype: Vec<Dim>) -> Option<Rc<dyn ThunkImpl_<Cel=SmpInnerCell>>> { None }
  fn gen_impl_gpu(&self, _spectype: Vec<Dim>) -> Option<Rc<dyn ThunkImpl_<Cel=GpuInnerCell>>> { None }
  // TODO: gen adj.
}

pub trait ThunkSpec_ {
  fn as_any(&self) -> &dyn Any;
  fn as_bytes_repr(&self) -> &[u8];
  fn thunk_eq(&self, other: &dyn ThunkSpec_) -> Option<bool>;
  fn arity(&self) -> (u16, u16);
  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr>;
  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr>;
  fn mode(&self) -> CellMode;
  fn gen_impl_smp(&self, spectype: Vec<Dim>) -> Option<Rc<dyn ThunkImpl_<Cel=SmpInnerCell>>>;
  fn gen_impl_gpu(&self, spectype: Vec<Dim>) -> Option<Rc<dyn ThunkImpl_<Cel=GpuInnerCell>>>;
  // TODO: gen adj.
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

  fn mode(&self) -> CellMode {
    ThunkSpec::mode(self)
  }

  fn gen_impl_smp(&self, spectype: Vec<Dim>) -> Option<Rc<dyn ThunkImpl_<Cel=SmpInnerCell>>> {
    ThunkSpec::gen_impl_smp(self, spectype)
  }

  fn gen_impl_gpu(&self, spectype: Vec<Dim>) -> Option<Rc<dyn ThunkImpl_<Cel=GpuInnerCell>>> {
    ThunkSpec::gen_impl_gpu(self, spectype)
  }
}

//pub type ThunkSmpImpl_ = ThunkImpl_<Cel=SmpInnerCell>;
//pub type ThunkGpuImpl_ = ThunkImpl_<Cel=GpuInnerCell>;

pub trait ThunkImpl {
  type Cel;

  fn apply(&self, _ctr: &CtxCtr, _env: &mut CtxEnv, _spec_: &dyn ThunkSpec_, _args: &[CellPtr], _cel: &mut Weak<Self::Cel>) -> ThunkRet {
    ThunkRet::NotImpl
  }
}

pub trait ThunkImpl_ {
  type Cel;

  fn as_any(&self) -> &dyn Any;
  //fn thunk_eq(&self, other: &dyn ThunkImpl_) -> Option<bool>;
  fn apply(&self, ctr: &CtxCtr, env: &mut CtxEnv, spec_: &dyn ThunkSpec_, args: &[CellPtr], cel: &mut Weak<Self::Cel>) -> ThunkRet;
}

impl<T: ThunkImpl + Any> ThunkImpl_ for T {
  type Cel = <T as ThunkImpl>::Cel;

  fn as_any(&self) -> &dyn Any {
    self
  }

  fn apply(&self, ctr: &CtxCtr, env: &mut CtxEnv, spec_: &dyn ThunkSpec_, args: &[CellPtr], cel: &mut Weak<Self::Cel>) -> ThunkRet {
    ThunkImpl::apply(self, ctr, env, spec_, args, cel)
  }
}

pub trait FutharkThunkSpec {
  fn arity(&self) -> (u16, u16);
  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr>;
  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr>;
  fn mode(&self) -> CellMode { CellMode::Aff }
  fn gen_futhark(&self) -> FutharkThunkCode;
  fn gen_adj_futhark(&self, _gen_code: &FutharkThunkCode) -> Option<FutharkThunkCode> { None }
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

  fn mode(&self) -> CellMode {
    FutharkThunkSpec::mode(self)
  }

  fn gen_impl_smp(&self, spectype: Vec<Dim>) -> Option<Rc<dyn ThunkImpl_<Cel=SmpInnerCell>>> {
    let (arityin, arityout) = self.arity();
    let code = self.gen_futhark();
    Some(Rc::new(FutharkThunkImpl::<MulticoreBackend>{
      arityin,
      arityout,
      spectype,
      code,
      object: RefCell::new(None),
    }))
  }

  fn gen_impl_gpu(&self, spectype: Vec<Dim>) -> Option<Rc<dyn ThunkImpl_<Cel=GpuInnerCell>>> {
    let (arityin, arityout) = self.arity();
    let code = self.gen_futhark();
    Some(Rc::new(FutharkThunkImpl::<CudaBackend>{
      arityin,
      arityout,
      spectype,
      code,
      object: RefCell::new(None),
    }))
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
}

pub struct FutharkThunkImpl<B: FutBackend> where FutharkThunkImpl<B>: FutharkThunkImpl_<B> {
  // TODO
  //pub f_decl:   Vec<u8>,
  //pub f_body:   Vec<u8>,
  //pub f_hash:   _,
  pub arityin:  u16,
  pub arityout: u16,
  pub spectype: Vec<Dim>,
  pub code:     FutharkThunkCode,
  pub object:   RefCell<Option<FutObject<B>>>,
}

impl FutharkThunkImpl_<MulticoreBackend> for FutharkThunkImpl<MulticoreBackend> {
  fn _dropck(&mut self) {
  }

  unsafe fn _setup_object(_obj: &mut FutObject<MulticoreBackend>) {
  }
}

impl FutharkThunkImpl_<CudaBackend> for FutharkThunkImpl<CudaBackend> {
  fn _dropck(&mut self) {
    assert!(LIBCUDA._inner.is_some());
    assert!(LIBCUDART._inner.is_some());
    assert!(LIBNVRTC._inner.is_some());
  }

  #[cfg(not(unix))]
  unsafe fn _setup_object(_obj: &mut FutObject<CudaBackend>) {
    unimplemented!();
  }

  #[cfg(unix)]
  unsafe fn _setup_object(obj: &mut FutObject<CudaBackend>) {
    println!("DEBUG: FutharkThunkImpl::_setup_object: cfg...");
    obj.cfg = (obj.ffi.ctx_cfg_new.as_ref().unwrap())();
    assert!(!obj.cfg.is_null());
    // TODO TODO
    // FIXME FIXME
    //(obj.ffi.ctx_cfg_set_gpu_alloc.as_ref().unwrap())(obj.cfg, LIBCUDA.cuMemAlloc.as_ref().unwrap().as_ptr());
    //(obj.ffi.ctx_cfg_set_gpu_free.as_ref().unwrap())(obj.cfg, LIBCUDA.cuMemFree.as_ref().unwrap().as_ptr());
    (obj.ffi.ctx_cfg_set_gpu_alloc.as_ref().unwrap())(obj.cfg, tl_pctx_gpu_alloc_hook as *const c_void as _);
    (obj.ffi.ctx_cfg_set_gpu_free.as_ref().unwrap())(obj.cfg, tl_pctx_gpu_free_hook as *const c_void as _);
    (obj.ffi.ctx_cfg_set_gpu_back_alloc.as_ref().unwrap())(obj.cfg, tl_pctx_gpu_back_alloc_hook as *const c_void as _);
    (obj.ffi.ctx_cfg_set_gpu_back_free.as_ref().unwrap())(obj.cfg, tl_pctx_gpu_back_free_hook as *const c_void as _);
    // TODO TODO
    (obj.ffi.ctx_cfg_set_cuGetErrorString.as_ref().unwrap())(obj.cfg, LIBCUDA.cuGetErrorString.as_ref().unwrap().as_ptr());
    (obj.ffi.ctx_cfg_set_cuInit.as_ref().unwrap())(obj.cfg, LIBCUDA.cuInit.as_ref().unwrap().as_ptr());
    (obj.ffi.ctx_cfg_set_cuDeviceGetCount.as_ref().unwrap())(obj.cfg, LIBCUDA.cuDeviceGetCount.as_ref().unwrap().as_ptr());
    (obj.ffi.ctx_cfg_set_cuDeviceGetName.as_ref().unwrap())(obj.cfg, LIBCUDA.cuDeviceGetName.as_ref().unwrap().as_ptr());
    (obj.ffi.ctx_cfg_set_cuDeviceGet.as_ref().unwrap())(obj.cfg, LIBCUDA.cuDeviceGet.as_ref().unwrap().as_ptr());
    (obj.ffi.ctx_cfg_set_cuDeviceGetAttribute.as_ref().unwrap())(obj.cfg, LIBCUDA.cuDeviceGetAttribute.as_ref().unwrap().as_ptr());
    (obj.ffi.ctx_cfg_set_cuDevicePrimaryCtxRetain.as_ref().unwrap())(obj.cfg, LIBCUDA.cuDevicePrimaryCtxRetain.as_ref().unwrap().as_ptr());
    (obj.ffi.ctx_cfg_set_cuDevicePrimaryCtxRelease.as_ref().unwrap())(obj.cfg, LIBCUDA.cuDevicePrimaryCtxRelease.as_ref().unwrap().as_ptr());
    (obj.ffi.ctx_cfg_set_cuCtxCreate.as_ref().unwrap())(obj.cfg, LIBCUDA.cuCtxCreate.as_ref().unwrap().as_ptr());
    (obj.ffi.ctx_cfg_set_cuCtxDestroy.as_ref().unwrap())(obj.cfg, LIBCUDA.cuCtxDestroy.as_ref().unwrap().as_ptr());
    (obj.ffi.ctx_cfg_set_cuCtxPopCurrent.as_ref().unwrap())(obj.cfg, LIBCUDA.cuCtxPopCurrent.as_ref().unwrap().as_ptr());
    (obj.ffi.ctx_cfg_set_cuCtxPushCurrent.as_ref().unwrap())(obj.cfg, LIBCUDA.cuCtxPushCurrent.as_ref().unwrap().as_ptr());
    (obj.ffi.ctx_cfg_set_cuCtxSynchronize.as_ref().unwrap())(obj.cfg, LIBCUDA.cuCtxSynchronize.as_ref().unwrap().as_ptr());
    (obj.ffi.ctx_cfg_set_cuMemAlloc.as_ref().unwrap())(obj.cfg, LIBCUDA.cuMemAlloc.as_ref().unwrap().as_ptr());
    (obj.ffi.ctx_cfg_set_cuMemFree.as_ref().unwrap())(obj.cfg, LIBCUDA.cuMemFree.as_ref().unwrap().as_ptr());
    (obj.ffi.ctx_cfg_set_cuMemcpy.as_ref().unwrap())(obj.cfg, LIBCUDA.cuMemcpy.as_ref().unwrap().as_ptr());
    (obj.ffi.ctx_cfg_set_cuMemcpyHtoD.as_ref().unwrap())(obj.cfg, LIBCUDA.cuMemcpyHtoD.as_ref().unwrap().as_ptr());
    (obj.ffi.ctx_cfg_set_cuMemcpyDtoH.as_ref().unwrap())(obj.cfg, LIBCUDA.cuMemcpyDtoH.as_ref().unwrap().as_ptr());
    (obj.ffi.ctx_cfg_set_cuMemcpyAsync.as_ref().unwrap())(obj.cfg, LIBCUDA.cuMemcpyAsync.as_ref().unwrap().as_ptr());
    (obj.ffi.ctx_cfg_set_cuMemcpyHtoDAsync.as_ref().unwrap())(obj.cfg, LIBCUDA.cuMemcpyHtoDAsync.as_ref().unwrap().as_ptr());
    (obj.ffi.ctx_cfg_set_cuMemcpyDtoHAsync.as_ref().unwrap())(obj.cfg, LIBCUDA.cuMemcpyDtoHAsync.as_ref().unwrap().as_ptr());
    (obj.ffi.ctx_cfg_set_cudaEventCreate.as_ref().unwrap())(obj.cfg, LIBCUDART.cudaEventCreate.as_ref().unwrap().as_ptr());
    (obj.ffi.ctx_cfg_set_cudaEventDestroy.as_ref().unwrap())(obj.cfg, LIBCUDART.cudaEventDestroy.as_ref().unwrap().as_ptr());
    (obj.ffi.ctx_cfg_set_cudaEventRecord.as_ref().unwrap())(obj.cfg, LIBCUDART.cudaEventRecord.as_ref().unwrap().as_ptr());
    (obj.ffi.ctx_cfg_set_cudaEventElapsedTime.as_ref().unwrap())(obj.cfg, LIBCUDART.cudaEventElapsedTime.as_ref().unwrap().as_ptr());
    (obj.ffi.ctx_cfg_set_nvrtcGetErrorString.as_ref().unwrap())(obj.cfg, LIBNVRTC.nvrtcGetErrorString.as_ref().unwrap().as_ptr());
    (obj.ffi.ctx_cfg_set_nvrtcCreateProgram.as_ref().unwrap())(obj.cfg, LIBNVRTC.nvrtcCreateProgram.as_ref().unwrap().as_ptr());
    (obj.ffi.ctx_cfg_set_nvrtcDestroyProgram.as_ref().unwrap())(obj.cfg, LIBNVRTC.nvrtcDestroyProgram.as_ref().unwrap().as_ptr());
    (obj.ffi.ctx_cfg_set_nvrtcCompileProgram.as_ref().unwrap())(obj.cfg, LIBNVRTC.nvrtcCompileProgram.as_ref().unwrap().as_ptr());
    (obj.ffi.ctx_cfg_set_nvrtcGetProgramLogSize.as_ref().unwrap())(obj.cfg, LIBNVRTC.nvrtcGetProgramLogSize.as_ref().unwrap().as_ptr());
    (obj.ffi.ctx_cfg_set_nvrtcGetProgramLog.as_ref().unwrap())(obj.cfg, LIBNVRTC.nvrtcGetProgramLog.as_ref().unwrap().as_ptr());
    (obj.ffi.ctx_cfg_set_nvrtcGetPTXSize.as_ref().unwrap())(obj.cfg, LIBNVRTC.nvrtcGetPTXSize.as_ref().unwrap().as_ptr());
    (obj.ffi.ctx_cfg_set_nvrtcGetPTX.as_ref().unwrap())(obj.cfg, LIBNVRTC.nvrtcGetPTX.as_ref().unwrap().as_ptr());
    (obj.ffi.ctx_cfg_set_cuModuleLoadData.as_ref().unwrap())(obj.cfg, LIBCUDA.cuModuleLoadData.as_ref().unwrap().as_ptr());
    (obj.ffi.ctx_cfg_set_cuModuleUnload.as_ref().unwrap())(obj.cfg, LIBCUDA.cuModuleUnload.as_ref().unwrap().as_ptr());
    (obj.ffi.ctx_cfg_set_cuModuleGetFunction.as_ref().unwrap())(obj.cfg, LIBCUDA.cuModuleGetFunction.as_ref().unwrap().as_ptr());
    (obj.ffi.ctx_cfg_set_cuFuncGetAttribute.as_ref().unwrap())(obj.cfg, LIBCUDA.cuFuncGetAttribute.as_ref().unwrap().as_ptr());
    (obj.ffi.ctx_cfg_set_cuLaunchKernel.as_ref().unwrap())(obj.cfg, LIBCUDA.cuLaunchKernel.as_ref().unwrap().as_ptr());
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
}

impl<B: FutBackend> Drop for FutharkThunkImpl<B> where FutharkThunkImpl<B>: FutharkThunkImpl_<B> {
  fn drop(&mut self) {
    *self.object.borrow_mut() = None;
    self._dropck();
  }
}

impl<B: FutBackend> FutharkThunkImpl<B> where FutharkThunkImpl<B>: FutharkThunkImpl_<B> {
  pub fn _to_futhark_entry_type(ty_: Dim) -> String {
    let mut s = String::new();
    // NB: futhark scalars in entry arg position are always emitted as
    // pointers to host memory, so we must coerce those to 1-d arrays.
    if ty_.ndim == 0 {
      s.push_str("[1]");
    }
    for _ in 0 .. ty_.ndim {
      s.push_str("[]");
    }
    s.push_str(ty_.dtype.format_futhark());
    s
  }

  pub fn _try_build_object(&self, ctr: &CtxCtr, env: &mut CtxEnv, /*args: &[CellPtr]*/) {
    let mut s = String::new();
    write!(&mut s, "entry kernel").unwrap();
    for k in 0 .. self.arityin {
      let ty_ = self.spectype[k as usize];
      write!(&mut s, " (x_{}: {})", k, Self::_to_futhark_entry_type(ty_)).unwrap();
    }
    if self.arityout == 1 {
      let ty_ = self.spectype[self.arityin as usize];
      write!(&mut s, " : {}", Self::_to_futhark_entry_type(ty_)).unwrap();
    } else {
      write!(&mut s, " : (").unwrap();
      for k in 0 .. self.arityout {
        let ty_ = self.spectype[(self.arityin + k) as usize];
        write!(&mut s, "{}, ", Self::_to_futhark_entry_type(ty_)).unwrap();
      }
      write!(&mut s, ")").unwrap();
    }
    write!(&mut s, " =\n").unwrap();
    for line in self.code.body.iter() {
      // FIXME FIXME: better pattern match/replace.
      let mut line = line.clone();
      for k in 0 .. self.arityin {
        line = line.replace(&format!("{{%{}}}", k), &format!("x_{}", k));
      }
      for k in 0 .. self.arityout {
        line = line.replace(&format!("{{%{}}}", self.arityin + k), &format!("y_{}", k));
      }
      write!(&mut s, "    ").unwrap();
      s.push_str(&line);
      write!(&mut s, "\n").unwrap();
    }
    write!(&mut s, "    ").unwrap();
    if self.arityout == 1 {
      write!(&mut s, "y_{}", 0).unwrap();
    } else {
      write!(&mut s, "(").unwrap();
      for k in 0 .. self.arityout {
        write!(&mut s, "y_{}, ", k).unwrap();
      }
      write!(&mut s, " )").unwrap();
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
    assert!(TL_LIBNVRTC_BUILTINS_BARRIER.with(|&bar| bar));
    // FIXME FIXME: get the correct dev from pctx.
    cudart_set_cur_dev(0).unwrap();
    match config.cached_or_new_object::<B>(s.as_bytes()) {
      Err(e) => println!("WARNING: FutharkThunkImpl::_try_build_object: {} build error: {:?}", B::cmd_arg(), e),
      Ok(mut obj) => {
        // FIXME FIXME: object ctx may create constants that need to be tracked.
        env.reset_tmp();
        ctr.reset_tmp();
        unsafe { FutharkThunkImpl::<B>::_setup_object(&mut obj); }
        *self.object.borrow_mut() = Some(obj);
        let x0 = ctr.peek_tmp();
        let tmp_ct = -x0.to_unchecked();
        if tmp_ct > 0 {
          println!("DEBUG: FutharkThunkImpl::<CudaBackend>::apply: alloc tmp: {}", tmp_ct);
        }
        for x in (x0.to_unchecked() .. 0).rev() {
          let x = CellPtr::from_unchecked(x);
          // FIXME FIXME
          TL_PCTX.with(|pctx| {
            match pctx.nvgpu.cel_map.borrow().get(&x) {
              None => panic!("bug"),
              Some(cel) => {
                let ty = CellType::top();
                let cel = Rc::downgrade(cel);
                let mut pcel = PCell::new(x, ty.clone());
                pcel.compute = InnerCell::Gpu(cel);
                env.insert(x, ty, pcel);
              }
            }
          });
          let y = env.unify(ctr, x, None);
          // FIXME FIXME
        }
      }
    }
  }
}

impl ThunkImpl for FutharkThunkImpl<MulticoreBackend> {
  type Cel = SmpInnerCell;

  fn apply(&self, ctr: &CtxCtr, env: &mut CtxEnv, spec_: &dyn ThunkSpec_, args: &[CellPtr], cel: &mut Weak<SmpInnerCell>) -> ThunkRet {
    // FIXME
    if self.object.borrow().is_none() {
      self._try_build_object(ctr, env);
    }
    if self.object.borrow().is_none() {
      panic!("bug: FutharkThunkImpl::<MulticoreBackend>::apply: build error");
    }
    unimplemented!();
  }
}

impl ThunkImpl for FutharkThunkImpl<CudaBackend> {
  type Cel = GpuInnerCell;

  fn apply(&self, ctr: &CtxCtr, env: &mut CtxEnv, spec_: &dyn ThunkSpec_, args: &[CellPtr], cel: &mut Weak<GpuInnerCell>) -> ThunkRet {
    if self.object.borrow().is_none() {
      self._try_build_object(ctr, env);
    }
    if self.object.borrow().is_none() {
      panic!("bug: FutharkThunkImpl::<CudaBackend>::apply: build error");
    }
    assert_eq!(args.len(), self.arityin as usize);
    let mut arg_ty_ = Vec::with_capacity(self.arityin as usize);
    let mut arg_arr = Vec::with_capacity(self.arityin as usize);
    for k in 0 .. self.arityin as usize {
      let ty_ = env.lookup(args[k]).unwrap().ty.clone();
      assert_eq!(self.spectype[k], ty_.to_dim());
      let a = match self.spectype[k].ndim {
        0 | 1 => FutArrayDev::alloc_1d(),
        2 => FutArrayDev::alloc_2d(),
        3 => FutArrayDev::alloc_3d(),
        4 => FutArrayDev::alloc_4d(),
        _ => unimplemented!()
      };
      // FIXME FIXME: actually init the array.
      if self.spectype[k].ndim == 0 {
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
      assert_eq!(self.spectype[self.arityin as usize + k], ty_.to_dim());
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
    // TODO TODO
    println!("DEBUG: FutharkThunkImpl::<CudaBackend>::apply: out: rc={:?}", out_arr[0].refcount());
    let (mem_dptr, mem_size) = out_arr[0].parts().unwrap();
    println!("DEBUG: FutharkThunkImpl::<CudaBackend>::apply: out: dptr=0x{:016x} size={}", mem_dptr, mem_size);
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
}

// TODO

pub struct PThunk {
  // FIXME
  pub ptr:      ThunkPtr,
  pub clk:      Clock,
  pub arityin:  u16,
  pub arityout: u16,
  pub spectype: Vec<Dim>,
  pub spec_:    Rc<dyn ThunkSpec_>,
  //pub sub:      Vec<ThunkArg>,
  //pub sub:      RefCell<Vec<ThunkArg>>,
  pub impl_smp: RefCell<Option<Rc<dyn ThunkImpl_<Cel=SmpInnerCell>>>>,
  pub impl_gpu: RefCell<Option<Rc<dyn ThunkImpl_<Cel=GpuInnerCell>>>>,
}

impl PThunk {
  //pub fn new<Th: ThunkSpec + Any + Copy + Eq>(ptr: ThunkPtr, thunk: Th) -> PThunk {}
  pub fn new(ptr: ThunkPtr, spectype: Vec<Dim>, spec_: Rc<dyn ThunkSpec_>) -> PThunk {
    let clk = Clock::default();
    let (arityin, arityout) = spec_.arity();
    assert_eq!(spectype.len(), (arityin + arityout) as usize);
    //let spec_ = Rc::new(thunk);
    //let sub = Vec::new();
    PThunk{
      ptr,
      clk,
      arityin,
      arityout,
      spectype,
      spec_,
      //sub,
      impl_smp: RefCell::new(None),
      impl_gpu: RefCell::new(None),
    }
  }

  pub fn apply_smp(&self, ctr: &CtxCtr, env: &mut CtxEnv, args: &[CellPtr], cel: &mut Weak<SmpInnerCell>) -> ThunkRet {
    // FIXME FIXME
    //unimplemented!();
    //self.spec_.apply_smp(cel)
    if self.impl_smp.borrow().is_none() {
      // FIXME FIXME: potentially load from thunk impl cache.
      *self.impl_smp.borrow_mut() = self.spec_.gen_impl_smp(self.spectype.clone());
    }
    match self.impl_smp.borrow().as_ref() {
      None => panic!("bug"),
      Some(th) => th.apply(ctr, env, &*self.spec_, args, cel)
    }
  }

  pub fn apply_gpu(&self, ctr: &CtxCtr, env: &mut CtxEnv, args: &[CellPtr], cel: &mut Weak<GpuInnerCell>) -> ThunkRet {
    // FIXME FIXME
    //unimplemented!();
    //self.spec_.apply_gpu(cel)
    if self.impl_gpu.borrow().is_none() {
      // FIXME FIXME: potentially load from thunk impl cache.
      *self.impl_gpu.borrow_mut() = self.spec_.gen_impl_gpu(self.spectype.clone());
    }
    match self.impl_gpu.borrow().as_ref() {
      None => panic!("bug"),
      Some(th) => th.apply(ctr, env, &*self.spec_, args, cel)
    }
  }

  // FIXME FIXME: think about this api.
  pub fn apply(&self, ctr: &CtxCtr, env: &mut CtxEnv, args: &[CellPtr], out: CellPtr, out_ty: &CellType, out_cel: &mut PCell) {
  //pub fn apply(&self, args: &[(CellPtr, &CellType, &PCell)], out: CellPtr, out_ty: &CellType, out_cel: &mut PCell) {}
    // FIXME FIXME
    match &mut out_cel.compute {
      &mut InnerCell::Primary => {
        match &mut out_cel.primary {
          &mut InnerCell::Gpu(ref mut cel) => {
            // FIXME FIXME
            self.apply_gpu(ctr, env, args, cel);
          }
          _ => unimplemented!()
        }
      }
      &mut InnerCell::Gpu(ref mut cel) => {
        // FIXME FIXME
        self.apply_gpu(ctr, env, args, cel);
      }
      _ => unimplemented!()
    }
  }
}
