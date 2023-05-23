use crate::algo::fp::*;
use crate::cell::*;
use crate::cell::gpu::*;
use crate::cell::smp::*;
use crate::clock::*;
use crate::op::*;

use cacti_gpu_cu_ffi::{LIBCUDA, LIBCUDART, LIBNVRTC};

use futhark_ffi::{
  Config as FutConfig,
  Object as FutObject,
  Backend as FutBackend,
  MulticoreBackend,
  CudaBackend,
  FutharkFloatFormatter,
};
use libc::{c_void};

use std::any::{Any};
use std::cell::{RefCell};
use std::convert::{TryFrom};
use std::fmt::{Debug, Formatter, Result as FmtResult, Write as FmtWrite};
use std::hash::{Hash};
use std::mem::{size_of};
use std::rc::{Rc};
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

#[derive(Clone, Copy)]
pub struct ThunkArg {
  pub ptr:  CellPtr,
  pub mode: CellMode,
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

pub trait ThunkSpec {
  fn gen_impl_smp(&self, ) -> Option<Box<dyn ThunkImpl_<Cel=SmpInnerCell>>> { None }
  fn gen_impl_gpu(&self, ) -> Option<Box<dyn ThunkImpl_<Cel=GpuInnerCell>>> { None }
  // TODO: gen adj.
}

pub trait ThunkSpec_ {
  fn as_any(&self) -> &dyn Any;
  fn as_bytes_repr(&self) -> &[u8];
  fn thunk_eq(&self, other: &dyn ThunkSpec_) -> Option<bool>;
  fn gen_impl_smp(&self, ) -> Option<Box<dyn ThunkImpl_<Cel=SmpInnerCell>>>;
  fn gen_impl_gpu(&self, ) -> Option<Box<dyn ThunkImpl_<Cel=GpuInnerCell>>>;
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

  fn gen_impl_smp(&self, ) -> Option<Box<dyn ThunkImpl_<Cel=SmpInnerCell>>> {
    ThunkSpec::gen_impl_smp(self)
  }

  fn gen_impl_gpu(&self, ) -> Option<Box<dyn ThunkImpl_<Cel=GpuInnerCell>>> {
    ThunkSpec::gen_impl_gpu(self)
  }
}

//pub type ThunkSmpImpl_ = ThunkImpl_<Cel=SmpInnerCell>;
//pub type ThunkGpuImpl_ = ThunkImpl_<Cel=GpuInnerCell>;

pub trait ThunkImpl {
  type Cel;

  fn apply(&self, _cel: &Self::Cel) -> ThunkRet {
    ThunkRet::NotImpl
  }
}

pub trait ThunkImpl_ {
  type Cel;

  fn as_any(&self) -> &dyn Any;
  //fn thunk_eq(&self, other: &dyn ThunkImpl_) -> Option<bool>;
  //fn apply_smp(&self, cel: &SmpInnerCell) -> ThunkRet;
  //fn apply_gpu(&self, cel: &GpuInnerCell) -> ThunkRet;
  fn apply(&self, cel: &Self::Cel) -> ThunkRet;
}

impl<T: ThunkImpl + Any> ThunkImpl_ for T {
  type Cel = <T as ThunkImpl>::Cel;

  fn as_any(&self) -> &dyn Any {
    self
  }

  fn apply(&self, cel: &Self::Cel) -> ThunkRet {
    ThunkImpl::apply(self, cel)
  }
}

/*pub enum InnerThunk {
  _Top,
  Futhark(FutharkThunk),
  Custom(Rc<dyn CustomThunk>),
}*/

pub trait FutharkThunkSpec {
  fn gen_futhark(&self) -> FutharkThunkCode;
  fn gen_adj_futhark(&self, _gen_code: &FutharkThunkCode) -> Option<FutharkThunkCode> { None }
}

impl<T: FutharkThunkSpec> ThunkSpec for T {
  fn gen_impl_smp(&self, ) -> Option<Box<dyn ThunkImpl_<Cel=SmpInnerCell>>> {
    let code = self.gen_futhark();
    Some(Box::new(FutharkThunkImpl::<MulticoreBackend>{
      code,
      object: RefCell::new(None),
    }))
  }

  fn gen_impl_gpu(&self, ) -> Option<Box<dyn ThunkImpl_<Cel=GpuInnerCell>>> {
    let code = self.gen_futhark();
    Some(Box::new(FutharkThunkImpl::<CudaBackend>{
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
  pub arityin:  u16,
  pub arityout: u16,
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
    obj.cfg = (obj.ffi.ctx_cfg_new.as_ref().unwrap())();
    assert!(!obj.cfg.is_null());
    // TODO TODO
    (obj.ffi.ctx_cfg_set_gpu_alloc.as_ref().unwrap())(obj.cfg, tl_ctx_gpu_alloc_hook as *const c_void as _);
    (obj.ffi.ctx_cfg_set_gpu_free.as_ref().unwrap())(obj.cfg, tl_ctx_gpu_free_hook as *const c_void as _);
    // TODO TODO
    (obj.ffi.ctx_cfg_set_cuGetErrorString.as_ref().unwrap())(obj.cfg, LIBCUDA.cuGetErrorString.as_ref().unwrap().as_ptr());
    (obj.ffi.ctx_cfg_set_cuInit.as_ref().unwrap())(obj.cfg, LIBCUDA.cuInit.as_ref().unwrap().as_ptr());
    (obj.ffi.ctx_cfg_set_cuDeviceGetCount.as_ref().unwrap())(obj.cfg, LIBCUDA.cuDeviceGetCount.as_ref().unwrap().as_ptr());
    (obj.ffi.ctx_cfg_set_cuDeviceGetName.as_ref().unwrap())(obj.cfg, LIBCUDA.cuDeviceGetName.as_ref().unwrap().as_ptr());
    (obj.ffi.ctx_cfg_set_cuDeviceGet.as_ref().unwrap())(obj.cfg, LIBCUDA.cuDeviceGet.as_ref().unwrap().as_ptr());
    (obj.ffi.ctx_cfg_set_cuDeviceGetAttribute.as_ref().unwrap())(obj.cfg, LIBCUDA.cuDeviceGetAttribute.as_ref().unwrap().as_ptr());
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
  pub fn _try_build_object(&self) {
    let mut s = String::new();
    write!(&mut s, "entry kernel").unwrap();
    for k in 0 .. self.code.arityin {
      write!(&mut s, " x_{}", k).unwrap();
    }
    write!(&mut s, " =\n").unwrap();
    for line in self.code.body.iter() {
      // FIXME FIXME: better pattern match/replace.
      let mut line = line.clone();
      for k in 0 .. self.code.arityin {
        line = line.replace(&format!("{{%{}}}", k), &format!("x_{}", k));
      }
      for k in 0 .. self.code.arityout {
        line = line.replace(&format!("{{%{}}}", self.code.arityin + k), &format!("y_{}", k));
      }
      write!(&mut s, "    ").unwrap();
      s.push_str(&line);
      write!(&mut s, "\n").unwrap();
    }
    write!(&mut s, "    ").unwrap();
    if self.code.arityout == 1 {
      write!(&mut s, "y_{}", 0).unwrap();
    } else {
      write!(&mut s, "(").unwrap();
      for k in 0 .. self.code.arityout {
        write!(&mut s, "y_{}, ", k).unwrap();
      }
      write!(&mut s, " )").unwrap();
    }
    write!(&mut s, "\n").unwrap();
    match FutConfig::default().cached_or_new_object::<B>(s.as_bytes()) {
      Err(e) => println!("WARNING: FutharkThunkImpl::_try_build_object: {} build error: {:?}", B::cmd_arg(), e),
      Ok(mut obj) => {
        unsafe { FutharkThunkImpl::<B>::_setup_object(&mut obj); }
        *self.object.borrow_mut() = Some(obj);
      }
    }
  }
}

impl ThunkImpl for FutharkThunkImpl<MulticoreBackend> {
  type Cel = SmpInnerCell;

  fn apply(&self, cel: &SmpInnerCell) -> ThunkRet {
    // FIXME
    if self.object.borrow().is_none() {
      self._try_build_object();
    }
    assert!(self.object.borrow().is_some());
    unimplemented!();
  }
}

impl ThunkImpl for FutharkThunkImpl<CudaBackend> {
  type Cel = GpuInnerCell;

  fn apply(&self, cel: &GpuInnerCell) -> ThunkRet {
    // FIXME
    if self.object.borrow().is_none() {
      self._try_build_object();
    }
    assert!(self.object.borrow().is_some());
    unimplemented!();
  }
}

/*pub trait CustomThunkImpl: Any {
  // FIXME FIXME
}*/

// TODO

pub struct PThunk {
  // FIXME
  pub optr:     ThunkPtr,
  pub clk:      Clock,
  //pub localsub: Vec<CellPtr>,
  pub localsub: Vec<ThunkArg>,
  // FIXME FIXME
  //pub inner:    InnerThunk,
  pub spec_:    Box<dyn ThunkSpec_>,
  pub impl_smp: RefCell<Option<Box<dyn ThunkImpl_<Cel=SmpInnerCell>>>>,
  pub impl_gpu: RefCell<Option<Box<dyn ThunkImpl_<Cel=GpuInnerCell>>>>,
}

impl PThunk {
  pub fn new0<Th: ThunkSpec + Any + Copy + Eq>(ptr: ThunkPtr, thunk: Th) -> PThunk {
    let clk = Clock::default();
    let localsub = Vec::new();
    let spec_ = Box::new(thunk);
    PThunk{
      optr: ptr,
      clk,
      localsub,
      spec_,
      impl_smp: RefCell::new(None),
      impl_gpu: RefCell::new(None),
    }
  }

  pub fn apply_smp(&self, cel: &SmpInnerCell) -> ThunkRet {
    // FIXME FIXME
    //unimplemented!();
    //self.spec_.apply_smp(cel)
    if self.impl_smp.borrow().is_none() {
      *self.impl_smp.borrow_mut() = self.spec_.gen_impl_smp();
    }
    match self.impl_smp.borrow().as_ref() {
      None => panic!("bug"),
      Some(th) => th.apply(cel)
    }
  }

  pub fn apply_gpu(&self, cel: &GpuInnerCell) -> ThunkRet {
    // FIXME FIXME
    //unimplemented!();
    //self.spec_.apply_gpu(cel)
    if self.impl_gpu.borrow().is_none() {
      *self.impl_gpu.borrow_mut() = self.spec_.gen_impl_gpu();
    }
    match self.impl_gpu.borrow().as_ref() {
      None => panic!("bug"),
      Some(th) => th.apply(cel)
    }
  }
}
