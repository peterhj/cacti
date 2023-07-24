use crate::algo::{SortKey8, SortMap8, RevSortMap8};
use crate::algo::fp::*;
use crate::algo::hash::*;
use crate::algo::str::*;
use crate::cell::*;
use crate::clock::*;
use crate::ctx::{TL_CTX, CtxCtr, CtxEnv, Cell_, CellClosure, CowCell, ctx_lookup_type, ctx_clean_arg, ctx_push_cell_arg, ctx_pop_thunk};
//use crate::op::*;
use crate::pctx::{TL_PCTX, PCtxImpl, Locus, PMach, PAddr, TagUnifier};
#[cfg(feature = "nvgpu")]
use crate::pctx::nvgpu::*;
use crate::pctx::smp::*;
//use crate::spine::{Spine};
use crate::thunk::op::{FutharkCodeThunkSpec};
use crate::util::time::{Stopwatch};
use cacti_cfg_env::*;
#[cfg(feature = "nvgpu")]
use cacti_gpu_cu_ffi::{LIBCUDA, LIBCUDART, LIBNVRTC, TL_LIBNVRTC_BUILTINS_BARRIER};
#[cfg(feature = "nvgpu")]
use cacti_gpu_cu_ffi::{cuda_memcpy_d2h, cuda_memcpy_d2h_async, CudartStream, cudart_set_cur_dev};

use aho_corasick::{AhoCorasick};
use futhark_ffi::{
  Config as FutConfig,
  Stage as FutStage,
  Object as FutObject,
  ObjectExt,
  bindings::ObjectFFI,
  Backend as FutBackend,
  MulticoreBackend,
  Abi as FutAbi,
  AbiOutput as FutAbiOutput,
  AbiArrayRepr as FutAbiArrayRepr,
  AbiScalarType as FutAbiScalarType,
  AbiScalar as FutAbiScalar,
  AbiSpace as FutAbiSpace,
  FutharkFloatFormatter,
};
#[cfg(feature = "nvgpu")]
use futhark_ffi::{
  CudaBackend,
  ArrayDev as FutArrayDev,
};
use home::{home_dir};
//use smol_str::{SmolStr};

use std::any::{Any};
use std::borrow::{Borrow};
use std::cell::{RefCell, UnsafeCell};
use std::cmp::{max};
//use std::convert::{TryFrom};
use std::ffi::{c_void};
use std::fmt::{Debug, Formatter, Result as FmtResult, Write as FmtWrite};
use std::hash::{Hash, Hasher};
use std::mem::{size_of, swap};
use std::ptr::{null_mut};
use std::rc::{Rc, Weak};
use std::slice::{from_raw_parts};
use std::str::{from_utf8};

pub mod op;
//pub mod op_gpu;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct ThunkPtr(pub i32);

impl Debug for ThunkPtr {
  fn fmt(&self, f: &mut Formatter) -> FmtResult {
    write!(f, "ThunkPtr({})", self.0)
  }
}

impl ThunkPtr {
  pub fn nil() -> ThunkPtr {
    ThunkPtr(0)
  }

  pub fn opaque() -> ThunkPtr {
    // FIXME: make sure that ctr never allocates this value.
    ThunkPtr(i32::max_value())
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

  pub fn is_opaque(&self) -> bool {
    self.0 == i32::max_value()
  }

  pub fn as_bytes_repr(&self) -> &[u8] {
    // SAFETY: This is safe because the type is `Copy` and transparent.
    let ptr = ((self as *const ThunkPtr) as *const i32) as *const u8;
    let len = size_of::<ThunkPtr>();
    assert_eq!(len, 4);
    unsafe { from_raw_parts(ptr, len) }
  }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[repr(u8)]
pub enum ThunkMode {
  Apply0 = 0,
  //Apply1 = 1,
  Accumulate = 2,
  Initialize = 1,
}

pub type ThunkResult = Result<(), ThunkErr>;

#[derive(Clone, Copy, Debug)]
#[repr(u8)]
pub enum ThunkErr {
  Failure = 1,
  NotImpl,
}

impl From<ThunkErr> for ThunkResult {
  fn from(e: ThunkErr) -> ThunkResult {
    Err(e)
  }
}

#[derive(Clone, Copy, Debug)]
#[repr(u8)]
pub enum ThunkDimErr {
  // FIXME FIXME
  Deferred = 1,
  Immutable,
  Nondeterm,
  _Bot,
}

impl Default for ThunkDimErr {
  fn default() -> ThunkDimErr {
    ThunkDimErr::_Bot
  }
}
impl ThunkDimErr {
  pub fn into_gen(self) -> FutharkGenErr {
    FutharkGenErr::Dim(self)
  }

  pub fn into_ty_(self) -> ThunkTypeErr {
    match self {
      ThunkDimErr::Deferred   => ThunkTypeErr::Deferred,
      ThunkDimErr::Immutable  => ThunkTypeErr::Immutable,
      ThunkDimErr::Nondeterm  => ThunkTypeErr::Nondeterm,
      ThunkDimErr::_Bot       => ThunkTypeErr::_Bot,
    }
  }
}

#[derive(Clone, Copy, Debug)]
#[repr(u8)]
pub enum ThunkTypeErr {
  // FIXME FIXME
  Deferred = 1,
  Immutable,
  Nondeterm,
  _Bot,
}

impl Default for ThunkTypeErr {
  fn default() -> ThunkTypeErr {
    ThunkTypeErr::_Bot
  }
}

#[derive(Clone, Copy, Debug)]
#[repr(u8)]
pub enum ThunkAdjErr {
  // FIXME FIXME
  NotImpl = 1,
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
    (self.0).hash(hasher)
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub enum ThunkCostR0 {
  Space,
  Time,
}

pub trait ThunkSpec {
  fn debug_name(&self) -> Option<&'static str> { None }
  fn cost_r0(&self) -> Option<ThunkCostR0> { None }
  fn arity(&self) -> (u16, u16);
  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr>;
  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr>;
  fn set_out_dim(&self, _arg: &[Dim], _out: Dim) -> Result<(), ThunkDimErr> { Err(ThunkDimErr::Immutable) }
  fn set_out_ty_(&self, _arg: &[CellType], _out: CellType) -> Result<(), ThunkTypeErr> { Err(ThunkTypeErr::Immutable) }
  //fn scalar_val(&self) -> Option<&dyn DtypeExt> { None }
  //fn mode(&self) -> CellMode { CellMode::Aff }
  fn gen_impl_(&self, _spec_dim: Vec<Dim>, _pmach: PMach) -> Option<Rc<dyn ThunkImpl_>> { None }
  fn pop_adj(&self, _arg: &[(CellPtr, Clock)], _out: CellPtr, _out_clk: Clock, _out_mode: ThunkMode, _out_adj: CellPtr, _arg_adj: &mut [CellPtr]) -> Result<(), ThunkAdjErr> { Err(ThunkAdjErr::NotImpl) }
}

pub trait ThunkSpec_ {
  fn as_any(&self) -> &dyn Any;
  //fn as_bytes_repr(&self) -> &[u8];
  fn hash(&self, hasher: &mut dyn Hasher);
  fn thunk_eq(&self, other: &dyn ThunkSpec_) -> Option<bool>;
  fn debug_name(&self) -> Option<&'static str>;
  fn cost_r0(&self) -> Option<ThunkCostR0>;
  fn arity(&self) -> (u16, u16);
  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr>;
  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr>;
  fn set_out_dim(&self, arg: &[Dim], out: Dim) -> Result<(), ThunkDimErr>;
  fn set_out_ty_(&self, arg: &[CellType], out: CellType) -> Result<(), ThunkTypeErr>;
  //fn scalar_val(&self) -> Option<&dyn DtypeExt>;
  //fn mode(&self) -> CellMode;
  fn gen_impl_(&self, spec_dim: Vec<Dim>, pmach: PMach) -> Option<Rc<dyn ThunkImpl_>>;
  fn pop_adj(&self, arg: &[(CellPtr, Clock)], out: CellPtr, out_clk: Clock, out_mode: ThunkMode, out_adj: CellPtr, arg_adj: &mut [CellPtr]) -> Result<(), ThunkAdjErr>;
}

impl<T: ThunkSpec + Sized + Eq + Hash + Any> ThunkSpec_ for T {
  fn as_any(&self) -> &dyn Any {
    self
  }

  /*fn as_bytes_repr(&self) -> &[u8] {
    // SAFETY: This is safe as the type is `Sized`.
    let ptr = (self as *const T) as *const u8;
    let len = size_of::<T>();
    unsafe { from_raw_parts(ptr, len) }
  }*/

  fn hash(&self, inner: &mut dyn Hasher) {
    let mut h = Hasher_{inner};
    Hash::hash(self, &mut h);
  }

  fn thunk_eq(&self, other: &dyn ThunkSpec_) -> Option<bool> {
    other.as_any().downcast_ref::<T>().map(|other| self == other)
  }

  fn debug_name(&self) -> Option<&'static str> {
    ThunkSpec::debug_name(self)
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    ThunkSpec::cost_r0(self)
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

  fn set_out_dim(&self, arg: &[Dim], out: Dim) -> Result<(), ThunkDimErr> {
    ThunkSpec::set_out_dim(self, arg, out)
  }

  fn set_out_ty_(&self, arg: &[CellType], out: CellType) -> Result<(), ThunkTypeErr> {
    ThunkSpec::set_out_ty_(self, arg, out)
  }

  fn gen_impl_(&self, spec_dim: Vec<Dim>, pmach: PMach) -> Option<Rc<dyn ThunkImpl_>> {
    ThunkSpec::gen_impl_(self, spec_dim, pmach)
  }

  fn pop_adj(&self, arg: &[(CellPtr, Clock)], out: CellPtr, out_clk: Clock, out_mode: ThunkMode, out_adj: CellPtr, arg_adj: &mut [CellPtr]) -> Result<(), ThunkAdjErr> {
    ThunkSpec::pop_adj(self, /*ctr, env, spine,*/ arg, out, out_clk, out_mode, out_adj, arg_adj)
  }
}

pub trait ThunkImpl {
  fn apply(&self, _ctr: &CtxCtr, _env: &mut CtxEnv, _spec_: &dyn ThunkSpec_, _args: &[(CellPtr, Clock)], _th: ThunkPtr, _out: CellPtr, _oclk: Clock) -> ThunkResult { ThunkErr::NotImpl.into() }
  fn accumulate(&self, _ctr: &CtxCtr, _env: &mut CtxEnv, _spec_: &dyn ThunkSpec_, _arg: &[(CellPtr, Clock)], _th: ThunkPtr, _out: CellPtr, _oclk: Clock) -> ThunkResult { ThunkErr::NotImpl.into() }
  fn initialize(&self, _ctr: &CtxCtr, _env: &mut CtxEnv, _spec_: &dyn ThunkSpec_, _arg: &[(CellPtr, Clock)], _th: ThunkPtr, _out: CellPtr, _oclk: Clock) -> ThunkResult { ThunkErr::NotImpl.into() }
}

pub trait ThunkImpl_ {
  fn as_any(&self) -> &dyn Any;
  fn apply(&self, ctr: &CtxCtr, env: &mut CtxEnv, spec_: &dyn ThunkSpec_, args: &[(CellPtr, Clock)], th: ThunkPtr, out: CellPtr, oclk: Clock) -> ThunkResult;
  fn accumulate(&self, ctr: &CtxCtr, env: &mut CtxEnv, spec_: &dyn ThunkSpec_, arg: &[(CellPtr, Clock)], th: ThunkPtr, out: CellPtr, oclk: Clock) -> ThunkResult;
  fn initialize(&self, ctr: &CtxCtr, env: &mut CtxEnv, spec_: &dyn ThunkSpec_, arg: &[(CellPtr, Clock)], th: ThunkPtr, out: CellPtr, oclk: Clock) -> ThunkResult;
}

impl<T: ThunkImpl + Any> ThunkImpl_ for T {
  fn as_any(&self) -> &dyn Any {
    self
  }

  fn apply(&self, ctr: &CtxCtr, env: &mut CtxEnv, spec_: &dyn ThunkSpec_, arg: &[(CellPtr, Clock)], th: ThunkPtr, out: CellPtr, oclk: Clock) -> ThunkResult {
    ThunkImpl::apply(self, ctr, env, spec_, arg, th, out, oclk)
  }

  fn accumulate(&self, ctr: &CtxCtr, env: &mut CtxEnv, spec_: &dyn ThunkSpec_, arg: &[(CellPtr, Clock)], th: ThunkPtr, out: CellPtr, oclk: Clock) -> ThunkResult {
    ThunkImpl::accumulate(self, ctr, env, spec_, arg, th, out, oclk)
  }

  fn initialize(&self, ctr: &CtxCtr, env: &mut CtxEnv, spec_: &dyn ThunkSpec_, arg: &[(CellPtr, Clock)], th: ThunkPtr, out: CellPtr, oclk: Clock) -> ThunkResult {
    unimplemented!();
  }
}

#[derive(Clone, Copy, Debug)]
//#[repr(u8)]
pub enum FutharkGenErr {
  NotImpl,
  Dim(ThunkDimErr),
  _Bot,
}

/*impl From<ThunkDimErr> for FutharkGenErr {
  fn from(e: ThunkDimErr) -> FutharkGenErr {
    FutharkGenErr::Dim(e)
  }
}*/

impl From<FutharkThunkCode> for Result<FutharkThunkCode, FutharkGenErr> {
  #[inline]
  fn from(code: FutharkThunkCode) -> Result<FutharkThunkCode, FutharkGenErr> {
    Ok(code)
  }
}

#[derive(Clone, Copy, Debug)]
#[repr(u8)]
pub enum FutharkThunkAdj {
  Auto,
  Spec,
}

pub trait FutharkThunkSpec {
  //fn tag(&self) -> Option<&'static str> { None }
  fn debug_name(&self) -> Option<&'static str> { None }
  fn cost_r0(&self) -> Option<ThunkCostR0> { None }
  /*fn arity(&self) -> (u16, u16);*/
  fn abi(&self) -> FutAbi;
  fn abi_param(&self, _param: &mut [FutAbiScalar]) -> usize { 0 }
  // FIXME: now that we have params, deprecate scalar_val.
  //fn scalar_val(&self) -> Option<&dyn DtypeExt> { None }
  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr>;
  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr>;
  fn set_out_dim(&self, _arg: &[Dim], _out: Dim) -> Result<(), ThunkDimErr> { Err(ThunkDimErr::Immutable) }
  fn set_out_ty_(&self, _arg: &[CellType], _out: CellType) -> Result<(), ThunkTypeErr> { Err(ThunkTypeErr::Immutable) }
  //fn mode(&self) -> CellMode { CellMode::Aff }
  fn gen_futhark(&self, abi: &mut FutAbi, arg: &[Dim], out: &[Dim]) -> Result<FutharkThunkCode, FutharkGenErr>;
  fn pop_adj(&self, _arg: &[(CellPtr, Clock)], _out: CellPtr, _out_clk: Clock, _out_adj: CellPtr, _arg_adj: &mut [CellPtr]) -> Result<FutharkThunkAdj, ThunkAdjErr> { Ok(FutharkThunkAdj::Auto) }
}

impl<T: FutharkThunkSpec> ThunkSpec for T {
  /*fn tag(&self) -> Option<&'static str> {
    FutharkThunkSpec::tag(self)
  }*/

  fn debug_name(&self) -> Option<&'static str> {
    FutharkThunkSpec::debug_name(self)
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    FutharkThunkSpec::cost_r0(self)
  }

  fn arity(&self) -> (u16, u16) {
    /*FutharkThunkSpec::arity(self)*/
    let abi = FutharkThunkSpec::abi(self);
    (abi.arityin, abi.arityout)
  }

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    FutharkThunkSpec::out_dim(self, arg)
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    FutharkThunkSpec::out_ty_(self, arg)
  }

  fn set_out_dim(&self, arg: &[Dim], out: Dim) -> Result<(), ThunkDimErr> {
    FutharkThunkSpec::set_out_dim(self, arg, out)
  }

  fn set_out_ty_(&self, arg: &[CellType], out: CellType) -> Result<(), ThunkTypeErr> {
    FutharkThunkSpec::set_out_ty_(self, arg, out)
  }

  /*fn mode(&self) -> CellMode {
    FutharkThunkSpec::mode(self)
  }*/

  fn gen_impl_(&self, spec_dim: Vec<Dim>, pmach: PMach) -> Option<Rc<dyn ThunkImpl_>> {
    let mut abi = FutharkThunkSpec::abi(self);
    assert_eq!(abi.space, FutAbiSpace::Unspec);
    let (arg_dim, out_dim) = (&spec_dim).split_at(abi.arityin as usize);
    assert_eq!(out_dim.len(), abi.arityout as usize);
    assert_eq!(arg_dim.len(), abi.arityin as usize);
    let np0 = abi.num_param();
    let mut param: Vec<FutAbiScalar> = Vec::with_capacity(np0);
    param.resize(np0, FutAbiScalar::Unspec);
    assert_eq!(FutharkThunkSpec::abi_param(self, &mut param), np0);
    let code = match FutharkThunkSpec::gen_futhark(self, &mut abi, arg_dim, out_dim) {
      Err(e) => {
        println!("DEBUG: ThunkSpec::gen_impl_: name={:?} arg={:?} out={:?}",
            FutharkThunkSpec::debug_name(self), arg_dim, out_dim);
        println!("ERROR: failed to generate futhark thunk code: {:?}", e);
        panic!();
      }
      Ok(code) => code
    };
    if code.cfg.emit_out0_shape_param {
      for d in 0 .. out_dim[0].ndim {
        abi.push_param(np0 as u16 + d as u16, FutAbiScalarType::I64);
      }
    }
    let name = FutharkThunkSpec::debug_name(self);
    Some(match pmach {
      PMach::Smp => {
        abi.space = FutAbiSpace::Default;
        Rc::new(FutharkThunkImpl::<MulticoreBackend>{
          //arityin,
          //arityout,
          abi,
          param,
          spec_dim,
          code,
          name,
          source: RefCell::new(String::new()),
          //object: RefCell::new(None),
          //consts: RefCell::new(Vec::new()),
          /*modekeys: RefCell::new(Vec::new()),
          object: RefCell::new(SortMap8::new()),
          consts: RefCell::new(SortMap8::new()),*/
          objects: RefCell::new(SortMap8::new()),
        })
      }
      #[cfg(not(feature = "nvgpu"))]
      PMach::NvGpu => {
        panic!("ERROR: not compiled with gpu support");
      }
      #[cfg(feature = "nvgpu")]
      PMach::NvGpu => {
        abi.space = FutAbiSpace::Device;
        Rc::new(FutharkThunkImpl::<CudaBackend>{
          //arityin,
          //arityout,
          abi,
          param,
          spec_dim,
          code,
          name,
          source: RefCell::new(String::new()),
          //object: RefCell::new(None),
          //consts: RefCell::new(Vec::new()),
          /*modekeys: RefCell::new(Vec::new()),
          object: RefCell::new(SortMap8::new()),
          consts: RefCell::new(SortMap8::new()),*/
          objects: RefCell::new(SortMap8::new()),
        })
      }
      _ => unimplemented!()
    })
  }

  fn pop_adj(&self, arg: &[(CellPtr, Clock)], out: CellPtr, out_clk: Clock, out_mode: ThunkMode, out_adj: CellPtr, arg_adj: &mut [CellPtr]) -> Result<(), ThunkAdjErr> {
    assert!(!out.is_nil());
    match FutharkThunkSpec::pop_adj(self, arg, out, out_clk, out_adj, arg_adj) {
      Ok(FutharkThunkAdj::Auto) => {
        // FIXME
        //println!("DEBUG: FutharkThunkSpec::pop_adj: auto adj not implemented: name={:?}", self.debug_name());
        let abi = FutharkThunkSpec::abi(self);
        assert_eq!(1, abi.arityout);
        assert_eq!(arg.len(), abi.arityin as usize);
        assert_eq!(arg_adj.len(), abi.arityin as usize);
        let mut dim = Vec::with_capacity((abi.arityin + abi.arityout) as usize);
        let mut ty_ = Vec::with_capacity((abi.arityin + abi.arityout) as usize);
        for &(x, _) in arg.iter() {
          let xty_ = ctx_lookup_type(x);
          let xdim = xty_.to_dim();
          dim.push(xdim);
          ty_.push(xty_);
        }
        let yty_ = ctx_lookup_type(out);
        let ydim = yty_.to_dim();
        dim.push(ydim);
        ty_.push(yty_);
        let (arg_dim, out_dim) = (&dim).split_at(abi.arityin as usize);
        assert_eq!(out_dim.len(), abi.arityout as usize);
        assert_eq!(arg_dim.len(), abi.arityin as usize);
        let np0 = abi.num_param();
        let mut param: Vec<FutAbiScalar> = Vec::with_capacity(np0);
        param.resize(np0, FutAbiScalar::Unspec);
        assert_eq!(FutharkThunkSpec::abi_param(self, &mut param), np0);
        // FIXME: need to capture the out0 shape param in a closure.
        let mut tmp_abi = abi.clone();
        let mut code = match self.gen_futhark(&mut tmp_abi, arg_dim, out_dim) {
          Err(e) => panic!("ERROR: failed to generate futhark thunk code: {:?}", e),
          Ok(code) => code
        };
        let restore_cfg = code.cfg;
        code.cfg.emit_primal_def = true;
        // FIXME
        let adj_source = code.gen_source(&abi, &dim, out_mode, FutharkThunkBuildConfig::default())
          .map_err(|_| ThunkAdjErr::_Bot)?;
        let cost = FutharkThunkSpec::cost_r0(self);
        for k in 0 .. abi.arityin {
          if arg_adj[k as usize].is_nil() {
            continue;
          }
          let mut adj_abi = FutAbi::default();
          adj_abi.arityin = abi.arityin + 1;
          adj_abi.arityout = 1;
          let mut adj_dim = Vec::with_capacity((abi.arityin + abi.arityout + 1) as usize);
          let mut adj_ty_ = Vec::with_capacity((abi.arityin + abi.arityout + 1) as usize);
          adj_dim.extend_from_slice(&dim);
          adj_ty_.extend_from_slice(&ty_);
          adj_dim.push(dim[k as usize]);
          adj_ty_.push(ty_[k as usize].clone());
          assert_eq!(adj_dim.len(), (abi.arityin + abi.arityout + 1) as usize);
          assert_eq!(adj_ty_.len(), (abi.arityin + abi.arityout + 1) as usize);
          let mut adj_code = FutharkThunkCode::default();
          // FIXME: NB if emit_out0_shape{_param} was set in restore_cfg,
          // then in adj code, that corresponds to the arityin-th input...
          adj_code.cfg = restore_cfg;
          if restore_cfg.emit_out0_shape_param {
            adj_code.cfg.emit_arg_shapes = true;
          } else {
            adj_code.cfg.emit_arg_shapes = false;
          }
          adj_code.cfg.emit_out0_shape = false;
          adj_code.cfg.emit_out0_shape_param = false;
          for line in adj_source.iter() {
            adj_code.pre_append(line);
          }
          // FIXME
          let mut line = format!(r"let {{%{}}} = vjp (\t -> primal", abi.arityin + 1);
          for j in 0 .. k {
            write!(&mut line, r" {{%{}}}", j).unwrap();
          }
          write!(&mut line, r" t").unwrap();
          for j in k + 1 .. abi.arityin {
            write!(&mut line, r" {{%{}}}", j).unwrap();
          }
          if restore_cfg.emit_out0_shape_param {
            let out0_dim = adj_dim[abi.arityin as usize];
            for d in 0 .. out0_dim.ndim {
              write!(&mut line, " {{%{}.s[{}]}}", abi.arityin, d).unwrap();
            }
          }
          write!(&mut line, r") {{%{}}} {{%{}}} in", k, abi.arityin).unwrap();
          adj_code.append(line);
          let adj_spec = FutharkCodeThunkSpec{
            primal_mode: out_mode,
            cost,
            abi: adj_abi,
            dim: adj_dim,
            ty_: adj_ty_,
            code: adj_code,
          };
          assert!(ctx_clean_arg());
          for &(x, _) in arg.iter() {
            ctx_push_cell_arg(x);
          }
          ctx_push_cell_arg(out_adj);
          let adj_rval = ctx_pop_thunk(adj_spec);
          arg_adj[k as usize] += adj_rval;
          // TODO TODO
        }
        // TODO TODO
        Ok(())
      }
      Ok(FutharkThunkAdj::Spec) => Ok(()),
      Err(e) => Err(e)
    }
  }
}

pub trait FutharkNumExt {
  fn as_any(&self) -> &dyn Any;
  fn dtype(&self) -> Dtype;
}

impl<T: DtypeConstExt + Eq + Any> FutharkNumExt for T {
  fn as_any(&self) -> &dyn Any {
    self
  }

  fn dtype(&self) -> Dtype {
    <T as DtypeConstExt>::dtype()
  }
}

#[derive(Default)]
pub struct FutharkNumFormatter {
  ffmt: FutharkFloatFormatter,
}

impl FutharkNumFormatter {
  pub fn format(&self, x: &dyn FutharkNumExt) -> String {
    match x.dtype() {
      Dtype::Fp64       => {
        unimplemented!();
        /*let x = x.as_any();
        self.ffmt.format_f64(*x.downcast_ref::<TotalOrd<f64>>().map(|x| x.as_ref()).unwrap())*/
      }
      Dtype::Fp32       => {
        let x = x.as_any();
        self.ffmt.format_f32(*x.downcast_ref::<TotalOrd<f32>>().map(|x| x.as_ref())/*.or_else(||
                              x.downcast_ref::<NonNan<f32>>().map(|x| x.as_ref()))*/.unwrap())
      }
      Dtype::Fp16       => unimplemented!(),
      Dtype::Bfloat16   => unimplemented!(),
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

  pub fn format_f32_as_dtype(&self, x: f32, dtype: Dtype) -> String {
    // FIXME FIXME
    unimplemented!();
    /*
    let mut s = String::new();
    self.ffmt.format_generic_f32(&mut s, x);
    write!(&mut s, "{}", dtype.to_futhark()).unwrap();
    s
    */
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct FutharkNdBroadcastMap2MonomorphicSpec {
  pub lmsk: u8,
  pub rmsk: u8,
  pub nd:   i8,
}

impl FutharkNdBroadcastMap2MonomorphicSpec {
  pub fn from2(l_ty: &CellType, r_ty: &CellType) -> FutharkNdBroadcastMap2MonomorphicSpec {
    assert_eq!(l_ty.ndim(), r_ty.ndim());
    let nd = l_ty.ndim();
    assert!(nd <= 7);
    let mut lmsk = 0;
    let mut rmsk = 0;
    for d in 0 .. nd {
      let l_len = l_ty.shape[d as usize];
      let r_len = r_ty.shape[d as usize];
      let o_len = max(l_len, r_len);
      if l_len == o_len {
      } else if l_len == 1 {
        lmsk |= (1 << d);
      } else {
        panic!("bug");
      }
      if r_len == o_len {
      } else if r_len == 1 {
        rmsk |= (1 << d);
      } else {
        panic!("bug");
      }
    }
    FutharkNdBroadcastMap2MonomorphicSpec{lmsk, rmsk, nd}
  }

  pub fn ndim(&self) -> i8 {
    self.nd
  }

  pub fn gen_futhark<S: Borrow<str>>(&self, abi: &mut FutAbi, arg0: Dim, arg1: Dim, lam: S) -> Result<FutharkThunkCode, FutharkGenErr> {
    assert_eq!(arg0.ndim(), self.nd);
    assert_eq!(arg1.ndim(), self.nd);
    abi.set_out_arr(0, FutAbiOutput::Pure, FutAbiArrayRepr::Nd, FutAbiScalarType::Unspec);
    abi.set_arg_arr(0, FutAbiArrayRepr::Nd, FutAbiScalarType::Unspec);
    abi.set_arg_arr(1, FutAbiArrayRepr::Nd, FutAbiScalarType::Unspec);
    let lam = lam.borrow();
    match self.nd {
      0 => {
        let mut code = FutharkThunkCode::default();
        code.append(format!(r"let {{%2}} = ({}) {{%0}} {{%1}} in", lam));
        code.into()
      }
      1 => {
        let mut code = FutharkThunkCode::default();
        code.cfg.emit_out0_shape = true;
        code.cfg.emit_out0_shape_param = true;
        match self.lmsk & 1 {
          0 => {
            code.pre_append(format!(r"def f0_dim_0 t_0 {{%2.s**}} = t_0 :> [{{%2.s[0]}}]{}",
                arg0.dtype.format_futhark(),
            ));
          }
          1 => {
            code.pre_append(format!(r"def f0_dim_0 t_0 {{%2.s**}} = (\t -> replicate {{%2.s[0]}} t[0]) t_0"));
          }
          _ => unreachable!()
        }
        match self.rmsk & 1 {
          0 => {
            code.pre_append(format!(r"def f1_dim_0 t_0 {{%2.s**}} = t_0 :> [{{%2.s[0]}}]{}",
                arg1.dtype.format_futhark(),
            ));
          }
          1 => {
            code.pre_append(format!(r"def f1_dim_0 t_0 {{%2.s**}} = (\t -> replicate {{%2.s[0]}} t[0]) t_0"));
          }
          _ => unreachable!()
        }
        code.append(format!(r"let t0 = f0_dim_0 {{%0}} {{%2.s*}} in"));
        code.append(format!(r"let t1 = f1_dim_0 {{%1}} {{%2.s*}} in"));
        code.append(format!(r"let {{%2}} = map2 ({}) t0 t1 in", lam));
        code.into()
      }
      2 => {
        let mut code = FutharkThunkCode::default();
        code.cfg.emit_out0_shape = true;
        code.cfg.emit_out0_shape_param = true;
        match (self.lmsk >> 1) & 1 {
          0 => {
            code.pre_append(format!(r"def f0_dim_1 t_0 {{%2.s**}} = t_0 :> [{{%2.s[1]}}]{}",
                arg0.dtype.format_futhark(),
            ));
          }
          1 => {
            code.pre_append(format!(r"def f0_dim_1 t_0 {{%2.s**}} = (\t -> replicate {{%2.s[1]}} t[0]) t_0"));
          }
          _ => unreachable!()
        }
        match self.lmsk & 1 {
          0 => {
            code.pre_append(format!(r"def f0_dim_0 t_0 {{%2.s**}} = (\t -> map (\u -> f0_dim_1 u {{%2.s*}}) t) t_0 :> [{{%2.s[0]}}][{{%2.s[1]}}]{}",
                arg0.dtype.format_futhark(),
            ));
          }
          1 => {
            code.pre_append(format!(r"def f0_dim_0 t_0 {{%2.s**}} = (\t -> replicate {{%2.s[0]}} (f0_dim_1 t[0] {{%2.s*}})) t_0"));
          }
          _ => unreachable!()
        }
        match (self.rmsk >> 1) & 1 {
          0 => {
            code.pre_append(format!(r"def f1_dim_1 t_0 {{%2.s**}} = t_0 :> [{{%2.s[1]}}]{}",
                arg1.dtype.format_futhark(),
            ));
          }
          1 => {
            code.pre_append(format!(r"def f1_dim_1 t_0 {{%2.s**}} = (\t -> replicate {{%2.s[1]}} t[0]) t_0"));
          }
          _ => unreachable!()
        }
        match self.rmsk & 1 {
          0 => {
            code.pre_append(format!(r"def f1_dim_0 t_0 {{%2.s**}} = (\t -> map (\u -> f1_dim_1 u {{%2.s*}}) t) t_0 :> [{{%2.s[0]}}][{{%2.s[1]}}]{}",
                arg1.dtype.format_futhark(),
            ));
          }
          1 => {
            code.pre_append(format!(r"def f1_dim_0 t_0 {{%2.s**}} = (\t -> replicate {{%2.s[0]}} (f1_dim_1 t[0] {{%2.s*}})) t_0"));
          }
          _ => unreachable!()
        }
        code.append(format!(r"let t0 = flatten (f0_dim_0 {{%0}} {{%2.s*}}) in"));
        code.append(format!(r"let t1 = flatten (f1_dim_0 {{%1}} {{%2.s*}}) in"));
        code.append(format!(r"let t2 = map2 ({}) t0 t1 in", lam));
        code.append(format!(r"let {{%2}} = unflatten t2 in"));
        code.into()
      }
      3 => {
        let mut code = FutharkThunkCode::default();
        code.cfg.emit_out0_shape = true;
        code.cfg.emit_out0_shape_param = true;
        match (self.lmsk >> 2) & 1 {
          0 => {
            code.pre_append(format!(r"def f0_dim_2 t_0 {{%2.s**}} = t_0 :> [{{%2.s[2]}}]{}",
                arg0.dtype.format_futhark(),
            ));
          }
          1 => {
            code.pre_append(format!(r"def f0_dim_2 t_0 {{%2.s**}} = (\t -> replicate {{%2.s[2]}} t[0]) t_0"));
          }
          _ => unreachable!()
        }
        match (self.lmsk >> 1) & 1 {
          0 => {
            code.pre_append(format!(r"def f0_dim_1 t_0 {{%2.s**}} = (\t -> map (\u -> f0_dim_2 u {{%2.s*}}) t) t_0 :> [{{%2.s[1]}}][{{%2.s[2]}}]{}",
                arg0.dtype.format_futhark(),
            ));
          }
          1 => {
            code.pre_append(format!(r"def f0_dim_1 t_0 {{%2.s**}} = (\t -> replicate {{%2.s[1]}} (f0_dim_2 t[0] {{%2.s*}})) t_0"));
          }
          _ => unreachable!()
        }
        match self.lmsk & 1 {
          0 => {
            code.pre_append(format!(r"def f0_dim_0 t_0 {{%2.s**}} = (\t -> map (\u -> f0_dim_1 u {{%2.s*}}) t) t_0 :> [{{%2.s[0]}}][{{%2.s[1]}}][{{%2.s[2]}}]{}",
                arg0.dtype.format_futhark(),
            ));
          }
          1 => {
            code.pre_append(format!(r"def f0_dim_0 t_0 {{%2.s**}} = (\t -> replicate {{%2.s[0]}} (f0_dim_1 t[0] {{%2.s*}})) t_0"));
          }
          _ => unreachable!()
        }
        match (self.rmsk >> 2) & 1 {
          0 => {
            code.pre_append(format!(r"def f1_dim_2 t_0 {{%2.s**}} = t_0 :> [{{%2.s[2]}}]{}",
                arg1.dtype.format_futhark(),
            ));
          }
          1 => {
            code.pre_append(format!(r"def f1_dim_2 t_0 {{%2.s**}} = (\t -> replicate {{%2.s[2]}} t[0]) t_0"));
          }
          _ => unreachable!()
        }
        match (self.rmsk >> 1) & 1 {
          0 => {
            code.pre_append(format!(r"def f1_dim_1 t_0 {{%2.s**}} = (\t -> map (\u -> f1_dim_2 u {{%2.s*}}) t) t_0 :> [{{%2.s[1]}}][{{%2.s[2]}}]{}",
                arg1.dtype.format_futhark(),
            ));
          }
          1 => {
            code.pre_append(format!(r"def f1_dim_1 t_0 {{%2.s**}} = (\t -> replicate {{%2.s[1]}} (f1_dim_2 t[0] {{%2.s*}})) t_0"));
          }
          _ => unreachable!()
        }
        match self.rmsk & 1 {
          0 => {
            code.pre_append(format!(r"def f1_dim_0 t_0 {{%2.s**}} = (\t -> map (\u -> f1_dim_1 u {{%2.s*}}) t) t_0 :> [{{%2.s[0]}}][{{%2.s[1]}}][{{%2.s[2]}}]{}",
                arg1.dtype.format_futhark(),
            ));
          }
          1 => {
            code.pre_append(format!(r"def f1_dim_0 t_0 {{%2.s**}} = (\t -> replicate {{%2.s[0]}} (f1_dim_1 t[0] {{%2.s*}})) t_0"));
          }
          _ => unreachable!()
        }
        code.append(format!(r"let t0 = flatten_3d (f0_dim_0 {{%0}} {{%2.s*}}) in"));
        code.append(format!(r"let t1 = flatten_3d (f1_dim_0 {{%1}} {{%2.s*}}) in"));
        code.append(format!(r"let t2 = map2 ({}) t0 t1 in", lam));
        code.append(format!(r"let {{%2}} = unflatten_3d t2 in"));
        code.into()
      }
      4 => {
        let mut code = FutharkThunkCode::default();
        code.cfg.emit_out0_shape = true;
        code.cfg.emit_out0_shape_param = true;
        match (self.lmsk >> 3) & 1 {
          0 => {
            code.pre_append(format!(r"def f0_dim_3 t_0 {{%2.s**}} = t_0 :> [{{%2.s[3]}}]{}",
                arg0.dtype.format_futhark(),
            ));
          }
          1 => {
            code.pre_append(format!(r"def f0_dim_3 t_0 {{%2.s**}} = (\t -> replicate {{%2.s[3]}} t[0]) t_0"));
          }
          _ => unreachable!()
        }
        match (self.lmsk >> 2) & 1 {
          0 => {
            code.pre_append(format!(r"def f0_dim_2 t_0 {{%2.s**}} = (\t -> map (\u -> f0_dim_3 u {{%2.s*}}) t) t_0 :> [{{%2.s[2]}}][{{%2.s[3]}}]{}",
                arg0.dtype.format_futhark(),
            ));
          }
          1 => {
            code.pre_append(format!(r"def f0_dim_2 t_0 {{%2.s**}} = (\t -> replicate {{%2.s[2]}} (f0_dim_3 t[0] {{%2.s*}})) t_0"));
          }
          _ => unreachable!()
        }
        match (self.lmsk >> 1) & 1 {
          0 => {
            code.pre_append(format!(r"def f0_dim_1 t_0 {{%2.s**}} = (\t -> map (\u -> f0_dim_2 u {{%2.s*}}) t) t_0 :> [{{%2.s[1]}}][{{%2.s[2]}}][{{%2.s[3]}}]{}",
                arg0.dtype.format_futhark(),
            ));
          }
          1 => {
            code.pre_append(format!(r"def f0_dim_1 t_0 {{%2.s**}} = (\t -> replicate {{%2.s[1]}} (f0_dim_2 t[0] {{%2.s*}})) t_0"));
          }
          _ => unreachable!()
        }
        match self.lmsk & 1 {
          0 => {
            code.pre_append(format!(r"def f0_dim_0 t_0 {{%2.s**}} = (\t -> map (\u -> f0_dim_1 u {{%2.s*}}) t) t_0 :> [{{%2.s[0]}}][{{%2.s[1]}}][{{%2.s[2]}}][{{%2.s[3]}}]{}",
                arg0.dtype.format_futhark(),
            ));
          }
          1 => {
            code.pre_append(format!(r"def f0_dim_0 t_0 {{%2.s**}} = (\t -> replicate {{%2.s[0]}} (f0_dim_1 t[0] {{%2.s*}})) t_0"));
          }
          _ => unreachable!()
        }
        match (self.rmsk >> 3) & 1 {
          0 => {
            code.pre_append(format!(r"def f1_dim_3 t_0 {{%2.s**}} = t_0 :> [{{%2.s[3]}}]{}",
                arg1.dtype.format_futhark(),
            ));
          }
          1 => {
            code.pre_append(format!(r"def f1_dim_3 t_0 {{%2.s**}} = (\t -> replicate {{%2.s[3]}} t[0]) t_0"));
          }
          _ => unreachable!()
        }
        match (self.rmsk >> 2) & 1 {
          0 => {
            code.pre_append(format!(r"def f1_dim_2 t_0 {{%2.s**}} = (\t -> map (\u -> f1_dim_3 u {{%2.s*}}) t) t_0 :> [{{%2.s[2]}}][{{%2.s[3]}}]{}",
                arg1.dtype.format_futhark(),
            ));
          }
          1 => {
            code.pre_append(format!(r"def f1_dim_2 t_0 {{%2.s**}} = (\t -> replicate {{%2.s[2]}} (f1_dim_3 t[0] {{%2.s*}})) t_0"));
          }
          _ => unreachable!()
        }
        match (self.rmsk >> 1) & 1 {
          0 => {
            code.pre_append(format!(r"def f1_dim_1 t_0 {{%2.s**}} = (\t -> map (\u -> f1_dim_2 u {{%2.s*}}) t) t_0 :> [{{%2.s[1]}}][{{%2.s[2]}}][{{%2.s[3]}}]{}",
                arg1.dtype.format_futhark(),
            ));
          }
          1 => {
            code.pre_append(format!(r"def f1_dim_1 t_0 {{%2.s**}} = (\t -> replicate {{%2.s[1]}} (f1_dim_2 t[0] {{%2.s*}})) t_0"));
          }
          _ => unreachable!()
        }
        match self.rmsk & 1 {
          0 => {
            code.pre_append(format!(r"def f1_dim_0 t_0 {{%2.s**}} = (\t -> map (\u -> f1_dim_1 u {{%2.s*}}) t) t_0 :> [{{%2.s[0]}}][{{%2.s[1]}}][{{%2.s[2]}}][{{%2.s[3]}}]{}",
                arg1.dtype.format_futhark(),
            ));
          }
          1 => {
            code.pre_append(format!(r"def f1_dim_0 t_0 {{%2.s**}} = (\t -> replicate {{%2.s[0]}} (f1_dim_1 t[0] {{%2.s*}})) t_0"));
          }
          _ => unreachable!()
        }
        code.append(format!(r"let t0 = flatten_4d (f0_dim_0 {{%0}} {{%2.s*}}) in"));
        code.append(format!(r"let t1 = flatten_4d (f1_dim_0 {{%1}} {{%2.s*}}) in"));
        code.append(format!(r"let t2 = map2 ({}) t0 t1 in", lam));
        code.append(format!(r"let {{%2}} = unflatten_4d t2 in"));
        code.into()
      }
      _ => {
        println!("WARNING: FutharkNdBroadcastMap2MonomorphicSpec::gen_futhark: not implemented: {:?} {:?}", arg0, arg1);
        return Err(FutharkGenErr::NotImpl);
      }
    }
  }
}

#[derive(Clone, PartialEq, Eq, Hash, Default)]
pub struct FutharkThunkCode {
  pub cfg:  FutharkThunkBuildConfig,
  pub head: Vec<String>,
  pub body: Vec<String>,
}

impl FutharkThunkCode {
  pub fn flat_map<S: Borrow<str>>(lam: S) -> Result<FutharkThunkCode, FutharkGenErr> {
    let mut code = FutharkThunkCode::default();
    code.append_flat_map(lam)?;
    code.into()
  }

  pub fn append_flat_map<S: Borrow<str>>(&mut self, lam: S) -> Result<(), FutharkGenErr> {
    let lam = lam.borrow();
    self.body.push(format!(r"let {{%1}} = map ({}) {{%0}} in", lam));
    Ok(())
  }

  pub fn nd_replicate<S: Borrow<str>>(abi: &mut FutAbi, out0: Dim, val: S) -> Result<FutharkThunkCode, FutharkGenErr> {
    abi.push_out_arr(0, FutAbiOutput::Pure, FutAbiArrayRepr::Nd, FutAbiScalarType::Unspec);
    let mut code = FutharkThunkCode::default();
    code.append_nd_replicate(out0, val)?;
    code.into()
  }

  pub fn append_nd_replicate<S: Borrow<str>>(&mut self, out0: Dim, val: S) -> Result<(), FutharkGenErr> {
    let val = val.borrow();
    match out0.ndim() {
      0 => {
        self.append(format!(r"let {{%0}} = ({}) in", val));
      }
      1 => {
        self.cfg.emit_out0_shape = true;
        self.cfg.emit_out0_shape_param = true;
        self.append(format!(r"let {{%0}} = replicate {{%0.s[0]}} ({}) in", val));
      }
      2 => {
        self.cfg.emit_out0_shape = true;
        self.cfg.emit_out0_shape_param = true;
        self.append(format!(r"let t0 = replicate ({{%0.s[0]}} * {{%0.s[1]}}) ({}) in", val));
        //self.append(format!(r"let {{%0}} = unflatten {{%0.s[0]}} {{%0.s[1]}} t0 in"));
        self.append(format!(r"let {{%0}} = unflatten t0 in"));
      }
      3 => {
        self.cfg.emit_out0_shape = true;
        self.cfg.emit_out0_shape_param = true;
        self.append(format!(r"let t0 = replicate ({{%0.s[0]}} * {{%0.s[1]}} * {{%0.s[2]}}) ({}) in", val));
        //self.append(format!(r"let {{%0}} = unflatten_3d {{%0.s[0]}} {{%0.s[1]}} {{%0.s[2]}} t0 in"));
        self.append(format!(r"let {{%0}} = unflatten_3d t0 in"));
      }
      4 => {
        self.cfg.emit_out0_shape = true;
        self.cfg.emit_out0_shape_param = true;
        self.append(format!(r"let t0 = replicate ({{%0.s[0]}} * {{%0.s[1]}} * {{%0.s[2]}} * {{%0.s[3]}}) ({}) in", val));
        //self.append(format!(r"let {{%0}} = unflatten_4d {{%0.s[0]}} {{%0.s[1]}} {{%0.s[2]}} {{%0.s[3]}} t0 in"));
        self.append(format!(r"let {{%0}} = unflatten_4d t0 in"));
      }
      _ => {
        println!("WARNING: FutharkThunkCode::nd_replicate: not implemented: {:?}", out0);
        return Err(FutharkGenErr::NotImpl);
      }
    }
    Ok(())
  }

  /*pub fn map_nd<S: Borrow<str>>(arg0: Dim, lam: S) -> Result<FutharkThunkCode, FutharkGenErr> {
    FutharkThunkCode::nd_map(arg0, lam)
  }*/

  pub fn nd_map<S: Borrow<str>>(abi: &mut FutAbi, arg0: Dim, lam: S) -> Result<FutharkThunkCode, FutharkGenErr> {
    abi.push_out_arr(0, FutAbiOutput::Pure, FutAbiArrayRepr::Nd, FutAbiScalarType::Unspec);
    abi.push_arg_arr(0, FutAbiArrayRepr::Nd, FutAbiScalarType::Unspec);
    let mut code = FutharkThunkCode::default();
    code.append_nd_map(arg0, lam)?;
    code.into()
  }

  pub fn append_nd_map<S: Borrow<str>>(&mut self, arg0: Dim, lam: S) -> Result<(), FutharkGenErr> {
    let lam = lam.borrow();
    match arg0.ndim() {
      0 => {
        self.append(format!(r"let {{%1}} = ({}) {{%0}} in", lam));
      }
      1 => {
        self.append(format!(r"let {{%1}} = map ({}) {{%0}} in", lam));
      }
      2 => {
        //self.cfg.emit_arg_shapes = true;
        self.append(format!(r"let t0 = flatten {{%0}} in"));
        self.append(format!(r"let t1 = map ({}) t0 in", lam));
        //self.append(format!(r"let {{%1}} = unflatten {{%0.s[0]}} {{%0.s[1]}} t1 in"));
        self.append(format!(r"let {{%1}} = unflatten t1 in"));
      }
      3 => {
        //self.cfg.emit_arg_shapes = true;
        self.append(format!(r"let t0 = flatten_3d {{%0}} in"));
        self.append(format!(r"let t1 = map ({}) t0 in", lam));
        //self.append(format!(r"let {{%1}} = unflatten_3d {{%0.s[0]}} {{%0.s[1]}} {{%0.s[2]}} t1 in"));
        self.append(format!(r"let {{%1}} = unflatten_3d t1 in"));
      }
      4 => {
        //self.cfg.emit_arg_shapes = true;
        self.append(format!(r"let t0 = flatten_4d {{%0}} in"));
        self.append(format!(r"let t1 = map ({}) t0 in", lam));
        //self.append(format!(r"let {{%1}} = unflatten_4d {{%0.s[0]}} {{%0.s[1]}} {{%0.s[2]}} {{%0.s[3]}} t1 in"));
        self.append(format!(r"let {{%1}} = unflatten_4d t1 in"));
      }
      _ => {
        println!("WARNING: FutharkThunkCode::nd_map: not implemented: {:?}", arg0);
        return Err(FutharkGenErr::NotImpl);
      }
    }
    Ok(())
  }

  /*pub fn map2_nd<S: Borrow<str>>(arg0: Dim, arg1: Dim, lam: S) -> Result<FutharkThunkCode, FutharkGenErr> {
    FutharkThunkCode::nd_map2(arg0, arg1, lam)
  }*/

  pub fn nd_map2<S: Borrow<str>>(abi: &mut FutAbi, arg0: Dim, arg1: Dim, lam: S) -> Result<FutharkThunkCode, FutharkGenErr> {
    abi.push_out_arr(0, FutAbiOutput::Pure, FutAbiArrayRepr::Nd, FutAbiScalarType::Unspec);
    abi.push_arg_arr(0, FutAbiArrayRepr::Nd, FutAbiScalarType::Unspec);
    abi.push_arg_arr(1, FutAbiArrayRepr::Nd, FutAbiScalarType::Unspec);
    let mut code = FutharkThunkCode::default();
    code.append_nd_map2(arg0, arg1, lam)?;
    code.into()
  }

  pub fn append_nd_map2<S: Borrow<str>>(&mut self, arg0: Dim, arg1: Dim, lam: S) -> Result<(), FutharkGenErr> {
    let lam = lam.borrow();
    match (arg0.ndim(), arg1.ndim()) {
      (0, 0) => {
        self.append(format!(r"let {{%2}} = ({}) {{%0}} {{%1}} in", lam));
      }
      (1, 1) => {
        //self.cfg.emit_arg_shapes = true;
        //self.append(format!(r"let a = {{%0.s[0]}} in"));
        //self.append(format!(r"let t0 = {{%0}} :> [a]{} in", arg0.dtype.format_futhark()));
        //self.append(format!(r"let t1 = {{%1}} :> [a]{} in", arg1.dtype.format_futhark()));
        self.append(format!(r"let t0 = {{%0}} in"));
        self.append(format!(r"let t1 = {{%1}} in"));
        self.append(format!(r"let {{%2}} = map2 ({}) t0 t1 in", lam));
      }
      (2, 2) => {
        //self.cfg.emit_arg_shapes = true;
        //self.append(format!(r"let a = {{%0.s[0]}} * {{%0.s[1]}} in"));
        //self.append(format!(r"let t0 = flatten {{%0}} :> [a]{} in", arg0.dtype.format_futhark()));
        //self.append(format!(r"let t1 = flatten {{%1}} :> [a]{} in", arg1.dtype.format_futhark()));
        self.append(format!(r"let t0 = flatten {{%0}} in"));
        self.append(format!(r"let t1 = flatten {{%1}} in"));
        self.append(format!(r"let t2 = map2 ({}) t0 t1 in", lam));
        //self.append(format!(r"let {{%2}} = unflatten {{%0.s[0]}} {{%0.s[1]}} t2 in"));
        self.append(format!(r"let {{%2}} = unflatten t2 in"));
      }
      (3, 3) => {
        //self.cfg.emit_arg_shapes = true;
        //self.append(format!(r"let a = {{%0.s[0]}} * {{%0.s[1]}} * {{%0.s[2]}} in"));
        //self.append(format!(r"let t0 = flatten_3d {{%0}} :> [a]{} in", arg0.dtype.format_futhark()));
        //self.append(format!(r"let t1 = flatten_3d {{%1}} :> [a]{} in", arg1.dtype.format_futhark()));
        self.append(format!(r"let t0 = flatten_3d {{%0}} in"));
        self.append(format!(r"let t1 = flatten_3d {{%1}} in"));
        self.append(format!(r"let t2 = map2 ({}) t0 t1 in", lam));
        //self.append(format!(r"let {{%2}} = unflatten_3d {{%0.s[0]}} {{%0.s[1]}} {{%0.s[2]}} t2 in"));
        self.append(format!(r"let {{%2}} = unflatten_3d t2 in"));
      }
      (4, 4) => {
        //self.cfg.emit_arg_shapes = true;
        //self.append(format!(r"let a = {{%0.s[0]}} * {{%0.s[1]}} * {{%0.s[2]}} * {{%0.s[3]}} in"));
        //self.append(format!(r"let t0 = flatten_4d {{%0}} :> [a]{} in", arg0.dtype.format_futhark()));
        //self.append(format!(r"let t1 = flatten_4d {{%1}} :> [a]{} in", arg1.dtype.format_futhark()));
        self.append(format!(r"let t0 = flatten_4d {{%0}} in"));
        self.append(format!(r"let t1 = flatten_4d {{%1}} in"));
        self.append(format!(r"let t2 = map2 ({}) t0 t1 in", lam));
        //self.append(format!(r"let {{%2}} = unflatten_4d {{%0.s[0]}} {{%0.s[1]}} {{%0.s[2]}} {{%0.s[3]}} t2 in"));
        self.append(format!(r"let {{%2}} = unflatten_4d t2 in"));
      }
      _ => {
        println!("WARNING: FutharkThunkCode::nd_map2: not implemented: {:?} {:?}", arg0, arg1);
        return Err(FutharkGenErr::NotImpl);
      }
    }
    Ok(())
  }

  /*pub fn nd_broadcast_map2_v0<S: Borrow<str>>(arg0: Dim, arg1: Dim, lam: S) -> Result<FutharkThunkCode, FutharkGenErr> {
    let mut code = FutharkThunkCode::default();
    code.append_nd_broadcast_map2_v0(arg0, arg1, lam)?;
    code.into()
  }

  pub fn append_nd_broadcast_map2_v0<S: Borrow<str>>(&mut self, arg0: Dim, arg1: Dim, lam: S) -> Result<(), FutharkGenErr> {
    let lam = lam.borrow();
    let dtype = arg0.dtype.max(arg1.dtype).unwrap();
    match (arg0.ndim(), arg1.ndim()) {
      (0, 0) => {
        self.append(format!(r"let {{%2}} = ({}) ({}.{} {{%0}}) ({}.{} {{%1}}) in",
            lam,
            dtype.format_futhark(), arg0.dtype.format_futhark(),
            dtype.format_futhark(), arg1.dtype.format_futhark(),
        ));
      }
      (1, 1) => {
        self.cfg.emit_arg_shapes = true;
        self.pre_append(format!(r"let f_inner = if {{%0.s[0]}} == {{%1.s[0]}} then (\t0 t1 -> map2 (\u v -> ({}) ({}.{} u) ({}.{} v)) t0 t1) else if {{%0.s[0]}} == 1 then (\t0 t1 -> map (\v -> ({}) ({}.{} t0[0]) ({}.{} v)) t1) else (\t0 t1 -> map (\u -> ({}) ({}.{} u) ({}.{} t1[0])) t0) in",
            lam,
            dtype.format_futhark(), arg0.dtype.format_futhark(),
            dtype.format_futhark(), arg1.dtype.format_futhark(),
            lam,
            dtype.format_futhark(), arg0.dtype.format_futhark(),
            dtype.format_futhark(), arg1.dtype.format_futhark(),
            lam,
            dtype.format_futhark(), arg0.dtype.format_futhark(),
            dtype.format_futhark(), arg1.dtype.format_futhark(),
        ));
        self.append(format!(r"let {{%2}} = f_inner {{%0}} {{%1}} in"));
      }
      (2, 2) => {
        self.cfg.emit_arg_shapes = true;
        self.pre_append(format!(r"let f_inner = if {{%0.s[1]}} == {{%1.s[1]}} then (\t0 t1 -> map2 (\u v -> ({}) ({}.{} u) ({}.{} v)) t0 t1) else if {{%0.s[1]}} == 1 then (\t0 t1 -> map (\v -> ({}) ({}.{} t0[0]) ({}.{} v)) t1) else (\t0 t1 -> map (\u -> ({}) ({}.{} u) ({}.{} t1[0])) t0) in",
            lam,
            dtype.format_futhark(), arg0.dtype.format_futhark(),
            dtype.format_futhark(), arg1.dtype.format_futhark(),
            lam,
            dtype.format_futhark(), arg0.dtype.format_futhark(),
            dtype.format_futhark(), arg1.dtype.format_futhark(),
            lam,
            dtype.format_futhark(), arg0.dtype.format_futhark(),
            dtype.format_futhark(), arg1.dtype.format_futhark(),
        ));
        self.pre_append(format!(r"let f_outer = if {{%0.s[0]}} == {{%1.s[0]}} then (\t0 t1 -> map2 (\u v -> f_inner u v) t0 t1) else if {{%0.s[0]}} == 1 then (\t0 t1 -> map (\v -> f_inner t0[0] v) t1) else (\t0 t1 -> map (\u -> f_inner u t1[0]) t0) in"));
        self.append(format!(r"let {{%2}} = f_outer {{%0}} {{%1}} in"));
      }
      (3, 3) => {
        self.cfg.emit_arg_shapes = true;
        self.pre_append(format!(r"let f_inner = if {{%0.s[2]}} == {{%1.s[2]}} then (\t0 t1 -> map2 (\u v -> ({}) ({}.{} u) ({}.{} v)) t0 t1) else if {{%0.s[2]}} == 1 then (\t0 t1 -> map (\v -> ({}) ({}.{} t0[0]) ({}.{} v)) t1) else (\t0 t1 -> map (\u -> ({}) ({}.{} u) ({}.{} t1[0])) t0) in",
            lam,
            dtype.format_futhark(), arg0.dtype.format_futhark(),
            dtype.format_futhark(), arg1.dtype.format_futhark(),
            lam,
            dtype.format_futhark(), arg0.dtype.format_futhark(),
            dtype.format_futhark(), arg1.dtype.format_futhark(),
            lam,
            dtype.format_futhark(), arg0.dtype.format_futhark(),
            dtype.format_futhark(), arg1.dtype.format_futhark(),
        ));
        self.pre_append(format!(r"let f_dim_1 = if {{%0.s[1]}} == {{%1.s[1]}} then (\t0 t1 -> map2 (\u v -> f_inner u v) t0 t1) else if {{%0.s[1]}} == 1 then (\t0 t1 -> map (\v -> f_inner t0[0] v) t1) else (\t0 t1 -> map (\u -> f_inner u t1[0]) t0) in"));
        self.pre_append(format!(r"let f_outer = if {{%0.s[0]}} == {{%1.s[0]}} then (\t0 t1 -> map2 (\u v -> f_dim_1 u v) t0 t1) else if {{%0.s[0]}} == 1 then (\t0 t1 -> map (\v -> f_dim_1 t0[0] v) t1) else (\t0 t1 -> map (\u -> f_dim_1 u t1[0]) t0) in"));
        self.append(format!(r"let {{%2}} = f_outer {{%0}} {{%1}} in"));
      }
      (4, 4) => {
        self.cfg.emit_arg_shapes = true;
        self.cfg.emit_out0_shape = true;
        self.cfg.emit_out0_shape_param = true;
        self.pre_append(format!(r"def f_inner t_0 t_1 {{%0.s**}} {{%1.s**}} {{%2.s**}} = if {{%0.s[3]}} == {{%1.s[3]}} then (\t0 t1 -> map2 (\u v -> ({}) ({}.{} u) ({}.{} v)) (t0 :> [{{%2.s[3]}}]{}) (t1 :> [{{%2.s[3]}}]{})) t_0 t_1 else if {{%0.s[3]}} == 1 then (\t0 t1 -> map (\v -> ({}) ({}.{} t0[0]) ({}.{} v)) (t1 :> [{{%2.s[3]}}]{})) t_0 t_1 else (\t0 t1 -> map (\u -> ({}) ({}.{} u) ({}.{} t1[0])) (t0 :> [{{%2.s[3]}}]{})) t_0 t_1",
            lam,
            dtype.format_futhark(), arg0.dtype.format_futhark(),
            dtype.format_futhark(), arg1.dtype.format_futhark(),
            arg0.dtype.format_futhark(),
            arg1.dtype.format_futhark(),
            lam,
            dtype.format_futhark(), arg0.dtype.format_futhark(),
            dtype.format_futhark(), arg1.dtype.format_futhark(),
            arg1.dtype.format_futhark(),
            lam,
            dtype.format_futhark(), arg0.dtype.format_futhark(),
            dtype.format_futhark(), arg1.dtype.format_futhark(),
            arg0.dtype.format_futhark(),
        ));
        self.pre_append(format!(r"def f_dim_2 t_0 t_1 {{%0.s**}} {{%1.s**}} {{%2.s**}} = if {{%0.s[2]}} == {{%1.s[2]}} then (\t0 t1 -> map2 (\u v -> f_inner u v {{%0.s*}} {{%1.s*}} {{%2.s*}}) (t0 :> [{{%2.s[2]}}][{{%0.s[3]}}]{}) (t1 :> [{{%2.s[2]}}][{{%1.s[3]}}]{})) t_0 t_1 else if {{%0.s[2]}} == 1 then (\t0 t1 -> map (\v -> f_inner t0[0] v {{%0.s*}} {{%1.s*}} {{%2.s*}}) (t1 :> [{{%2.s[2]}}][{{%1.s[3]}}]{})) t_0 t_1 else (\t0 t1 -> map (\u -> f_inner u t1[0] {{%0.s*}} {{%1.s*}} {{%2.s*}}) (t0 :> [{{%2.s[2]}}][{{%0.s[3]}}]{})) t_0 t_1",
            arg0.dtype.format_futhark(),
            arg1.dtype.format_futhark(),
            arg1.dtype.format_futhark(),
            arg0.dtype.format_futhark(),
        ));
        self.pre_append(format!(r"def f_dim_1 t_0 t_1 {{%0.s**}} {{%1.s**}} {{%2.s**}} = if {{%0.s[1]}} == {{%1.s[1]}} then (\t0 t1 -> map2 (\u v -> f_dim_2 u v {{%0.s*}} {{%1.s*}} {{%2.s*}}) (t0 :> [{{%2.s[1]}}][{{%0.s[2]}}][{{%0.s[3]}}]{}) (t1 :> [{{%2.s[1]}}][{{%1.s[2]}}][{{%1.s[3]}}]{})) t_0 t_1 else if {{%0.s[1]}} == 1 then (\t0 t1 -> map (\v -> f_dim_2 t0[0] v {{%0.s*}} {{%1.s*}} {{%2.s*}}) (t1 :> [{{%2.s[1]}}][{{%1.s[2]}}][{{%1.s[3]}}]{})) t_0 t_1 else (\t0 t1 -> map (\u -> f_dim_2 u t1[0] {{%0.s*}} {{%1.s*}} {{%2.s*}}) (t0 :> [{{%2.s[1]}}][{{%0.s[2]}}][{{%0.s[3]}}]{})) t_0 t_1",
            arg0.dtype.format_futhark(),
            arg1.dtype.format_futhark(),
            arg1.dtype.format_futhark(),
            arg0.dtype.format_futhark(),
        ));
        self.pre_append(format!(r"def f_outer t_0 t_1 {{%0.s**}} {{%1.s**}} {{%2.s**}} = if {{%0.s[0]}} == {{%1.s[0]}} then (\t0 t1 -> map2 (\u v -> f_dim_1 u v {{%0.s*}} {{%1.s*}} {{%2.s*}}) (t0 :> [{{%2.s[0]}}][{{%0.s[1]}}][{{%0.s[2]}}][{{%0.s[3]}}]{}) (t1 :> [{{%2.s[0]}}][{{%1.s[1]}}][{{%1.s[2]}}][{{%1.s[3]}}]{})) t_0 t_1 else if {{%0.s[0]}} == 1 then (\t0 t1 -> map (\v -> f_dim_1 t0[0] v {{%0.s*}} {{%1.s*}} {{%2.s*}}) (t1 :> [{{%2.s[0]}}][{{%1.s[1]}}][{{%1.s[2]}}][{{%1.s[3]}}]{})) t_0 t_1 else (\t0 t1 -> map (\u -> f_dim_1 u t1[0] {{%0.s*}} {{%1.s*}} {{%2.s*}}) (t0 :> [{{%2.s[0]}}][{{%0.s[1]}}][{{%0.s[2]}}][{{%0.s[3]}}]{})) t_0 t_1",
            arg0.dtype.format_futhark(),
            arg1.dtype.format_futhark(),
            arg1.dtype.format_futhark(),
            arg0.dtype.format_futhark(),
        ));
        self.append(format!(r"let {{%2}} = f_outer {{%0}} {{%1}} {{%0.s*}} {{%1.s*}} {{%2.s*}} in"));
      }
      _ => {
        println!("WARNING: FutharkThunkCode::append_nd_broadcast_map2: not implemented: {:?} {:?}", arg0, arg1);
        return Err(FutharkGenErr::NotImpl);
      }
    }
    Ok(())
  }*/

  /*pub fn nd_broadcast_map2<S: Borrow<str>>(abi: &mut FutAbi, arg0: Dim, arg1: Dim, lam: S) -> Result<FutharkThunkCode, FutharkGenErr> {
    abi.push_out_arr(0, FutAbiOutput::Pure, FutAbiArrayRepr::Nd, FutAbiScalarType::Unspec);
    abi.push_arg_arr(0, FutAbiArrayRepr::Nd, FutAbiScalarType::Unspec);
    abi.push_arg_arr(1, FutAbiArrayRepr::Nd, FutAbiScalarType::Unspec);
    let mut code = FutharkThunkCode::default();
    code.append_nd_broadcast_map2(arg0, arg1, lam)?;
    code.into()
  }

  pub fn append_nd_broadcast_map2<S: Borrow<str>>(&mut self, arg0: Dim, arg1: Dim, lam: S) -> Result<(), FutharkGenErr> {
    let lam = lam.borrow();
    let dtype = arg0.dtype.max(arg1.dtype).unwrap();
    match (arg0.ndim(), arg1.ndim()) {
      (2, 2) => {
        self.cfg.emit_arg_shapes = true;
        self.cfg.emit_out0_shape = true;
        self.cfg.emit_out0_shape_param = true;
        self.pre_append(format!(r"def f0_inner t_0 {{%0.s**}} {{%2.s**}} = if {{%0.s[1]}} == {{%2.s[1]}} then t_0 :> [{{%2.s[1]}}]{} else (\t -> replicate {{%2.s[1]}} t[0]) t_0",
            arg0.dtype.format_futhark(),
        ));
        self.pre_append(format!(r"def f0_outer t_0 {{%0.s**}} {{%2.s**}} = if {{%0.s[0]}} == {{%2.s[0]}} then (\t -> map (\u -> f0_inner u {{%0.s*}} {{%2.s*}}) t) t_0 :> [{{%2.s[0]}}][{{%2.s[1]}}]{} else (\t -> replicate {{%2.s[0]}} (f0_inner t[0] {{%0.s*}} {{%2.s*}})) t_0",
            arg0.dtype.format_futhark(),
        ));
        self.pre_append(format!(r"def f1_inner t_0 {{%0.s**}} {{%2.s**}} = if {{%0.s[1]}} == {{%2.s[1]}} then t_0 :> [{{%2.s[1]}}]{} else (\t -> replicate {{%2.s[1]}} t[0]) t_0",
            arg1.dtype.format_futhark(),
        ));
        self.pre_append(format!(r"def f1_outer t_0 {{%0.s**}} {{%2.s**}} = if {{%0.s[0]}} == {{%2.s[0]}} then (\t -> map (\u -> f1_inner u {{%0.s*}} {{%2.s*}}) t) t_0 :> [{{%2.s[0]}}][{{%2.s[1]}}]{} else (\t -> replicate {{%2.s[0]}} (f1_inner t[0] {{%0.s*}} {{%2.s*}})) t_0",
            arg1.dtype.format_futhark(),
        ));
        self.append(format!(r"let t0 = flatten (f0_outer {{%0}} {{%0.s*}} {{%2.s*}}) in"));
        self.append(format!(r"let t1 = flatten (f1_outer {{%1}} {{%1.s*}} {{%2.s*}}) in"));
        self.append(format!(r"let t2 = map2 ({}) t0 t1 in", lam));
        self.append(format!(r"let {{%2}} = unflatten t2 in"));
      }
      (3, 3) => {
        self.cfg.emit_arg_shapes = true;
        self.cfg.emit_out0_shape = true;
        self.cfg.emit_out0_shape_param = true;
        self.pre_append(format!(r"def f0_dim_2 t_0 {{%0.s**}} {{%2.s**}} = if {{%0.s[2]}} == {{%2.s[2]}} then t_0 :> [{{%2.s[2]}}]{} else (\t -> replicate {{%2.s[2]}} t[0]) t_0",
            arg0.dtype.format_futhark(),
        ));
        self.pre_append(format!(r"def f0_dim_1 t_0 {{%0.s**}} {{%2.s**}} = if {{%0.s[1]}} == {{%2.s[1]}} then (\t -> map (\u -> f0_dim_2 u {{%0.s*}} {{%2.s*}}) t) t_0 :> [{{%2.s[1]}}][{{%2.s[2]}}]{} else (\t -> replicate {{%2.s[1]}} (f0_dim_2 t[0] {{%0.s*}} {{%2.s*}})) t_0",
            arg0.dtype.format_futhark(),
        ));
        self.pre_append(format!(r"def f0_outer t_0 {{%0.s**}} {{%2.s**}} = if {{%0.s[0]}} == {{%2.s[0]}} then (\t -> map (\u -> f0_dim_1 u {{%0.s*}} {{%2.s*}}) t) t_0 :> [{{%2.s[0]}}][{{%2.s[1]}}][{{%2.s[2]}}]{} else (\t -> replicate {{%2.s[0]}} (f0_dim_1 t[0] {{%0.s*}} {{%2.s*}})) t_0",
            arg0.dtype.format_futhark(),
        ));
        self.pre_append(format!(r"def f1_dim_2 t_0 {{%0.s**}} {{%2.s**}} = if {{%0.s[2]}} == {{%2.s[2]}} then t_0 :> [{{%2.s[2]}}]{} else (\t -> replicate {{%2.s[2]}} t[0]) t_0",
            arg1.dtype.format_futhark(),
        ));
        self.pre_append(format!(r"def f1_dim_1 t_0 {{%0.s**}} {{%2.s**}} = if {{%0.s[1]}} == {{%2.s[1]}} then (\t -> map (\u -> f1_dim_2 u {{%0.s*}} {{%2.s*}}) t) t_0 :> [{{%2.s[1]}}][{{%2.s[2]}}]{} else (\t -> replicate {{%2.s[1]}} (f1_dim_2 t[0] {{%0.s*}} {{%2.s*}})) t_0",
            arg1.dtype.format_futhark(),
        ));
        self.pre_append(format!(r"def f1_outer t_0 {{%0.s**}} {{%2.s**}} = if {{%0.s[0]}} == {{%2.s[0]}} then (\t -> map (\u -> f1_dim_1 u {{%0.s*}} {{%2.s*}}) t) t_0 :> [{{%2.s[0]}}][{{%2.s[1]}}][{{%2.s[2]}}]{} else (\t -> replicate {{%2.s[0]}} (f1_dim_1 t[0] {{%0.s*}} {{%2.s*}})) t_0",
            arg1.dtype.format_futhark(),
        ));
        self.append(format!(r"let t0 = flatten_3d (f0_outer {{%0}} {{%0.s*}} {{%2.s*}}) in"));
        self.append(format!(r"let t1 = flatten_3d (f1_outer {{%1}} {{%1.s*}} {{%2.s*}}) in"));
        self.append(format!(r"let t2 = map2 ({}) t0 t1 in", lam));
        self.append(format!(r"let {{%2}} = unflatten_3d t2 in"));
      }
      (4, 4) => {
        self.cfg.emit_arg_shapes = true;
        self.cfg.emit_out0_shape = true;
        self.cfg.emit_out0_shape_param = true;
        self.pre_append(format!(r"def f0_dim_3 t_0 {{%0.s**}} {{%2.s**}} = if {{%0.s[3]}} == {{%2.s[3]}} then t_0 :> [{{%2.s[3]}}]{} else (\t -> replicate {{%2.s[3]}} t[0]) t_0",
            arg0.dtype.format_futhark(),
        ));
        self.pre_append(format!(r"def f0_dim_2 t_0 {{%0.s**}} {{%2.s**}} = if {{%0.s[2]}} == {{%2.s[2]}} then (\t -> map (\u -> f0_dim_3 u {{%0.s*}} {{%2.s*}}) t) t_0 :> [{{%2.s[2]}}][{{%2.s[3]}}]{} else (\t -> replicate {{%2.s[2]}} (f0_dim_3 t[0] {{%0.s*}} {{%2.s*}})) t_0",
            arg0.dtype.format_futhark(),
        ));
        self.pre_append(format!(r"def f0_dim_1 t_0 {{%0.s**}} {{%2.s**}} = if {{%0.s[1]}} == {{%2.s[1]}} then (\t -> map (\u -> f0_dim_2 u {{%0.s*}} {{%2.s*}}) t) t_0 :> [{{%2.s[1]}}][{{%2.s[2]}}][{{%2.s[3]}}]{} else (\t -> replicate {{%2.s[1]}} (f0_dim_2 t[0] {{%0.s*}} {{%2.s*}})) t_0",
            arg0.dtype.format_futhark(),
        ));
        self.pre_append(format!(r"def f0_outer t_0 {{%0.s**}} {{%2.s**}} = if {{%0.s[0]}} == {{%2.s[0]}} then (\t -> map (\u -> f0_dim_1 u {{%0.s*}} {{%2.s*}}) t) t_0 :> [{{%2.s[0]}}][{{%2.s[1]}}][{{%2.s[2]}}][{{%2.s[3]}}]{} else (\t -> replicate {{%2.s[0]}} (f0_dim_1 t[0] {{%0.s*}} {{%2.s*}})) t_0",
            arg0.dtype.format_futhark(),
        ));
        self.pre_append(format!(r"def f1_dim_3 t_0 {{%0.s**}} {{%2.s**}} = if {{%0.s[3]}} == {{%2.s[3]}} then t_0 :> [{{%2.s[3]}}]{} else (\t -> replicate {{%2.s[3]}} t[0]) t_0",
            arg1.dtype.format_futhark(),
        ));
        self.pre_append(format!(r"def f1_dim_2 t_0 {{%0.s**}} {{%2.s**}} = if {{%0.s[2]}} == {{%2.s[2]}} then (\t -> map (\u -> f1_dim_3 u {{%0.s*}} {{%2.s*}}) t) t_0 :> [{{%2.s[2]}}][{{%2.s[3]}}]{} else (\t -> replicate {{%2.s[2]}} (f1_dim_3 t[0] {{%0.s*}} {{%2.s*}})) t_0",
            arg1.dtype.format_futhark(),
        ));
        self.pre_append(format!(r"def f1_dim_1 t_0 {{%0.s**}} {{%2.s**}} = if {{%0.s[1]}} == {{%2.s[1]}} then (\t -> map (\u -> f1_dim_2 u {{%0.s*}} {{%2.s*}}) t) t_0 :> [{{%2.s[1]}}][{{%2.s[2]}}][{{%2.s[3]}}]{} else (\t -> replicate {{%2.s[1]}} (f1_dim_2 t[0] {{%0.s*}} {{%2.s*}})) t_0",
            arg1.dtype.format_futhark(),
        ));
        self.pre_append(format!(r"def f1_outer t_0 {{%0.s**}} {{%2.s**}} = if {{%0.s[0]}} == {{%2.s[0]}} then (\t -> map (\u -> f1_dim_1 u {{%0.s*}} {{%2.s*}}) t) t_0 :> [{{%2.s[0]}}][{{%2.s[1]}}][{{%2.s[2]}}][{{%2.s[3]}}]{} else (\t -> replicate {{%2.s[0]}} (f1_dim_1 t[0] {{%0.s*}} {{%2.s*}})) t_0",
            arg1.dtype.format_futhark(),
        ));
        self.append(format!(r"let t0 = flatten_4d (f0_outer {{%0}} {{%0.s*}} {{%2.s*}}) in"));
        self.append(format!(r"let t1 = flatten_4d (f1_outer {{%1}} {{%1.s*}} {{%2.s*}}) in"));
        self.append(format!(r"let t2 = map2 ({}) t0 t1 in", lam));
        self.append(format!(r"let {{%2}} = unflatten_4d t2 in"));
      }
      _ => {
        println!("WARNING: FutharkThunkCode::append_nd_broadcast_map2: not implemented: {:?} {:?}", arg0, arg1);
        return Err(FutharkGenErr::NotImpl);
      }
    }
    Ok(())
  }*/

  pub fn pre_append<S: Into<String>>(&mut self, line: S) {
    self.head.push(line.into());
  }

  pub fn append<S: Into<String>>(&mut self, line: S) {
    self.body.push(line.into());
  }

  pub fn gen_source(&self, abi: &FutAbi, spec_dim: &[Dim], mode: ThunkMode, mut cfg: FutharkThunkBuildConfig) -> Result<Vec<String>, ()> {
    // TODO
    cfg = self.cfg.merge(cfg);
    let mut pats = Vec::new();
    let mut reps = Vec::new();
    for k in 0 .. abi.arityin {
      pats.push(format!(r"{{%{}}}", k));
      reps.push(format!(r"x_{}", k));
      if cfg.emit_arg_shapes {
        let dim = spec_dim[k as usize];
        for d in 0 .. dim.ndim {
          pats.push(format!(r"{{%{}.s[{}]}}", k, d));
          reps.push(format!(r"x_{}_s_{}", k, d));
        }
        pats.push(format!(r"{{%{}.s*}}", k));
        let mut r = String::new();
        for d in 0 .. dim.ndim {
          if d > 0 {
            write!(&mut r, r" ").unwrap();
          }
          write!(&mut r, r"x_{}_s_{}", k, d).unwrap();
        }
        reps.push(r);
        pats.push(format!(r"{{%{}.s**}}", k));
        let mut r = String::new();
        for d in 0 .. dim.ndim {
          if d > 0 {
            write!(&mut r, r" ").unwrap();
          }
          write!(&mut r, r"(x_{}_s_{}: i64)", k, d).unwrap();
        }
        reps.push(r);
      }
    }
    for k in 0 .. abi.arityout {
      pats.push(format!(r"{{%{}}}", abi.arityin + k));
      reps.push(format!(r"y_{}", k));
      if k == 0 && cfg.emit_out0_shape {
        let dim = spec_dim[abi.arityin as usize];
        for d in 0 .. dim.ndim {
          pats.push(format!(r"{{%{}.s[{}]}}", abi.arityin + k, d));
          reps.push(format!(r"y_{}_s_{}", k, d));
        }
        pats.push(format!(r"{{%{}.s*}}", abi.arityin + k));
        let mut r = String::new();
        for d in 0 .. dim.ndim {
          if d > 0 {
            write!(&mut r, r" ").unwrap();
          }
          write!(&mut r, r"y_{}_s_{}", k, d).unwrap();
        }
        reps.push(r);
        pats.push(format!(r"{{%{}.s**}}", abi.arityin + k));
        let mut r = String::new();
        for d in 0 .. dim.ndim {
          if d > 0 {
            write!(&mut r, r" ").unwrap();
          }
          write!(&mut r, r"(y_{}_s_{}: i64)", k, d).unwrap();
        }
        reps.push(r);
      }
    }
    assert_eq!(pats.len(), reps.len());
    let matcher = AhoCorasick::new(&pats).unwrap();
    let mut s = String::new();
    let mut out_buf = Vec::new();
    for line in self.head.iter() {
      out_buf.clear();
      matcher.try_stream_replace_all(line.as_bytes(), &mut out_buf, &reps).unwrap();
      let out_line = from_utf8(&out_buf).unwrap();
      s.push_str(out_line);
      write!(&mut s, "\n").unwrap();
    }
    drop(out_buf);
    if cfg.emit_primal_def {
      write!(&mut s, "def primal").unwrap();
    } else {
      write!(&mut s, "entry kernel").unwrap();
    }
    if cfg.emit_arg_shapes {
      for k in 0 .. abi.arityin {
        let dim = spec_dim[k as usize];
        for d in 0 .. dim.ndim {
          write!(&mut s, " [x_{}_s_{}]", k, d).unwrap();
        }
      }
    }
    if cfg.emit_out0_shape && !cfg.emit_out0_shape_param {
      assert_eq!(abi.arityout, 1);
      let dim = spec_dim[abi.arityin as usize];
      for d in 0 .. dim.ndim {
        write!(&mut s, " [y_{}_s_{}]", 0, d).unwrap();
      }
    }
    for k in 0 .. abi.arityin {
      let dim = spec_dim[k as usize];
      if cfg.emit_arg_shapes {
        write!(&mut s, " (x_{}: {})", k, _to_futhark_entry_arg_type(dim, k)).unwrap();
      } else {
        write!(&mut s, " (x_{}: {})", k, _to_futhark_entry_type(dim)).unwrap();
      }
    }
    if abi.arityout == 1 {
      match mode {
        ThunkMode::Apply0 => {
          let dim = spec_dim[abi.arityin as usize];
          if cfg.emit_out0_shape_param {
            let np = abi.num_param();
            for d in 0 .. dim.ndim {
              // FIXME FIXME: don't push param here...
              //abi.push_param(np as u16 + d as u16, FutAbiScalarType::I64);
              write!(&mut s, " (y_{}_s_{}: i64)", 0, d).unwrap();
            }
            // FIXME
            //abi.set_emit_out0_shape_param(true);
          }
          if cfg.emit_out0_shape {
            write!(&mut s, " : {}", _to_futhark_entry_out0_type(dim)).unwrap();
          } else {
            write!(&mut s, " : {}", _to_futhark_entry_type(dim)).unwrap();
          }
        }
        //ThunkMode::Apply1 |
        ThunkMode::Accumulate => {
          let dim = spec_dim[abi.arityin as usize];
          if dim.ndim >= 1 && !cfg.emit_out0_shape {
            cfg.emit_out0_shape = true;
            return self.gen_source(abi, spec_dim, mode, cfg);
          }
          let fty = if cfg.emit_out0_shape {
            _to_futhark_entry_out0_type(dim)
          } else {
            _to_futhark_entry_type(dim)
          };
          write!(&mut s, " (oy_{}: *{}) : *{}", 0, fty, fty).unwrap();
        }
        _ => unimplemented!()
      }
    } else if mode == ThunkMode::Apply0 {
      write!(&mut s, " : (").unwrap();
      for k in 0 .. abi.arityout {
        let dim = spec_dim[(abi.arityin + k) as usize];
        if k == 0 && cfg.emit_out0_shape {
          write!(&mut s, "{}, ", _to_futhark_entry_out0_type(dim)).unwrap();
        } else {
          write!(&mut s, "{}, ", _to_futhark_entry_type(dim)).unwrap();
        }
      }
      write!(&mut s, ")").unwrap();
    } else {
      unimplemented!();
    }
    write!(&mut s, " =\n").unwrap();
    for k in 0 .. abi.arityin {
      let dim = spec_dim[k as usize];
      if dim.ndim == 0 {
        write!(&mut s, "\tlet x_{} = x_{}[0] in\n", k, k).unwrap();
      }
    }
    if abi.arityout == 1 {
      match mode {
        ThunkMode::Apply0 => {}
        //ThunkMode::Apply1 |
        ThunkMode::Accumulate => {
          let dim = spec_dim[abi.arityin as usize];
          if dim.ndim == 0 {
            write!(&mut s, "\tlet oy_{} = oy_{}[0] in\n", 0, 0).unwrap();
          }
        }
        _ => unimplemented!()
      }
    }
    let mut out_buf = Vec::new();
    for line in self.body.iter() {
      out_buf.clear();
      matcher.try_stream_replace_all(line.as_bytes(), &mut out_buf, &reps).unwrap();
      let out_line = from_utf8(&out_buf).unwrap();
      write!(&mut s, "\t").unwrap();
      s.push_str(out_line);
      write!(&mut s, "\n").unwrap();
    }
    drop(out_buf);
    if abi.arityout == 1 {
      match mode {
        ThunkMode::Apply0 => {}
        /*ThunkMode::Apply1 => {
          let dim = spec_dim[abi.arityin as usize];
          if dim.ndim >= 1 {
            assert!(cfg.emit_out0_shape);
            let fty = _to_futhark_entry_out0_type(dim);
            write!(&mut s, "\tlet y_{} = y_{} :> {} in\n", 0, 0, fty).unwrap();
          }
        }*/
        ThunkMode::Accumulate => {
          let dim = spec_dim[abi.arityin as usize];
          if dim.ndim >= 1 {
            assert!(cfg.emit_out0_shape);
          }
          match dim.ndim {
            0 => {
              write!(&mut s, "\tlet y_{} = oy_{} + y_{} in\n", 0, 0, 0).unwrap();
            }
            1 => {
              let fty = _to_futhark_entry_out0_type(dim);
              write!(&mut s, "\tlet y_{} = map2 (+) oy_{} (y_{} :> {}) in\n", 0, 0, 0, fty).unwrap();
            }
            2 => {
              let fty = _to_futhark_entry_out0_type(dim);
              write!(&mut s, "\tlet oy_{} = flatten oy_{} in\n", 0, 0).unwrap();
              write!(&mut s, "\tlet y_{} = flatten (y_{} :> {}) in\n", 0, 0, fty).unwrap();
              write!(&mut s, "\tlet y_{} = map2 (+) oy_{} y_{} in\n", 0, 0, 0).unwrap();
              //write!(&mut s, "\tlet y_{} = unflatten y_{}_s_0 y_{}_s_1 y_{} in\n", 0, 0, 0, 0).unwrap();
              write!(&mut s, "\tlet y_{} = unflatten y_{} in\n", 0, 0).unwrap();
            }
            3 => {
              let fty = _to_futhark_entry_out0_type(dim);
              write!(&mut s, "\tlet oy_{} = flatten_3d oy_{} in\n", 0, 0).unwrap();
              write!(&mut s, "\tlet y_{} = flatten_3d (y_{} :> {}) in\n", 0, 0, fty).unwrap();
              write!(&mut s, "\tlet y_{} = map2 (+) oy_{} y_{} in\n", 0, 0, 0).unwrap();
              //write!(&mut s, "\tlet y_{} = unflatten_3d y_{}_s_0 y_{}_s_1 y_{}_s_2 y_{} in\n", 0, 0, 0, 0, 0).unwrap();
              write!(&mut s, "\tlet y_{} = unflatten_3d y_{} in\n", 0, 0).unwrap();
            }
            4 => {
              let fty = _to_futhark_entry_out0_type(dim);
              write!(&mut s, "\tlet oy_{} = flatten_4d oy_{} in\n", 0, 0).unwrap();
              write!(&mut s, "\tlet y_{} = flatten_4d (y_{} :> {}) in\n", 0, 0, fty).unwrap();
              write!(&mut s, "\tlet y_{} = map2 (+) oy_{} y_{} in\n", 0, 0, 0).unwrap();
              //write!(&mut s, "\tlet y_{} = unflatten_4d y_{}_s_0 y_{}_s_1 y_{}_s_2 y_{}_s_3 y_{} in\n", 0, 0, 0, 0, 0, 0).unwrap();
              write!(&mut s, "\tlet y_{} = unflatten_4d y_{} in\n", 0, 0).unwrap();
            }
            _ => unimplemented!()
          }
        }
        _ => unimplemented!()
      }
    }
    for k in 0 .. abi.arityout {
      let dim = spec_dim[(abi.arityin + k) as usize];
      if dim.ndim == 0 {
        write!(&mut s, "\tlet y_{} = [y_{}] in\n", k, k).unwrap();
      }
    }
    write!(&mut s, "\t").unwrap();
    if abi.arityout == 1 {
      write!(&mut s, "y_{}", 0).unwrap();
    } else if mode == ThunkMode::Apply0 {
      write!(&mut s, "(").unwrap();
      for k in 0 .. abi.arityout {
        write!(&mut s, "y_{}, ", k).unwrap();
      }
      write!(&mut s, " )").unwrap();
    } else {
      panic!("bug");
    }
    write!(&mut s, "\n").unwrap();
    // FIXME FIXME
    let mut lines_out = Vec::new();
    //let lines_out = vec![s];
    for line in s.split('\n') {
      lines_out.push(line.into());
    }
    Ok(lines_out)
  }
}

pub struct FutharkThunkObject<B: FutBackend> {
  pub obj:      FutObject<B>,
  pub consts:   Vec<(PAddr, StableCell)>,
  pub out0_tag: Option<(u32, TagUnifier)>,
}

pub trait FutharkThunkImpl_<B: FutBackend> {
  fn _dropck(&mut self);
  unsafe fn _setup_object(obj: &mut FutObject<B>);
  fn _build_object(ctr: &CtxCtr, env: &mut CtxEnv, config: &FutConfig, name: Option<&str>, source: &str, rst: Counter) -> Option<(FutObject<B>, Vec<(PAddr, StableCell)>)>;
}

pub struct FutharkThunkImpl<B: FutBackend> where FutharkThunkImpl<B>: FutharkThunkImpl_<B> {
  pub abi:      FutAbi,
  pub param:    Vec<FutAbiScalar>,
  pub spec_dim: Vec<Dim>,
  pub code:     FutharkThunkCode,
  pub name:     Option<&'static str>,
  pub source:   RefCell<String>,
  pub objects:  RefCell<SortMap8<ThunkMode, FutharkThunkObject<B>>>,
}

impl FutharkThunkImpl_<MulticoreBackend> for FutharkThunkImpl<MulticoreBackend> {
  fn _dropck(&mut self) {
  }

  unsafe fn _setup_object(obj: &mut FutObject<MulticoreBackend>) {
    if cfg_debug() { println!("DEBUG: FutharkThunkImpl::<MulticoreBackend>::_setup_object: cfg..."); }
    obj.new_config();
    assert!(!obj.cfg.is_null());
    TL_PCTX.with(|pctx| {
      // FIXME: proper pctx multiplexing.
      let mut override_smp = false;
      #[cfg(feature = "nvgpu")]
      if pctx.nvgpu.is_some() {
        (obj.ffi.base().ctx_cfg_set_mem_alloc.as_ref().unwrap())(obj.cfg, tl_pctx_nvgpu_mem_alloc_hook as *const c_void as _);
        (obj.ffi.base().ctx_cfg_set_mem_free.as_ref().unwrap())(obj.cfg, tl_pctx_nvgpu_mem_free_hook as *const c_void as _);
        (obj.ffi.base().ctx_cfg_set_mem_unify.as_ref().unwrap())(obj.cfg, tl_pctx_nvgpu_mem_unify_hook as *const c_void as _);
        override_smp = true;
      }
      if !override_smp {
        (obj.ffi.base().ctx_cfg_set_mem_alloc.as_ref().unwrap())(obj.cfg, tl_pctx_smp_mem_alloc_hook as *const c_void as _);
        (obj.ffi.base().ctx_cfg_set_mem_free.as_ref().unwrap())(obj.cfg, tl_pctx_smp_mem_free_hook as *const c_void as _);
        (obj.ffi.base().ctx_cfg_set_mem_unify.as_ref().unwrap())(obj.cfg, tl_pctx_smp_mem_unify_hook as *const c_void as _);
      }
      obj.set_num_threads(pctx.smp.phy_core_ct() as _);
    });
    // TODO
    if cfg_debug() { println!("DEBUG: FutharkThunkImpl::<MulticoreBackend>::_setup_object: cfg done"); }
    if cfg_debug() { println!("DEBUG: FutharkThunkImpl::<MulticoreBackend>::_setup_object: ctx..."); }
    obj.new_context();
    assert!(!obj.ctx.is_null());
    if cfg_debug() { println!("DEBUG: FutharkThunkImpl::<MulticoreBackend>::_setup_object: ctx done"); }
  }

  fn _build_object(_ctr: &CtxCtr, _env: &mut CtxEnv, _config: &FutConfig, _name: Option<&str>, _source: &str, _rst: Counter) -> Option<(FutObject<MulticoreBackend>, Vec<(PAddr, StableCell)>)> {
    // FIXME FIXME
    None
  }
}

#[cfg(feature = "nvgpu")]
impl FutharkThunkImpl_<CudaBackend> for FutharkThunkImpl<CudaBackend> {
  fn _dropck(&mut self) {
    /*assert!(LIBCUDA._inner.is_some());
    assert!(LIBCUDART._inner.is_some());
    assert!(LIBNVRTC._inner.is_some());*/
  }

  unsafe fn _setup_object(obj: &mut FutObject<CudaBackend>) {
    if cfg_debug() { println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_setup_object: cfg..."); }
    obj.new_config();
    assert!(!obj.cfg.is_null());
    TL_CFG_ENV.with(|cfg| {
      if !cfg.no_kcache {
        if let Some(kcache_path) = obj.kcache_path() {
          if cfg_debug() {
          println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_setup_object: kcache path={:?}",
              kcache_path.to_str().map(|s| safe_ascii(s.as_bytes())).unwrap());
          }
          obj.set_cache_file(kcache_path);
        }
      }
    });
    TL_PCTX.with(|pctx| {
      let gpu = pctx.nvgpu.as_ref().unwrap();
      gpu.compute.sync().unwrap();
      obj.set_setup_device(gpu.dev());
      obj.set_setup_stream(gpu.compute.as_ptr() as *mut _);
    });
    (obj.ffi.base().ctx_cfg_set_mem_alloc.as_ref().unwrap())(obj.cfg, tl_pctx_nvgpu_mem_alloc_hook as *const c_void as _);
    (obj.ffi.base().ctx_cfg_set_mem_free.as_ref().unwrap())(obj.cfg, tl_pctx_nvgpu_mem_free_hook as *const c_void as _);
    (obj.ffi.base().ctx_cfg_set_mem_unify.as_ref().unwrap())(obj.cfg, tl_pctx_nvgpu_mem_unify_hook as *const c_void as _);
    (obj.ffi.ctx_cfg_set_gpu_alloc.as_ref().unwrap())(obj.cfg, tl_pctx_gpu_alloc_hook as *const c_void as _);
    (obj.ffi.ctx_cfg_set_gpu_free.as_ref().unwrap())(obj.cfg, tl_pctx_gpu_free_hook as *const c_void as _);
    (obj.ffi.ctx_cfg_set_gpu_unify.as_ref().unwrap())(obj.cfg, tl_pctx_gpu_unify_hook as *const c_void as _);
    (obj.ffi.ctx_cfg_set_gpu_global_failure_alloc.as_ref().unwrap())(obj.cfg, tl_pctx_gpu_failarg_alloc_hook as *const c_void as _);
    (obj.ffi.ctx_cfg_set_gpu_global_failure_free.as_ref().unwrap())(obj.cfg, tl_pctx_gpu_failarg_free_hook as *const c_void as _);
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
    (obj.ffi.ctx_cfg_set_cuStreamSynchronize.as_ref().unwrap())(obj.cfg, LIBCUDA.cuStreamSynchronize.as_ref().unwrap().as_ptr() as _);
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
    // TODO
    if cfg_debug() { println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_setup_object: cfg done"); }
    if cfg_debug() { println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_setup_object: ctx..."); }
    obj.new_context();
    assert!(!obj.ctx.is_null());
    TL_PCTX.with(|pctx| {
      let gpu = pctx.nvgpu.as_ref().unwrap();
      gpu.compute.sync().unwrap();
    });
    if cfg_debug() { println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_setup_object: ctx done"); }
  }

  fn _build_object(ctr: &CtxCtr, env: &mut CtxEnv, config: &FutConfig, name: Option<&str>, source: &str, rst: Counter) -> Option<(FutObject<CudaBackend>, Vec<(PAddr, StableCell)>)> {
    assert!(TL_LIBNVRTC_BUILTINS_BARRIER.with(|&bar| bar));
    TL_PCTX.with(|pctx| {
      let dev = pctx.nvgpu.as_ref().unwrap().dev();
      cudart_set_cur_dev(dev).unwrap();
    });
    let t0 = Stopwatch::tl_stamp();
    match config.build::<CudaBackend>(FutStage::Dylib, name, source.as_bytes()) {
      Err(e) => {
        println!("WARNING: FutharkThunkImpl::<CudaBackend>::_build_object: build error: {:?}", e);
        None
      }
      Ok(None) => panic!("bug"),
      Ok(Some(mut obj)) => {
        let t1 = Stopwatch::tl_stamp();
        if cfg_debug() { println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_build_object:   build elapsed: {:.09} s", t1 - t0); }
        let t0 = t1;
        // NB: futhark object ctx may create constants that need to be tracked.
        let mut consts = Vec::new();
        let pstart = TL_PCTX.with(|pctx| {
          let gpu = pctx.nvgpu.as_ref().unwrap();
          gpu.mem_pool.borrow().set_back_alloc(true);
          gpu.mem_pool.borrow().set_alloc_pin(true);
          pctx.ctr.next_addr()
        });
        unsafe { FutharkThunkImpl::<CudaBackend>::_setup_object(&mut obj); }
        let t1 = Stopwatch::tl_stamp();
        if cfg_debug() { println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_build_object:   setup elapsed: {:.09} s", t1 - t0); }
        let pfin = TL_PCTX.with(|pctx| {
          let gpu = pctx.nvgpu.as_ref().unwrap();
          gpu.mem_pool.borrow().set_back_alloc(false);
          gpu.mem_pool.borrow().set_alloc_pin(false);
          pctx.ctr.peek_addr()
        });
        for p in (pstart.to_unchecked() ..= pfin.to_unchecked()) {
          let p = PAddr::from_unchecked(p);
          let x = ctr.fresh_cel();
          if cfg_debug() { println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_build_object: const: {:?} {:?}", p, x); }
          // FIXME: futhark consts should be marked pin.
          TL_PCTX.with(|pctx| {
            // FIXME: the type of the constant could probably be inferred,
            // but easier to defer it until unification.
            let ty = CellType::top();
            let mut pcel = PCell::new(x, ty.clone());
            let pmach = PMach::NvGpu;
            let locus = match pctx.nvgpu.as_ref().unwrap().lookup_reg(p) {
              None => panic!("bug"),
              Some(reg) => reg.locus()
            };
            let base_clk: Clock = rst.into();
            let xclk = base_clk.init_once();
            pcel.push_new_replica(x, xclk, locus, pmach, p);
            env.insert_phy(x, ty, pcel);
          });
          consts.push((p, StableCell::retain(env, x)));
          // FIXME: what else?
        }
        if consts.len() > 0 {
          if cfg_debug() { println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_build_object: consts={:?}", consts); }
        }
        Some((obj, consts))
      }
    }
  }
}

impl<B: FutBackend> Drop for FutharkThunkImpl<B> where FutharkThunkImpl<B>: FutharkThunkImpl_<B> {
  fn drop(&mut self) {
    self.objects.borrow_mut().clear();
    /*// FIXME FIXME: in case of custom dropck that takes &mut CtxEnv.
    let mut consts = Vec::new();
    swap(&mut *self.consts.borrow_mut(), &mut consts);
    for x in consts.into_iter() {
      x.release(_);
    }*/
    self._dropck();
  }
}

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

pub fn _to_futhark_entry_arg_type(dim: Dim, k: u16) -> String {
  let mut s = String::new();
  if dim.ndim == 0 {
    s.push_str("[1]");
  }
  for i in 0 .. dim.ndim {
    write!(&mut s, "[x_{}_s_{}]", k, i).unwrap();
  }
  s.push_str(dim.dtype.format_futhark());
  s
}

pub fn _to_futhark_entry_out0_type(dim: Dim) -> String {
  let mut s = String::new();
  if dim.ndim == 0 {
    s.push_str("[1]");
  }
  for i in 0 .. dim.ndim {
    write!(&mut s, "[y_{}_s_{}]", 0, i).unwrap();
  }
  s.push_str(dim.dtype.format_futhark());
  s
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct FutharkThunkBuildConfig {
  // FIXME
  pub emit_arg_shapes: bool,
  pub emit_out0_shape: bool,
  pub emit_out0_shape_param: bool,
  pub emit_primal_def: bool,
}

impl Default for FutharkThunkBuildConfig {
  fn default() -> FutharkThunkBuildConfig {
    FutharkThunkBuildConfig{
      // FIXME
      emit_arg_shapes: false,
      emit_out0_shape: false,
      emit_out0_shape_param: false,
      emit_primal_def: false,
    }
  }
}

impl FutharkThunkBuildConfig {
  pub fn merge(self, rhs: FutharkThunkBuildConfig) -> FutharkThunkBuildConfig {
    FutharkThunkBuildConfig{
      emit_arg_shapes:  self.emit_arg_shapes || rhs.emit_arg_shapes,
      emit_out0_shape:  self.emit_out0_shape || rhs.emit_out0_shape,
      emit_out0_shape_param:  self.emit_out0_shape_param || rhs.emit_out0_shape_param,
      emit_primal_def:  self.emit_primal_def || rhs.emit_primal_def,
    }
  }
}

impl<B: FutBackend> FutharkThunkImpl<B> where FutharkThunkImpl<B>: FutharkThunkImpl_<B> {
  pub fn _try_build(&self, ctr: &CtxCtr, env: &mut CtxEnv, mode: ThunkMode, mut cfg: FutharkThunkBuildConfig, rst: Counter) {
    cfg = self.code.cfg.merge(cfg);
    let mut pats = Vec::new();
    let mut reps = Vec::new();
    for k in 0 .. self.abi.arityin {
      pats.push(format!(r"{{%{}}}", k));
      reps.push(format!(r"x_{}", k));
      if cfg.emit_arg_shapes {
        let dim = self.spec_dim[k as usize];
        for d in 0 .. dim.ndim {
          pats.push(format!(r"{{%{}.s[{}]}}", k, d));
          reps.push(format!(r"x_{}_s_{}", k, d));
        }
        pats.push(format!(r"{{%{}.s*}}", k));
        let mut r = String::new();
        for d in 0 .. dim.ndim {
          if d > 0 {
            write!(&mut r, r" ").unwrap();
          }
          write!(&mut r, r"x_{}_s_{}", k, d).unwrap();
        }
        reps.push(r);
        pats.push(format!(r"{{%{}.s**}}", k));
        let mut r = String::new();
        for d in 0 .. dim.ndim {
          if d > 0 {
            write!(&mut r, r" ").unwrap();
          }
          write!(&mut r, r"(x_{}_s_{}: i64)", k, d).unwrap();
        }
        reps.push(r);
      }
    }
    for k in 0 .. self.abi.arityout {
      pats.push(format!(r"{{%{}}}", self.abi.arityin + k));
      reps.push(format!(r"y_{}", k));
      if k == 0 && cfg.emit_out0_shape {
        let dim = self.spec_dim[self.abi.arityin as usize];
        for d in 0 .. dim.ndim {
          pats.push(format!(r"{{%{}.s[{}]}}", self.abi.arityin + k, d));
          reps.push(format!(r"y_{}_s_{}", k, d));
        }
        pats.push(format!(r"{{%{}.s*}}", self.abi.arityin + k));
        let mut r = String::new();
        for d in 0 .. dim.ndim {
          if d > 0 {
            write!(&mut r, r" ").unwrap();
          }
          write!(&mut r, r"y_{}_s_{}", k, d).unwrap();
        }
        reps.push(r);
        pats.push(format!(r"{{%{}.s**}}", self.abi.arityin + k));
        let mut r = String::new();
        for d in 0 .. dim.ndim {
          if d > 0 {
            write!(&mut r, r" ").unwrap();
          }
          write!(&mut r, r"(y_{}_s_{}: i64)", k, d).unwrap();
        }
        reps.push(r);
      }
    }
    assert_eq!(pats.len(), reps.len());
    let matcher = AhoCorasick::new(&pats).unwrap();
    let mut s = String::new();
    let mut out_buf = Vec::new();
    for line in self.code.head.iter() {
      out_buf.clear();
      matcher.try_stream_replace_all(line.as_bytes(), &mut out_buf, &reps).unwrap();
      let out_line = from_utf8(&out_buf).unwrap();
      s.push_str(out_line);
      write!(&mut s, "\n").unwrap();
    }
    drop(out_buf);
    write!(&mut s, "entry kernel").unwrap();
    if cfg.emit_arg_shapes {
      for k in 0 .. self.abi.arityin {
        let dim = self.spec_dim[k as usize];
        for d in 0 .. dim.ndim {
          write!(&mut s, " [x_{}_s_{}]", k, d).unwrap();
        }
      }
    }
    if cfg.emit_out0_shape && !cfg.emit_out0_shape_param {
      assert_eq!(self.abi.arityout, 1);
      let dim = self.spec_dim[self.abi.arityin as usize];
      for d in 0 .. dim.ndim {
        write!(&mut s, " [y_{}_s_{}]", 0, d).unwrap();
      }
    }
    for k in 0 .. self.abi.arityin {
      let dim = self.spec_dim[k as usize];
      if cfg.emit_arg_shapes {
        write!(&mut s, " (x_{}: {})", k, _to_futhark_entry_arg_type(dim, k)).unwrap();
      } else {
        write!(&mut s, " (x_{}: {})", k, _to_futhark_entry_type(dim)).unwrap();
      }
    }
    if self.abi.arityout == 1 {
      match mode {
        ThunkMode::Apply0 => {
          let dim = self.spec_dim[self.abi.arityin as usize];
          if cfg.emit_out0_shape_param {
            let np = self.abi.num_param();
            for d in 0 .. dim.ndim {
              // FIXME FIXME: don't push param here...
              //self.abi.push_param(np as u16 + d as u16, FutAbiScalarType::I64);
              write!(&mut s, " (y_{}_s_{}: i64)", 0, d).unwrap();
            }
            // FIXME
            //self.abi.set_emit_out0_shape_param(true);
          }
          if cfg.emit_out0_shape {
            write!(&mut s, " : {}", _to_futhark_entry_out0_type(dim)).unwrap();
          } else {
            write!(&mut s, " : {}", _to_futhark_entry_type(dim)).unwrap();
          }
        }
        //ThunkMode::Apply1 |
        ThunkMode::Accumulate => {
          let dim = self.spec_dim[self.abi.arityin as usize];
          if dim.ndim >= 1 && !cfg.emit_out0_shape {
            cfg.emit_out0_shape = true;
            return self._try_build(ctr, env, mode, cfg, rst);
          }
          let fty = if cfg.emit_out0_shape {
            _to_futhark_entry_out0_type(dim)
          } else {
            _to_futhark_entry_type(dim)
          };
          write!(&mut s, " (oy_{}: *{}) : *{}", 0, fty, fty).unwrap();
        }
        _ => unimplemented!()
      }
    } else if mode == ThunkMode::Apply0 {
      write!(&mut s, " : (").unwrap();
      for k in 0 .. self.abi.arityout {
        let dim = self.spec_dim[(self.abi.arityin + k) as usize];
        if k == 0 && cfg.emit_out0_shape {
          write!(&mut s, "{}, ", _to_futhark_entry_out0_type(dim)).unwrap();
        } else {
          write!(&mut s, "{}, ", _to_futhark_entry_type(dim)).unwrap();
        }
      }
      write!(&mut s, ")").unwrap();
    } else {
      unimplemented!();
    }
    write!(&mut s, " =\n").unwrap();
    for k in 0 .. self.abi.arityin {
      let dim = self.spec_dim[k as usize];
      if dim.ndim == 0 {
        write!(&mut s, "\tlet x_{} = x_{}[0] in\n", k, k).unwrap();
      }
    }
    if self.abi.arityout == 1 {
      match mode {
        ThunkMode::Apply0 => {}
        //ThunkMode::Apply1 |
        ThunkMode::Accumulate => {
          let dim = self.spec_dim[self.abi.arityin as usize];
          if dim.ndim == 0 {
            write!(&mut s, "\tlet oy_{} = oy_{}[0] in\n", 0, 0).unwrap();
          }
        }
        _ => unimplemented!()
      }
    }
    let mut out_buf = Vec::new();
    for line in self.code.body.iter() {
      out_buf.clear();
      matcher.try_stream_replace_all(line.as_bytes(), &mut out_buf, &reps).unwrap();
      let out_line = from_utf8(&out_buf).unwrap();
      write!(&mut s, "\t").unwrap();
      s.push_str(out_line);
      write!(&mut s, "\n").unwrap();
    }
    drop(out_buf);
    if self.abi.arityout == 1 {
      match mode {
        ThunkMode::Apply0 => {}
        /*ThunkMode::Apply1 => {
          let dim = self.spec_dim[self.abi.arityin as usize];
          if dim.ndim >= 1 {
            assert!(cfg.emit_out0_shape);
            let fty = _to_futhark_entry_out0_type(dim);
            write!(&mut s, "\tlet y_{} = y_{} :> {} in\n", 0, 0, fty).unwrap();
          }
        }*/
        ThunkMode::Accumulate => {
          let dim = self.spec_dim[self.abi.arityin as usize];
          if dim.ndim >= 1 {
            assert!(cfg.emit_out0_shape);
          }
          match dim.ndim {
            0 => {
              write!(&mut s, "\tlet y_{} = oy_{} + y_{} in\n", 0, 0, 0).unwrap();
            }
            1 => {
              let fty = _to_futhark_entry_out0_type(dim);
              write!(&mut s, "\tlet y_{} = map2 (+) oy_{} (y_{} :> {}) in\n", 0, 0, 0, fty).unwrap();
            }
            2 => {
              let fty = _to_futhark_entry_out0_type(dim);
              write!(&mut s, "\tlet oy_{} = flatten oy_{} in\n", 0, 0).unwrap();
              write!(&mut s, "\tlet y_{} = flatten (y_{} :> {}) in\n", 0, 0, fty).unwrap();
              write!(&mut s, "\tlet y_{} = map2 (+) oy_{} y_{} in\n", 0, 0, 0).unwrap();
              //write!(&mut s, "\tlet y_{} = unflatten y_{}_s_0 y_{}_s_1 y_{} in\n", 0, 0, 0, 0).unwrap();
              write!(&mut s, "\tlet y_{} = unflatten y_{} in\n", 0, 0).unwrap();
            }
            3 => {
              let fty = _to_futhark_entry_out0_type(dim);
              write!(&mut s, "\tlet oy_{} = flatten_3d oy_{} in\n", 0, 0).unwrap();
              write!(&mut s, "\tlet y_{} = flatten_3d (y_{} :> {}) in\n", 0, 0, fty).unwrap();
              write!(&mut s, "\tlet y_{} = map2 (+) oy_{} y_{} in\n", 0, 0, 0).unwrap();
              //write!(&mut s, "\tlet y_{} = unflatten_3d y_{}_s_0 y_{}_s_1 y_{}_s_2 y_{} in\n", 0, 0, 0, 0, 0).unwrap();
              write!(&mut s, "\tlet y_{} = unflatten_3d y_{} in\n", 0, 0).unwrap();
            }
            4 => {
              let fty = _to_futhark_entry_out0_type(dim);
              write!(&mut s, "\tlet oy_{} = flatten_4d oy_{} in\n", 0, 0).unwrap();
              write!(&mut s, "\tlet y_{} = flatten_4d (y_{} :> {}) in\n", 0, 0, fty).unwrap();
              write!(&mut s, "\tlet y_{} = map2 (+) oy_{} y_{} in\n", 0, 0, 0).unwrap();
              //write!(&mut s, "\tlet y_{} = unflatten_4d y_{}_s_0 y_{}_s_1 y_{}_s_2 y_{}_s_3 y_{} in\n", 0, 0, 0, 0, 0, 0).unwrap();
              write!(&mut s, "\tlet y_{} = unflatten_4d y_{} in\n", 0, 0).unwrap();
            }
            _ => unimplemented!()
          }
        }
        _ => unimplemented!()
      }
    }
    for k in 0 .. self.abi.arityout {
      let dim = self.spec_dim[(self.abi.arityin + k) as usize];
      if dim.ndim == 0 {
        write!(&mut s, "\tlet y_{} = [y_{}] in\n", k, k).unwrap();
      }
    }
    write!(&mut s, "\t").unwrap();
    if self.abi.arityout == 1 {
      write!(&mut s, "y_{}", 0).unwrap();
    } else if mode == ThunkMode::Apply0 {
      write!(&mut s, "(").unwrap();
      for k in 0 .. self.abi.arityout {
        write!(&mut s, "y_{}, ", k).unwrap();
      }
      write!(&mut s, " )").unwrap();
    } else {
      panic!("bug");
    }
    write!(&mut s, "\n").unwrap();
    *self.source.borrow_mut() = s;
    let mut config = FutConfig::default();
    // FIXME FIXME: os-specific paths.
    config.cachedir = home_dir().unwrap().join(".cacti").join("cache");
    TL_CFG_ENV.with(|cfg| {
      if let Some(path) = cfg.cabalpath.first() {
        config.futhark = path.join("cacti-futhark");
      }
      if let Some(prefix) = cfg.cudaprefix.first() {
        config.include = prefix.join("include");
      }
      if cfg.debug >= 3 {
        config.verbose = true;
      }
      if cfg.debug >= 1 {
        config.debug = true;
      }
    });
    if let Some((obj, consts)) = FutharkThunkImpl::<B>::_build_object(ctr, env, &config, self.name, &*self.source.borrow(), rst) {
      //*self.object.borrow_mut() = Some(obj);
      //*self.consts.borrow_mut() = consts;
      /*for key in self.modekeys.borrow().iter() {
        if key.key == mode {
          let key2 = key.clone();
          self.object.borrow_mut().swap(&key2, &mut obj);
          self.consts.borrow_mut().swap(key, &mut consts);
          assert_eq!(key, &key2);
          return;
        }
      }
      let key2 = self.object.borrow_mut().insert(mode, obj);
      let key = self.consts.borrow_mut().insert(mode, consts);
      assert_eq!(&key, &key2);
      self.modekeys.borrow_mut().push(key);*/
      // FIXME: or swap.
      let object = FutharkThunkObject{obj, consts, out0_tag: None};
      self.objects.borrow_mut().insert(mode, object);
    }
  }
}

impl ThunkImpl for FutharkThunkImpl<MulticoreBackend> {
  fn apply(&self, ctr: &CtxCtr, env: &mut CtxEnv, spec_: &dyn ThunkSpec_, arg: &[(CellPtr, Clock)], th: ThunkPtr, out: CellPtr, oclk: Clock) -> ThunkResult {
    // FIXME
    if self.objects.borrow().find(ThunkMode::Apply0).is_none() {
      self._try_build(ctr, env, ThunkMode::Apply0, FutharkThunkBuildConfig::default(), oclk.ctr());
    }
    if self.objects.borrow().find(ThunkMode::Apply0).is_none() {
      panic!("bug: FutharkThunkImpl::<MulticoreBackend>::apply: build error");
    }
    unimplemented!();
  }
}

#[cfg(feature = "nvgpu")]
impl FutharkThunkImpl<CudaBackend> {
  pub fn _enter(&self, ctr: &CtxCtr, env: &mut CtxEnv, spec_: &dyn ThunkSpec_, arg: &[(CellPtr, Clock)], th: ThunkPtr, out: CellPtr, oclk: Clock, mode: ThunkMode) -> ThunkResult {
    if cfg_debug() {
    println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: name={:?} mode={:?}",
        spec_.debug_name(), mode);
    }
    if self.objects.borrow().find(mode).is_none() {
      self._try_build(ctr, env, mode, FutharkThunkBuildConfig::default(), oclk.ctr());
    }
    if self.objects.borrow().find(mode).is_none() {
      panic!("BUG: FutharkThunkImpl::<CudaBackend>::_enter: build error");
    }
    assert_eq!(arg.len(), self.abi.arityin as usize);
    let extrain = match mode {
      ThunkMode::Apply0 => {
        if 1 != self.abi.arityout {
          unimplemented!();
        }
        0
      }
      /*ThunkMode::Apply1 => {
        assert_eq!(1, self.abi.arityout);
        1
      }*/
      ThunkMode::Accumulate => {
        assert_eq!(1, self.abi.arityout);
        1
      }
      _ => panic!("bug")
    };
    //assert_eq!(extra, 0);
    if cfg_debug() { println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: arg={:?}", arg); }
    let mut arg_ty_: Vec<CellType> = Vec::with_capacity((self.abi.arityin + extrain) as usize);
    let mut arg_arr: Vec<UnsafeCell<FutArrayDev>> = Vec::with_capacity((self.abi.arityin + extrain) as usize);
    'for_k: for k in 0 .. self.abi.arityin as usize {
      let xroot = match env.lookup_ref(arg[k].0) {
        None => panic!("bug"),
        Some(e) => e.root
      };
      for j in 0 .. k {
        match env.lookup_ref(arg[j].0) {
          None => panic!("bug"),
          Some(e_j) => {
            let xroot_j = e_j.root;
            if xroot_j == xroot {
              if cfg_debug() {
              println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: aliased args: xroot[{}]={:?} xroot[{}]={:?}",
                  j, xroot_j, k, xroot);
              }
              match env.lookup_ref(arg[k].0) {
                None => panic!("bug"),
                Some(e) => {
                  if &arg_ty_[j] == e.ty {
                    arg_ty_.push(arg_ty_[j].clone());
                    let a = arg_arr[j].get_mut().clone();
                    arg_arr.push(a.into());
                  } else {
                    println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: aliased args: xty[{}]={:?} xty[{}]={:?}",
                        j, &arg_ty_[j], k, e.ty);
                    unimplemented!();
                  }
                  continue 'for_k;
                }
              }
            }
          }
        }
      }
      let e = match env.pread_ref(arg[k].0, arg[k].1) {
        None => panic!("bug"),
        Some(e) => e
      };
      assert_eq!(self.spec_dim[k], e.ty.to_dim());
      let a = match self.spec_dim[k].ndim {
        0 | 1 => FutArrayDev::new_1d(),
        2 => FutArrayDev::new_2d(),
        3 => FutArrayDev::new_3d(),
        4 => FutArrayDev::new_4d(),
        _ => unimplemented!()
      };
      TL_PCTX.with(|pctx| {
        let gpu = pctx.nvgpu.as_ref().unwrap();
        let loc = gpu.device_locus();
        match e.cel_ {
          &mut Cell_::Phy(.., ref mut pcel) => {
            let addr = pcel.get(arg[k].0, arg[k].1, &e.ty, loc, PMach::NvGpu);
            let (dptr, size) = gpu.lookup_dev(addr).unwrap();
            a.set_mem_parts(dptr, size);
          }
          _ => panic!("bug")
        }
      });
      if self.spec_dim[k].ndim == 0 {
        a.set_shape(&[1]);
      } else {
        a.set_shape(&e.ty.shape);
      }
      arg_ty_.push(e.ty);
      arg_arr.push(a.into());
    }
    //println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: arg_arr={:?}", &arg_arr);
    let mut arg_ndim = Vec::with_capacity(arg_arr.len());
    for (k, arr) in arg_arr.iter_mut().enumerate() {
      if cfg_debug() { println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: arg_arr[{}]={:?}", k, arr.get_mut()); }
      arg_ndim.push(arr.get_mut()._unset_ndim());
    }
    if cfg_debug() { println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: out={:?} oclk={:?}", out, oclk); }
    let mut out_ty_ = Vec::with_capacity(self.abi.arityout as usize);
    let mut out_arr: Vec<UnsafeCell<FutArrayDev>> = Vec::with_capacity(self.abi.arityout as usize);
    for k in 0 .. self.abi.arityout as usize {
      assert_eq!(k, 0);
      let ty_ = match env.lookup_ref(out) {
        None => panic!("bug"),
        Some(e) => {
          if e.ty.is_top() {
            match spec_.out_ty_(&arg_ty_) {
              Err(ThunkTypeErr::Nondeterm) => panic!("bug: nondeterm type"),
              Err(_) => panic!("bug"),
              Ok(ty_) => ty_
            }
          } else {
            e.ty.clone()
          }
        }
      };
      //let ty_ = &out_ty_[k];
      assert_eq!(self.spec_dim[self.abi.arityin as usize + k], ty_.to_dim());
      match mode {
        ThunkMode::Accumulate => {
          // FIXME: double check that out does not alias any args.
          let e = match env.pwrite_ref(out, oclk) {
            None => panic!("bug"),
            Some(e) => e
          };
          assert_eq!(self.spec_dim[self.abi.arityin as usize + k], e.ty.to_dim());
          let a = match self.spec_dim[self.abi.arityin as usize + k].ndim {
            0 | 1 => FutArrayDev::new_1d(),
            2 => FutArrayDev::new_2d(),
            3 => FutArrayDev::new_3d(),
            4 => FutArrayDev::new_4d(),
            _ => unimplemented!()
          };
          TL_PCTX.with(|pctx| {
            let gpu = pctx.nvgpu.as_ref().unwrap();
            let loc = gpu.device_locus();
            match e.cel_ {
              &mut Cell_::Phy(.., ref mut pcel) => {
                let addr = pcel.get(out, oclk, &e.ty, loc, PMach::NvGpu);
                let (dptr, size) = gpu.lookup_dev(addr).unwrap();
                a.set_mem_parts(dptr, size);
              }
              _ => panic!("bug")
            }
          });
          if self.spec_dim[self.abi.arityin as usize + k].ndim == 0 {
            a.set_shape(&[1]);
          } else {
            a.set_shape(&e.ty.shape);
          }
          arg_ty_.push(ty_.clone());
          arg_arr.push(a.into());
        }
        _ => {}
      }
      out_ty_.push(ty_);
      out_arr.push(FutArrayDev::null().into());
    }
    for (k, arr) in (&mut arg_arr[self.abi.arityin as usize ..]).iter_mut().enumerate() {
      if cfg_debug() {
      println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: arg_arr[{}]={:?} (out in-place)",
          self.abi.arityin as usize + k, arr.get_mut());
      }
      arg_ndim.push(arr.get_mut()._unset_ndim());
    }
    //println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: out_arr={:?}", &out_arr);
    for (k, arr) in out_arr.iter_mut().enumerate() {
      if cfg_debug() { println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: out_arr[{}]={:?}", k, arr.get_mut()); }
    }
    // FIXME FIXME: param.
    let np = self.abi.num_param();
    let mut param: Vec<FutAbiScalar> = Vec::with_capacity(np);
    //param.resize(np, FutAbiScalar::Empty);
    //spec_.set_param(&mut param);
    if np == 0 && self.param.len() == 0 {
    } else if np == 1 && self.param.len() == 0 {
      // FIXME FIXME: hack.
      param.push(FutAbiScalar::I64(out_ty_[0].shape[0].into()));
    } else if np == 2 && self.param.len() == 0 {
      // FIXME FIXME: hack.
      param.push(FutAbiScalar::I64(out_ty_[0].shape[0].into()));
      param.push(FutAbiScalar::I64(out_ty_[0].shape[1].into()));
    } else if np == 3 && self.param.len() == 0 {
      // FIXME FIXME: hack.
      param.push(FutAbiScalar::I64(out_ty_[0].shape[0].into()));
      param.push(FutAbiScalar::I64(out_ty_[0].shape[1].into()));
      param.push(FutAbiScalar::I64(out_ty_[0].shape[2].into()));
    } else if np == 4 && self.param.len() == 0 {
      // FIXME FIXME: hack.
      param.push(FutAbiScalar::I64(out_ty_[0].shape[0].into()));
      param.push(FutAbiScalar::I64(out_ty_[0].shape[1].into()));
      param.push(FutAbiScalar::I64(out_ty_[0].shape[2].into()));
      param.push(FutAbiScalar::I64(out_ty_[0].shape[3].into()));
    } else {
      unimplemented!();
    }
    let restore_out = match mode {
      ThunkMode::Accumulate => {
        let (out, rep, dty) = self.abi.get_out_arr(0);
        assert_eq!(out, FutAbiOutput::Pure);
        let _ = self.abi.set_out_arr(0, FutAbiOutput::ImplicitInPlace, rep, dty);
        Some((out, rep, dty))
      }
      _ => None
    };
    let mut objects = self.objects.borrow_mut();
    let mut object = objects.find_mut(mode).unwrap().1;
    let &mut FutharkThunkObject{ref mut obj, ref mut out0_tag, ..} = &mut object;
    if cfg_debug() { println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: hash={}", &obj.src_hash); }
    TL_PCTX.with(|pctx| {
      let gpu = pctx.nvgpu.as_ref().unwrap();
      gpu.compute.sync().unwrap();
      obj.set_stream(gpu.compute.as_ptr() as *mut _);
    });
    let t0 = Stopwatch::tl_stamp();
    obj.reset();
    let tmp_t1 = Stopwatch::tl_stamp();
    if cfg_debug() { println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter:   reset elapsed: {:.09} s", tmp_t1 - t0); }
    // FIXME FIXME: pre-entry setup.
    /*obj.unify_abi(self.abi).unwrap();*/
    if cfg_debug() { println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: enter kernel..."); }
    let (pre_front_dptr, pre_backoffset) = TL_PCTX.with(|pctx| {
      let gpu = pctx.nvgpu.as_ref().unwrap();
      //gpu.compute.sync().unwrap();
      if let Some(&mut (tag, ref mut unify)) = out0_tag.as_mut() {
        // FIXME FIXME: rather than swap, start a new unifier
        // and verify that it matches the existing one.
        swap(&mut *pctx.tagunify.borrow_mut(), unify);
        gpu.mem_pool.set_front_tag(Some(tag));
      } else {
        pctx.tagunify.borrow_mut().reset();
      }
      gpu.mem_pool.set_back_alloc(true);
      gpu.mem_pool.free_list.borrow_mut().clear();
      (gpu.mem_pool.front_dptr(), gpu.mem_pool.back_offset())
    });
    let o_ret = obj.enter_kernel(&self.abi, &param, &arg_arr, &out_arr);
    if o_ret.is_err() {
      // FIXME FIXME: error handling.
      panic!("BUG: FutharkThunkImpl::<CudaBackend>::_enter: runtime error");
    }
    let may_fail = obj.may_fail();
    if cfg_debug() { println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: may fail? {:?}", may_fail); }
    if may_fail {
      let ret = obj.sync();
      if ret.is_err() {
        // FIXME FIXME: failure handling.
        println!("ERROR: FutharkThunkImpl::<CudaBackend>::_enter: runtime failure: {:?}", ret);
        if let Some(e) = obj.error().map(|c| safe_ascii(c.to_bytes())) {
          println!("ERROR: FutharkThunkImpl::<CudaBackend>::_enter: runtime failure: {}", e);
        }
        panic!();
      }
    }
    obj.release();
    let (post_front_dptr, post_backoffset) = TL_PCTX.with(|pctx| {
      let gpu = pctx.nvgpu.as_ref().unwrap();
      gpu.compute.sync().unwrap();
      if let Some(&mut (_, ref mut unify)) = out0_tag.as_mut() {
        // FIXME FIXME: rather than swap, start a new unifier
        // and verify that it matches the existing one.
        swap(&mut *pctx.tagunify.borrow_mut(), unify);
        gpu.mem_pool.set_front_tag(None);
      }
      gpu.mem_pool.set_back_alloc(false);
      if !gpu.mem_pool.free_list.borrow().is_empty() {
        if cfg_debug() { println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: free={:?}", &*gpu.mem_pool.free_list.borrow()); }
      }
      (gpu.mem_pool.front_dptr(), gpu.mem_pool.back_offset())
    });
    let t1 = Stopwatch::tl_stamp();
    if cfg_debug() { println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter:   elapsed: {:.09} s", t1 - t0); }
    TL_CTX.with(|ctx| {
      if oclk.rst <= 0 {
        panic!("bug");
      } else if oclk.rst == 1 {
        ctx.timing.futhark1.borrow_mut().push(t1 - t0);
      } else {
        ctx.timing.futhark.borrow_mut().push(t1 - t0);
      }
    });
    drop(obj);
    if cfg_debug() { println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: ret={:?}", o_ret); }
    //println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: out={:?} oclk={:?}", out, oclk);
    match mode {
      ThunkMode::Accumulate => {
        let (out, rep, dty) = restore_out.unwrap();
        let _ = self.abi.set_out_arr(0, out, rep, dty);
      }
      _ => {}
    }
    for (k, arr) in arg_arr.iter_mut().enumerate() {
      arr.get_mut()._set_ndim(arg_ndim[k]);
      if cfg_debug() { println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: arg_arr[{}]={:?}", k, arr.get_mut()); }
    }
    // FIXME: at this point, the remaining memblocks are the outputs.
    // but, if any of the inputs were clobbered, then we have to unset those.
    // so, do some kind of unification here.
    for k in 0 .. self.abi.arityout as usize {
      assert!(!out_arr[k].get_mut().as_ptr().is_null());
      out_arr[k].get_mut()._set_ndim(max(1, out_ty_[k].ndim()));
    }
    //println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: out_arr={:?}", &out_arr);
    for (k, arr) in out_arr.iter_mut().enumerate() {
      if cfg_debug() { println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: out_arr[{}]={:?}", k, arr.get_mut()); }
    }
    if cfg_debug() { println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: out: shape={:?}", out_arr[0].get_mut().shape().unwrap()); }
    if out_ty_[0].ndim() == 0 {
      assert_eq!(&[1], out_arr[0].get_mut().shape().unwrap());
    } else {
      assert_eq!(&out_ty_[0].shape, out_arr[0].get_mut().shape().unwrap());
    }
    // TODO TODO
    let (mem_dptr, mem_size) = out_arr[0].get_mut().mem_parts().unwrap();
    if cfg_debug() { println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: out: rc={:?} dptr=0x{:016x} size={}", out_arr[0].get_mut().refcount(), mem_dptr, mem_size); }
    if mem_dptr == pre_front_dptr {
      if cfg_debug() { println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: out:   no fragmentation"); }
    } else if mem_dptr > pre_front_dptr {
      if cfg_debug() { println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: out:   fragmentation sz={}", mem_dptr - pre_front_dptr); }
    } else {
      if cfg_debug() { println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: out:   internal allocation; pre_front_dptr=0x{:016x}", pre_front_dptr); }
      /*match mode {
        ThunkMode::Accumulate => {}
        _ => {
          unimplemented!();
        }
      }*/
    }
    match out_arr[0].get_mut().tag() {
      None => {
        if cfg_debug() { println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: out:   tag=null"); }
      }
      Some(ctag) => {
        if cfg_debug() { println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: out:   tag=\"{}\"", safe_ascii(ctag.to_bytes())); }
        let tag = TagUnifier::parse_tag(ctag.to_bytes()).unwrap();
        match out0_tag.as_ref() {
          None => {
            TL_PCTX.with(|pctx| {
              let unify = pctx.tagunify.borrow().clone();
              *out0_tag = Some((tag, unify));
            });
          }
          Some(&(otag, _)) => {
            assert_eq!(otag, tag);
          }
        }
      }
    }
    TL_PCTX.with(|pctx| {
      let gpu = pctx.nvgpu.as_ref().unwrap();
      match gpu.mem_pool.rev_lookup(mem_dptr) {
        None => {
          // FIXME FIXME
          panic!("bug");
        }
        Some((_region, None)) => {
          // FIXME FIXME
          panic!("bug");
        }
        Some((region, Some(p))) => {
          if cfg_debug() {
          println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: out: region={:?} addr={:?} root={:?}",
              region, p, pctx.lookup_root(p));
          }
          let mut f = false;
          if !f {
            //for k in self.consts.borrow().iter() {}
            for k in objects.find(mode).unwrap().1.consts.iter() {
              if k.0 == p {
                if cfg_debug() { println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: out:   is const"); }
                match env.lookup_mut_ref(out) {
                  None => panic!("bug"),
                  Some(e) => {
                    match e.cel_ {
                      &mut Cell_::Top(ref state, optr) => {
                        // FIXME: defaults below are placeholders for...?
                        let state = RefCell::new(state.borrow().clone());
                        let clo = RefCell::new(CellClosure::default());
                        *e.cel_ = Cell_::Cow(state, clo, CowCell{optr, pcel: *(k.1).as_ref(), pclk: Clock::default()});
                        f = true;
                        if cfg_debug() { println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: out: cow {:?} -> {:?}", out, p); }
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
          /*if !f && mode == ThunkMode::Apply1 {
            // FIXME FIXME
            unimplemented!();
          }*/
          if !f /*&& mode == ThunkMode::Apply0 */{
            match env.lookup_mut_ref(out) {
              None => panic!("bug"),
              Some(e) => {
                match e.cel_ {
                  &mut Cell_::Top(ref state, optr) => {
                    if cfg_debug() { println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: out: try new phy..."); }
                    assert_eq!(e.root, optr);
                    // FIXME: defaults below are placeholders for...?
                    let state = RefCell::new(state.borrow().clone());
                    let clo = RefCell::new(CellClosure::default());
                    let mut pcel = PCell::new(optr, out_ty_[0].clone());
                    pcel.push_new_replica(optr, oclk, Locus::VMem, PMach::NvGpu, p);
                    *e.cel_ = Cell_::Phy(state, clo, pcel);
                    f = true;
                    if cfg_debug() {
                    println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: out: new phy {:?} --> {:?} -> {:?}",
                        out, optr, p);
                    }
                  }
                  // FIXME FIXME
                  &mut Cell_::Phy(ref state, .., ref mut pcel) => {
                    if cfg_debug() { println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: out: try old phy..."); }
                    let optr = pcel.optr;
                    if let Some(replica) = pcel.lookup(Locus::VMem, PMach::NvGpu) {
                      // NB: clk equal b/c we did clock_sync.
                      assert_eq!(replica.clk.get(), oclk);
                      // FIXME: gc.
                      //let _ = replica.addr.get();
                      replica.addr.set(p);
                    } else {
                      pcel.push_new_replica(optr, oclk, Locus::VMem, PMach::NvGpu, p);
                    }
                    f = true;
                    if cfg_debug() {
                    println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: out: old phy {:?} --> {:?} -> {:?}",
                        out, optr, p);
                    }
                  }
                  _ => unimplemented!()
                }
              }
            }
            let p_out = p;
            if cfg_debug() {
            println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: out: pre front dptr=0x{:016x}",
                pre_front_dptr);
            println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: out: postfront dptr=0x{:016x}",
                post_front_dptr);
            println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: out: pre backoffset=0x{:016x}",
                pre_backoffset);
            println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: out: postbackoffset=0x{:016x}",
                post_backoffset);
            }
            if pre_front_dptr == post_front_dptr &&
               mem_dptr > pre_front_dptr
            {
              gpu.mem_pool.front_relocate(p_out, post_front_dptr, &gpu.compute);
              if cfg_debug() {
              println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: out: relocate src=0x{:016x} dst=0x{:016x}",
                  mem_dptr, post_front_dptr);
              }
            }
            let mut free_list = gpu.mem_pool.free_list.borrow_mut();
            free_list.sort();
            loop {
              let p = match free_list.pop() {
                None => break,
                Some(p) => p
              };
              assert!(p != p_out);
              let icel = gpu.mem_pool.try_free(p).unwrap();
              assert!(InnerCell::root(&*icel).is_none());
              assert!(icel.back());
            }
            gpu.mem_pool.set_back_offset(pre_backoffset);
          }
          /*if !f && mode == ThunkMode::Accumulate {
            // TODO
            let _ = arg_arr[self.abi.arityin as usize].get_mut().take_ptr();
            f = true;
          }*/
          if !f {
            panic!("bug");
          }
        }
      }
    /*let ret = gpu.compute.sync();
    if ret.is_err() {
      println!("ERROR: FutharkThunkImpl::<CudaBackend>::_enter: cuda stream sync failed: {:?}", ret);
      println!("ERROR: FutharkThunkImpl::<CudaBackend>::_enter: source:");
      println!("{}", &*self.source.borrow());
      panic!();
    }
    unsafe {
      let mut dst_buf: Vec<u8> = Vec::with_capacity(mem_size + 4096);
      dst_buf.set_len(mem_size + 4096);
      let mut dst_ptr = dst_buf.as_mut_ptr() as usize;
      dst_ptr = (dst_ptr + (4096 - 1)) / 4096 * 4096;
      /*let ret = cuda_memcpy_d2h_async(dst_buf.as_mut_ptr(), mem_dptr, mem_size, &CudartStream::null());*/
      let ret = cuda_memcpy_d2h(dst_ptr as *mut _, mem_dptr, mem_size);
      if ret.is_err() {
        println!("ERROR: FutharkThunkImpl::<CudaBackend>::_enter: cuda memcpy failed: {:?}", ret);
        println!("ERROR: FutharkThunkImpl::<CudaBackend>::_enter:   dst ptr=0x{:016x}", dst_ptr);
        panic!();
      }
      let out_val = *(dst_buf.as_ptr() as *const f32);
      println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: out: val={:?}", out_val);
    }*/
    });
    Ok(())
  }
}

#[cfg(feature = "nvgpu")]
impl ThunkImpl for FutharkThunkImpl<CudaBackend> {
  fn apply(&self, ctr: &CtxCtr, env: &mut CtxEnv, spec_: &dyn ThunkSpec_, arg: &[(CellPtr, Clock)], th: ThunkPtr, out: CellPtr, oclk: Clock) -> ThunkResult {
    let mode = ThunkMode::Apply0;
    self._enter(ctr, env, spec_, arg, th, out, oclk, mode)
  }

  fn accumulate(&self, ctr: &CtxCtr, env: &mut CtxEnv, spec_: &dyn ThunkSpec_, arg: &[(CellPtr, Clock)], th: ThunkPtr, out: CellPtr, oclk: Clock) -> ThunkResult {
    let mode = ThunkMode::Accumulate;
    self._enter(ctr, env, spec_, arg, th, out, oclk, mode)
    /*if self.objects.borrow().find(mode).is_none() {
      self._try_build(ctr, env, mode, FutharkThunkBuildConfig::default(), oclk.ctr());
    }
    if self.objects.borrow().find(mode).is_none() {
      panic!("bug: FutharkThunkImpl::<CudaBackend>::accumulate: build error");
    }
    // FIXME FIXME
    unimplemented!();*/
    /*
    assert_eq!(arg.len(), self.abi.arityin as usize);
    assert_eq!(1, self.abi.arityout);
    let mut arg_ty_ = Vec::with_capacity((self.abi.arityin + 1) as usize);
    let mut arg_arr = Vec::with_capacity((self.abi.arityin + 1) as usize);
    for k in 0 .. self.abi.arityin as usize {
      let ty_ = env.lookup_ref(arg[k].0).unwrap().ty.clone();
      assert_eq!(self.spec_dim[k], ty_.to_dim());
      let a = match self.spec_dim[k].ndim {
        0 | 1 => FutArrayDev::new_1d(),
        2 => FutArrayDev::new_2d(),
        3 => FutArrayDev::new_3d(),
        4 => FutArrayDev::new_4d(),
        _ => unimplemented!()
      };
      // FIXME FIXME: actually init the array.
      /*
      a.set_mem_parts(...);
      */
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
        let k = self.abi.arityin as usize;
        assert_eq!(self.spec_dim[k], ty_.to_dim());
        let a = match self.spec_dim[k].ndim {
          0 | 1 => FutArrayDev::new_1d(),
          2 => FutArrayDev::new_2d(),
          3 => FutArrayDev::new_3d(),
          4 => FutArrayDev::new_4d(),
          _ => unimplemented!()
        };
        // FIXME FIXME: actually init the array.
        /*
        a.set_mem_parts(...);
        */
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
    //let mut out_raw_arr = Vec::with_capacity(1);
    let mut out_arr = Vec::with_capacity(1);
    for k in 0 .. 1 {
      let ty_ = &out_ty_[k];
      assert_eq!(self.spec_dim[self.abi.arityin as usize + k], ty_.to_dim());
      // FIXME FIXME
      //out_raw_arr.push(null_mut());
      out_arr.push(FutArrayDev::null());
    }
    /*let mut obj = self.object.borrow_mut();
    let obj = obj.as_mut().unwrap();*/
    /*let mut objs = self.object.borrow_mut();
    let obj = objs.find_mut(mode).unwrap().1;*/
    let mut objects = self.objects.borrow_mut();
    let obj = &mut objects.find_mut(mode).unwrap().1.obj;
    // FIXME FIXME: pre-entry setup.
    obj.reset();
    /*obj.unify_abi(self.abi).unwrap();*/
    let o_ret = obj.enter_kernel(&self.abi, &self.param, &arg_arr, &mut out_arr);
    //let o_ret = obj.enter_kernel(self.abi.arityin + 1, 1, &self.param, &arg_arr, &mut out_arr);
    if o_ret.is_err() || (obj.may_fail() && obj.sync().is_err()) {
      // FIXME FIXME: error handling.
      panic!("bug: FutharkThunkImpl::<CudaBackend>::accumulate: runtime error");
    }
    println!("DEBUG: FutharkThunkImpl::<CudaBackend>::accumulate: out={:?}", out);
    drop(obj);
    // FIXME: because of uniqueness, the lone output should the same memblock as
    // the last input; so, make sure not to double free.
    //let mut out_arr = Vec::with_capacity(1);
    //for (k, raw) in out_raw_arr.into_iter().enumerate() {}
    for k in 0 .. 1 {
      //out_arr.push(FutArrayDev::from_raw(raw, max(1, out_ty_[k].ndim())));
      assert!(!out_arr[0].as_ptr().is_null());
      out_arr[0]._set_ndim(max(1, out_ty_[k].ndim()));
    }
    /*assert_eq!(arg_arr[self.abi.arityin as usize].as_ptr(), out_arr[0].as_ptr());*/
    /*let (out_ptr, out_ndim) = arg_arr.pop().unwrap().into_raw();*/
    let out_ndim = arg_arr.last().unwrap().ndim();
    let out_ptr = arg_arr.last_mut().unwrap().take_ptr();
    assert_eq!(out_ptr, out_arr[0].as_ptr());
    assert_eq!(out_ndim, out_arr[0].ndim());
    // FIXME FIXME
    unimplemented!();
    */
  }

  fn initialize(&self, ctr: &CtxCtr, env: &mut CtxEnv, spec_: &dyn ThunkSpec_, arg: &[(CellPtr, Clock)], th: ThunkPtr, out: CellPtr, oclk: Clock) -> ThunkResult {
    let mode = ThunkMode::Initialize;
    self._enter(ctr, env, spec_, arg, th, out, oclk, mode)
  }
}

// TODO

pub struct PThunk {
  pub ptr:      ThunkPtr,
  pub clk:      Clock,
  pub arityin:  u16,
  pub arityout: u16,
  pub spec_dim: Vec<Dim>,
  pub spec_:    Rc<dyn ThunkSpec_>,
  // TODO TODO
  //pub impl_:    RefCell<Vec<(PMach, Rc<dyn ThunkImpl_>)>>,
  pub impl_:    RefCell<RevSortMap8<PMach, Rc<dyn ThunkImpl_>>>,
}

impl PThunk {
  pub fn new(ptr: ThunkPtr, spec_dim: Vec<Dim>, spec_: Rc<dyn ThunkSpec_>) -> PThunk {
    let clk = Clock::default();
    let (arityin, arityout) = spec_.arity();
    assert_eq!(spec_dim.len(), (arityin + arityout) as usize);
    let impl_ = RefCell::new(RevSortMap8::new());
    PThunk{
      ptr,
      clk,
      arityin,
      arityout,
      spec_dim,
      spec_,
      //impl_: RefCell::new(Vec::new()),
      impl_,
    }
  }

  pub fn push_new_impl_(&self, pmach: PMach, thimpl_: Rc<dyn ThunkImpl_>) {
    /*for &(o_pmach, _) in self.impl_.borrow().iter() {
      if o_pmach == pmach {
        panic!("bug");
      }
    }
    let mut impl_ = self.impl_.borrow_mut();
    impl_.push((pmach, thimpl_));
    impl_.sort_by(|&(l_pmach, _), &(r_pmach, _)| r_pmach.cmp(&l_pmach));*/
    match self.impl_.borrow().find(pmach) {
      None => {}
      Some(_) => panic!("bug")
    }
    self.impl_.borrow_mut().insert(pmach, thimpl_);
  }

  pub fn lookup_impl_(&self, q_pmach: PMach) -> Option<Rc<dyn ThunkImpl_>> {
    /*for &(pmach, ref thimpl_) in self.impl_.borrow().iter() {
      if pmach == q_pmach {
        return Some(thimpl_.clone());
      }
    }
    None*/
    match self.impl_.borrow().find(q_pmach) {
      None => None,
      Some((_, thimpl_)) => Some(thimpl_.clone())
    }
  }

  pub fn apply(&self, ctr: &CtxCtr, env: &mut CtxEnv, arg: &[(CellPtr, Clock)], th: ThunkPtr, out: CellPtr, oclk: Clock) -> ThunkResult {
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
        thimpl_.apply(ctr, env, &*self.spec_, arg, th, out, oclk)
      }
    }
  }

  pub fn accumulate(&self, ctr: &CtxCtr, env: &mut CtxEnv, arg: &[(CellPtr, Clock)], th: ThunkPtr, out: CellPtr, oclk: Clock) -> ThunkResult {
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
        thimpl_.accumulate(ctr, env, &*self.spec_, arg, th, out, oclk)
      }
    }
  }

  pub fn initialize(&self, ctr: &CtxCtr, env: &mut CtxEnv, arg: &[(CellPtr, Clock)], th: ThunkPtr, out: CellPtr, oclk: Clock) -> ThunkResult {
    unimplemented!();
  }
}
