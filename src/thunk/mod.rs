use crate::algo::{SortKey8, SortMap8, RevSortMap8, StdCellExt};
use crate::algo::fp::*;
use crate::algo::hash::*;
use crate::algo::str::*;
use crate::cell::*;
use crate::clock::*;
use crate::ctx::{TL_CTX, CtxCtr, CtxEnv, Cell_, CellClosure, CowCell, ctx_lookup_type, ctx_clean_arg, ctx_push_cell_arg, ctx_pop_thunk};
use crate::pctx::{TL_PCTX, PCtxImpl, Locus, PMach, PAddr, TagUnifier};
#[cfg(feature = "nvgpu")]
use crate::pctx::nvgpu::*;
use crate::pctx::smp::*;
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
  //Abi as FutAbi,
  //AbiOutput as FutAbiOutput,
  //AbiInput as FutAbiInput,
  //AbiArrayRepr as FutAbiArrayRepr,
  AbiScalarType as FutAbiScalarType,
  AbiScalar as FutAbiScalar,
  AbiSpace as FutAbiSpace,
  EntryAbi as FutEntryAbi,
  Array as FutArray,
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
use std::cmp::{max, min};
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

pub fn _cfg_debug_mode(mode: ThunkMode) -> bool {
  TL_CFG_ENV.with(|cfg| {
    !cfg.silent && (
        cfg.debug >= 1
        || match mode {
          ThunkMode::Apply => cfg.debug_apply >= 1,
          ThunkMode::Accumulate => cfg.debug_accumulate >= 1,
          _ => false
        }
    )
  })
}

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

#[derive(Clone, Debug)]
pub struct ThunkSpineType {
  // TODO
  pub arityin:  u16,
  pub arityout: u16,
  pub data: Vec<u32>,
}

impl ThunkSpineType {
  pub fn _parse_datum(u: u32) -> (i16, i8) {
    let var = (u >> 16) as i16;
    let up = (u & 0xff) as i8;
    assert!(up >= -1);
    assert!(up < 0x7f);
    (var, up)
  }

  pub fn pre_cond(&self, idx: u16, ) -> (i16, i8) {
    assert!(idx < self.arityin);
    let u = self.data[idx as usize];
    ThunkSpineType::_parse_datum(u)
  }

  pub fn post_cond(&self, idx: u16, ) -> (i16, i8) {
    assert!(self.arityin + idx < self.arityout);
    let u = self.data[idx as usize];
    ThunkSpineType::_parse_datum(u)
  }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[repr(u8)]
pub enum ThunkMode {
  Apply = 0,
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
  pub fn into_gen(self) -> FutharkThunkGenErr {
    FutharkThunkGenErr::Dim(self)
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
  fn arity(&self) -> Option<(u16, u16)>;
  //fn spine_type(&self) -> ThunkSpineType { unimplemented!(); }
  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr>;
  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr>;
  fn set_out_dim(&self, _arg: &[Dim], _out: Dim) -> Result<(), ThunkDimErr> { Err(ThunkDimErr::Immutable) }
  fn set_out_ty_(&self, _arg: &[CellType], _out: CellType) -> Result<(), ThunkTypeErr> { Err(ThunkTypeErr::Immutable) }
  fn gen_impl_(&self, _spec_dim: Vec<Dim>, _pmach: PMach) -> Option<Rc<dyn ThunkImpl_>> { None }
  fn pop_adj(&self, _arg: &[(CellPtr, Clock)], _out: CellPtr, _out_clk: Clock, _out_mode: ThunkMode, _out_adj: CellPtr, _arg_adj: &mut [CellPtr]) -> Result<(), ThunkAdjErr> { Err(ThunkAdjErr::NotImpl) }
}

pub trait ThunkSpec_ {
  fn as_any(&self) -> &dyn Any;
  fn hash(&self, hasher: &mut dyn Hasher);
  fn thunk_eq(&self, other: &dyn ThunkSpec_) -> Option<bool>;
  fn debug_name(&self) -> Option<&'static str>;
  fn cost_r0(&self) -> Option<ThunkCostR0>;
  fn arity(&self) -> Option<(u16, u16)>;
  //fn spine_type(&self) -> ThunkSpineType;
  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr>;
  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr>;
  fn set_out_dim(&self, arg: &[Dim], out: Dim) -> Result<(), ThunkDimErr>;
  fn set_out_ty_(&self, arg: &[CellType], out: CellType) -> Result<(), ThunkTypeErr>;
  fn gen_impl_(&self, spec_dim: Vec<Dim>, pmach: PMach) -> Option<Rc<dyn ThunkImpl_>>;
  fn pop_adj(&self, arg: &[(CellPtr, Clock)], out: CellPtr, out_clk: Clock, out_mode: ThunkMode, out_adj: CellPtr, arg_adj: &mut [CellPtr]) -> Result<(), ThunkAdjErr>;
}

impl<T: ThunkSpec + Sized + Eq + Hash + Any> ThunkSpec_ for T {
  fn as_any(&self) -> &dyn Any {
    self
  }

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

  fn arity(&self) -> Option<(u16, u16)> {
    ThunkSpec::arity(self)
  }

  /*fn spine_type(&self) -> ThunkSpineType {
    ThunkSpec::spine_type(self)
  }*/

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

pub type FutharkGenErr = FutharkThunkGenErr;

#[derive(Clone, Copy, Debug)]
//#[repr(u8)]
pub enum FutharkThunkGenErr {
  NotImpl,
  Dim(ThunkDimErr),
  _Bot,
}

/*impl From<ThunkDimErr> for FutharkThunkGenErr {
  fn from(e: ThunkDimErr) -> FutharkThunkGenErr {
    FutharkThunkGenErr::Dim(e)
  }
}*/

impl From<FutharkThunkGenCode> for Result<FutharkThunkGenCode, FutharkThunkGenErr> {
  #[inline]
  fn from(code: FutharkThunkGenCode) -> Result<FutharkThunkGenCode, FutharkThunkGenErr> {
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
  fn debug_name(&self) -> Option<&'static str> { None }
  fn cost_r0(&self) -> Option<ThunkCostR0> { None }
  fn arity(&self) -> Option<(u16, u16)> { None }
  //fn abi(&self) -> FutAbi;
  fn abi_param(&self, _param: &mut [FutAbiScalar]) -> usize { 0 }
  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr>;
  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr>;
  fn set_out_dim(&self, _arg: &[Dim], _out: Dim) -> Result<(), ThunkDimErr> { Err(ThunkDimErr::Immutable) }
  fn set_out_ty_(&self, _arg: &[CellType], _out: CellType) -> Result<(), ThunkTypeErr> { Err(ThunkTypeErr::Immutable) }
  fn gen_futhark(&self, /*abi: &mut FutAbi,*/ arg: &[Dim], out: &[Dim]) -> Result<FutharkThunkGenCode, FutharkThunkGenErr>;
  fn pop_adj(&self, _arg: &[(CellPtr, Clock)], _out: CellPtr, _out_clk: Clock, _out_adj: CellPtr, _arg_adj: &mut [CellPtr]) -> Result<FutharkThunkAdj, ThunkAdjErr> { Ok(FutharkThunkAdj::Auto) }
}

impl<T: FutharkThunkSpec> ThunkSpec for T {
  fn debug_name(&self) -> Option<&'static str> {
    FutharkThunkSpec::debug_name(self)
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    FutharkThunkSpec::cost_r0(self)
  }

  fn arity(&self) -> Option<(u16, u16)> {
    /*let abi = FutharkThunkSpec::abi(self);
    (abi.arityin, abi.arityout)*/
    FutharkThunkSpec::arity(self)
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

  fn gen_impl_(&self, spec_dim: Vec<Dim>, pmach: PMach) -> Option<Rc<dyn ThunkImpl_>> {
    let (lar, rar) = match FutharkThunkSpec::arity(self) {
      None => unimplemented!(),
      Some(ar) => ar
    };
    /*let mut abi = FutharkThunkSpec::abi(self);
    assert_eq!(abi.space, FutAbiSpace::Unspec);*/
    let (arg_dim, out_dim) = (&spec_dim).split_at(lar as usize);
    assert_eq!(out_dim.len(), rar as usize);
    assert_eq!(arg_dim.len(), lar as usize);
    //let np0 = abi.num_param();
    let np0 = 0;
    let mut param: Vec<FutAbiScalar> = Vec::with_capacity(np0);
    //param.resize(np0, FutAbiScalar::Unspec);
    assert_eq!(FutharkThunkSpec::abi_param(self, &mut param), np0);
    //let mut tmp_abi = FutAbi::default();
    let code = match FutharkThunkSpec::gen_futhark(self, /*&mut tmp_abi,*/ arg_dim, out_dim) {
      Err(e) => {
        println!("DEBUG: ThunkSpec::gen_impl_: name={:?} arg={:?} out={:?}",
            FutharkThunkSpec::debug_name(self), arg_dim, out_dim);
        println!("ERROR: failed to generate futhark thunk code: {:?}", e);
        panic!();
      }
      Ok(code) => code
    };
    /*if code.cfg.emit_out0_shape_param {
      for d in 0 .. out_dim[0].ndim {
        abi.push_param(np0 as u16 + d as u16, FutAbiScalarType::I64);
      }
    }*/
    let name = FutharkThunkSpec::debug_name(self);
    Some(match pmach {
      PMach::Smp => {
        //abi.space = FutAbiSpace::Default;
        Rc::new(FutharkThunkImpl::<MulticoreBackend>{
          //abi,
          lar,
          rar,
          param,
          spec_dim,
          code,
          name,
          //source: RefCell::new(String::new()),
          objects: RefCell::new(SortMap8::new()),
        })
      }
      #[cfg(not(feature = "nvgpu"))]
      PMach::NvGpu => {
        panic!("ERROR: not compiled with gpu support");
      }
      #[cfg(feature = "nvgpu")]
      PMach::NvGpu => {
        //abi.space = FutAbiSpace::Device;
        Rc::new(FutharkThunkImpl::<CudaBackend>{
          //abi,
          lar,
          rar,
          param,
          spec_dim,
          code,
          name,
          //source: RefCell::new(String::new()),
          objects: RefCell::new(SortMap8::new()),
        })
      }
      _ => unimplemented!()
    })
  }

  fn pop_adj(&self, arg: &[(CellPtr, Clock)], out: CellPtr, out_clk: Clock, out_mode: ThunkMode, out_adj: CellPtr, arg_adj: &mut [CellPtr]) -> Result<(), ThunkAdjErr> {
    assert!(!out_adj.is_nil());
    match FutharkThunkSpec::pop_adj(self, arg, out, out_clk, out_adj, arg_adj) {
      Ok(FutharkThunkAdj::Auto) => {
        // FIXME
        if cfg_debug() { println!("DEBUG: <FutharkThunkSpec as ThunkSpec>::pop_adj: name={:?}", self.debug_name()); }
        let primal_ar = FutharkThunkSpec::arity(self);
        let (primal_lar, primal_rar) = match primal_ar {
          None => unimplemented!(),
          Some(ar) => ar
        };
        assert_eq!(1, primal_rar);
        assert_eq!(arg.len(), primal_lar as usize);
        assert_eq!(arg_adj.len(), primal_lar as usize);
        /*let abi = FutharkThunkSpec::abi(self);
        assert_eq!(1, abi.arityout);
        assert_eq!(arg.len(), abi.arityin as usize);
        assert_eq!(arg_adj.len(), abi.arityin as usize);*/
        let mut dim = Vec::with_capacity((primal_lar + primal_rar) as usize);
        let mut ty_ = Vec::with_capacity((primal_lar + primal_rar) as usize);
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
        let (arg_dim, out_dim) = (&dim).split_at(primal_lar as usize);
        assert_eq!(out_dim.len(), primal_rar as usize);
        assert_eq!(arg_dim.len(), primal_lar as usize);
        /*
        let np0 = abi.num_param();
        let mut param: Vec<FutAbiScalar> = Vec::with_capacity(np0);
        param.resize(np0, FutAbiScalar::Unspec);
        assert_eq!(FutharkThunkSpec::abi_param(self, &mut param), np0);
        // FIXME: need to capture the out0 shape param in a closure.
        */
        //let mut tmp_abi = abi.clone();
        //let mut tmp_abi = FutAbi::default();
        let mut code = match self.gen_futhark(/*&mut tmp_abi,*/ arg_dim, out_dim) {
          Err(e) => panic!("ERROR: failed to generate futhark thunk code: {:?}", e),
          Ok(code) => code
        };
        let restore_cfg = code.cfg;
        code.cfg.emit_primal_def = true;
        // FIXME
        /*if code.abi.param_ct != 0 {
          unimplemented!();
        }*/
        //if cfg_debug() { println!("DEBUG: <FutharkThunkSpec as ThunkSpec>::pop_adj:   primal lar={} rar={}", primal_lar, primal_rar, ); }
        if cfg_debug() { println!("DEBUG: <FutharkThunkSpec as ThunkSpec>::pop_adj:   gen source..."); }
        let (primal_genabi, primal_source) = code.gen_source(/*&abi,*/ primal_lar, primal_rar, &dim, out_mode, FutharkThunkGenConfig::default(), code.abi.clone())
          .map_err(|_| ThunkAdjErr::_Bot)?;
        if cfg_debug() { println!("DEBUG: <FutharkThunkSpec as ThunkSpec>::pop_adj:   primal genabi={:?}", &primal_genabi); }
        assert_eq!(primal_genabi.arityout, primal_rar);
        assert_eq!(primal_genabi.arityin, primal_lar);
        let primal_cost = FutharkThunkSpec::cost_r0(self);
        for k in 0 .. primal_lar {
          if arg_adj[k as usize].is_nil() {
            continue;
          }
          /*let (adj_lar, adj_rar) = match primal_ar {
            None => unimplemented!(),
            Some((lar, rar)) => (lar + 1, rar)
          };*/
          /*let mut adj_abi = FutAbi::default();
          adj_abi.arityin = abi.arityin + 1;
          adj_abi.arityout = 1;*/
          let mut adj_dim = Vec::with_capacity((primal_lar + primal_rar + 1) as usize);
          let mut adj_ty_ = Vec::with_capacity((primal_lar + primal_rar + 1) as usize);
          adj_dim.extend_from_slice(&dim);
          adj_ty_.extend_from_slice(&ty_);
          adj_dim.push(dim[k as usize]);
          adj_ty_.push(ty_[k as usize].clone());
          assert_eq!(adj_dim.len(), (primal_lar + primal_rar + 1) as usize);
          assert_eq!(adj_ty_.len(), (primal_lar + primal_rar + 1) as usize);
          let mut adj_code = FutharkThunkGenCode::default();
          adj_code.abi.arityout = 1;
          adj_code.abi.arityin = primal_genabi.arityin + 1;
          adj_code.abi.set_out(0, primal_genabi.get_arg(k));
          for j in 0 .. primal_genabi.arityin {
            adj_code.abi.set_arg(j, primal_genabi.get_arg(j));
          }
          adj_code.abi.set_arg(primal_genabi.arityin, primal_genabi.get_out(0));
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
          for line in primal_source.iter() {
            adj_code.pre_append(line);
          }
          // FIXME
          let mut line = format!(r"let {{%{}}} = vjp (\t -> primal", primal_lar + 1);
          for j in 0 .. k {
            write!(&mut line, r" {{%{}}}", j).unwrap();
          }
          write!(&mut line, r" t").unwrap();
          for j in k + 1 .. primal_lar {
            write!(&mut line, r" {{%{}}}", j).unwrap();
          }
          if restore_cfg.emit_out0_shape_param {
            let dim = adj_dim[primal_lar as usize];
            let nd = match adj_code.abi.get_arg(primal_lar) {
              FutharkArrayRepr::Nd => dim.ndim,
              FutharkArrayRepr::Flat => min(1, dim.ndim),
              _ => unimplemented!()
            };
            for d in 0 .. nd {
              write!(&mut line, " {{%{}.s[{}]}}", primal_lar, d).unwrap();
            }
          }
          write!(&mut line, r") {{%{}}} {{%{}}} in", k, primal_lar).unwrap();
          adj_code.append(line);
          let adj_spec = FutharkCodeThunkSpec{
            primal_mode: out_mode,
            cost: primal_cost,
            lar: primal_lar + 1,
            rar: primal_rar,
            //abi: adj_abi,
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
        if cfg_debug() { println!("DEBUG: <FutharkThunkSpec as ThunkSpec>::pop_adj:   ... done!"); }
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

  pub fn gen_futhark<S: Borrow<str>>(&self, /*abi: &mut FutAbi,*/ arg0: Dim, arg1: Dim, lam: S) -> Result<FutharkThunkGenCode, FutharkThunkGenErr> {
    assert_eq!(arg0.ndim(), self.nd);
    assert_eq!(arg1.ndim(), self.nd);
    /*abi.set_out_arr(0, FutAbiOutput::Pure, FutAbiArrayRepr::Nd, FutAbiScalarType::Unspec);
    abi.set_arg_arr(0, FutAbiInput::Shared, FutAbiArrayRepr::Nd, FutAbiScalarType::Unspec);
    abi.set_arg_arr(1, FutAbiInput::Shared, FutAbiArrayRepr::Nd, FutAbiScalarType::Unspec);*/
    let lam = lam.borrow();
    let mut code = FutharkThunkGenCode::default();
    code.abi.arityout = 1;
    code.abi.set_out(0, FutharkArrayRepr::Nd);
    code.abi.arityin = 2;
    code.abi.set_arg(0, FutharkArrayRepr::Nd);
    code.abi.set_arg(1, FutharkArrayRepr::Nd);
    match self.nd {
      0 => {
        code.append(format!(r"let {{%2}} = ({}) {{%0}} {{%1}} in", lam));
      }
      1 => {
        code.cfg.emit_out0_shape = true;
        code.cfg.emit_out0_shape_param = true;
        match self.lmsk & 1 {
          0 => {
            code.pre_append(format!(r"def f0_dim_0 t_0 {{%2.s:~}} = t_0 :> [{{%2.s[0]}}]{}",
                arg0.dtype.format_futhark(),
            ));
          }
          1 => {
            code.pre_append(format!(r"def f0_dim_0 t_0 {{%2.s:~}} = (\t -> replicate {{%2.s[0]}} t[0]) t_0"));
          }
          _ => unreachable!()
        }
        match self.rmsk & 1 {
          0 => {
            code.pre_append(format!(r"def f1_dim_0 t_0 {{%2.s:~}} = t_0 :> [{{%2.s[0]}}]{}",
                arg1.dtype.format_futhark(),
            ));
          }
          1 => {
            code.pre_append(format!(r"def f1_dim_0 t_0 {{%2.s:~}} = (\t -> replicate {{%2.s[0]}} t[0]) t_0"));
          }
          _ => unreachable!()
        }
        code.append(format!(r"let t0 = f0_dim_0 {{%0}} {{%2.s~}} in"));
        code.append(format!(r"let t1 = f1_dim_0 {{%1}} {{%2.s~}} in"));
        code.append(format!(r"let {{%2}} = map2 ({}) t0 t1 in", lam));
      }
      2 => {
        code.cfg.emit_out0_shape = true;
        code.cfg.emit_out0_shape_param = true;
        match (self.lmsk >> 1) & 1 {
          0 => {
            code.pre_append(format!(r"def f0_dim_1 t_0 {{%2.s:~}} = t_0 :> [{{%2.s[1]}}]{}",
                arg0.dtype.format_futhark(),
            ));
          }
          1 => {
            code.pre_append(format!(r"def f0_dim_1 t_0 {{%2.s:~}} = (\t -> replicate {{%2.s[1]}} t[0]) t_0"));
          }
          _ => unreachable!()
        }
        match self.lmsk & 1 {
          0 => {
            code.pre_append(format!(r"def f0_dim_0 t_0 {{%2.s:~}} = (\t -> map (\u -> f0_dim_1 u {{%2.s~}}) t) t_0 :> [{{%2.s[0]}}][{{%2.s[1]}}]{}",
                arg0.dtype.format_futhark(),
            ));
          }
          1 => {
            code.pre_append(format!(r"def f0_dim_0 t_0 {{%2.s:~}} = (\t -> replicate {{%2.s[0]}} (f0_dim_1 t[0] {{%2.s~}})) t_0"));
          }
          _ => unreachable!()
        }
        match (self.rmsk >> 1) & 1 {
          0 => {
            code.pre_append(format!(r"def f1_dim_1 t_0 {{%2.s:~}} = t_0 :> [{{%2.s[1]}}]{}",
                arg1.dtype.format_futhark(),
            ));
          }
          1 => {
            code.pre_append(format!(r"def f1_dim_1 t_0 {{%2.s:~}} = (\t -> replicate {{%2.s[1]}} t[0]) t_0"));
          }
          _ => unreachable!()
        }
        match self.rmsk & 1 {
          0 => {
            code.pre_append(format!(r"def f1_dim_0 t_0 {{%2.s:~}} = (\t -> map (\u -> f1_dim_1 u {{%2.s~}}) t) t_0 :> [{{%2.s[0]}}][{{%2.s[1]}}]{}",
                arg1.dtype.format_futhark(),
            ));
          }
          1 => {
            code.pre_append(format!(r"def f1_dim_0 t_0 {{%2.s:~}} = (\t -> replicate {{%2.s[0]}} (f1_dim_1 t[0] {{%2.s~}})) t_0"));
          }
          _ => unreachable!()
        }
        code.append(format!(r"let t0 = flatten (f0_dim_0 {{%0}} {{%2.s~}}) in"));
        code.append(format!(r"let t1 = flatten (f1_dim_0 {{%1}} {{%2.s~}}) in"));
        code.append(format!(r"let t2 = map2 ({}) t0 t1 in", lam));
        code.append(format!(r"let {{%2}} = unflatten t2 in"));
      }
      3 => {
        code.cfg.emit_out0_shape = true;
        code.cfg.emit_out0_shape_param = true;
        match (self.lmsk >> 2) & 1 {
          0 => {
            code.pre_append(format!(r"def f0_dim_2 t_0 {{%2.s:~}} = t_0 :> [{{%2.s[2]}}]{}",
                arg0.dtype.format_futhark(),
            ));
          }
          1 => {
            code.pre_append(format!(r"def f0_dim_2 t_0 {{%2.s:~}} = (\t -> replicate {{%2.s[2]}} t[0]) t_0"));
          }
          _ => unreachable!()
        }
        match (self.lmsk >> 1) & 1 {
          0 => {
            code.pre_append(format!(r"def f0_dim_1 t_0 {{%2.s:~}} = (\t -> map (\u -> f0_dim_2 u {{%2.s~}}) t) t_0 :> [{{%2.s[1]}}][{{%2.s[2]}}]{}",
                arg0.dtype.format_futhark(),
            ));
          }
          1 => {
            code.pre_append(format!(r"def f0_dim_1 t_0 {{%2.s:~}} = (\t -> replicate {{%2.s[1]}} (f0_dim_2 t[0] {{%2.s~}})) t_0"));
          }
          _ => unreachable!()
        }
        match self.lmsk & 1 {
          0 => {
            code.pre_append(format!(r"def f0_dim_0 t_0 {{%2.s:~}} = (\t -> map (\u -> f0_dim_1 u {{%2.s~}}) t) t_0 :> [{{%2.s[0]}}][{{%2.s[1]}}][{{%2.s[2]}}]{}",
                arg0.dtype.format_futhark(),
            ));
          }
          1 => {
            code.pre_append(format!(r"def f0_dim_0 t_0 {{%2.s:~}} = (\t -> replicate {{%2.s[0]}} (f0_dim_1 t[0] {{%2.s~}})) t_0"));
          }
          _ => unreachable!()
        }
        match (self.rmsk >> 2) & 1 {
          0 => {
            code.pre_append(format!(r"def f1_dim_2 t_0 {{%2.s:~}} = t_0 :> [{{%2.s[2]}}]{}",
                arg1.dtype.format_futhark(),
            ));
          }
          1 => {
            code.pre_append(format!(r"def f1_dim_2 t_0 {{%2.s:~}} = (\t -> replicate {{%2.s[2]}} t[0]) t_0"));
          }
          _ => unreachable!()
        }
        match (self.rmsk >> 1) & 1 {
          0 => {
            code.pre_append(format!(r"def f1_dim_1 t_0 {{%2.s:~}} = (\t -> map (\u -> f1_dim_2 u {{%2.s~}}) t) t_0 :> [{{%2.s[1]}}][{{%2.s[2]}}]{}",
                arg1.dtype.format_futhark(),
            ));
          }
          1 => {
            code.pre_append(format!(r"def f1_dim_1 t_0 {{%2.s:~}} = (\t -> replicate {{%2.s[1]}} (f1_dim_2 t[0] {{%2.s~}})) t_0"));
          }
          _ => unreachable!()
        }
        match self.rmsk & 1 {
          0 => {
            code.pre_append(format!(r"def f1_dim_0 t_0 {{%2.s:~}} = (\t -> map (\u -> f1_dim_1 u {{%2.s~}}) t) t_0 :> [{{%2.s[0]}}][{{%2.s[1]}}][{{%2.s[2]}}]{}",
                arg1.dtype.format_futhark(),
            ));
          }
          1 => {
            code.pre_append(format!(r"def f1_dim_0 t_0 {{%2.s:~}} = (\t -> replicate {{%2.s[0]}} (f1_dim_1 t[0] {{%2.s~}})) t_0"));
          }
          _ => unreachable!()
        }
        code.append(format!(r"let t0 = flatten_3d (f0_dim_0 {{%0}} {{%2.s~}}) in"));
        code.append(format!(r"let t1 = flatten_3d (f1_dim_0 {{%1}} {{%2.s~}}) in"));
        code.append(format!(r"let t2 = map2 ({}) t0 t1 in", lam));
        code.append(format!(r"let {{%2}} = unflatten_3d t2 in"));
      }
      4 => {
        code.cfg.emit_out0_shape = true;
        code.cfg.emit_out0_shape_param = true;
        match (self.lmsk >> 3) & 1 {
          0 => {
            code.pre_append(format!(r"def f0_dim_3 t_0 {{%2.s:~}} = t_0 :> [{{%2.s[3]}}]{}",
                arg0.dtype.format_futhark(),
            ));
          }
          1 => {
            code.pre_append(format!(r"def f0_dim_3 t_0 {{%2.s:~}} = (\t -> replicate {{%2.s[3]}} t[0]) t_0"));
          }
          _ => unreachable!()
        }
        match (self.lmsk >> 2) & 1 {
          0 => {
            code.pre_append(format!(r"def f0_dim_2 t_0 {{%2.s:~}} = (\t -> map (\u -> f0_dim_3 u {{%2.s~}}) t) t_0 :> [{{%2.s[2]}}][{{%2.s[3]}}]{}",
                arg0.dtype.format_futhark(),
            ));
          }
          1 => {
            code.pre_append(format!(r"def f0_dim_2 t_0 {{%2.s:~}} = (\t -> replicate {{%2.s[2]}} (f0_dim_3 t[0] {{%2.s~}})) t_0"));
          }
          _ => unreachable!()
        }
        match (self.lmsk >> 1) & 1 {
          0 => {
            code.pre_append(format!(r"def f0_dim_1 t_0 {{%2.s:~}} = (\t -> map (\u -> f0_dim_2 u {{%2.s~}}) t) t_0 :> [{{%2.s[1]}}][{{%2.s[2]}}][{{%2.s[3]}}]{}",
                arg0.dtype.format_futhark(),
            ));
          }
          1 => {
            code.pre_append(format!(r"def f0_dim_1 t_0 {{%2.s:~}} = (\t -> replicate {{%2.s[1]}} (f0_dim_2 t[0] {{%2.s~}})) t_0"));
          }
          _ => unreachable!()
        }
        match self.lmsk & 1 {
          0 => {
            code.pre_append(format!(r"def f0_dim_0 t_0 {{%2.s:~}} = (\t -> map (\u -> f0_dim_1 u {{%2.s~}}) t) t_0 :> [{{%2.s[0]}}][{{%2.s[1]}}][{{%2.s[2]}}][{{%2.s[3]}}]{}",
                arg0.dtype.format_futhark(),
            ));
          }
          1 => {
            code.pre_append(format!(r"def f0_dim_0 t_0 {{%2.s:~}} = (\t -> replicate {{%2.s[0]}} (f0_dim_1 t[0] {{%2.s~}})) t_0"));
          }
          _ => unreachable!()
        }
        match (self.rmsk >> 3) & 1 {
          0 => {
            code.pre_append(format!(r"def f1_dim_3 t_0 {{%2.s:~}} = t_0 :> [{{%2.s[3]}}]{}",
                arg1.dtype.format_futhark(),
            ));
          }
          1 => {
            code.pre_append(format!(r"def f1_dim_3 t_0 {{%2.s:~}} = (\t -> replicate {{%2.s[3]}} t[0]) t_0"));
          }
          _ => unreachable!()
        }
        match (self.rmsk >> 2) & 1 {
          0 => {
            code.pre_append(format!(r"def f1_dim_2 t_0 {{%2.s:~}} = (\t -> map (\u -> f1_dim_3 u {{%2.s~}}) t) t_0 :> [{{%2.s[2]}}][{{%2.s[3]}}]{}",
                arg1.dtype.format_futhark(),
            ));
          }
          1 => {
            code.pre_append(format!(r"def f1_dim_2 t_0 {{%2.s:~}} = (\t -> replicate {{%2.s[2]}} (f1_dim_3 t[0] {{%2.s~}})) t_0"));
          }
          _ => unreachable!()
        }
        match (self.rmsk >> 1) & 1 {
          0 => {
            code.pre_append(format!(r"def f1_dim_1 t_0 {{%2.s:~}} = (\t -> map (\u -> f1_dim_2 u {{%2.s~}}) t) t_0 :> [{{%2.s[1]}}][{{%2.s[2]}}][{{%2.s[3]}}]{}",
                arg1.dtype.format_futhark(),
            ));
          }
          1 => {
            code.pre_append(format!(r"def f1_dim_1 t_0 {{%2.s:~}} = (\t -> replicate {{%2.s[1]}} (f1_dim_2 t[0] {{%2.s~}})) t_0"));
          }
          _ => unreachable!()
        }
        match self.rmsk & 1 {
          0 => {
            code.pre_append(format!(r"def f1_dim_0 t_0 {{%2.s:~}} = (\t -> map (\u -> f1_dim_1 u {{%2.s~}}) t) t_0 :> [{{%2.s[0]}}][{{%2.s[1]}}][{{%2.s[2]}}][{{%2.s[3]}}]{}",
                arg1.dtype.format_futhark(),
            ));
          }
          1 => {
            code.pre_append(format!(r"def f1_dim_0 t_0 {{%2.s:~}} = (\t -> replicate {{%2.s[0]}} (f1_dim_1 t[0] {{%2.s~}})) t_0"));
          }
          _ => unreachable!()
        }
        code.append(format!(r"let t0 = flatten_4d (f0_dim_0 {{%0}} {{%2.s~}}) in"));
        code.append(format!(r"let t1 = flatten_4d (f1_dim_0 {{%1}} {{%2.s~}}) in"));
        code.append(format!(r"let t2 = map2 ({}) t0 t1 in", lam));
        code.append(format!(r"let {{%2}} = unflatten_4d t2 in"));
      }
      _ => {
        println!("WARNING: FutharkNdBroadcastMap2MonomorphicSpec::gen_futhark: not implemented: {:?} {:?}", arg0, arg1);
        return Err(FutharkThunkGenErr::NotImpl);
      }
    }
    if code.cfg.emit_out0_shape_param {
      // FIXME: Abi should do this.
      // FIXME FIXME
      /*for d in 0 .. self.nd {
        //abi.push_param(FutAbiParam::ImplicitOutShape, d as _, FutAbiScalarType::I64);
        abi.push_implicit_out_shape_param(0, d as _);
      }*/
    }
    code.into()
  }
}

//#[derive(Clone, Default)]
#[derive(Clone, PartialEq, Eq, Hash, Default)]
pub struct FutharkThunkGenCode {
  pub cfg:  FutharkThunkGenConfig,
  pub abi:  FutharkThunkGenAbi,
  pub head: Vec<String>,
  pub body: Vec<String>,
}

impl FutharkThunkGenCode {
  pub fn flat_replicate<S: Borrow<str>>(/*abi: &mut FutAbi,*/ out0: Dim, val: S) -> Result<FutharkThunkGenCode, FutharkThunkGenErr> {
    //abi.set_out_arr(0, FutAbiOutput::Pure, FutAbiArrayRepr::Nd, FutAbiScalarType::Unspec);
    let mut code = FutharkThunkGenCode::default();
    code.abi.arityout = 1;
    code.abi.set_out(0, FutharkArrayRepr::Flat);
    code.abi.arityin = 0;
    code.append_flat_replicate(out0, val)?;
    code.into()
  }

  pub fn append_flat_replicate<S: Borrow<str>>(&mut self, out0: Dim, val: S) -> Result<(), FutharkThunkGenErr> {
    let val = val.borrow();
    if out0.ndim() == 0 {
      self.append(format!(r"let {{%0}} = ({}) in", val));
    } else {
      self.cfg.emit_out0_shape = true;
      self.cfg.emit_out0_shape_param = true;
      self.append(format!(r"let {{%0}} = replicate {{%0.s*}} ({}) in", val));
    }
    Ok(())
  }

  pub fn flat_map<S: Borrow<str>>(/*abi: &mut FutAbi,*/ arg0: Dim, lam: S) -> Result<FutharkThunkGenCode, FutharkThunkGenErr> {
    /*abi.set_out_arr(0, FutAbiOutput::Pure, FutAbiArrayRepr::Nd, FutAbiScalarType::Unspec);
    abi.set_arg_arr(0, FutAbiInput::Shared, FutAbiArrayRepr::Nd, FutAbiScalarType::Unspec);*/
    FutharkThunkGenCode::flat_map_(r"{%0}", arg0, r"{%1}", lam)
  }

  pub fn flat_map_<S: Borrow<str>>(x0: &str, arg0: Dim, y: &str, lam: S) -> Result<FutharkThunkGenCode, FutharkThunkGenErr> {
    let mut code = FutharkThunkGenCode::default();
    code.abi.arityout = 1;
    code.abi.set_out(0, FutharkArrayRepr::Flat);
    code.abi.arityin = 1;
    code.abi.set_arg(0, FutharkArrayRepr::Flat);
    code.append_flat_map(x0, arg0, y, lam)?;
    code.into()
  }

  pub fn append_flat_map<S: Borrow<str>>(&mut self, x0: &str, arg0: Dim, y: &str, lam: S) -> Result<(), FutharkThunkGenErr> {
    let lam = lam.borrow();
    if arg0.ndim() == 0 {
      self.append(format!(r"let {} = ({}) {} in", y, lam, x0));
    } else {
      self.append(format!(r"let {} = map ({}) {} in", y, lam, x0));
    }
    Ok(())
  }

  pub fn flat_map2<S: Borrow<str>>(/*abi: &mut FutAbi,*/ arg0: Dim, arg1: Dim, lam: S) -> Result<FutharkThunkGenCode, FutharkThunkGenErr> {
    /*abi.set_out_arr(0, FutAbiOutput::Pure, FutAbiArrayRepr::Nd, FutAbiScalarType::Unspec);
    abi.set_arg_arr(0, FutAbiInput::Shared, FutAbiArrayRepr::Nd, FutAbiScalarType::Unspec);
    abi.set_arg_arr(1, FutAbiInput::Shared, FutAbiArrayRepr::Nd, FutAbiScalarType::Unspec);*/
    FutharkThunkGenCode::flat_map2_(r"{%0}", arg0, r"{%1}", arg1, r"{%2}", lam)
  }

  pub fn flat_map2_<S: Borrow<str>>(x0: &str, arg0: Dim, x1: &str, arg1: Dim, y: &str, lam: S) -> Result<FutharkThunkGenCode, FutharkThunkGenErr> {
    let mut code = FutharkThunkGenCode::default();
    code.abi.arityout = 1;
    code.abi.set_out(0, FutharkArrayRepr::Flat);
    code.abi.arityin = 2;
    code.abi.set_arg(0, FutharkArrayRepr::Flat);
    code.abi.set_arg(1, FutharkArrayRepr::Flat);
    code.append_flat_map2(x0, arg0, x1, arg1, y, lam)?;
    code.into()
  }

  pub fn append_flat_map2<S: Borrow<str>>(&mut self, x0: &str, arg0: Dim, x1: &str, arg1: Dim, y: &str, lam: S) -> Result<(), FutharkThunkGenErr> {
    let lam = lam.borrow();
    if !(arg0.ndim() == arg1.ndim()) {
      println!("WARNING: FutharkThunkGenCode::append_flat_map2: not implemented: {:?} {:?}", arg0, arg1);
      return Err(FutharkThunkGenErr::NotImpl);
    }
    if arg0.ndim() == 0 {
      self.append(format!(r"let {} = ({}) {} {} in", y, lam, x0, x1));
    } else {
      self.append(format!(r"let {} = map2 ({}) {} {} in", y, lam, x0, x1));
    }
    Ok(())
  }

  pub fn flat_map3<S: Borrow<str>>(/*abi: &mut FutAbi,*/ arg0: Dim, arg1: Dim, arg2: Dim, lam: S) -> Result<FutharkThunkGenCode, FutharkThunkGenErr> {
    /*abi.set_out_arr(0, FutAbiOutput::Pure, FutAbiArrayRepr::Nd, FutAbiScalarType::Unspec);
    abi.set_arg_arr(0, FutAbiInput::Shared, FutAbiArrayRepr::Nd, FutAbiScalarType::Unspec);
    abi.set_arg_arr(1, FutAbiInput::Shared, FutAbiArrayRepr::Nd, FutAbiScalarType::Unspec);
    abi.set_arg_arr(2, FutAbiInput::Shared, FutAbiArrayRepr::Nd, FutAbiScalarType::Unspec);*/
    FutharkThunkGenCode::flat_map3_(r"{%0}", arg0, r"{%1}", arg1, r"{%2}", arg2, r"{%3}", lam)
  }

  pub fn flat_map3_<S: Borrow<str>>(x0: &str, arg0: Dim, x1: &str, arg1: Dim, x2: &str, arg2: Dim, y: &str, lam: S) -> Result<FutharkThunkGenCode, FutharkThunkGenErr> {
    let mut code = FutharkThunkGenCode::default();
    code.abi.arityout = 1;
    code.abi.set_out(0, FutharkArrayRepr::Flat);
    code.abi.arityin = 3;
    code.abi.set_arg(0, FutharkArrayRepr::Flat);
    code.abi.set_arg(1, FutharkArrayRepr::Flat);
    code.abi.set_arg(2, FutharkArrayRepr::Flat);
    code.append_flat_map3(x0, arg0, x1, arg1, x2, arg2, y, lam)?;
    code.into()
  }

  pub fn append_flat_map3<S: Borrow<str>>(&mut self, x0: &str, arg0: Dim, x1: &str, arg1: Dim, x2: &str, arg2: Dim, y: &str, lam: S) -> Result<(), FutharkThunkGenErr> {
    let lam = lam.borrow();
    if !(arg0.ndim() == arg1.ndim() && arg0.ndim() == arg2.ndim()) {
      println!("WARNING: FutharkThunkGenCode::append_flat_map3: not implemented: {:?} {:?} {:?}", arg0, arg1, arg2);
      return Err(FutharkThunkGenErr::NotImpl);
    }
    if arg0.ndim() == 0 {
      self.append(format!(r"let {} = ({}) {} {} {} in", y, lam, x0, x1, x2));
    } else {
      self.append(format!(r"let {} = map3 ({}) {} {} {} in", y, lam, x0, x1, x2));
    }
    Ok(())
  }

  pub fn nd_replicate<S: Borrow<str>>(/*abi: &mut FutAbi,*/ out0: Dim, val: S) -> Result<FutharkThunkGenCode, FutharkThunkGenErr> {
    //abi.set_out_arr(0, FutAbiOutput::Pure, FutAbiArrayRepr::Nd, FutAbiScalarType::Unspec);
    let mut code = FutharkThunkGenCode::default();
    code.abi.arityout = 1;
    code.abi.set_out(0, FutharkArrayRepr::Nd);
    code.abi.arityin = 0;
    code.append_nd_replicate(out0, val)?;
    code.into()
  }

  pub fn append_nd_replicate<S: Borrow<str>>(&mut self, out0: Dim, val: S) -> Result<(), FutharkThunkGenErr> {
    let val = val.borrow();
    match out0.ndim() {
      0 => {
        self.append(format!(r"let {{%0}} = ({}) in", val));
      }
      1 => {
        self.cfg.emit_out0_shape = true;
        self.cfg.emit_out0_shape_param = true;
        //self.append(format!(r"let {{%0}} = replicate {{%0.s[0]}} ({}) in", val));
        self.append(format!(r"let {{%0}} = replicate {{%0.s*}} ({}) in", val));
      }
      2 => {
        self.cfg.emit_out0_shape = true;
        self.cfg.emit_out0_shape_param = true;
        //self.append(format!(r"let t0 = replicate ({{%0.s[0]}} * {{%0.s[1]}}) ({}) in", val));
        self.append(format!(r"let t0 = replicate ({{%0.s*}}) ({}) in", val));
        self.append(format!(r"let {{%0}} = unflatten t0 in"));
      }
      3 => {
        self.cfg.emit_out0_shape = true;
        self.cfg.emit_out0_shape_param = true;
        //self.append(format!(r"let t0 = replicate ({{%0.s[0]}} * {{%0.s[1]}} * {{%0.s[2]}}) ({}) in", val));
        self.append(format!(r"let t0 = replicate ({{%0.s*}}) ({}) in", val));
        self.append(format!(r"let {{%0}} = unflatten_3d t0 in"));
      }
      4 => {
        self.cfg.emit_out0_shape = true;
        self.cfg.emit_out0_shape_param = true;
        //self.append(format!(r"let t0 = replicate ({{%0.s[0]}} * {{%0.s[1]}} * {{%0.s[2]}} * {{%0.s[3]}}) ({}) in", val));
        self.append(format!(r"let t0 = replicate ({{%0.s*}}) ({}) in", val));
        self.append(format!(r"let {{%0}} = unflatten_4d t0 in"));
      }
      _ => {
        println!("WARNING: FutharkThunkGenCode::append_nd_replicate: not implemented: {:?}", out0);
        return Err(FutharkThunkGenErr::NotImpl);
      }
    }
    Ok(())
  }

  pub fn nd_map<S: Borrow<str>>(/*abi: &mut FutAbi,*/ arg0: Dim, lam: S) -> Result<FutharkThunkGenCode, FutharkThunkGenErr> {
    //abi.set_out_arr(0, FutAbiOutput::Pure, FutAbiArrayRepr::Nd, FutAbiScalarType::Unspec);
    //abi.set_arg_arr(0, FutAbiInput::Shared, FutAbiArrayRepr::Nd, FutAbiScalarType::Unspec);
    let mut code = FutharkThunkGenCode::default();
    code.abi.arityout = 1;
    code.abi.set_out(0, FutharkArrayRepr::Nd);
    code.abi.arityin = 1;
    code.abi.set_arg(0, FutharkArrayRepr::Nd);
    code.append_nd_map(arg0, lam)?;
    code.into()
  }

  pub fn append_nd_map<S: Borrow<str>>(&mut self, arg0: Dim, lam: S) -> Result<(), FutharkThunkGenErr> {
    let lam = lam.borrow();
    match arg0.ndim() {
      0 => {
        self.append(format!(r"let {{%1}} = ({}) {{%0}} in", lam));
      }
      1 => {
        self.append(format!(r"let {{%1}} = map ({}) {{%0}} in", lam));
      }
      2 => {
        self.append(format!(r"let t0 = flatten {{%0}} in"));
        self.append(format!(r"let t1 = map ({}) t0 in", lam));
        self.append(format!(r"let {{%1}} = unflatten t1 in"));
      }
      3 => {
        self.append(format!(r"let t0 = flatten_3d {{%0}} in"));
        self.append(format!(r"let t1 = map ({}) t0 in", lam));
        self.append(format!(r"let {{%1}} = unflatten_3d t1 in"));
      }
      4 => {
        self.append(format!(r"let t0 = flatten_4d {{%0}} in"));
        self.append(format!(r"let t1 = map ({}) t0 in", lam));
        self.append(format!(r"let {{%1}} = unflatten_4d t1 in"));
      }
      _ => {
        println!("WARNING: FutharkThunkGenCode::append_nd_map: not implemented: {:?}", arg0);
        return Err(FutharkThunkGenErr::NotImpl);
      }
    }
    Ok(())
  }

  pub fn nd_map2<S: Borrow<str>>(/*abi: &mut FutAbi,*/ arg0: Dim, arg1: Dim, lam: S) -> Result<FutharkThunkGenCode, FutharkThunkGenErr> {
    //abi.set_out_arr(0, FutAbiOutput::Pure, FutAbiArrayRepr::Nd, FutAbiScalarType::Unspec);
    //abi.set_arg_arr(0, FutAbiInput::Shared, FutAbiArrayRepr::Nd, FutAbiScalarType::Unspec);
    //abi.set_arg_arr(1, FutAbiInput::Shared, FutAbiArrayRepr::Nd, FutAbiScalarType::Unspec);
    FutharkThunkGenCode::nd_map2_(r"{%0}", arg0, r"{%1}", arg1, r"{%2}", lam)
  }

  pub fn nd_map2_<S: Borrow<str>>(x0: &str, arg0: Dim, x1: &str, arg1: Dim, y: &str, lam: S) -> Result<FutharkThunkGenCode, FutharkThunkGenErr> {
    let mut code = FutharkThunkGenCode::default();
    code.abi.arityout = 1;
    code.abi.set_out(0, FutharkArrayRepr::Nd);
    code.abi.arityin = 2;
    code.abi.set_arg(0, FutharkArrayRepr::Nd);
    code.abi.set_arg(1, FutharkArrayRepr::Nd);
    code.append_nd_map2(x0, arg0, x1, arg1, y, lam)?;
    code.into()
  }

  pub fn append_nd_map2<S: Borrow<str>>(&mut self, x0: &str, arg0: Dim, x1: &str, arg1: Dim, y: &str, lam: S) -> Result<(), FutharkThunkGenErr> {
    let lam = lam.borrow();
    match (arg0.ndim(), arg1.ndim()) {
      (0, 0) => {
        self.append(format!(r"let {} = ({}) {} {} in", y, lam, x0, x1));
      }
      (1, 1) => {
        self.append(format!(r"let {} = map2 ({}) {} {} in", y, lam, x0, x1));
      }
      (2, 2) => {
        self.append(format!(r"let t0 = flatten {} in", x0));
        self.append(format!(r"let t1 = flatten {} in", x1));
        self.append(format!(r"let t2 = map2 ({}) t0 t1 in", lam));
        self.append(format!(r"let {} = unflatten t2 in", y));
      }
      (3, 3) => {
        self.append(format!(r"let t0 = flatten_3d {} in", x0));
        self.append(format!(r"let t1 = flatten_3d {} in", x1));
        self.append(format!(r"let t2 = map2 ({}) t0 t1 in", lam));
        self.append(format!(r"let {} = unflatten_3d t2 in", y));
      }
      (4, 4) => {
        self.append(format!(r"let t0 = flatten_4d {} in", x0));
        self.append(format!(r"let t1 = flatten_4d {} in", x1));
        self.append(format!(r"let t2 = map2 ({}) t0 t1 in", lam));
        self.append(format!(r"let {} = unflatten_4d t2 in", y));
      }
      _ => {
        println!("WARNING: FutharkThunkGenCode::append_nd_map2: not implemented: {:?} {:?}", arg0, arg1);
        return Err(FutharkThunkGenErr::NotImpl);
      }
    }
    Ok(())
  }

  pub fn nd_map3_<S: Borrow<str>>(x0: &str, arg0: Dim, x1: &str, arg1: Dim, x2: &str, arg2: Dim, y: &str, lam: S) -> Result<FutharkThunkGenCode, FutharkThunkGenErr> {
    let mut code = FutharkThunkGenCode::default();
    code.abi.arityout = 1;
    code.abi.set_out(0, FutharkArrayRepr::Nd);
    code.abi.arityin = 3;
    code.abi.set_arg(0, FutharkArrayRepr::Nd);
    code.abi.set_arg(1, FutharkArrayRepr::Nd);
    code.abi.set_arg(2, FutharkArrayRepr::Nd);
    code.append_nd_map3(x0, arg0, x1, arg1, x2, arg2, y, lam)?;
    code.into()
  }

  pub fn append_nd_map3<S: Borrow<str>>(&mut self, x0: &str, arg0: Dim, x1: &str, arg1: Dim, x2: &str, arg2: Dim, y: &str, lam: S) -> Result<(), FutharkThunkGenErr> {
    let lam = lam.borrow();
    match (arg0.ndim(), arg1.ndim(), arg2.ndim()) {
      (0, 0, 0) => {
        self.append(format!(r"let {} = ({}) {} {} {} in", y, lam, x0, x1, x2));
      }
      (1, 1, 1) => {
        self.append(format!(r"let {} = map3 ({}) {} {} {} in", y, lam, x0, x1, x2));
      }
      (2, 2, 2) => {
        self.append(format!(r"let t0 = flatten {} in", x0));
        self.append(format!(r"let t1 = flatten {} in", x1));
        self.append(format!(r"let t2 = flatten {} in", x2));
        self.append(format!(r"let t3 = map3 ({}) t0 t1 t2 in", lam));
        self.append(format!(r"let {} = unflatten t3 in", y));
      }
      (3, 3, 3) => {
        self.append(format!(r"let t0 = flatten_3d {} in", x0));
        self.append(format!(r"let t1 = flatten_3d {} in", x1));
        self.append(format!(r"let t2 = flatten_3d {} in", x2));
        self.append(format!(r"let t3 = map3 ({}) t0 t1 t2 in", lam));
        self.append(format!(r"let {} = unflatten_3d t3 in", y));
      }
      (4, 4, 4) => {
        self.append(format!(r"let t0 = flatten_4d {} in", x0));
        self.append(format!(r"let t1 = flatten_4d {} in", x1));
        self.append(format!(r"let t2 = flatten_4d {} in", x2));
        self.append(format!(r"let t3 = map3 ({}) t0 t1 t2 in", lam));
        self.append(format!(r"let {} = unflatten_4d t3 in", y));
      }
      _ => {
        println!("WARNING: FutharkThunkGenCode::append_nd_map3: not implemented: {:?} {:?} {:?}", arg0, arg1, arg2);
        return Err(FutharkThunkGenErr::NotImpl);
      }
    }
    Ok(())
  }

  pub fn pre_append<S: Into<String>>(&mut self, line: S) {
    self.head.push(line.into());
  }

  pub fn append<S: Into<String>>(&mut self, line: S) {
    self.body.push(line.into());
  }

  //pub fn gen_source(&self, abi: &FutAbi, spec_dim: &[Dim], mode: ThunkMode, mut cfg: FutharkThunkGenConfig) -> Result<Vec<String>, ()> {}
  pub fn gen_source(&self, /*abi: &FutAbi,*/ lar: u16, rar: u16, spec_dim: &[Dim], mode: ThunkMode, mut cfg: FutharkThunkGenConfig, mut genabi: FutharkThunkGenAbi) -> Result<(FutharkThunkGenAbi, Vec<String>), ()> {
    // TODO
    let mut warn = false;
    for k in 0 .. genabi.arityout {
      match genabi.get_out(k) {
        FutharkArrayRepr::Nd |
        FutharkArrayRepr::Flat => {}
        _ => {
          println!("WARNING: FutharkThunkGenCode::gen_source: invalid array repr: out={}", k);
          warn = true;
        }
      }
    }
    for k in 0 .. genabi.arityin {
      match genabi.get_arg(k) {
        FutharkArrayRepr::Nd |
        FutharkArrayRepr::Flat => {}
        _ => {
          println!("WARNING: FutharkThunkGenCode::gen_source: invalid array repr: arg={}", k);
          warn = true;
        }
      }
    }
    if warn {
      for line in self.head.iter() {
        println!("DEBUG: FutharkThunkGenCode::gen_source: dump head: {}", line);
      }
      for line in self.body.iter() {
        println!("DEBUG: FutharkThunkGenCode::gen_source: dump body: {}", line);
      }
    }
    cfg = self.cfg.merge(cfg);
    let mut pats = Vec::new();
    let mut reps = Vec::new();
    for k in 0 .. lar {
      pats.push(format!(r"{{%{}}}", k));
      reps.push(format!(r"x_{}", k));
      if cfg.emit_arg_shapes {
        let dim = spec_dim[k as usize];
        let nd = match genabi.get_arg(k) {
          FutharkArrayRepr::Nd => dim.ndim,
          FutharkArrayRepr::Flat => min(1, dim.ndim),
          //_ => unimplemented!()
          _ => {
            println!("DEBUG: FutharkThunkGenCode::gen_source: lar={} rar={} spec_dim={:?} mode={:?} cfg={:?}",
                lar, rar, spec_dim, mode, cfg);
            //println!("DEBUG: FutharkThunkGenCode::gen_source: futabi={:?}", abi);
            println!("DEBUG: FutharkThunkGenCode::gen_source: genabi={:?}", genabi);
            unimplemented!()
          }
        };
        for d in 0 .. nd {
          pats.push(format!(r"{{%{}.s[{}]}}", k, d));
          reps.push(format!(r"x_{}_s_{}", k, d));
        }
        pats.push(format!(r"{{%{}.s~}}", k));
        let mut r = String::new();
        for d in 0 .. nd {
          if d > 0 {
            write!(&mut r, r" ").unwrap();
          }
          write!(&mut r, r"x_{}_s_{}", k, d).unwrap();
        }
        reps.push(r);
        pats.push(format!(r"{{%{}.s:~}}", k));
        let mut r = String::new();
        for d in 0 .. nd {
          if d > 0 {
            write!(&mut r, r" ").unwrap();
          }
          write!(&mut r, r"(x_{}_s_{}: i64)", k, d).unwrap();
        }
        reps.push(r);
        pats.push(format!(r"{{%{}.s*}}", k));
        let mut r = String::new();
        for d in 0 .. nd {
          if d > 0 {
            write!(&mut r, r" * ").unwrap();
          }
          write!(&mut r, r"x_{}_s_{}", k, d).unwrap();
        }
        reps.push(r);
        pats.push(format!(r"{{%{}.s}}", k));
        let mut r = String::new();
        write!(&mut r, r"[").unwrap();
        for d in 0 .. nd {
          if d > 0 {
            write!(&mut r, r"][").unwrap();
          }
          write!(&mut r, r"x_{}_s_{}", k, d).unwrap();
        }
        write!(&mut r, r"]").unwrap();
        reps.push(r);
      }
    }
    for k in 0 .. rar {
      pats.push(format!(r"{{%{}}}", lar + k));
      reps.push(format!(r"y_{}", k));
      if k == 0 && cfg.emit_out0_shape {
        let dim = spec_dim[(lar + k) as usize];
        let nd = match genabi.get_out(k) {
          FutharkArrayRepr::Nd => dim.ndim,
          FutharkArrayRepr::Flat => min(1, dim.ndim),
          _ => unimplemented!()
        };
        for d in 0 .. nd {
          pats.push(format!(r"{{%{}.s[{}]}}", lar + k, d));
          reps.push(format!(r"y_{}_s_{}", k, d));
        }
        pats.push(format!(r"{{%{}.s~}}", lar + k));
        let mut r = String::new();
        for d in 0 .. nd {
          if d > 0 {
            write!(&mut r, r" ").unwrap();
          }
          write!(&mut r, r"y_{}_s_{}", k, d).unwrap();
        }
        reps.push(r);
        pats.push(format!(r"{{%{}.s:~}}", lar + k));
        let mut r = String::new();
        for d in 0 .. nd {
          if d > 0 {
            write!(&mut r, r" ").unwrap();
          }
          write!(&mut r, r"(y_{}_s_{}: i64)", k, d).unwrap();
        }
        reps.push(r);
        pats.push(format!(r"{{%{}.s*}}", lar + k));
        let mut r = String::new();
        for d in 0 .. nd {
          if d > 0 {
            write!(&mut r, r" * ").unwrap();
          }
          write!(&mut r, r"y_{}_s_{}", k, d).unwrap();
        }
        reps.push(r);
        pats.push(format!(r"{{%{}.s}}", lar + k));
        let mut r = String::new();
        write!(&mut r, r"[").unwrap();
        for d in 0 .. nd {
          if d > 0 {
            write!(&mut r, r"][").unwrap();
          }
          write!(&mut r, r"y_{}_s_{}", k, d).unwrap();
        }
        write!(&mut r, r"]").unwrap();
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
      for k in 0 .. lar {
        let dim = spec_dim[k as usize];
        let nd = match genabi.get_arg(k) {
          FutharkArrayRepr::Nd => dim.ndim,
          FutharkArrayRepr::Flat => min(1, dim.ndim),
          _ => unimplemented!()
        };
        for d in 0 .. nd {
          write!(&mut s, " [x_{}_s_{}]", k, d).unwrap();
        }
      }
    }
    if cfg.emit_out0_shape && !cfg.emit_out0_shape_param {
      assert_eq!(rar, 1);
      let dim = spec_dim[lar as usize];
      let nd = match genabi.get_out(0) {
        FutharkArrayRepr::Nd => dim.ndim,
        FutharkArrayRepr::Flat => min(1, dim.ndim),
        _ => unimplemented!()
      };
      for d in 0 .. nd {
        write!(&mut s, " [y_{}_s_{}]", 0, d).unwrap();
      }
    }
    for k in 0 .. lar {
      let dim = spec_dim[k as usize];
      write!(&mut s, " (x_{}: {})", k, genabi._to_futhark_entry_arg_type(k, dim, cfg.emit_arg_shapes)).unwrap();
    }
    if rar == 1 {
      match mode {
        ThunkMode::Apply => {
          let dim = spec_dim[lar as usize];
          if cfg.emit_out0_shape_param {
            //let np = abi.num_param();
            let nd = match genabi.get_out(0) {
              FutharkArrayRepr::Nd => dim.ndim,
              FutharkArrayRepr::Flat => min(1, dim.ndim),
              _ => unimplemented!()
            };
            for d in 0 .. nd {
              // FIXME FIXME: don't push param here...
              //abi.push_param(np as u16 + d as u16, FutAbiScalarType::I64);
              write!(&mut s, " (y_{}_s_{}: i64)", 0, d).unwrap();
            }
            // FIXME
            //abi.set_emit_out0_shape_param(true);
          }
          if cfg.emit_out0_shape {
            write!(&mut s, " : {}", genabi._to_futhark_entry_out0_type(0, dim)).unwrap();
          } else {
            write!(&mut s, " : {}", genabi._to_futhark_entry_out_type(0, dim)).unwrap();
          }
        }
        //ThunkMode::Apply1 |
        ThunkMode::Accumulate => {
          let dim = spec_dim[lar as usize];
          if dim.ndim >= 1 && !cfg.emit_out0_shape {
            cfg.emit_out0_shape = true;
            return self.gen_source(/*abi,*/ lar, rar, spec_dim, mode, cfg, genabi);
          }
          let fty = if cfg.emit_out0_shape {
            genabi._to_futhark_entry_out0_type(0, dim)
          } else {
            genabi._to_futhark_entry_out_type(0, dim)
          };
          write!(&mut s, " (oy_{}: *{}) : *{}", 0, fty, fty).unwrap();
        }
        _ => unimplemented!()
      }
    } else if mode == ThunkMode::Apply {
      write!(&mut s, " : (").unwrap();
      for k in 0 .. rar {
        let dim = spec_dim[(lar + k) as usize];
        if k == 0 && cfg.emit_out0_shape {
          write!(&mut s, "{}, ", genabi._to_futhark_entry_out0_type(0, dim)).unwrap();
        } else {
          write!(&mut s, "{}, ", genabi._to_futhark_entry_out_type(0, dim)).unwrap();
        }
      }
      write!(&mut s, ")").unwrap();
    } else {
      unimplemented!();
    }
    write!(&mut s, " =\n").unwrap();
    for k in 0 .. lar {
      let dim = spec_dim[k as usize];
      if dim.ndim == 0 {
        write!(&mut s, "\tlet x_{} = x_{}[0] in\n", k, k).unwrap();
      }
    }
    if rar == 1 {
      match mode {
        ThunkMode::Apply => {}
        /*//ThunkMode::Apply1 |
        ThunkMode::Accumulate => {
          let dim = spec_dim[lar as usize];
          if dim.ndim == 0 {
            write!(&mut s, "\tlet oy_{} = oy_{}[0] in\n", 0, 0).unwrap();
          }
        }*/
        ThunkMode::Accumulate => {}
        ThunkMode::Initialize => {
          let dim = spec_dim[lar as usize];
          if dim.ndim == 0 {
            write!(&mut s, "\tlet oy_{} = oy_{}[0] in\n", 0, 0).unwrap();
          }
          write!(&mut s, "\tlet y_{} = oy_{} in\n", 0, 0).unwrap();
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
    if rar == 1 {
      match mode {
        ThunkMode::Apply => {}
        ThunkMode::Initialize => {}
        ThunkMode::Accumulate => {
          let dim = spec_dim[lar as usize];
          if dim.ndim >= 1 {
            assert!(cfg.emit_out0_shape);
          }
          let nd = match genabi.get_out(0) {
            FutharkArrayRepr::Nd => dim.ndim,
            FutharkArrayRepr::Flat => min(1, dim.ndim),
            _ => unimplemented!()
          };
          match nd {
            0 => {
              write!(&mut s, "\tlet oy_{} = oy_{}[0] in\n", 0, 0).unwrap();
              write!(&mut s, "\tlet y_{} = oy_{} + y_{} in\n", 0, 0, 0).unwrap();
            }
            1 => {
              let fty = genabi._to_futhark_entry_out0_type(0, dim);
              write!(&mut s, "\tlet y_{} = map2 (+) oy_{} (y_{} :> {}) in\n", 0, 0, 0, fty).unwrap();
            }
            2 => {
              let fty = genabi._to_futhark_entry_out0_type(0, dim);
              write!(&mut s, "\tlet oy_{} = flatten oy_{} in\n", 0, 0).unwrap();
              write!(&mut s, "\tlet y_{} = flatten (y_{} :> {}) in\n", 0, 0, fty).unwrap();
              write!(&mut s, "\tlet y_{} = map2 (+) oy_{} y_{} in\n", 0, 0, 0).unwrap();
              write!(&mut s, "\tlet y_{} = unflatten y_{} in\n", 0, 0).unwrap();
            }
            3 => {
              let fty = genabi._to_futhark_entry_out0_type(0, dim);
              write!(&mut s, "\tlet oy_{} = flatten_3d oy_{} in\n", 0, 0).unwrap();
              write!(&mut s, "\tlet y_{} = flatten_3d (y_{} :> {}) in\n", 0, 0, fty).unwrap();
              write!(&mut s, "\tlet y_{} = map2 (+) oy_{} y_{} in\n", 0, 0, 0).unwrap();
              write!(&mut s, "\tlet y_{} = unflatten_3d y_{} in\n", 0, 0).unwrap();
            }
            4 => {
              let fty = genabi._to_futhark_entry_out0_type(0, dim);
              write!(&mut s, "\tlet oy_{} = flatten_4d oy_{} in\n", 0, 0).unwrap();
              write!(&mut s, "\tlet y_{} = flatten_4d (y_{} :> {}) in\n", 0, 0, fty).unwrap();
              write!(&mut s, "\tlet y_{} = map2 (+) oy_{} y_{} in\n", 0, 0, 0).unwrap();
              write!(&mut s, "\tlet y_{} = unflatten_4d y_{} in\n", 0, 0).unwrap();
            }
            _ => unimplemented!()
          }
        }
        _ => unimplemented!()
      }
    }
    for k in 0 .. rar {
      let dim = spec_dim[(lar + k) as usize];
      if dim.ndim == 0 {
        write!(&mut s, "\tlet y_{} = [y_{}] in\n", k, k).unwrap();
      }
    }
    write!(&mut s, "\t").unwrap();
    if rar == 1 {
      write!(&mut s, "y_{}", 0).unwrap();
    } else if mode == ThunkMode::Apply {
      write!(&mut s, "(").unwrap();
      for k in 0 .. rar {
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
    match mode {
      ThunkMode::Accumulate |
      ThunkMode::Initialize => {
        let arityin = genabi.arityin;
        assert_eq!(arityin, lar);
        genabi.arityin += 1;
        genabi.set_arg(arityin, genabi.get_out(0));
      }
      _ => {}
    }
    if cfg.emit_out0_shape_param {
      let prev_np = genabi.param_ct;
      let dim = spec_dim[lar as usize];
      let nd = match genabi.get_out(0) {
        FutharkArrayRepr::Nd => dim.ndim,
        FutharkArrayRepr::Flat => min(1, dim.ndim),
        _ => unimplemented!()
      };
      genabi.param_ct += nd as u16;
      for d in 0 .. nd {
        genabi.set_param(prev_np + d as u16, FutharkParam::Out0Shape, FutAbiScalarType::I64);
      }
    }
    Ok((genabi, lines_out))
  }
}

pub struct FutharkThunkObject<B: FutBackend> {
  pub genabi:   FutharkThunkGenAbi,
  pub source:   Vec<String>,
  pub obj:      FutObject<B>,
  pub consts:   Vec<(PAddr, StableCell)>,
  pub out0_tag: Option<(u32, TagUnifier)>,
}

pub trait FutharkThunkImpl_<B: FutBackend> {
  fn _dropck(&mut self);
  unsafe fn _setup_object(obj: &mut FutObject<B>);
  fn _build_object(ctr: &CtxCtr, env: &mut CtxEnv, config: &FutConfig, name: Option<&str>, source: &[String], rst: Counter) -> Option<(FutObject<B>, Vec<(PAddr, StableCell)>)>;
}

pub struct FutharkThunkImpl<B: FutBackend> where FutharkThunkImpl<B>: FutharkThunkImpl_<B> {
  //pub abi:      FutAbi,
  pub lar:  u16,
  pub rar:  u16,
  pub param:    Vec<FutAbiScalar>,
  pub spec_dim: Vec<Dim>,
  pub code:     FutharkThunkGenCode,
  pub name:     Option<&'static str>,
  //pub source:   RefCell<String>,
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

  fn _build_object(ctr: &CtxCtr, env: &mut CtxEnv, config: &FutConfig, name: Option<&str>, source: &[String], rst: Counter) -> Option<(FutObject<MulticoreBackend>, Vec<(PAddr, StableCell)>)> {
    // FIXME FIXME
    let t0 = Stopwatch::tl_stamp();
    match config.build::<MulticoreBackend>(FutStage::Dylib, name, source) {
      Err(e) => {
        println!("WARNING: FutharkThunkImpl::<MulticoreBackend>::_build_object: build error: {:?}", e);
        None
      }
      Ok(None) => panic!("bug"),
      Ok(Some(mut obj)) => {
        let t1 = Stopwatch::tl_stamp();
        if cfg_debug() { println!("DEBUG: FutharkThunkImpl::<MulticoreBackend>::_build_object:   build elapsed: {:.09} s", t1 - t0); }
        TL_CTX.with(|ctx| {
          if rst.rst <= 0 {
            panic!("bug");
          } else if rst.rst == 1 {
            ctx.timing.f_build1.borrow_mut().push(t1 - t0);
          } else {
            ctx.timing.f_build.borrow_mut().push(t1 - t0);
          }
        });
        let t0 = t1;
        // NB: futhark object ctx may create constants that need to be tracked.
        let mut consts = Vec::new();
        let pstart = TL_PCTX.with(|pctx| {
          pctx.ctr.next_addr()
        });
        unsafe { FutharkThunkImpl::<MulticoreBackend>::_setup_object(&mut obj); }
        let t1 = Stopwatch::tl_stamp();
        if cfg_debug() { println!("DEBUG: FutharkThunkImpl::<MulticoreBackend>::_build_object:   setup elapsed: {:.09} s", t1 - t0); }
        TL_CTX.with(|ctx| {
          if rst.rst <= 0 {
            panic!("bug");
          } else if rst.rst == 1 {
            ctx.timing.f_setup1.borrow_mut().push(t1 - t0);
          } else {
            ctx.timing.f_setup.borrow_mut().push(t1 - t0);
          }
        });
        let pfin = TL_PCTX.with(|pctx| {
          pctx.ctr.peek_addr()
        });
        for p in (pstart.to_unchecked() ..= pfin.to_unchecked()) {
          let p = PAddr::from_unchecked(p);
          let x = ctr.fresh_cel();
          if cfg_debug() { println!("DEBUG: FutharkThunkImpl::<MulticoreBackend>::_build_object: const: {:?} {:?}", p, x); }
          // FIXME: futhark consts should be marked pin.
          TL_PCTX.with(|pctx| {
            // FIXME: the type of the constant could probably be inferred,
            // but easier to defer it until unification.
            let ty = CellType::top();
            let mut pcel = PCell::new(x, ty.clone());
            // FIXME FIXME: could be gpu pmach (for page-locked mem).
            //let pmach = PMach::Smp;
            let pmach = PMach::NvGpu;
            let locus = Locus::Mem;
            let base_clk: Clock = rst.into();
            let xclk = base_clk.init_once();
            pcel.push_new_replica(x, xclk, locus, pmach, p);
            env.insert_phy(x, ty, pcel);
          });
          consts.push((p, StableCell::retain(env, x)));
          // FIXME: what else?
        }
        if consts.len() > 0 {
          if cfg_debug() { println!("DEBUG: FutharkThunkImpl::<MulticoreBackend>::_build_object: consts={:?}", consts); }
        }
        Some((obj, consts))
      }
    }
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
    (obj.ffi.ctx_cfg_set_cuEventCreate.as_ref().unwrap())(obj.cfg, LIBCUDA.cuEventCreate.as_ref().unwrap().as_ptr() as _);
    (obj.ffi.ctx_cfg_set_cuEventDestroy.as_ref().unwrap())(obj.cfg, LIBCUDA.cuEventDestroy.as_ref().unwrap().as_ptr() as _);
    (obj.ffi.ctx_cfg_set_cuEventRecord.as_ref().unwrap())(obj.cfg, LIBCUDA.cuEventRecord.as_ref().unwrap().as_ptr() as _);
    (obj.ffi.ctx_cfg_set_cuEventElapsedTime.as_ref().unwrap())(obj.cfg, LIBCUDA.cuEventElapsedTime.as_ref().unwrap().as_ptr() as _);
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

  fn _build_object(ctr: &CtxCtr, env: &mut CtxEnv, config: &FutConfig, name: Option<&str>, source: &[String], rst: Counter) -> Option<(FutObject<CudaBackend>, Vec<(PAddr, StableCell)>)> {
    assert!(TL_LIBNVRTC_BUILTINS_BARRIER.with(|&bar| bar));
    TL_PCTX.with(|pctx| {
      let dev = pctx.nvgpu.as_ref().unwrap().dev();
      cudart_set_cur_dev(dev).unwrap();
    });
    let t0 = Stopwatch::tl_stamp();
    match config.build::<CudaBackend>(FutStage::Dylib, name, source) {
      Err(e) => {
        println!("WARNING: FutharkThunkImpl::<CudaBackend>::_build_object: build error: {:?}", e);
        None
      }
      Ok(None) => panic!("bug"),
      Ok(Some(mut obj)) => {
        let t1 = Stopwatch::tl_stamp();
        if cfg_debug() { println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_build_object:   build elapsed: {:.09} s", t1 - t0); }
        TL_CTX.with(|ctx| {
          if rst.rst <= 0 {
            panic!("bug");
          } else if rst.rst == 1 {
            ctx.timing.f_build1.borrow_mut().push(t1 - t0);
          } else {
            ctx.timing.f_build.borrow_mut().push(t1 - t0);
          }
        });
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
        TL_CTX.with(|ctx| {
          if rst.rst <= 0 {
            panic!("bug");
          } else if rst.rst == 1 {
            ctx.timing.f_setup1.borrow_mut().push(t1 - t0);
          } else {
            ctx.timing.f_setup.borrow_mut().push(t1 - t0);
          }
        });
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

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub enum FutharkArrayRepr {
  _Top = 0,
  Nd = 1,
  Flat = 2,
}

impl FutharkArrayRepr {
  pub fn from_bits(u: u8) -> FutharkArrayRepr {
    match u {
      0 => FutharkArrayRepr::_Top,
      1 => FutharkArrayRepr::Nd,
      2 => FutharkArrayRepr::Flat,
      _ => panic!("bug")
    }
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub enum FutharkParam {
  _Top = 0,
  Spec = 1,
  Out0Shape = 2,
}

impl FutharkParam {
  pub fn from_bits(u: u8) -> FutharkParam {
    match u {
      0 => FutharkParam::_Top,
      1 => FutharkParam::Spec,
      2 => FutharkParam::Out0Shape,
      _ => panic!("bug")
    }
  }
}

//#[derive(Clone, Debug)]
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct FutharkThunkGenAbi {
  // TODO
  pub arityout: u16,
  pub arityin:  u16,
  pub param_ct: u16,
  pub data: Vec<u8>,
}

impl Default for FutharkThunkGenAbi {
  fn default() -> FutharkThunkGenAbi {
    FutharkThunkGenAbi{
      arityout: 0,
      arityin:  0,
      param_ct: 0,
      data: Vec::new(),
    }
  }
}

impl FutharkThunkGenAbi {
  pub fn to_eabi(&self, space: FutAbiSpace) -> FutEntryAbi {
    let mut eabi = FutEntryAbi{
      arityout: self.arityout,
      arityin:  self.arityin,
      param_ct: self.param_ct,
      space,
      data: Vec::new(),
    };
    for idx in 0 .. self.param_ct {
      let (_, sty) = self.get_param(idx);
      eabi.set_param(idx, sty);
    }
    eabi
  }

  pub fn get_out(&self, idx: u16) -> FutharkArrayRepr {
    assert!(idx < self.arityout);
    let u = self.data[idx as usize];
    FutharkArrayRepr::from_bits(u)
  }

  pub fn set_out(&mut self, idx: u16, rep: FutharkArrayRepr) {
    assert!(idx < self.arityout);
    if self.data.len() <= idx as usize {
      self.data.resize(idx as usize + 1, 0);
    }
    self.data[idx as usize] = rep as u8;
  }

  pub fn get_arg(&self, idx: u16) -> FutharkArrayRepr {
    assert!(idx < self.arityin);
    let u = self.data[(self.arityout + idx) as usize];
    FutharkArrayRepr::from_bits(u)
  }

  pub fn set_arg(&mut self, idx: u16, rep: FutharkArrayRepr) {
    assert!(idx < self.arityin);
    if self.data.len() <= (self.arityout + idx) as usize {
      self.data.resize((self.arityout + idx) as usize + 1, 0);
    }
    self.data[(self.arityout + idx) as usize] = rep as u8;
  }

  pub fn get_param(&self, idx: u16) -> (FutharkParam, FutAbiScalarType) {
    assert!(idx < self.param_ct);
    let u = self.data[(self.arityout + self.arityin + idx) as usize];
    (FutharkParam::from_bits(u & 3), FutAbiScalarType::from_bits(u >> 4))
  }

  pub fn set_param(&mut self, idx: u16, param: FutharkParam, sty: FutAbiScalarType) {
    assert!(idx < self.param_ct);
    if self.data.len() <= (self.arityout + self.arityin + idx) as usize {
      self.data.resize((self.arityout + self.arityin + idx) as usize + 1, 0);
    }
    self.data[(self.arityout + self.arityin + idx) as usize] = ((sty as u8) << 4) | (param as u8);
  }
}

impl FutharkThunkGenAbi {
  pub fn _to_futhark_entry_out0_type(&self, k: u16, dim: Dim) -> String {
    assert_eq!(k, 0);
    let mut s = String::new();
    if dim.ndim == 0 {
      s.push_str("[1]");
    }
    let nd = match self.get_out(k) {
      FutharkArrayRepr::Nd => dim.ndim,
      FutharkArrayRepr::Flat => min(1, dim.ndim),
      _ => unimplemented!()
    };
    for i in 0 .. nd {
      write!(&mut s, "[y_{}_s_{}]", 0, i).unwrap();
    }
    s.push_str(dim.dtype.format_futhark());
    s
  }

  pub fn _to_futhark_entry_out_type(&self, k: u16, dim: Dim) -> String {
    let mut s = String::new();
    if dim.ndim == 0 {
      s.push_str("[1]");
    }
    let nd = match self.get_out(k) {
      FutharkArrayRepr::Nd => dim.ndim,
      FutharkArrayRepr::Flat => min(1, dim.ndim),
      _ => unimplemented!()
    };
    for _ in 0 .. nd {
      s.push_str("[]");
    }
    s.push_str(dim.dtype.format_futhark());
    s
  }

  pub fn _to_futhark_entry_arg_type(&self, k: u16, dim: Dim, shape: bool) -> String {
    let mut s = String::new();
    if dim.ndim == 0 {
      s.push_str("[1]");
    }
    let nd = match self.get_arg(k) {
      FutharkArrayRepr::Nd => dim.ndim,
      FutharkArrayRepr::Flat => min(1, dim.ndim),
      _ => unimplemented!()
    };
    for i in 0 .. nd {
      if shape {
        write!(&mut s, "[x_{}_s_{}]", k, i).unwrap();
      } else {
        s.push_str("[]");
      }
    }
    s.push_str(dim.dtype.format_futhark());
    s
  }
}

/*pub fn _to_futhark_entry_type(dim: Dim) -> String {
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
}*/

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct FutharkThunkGenConfig {
  // FIXME
  pub emit_arg_shapes: bool,
  pub emit_out0_shape: bool,
  pub emit_out0_shape_param: bool,
  pub emit_primal_def: bool,
}

impl Default for FutharkThunkGenConfig {
  fn default() -> FutharkThunkGenConfig {
    FutharkThunkGenConfig{
      // FIXME
      emit_arg_shapes: false,
      emit_out0_shape: false,
      emit_out0_shape_param: false,
      emit_primal_def: false,
    }
  }
}

impl FutharkThunkGenConfig {
  pub fn merge(self, rhs: FutharkThunkGenConfig) -> FutharkThunkGenConfig {
    FutharkThunkGenConfig{
      emit_arg_shapes:  self.emit_arg_shapes || rhs.emit_arg_shapes,
      emit_out0_shape:  self.emit_out0_shape || rhs.emit_out0_shape,
      emit_out0_shape_param:  self.emit_out0_shape_param || rhs.emit_out0_shape_param,
      emit_primal_def:  self.emit_primal_def || rhs.emit_primal_def,
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

impl<B: FutBackend> FutharkThunkImpl<B> where FutharkThunkImpl<B>: FutharkThunkImpl_<B> {
  pub fn _try_build(&self, ctr: &CtxCtr, env: &mut CtxEnv, mode: ThunkMode, rst: Counter) {
    let gencfg = FutharkThunkGenConfig::default();
    // FIXME: gen abi.
    if cfg_debug() { println!("DEBUG: FutharkThunkImpl::_try_build: name={:?}", self.name); }
    //let source = self.code.gen_source(&self.abi, &self.spec_dim, mode, gencfg).unwrap();
    let (genabi, source) = self.code.gen_source(/*&self.abi,*/ self.lar, self.rar, &self.spec_dim, mode, gencfg, self.code.abi.clone()).unwrap();
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
    if let Some((obj, consts)) = FutharkThunkImpl::<B>::_build_object(ctr, env, &config, self.name, &source, rst) {
      // FIXME: or swap.
      let object = FutharkThunkObject{genabi, source, obj, consts, out0_tag: None};
      self.objects.borrow_mut().insert(mode, object);
    }
  }
}

impl FutharkThunkImpl<MulticoreBackend> {
  pub fn _enter(&self, ctr: &CtxCtr, env: &mut CtxEnv, spec_: &dyn ThunkSpec_, arg: &[(CellPtr, Clock)], th: ThunkPtr, out: CellPtr, oclk: Clock, mode: ThunkMode) -> ThunkResult {
    if _cfg_debug_mode(mode) {
    println!("DEBUG: FutharkThunkImpl::<MulticoreBackend>::_enter: name={:?} mode={:?}",
        spec_.debug_name(), mode);
    }
    if self.objects.borrow().find(mode).is_none() {
      self._try_build(ctr, env, mode, oclk.ctr());
    }
    if self.objects.borrow().find(mode).is_none() {
      println!("BUG: FutharkThunkImpl::<MulticoreBackend>::apply: build error");
      panic!();
    }
    let mut objects = self.objects.borrow_mut();
    let mut object = objects.find_mut(mode).unwrap().1;
    let &mut FutharkThunkObject{ref genabi, ref mut obj, ref mut out0_tag, ..} = &mut object;
    if _cfg_debug_mode(mode) { println!("DEBUG: FutharkThunkImpl::<MulticoreBackend>::_enter: hash={}", &obj.src_hash); }
    if _cfg_debug_mode(mode) { println!("DEBUG: FutharkThunkImpl::<MulticoreBackend>::_enter: arg={:?}", arg); }
    let (lar, rar) = (self.lar, self.rar);
    let extra_lar = match mode {
      ThunkMode::Apply => {
        if 1 != rar {
          unimplemented!();
        }
        0
      }
      ThunkMode::Accumulate |
      ThunkMode::Initialize => {
        assert_eq!(1, rar);
        1
      }
      _ => unimplemented!()
    };
    assert_eq!(arg.len(), lar as usize);
    assert_eq!(genabi.arityin, lar + extra_lar);
    let mut arg_ty_: Vec<CellType> = Vec::with_capacity(genabi.arityin as usize);
    let mut arg_arr: Vec<UnsafeCell<FutArray>> = Vec::with_capacity(genabi.arityin as usize);
    'for_k: for k in 0 .. lar {
      let xroot = match env.lookup_ref(arg[k as usize].0) {
        None => panic!("bug"),
        Some(e) => e.root
      };
      for j in 0 .. k as usize {
        match env.lookup_ref(arg[j].0) {
          None => panic!("bug"),
          Some(e_j) => {
            let xroot_j = e_j.root;
            if xroot_j == xroot {
              if _cfg_debug_mode(mode) {
              println!("DEBUG: FutharkThunkImpl::<MulticoreBackend>::_enter: aliased args: xroot[{}]={:?} xroot[{}]={:?}",
                  j, xroot_j, k, xroot);
              }
              match env.lookup_ref(arg[k as usize].0) {
                None => panic!("bug"),
                Some(e) => {
                  if &arg_ty_[j] == e.ty {
                    arg_ty_.push(arg_ty_[j].clone());
                    let a = arg_arr[j].get_mut().clone();
                    arg_arr.push(a.into());
                  } else {
                    println!("DEBUG: FutharkThunkImpl::<MulticoreBackend>::_enter: aliased args: xty[{}]={:?} xty[{}]={:?}",
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
      let e = match env.pread_ref(arg[k as usize].0, arg[k as usize].1) {
        None => panic!("bug"),
        Some(e) => e
      };
      assert_eq!(self.spec_dim[k as usize], e.ty.to_dim());
      let a = match (genabi.get_arg(k), self.spec_dim[k as usize].ndim) {
        (FutharkArrayRepr::Nd, 0) |
        (FutharkArrayRepr::Nd, 1) |
        (FutharkArrayRepr::Flat, _) => FutArray::new_1d(),
        (FutharkArrayRepr::Nd, 2) => FutArray::new_2d(),
        (FutharkArrayRepr::Nd, 3) => FutArray::new_3d(),
        (FutharkArrayRepr::Nd, 4) => FutArray::new_4d(),
        _ => unimplemented!()
      };
      TL_PCTX.with(|pctx| {
        let loc = Locus::Mem;
        match e.cel_ {
          &mut Cell_::Phy(.., ref mut pcel) => {
            // FIXME FIXME
            //let pmach = PMach::Smp;
            let pmach = PMach::NvGpu;
            let addr = pcel.get(arg[k as usize].0, arg[k as usize].1, &e.ty, loc, pmach);
            //let (ptr, size) = pctx.lookup_mem_reg(addr).unwrap();
            match pctx.lookup(addr).and_then(|(_, _, icel)| icel.as_mem_reg()) {
              Some(reg) => {
                a.set_mem_parts(reg.ptr, reg.sz);
              }
              None => unimplemented!()
            }
          }
          _ => panic!("bug")
        }
      });
      if self.spec_dim[k as usize].ndim == 0 {
        a.set_shape(&[1]);
      } else {
        match genabi.get_arg(k) {
          FutharkArrayRepr::Nd => {
            a.set_shape(&e.ty.shape);
          }
          FutharkArrayRepr::Flat => {
            a.set_shape(&[e.ty.flat_len()]);
          }
          _ => unimplemented!()
        }
      }
      arg_ty_.push(e.ty);
      arg_arr.push(a.into());
    }
    //println!("DEBUG: FutharkThunkImpl::<MulticoreBackend>::_enter: arg_arr={:?}", &arg_arr);
    let mut arg_ndim = Vec::with_capacity(arg_arr.len());
    for (k, arr) in arg_arr.iter_mut().enumerate() {
      if _cfg_debug_mode(mode) {
      println!("DEBUG: FutharkThunkImpl::<MulticoreBackend>::_enter: arg_arr[{}]={:?}",
          k, arr.get_mut());
      }
      arg_ndim.push(arr.get_mut()._unset_ndim());
    }
    if _cfg_debug_mode(mode) { println!("DEBUG: FutharkThunkImpl::<MulticoreBackend>::_enter: out={:?} oclk={:?}", out, oclk); }
    assert_eq!(genabi.arityout, rar);
    let mut out_ty_ = Vec::with_capacity(genabi.arityout as usize);
    let mut out_arr: Vec<UnsafeCell<FutArray>> = Vec::with_capacity(genabi.arityout as usize);
    let mut out_org = None;
    for k in 0 .. genabi.arityout {
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
      assert_eq!(self.spec_dim[(lar + k) as usize], ty_.to_dim());
      match mode {
        // FIXME: if we no longer match on mode, then the pwrite check
        // needs to test which of the args are also outs.
        ThunkMode::Accumulate |
        ThunkMode::Initialize => {
          // FIXME: double check that out does not alias any args.
          let e = match env.pwrite_ref(out, oclk) {
            None => panic!("bug"),
            Some(e) => e
          };
          assert_eq!(self.spec_dim[(lar + k) as usize], e.ty.to_dim());
          assert_eq!(genabi.get_out(k), genabi.get_arg(lar + k));
          let a = match (genabi.get_arg(lar + k), self.spec_dim[(lar + k) as usize].ndim) {
            (FutharkArrayRepr::Nd, 0) |
            (FutharkArrayRepr::Nd, 1) |
            (FutharkArrayRepr::Flat, _) => FutArray::new_1d(),
            (FutharkArrayRepr::Nd, 2) => FutArray::new_2d(),
            (FutharkArrayRepr::Nd, 3) => FutArray::new_3d(),
            (FutharkArrayRepr::Nd, 4) => FutArray::new_4d(),
            _ => unimplemented!()
          };
          TL_PCTX.with(|pctx| {
            let loc = Locus::Mem;
            match e.cel_ {
              &mut Cell_::Phy(.., ref mut pcel) => {
                // FIXME FIXME
                //let pmach = PMach::Smp;
                let pmach = PMach::NvGpu;
                let addr = pcel.get(out, oclk, &e.ty, loc, pmach);
                //let (ptr, size) = pctx.lookup_mem_reg(addr).unwrap();
                match pctx.lookup(addr).and_then(|(_, _, icel)| icel.as_mem_reg()) {
                  Some(reg) => {
                    a.set_mem_parts(reg.ptr, reg.sz);
                    out_org = Some((addr, reg.ptr, reg.sz));
                  }
                  None => unimplemented!()
                }
              }
              _ => panic!("bug")
            }
          });
          if self.spec_dim[(lar + k) as usize].ndim == 0 {
            a.set_shape(&[1]);
          } else {
            match genabi.get_arg(lar + k) {
              FutharkArrayRepr::Nd => {
                a.set_shape(&e.ty.shape);
              }
              FutharkArrayRepr::Flat => {
                a.set_shape(&[e.ty.flat_len()]);
              }
              _ => unimplemented!()
            }
          }
          arg_ty_.push(ty_.clone());
          arg_arr.push(a.into());
        }
        _ => {}
      }
      out_ty_.push(ty_);
      out_arr.push(FutArray::null().into());
    }
    for (k, arr) in (&mut arg_arr[lar as usize ..]).iter_mut().enumerate() {
      if _cfg_debug_mode(mode) {
      println!("DEBUG: FutharkThunkImpl::<MulticoreBackend>::_enter: arg_arr[{}]={:?} (out in-place)",
          lar as usize + k, arr.get_mut());
      }
      arg_ndim.push(arr.get_mut()._unset_ndim());
    }
    //println!("DEBUG: FutharkThunkImpl::<MulticoreBackend>::_enter: out_arr={:?}", &out_arr);
    if _cfg_debug_mode(mode) {
    for (k, arr) in out_arr.iter_mut().enumerate() {
      println!("DEBUG: FutharkThunkImpl::<MulticoreBackend>::_enter: out_arr[{}]={:?}", k, arr.get_mut());
    }
    }
    // FIXME: implicit output shape params during gen.
    let np = genabi.param_ct as usize;
    let mut param: Vec<FutAbiScalar> = Vec::with_capacity(np);
    param.resize(np, FutAbiScalar::Unspec);
    //spec_.set_param(&mut param);
    let mut flat_out0_shape = false;
    for pidx in 0 .. np {
      match genabi.get_param(pidx as _) {
        (FutharkParam::Out0Shape, FutAbiScalarType::I64) => {
          match genabi.get_out(0) {
            FutharkArrayRepr::Nd => {
              param[pidx] = FutAbiScalar::I64(out_ty_[0].shape[pidx].into());
            }
            FutharkArrayRepr::Flat => {
              assert!(!flat_out0_shape);
              param[pidx] = FutAbiScalar::I64(out_ty_[0].flat_len().into());
              flat_out0_shape = true;
            }
            _ => unimplemented!()
          }
        }
        _ => unimplemented!()
      }
    }
    match mode {
      ThunkMode::Accumulate => {
        TL_CTX.with(|ctx| {
          let mut h = ctx.debugctr.accumulate_hashes.borrow_mut();
          match h.get_mut(&obj.src_hash) {
            None => {
              h.insert(obj.src_hash.clone(), 1);
            }
            Some(ct) => {
              *ct += 1;
            }
          }
        });
      }
      _ => {}
    }
    TL_PCTX.with(|pctx| {
      // TODO: barrier?
    });
    let t0 = Stopwatch::tl_stamp();
    obj.reset();
    if _cfg_debug_mode(mode) {
    let tmp_t1 = Stopwatch::tl_stamp();
    println!("DEBUG: FutharkThunkImpl::<MulticoreBackend>::_enter:   reset elapsed: {:.09} s", tmp_t1 - t0);
    }
    if _cfg_debug_mode(mode) { println!("DEBUG: FutharkThunkImpl::<MulticoreBackend>::_enter: enter kernel..."); }
    let eabi = genabi.to_eabi(FutAbiSpace::Default);
    let o_ret = obj.enter_kernel(eabi, &param, &arg_arr, &out_arr);
    if o_ret.is_err() {
      // FIXME FIXME: error handling.
      println!("ERROR: FutharkThunkImpl::<MulticoreBackend>::_enter: runtime error");
      if let Some(e) = obj.error().map(|c| safe_ascii(c.to_bytes())) {
        println!("ERROR: FutharkThunkImpl::<MulticoreBackend>::_enter: runtime error: {}", e);
      }
      panic!();
    }
    let may_fail = obj.may_fail();
    if _cfg_debug_mode(mode) { println!("DEBUG: FutharkThunkImpl::<MulticoreBackend>::_enter: may fail? {:?}", may_fail); }
    if may_fail {
      let ret = obj.sync();
      if ret.is_err() {
        // FIXME FIXME: failure handling.
        println!("ERROR: FutharkThunkImpl::<MulticoreBackend>::_enter: runtime failure: {:?}", ret);
        if let Some(e) = obj.error().map(|c| safe_ascii(c.to_bytes())) {
          println!("ERROR: FutharkThunkImpl::<MulticoreBackend>::_enter: runtime failure: {}", e);
        }
        panic!();
      }
    }
    obj.release();
    let t1 = Stopwatch::tl_stamp();
    if _cfg_debug_mode(mode) { println!("DEBUG: FutharkThunkImpl::<MulticoreBackend>::_enter:   elapsed: {:.09} s", t1 - t0); }
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
    if _cfg_debug_mode(mode) { println!("DEBUG: FutharkThunkImpl::<MulticoreBackend>::_enter: ret={:?}", o_ret); }
    //println!("DEBUG: FutharkThunkImpl::<MulticoreBackend>::_enter: out={:?} oclk={:?}", out, oclk);
    for (k, arr) in (&mut arg_arr[ .. lar as usize]).iter_mut().enumerate() {
      arr.get_mut()._set_ndim(arg_ndim[k]);
      if _cfg_debug_mode(mode) {
      println!("DEBUG: FutharkThunkImpl::<MulticoreBackend>::_enter: arg_arr[{}]={:?}",
          k, arr.get_mut());
      }
    }
    for (k, arr) in (&mut arg_arr[lar as usize .. ]).iter_mut().enumerate() {
      arr.get_mut()._set_ndim(arg_ndim[lar as usize + k]);
      if _cfg_debug_mode(mode) {
      println!("DEBUG: FutharkThunkImpl::<MulticoreBackend>::_enter: arg_arr[{}]={:?} (out in-place)",
          lar as usize + k, arr.get_mut());
      }
    }
    // FIXME: at this point, the remaining memblocks are the outputs.
    // but, if any of the inputs were clobbered, then we have to unset those.
    // so, do some kind of unification here.
    for k in 0 .. genabi.arityout {
      assert!(!out_arr[k as usize].get_mut().as_ptr().is_null());
      match genabi.get_out(k) {
        FutharkArrayRepr::Nd => {
          out_arr[k as usize].get_mut()._set_ndim(max(1, out_ty_[k as usize].ndim()));
        }
        FutharkArrayRepr::Flat => {
          out_arr[k as usize].get_mut()._set_ndim(1);
        }
        _ => unimplemented!()
      }
    }
    //println!("DEBUG: FutharkThunkImpl::<MulticoreBackend>::_enter: out_arr={:?}", &out_arr);
    if _cfg_debug_mode(mode) {
    for (k, arr) in out_arr.iter_mut().enumerate() {
      println!("DEBUG: FutharkThunkImpl::<MulticoreBackend>::_enter: out_arr[{}]={:?}", k, arr.get_mut());
    }
    }
    // TODO TODO
    let (out_ptr, out_size) = out_arr[0].get_mut().mem_parts().unwrap();
    match mode {
      ThunkMode::Accumulate => {
        assert!(out_org.is_some());
        let (out_org_addr, out_org_ptr, out_org_size) = out_org.unwrap();
        if out_org_ptr != out_ptr {
          if _cfg_debug_mode(mode) {
          println!("WARNING: FutharkThunkImpl::<MulticoreBackend>::_enter: out: Accumulate not in-place!");
          println!("WARNING: FutharkThunkImpl::<MulticoreBackend>::_enter: out:   org ptr =0x{:016x} sz={}",
              out_org_ptr as usize, out_org_size);
          println!("WARNING: FutharkThunkImpl::<MulticoreBackend>::_enter: out:   org addr={:?}",
              out_org_addr);
          println!("WARNING: FutharkThunkImpl::<MulticoreBackend>::_enter: out:   new ptr =0x{:016x} sz={}",
              out_ptr as usize, out_size);
          /*let new_addr = TL_PCTX.with(|pctx| {
            let gpu = pctx.nvgpu.as_ref().unwrap();
            match gpu.mem_pool.rev_lookup(out_dptr) {
              None => None,
              Some((_region, None)) => None,
              Some((_region, Some(p))) => Some(p)
            }
          });
          println!("WARNING: FutharkThunkImpl::<MulticoreBackend>::_enter: out:   new addr={:?}",
              new_addr);*/
          }
          TL_CTX.with(|ctx| {
            ctx.debugctr.accumulate_not_in_place.fetch_add(1);
          });
        } else {
          TL_CTX.with(|ctx| {
            ctx.debugctr.accumulate_in_place.fetch_add(1);
          });
        }
        // TODO
      }
      _ => {}
    }
    if _cfg_debug_mode(mode) {
    println!("DEBUG: FutharkThunkImpl::<MulticoreBackend>::_enter: out: shape={:?}",
        out_arr[0].get_mut().shape().unwrap());
    println!("DEBUG: FutharkThunkImpl::<MulticoreBackend>::_enter: out: rc={:?} ptr=0x{:016x} sz={}",
        out_arr[0].get_mut().refcount(), out_ptr as usize, out_size);
    }
    if out_ty_[0].ndim() == 0 {
      assert_eq!(Some(&[1_i64] as &[_]), out_arr[0].get_mut().shape());
    } else {
      match genabi.get_out(0) {
        FutharkArrayRepr::Nd => {
          assert_eq!(Some(&out_ty_[0].shape as &[_]), out_arr[0].get_mut().shape());
        }
        FutharkArrayRepr::Flat => {
          assert_eq!(Some(&[out_ty_[0].flat_len()] as &[_]), out_arr[0].get_mut().shape());
        }
        _ => unimplemented!()
      }
    }
    match out_arr[0].get_mut().tag() {
      None => {
        if _cfg_debug_mode(mode) { println!("DEBUG: FutharkThunkImpl::<MulticoreBackend>::_enter: out:   tag=null"); }
      }
      Some(ctag) => {
        if _cfg_debug_mode(mode) { println!("DEBUG: FutharkThunkImpl::<MulticoreBackend>::_enter: out:   tag=\"{}\"", safe_ascii(ctag.to_bytes())); }
      }
    }
    // TODO
    TL_PCTX.with(|pctx| {
      let gpu = pctx.nvgpu.as_ref().unwrap();
      let addr = match gpu.page_map.rev_lookup(out_ptr) {
        Some(addr) => addr,
        None => unimplemented!(),
      };
      let mut f = false;
      if !f {
        for k in objects.find(mode).unwrap().1.consts.iter() {
          if k.0 == addr {
            if _cfg_debug_mode(mode) { println!("DEBUG: FutharkThunkImpl::<MulticoreBackend>::_enter: out:   is const"); }
            match env.lookup_mut_ref(out) {
              None => panic!("bug"),
              Some(e) => {
                match e.cel_ {
                  &mut Cell_::Top(ref state, optr) => {
                    assert_eq!(e.root, optr);
                    // FIXME: defaults below are placeholders for...?
                    let state = RefCell::new(state.borrow().clone());
                    let clo = RefCell::new(CellClosure::default());
                    *e.cel_ = Cell_::Cow(state, clo, CowCell{optr, pcel: *(k.1).as_ref(), pclk: Clock::default()});
                    f = true;
                    if _cfg_debug_mode(mode) { println!("DEBUG: FutharkThunkImpl::<MulticoreBackend>::_enter: out: cow {:?} -> {:?}", out, addr); }
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
      if !f {
        match env.lookup_mut_ref(out) {
          None => panic!("bug"),
          Some(e) => {
            match e.cel_ {
              &mut Cell_::Top(ref state, optr) => {
                if _cfg_debug_mode(mode) { println!("DEBUG: FutharkThunkImpl::<MulticoreBackend>::_enter: out: try new phy..."); }
                assert_eq!(e.root, optr);
                // FIXME: defaults below are placeholders for...?
                let state = RefCell::new(state.borrow().clone());
                let clo = RefCell::new(CellClosure::default());
                let mut pcel = PCell::new(optr, out_ty_[0].clone());
                pcel.push_new_replica(optr, oclk, Locus::Mem, PMach::NvGpu, addr);
                *e.cel_ = Cell_::Phy(state, clo, pcel);
                f = true;
                if _cfg_debug_mode(mode) {
                println!("DEBUG: FutharkThunkImpl::<MulticoreBackend>::_enter: out: new phy {:?} --> {:?} -> {:?}",
                    out, optr, addr);
                }
              }
              &mut Cell_::Phy(ref state, .., ref mut pcel) => {
                if _cfg_debug_mode(mode) { println!("DEBUG: FutharkThunkImpl::<MulticoreBackend>::_enter: out: try old phy..."); }
                assert_eq!(e.root, pcel.optr);
                let optr = pcel.optr;
                if let Some(replica) = pcel.lookup(Locus::Mem, PMach::NvGpu) {
                  // NB: clk equal b/c spine did clock_sync.
                  assert_eq!(replica.clk.get(), oclk);
                  let prev_addr = replica.addr.get();
                  if prev_addr != addr {
                    if _cfg_debug_mode(mode) {
                    println!("DEBUG: FutharkThunkImpl::<MulticoreBackend>::_enter: out:   prev addr={:?}",
                        prev_addr);
                    println!("DEBUG: FutharkThunkImpl::<MulticoreBackend>::_enter: out:   next addr={:?}",
                        addr);
                    }
                    replica.addr.set(addr);
                    assert!(pctx.set_root(addr, optr).is_none());
                    let prev_root = pctx.unset_root(prev_addr);
                    let prev_root_root =
                    if let Some(prev_root) = prev_root {
                      let prev_root_root = match env.lookup_ref(prev_root) {
                        None => panic!("bug"),
                        Some(e) => e.root
                      };
                      if _cfg_debug_mode(mode) {
                      println!("DEBUG: FutharkThunkImpl::<MulticoreBackend>::_enter: out:   prev root={:?} --> {:?}",
                          prev_root, prev_root_root);
                      }
                      Some(prev_root_root)
                    } else {
                      None
                    };
                    if _cfg_debug_mode(mode) {
                    println!("DEBUG: FutharkThunkImpl::<MulticoreBackend>::_enter: out:   next root={:?} --> {:?}",
                        out, optr);
                    }
                    if !(prev_root == Some(optr)) && prev_root_root == Some(optr) {
                      if _cfg_debug_mode(mode) {
                      println!("WARNING: FutharkThunkImpl::<MulticoreBackend>::_enter: out:   slightly invalid prev root, but mostly harmless");
                      }
                    } else {
                      assert_eq!(prev_root, Some(optr));
                    }
                    // FIXME: gc the prev addr.
                    match gpu.page_map.release(prev_addr) {
                      Some(_) => {}
                      None => unimplemented!()
                    }
                  }
                } else {
                  pcel.push_new_replica(optr, oclk, Locus::Mem, PMach::NvGpu, addr);
                }
                f = true;
                if _cfg_debug_mode(mode) {
                println!("DEBUG: FutharkThunkImpl::<MulticoreBackend>::_enter: out: old phy {:?} --> {:?} -> {:?}",
                    out, optr, addr);
                }
              }
              _ => unimplemented!()
            }
          }
        }
      }
      if !f {
        panic!("bug");
      }
    });
    Ok(())
  }
}

impl ThunkImpl for FutharkThunkImpl<MulticoreBackend> {
  fn apply(&self, ctr: &CtxCtr, env: &mut CtxEnv, spec_: &dyn ThunkSpec_, arg: &[(CellPtr, Clock)], th: ThunkPtr, out: CellPtr, oclk: Clock) -> ThunkResult {
    let mode = ThunkMode::Apply;
    self._enter(ctr, env, spec_, arg, th, out, oclk, mode)
  }

  fn accumulate(&self, ctr: &CtxCtr, env: &mut CtxEnv, spec_: &dyn ThunkSpec_, arg: &[(CellPtr, Clock)], th: ThunkPtr, out: CellPtr, oclk: Clock) -> ThunkResult {
    let mode = ThunkMode::Accumulate;
    self._enter(ctr, env, spec_, arg, th, out, oclk, mode)
  }

  fn initialize(&self, ctr: &CtxCtr, env: &mut CtxEnv, spec_: &dyn ThunkSpec_, arg: &[(CellPtr, Clock)], th: ThunkPtr, out: CellPtr, oclk: Clock) -> ThunkResult {
    let mode = ThunkMode::Initialize;
    self._enter(ctr, env, spec_, arg, th, out, oclk, mode)
  }
}

#[cfg(feature = "nvgpu")]
impl FutharkThunkImpl<CudaBackend> {
  pub fn _enter(&self, ctr: &CtxCtr, env: &mut CtxEnv, spec_: &dyn ThunkSpec_, arg: &[(CellPtr, Clock)], th: ThunkPtr, out: CellPtr, oclk: Clock, mode: ThunkMode) -> ThunkResult {
    if _cfg_debug_mode(mode) {
    println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: name={:?} mode={:?}",
        spec_.debug_name(), mode);
    }
    if self.objects.borrow().find(mode).is_none() {
      self._try_build(ctr, env, mode, oclk.ctr());
    }
    if self.objects.borrow().find(mode).is_none() {
      println!("BUG: FutharkThunkImpl::<CudaBackend>::_enter: build error");
      panic!();
    }
    let mut objects = self.objects.borrow_mut();
    let mut object = objects.find_mut(mode).unwrap().1;
    let &mut FutharkThunkObject{ref genabi, ref mut obj, ref mut out0_tag, ..} = &mut object;
    if _cfg_debug_mode(mode) { println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: hash={}", &obj.src_hash); }
    if _cfg_debug_mode(mode) { println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: arg={:?}", arg); }
    let (lar, rar) = (self.lar, self.rar);
    let extra_lar = match mode {
      ThunkMode::Apply => {
        if 1 != rar {
          unimplemented!();
        }
        0
      }
      ThunkMode::Accumulate |
      ThunkMode::Initialize => {
        assert_eq!(1, rar);
        1
      }
      _ => unimplemented!()
    };
    assert_eq!(arg.len(), lar as usize);
    assert_eq!(genabi.arityin, lar + extra_lar);
    let mut arg_ty_: Vec<CellType> = Vec::with_capacity(genabi.arityin as usize);
    let mut arg_arr: Vec<UnsafeCell<FutArrayDev>> = Vec::with_capacity(genabi.arityin as usize);
    'for_k: for k in 0 .. lar {
      let xroot = match env.lookup_ref(arg[k as usize].0) {
        None => panic!("bug"),
        Some(e) => e.root
      };
      for j in 0 .. k as usize {
        match env.lookup_ref(arg[j].0) {
          None => panic!("bug"),
          Some(e_j) => {
            let xroot_j = e_j.root;
            if xroot_j == xroot {
              if _cfg_debug_mode(mode) {
              println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: aliased args: xroot[{}]={:?} xroot[{}]={:?}",
                  j, xroot_j, k, xroot);
              }
              match env.lookup_ref(arg[k as usize].0) {
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
      let e = match env.pread_ref(arg[k as usize].0, arg[k as usize].1) {
        None => panic!("bug"),
        Some(e) => e
      };
      assert_eq!(self.spec_dim[k as usize], e.ty.to_dim());
      let a = match (genabi.get_arg(k), self.spec_dim[k as usize].ndim) {
        (FutharkArrayRepr::Nd, 0) |
        (FutharkArrayRepr::Nd, 1) |
        (FutharkArrayRepr::Flat, _) => FutArrayDev::new_1d(),
        (FutharkArrayRepr::Nd, 2) => FutArrayDev::new_2d(),
        (FutharkArrayRepr::Nd, 3) => FutArrayDev::new_3d(),
        (FutharkArrayRepr::Nd, 4) => FutArrayDev::new_4d(),
        _ => unimplemented!()
      };
      TL_PCTX.with(|pctx| {
        let gpu = pctx.nvgpu.as_ref().unwrap();
        let loc = gpu.device_locus();
        match e.cel_ {
          &mut Cell_::Phy(.., ref mut pcel) => {
            let addr = pcel.get(arg[k as usize].0, arg[k as usize].1, &e.ty, loc, PMach::NvGpu);
            let (dptr, size) = gpu.lookup_dev(addr).unwrap();
            a.set_mem_parts(dptr, size);
          }
          _ => panic!("bug")
        }
      });
      if self.spec_dim[k as usize].ndim == 0 {
        a.set_shape(&[1]);
      } else {
        match genabi.get_arg(k) {
          FutharkArrayRepr::Nd => {
            a.set_shape(&e.ty.shape);
          }
          FutharkArrayRepr::Flat => {
            a.set_shape(&[e.ty.flat_len()]);
          }
          _ => unimplemented!()
        }
      }
      arg_ty_.push(e.ty);
      arg_arr.push(a.into());
    }
    //println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: arg_arr={:?}", &arg_arr);
    let mut arg_ndim = Vec::with_capacity(arg_arr.len());
    for (k, arr) in arg_arr.iter_mut().enumerate() {
      if _cfg_debug_mode(mode) {
      println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: arg_arr[{}]={:?}",
          k, arr.get_mut());
      }
      arg_ndim.push(arr.get_mut()._unset_ndim());
    }
    if _cfg_debug_mode(mode) { println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: out={:?} oclk={:?}", out, oclk); }
    assert_eq!(genabi.arityout, rar);
    let mut out_ty_ = Vec::with_capacity(genabi.arityout as usize);
    let mut out_arr: Vec<UnsafeCell<FutArrayDev>> = Vec::with_capacity(genabi.arityout as usize);
    let mut out_org = None;
    for k in 0 .. genabi.arityout {
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
      if !(self.spec_dim[(lar + k) as usize] == ty_.to_dim()) {
        println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: lar={} rar={} k={}", lar, rar, k);
        println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: {:?}", &self.spec_dim);
        println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: {:?}", self.spec_dim[(lar + k) as usize]);
        println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: arg={:?}", &arg);
        println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: arg ty={:?}", &arg_ty_);
        println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: out={:?}", out);
        println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: out ty={:?}", ty_);
      }
      assert_eq!(self.spec_dim[(lar + k) as usize], ty_.to_dim());
      match mode {
        // FIXME: if we no longer match on mode, then the pwrite check
        // needs to test which of the args are also outs.
        ThunkMode::Accumulate |
        ThunkMode::Initialize => {
          // FIXME: double check that out does not alias any args.
          let e = match env.pwrite_ref(out, oclk) {
            None => panic!("bug"),
            Some(e) => e
          };
          assert_eq!(self.spec_dim[(lar + k) as usize], e.ty.to_dim());
          assert_eq!(genabi.get_out(k), genabi.get_arg(lar + k));
          let a = match (genabi.get_arg(lar + k), self.spec_dim[(lar + k) as usize].ndim) {
            (FutharkArrayRepr::Nd, 0) |
            (FutharkArrayRepr::Nd, 1) |
            (FutharkArrayRepr::Flat, _) => FutArrayDev::new_1d(),
            (FutharkArrayRepr::Nd, 2) => FutArrayDev::new_2d(),
            (FutharkArrayRepr::Nd, 3) => FutArrayDev::new_3d(),
            (FutharkArrayRepr::Nd, 4) => FutArrayDev::new_4d(),
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
                out_org = Some((addr, dptr, size));
              }
              _ => panic!("bug")
            }
          });
          if self.spec_dim[(lar + k) as usize].ndim == 0 {
            a.set_shape(&[1]);
          } else {
            match genabi.get_arg(lar + k) {
              FutharkArrayRepr::Nd => {
                a.set_shape(&e.ty.shape);
              }
              FutharkArrayRepr::Flat => {
                a.set_shape(&[e.ty.flat_len()]);
              }
              _ => unimplemented!()
            }
          }
          arg_ty_.push(ty_.clone());
          arg_arr.push(a.into());
        }
        _ => {}
      }
      out_ty_.push(ty_);
      out_arr.push(FutArrayDev::null().into());
    }
    for (k, arr) in (&mut arg_arr[lar as usize ..]).iter_mut().enumerate() {
      if _cfg_debug_mode(mode) {
      println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: arg_arr[{}]={:?} (out in-place)",
          lar as usize + k, arr.get_mut());
      }
      arg_ndim.push(arr.get_mut()._unset_ndim());
    }
    //println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: out_arr={:?}", &out_arr);
    if _cfg_debug_mode(mode) {
    for (k, arr) in out_arr.iter_mut().enumerate() {
      println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: out_arr[{}]={:?}", k, arr.get_mut());
    }
    }
    // FIXME: implicit output shape params during gen.
    let np = genabi.param_ct as usize;
    let mut param: Vec<FutAbiScalar> = Vec::with_capacity(np);
    param.resize(np, FutAbiScalar::Unspec);
    //spec_.set_param(&mut param);
    let mut flat_out0_shape = false;
    for pidx in 0 .. np {
      match genabi.get_param(pidx as _) {
        (FutharkParam::Out0Shape, FutAbiScalarType::I64) => {
          match genabi.get_out(0) {
            FutharkArrayRepr::Nd => {
              param[pidx] = FutAbiScalar::I64(out_ty_[0].shape[pidx].into());
            }
            FutharkArrayRepr::Flat => {
              assert!(!flat_out0_shape);
              param[pidx] = FutAbiScalar::I64(out_ty_[0].flat_len().into());
              flat_out0_shape = true;
            }
            _ => unimplemented!()
          }
        }
        _ => unimplemented!()
      }
    }
    /*if np == 0 && self.param.len() == 0 {
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
    }*/
    // FIXME: should distinguish the "spec" abi from the "real" abi.
    /*let restore_out = match mode {
      ThunkMode::Accumulate => {
        let (out, rep, dty) = self.abi.get_out_arr(0);
        assert_eq!(out, FutAbiOutput::Pure);
        self.abi.set_out_arr(0, FutAbiOutput::ImplicitInPlace, rep, dty);
        Some((out, rep, dty))
      }
      _ => None
    };*/
    match mode {
      ThunkMode::Accumulate => {
        TL_CTX.with(|ctx| {
          let mut h = ctx.debugctr.accumulate_hashes.borrow_mut();
          match h.get_mut(&obj.src_hash) {
            None => {
              h.insert(obj.src_hash.clone(), 1);
            }
            Some(ct) => {
              *ct += 1;
            }
          }
        });
      }
      _ => {}
    }
    TL_PCTX.with(|pctx| {
      let gpu = pctx.nvgpu.as_ref().unwrap();
      gpu.compute.sync().unwrap();
      obj.set_stream(gpu.compute.as_ptr() as *mut _);
    });
    let t0 = Stopwatch::tl_stamp();
    obj.reset();
    if _cfg_debug_mode(mode) {
    let tmp_t1 = Stopwatch::tl_stamp();
    println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter:   reset elapsed: {:.09} s", tmp_t1 - t0);
    }
    if _cfg_debug_mode(mode) { println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: enter kernel..."); }
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
      gpu.mem_pool.tmp_freelist.borrow_mut().clear();
      (gpu.mem_pool.front_dptr(), gpu.mem_pool.back_offset())
    });
    /*obj.unify_abi(self.abi).unwrap();*/
    let eabi = genabi.to_eabi(FutAbiSpace::Device);
    let o_ret = obj.enter_kernel(/*&self.abi,*/ eabi, &param, &arg_arr, &out_arr);
    if o_ret.is_err() {
      // FIXME FIXME: error handling.
      println!("ERROR: FutharkThunkImpl::<CudaBackend>::_enter: runtime error");
      if let Some(e) = obj.error().map(|c| safe_ascii(c.to_bytes())) {
        println!("ERROR: FutharkThunkImpl::<CudaBackend>::_enter: runtime error: {}", e);
      }
      panic!();
    }
    let may_fail = obj.may_fail();
    if _cfg_debug_mode(mode) { println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: may fail? {:?}", may_fail); }
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
      if !gpu.mem_pool.tmp_freelist.borrow().is_empty() {
        if _cfg_debug_mode(mode) { println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: free={:?}", &*gpu.mem_pool.tmp_freelist.borrow()); }
      }
      (gpu.mem_pool.front_dptr(), gpu.mem_pool.back_offset())
    });
    let t1 = Stopwatch::tl_stamp();
    if _cfg_debug_mode(mode) { println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter:   elapsed: {:.09} s", t1 - t0); }
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
    if _cfg_debug_mode(mode) { println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: ret={:?}", o_ret); }
    //println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: out={:?} oclk={:?}", out, oclk);
    /*match mode {
      ThunkMode::Accumulate => {
        let (out, rep, dty) = restore_out.unwrap();
        let _ = self.abi.set_out_arr(0, out, rep, dty);
      }
      _ => {}
    }*/
    for (k, arr) in (&mut arg_arr[ .. lar as usize]).iter_mut().enumerate() {
      arr.get_mut()._set_ndim(arg_ndim[k]);
      if _cfg_debug_mode(mode) {
      println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: arg_arr[{}]={:?}",
          k, arr.get_mut());
      }
    }
    for (k, arr) in (&mut arg_arr[lar as usize .. ]).iter_mut().enumerate() {
      /*let _ = arr.get_mut().take_ptr();*/
      arr.get_mut()._set_ndim(arg_ndim[lar as usize + k]);
      if _cfg_debug_mode(mode) {
      println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: arg_arr[{}]={:?} (out in-place)",
          lar as usize + k, arr.get_mut());
      }
    }
    // FIXME: at this point, the remaining memblocks are the outputs.
    // but, if any of the inputs were clobbered, then we have to unset those.
    // so, do some kind of unification here.
    for k in 0 .. genabi.arityout {
      assert!(!out_arr[k as usize].get_mut().as_ptr().is_null());
      match genabi.get_out(k) {
        FutharkArrayRepr::Nd => {
          out_arr[k as usize].get_mut()._set_ndim(max(1, out_ty_[k as usize].ndim()));
        }
        FutharkArrayRepr::Flat => {
          out_arr[k as usize].get_mut()._set_ndim(1);
        }
        _ => unimplemented!()
      }
    }
    //println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: out_arr={:?}", &out_arr);
    if _cfg_debug_mode(mode) {
    for (k, arr) in out_arr.iter_mut().enumerate() {
      println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: out_arr[{}]={:?}", k, arr.get_mut());
    }
    }
    // TODO TODO
    let (out_dptr, out_size) = out_arr[0].get_mut().mem_parts().unwrap();
    match mode {
      ThunkMode::Accumulate => {
        assert!(out_org.is_some());
        let (out_org_addr, out_org_dptr, out_org_size) = out_org.unwrap();
        if out_org_dptr != out_dptr {
          if _cfg_debug_mode(mode) {
          println!("WARNING: FutharkThunkImpl::<CudaBackend>::_enter: out: Accumulate not in-place!");
          println!("WARNING: FutharkThunkImpl::<CudaBackend>::_enter: out:   org dptr=0x{:016x} sz={}",
              out_org_dptr, out_org_size);
          println!("WARNING: FutharkThunkImpl::<CudaBackend>::_enter: out:   org addr={:?}",
              out_org_addr);
          println!("WARNING: FutharkThunkImpl::<CudaBackend>::_enter: out:   new dptr=0x{:016x} sz={}",
              out_dptr, out_size);
          let new_addr = TL_PCTX.with(|pctx| {
            let gpu = pctx.nvgpu.as_ref().unwrap();
            match gpu.mem_pool.rev_lookup(out_dptr) {
              None => None,
              Some((_region, None)) => None,
              Some((_region, Some(p))) => Some(p)
            }
          });
          println!("WARNING: FutharkThunkImpl::<CudaBackend>::_enter: out:   new addr={:?}",
              new_addr);
          }
          TL_CTX.with(|ctx| {
            ctx.debugctr.accumulate_not_in_place.fetch_add(1);
          });
        } else {
          TL_CTX.with(|ctx| {
            ctx.debugctr.accumulate_in_place.fetch_add(1);
          });
        }
        // TODO
      }
      _ => {}
    }
    if _cfg_debug_mode(mode) {
    println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: out: shape={:?}",
        out_arr[0].get_mut().shape().unwrap());
    println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: out: rc={:?} dptr=0x{:016x} sz={}",
        out_arr[0].get_mut().refcount(), out_dptr, out_size);
    if out_dptr == pre_front_dptr {
      println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: out:   no fragmentation");
    } else if out_dptr > pre_front_dptr {
      println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: out:   fragmentation sz={}", out_dptr - pre_front_dptr);
    } else {
      println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: out:   internal allocation; pre_front_dptr=0x{:016x}", pre_front_dptr);
      /*match mode {
        ThunkMode::Accumulate => {}
        _ => {
          unimplemented!();
        }
      }*/
    }
    }
    if out_ty_[0].ndim() == 0 {
      assert_eq!(Some(&[1_i64] as &[_]), out_arr[0].get_mut().shape());
    } else {
      match genabi.get_out(0) {
        FutharkArrayRepr::Nd => {
          assert_eq!(Some(&out_ty_[0].shape as &[_]), out_arr[0].get_mut().shape());
        }
        FutharkArrayRepr::Flat => {
          assert_eq!(Some(&[out_ty_[0].flat_len()] as &[_]), out_arr[0].get_mut().shape());
        }
        _ => unimplemented!()
      }
    }
    match out_arr[0].get_mut().tag() {
      None => {
        if _cfg_debug_mode(mode) { println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: out:   tag=null"); }
      }
      Some(ctag) => {
        if _cfg_debug_mode(mode) { println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: out:   tag=\"{}\"", safe_ascii(ctag.to_bytes())); }
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
      match gpu.mem_pool.rev_lookup(out_dptr) {
        None => panic!("bug"),
        Some((_region, None)) => panic!("bug"),
        Some((region, Some(addr))) => {
          if _cfg_debug_mode(mode) {
          println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: out: region={:?} addr={:?} root={:?}",
              region, addr, pctx.lookup_root(addr));
          }
          let mut f = false;
          if !f {
            //for k in self.consts.borrow().iter() {}
            for k in objects.find(mode).unwrap().1.consts.iter() {
              if k.0 == addr {
                if _cfg_debug_mode(mode) { println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: out:   is const"); }
                match env.lookup_mut_ref(out) {
                  None => panic!("bug"),
                  Some(e) => {
                    match e.cel_ {
                      &mut Cell_::Top(ref state, optr) => {
                        assert_eq!(e.root, optr);
                        // FIXME: defaults below are placeholders for...?
                        let state = RefCell::new(state.borrow().clone());
                        let clo = RefCell::new(CellClosure::default());
                        *e.cel_ = Cell_::Cow(state, clo, CowCell{optr, pcel: *(k.1).as_ref(), pclk: Clock::default()});
                        f = true;
                        if _cfg_debug_mode(mode) { println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: out: cow {:?} -> {:?}", out, addr); }
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
          if !f {
            match env.lookup_mut_ref(out) {
              None => panic!("bug"),
              Some(e) => {
                match e.cel_ {
                  &mut Cell_::Top(ref state, optr) => {
                    if _cfg_debug_mode(mode) { println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: out: try new phy..."); }
                    assert_eq!(e.root, optr);
                    // FIXME: defaults below are placeholders for...?
                    let state = RefCell::new(state.borrow().clone());
                    let clo = RefCell::new(CellClosure::default());
                    let mut pcel = PCell::new(optr, out_ty_[0].clone());
                    pcel.push_new_replica(optr, oclk, Locus::VMem, PMach::NvGpu, addr);
                    *e.cel_ = Cell_::Phy(state, clo, pcel);
                    f = true;
                    if _cfg_debug_mode(mode) {
                    println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: out: new phy {:?} --> {:?} -> {:?}",
                        out, optr, addr);
                    }
                  }
                  &mut Cell_::Phy(ref state, .., ref mut pcel) => {
                    if _cfg_debug_mode(mode) { println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: out: try old phy..."); }
                    assert_eq!(e.root, pcel.optr);
                    let optr = pcel.optr;
                    if let Some(replica) = pcel.lookup(Locus::VMem, PMach::NvGpu) {
                      // NB: clk equal b/c spine did clock_sync.
                      assert_eq!(replica.clk.get(), oclk);
                      let prev_addr = replica.addr.get();
                      if prev_addr != addr {
                        if _cfg_debug_mode(mode) {
                        println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: out:   prev addr={:?}",
                            prev_addr);
                        println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: out:   next addr={:?}",
                            addr);
                        }
                        replica.addr.set(addr);
                        assert!(pctx.set_root(addr, optr).is_none());
                        let prev_root = pctx.unset_root(prev_addr);
                        let prev_root_root =
                        if let Some(prev_root) = prev_root {
                          let prev_root_root = match env.lookup_ref(prev_root) {
                            None => panic!("bug"),
                            Some(e) => e.root
                          };
                          if _cfg_debug_mode(mode) {
                          println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: out:   prev root={:?} --> {:?}",
                              prev_root, prev_root_root);
                          }
                          Some(prev_root_root)
                        } else {
                          None
                        };
                        if _cfg_debug_mode(mode) {
                        println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: out:   next root={:?} --> {:?}",
                            out, optr);
                        }
                        if !(prev_root == Some(optr)) && prev_root_root == Some(optr) {
                          if _cfg_debug_mode(mode) {
                          println!("WARNING: FutharkThunkImpl::<CudaBackend>::_enter: out:   slightly invalid prev root, but mostly harmless");
                          }
                        } else {
                          assert_eq!(prev_root, Some(optr));
                        }
                        // FIXME: gc the prev addr.
                        let _ = gpu.mem_pool.try_free(prev_addr);
                      }
                    } else {
                      pcel.push_new_replica(optr, oclk, Locus::VMem, PMach::NvGpu, addr);
                    }
                    f = true;
                    if _cfg_debug_mode(mode) {
                    println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: out: old phy {:?} --> {:?} -> {:?}",
                        out, optr, addr);
                    }
                  }
                  _ => unimplemented!()
                }
              }
            }
            let p_out = addr;
            if _cfg_debug_mode(mode) {
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
               out_dptr > pre_front_dptr
            {
              // FIXME: could also relocate into a free region.
              let new_offset = gpu.mem_pool.front_cursor.get();
              let new_dptr = gpu.mem_pool.front_base + new_offset as u64;
              assert!(new_dptr <= pre_front_dptr);
              gpu.mem_pool.front_relocate(p_out, new_dptr, &gpu.compute);
              if _cfg_debug_mode(mode) {
              println!("DEBUG: FutharkThunkImpl::<CudaBackend>::_enter: out: relocate src=0x{:016x} dst=0x{:016x}",
                  out_dptr, new_dptr);
              }
            }
            let mut tmp_freelist = gpu.mem_pool.tmp_freelist.borrow_mut();
            tmp_freelist.sort();
            loop {
              let p = match tmp_freelist.pop() {
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
            let _ = arg_arr[lar as usize].get_mut().take_ptr();
            f = true;
          }*/
          if !f {
            panic!("bug");
          }
        }
      }
    });
    Ok(())
  }
}

#[cfg(feature = "nvgpu")]
impl ThunkImpl for FutharkThunkImpl<CudaBackend> {
  fn apply(&self, ctr: &CtxCtr, env: &mut CtxEnv, spec_: &dyn ThunkSpec_, arg: &[(CellPtr, Clock)], th: ThunkPtr, out: CellPtr, oclk: Clock) -> ThunkResult {
    let mode = ThunkMode::Apply;
    self._enter(ctr, env, spec_, arg, th, out, oclk, mode)
  }

  fn accumulate(&self, ctr: &CtxCtr, env: &mut CtxEnv, spec_: &dyn ThunkSpec_, arg: &[(CellPtr, Clock)], th: ThunkPtr, out: CellPtr, oclk: Clock) -> ThunkResult {
    let mode = ThunkMode::Accumulate;
    self._enter(ctr, env, spec_, arg, th, out, oclk, mode)
  }

  fn initialize(&self, ctr: &CtxCtr, env: &mut CtxEnv, spec_: &dyn ThunkSpec_, arg: &[(CellPtr, Clock)], th: ThunkPtr, out: CellPtr, oclk: Clock) -> ThunkResult {
    let mode = ThunkMode::Initialize;
    self._enter(ctr, env, spec_, arg, th, out, oclk, mode)
  }
}

// TODO

pub struct PThunk {
  pub ptr:  ThunkPtr,
  //pub clk:  Clock,
  pub lar:  u16,
  pub rar:  u16,
  pub spec_dim: Vec<Dim>,
  pub spec_:    Rc<dyn ThunkSpec_>,
  pub impl_:    RefCell<RevSortMap8<PMach, Rc<dyn ThunkImpl_>>>,
}

impl PThunk {
  pub fn new(ptr: ThunkPtr, spec_dim: Vec<Dim>, spec_: Rc<dyn ThunkSpec_>) -> PThunk {
    //let clk = Clock::default();
    let (lar, rar) = match spec_.arity() {
      None => unimplemented!(),
      Some(ar) => ar
    };
    assert_eq!(spec_dim.len(), (lar + rar) as usize);
    let impl_ = RefCell::new(RevSortMap8::new());
    PThunk{
      ptr,
      //clk,
      lar,
      rar,
      spec_dim,
      spec_,
      impl_,
    }
  }

  pub fn push_new_impl_(&self, pmach: PMach, thimpl_: Rc<dyn ThunkImpl_>) {
    match self.impl_.borrow().find(pmach) {
      None => {}
      Some(_) => panic!("bug")
    }
    self.impl_.borrow_mut().insert(pmach, thimpl_);
  }

  pub fn lookup_impl_(&self, q_pmach: PMach) -> Option<Rc<dyn ThunkImpl_>> {
    match self.impl_.borrow().find(q_pmach) {
      None => None,
      Some((_, thimpl_)) => Some(thimpl_.clone())
    }
  }

  pub fn apply(&self, ctr: &CtxCtr, env: &mut CtxEnv, arg: &[(CellPtr, Clock)], th: ThunkPtr, out: CellPtr, oclk: Clock, pmach: PMach) -> ThunkResult {
    //let primary = TL_CTX.with(|ctx| ctx.primary.get());
    //let pmach = primary.unwrap_or_else(|| TL_PCTX.with(|pctx| pctx.fastest_pmach()));
    //println!("DEBUG: PThunk::apply: pmach={:?} primary={:?}", pmach, primary);
    match self.lookup_impl_(pmach) {
      None => {
        match self.spec_.gen_impl_(self.spec_dim.clone(), pmach) {
          None => {
            // FIXME: fail stop here.
          }
          Some(thimpl_) => {
            self.push_new_impl_(pmach, thimpl_);
          }
        }
      }
      _ => {}
    }
    match self.lookup_impl_(pmach) {
      None => panic!("bug"),
      Some(thimpl_) => {
        thimpl_.apply(ctr, env, &*self.spec_, arg, th, out, oclk)
      }
    }
  }

  pub fn accumulate(&self, ctr: &CtxCtr, env: &mut CtxEnv, arg: &[(CellPtr, Clock)], th: ThunkPtr, out: CellPtr, oclk: Clock, pmach: PMach) -> ThunkResult {
    //let primary = TL_CTX.with(|ctx| ctx.primary.get());
    //let pmach = primary.unwrap_or_else(|| TL_PCTX.with(|pctx| pctx.fastest_pmach()));
    //println!("DEBUG: PThunk::accumulate: pmach={:?} primary={:?}", pmach, primary);
    match self.lookup_impl_(pmach) {
      None => {
        match self.spec_.gen_impl_(self.spec_dim.clone(), pmach) {
          None => {
            // FIXME: fail stop here.
          }
          Some(thimpl_) => {
            self.push_new_impl_(pmach, thimpl_);
          }
        }
      }
      _ => {}
    }
    match self.lookup_impl_(pmach) {
      None => panic!("bug"),
      Some(thimpl_) => {
        thimpl_.accumulate(ctr, env, &*self.spec_, arg, th, out, oclk)
      }
    }
  }

  pub fn initialize(&self, ctr: &CtxCtr, env: &mut CtxEnv, arg: &[(CellPtr, Clock)], th: ThunkPtr, out: CellPtr, oclk: Clock, pmach: PMach) -> ThunkResult {
    //let primary = TL_CTX.with(|ctx| ctx.primary.get());
    //let pmach = primary.unwrap_or_else(|| TL_PCTX.with(|pctx| pctx.fastest_pmach()));
    //println!("DEBUG: PThunk::initialize: pmach={:?} primary={:?}", pmach, primary);
    match self.lookup_impl_(pmach) {
      None => {
        match self.spec_.gen_impl_(self.spec_dim.clone(), pmach) {
          None => {
            // FIXME: fail stop here.
          }
          Some(thimpl_) => {
            self.push_new_impl_(pmach, thimpl_);
          }
        }
      }
      _ => {}
    }
    match self.lookup_impl_(pmach) {
      None => panic!("bug"),
      Some(thimpl_) => {
        thimpl_.initialize(ctr, env, &*self.spec_, arg, th, out, oclk)
      }
    }
  }
}
