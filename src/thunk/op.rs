use super::*;
use crate::algo::fp::{TotalOrd};
use crate::cell::{DtypeExt, Dim, ScalarVal_};
use crate::op::*;
use cacti_cfg_env::*;
use cacti_gpu_cu_ffi::{cuda_memcpy_async, cublas_gemm, cublas_gemm_batched};
use cacti_gpu_cu_ffi::types::{CUDA_R_32F, CUDA_R_16F, CUDA_R_16BF};

//use futhark_syntax::{Exp};

use std::borrow::{Cow};
use std::cell::{Cell};
use std::ffi::{c_void};
//use std::io::{Write};
use std::rc::{Weak};
use std::slice::{from_raw_parts, from_raw_parts_mut};

/*#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct LamFutExpThunkSpec {
  pub lam_src: Cow<'static, str>,
  pub lam_exp: Exp,
  pub ar_in: u16,
  pub ar_out: u16,
  // FIXME FIXME
  //pub arg_dim: Vec<Dim>,
  //pub out_dim: RefCell<Option<Vec<Dim>>>,
  pub wrap_parens: bool,
}

impl FutharkThunkSpec for LamFutExpThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.lam")
  }

  fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = self.ar_in;
    abi.arityout = self.ar_out;
    abi
  }

  fn out_dim(&self, _arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    // FIXME FIXME
    //Ok(Dim{ndim: 0, dtype: T::dtype_()})
    unimplemented!();
  }

  fn out_ty_(&self, _arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    // FIXME FIXME
    //Ok(CellType{shape: Vec::new(), dtype: T::dtype_()})
    unimplemented!();
  }

  fn gen_futhark(&self, abi: &mut FutAbi, _arg: &[Dim], _out: &[Dim]) -> Result<FutharkThunkGenCode, FutharkGenErr> {
    let mut s = String::new();
    if self.wrap_parens {
      write!(&mut s, "(").unwrap();
    }
    s.push_str(&self.lam_src);
    if self.wrap_parens {
      write!(&mut s, ")").unwrap();
    }
    for k in 0 .. self.ar_in {
      write!(&mut s, " {{%{}}}", k).unwrap();
    }
    let mut code = FutharkThunkGenCode::default();
    code.body.push(s);
    code.into()
  }
}*/

#[derive(PartialEq, Eq, Hash)]
pub struct FutharkCodeThunkSpec {
  // TODO
  pub primal_mode: ThunkMode,
  pub cost: Option<ThunkCostR0>,
  pub lar:  u16,
  pub rar:  u16,
  //pub abi: FutAbi,
  //pub param: _,
  pub dim:  Vec<Dim>,
  pub ty_:  Vec<CellType>,
  pub code: FutharkThunkGenCode,
}

impl FutharkThunkSpec for FutharkCodeThunkSpec {
  fn cost_r0(&self) -> Option<ThunkCostR0> {
    self.cost
  }

  fn arity(&self) -> Option<(u16, u16)> {
    Some((self.lar, self.rar))
  }

  /*fn abi(&self) -> Abi {
    //self.abi.clone()
    unimplemented!();
  }*/

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    // TODO
    if arg.len() != self.lar as usize {
      return Err(ThunkDimErr::_Bot);
    }
    if arg != &self.dim[ .. self.lar as usize] {
      return Err(ThunkDimErr::_Bot);
    }
    assert_eq!(1, self.rar);
    Ok(self.dim[self.lar as usize])
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    // TODO
    if arg.len() != self.lar as usize {
      return Err(ThunkTypeErr::_Bot);
    }
    if arg != &self.ty_[ .. self.lar as usize] {
      return Err(ThunkTypeErr::_Bot);
    }
    assert_eq!(1, self.rar);
    Ok(self.ty_[self.lar as usize].clone())
  }

  fn gen_futhark(&self, /*abi: &mut FutAbi,*/ _arg: &[Dim], _out: &[Dim]) -> Result<FutharkThunkGenCode, FutharkGenErr> {
    // TODO
    /*assert_eq!(1, abi.arityout);
    abi.set_out_arr(0, AbiOutput::Pure, AbiArrayRepr::Nd, AbiScalarType::Unspec);
    for k in 0 .. abi.arityin {
      abi.set_arg_arr(k, AbiInput::Shared, AbiArrayRepr::Nd, AbiScalarType::Unspec);
    }*/
    Ok(self.code.clone())
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct IdentityFutThunkSpec;

impl FutharkThunkSpec for IdentityFutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.identity")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn arity(&self) -> Option<(u16, u16)> {
    Some((1, 1))
  }

  /*fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }*/

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(arg[0])
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(arg[0].clone())
  }

  fn gen_futhark(&self, /*abi: &mut FutAbi,*/ arg: &[Dim], _out: &[Dim]) -> Result<FutharkThunkGenCode, FutharkGenErr> {
    /*abi.set_out_arr(0, AbiOutput::Pure, AbiArrayRepr::Nd, AbiScalarType::Unspec);
    abi.set_arg_arr(0, AbiInput::Shared, AbiArrayRepr::Nd, AbiScalarType::Unspec);
    let mut code = FutharkThunkGenCode::default();
    code.append(r"let {%1} = {%0} in");
    code.into()*/
    FutharkThunkGenCode::flat_map(arg[0], r"\u -> u")
  }

  fn pop_adj(&self, _arg: &[(CellPtr, Clock)], _out: CellPtr, _out_clk: Clock, out_adj: CellPtr, arg_adj: &mut [CellPtr]) -> Result<FutharkThunkAdj, ThunkAdjErr> {
    // FIXME: snapshot.
    arg_adj[0] += out_adj._memcpy();
    Ok(FutharkThunkAdj::Spec)
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct IotaFutThunkSpec { pub len: i64 }

impl FutharkThunkSpec for IotaFutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.iota")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn arity(&self) -> Option<(u16, u16)> {
    Some((0, 1))
  }

  /*fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 0;
    abi.arityout = 1;
    abi
  }*/

  fn out_dim(&self, _arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(Dim{ndim: 1, dtype: Dtype::I64})
  }

  fn out_ty_(&self, _arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(CellType{shape: vec![self.len], dtype: Dtype::I64})
  }

  fn gen_futhark(&self, /*abi: &mut FutAbi,*/ _arg: &[Dim], _out: &[Dim]) -> Result<FutharkThunkGenCode, FutharkGenErr> {
    // FIXME: use param.
    //abi.set_out_arr(0, AbiOutput::Pure, AbiArrayRepr::Nd, AbiScalarType::Unspec);
    let mut code = FutharkThunkGenCode::default();
    code.abi.arityout = 1;
    code.abi.set_out(0, FutharkArrayRepr::Nd);
    code.abi.arityin = 0;
    code.append(format!(r"let {{%0}} = iota {} in", self.len));
    code.into()
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SetScalarFutThunkSpec { pub val: ScalarVal_ }

impl FutharkThunkSpec for SetScalarFutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.set_scalar")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn arity(&self) -> Option<(u16, u16)> {
    Some((0, 1))
  }

  /*fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 0;
    abi.arityout = 1;
    abi
  }*/

  fn out_dim(&self, _arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Err(ThunkDimErr::Nondeterm)
  }

  fn out_ty_(&self, _arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Err(ThunkTypeErr::Nondeterm)
  }

  /*fn scalar_val(&self) -> Option<&dyn DtypeExt> {
    Some(&self.val)
  }*/

  fn gen_futhark(&self, /*abi: &mut FutAbi,*/ _arg: &[Dim], out: &[Dim]) -> Result<FutharkThunkGenCode, FutharkGenErr> {
    /*// FIXME FIXME: rank polymorphic.
    //let fmt = FutharkNumFormatter::default();
    let mut code = FutharkThunkGenCode::default();
    // FIXME FIXME: futhark treats actual scalars as simply pointers to cpu mem.
    /*body:     vec![format!("let {{%0}} = [{}] in", fmt.format(&self.val))],*/
    //code.body.push(format!("let {{%0}} = {} in", fmt.format(&self.val)));
    code.append(format!(r"let {{%0}} = {} in", self.val.format_futhark()));
    code.into()*/
    FutharkThunkGenCode::flat_replicate(out[0], format!(r"{}", self.val.format_futhark()))
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct NanCountFutThunkSpec;

impl FutharkThunkSpec for NanCountFutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.nan_count")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn arity(&self) -> Option<(u16, u16)> {
    Some((1, 1))
  }

  /*fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }*/

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(Dim{ndim: 0, dtype: Dtype::I64})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(CellType{shape: vec![], dtype: Dtype::I64})
  }

  fn gen_futhark(&self, /*abi: &mut FutAbi,*/ arg: &[Dim], out: &[Dim]) -> Result<FutharkThunkGenCode, FutharkGenErr> {
    //abi.set_out_arr(0, AbiOutput::Pure, AbiArrayRepr::Nd, AbiScalarType::Unspec);
    //abi.set_arg_arr(0, AbiInput::Shared, AbiArrayRepr::Nd, AbiScalarType::Unspec);
    let mut code = FutharkThunkGenCode::default();
    code.abi.arityout = 1;
    code.abi.set_out(0, FutharkArrayRepr::Nd);
    code.abi.arityin = 1;
    code.abi.set_arg(0, FutharkArrayRepr::Nd);
    match arg[0].dtype {
      //Dtype::F64 |
      //Dtype::F32 |
      Dtype::F16 => {
        match arg[0].ndim() {
          0 => {
            code.append(format!(r"let t0 = [{{%0}}] in"));
          }
          1 => {
            code.append(format!(r"let t0 = {{%0}} in"));
          }
          2 => {
            code.append(format!(r"let t0 = flatten {{%0}} in"));
          }
          3 => {
            code.append(format!(r"let t0 = flatten_3d {{%0}} in"));
          }
          4 => {
            code.append(format!(r"let t0 = flatten_4d {{%0}} in"));
          }
          _ => unimplemented!()
        }
        /*code.append(format!(r"let {{%1}} = reduce (+) 0 (map (\u -> if {}.isnan u then 1 else 0) t0)",
            arg[0].dtype.format_futhark(),
        ));*/
        code.append(format!(r"let n = length t0 in"));
        code.append(format!(r"let {{%1}} = reduce (+) 0 (map (\u ->"));
        code.append(format!(r"let u = {}.to_bits u in",
            arg[0].dtype.format_futhark(),
        ));
        code.append(format!(r"let b = (u >> 10) & 0x1f in"));
        code.append(format!(r"let m = u & 0x3ff in"));
        code.append(format!(r"if (b == 0x1f && m != 0) then 1 else 0) (t0 :> [n]{})) in",
            arg[0].dtype.format_futhark(),
        ));
      }
      _ => unimplemented!()
    }
    code.into()
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct AbsLog2Hist8FutThunkSpec;

impl FutharkThunkSpec for AbsLog2Hist8FutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.abs_log2_hist8")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn arity(&self) -> Option<(u16, u16)> {
    Some((1, 1))
  }

  /*fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }*/

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(Dim{ndim: 1, dtype: Dtype::I64})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(CellType{shape: vec![0x100], dtype: Dtype::I64})
  }

  fn gen_futhark(&self, /*abi: &mut FutAbi,*/ arg: &[Dim], out: &[Dim]) -> Result<FutharkThunkGenCode, FutharkGenErr> {
    //abi.set_out_arr(0, AbiOutput::Pure, AbiArrayRepr::Nd, AbiScalarType::Unspec);
    //abi.set_arg_arr(0, AbiInput::Shared, AbiArrayRepr::Nd, AbiScalarType::Unspec);
    let mut code = FutharkThunkGenCode::default();
    code.abi.arityout = 1;
    code.abi.set_out(0, FutharkArrayRepr::Nd);
    code.abi.arityin = 1;
    code.abi.set_arg(0, FutharkArrayRepr::Nd);
    match arg[0].dtype {
      Dtype::F16 => {
        code.pre_append(format!(r"def u16_nz_log2 (x: u16): i8 ="));
        code.pre_append(format!(r"{}let v_tab = [0, 7, 1, 13, 8, 10, 2, 14, 6, 12, 9, 5, 11, 4, 3, 15] in", "\t", ));
        code.pre_append(format!(r"{}let c = 0xf2d_u16 in", "\t", ));
        code.pre_append(format!(r"{}let x = x | (x >> 1) in", "\t", ));
        code.pre_append(format!(r"{}let x = x | (x >> 2) in", "\t", ));
        code.pre_append(format!(r"{}let x = x | (x >> 4) in", "\t", ));
        code.pre_append(format!(r"{}let x = x | (x >> 8) in", "\t", ));
        code.pre_append(format!(r"{}v_tab[i64.u16 ((c * x) >> 12)]", "\t", ));
        match arg[0].ndim() {
          0 => {
            code.append(format!(r"let t0 = [{{%0}}] in"));
          }
          1 => {
            code.append(format!(r"let t0 = {{%0}} in"));
          }
          2 => {
            code.append(format!(r"let t0 = flatten {{%0}} in"));
          }
          3 => {
            code.append(format!(r"let t0 = flatten_3d {{%0}} in"));
          }
          4 => {
            code.append(format!(r"let t0 = flatten_4d {{%0}} in"));
          }
          _ => unimplemented!()
        }
        code.append(format!(r"let n = length t0 in"));
        code.append(format!(r"let t1 = map (\u ->"));
        code.append(format!(r"let u = f16.to_bits u in"));
        code.append(format!(r"let b = (u >> 10) & 0x1f in"));
        code.append(format!(r"let e = (i8.u16 b) - 0xf in"));
        code.append(format!(r"let m = u & 0x3ff in"));
        code.append(format!(r"let v ="));
        code.append(format!(r"{}if b == 0x1f then -0x80", "\t", ));
        code.append(format!(r"{}else if b == 0 then", "\t", ));
        code.append(format!(r"{}if m == 0 then -0x7f", "\t\t", ));
        code.append(format!(r"{}else (-14 - 10 + (u16_nz_log2 m))", "\t\t", ));
        code.append(format!(r"{}else e", "\t", ));
        code.append(format!(r"in i64.u8 (u8.i8 (v + 0x7f))"));
        code.append(format!(r") (t0 :> [n]{}) in", arg[0].dtype.format_futhark()));
        code.append(format!(r"let {{%1}} = hist (+) 0 0x100 t1 (replicate n 1) in"));
      }
      _ => unimplemented!()
    }
    code.into()
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct AbsLog2Hist16FutThunkSpec;

impl FutharkThunkSpec for AbsLog2Hist16FutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.abs_log2_hist16")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn arity(&self) -> Option<(u16, u16)> {
    Some((1, 1))
  }

  /*fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }*/

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(Dim{ndim: 1, dtype: Dtype::I64})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(CellType{shape: vec![0x10000], dtype: Dtype::I64})
  }

  fn gen_futhark(&self, /*abi: &mut FutAbi,*/ arg: &[Dim], out: &[Dim]) -> Result<FutharkThunkGenCode, FutharkGenErr> {
    //abi.set_out_arr(0, AbiOutput::Pure, AbiArrayRepr::Nd, AbiScalarType::Unspec);
    //abi.set_arg_arr(0, AbiInput::Shared, AbiArrayRepr::Nd, AbiScalarType::Unspec);
    let mut code = FutharkThunkGenCode::default();
    code.abi.arityout = 1;
    code.abi.set_out(0, FutharkArrayRepr::Nd);
    code.abi.arityin = 1;
    code.abi.set_arg(0, FutharkArrayRepr::Nd);
    match arg[0].dtype {
      Dtype::F16 => {
        code.pre_append(format!(r"def u16_nz_log2 (x: u16): i16 ="));
        code.pre_append(format!(r"{}let v_tab = [0, 7, 1, 13, 8, 10, 2, 14, 6, 12, 9, 5, 11, 4, 3, 15] in", "\t", ));
        code.pre_append(format!(r"{}let c = 0xf2d_u16 in", "\t", ));
        code.pre_append(format!(r"{}let x = x | (x >> 1) in", "\t", ));
        code.pre_append(format!(r"{}let x = x | (x >> 2) in", "\t", ));
        code.pre_append(format!(r"{}let x = x | (x >> 4) in", "\t", ));
        code.pre_append(format!(r"{}let x = x | (x >> 8) in", "\t", ));
        code.pre_append(format!(r"{}v_tab[i64.u16 ((c * x) >> 12)]", "\t", ));
        match arg[0].ndim() {
          0 => {
            code.append(format!(r"let t0 = [{{%0}}] in"));
          }
          1 => {
            code.append(format!(r"let t0 = {{%0}} in"));
          }
          2 => {
            code.append(format!(r"let t0 = flatten {{%0}} in"));
          }
          3 => {
            code.append(format!(r"let t0 = flatten_3d {{%0}} in"));
          }
          4 => {
            code.append(format!(r"let t0 = flatten_4d {{%0}} in"));
          }
          _ => unimplemented!()
        }
        code.append(format!(r"let n = length t0 in"));
        code.append(format!(r"let t1 = map (\u ->"));
        code.append(format!(r"let u = f16.to_bits u in"));
        code.append(format!(r"let b = (u >> 10) & 0x1f in"));
        code.append(format!(r"let e = (i16.u16 b) - 0xf in"));
        code.append(format!(r"let m = u & 0x3ff in"));
        code.append(format!(r"let v ="));
        code.append(format!(r"{}if b == 0x1f then -0x8000", "\t", ));
        code.append(format!(r"{}else if b == 0 then", "\t", ));
        code.append(format!(r"{}if m == 0 then -0x7fff", "\t\t", ));
        code.append(format!(r"{}else (-14 - 10 + (u16_nz_log2 m))", "\t\t", ));
        code.append(format!(r"{}else e", "\t", ));
        code.append(format!(r"in i64.u16 (u16.i16 (v + 0x7fff))"));
        code.append(format!(r") (t0 :> [n]{}) in", arg[0].dtype.format_futhark()));
        code.append(format!(r"let {{%1}} = hist (+) 0 0x10000 t1 (replicate n 1) in"));
      }
      _ => unimplemented!()
    }
    code.into()
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct CastFutThunkSpec { pub new_dtype: Dtype }

impl FutharkThunkSpec for CastFutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.cast")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn arity(&self) -> Option<(u16, u16)> {
    Some((1, 1))
  }

  /*fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }*/

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(Dim{ndim: arg[0].ndim, dtype: self.new_dtype})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(CellType{shape: arg[0].shape.clone(), dtype: self.new_dtype})
  }

  fn gen_futhark(&self, /*abi: &mut FutAbi,*/ arg: &[Dim], out: &[Dim]) -> Result<FutharkThunkGenCode, FutharkGenErr> {
    assert_eq!(out[0].dtype, self.new_dtype);
    FutharkThunkGenCode::flat_map(arg[0], format!(r"\u -> {}.{} u",
                                                  self.new_dtype.format_futhark(),
                                                  arg[0].dtype.format_futhark()))
  }

  fn pop_adj(&self, arg: &[(CellPtr, Clock)], _out: CellPtr, _out_clk: Clock, out_adj: CellPtr, arg_adj: &mut [CellPtr]) -> Result<FutharkThunkAdj, ThunkAdjErr> {
    let x_ty = arg[0].0.type_();
    arg_adj[0] += out_adj.cast(x_ty.dtype);
    Ok(FutharkThunkAdj::Spec)
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct CastBf16F16FutThunkSpec;

impl FutharkThunkSpec for CastBf16F16FutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.f16.bf16_cast")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn arity(&self) -> Option<(u16, u16)> {
    Some((1, 1))
  }

  /*fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }*/

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    if arg[0].dtype != Dtype::Bf16 {
      return Err(ThunkDimErr::_Bot);
    }
    Ok(Dim{ndim: arg[0].ndim, dtype: Dtype::F16})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    if arg[0].dtype != Dtype::Bf16 {
      return Err(ThunkTypeErr::_Bot);
    }
    Ok(CellType{shape: arg[0].shape.clone(), dtype: Dtype::F16})
  }

  fn gen_futhark(&self, /*abi: &mut FutAbi,*/ arg: &[Dim], _out: &[Dim]) -> Result<FutharkThunkGenCode, FutharkGenErr> {
    FutharkThunkGenCode::flat_map(arg[0], r"\u -> f16.f32 (f32.from_bits ((u32.u16 u) << 16))")
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct CastBf16F32FutThunkSpec;

impl FutharkThunkSpec for CastBf16F32FutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.f32.bf16_cast")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn arity(&self) -> Option<(u16, u16)> {
    Some((1, 1))
  }

  /*fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }*/

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    if arg[0].dtype != Dtype::Bf16 {
      return Err(ThunkDimErr::_Bot);
    }
    Ok(Dim{ndim: arg[0].ndim, dtype: Dtype::F32})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    if arg[0].dtype != Dtype::Bf16 {
      return Err(ThunkTypeErr::_Bot);
    }
    Ok(CellType{shape: arg[0].shape.clone(), dtype: Dtype::F32})
  }

  fn gen_futhark(&self, /*abi: &mut FutAbi,*/ arg: &[Dim], _out: &[Dim]) -> Result<FutharkThunkGenCode, FutharkGenErr> {
    FutharkThunkGenCode::flat_map(arg[0], r"\u -> f32.from_bits ((u32.u16 u) << 16)")
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct CastF32Bf16FutThunkSpec;

impl FutharkThunkSpec for CastF32Bf16FutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.bf16.f32_cast")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn arity(&self) -> Option<(u16, u16)> {
    Some((1, 1))
  }

  /*fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }*/

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    if arg[0].dtype != Dtype::F32 {
      return Err(ThunkDimErr::_Bot);
    }
    Ok(Dim{ndim: arg[0].ndim, dtype: Dtype::Bf16})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    if arg[0].dtype != Dtype::F32 {
      return Err(ThunkTypeErr::_Bot);
    }
    Ok(CellType{shape: arg[0].shape.clone(), dtype: Dtype::Bf16})
  }

  fn gen_futhark(&self, /*abi: &mut FutAbi,*/ arg: &[Dim], _out: &[Dim]) -> Result<FutharkThunkGenCode, FutharkGenErr> {
    FutharkThunkGenCode::flat_map(arg[0], r"\u -> u16.u32 ((f32.to_bits u) >> 16)")
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct InnerArgMaxFutThunkSpec;

impl FutharkThunkSpec for InnerArgMaxFutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.inner_arg_max")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn arity(&self) -> Option<(u16, u16)> {
    Some((1, 1))
  }

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    if arg[0].ndim() < 1 {
      return Err(ThunkDimErr::_Bot);
    }
    Ok(Dim{ndim: arg[0].ndim() - 1, dtype: Dtype::I64})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    if arg[0].ndim() < 1 {
      return Err(ThunkTypeErr::_Bot);
    }
    let mut shape = arg[0].shape.clone();
    shape.pop();
    Ok(CellType{shape, dtype: Dtype::I64})
  }

  fn gen_futhark(&self, arg: &[Dim], _out: &[Dim]) -> Result<FutharkThunkGenCode, FutharkGenErr> {
    let mut code = FutharkThunkGenCode::default();
    code.abi.arityout = 1;
    code.abi.set_out(0, FutharkArrayRepr::Nd);
    code.abi.arityin = 1;
    code.abi.set_arg(0, FutharkArrayRepr::Nd);
    match arg[0].ndim {
      2 => {
        code.cfg.emit_arg_shapes = true;
        code.body.push(format!(r"let a_pre = {{%0.s[0]}} in"));
        code.body.push(format!(r"let a_suf = {{%0.s[1]}} in"));
        code.body.push(format!(r"let t_iota = iota a_suf in"));
        code.body.push(format!(r"let t0 = unflatten (flatten {{%0}} :> [a_pre * a_suf]{}) in",
            arg[0].dtype.format_futhark(),
        ));
        code.body.push(format!(r"let t1 = map (\t -> let umax = reduce ({}.max) (-{}.inf) t in let tkey = map2 (\idx u -> if u == umax then idx else a_suf) t_iota t in reduce (i64.min) (i64.highest) tkey) t0 in",
            arg[0].dtype.format_futhark(),
            arg[0].dtype.format_futhark(),
        ));
        code.body.push(format!(r"let {{%1}} = t1 in"));
      }
      3 => {
        code.cfg.emit_arg_shapes = true;
        code.body.push(format!(r"let a_pre = {{%0.s[0]}} * {{%0.s[1]}} in"));
        code.body.push(format!(r"let a_suf = {{%0.s[2]}} in"));
        code.body.push(format!(r"let t_iota = iota a_suf in"));
        code.body.push(format!(r"let t0 = unflatten (flatten_3d {{%0}} :> [a_pre * a_suf]{}) in",
            arg[0].dtype.format_futhark(),
        ));
        code.body.push(format!(r"let t1 = map (\t -> let umax = reduce ({}.max) (-{}.inf) t in let tkey = map2 (\idx u -> if u == umax then idx else a_suf) t_iota t in reduce (i64.min) (i64.highest) tkey) t0 in",
            arg[0].dtype.format_futhark(),
            arg[0].dtype.format_futhark(),
        ));
        code.body.push(format!(r"let {{%1}} = unflatten (t1 :> [{{%0.s[0]}} * {{%0.s[1]}}]i64) in"));
      }
      4 => {
        unimplemented!();
      }
      _ => unimplemented!()
    }
    code.into()
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct InnerOneHotFutThunkSpec { pub inner_len: i64, /*pub org_dtype: Dtype,*/ pub new_dtype: Dtype }

impl FutharkThunkSpec for InnerOneHotFutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.inner_one_hot")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn arity(&self) -> Option<(u16, u16)> {
    Some((1, 1))
  }

  /*fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }*/

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    /*if arg[0].dtype != self.org_dtype {
      return Err(ThunkDimErr::_Bot);
    }*/
    if !arg[0].dtype.is_uint() {
      return Err(ThunkDimErr::_Bot);
    }
    Ok(Dim{ndim: arg[0].ndim + 1, dtype: self.new_dtype})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    /*if arg[0].dtype != self.org_dtype {
      return Err(ThunkTypeErr::_Bot);
    }*/
    if !arg[0].dtype.is_uint() {
      return Err(ThunkTypeErr::_Bot);
    }
    let mut shape = arg[0].shape.clone();
    shape.push(self.inner_len);
    Ok(CellType{shape, dtype: self.new_dtype})
  }

  fn gen_futhark(&self, /*abi: &mut FutAbi,*/ arg: &[Dim], _out: &[Dim]) -> Result<FutharkThunkGenCode, FutharkGenErr> {
    //abi.set_out_arr(0, AbiOutput::Pure, AbiArrayRepr::Nd, AbiScalarType::Unspec);
    //abi.set_arg_arr(0, AbiInput::Shared, AbiArrayRepr::Nd, AbiScalarType::Unspec);
    let out = FutharkThunkSpec::out_dim(self, arg).map_err(|e| e.into_gen())?;
    let mut code = FutharkThunkGenCode::default();
    code.abi.arityout = 1;
    code.abi.set_out(0, FutharkArrayRepr::Nd);
    code.abi.arityin = 1;
    code.abi.set_arg(0, FutharkArrayRepr::Nd);
    match out.ndim {
      0 => {
        unimplemented!();
      }
      1 => {
        unimplemented!();
      }
      2 => {
        code.cfg.emit_arg_shapes = true;
        code.body.push(format!(r"let t_oidx = {{%0}} in"));
        code.body.push(format!(r"let t_iota = indices t_oidx in"));
        code.body.push(format!(r"let t_key = map (\idx k -> let k = (i64.{} k) in (assert (k >= 0 && k < {}) k) + {} * idx) t_iota t_oidx in",
            arg[0].dtype.format_futhark(),
            self.inner_len,
            self.inner_len,
        ));
        code.body.push(format!(r"let t_val = replicate {{%0.s[0]}} 1.0{} in",
            //fmt.format(&TotalOrd::from(1.0_f32)),
            out.dtype.format_futhark(),
        ));
        code.body.push(format!(r"let t1 = spread ({{%0.s[0]}} * {}) 0.0{} t_key t_val in",
            self.inner_len,
            out.dtype.format_futhark(),
        ));
        code.body.push(format!(r"let {{%1}} = unflatten (t1 :> [{{%0.s[0]}} * {}]{}) in",
            self.inner_len,
            out.dtype.format_futhark(),
        ));
      }
      3 => {
        code.cfg.emit_arg_shapes = true;
        code.body.push(format!(r"let a = {{%0.s[0]}} * {{%0.s[1]}} in"));
        code.body.push(format!(r"let t_oidx = flatten {{%0}} :> [a]{} in",
            arg[0].dtype.format_futhark(),
        ));
        code.body.push(format!(r"let t_iota = indices t_oidx in"));
        code.body.push(format!(r"let t_key = map2 (\idx k -> let k = (i64.{} k) in (assert (k >= 0 && k < {}) k) + {} * idx) t_iota t_oidx in",
            arg[0].dtype.format_futhark(),
            self.inner_len,
            self.inner_len,
        ));
        code.body.push(format!(r"let t_val = replicate a 1.0{} in",
            out.dtype.format_futhark(),
        ));
        code.body.push(format!(r"let t1 = spread (a * {}) 0.0{} t_key t_val in",
            self.inner_len,
            out.dtype.format_futhark(),
        ));
        code.body.push(format!(r"let {{%1}} = unflatten_3d (t1 :> [{{%0.s[0]}} * {{%0.s[1]}} * {}]{}) in",
            self.inner_len,
            out.dtype.format_futhark(),
        ));
      }
      4 => {
        unimplemented!();
      }
      _ => unimplemented!()
    }
    code.into()
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct InnerSelectFutThunkSpec;

impl FutharkThunkSpec for InnerSelectFutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.inner_select")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn arity(&self) -> Option<(u16, u16)> {
    Some((2, 1))
  }

  /*fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 2;
    abi.arityout = 1;
    abi
  }*/

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    // FIXME
    if !arg[1].dtype.is_uint() {
      return Err(ThunkDimErr::_Bot);
    }
    if arg[0].ndim() != arg[1].ndim() + 1 {
      return Err(ThunkDimErr::_Bot);
    }
    Ok(Dim{ndim: arg[1].ndim(), dtype: arg[0].dtype})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    // FIXME
    if !arg[1].dtype.is_uint() {
      return Err(ThunkTypeErr::_Bot);
    }
    if arg[0].ndim() != arg[1].ndim() + 1 {
      return Err(ThunkTypeErr::_Bot);
    }
    let shape = arg[1].shape.clone();
    if &arg[0].shape[ .. shape.len()] != shape {
      return Err(ThunkTypeErr::_Bot);
    }
    Ok(CellType{shape, dtype: arg[0].dtype})
  }

  fn gen_futhark(&self, /*abi: &mut FutAbi,*/ arg: &[Dim], out: &[Dim]) -> Result<FutharkThunkGenCode, FutharkGenErr> {
    //abi.set_out_arr(0, AbiOutput::Pure, AbiArrayRepr::Nd, AbiScalarType::Unspec);
    //abi.set_arg_arr(0, AbiInput::Shared, AbiArrayRepr::Nd, AbiScalarType::Unspec);
    //abi.set_arg_arr(1, AbiInput::Shared, AbiArrayRepr::Nd, AbiScalarType::Unspec);
    //let out = FutharkThunkSpec::out_dim(self, arg).map_err(|e| e.into_gen())?;
    let mut code = FutharkThunkGenCode::default();
    code.abi.arityout = 1;
    code.abi.set_out(0, FutharkArrayRepr::Nd);
    code.abi.arityin = 2;
    code.abi.set_arg(0, FutharkArrayRepr::Nd);
    code.abi.set_arg(1, FutharkArrayRepr::Nd);
    match out[0].ndim {
      0 => {
        unimplemented!();
      }
      1 => {
        unimplemented!();
      }
      2 => {
        code.cfg.emit_arg_shapes = true;
        code.append(format!(r"let a_pre = {{%0.s[0]}} * {{%0.s[1]}} in"));
        code.append(format!(r"let a_suf = {{%0.s[2]}} in"));
        //code.append(format!(r"let a = a_pre * a_suf in"));
        //code.append(format!(r"let t_val = flatten_3d {{%0}} :> [a]{} in", arg[0].dtype.format_futhark()));
        code.append(format!(r"let t_val = flatten_3d {{%0}} :> [a_pre * a_suf]{} in",
            arg[0].dtype.format_futhark(),
        ));
        //code.append(format!(r"let t_val = unflatten a_pre a_suf t_val in"));
        code.append(format!(r"let t_val = unflatten t_val in"));
        /*code.append(format!(r"let t_key = flatten {{%1}} :> [a_pre]{} in", arg[1].dtype.format_futhark()));*/
        code.append(format!(r"let t1 = flatten {{%1}} :> [a_pre]{} in",
            arg[1].dtype.format_futhark(),
        ));
        // FIXME FIXME
        /*code.append(format!(r"let t_iota = iota a_pre in"));
        code.append(format!(r"let t_key = map2 (\idx k -> let k = (i64.{} k) in (assert (k >= 0 && k < a_suf) k) + a_suf * idx) t_iota t1 in",
            arg[1].dtype.format_futhark(),
        ));*/
        code.append(format!(r"let t_key = map2 (\k -> let k = (i64.{} k) in (assert (k >= 0 && k < a_suf) k)) t1 in",
            arg[1].dtype.format_futhark(),
        ));
        code.append(format!(r"let t2 = map2 (\k v -> v[k]) t_key t_val in"));
        //code.append(format!(r"let {{%2}} = unflatten {{%0.s[0]}} {{%0.s[1]}} t2 in"));
        code.append(format!(r"let {{%2}} = unflatten t2 in"));
        /*code.append(format!(r"let {{%2}} = unflatten t2 :> [{{%0.s[0]}}][{{%0.s[1]}}]{} in",
            arg[0].dtype.format_futhark(),
        ));*/
      }
      3 => {
        unimplemented!();
      }
      4 => {
        unimplemented!();
      }
      _ => {
        unimplemented!();
      }
    }
    code.into()
  }

  fn pop_adj(&self, arg: &[(CellPtr, Clock)], out: CellPtr, out_clk: Clock, out_adj: CellPtr, arg_adj: &mut [CellPtr]) -> Result<FutharkThunkAdj, ThunkAdjErr> {
    let arg0_ty = arg[0].0.type_();
    let arg0_nd = arg0_ty.ndim() as usize;
    let inner_len = arg0_ty.shape[arg0_nd - 1];
    arg_adj[0] += out_adj.inner_inv_select(arg[1].0, inner_len);
    Ok(FutharkThunkAdj::Spec)
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct InnerInvSelectFutThunkSpec { pub inner_len: i64 }

impl FutharkThunkSpec for InnerInvSelectFutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.inner_inv_select")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn arity(&self) -> Option<(u16, u16)> {
    Some((2, 1))
  }

  /*fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 2;
    abi.arityout = 1;
    abi
  }*/

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    // FIXME
    if !arg[1].dtype.is_uint() {
      return Err(ThunkDimErr::_Bot);
    }
    if arg[0].ndim() != arg[1].ndim() {
      return Err(ThunkDimErr::_Bot);
    }
    Ok(Dim{ndim: arg[1].ndim() + 1, dtype: arg[0].dtype})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    // FIXME
    if !arg[1].dtype.is_uint() {
      return Err(ThunkTypeErr::_Bot);
    }
    if arg[0].ndim() != arg[1].ndim() {
      return Err(ThunkTypeErr::_Bot);
    }
    if &arg[0].shape != &arg[1].shape {
      return Err(ThunkTypeErr::_Bot);
    }
    let mut shape = arg[1].shape.clone();
    shape.push(self.inner_len);
    Ok(CellType{shape, dtype: arg[0].dtype})
  }

  fn gen_futhark(&self, /*abi: &mut FutAbi,*/ arg: &[Dim], out: &[Dim]) -> Result<FutharkThunkGenCode, FutharkGenErr> {
    //abi.set_out_arr(0, AbiOutput::Pure, AbiArrayRepr::Nd, AbiScalarType::Unspec);
    //abi.set_arg_arr(0, AbiInput::Shared, AbiArrayRepr::Nd, AbiScalarType::Unspec);
    //abi.set_arg_arr(1, AbiInput::Shared, AbiArrayRepr::Nd, AbiScalarType::Unspec);
    //let out = FutharkThunkSpec::out_dim(self, arg).map_err(|e| e.into_gen())?;
    let mut code = FutharkThunkGenCode::default();
    code.abi.arityout = 1;
    code.abi.set_out(0, FutharkArrayRepr::Nd);
    code.abi.arityin = 2;
    code.abi.set_arg(0, FutharkArrayRepr::Nd);
    code.abi.set_arg(1, FutharkArrayRepr::Nd);
    match out[0].ndim {
      0 => {
        unimplemented!();
      }
      1 => {
        unimplemented!();
      }
      2 => {
        unimplemented!();
      }
      3 => {
        code.cfg.emit_arg_shapes = true;
        code.append(format!(r"let a = {{%0.s[0]}} * {{%0.s[1]}} in"));
        code.append(format!(r"let t_val = flatten {{%0}} :> [a]{} in", arg[0].dtype.format_futhark()));
        /*code.append(format!(r"let t_key = flatten {{%1}} :> [a]{} in", arg[1].dtype.format_futhark()));
        code.append(format!(r"let t_key = flatten t_key :> [a]{} in", arg[1].dtype.format_futhark()));*/
        code.append(format!(r"let t1 = flatten {{%1}} :> [a]{} in", arg[1].dtype.format_futhark()));
        code.append(format!(r"let t_iota = iota a in"));
        code.append(format!(r"let t_key = map2 (\idx k -> let k = (i64.{} k) in (assert (k >= 0 && k < {}) k) + {} * idx) t_iota t1 in",
            arg[1].dtype.format_futhark(),
            self.inner_len,
            self.inner_len,
        ));
        code.append(format!(r"let t2 = spread (a * {}) 0.0{} t_key t_val in",
            self.inner_len,
            arg[0].dtype.format_futhark(),
        ));
        /*code.append(format!(r"let {{%2}} = unflatten_3d {{%0.s[0]}} {{%0.s[1]}} {} t2 in",
            self.inner_len,
        ));*/
        code.append(format!(r"let {{%2}} = unflatten_3d t2 in"));
      }
      4 => {
        unimplemented!();
      }
      _ => {
        unimplemented!();
      }
    }
    code.into()
  }

  fn pop_adj(&self, arg: &[(CellPtr, Clock)], out: CellPtr, out_clk: Clock, out_adj: CellPtr, arg_adj: &mut [CellPtr]) -> Result<FutharkThunkAdj, ThunkAdjErr> {
    arg_adj[0] += out_adj.inner_select(arg[1].0);
    Ok(FutharkThunkAdj::Spec)
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct OuterSelectFutThunkSpec;

impl FutharkThunkSpec for OuterSelectFutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.outer_select")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn arity(&self) -> Option<(u16, u16)> {
    Some((2, 1))
  }

  /*fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 2;
    abi.arityout = 1;
    abi
  }*/

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    // FIXME
    if !arg[1].dtype.is_uint() {
      return Err(ThunkDimErr::_Bot);
    }
    if arg[0].ndim() < 1 {
      return Err(ThunkDimErr::_Bot);
    }
    let ndim = arg[1].ndim() + arg[0].ndim() - 1;
    Ok(Dim{ndim, dtype: arg[0].dtype})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    // FIXME
    if !arg[1].dtype.is_uint() {
      return Err(ThunkTypeErr::_Bot);
    }
    if arg[0].ndim() < 1 {
      return Err(ThunkTypeErr::_Bot);
    }
    let mut shape = arg[1].shape.clone();
    shape.extend_from_slice(&arg[0].shape[1 .. ]);
    Ok(CellType{shape, dtype: arg[0].dtype})
  }

  fn gen_futhark(&self, /*abi: &mut FutAbi,*/ arg: &[Dim], _out: &[Dim]) -> Result<FutharkThunkGenCode, FutharkGenErr> {
    //abi.set_out_arr(0, AbiOutput::Pure, AbiArrayRepr::Nd, AbiScalarType::Unspec);
    //abi.set_arg_arr(0, AbiInput::Shared, AbiArrayRepr::Nd, AbiScalarType::Unspec);
    //abi.set_arg_arr(1, AbiInput::Shared, AbiArrayRepr::Nd, AbiScalarType::Unspec);
    //let out = FutharkThunkSpec::out_dim(self, arg).map_err(|e| e.into_gen())?;
    let mut code = FutharkThunkGenCode::default();
    code.abi.arityout = 1;
    code.abi.set_out(0, FutharkArrayRepr::Nd);
    code.abi.arityin = 2;
    code.abi.set_arg(0, FutharkArrayRepr::Nd);
    code.abi.set_arg(1, FutharkArrayRepr::Nd);
    match arg[1].ndim {
      0 => panic!("bug"),
      1 => {
        code.append(format!(r"let {{%2}} = map (\k -> {{%0}}[i64.{} k]) {{%1}} in",
            arg[1].dtype.format_futhark(),
        ));
      }
      2 => {
        code.append(format!(r"let {{%2}} = map (\t -> map (\k -> {{%0}}[i64.{} k]) t) {{%1}} in",
            arg[1].dtype.format_futhark(),
        ));
      }
      3 => {
        code.append(format!(r"let {{%2}} = map (\t1 -> map (\t2 -> map (\k -> {{%0}}[i64.{} k]) t2) t1) {{%1}} in",
            arg[1].dtype.format_futhark(),
        ));
      }
      4 => {
        unimplemented!();
      }
      _ => {
        unimplemented!();
      }
    }
    code.into()
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct OuterMulFutThunkSpec;

impl FutharkThunkSpec for OuterMulFutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.outer_mul")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn arity(&self) -> Option<(u16, u16)> {
    Some((2, 1))
  }

  /*fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 2;
    abi.arityout = 1;
    abi
  }*/

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    if arg[0].ndim() != 1 {
      return Err(ThunkDimErr::_Bot);
    }
    if 1 != arg[1].ndim() {
      return Err(ThunkDimErr::_Bot);
    }
    if arg[0].dtype != arg[1].dtype {
      return Err(ThunkDimErr::_Bot);
    }
    let ndim = 2;
    let dtype = arg[0].dtype;
    Ok(Dim{ndim, dtype})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    if arg[0].ndim() != 1 {
      return Err(ThunkTypeErr::_Bot);
    }
    if 1 != arg[1].ndim() {
      return Err(ThunkTypeErr::_Bot);
    }
    if arg[0].dtype != arg[1].dtype {
      return Err(ThunkTypeErr::_Bot);
    }
    let mut shape = arg[0].shape.clone();
    shape.push(arg[1].shape[0]);
    assert_eq!(shape.len(), 2);
    let dtype = arg[0].dtype;
    Ok(CellType{shape, dtype})
  }

  fn gen_futhark(&self, /*abi: &mut FutAbi,*/ arg: &[Dim], _out: &[Dim]) -> Result<FutharkThunkGenCode, FutharkGenErr> {
    //abi.set_out_arr(0, AbiOutput::Pure, AbiArrayRepr::Nd, AbiScalarType::Unspec);
    //abi.set_arg_arr(0, AbiInput::Shared, AbiArrayRepr::Nd, AbiScalarType::Unspec);
    //abi.set_arg_arr(1, AbiInput::Shared, AbiArrayRepr::Nd, AbiScalarType::Unspec);
    let mut code = FutharkThunkGenCode::default();
    code.abi.arityout = 1;
    code.abi.set_out(0, FutharkArrayRepr::Nd);
    code.abi.arityin = 2;
    code.abi.set_arg(0, FutharkArrayRepr::Nd);
    code.abi.set_arg(1, FutharkArrayRepr::Nd);
    match (arg[0].ndim(), arg[1].ndim()) {
      (1, 1) => {
        code.append(format!(r"let {{%2}} = map (\u -> map (\v -> u * v) {{%1}}) {{%0}} in"));
      }
      _ => {
        unimplemented!();
      }
    }
    code.into()
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct AddScalarFutThunkSpec { pub val: ScalarVal_ }

impl FutharkThunkSpec for AddScalarFutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.add_scalar")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn arity(&self) -> Option<(u16, u16)> {
    Some((1, 1))
  }

  /*fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }*/

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(arg[0])
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(arg[0].clone())
  }

  fn gen_futhark(&self, /*abi: &mut FutAbi,*/ arg: &[Dim], _out: &[Dim]) -> Result<FutharkThunkGenCode, FutharkGenErr> {
    FutharkThunkGenCode::flat_map(arg[0], format!(r"\u -> u + {}", self.val.format_futhark()))
  }

  fn pop_adj(&self, _arg: &[(CellPtr, Clock)], _out: CellPtr, _out_clk: Clock, out_adj: CellPtr, arg_adj: &mut [CellPtr]) -> Result<FutharkThunkAdj, ThunkAdjErr> {
    // FIXME: snapshot.
    arg_adj[0] += out_adj._memcpy();
    Ok(FutharkThunkAdj::Spec)
  }
}

/*#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct AddScalarF32FutThunkSpec { pub val: TotalOrd<f32> }

impl FutharkThunkSpec for AddScalarF32FutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.f32.add_scalar")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn arity(&self) -> Option<(u16, u16)> {
    Some((1, 1))
  }

  /*fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }*/

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(Dim{ndim: arg[0].ndim, dtype: f32::dtype_()})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(CellType{shape: arg[0].shape.clone(), dtype: f32::dtype_()})
  }

  fn gen_futhark(&self, /*abi: &mut FutAbi,*/ arg: &[Dim], _out: &[Dim]) -> Result<FutharkThunkGenCode, FutharkGenErr> {
    let fmt = FutharkNumFormatter::default();
    FutharkThunkGenCode::flat_map(arg[0], format!(r"\u -> u + {}", fmt.format(&self.val)))
  }

  fn pop_adj(&self, _arg: &[(CellPtr, Clock)], _out: CellPtr, _out_clk: Clock, out_adj: CellPtr, arg_adj: &mut [CellPtr]) -> Result<FutharkThunkAdj, ThunkAdjErr> {
    // FIXME: snapshot.
    arg_adj[0] += out_adj._memcpy();
    Ok(FutharkThunkAdj::Spec)
  }
}*/

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct AddFutThunkSpec;

impl FutharkThunkSpec for AddFutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.add")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn arity(&self) -> Option<(u16, u16)> {
    Some((2, 1))
  }

  /*fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 2;
    abi.arityout = 1;
    abi
  }*/

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    if arg[0].ndim() != arg[1].ndim() {
      return Err(ThunkDimErr::_Bot);
    }
    if arg[0].dtype != arg[1].dtype {
      return Err(ThunkDimErr::_Bot);
    }
    Ok(Dim{ndim: arg[0].ndim, dtype: arg[0].dtype})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    if arg[0].ndim() != arg[1].ndim() {
      return Err(ThunkTypeErr::_Bot);
    }
    if arg[0].dtype != arg[1].dtype {
      return Err(ThunkTypeErr::_Bot);
    }
    if &arg[0].shape != &arg[1].shape {
      return Err(ThunkTypeErr::_Bot);
    }
    Ok(CellType{shape: arg[0].shape.clone(), dtype: arg[0].dtype})
  }

  fn gen_futhark(&self, /*abi: &mut FutAbi,*/ arg: &[Dim], _out: &[Dim]) -> Result<FutharkThunkGenCode, FutharkGenErr> {
    FutharkThunkGenCode::flat_map2(arg[0], arg[1], r"+")
  }

  fn pop_adj(&self, _arg: &[(CellPtr, Clock)], _out: CellPtr, _out_clk: Clock, out_adj: CellPtr, arg_adj: &mut [CellPtr]) -> Result<FutharkThunkAdj, ThunkAdjErr> {
    // FIXME: snapshot.
    arg_adj[1] += out_adj._memcpy();
    arg_adj[0] += out_adj._memcpy();
    Ok(FutharkThunkAdj::Spec)
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct BroadcastAddFutThunkSpec { pub mono: FutharkNdBroadcastMap2MonomorphicSpec }

impl FutharkThunkSpec for BroadcastAddFutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.broadcast_add")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn arity(&self) -> Option<(u16, u16)> {
    Some((2, 1))
  }

  /*fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 2;
    abi.arityout = 1;
    abi
  }*/

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    if arg[0].ndim() != self.mono.ndim() {
      return Err(ThunkDimErr::_Bot);
    }
    if arg[1].ndim() != self.mono.ndim() {
      return Err(ThunkDimErr::_Bot);
    }
    let dtype = arg[0].dtype.max(arg[1].dtype);
    if dtype.is_none() {
      return Err(ThunkDimErr::_Bot);
    }
    let dtype = dtype.unwrap();
    Ok(Dim{ndim: arg[0].ndim(), dtype})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    if arg[0].ndim() != self.mono.ndim() {
      return Err(ThunkTypeErr::_Bot);
    }
    if arg[1].ndim() != self.mono.ndim() {
      return Err(ThunkTypeErr::_Bot);
    }
    let dtype = arg[0].dtype.max(arg[1].dtype);
    if dtype.is_none() {
      return Err(ThunkTypeErr::_Bot);
    }
    let dtype = dtype.unwrap();
    let nd = arg[0].ndim() as usize;
    let mut shape = Vec::with_capacity(nd);
    for d in 0 .. nd {
      if !(arg[0].shape[d] == arg[1].shape[d] ||
           arg[0].shape[d] == 1 ||
           arg[1].shape[d] == 1)
      {
        return Err(ThunkTypeErr::_Bot);
      }
      shape.push(max(arg[0].shape[d], arg[1].shape[d]));
    }
    Ok(CellType{shape, dtype})
  }

  fn gen_futhark(&self, /*abi: &mut FutAbi,*/ arg: &[Dim], _out: &[Dim]) -> Result<FutharkThunkGenCode, FutharkGenErr> {
    // FIXME
    //FutharkThunkGenCode::nd_broadcast_map2_v0(arg[0], arg[1], r"+")
    let dtype = arg[0].dtype.max(arg[1].dtype).unwrap();
    /*FutharkThunkGenCode::nd_broadcast_map2(abi, arg[0], arg[1], format!(r"\u v -> ({}.{} u) + ({}.{} v)",
        dtype.format_futhark(), arg[0].dtype.format_futhark(),
        dtype.format_futhark(), arg[1].dtype.format_futhark(),
    ))*/
    self.mono.gen_futhark(arg[0], arg[1], format!(r"\u v -> ({}.{} u) + ({}.{} v)",
        dtype.format_futhark(), arg[0].dtype.format_futhark(),
        dtype.format_futhark(), arg[1].dtype.format_futhark(),
    ))
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct LSubScalarFutThunkSpec { pub val: ScalarVal_ }

impl FutharkThunkSpec for LSubScalarFutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.left_sub_scalar")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn arity(&self) -> Option<(u16, u16)> {
    Some((1, 1))
  }

  /*fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }*/

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(arg[0])
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(arg[0].clone())
  }

  fn gen_futhark(&self, /*abi: &mut FutAbi,*/ arg: &[Dim], _out: &[Dim]) -> Result<FutharkThunkGenCode, FutharkGenErr> {
    FutharkThunkGenCode::flat_map(arg[0], format!(r"\u -> {} - u", self.val.format_futhark()))
  }

  fn pop_adj(&self, _arg: &[(CellPtr, Clock)], _out: CellPtr, _out_clk: Clock, out_adj: CellPtr, arg_adj: &mut [CellPtr]) -> Result<FutharkThunkAdj, ThunkAdjErr> {
    arg_adj[0] += -out_adj;
    Ok(FutharkThunkAdj::Spec)
  }
}

/*#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct LSubScalarF32FutThunkSpec { pub val: TotalOrd<f32> }

impl FutharkThunkSpec for LSubScalarF32FutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.f32.left_sub_scalar")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  /*fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }*/

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(Dim{ndim: arg[0].ndim, dtype: f32::dtype_()})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(CellType{shape: arg[0].shape.clone(), dtype: f32::dtype_()})
  }

  fn gen_futhark(&self, /*abi: &mut FutAbi,*/ arg: &[Dim], _out: &[Dim]) -> Result<FutharkThunkGenCode, FutharkGenErr> {
    let fmt = FutharkNumFormatter::default();
    FutharkThunkGenCode::flat_map(arg[0], format!(r"\u -> {} - u", fmt.format(&self.val)))
  }

  fn pop_adj(&self, _arg: &[(CellPtr, Clock)], _out: CellPtr, _out_clk: Clock, out_adj: CellPtr, arg_adj: &mut [CellPtr]) -> Result<FutharkThunkAdj, ThunkAdjErr> {
    arg_adj[0] += -out_adj;
    Ok(FutharkThunkAdj::Spec)
  }
}*/

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct RSubScalarFutThunkSpec { pub val: ScalarVal_ }

impl FutharkThunkSpec for RSubScalarFutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.right_sub_scalar")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn arity(&self) -> Option<(u16, u16)> {
    Some((1, 1))
  }

  /*fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }*/

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(arg[0])
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(arg[0].clone())
  }

  fn gen_futhark(&self, /*abi: &mut FutAbi,*/ arg: &[Dim], _out: &[Dim]) -> Result<FutharkThunkGenCode, FutharkGenErr> {
    FutharkThunkGenCode::flat_map(arg[0], format!(r"\u -> u - {}", self.val.format_futhark()))
  }

  fn pop_adj(&self, _arg: &[(CellPtr, Clock)], _out: CellPtr, _out_clk: Clock, out_adj: CellPtr, arg_adj: &mut [CellPtr]) -> Result<FutharkThunkAdj, ThunkAdjErr> {
    // FIXME: snapshot.
    arg_adj[0] += out_adj._memcpy();
    Ok(FutharkThunkAdj::Spec)
  }
}

/*#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct RSubScalarF32FutThunkSpec { pub val: TotalOrd<f32> }

impl FutharkThunkSpec for RSubScalarF32FutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.f32.right_sub_scalar")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn arity(&self) -> Option<(u16, u16)> {
    Some((1, 1))
  }

  /*fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }*/

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(Dim{ndim: arg[0].ndim, dtype: f32::dtype_()})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(CellType{shape: arg[0].shape.clone(), dtype: f32::dtype_()})
  }

  fn gen_futhark(&self, /*abi: &mut FutAbi,*/ arg: &[Dim], _out: &[Dim]) -> Result<FutharkThunkGenCode, FutharkGenErr> {
    let fmt = FutharkNumFormatter::default();
    FutharkThunkGenCode::flat_map(arg[0], format!(r"\u -> u - {}", fmt.format(&self.val)))
  }

  fn pop_adj(&self, _arg: &[(CellPtr, Clock)], _out: CellPtr, _out_clk: Clock, out_adj: CellPtr, arg_adj: &mut [CellPtr]) -> Result<FutharkThunkAdj, ThunkAdjErr> {
    // FIXME: snapshot.
    arg_adj[0] += out_adj._memcpy();
    Ok(FutharkThunkAdj::Spec)
  }
}*/

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct SubFutThunkSpec;

impl FutharkThunkSpec for SubFutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.sub")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn arity(&self) -> Option<(u16, u16)> {
    Some((2, 1))
  }

  /*fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 2;
    abi.arityout = 1;
    abi
  }*/

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    if arg[0].ndim() != arg[1].ndim() {
      return Err(ThunkDimErr::_Bot);
    }
    if arg[0].dtype != arg[1].dtype {
      return Err(ThunkDimErr::_Bot);
    }
    Ok(Dim{ndim: arg[0].ndim, dtype: arg[0].dtype})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    if arg[0].ndim() != arg[1].ndim() {
      return Err(ThunkTypeErr::_Bot);
    }
    if arg[0].dtype != arg[1].dtype {
      return Err(ThunkTypeErr::_Bot);
    }
    if &arg[0].shape != &arg[1].shape {
      return Err(ThunkTypeErr::_Bot);
    }
    Ok(CellType{shape: arg[0].shape.clone(), dtype: arg[0].dtype})
  }

  fn gen_futhark(&self, /*abi: &mut FutAbi,*/ arg: &[Dim], _out: &[Dim]) -> Result<FutharkThunkGenCode, FutharkGenErr> {
    FutharkThunkGenCode::flat_map2(arg[0], arg[1], r"-")
  }

  fn pop_adj(&self, _arg: &[(CellPtr, Clock)], _out: CellPtr, _out_clk: Clock, out_adj: CellPtr, arg_adj: &mut [CellPtr]) -> Result<FutharkThunkAdj, ThunkAdjErr> {
    arg_adj[1] += -out_adj;
    // FIXME: snapshot.
    arg_adj[0] += out_adj._memcpy();
    Ok(FutharkThunkAdj::Spec)
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct BroadcastSubFutThunkSpec { pub mono: FutharkNdBroadcastMap2MonomorphicSpec }

impl FutharkThunkSpec for BroadcastSubFutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.broadcast_sub")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn arity(&self) -> Option<(u16, u16)> {
    Some((2, 1))
  }

  /*fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 2;
    abi.arityout = 1;
    abi
  }*/

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    if arg[0].ndim() != self.mono.ndim() {
      return Err(ThunkDimErr::_Bot);
    }
    if arg[1].ndim() != self.mono.ndim() {
      return Err(ThunkDimErr::_Bot);
    }
    let dtype = arg[0].dtype.max(arg[1].dtype);
    if dtype.is_none() {
      return Err(ThunkDimErr::_Bot);
    }
    let dtype = dtype.unwrap();
    Ok(Dim{ndim: arg[0].ndim(), dtype})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    if arg[0].ndim() != self.mono.ndim() {
      return Err(ThunkTypeErr::_Bot);
    }
    if arg[1].ndim() != self.mono.ndim() {
      return Err(ThunkTypeErr::_Bot);
    }
    let dtype = arg[0].dtype.max(arg[1].dtype);
    if dtype.is_none() {
      return Err(ThunkTypeErr::_Bot);
    }
    let dtype = dtype.unwrap();
    let nd = arg[0].ndim() as usize;
    let mut shape = Vec::with_capacity(nd);
    for d in 0 .. nd {
      if !(arg[0].shape[d] == arg[1].shape[d] ||
           arg[0].shape[d] == 1 ||
           arg[1].shape[d] == 1)
      {
        return Err(ThunkTypeErr::_Bot);
      }
      shape.push(max(arg[0].shape[d], arg[1].shape[d]));
    }
    Ok(CellType{shape, dtype})
  }

  fn gen_futhark(&self, /*abi: &mut FutAbi,*/ arg: &[Dim], _out: &[Dim]) -> Result<FutharkThunkGenCode, FutharkGenErr> {
    // FIXME
    //FutharkThunkGenCode::nd_broadcast_map2_v0(arg[0], arg[1], r"-")
    let dtype = arg[0].dtype.max(arg[1].dtype).unwrap();
    /*FutharkThunkGenCode::nd_broadcast_map2(abi, arg[0], arg[1], format!(r"\u v -> ({}.{} u) - ({}.{} v)",
        dtype.format_futhark(), arg[0].dtype.format_futhark(),
        dtype.format_futhark(), arg[1].dtype.format_futhark(),
    ))*/
    self.mono.gen_futhark(arg[0], arg[1], format!(r"\u v -> ({}.{} u) - ({}.{} v)",
        dtype.format_futhark(), arg[0].dtype.format_futhark(),
        dtype.format_futhark(), arg[1].dtype.format_futhark(),
    ))
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct MulScalarFutThunkSpec { pub val: ScalarVal_ }

impl FutharkThunkSpec for MulScalarFutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.mul_scalar")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn arity(&self) -> Option<(u16, u16)> {
    Some((1, 1))
  }

  /*fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }*/

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(arg[0])
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(arg[0].clone())
  }

  fn gen_futhark(&self, /*abi: &mut FutAbi,*/ arg: &[Dim], _out: &[Dim]) -> Result<FutharkThunkGenCode, FutharkGenErr> {
    // FIXME: param.
    FutharkThunkGenCode::flat_map(arg[0], format!(r"\u -> u * {}", self.val.format_futhark()))
  }

  fn pop_adj(&self, _arg: &[(CellPtr, Clock)], _out: CellPtr, _out_clk: Clock, out_adj: CellPtr, arg_adj: &mut [CellPtr]) -> Result<FutharkThunkAdj, ThunkAdjErr> {
    arg_adj[0] += out_adj * self.val;
    Ok(FutharkThunkAdj::Spec)
  }
}

/*#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct MulScalarF32FutThunkSpec { pub val: TotalOrd<f32> }

impl FutharkThunkSpec for MulScalarF32FutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.f32.mul_scalar")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  /*fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }*/

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(Dim{ndim: arg[0].ndim, dtype: f32::dtype_()})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(CellType{shape: arg[0].shape.clone(), dtype: arg[0].dtype})
  }

  fn gen_futhark(&self, /*abi: &mut FutAbi,*/ arg: &[Dim], _out: &[Dim]) -> Result<FutharkThunkGenCode, FutharkGenErr> {
    // FIXME: param.
    let fmt = FutharkNumFormatter::default();
    FutharkThunkGenCode::flat_map(arg[0], format!(r"\u -> u * {}", fmt.format(&self.val)))
  }

  fn pop_adj(&self, _arg: &[(CellPtr, Clock)], _out: CellPtr, _out_clk: Clock, out_adj: CellPtr, arg_adj: &mut [CellPtr]) -> Result<FutharkThunkAdj, ThunkAdjErr> {
    arg_adj[0] += out_adj * self.val.0;
    Ok(FutharkThunkAdj::Spec)
  }
}*/

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct MulFutThunkSpec;

impl FutharkThunkSpec for MulFutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.mul")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn arity(&self) -> Option<(u16, u16)> {
    Some((2, 1))
  }

  /*fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 2;
    abi.arityout = 1;
    abi
  }*/

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    if arg[0].ndim() != arg[1].ndim() {
      return Err(ThunkDimErr::_Bot);
    }
    if arg[0].dtype != arg[1].dtype {
      return Err(ThunkDimErr::_Bot);
    }
    Ok(Dim{ndim: arg[0].ndim, dtype: arg[0].dtype})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    if arg[0].ndim() != arg[1].ndim() {
      return Err(ThunkTypeErr::_Bot);
    }
    if arg[0].dtype != arg[1].dtype {
      return Err(ThunkTypeErr::_Bot);
    }
    if &arg[0].shape != &arg[1].shape {
      return Err(ThunkTypeErr::_Bot);
    }
    Ok(CellType{shape: arg[0].shape.clone(), dtype: arg[0].dtype})
  }

  fn gen_futhark(&self, /*abi: &mut FutAbi,*/ arg: &[Dim], _out: &[Dim]) -> Result<FutharkThunkGenCode, FutharkGenErr> {
    FutharkThunkGenCode::flat_map2(arg[0], arg[1], r"*")
  }

  fn pop_adj(&self, arg: &[(CellPtr, Clock)], _out: CellPtr, _out_clk: Clock, out_adj: CellPtr, arg_adj: &mut [CellPtr]) -> Result<FutharkThunkAdj, ThunkAdjErr> {
    arg_adj[1] += arg[0].0 * out_adj;
    arg_adj[0] += out_adj * arg[1].0;
    Ok(FutharkThunkAdj::Spec)
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct BroadcastMulFutThunkSpec { pub mono: FutharkNdBroadcastMap2MonomorphicSpec }

impl FutharkThunkSpec for BroadcastMulFutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.broadcast_mul")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn arity(&self) -> Option<(u16, u16)> {
    Some((2, 1))
  }

  /*fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 2;
    abi.arityout = 1;
    abi
  }*/

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    if arg[0].ndim() != self.mono.ndim() {
      return Err(ThunkDimErr::_Bot);
    }
    if arg[1].ndim() != self.mono.ndim() {
      return Err(ThunkDimErr::_Bot);
    }
    let dtype = arg[0].dtype.max(arg[1].dtype);
    if dtype.is_none() {
      return Err(ThunkDimErr::_Bot);
    }
    let dtype = dtype.unwrap();
    Ok(Dim{ndim: arg[0].ndim(), dtype})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    if arg[0].ndim() != self.mono.ndim() {
      return Err(ThunkTypeErr::_Bot);
    }
    if arg[1].ndim() != self.mono.ndim() {
      return Err(ThunkTypeErr::_Bot);
    }
    let dtype = arg[0].dtype.max(arg[1].dtype);
    if dtype.is_none() {
      return Err(ThunkTypeErr::_Bot);
    }
    let dtype = dtype.unwrap();
    let nd = arg[0].ndim() as usize;
    let mut shape = Vec::with_capacity(nd);
    for d in 0 .. nd {
      if !(arg[0].shape[d] == arg[1].shape[d] ||
           arg[0].shape[d] == 1 ||
           arg[1].shape[d] == 1)
      {
        return Err(ThunkTypeErr::_Bot);
      }
      shape.push(max(arg[0].shape[d], arg[1].shape[d]));
    }
    Ok(CellType{shape, dtype})
  }

  fn gen_futhark(&self, /*abi: &mut FutAbi,*/ arg: &[Dim], _out: &[Dim]) -> Result<FutharkThunkGenCode, FutharkGenErr> {
    // FIXME
    let dtype = arg[0].dtype.max(arg[1].dtype).unwrap();
    /*FutharkThunkGenCode::nd_broadcast_map2(abi, arg[0], arg[1], format!(r"\u v -> ({}.{} u) * ({}.{} v)",
        dtype.format_futhark(), arg[0].dtype.format_futhark(),
        dtype.format_futhark(), arg[1].dtype.format_futhark(),
    ))*/
    self.mono.gen_futhark(arg[0], arg[1], format!(r"\u v -> ({}.{} u) * ({}.{} v)",
        dtype.format_futhark(), arg[0].dtype.format_futhark(),
        dtype.format_futhark(), arg[1].dtype.format_futhark(),
    ))
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct LDivScalarFutThunkSpec { pub val: ScalarVal_ }

impl FutharkThunkSpec for LDivScalarFutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.left_div_scalar")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn arity(&self) -> Option<(u16, u16)> {
    Some((1, 1))
  }

  /*fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }*/

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(Dim{ndim: arg[0].ndim, dtype: self.val.dtype()})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(CellType{shape: arg[0].shape.clone(), dtype: self.val.dtype()})
  }

  fn gen_futhark(&self, /*abi: &mut FutAbi,*/ arg: &[Dim], _out: &[Dim]) -> Result<FutharkThunkGenCode, FutharkGenErr> {
    // FIXME: param.
    FutharkThunkGenCode::flat_map(arg[0], format!(r"\u -> {} / u", self.val.format_futhark()))
  }

  fn pop_adj(&self, arg: &[(CellPtr, Clock)], _out: CellPtr, _out_clk: Clock, out_adj: CellPtr, arg_adj: &mut [CellPtr]) -> Result<FutharkThunkAdj, ThunkAdjErr> {
    arg_adj[0] += (out_adj / arg[0].0) * (-self.val / arg[0].0);
    Ok(FutharkThunkAdj::Spec)
  }
}

/*#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct LDivScalarF32FutThunkSpec { pub val: TotalOrd<f32> }

impl FutharkThunkSpec for LDivScalarF32FutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.f32.left_div_scalar")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  /*fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }*/

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(Dim{ndim: arg[0].ndim, dtype: f32::dtype_()})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(CellType{shape: arg[0].shape.clone(), dtype: f32::dtype_()})
  }

  fn gen_futhark(&self, /*abi: &mut FutAbi,*/ arg: &[Dim], _out: &[Dim]) -> Result<FutharkThunkGenCode, FutharkGenErr> {
    // FIXME: param.
    let fmt = FutharkNumFormatter::default();
    FutharkThunkGenCode::flat_map(arg[0], format!(r"\u -> {} / u", fmt.format(&self.val)))
  }

  fn pop_adj(&self, arg: &[(CellPtr, Clock)], _out: CellPtr, _out_clk: Clock, out_adj: CellPtr, arg_adj: &mut [CellPtr]) -> Result<FutharkThunkAdj, ThunkAdjErr> {
    //arg_adj[0] += -((out_adj * self.val) / (arg[0].0 * arg[0].0));
    arg_adj[0] += (out_adj / arg[0].0) * (-self.val / arg[0].0);
    Ok(FutharkThunkAdj::Spec)
  }
}*/

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct RDivScalarFutThunkSpec { pub val: ScalarVal_ }

impl FutharkThunkSpec for RDivScalarFutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.right_div_scalar")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn arity(&self) -> Option<(u16, u16)> {
    Some((1, 1))
  }

  /*fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }*/

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(Dim{ndim: arg[0].ndim, dtype: self.val.dtype()})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(CellType{shape: arg[0].shape.clone(), dtype: self.val.dtype()})
  }

  fn gen_futhark(&self, /*abi: &mut FutAbi,*/ arg: &[Dim], _out: &[Dim]) -> Result<FutharkThunkGenCode, FutharkGenErr> {
    // FIXME: param.
    FutharkThunkGenCode::flat_map(arg[0], format!(r"\u -> u / {}", self.val.format_futhark()))
  }

  fn pop_adj(&self, arg: &[(CellPtr, Clock)], _out: CellPtr, _out_clk: Clock, out_adj: CellPtr, arg_adj: &mut [CellPtr]) -> Result<FutharkThunkAdj, ThunkAdjErr> {
    arg_adj[0] += out_adj / self.val;
    Ok(FutharkThunkAdj::Spec)
  }
}

/*#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct RDivScalarF32FutThunkSpec { pub val: TotalOrd<f32> }

impl FutharkThunkSpec for RDivScalarF32FutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.f32.right_div_scalar")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  /*fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }*/

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(Dim{ndim: arg[0].ndim, dtype: f32::dtype_()})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(CellType{shape: arg[0].shape.clone(), dtype: f32::dtype_()})
  }

  fn gen_futhark(&self, /*abi: &mut FutAbi,*/ arg: &[Dim], _out: &[Dim]) -> Result<FutharkThunkGenCode, FutharkGenErr> {
    // FIXME: param.
    let fmt = FutharkNumFormatter::default();
    FutharkThunkGenCode::flat_map(arg[0], format!(r"\u -> u / {}", fmt.format(&self.val)))
  }

  fn pop_adj(&self, _arg: &[(CellPtr, Clock)], _out: CellPtr, _out_clk: Clock, out_adj: CellPtr, arg_adj: &mut [CellPtr]) -> Result<FutharkThunkAdj, ThunkAdjErr> {
    arg_adj[0] += out_adj / self.val.0;
    Ok(FutharkThunkAdj::Spec)
  }
}*/

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct DivFutThunkSpec;

impl FutharkThunkSpec for DivFutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.div")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn arity(&self) -> Option<(u16, u16)> {
    Some((2, 1))
  }

  /*fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 2;
    abi.arityout = 1;
    abi
  }*/

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    if arg[0].ndim() != arg[1].ndim() {
      return Err(ThunkDimErr::_Bot);
    }
    if arg[0].dtype != arg[1].dtype {
      return Err(ThunkDimErr::_Bot);
    }
    Ok(Dim{ndim: arg[0].ndim, dtype: arg[0].dtype})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    if arg[0].ndim() != arg[1].ndim() {
      return Err(ThunkTypeErr::_Bot);
    }
    if arg[0].dtype != arg[1].dtype {
      return Err(ThunkTypeErr::_Bot);
    }
    if &arg[0].shape != &arg[1].shape {
      return Err(ThunkTypeErr::_Bot);
    }
    Ok(CellType{shape: arg[0].shape.clone(), dtype: arg[0].dtype})
  }

  fn gen_futhark(&self, /*abi: &mut FutAbi,*/ arg: &[Dim], _out: &[Dim]) -> Result<FutharkThunkGenCode, FutharkGenErr> {
    FutharkThunkGenCode::flat_map2(arg[0], arg[1], r"/")
  }

  fn pop_adj(&self, arg: &[(CellPtr, Clock)], _out: CellPtr, _out_clk: Clock, out_adj: CellPtr, arg_adj: &mut [CellPtr]) -> Result<FutharkThunkAdj, ThunkAdjErr> {
    //arg_adj[1] += (out_adj * arg[0].0) / -arg[1].0.square();
    arg_adj[1] += (out_adj / arg[1].0) * (-arg[0].0 / arg[1].0);
    arg_adj[0] += out_adj / arg[1].0;
    Ok(FutharkThunkAdj::Spec)
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct BroadcastDivFutThunkSpec { pub mono: FutharkNdBroadcastMap2MonomorphicSpec }

impl FutharkThunkSpec for BroadcastDivFutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.broadcast_div")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn arity(&self) -> Option<(u16, u16)> {
    Some((2, 1))
  }

  /*fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 2;
    abi.arityout = 1;
    abi
  }*/

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    if arg[0].ndim() != self.mono.ndim() {
      return Err(ThunkDimErr::_Bot);
    }
    if arg[1].ndim() != self.mono.ndim() {
      return Err(ThunkDimErr::_Bot);
    }
    let dtype = arg[0].dtype.max(arg[1].dtype);
    if dtype.is_none() {
      return Err(ThunkDimErr::_Bot);
    }
    let dtype = dtype.unwrap();
    Ok(Dim{ndim: arg[0].ndim(), dtype})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    if arg[0].ndim() != self.mono.ndim() {
      return Err(ThunkTypeErr::_Bot);
    }
    if arg[1].ndim() != self.mono.ndim() {
      return Err(ThunkTypeErr::_Bot);
    }
    let dtype = arg[0].dtype.max(arg[1].dtype);
    if dtype.is_none() {
      return Err(ThunkTypeErr::_Bot);
    }
    let dtype = dtype.unwrap();
    let nd = arg[0].ndim() as usize;
    let mut shape = Vec::with_capacity(nd);
    for d in 0 .. nd {
      if !(arg[0].shape[d] == arg[1].shape[d] ||
           arg[0].shape[d] == 1 ||
           arg[1].shape[d] == 1)
      {
        return Err(ThunkTypeErr::_Bot);
      }
      shape.push(max(arg[0].shape[d], arg[1].shape[d]));
    }
    Ok(CellType{shape, dtype})
  }

  fn gen_futhark(&self, /*abi: &mut FutAbi,*/ arg: &[Dim], _out: &[Dim]) -> Result<FutharkThunkGenCode, FutharkGenErr> {
    // FIXME
    //FutharkThunkGenCode::nd_broadcast_map2_v0(arg[0], arg[1], r"/")
    let dtype = arg[0].dtype.max(arg[1].dtype).unwrap();
    /*FutharkThunkGenCode::nd_broadcast_map2(abi, arg[0], arg[1], format!(r"\u v -> ({}.{} u) / ({}.{} v)",
        dtype.format_futhark(), arg[0].dtype.format_futhark(),
        dtype.format_futhark(), arg[1].dtype.format_futhark(),
    ))*/
    self.mono.gen_futhark(arg[0], arg[1], format!(r"\u v -> ({}.{} u) / ({}.{} v)",
        dtype.format_futhark(), arg[0].dtype.format_futhark(),
        dtype.format_futhark(), arg[1].dtype.format_futhark(),
    ))
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct NegFutThunkSpec;

impl FutharkThunkSpec for NegFutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.neg")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn arity(&self) -> Option<(u16, u16)> {
    Some((1, 1))
  }

  /*fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }*/

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(Dim{ndim: arg[0].ndim, dtype: arg[0].dtype})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(CellType{shape: arg[0].shape.clone(), dtype: arg[0].dtype})
  }

  fn gen_futhark(&self, /*abi: &mut FutAbi,*/ arg: &[Dim], _out: &[Dim]) -> Result<FutharkThunkGenCode, FutharkGenErr> {
    FutharkThunkGenCode::flat_map(arg[0], r"\u -> -u")
  }

  fn pop_adj(&self, _arg: &[(CellPtr, Clock)], _out: CellPtr, _out_clk: Clock, out_adj: CellPtr, arg_adj: &mut [CellPtr]) -> Result<FutharkThunkAdj, ThunkAdjErr> {
    arg_adj[0] += -out_adj;
    Ok(FutharkThunkAdj::Spec)
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct NegF16FutThunkSpec;

impl FutharkThunkSpec for NegF16FutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.f16.neg")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn arity(&self) -> Option<(u16, u16)> {
    Some((1, 1))
  }

  /*fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }*/

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    assert_eq!(arg[0].dtype, Dtype::F16);
    Ok(Dim{ndim: arg[0].ndim, dtype: arg[0].dtype})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    assert_eq!(arg[0].dtype, Dtype::F16);
    Ok(CellType{shape: arg[0].shape.clone(), dtype: arg[0].dtype})
  }

  fn gen_futhark(&self, /*abi: &mut FutAbi,*/ arg: &[Dim], _out: &[Dim]) -> Result<FutharkThunkGenCode, FutharkGenErr> {
    FutharkThunkGenCode::flat_map(arg[0], r"\u -> f16.from_bits ((f16.to_bits u) ^ 0x8000u16)")
  }

  fn pop_adj(&self, _arg: &[(CellPtr, Clock)], _out: CellPtr, _out_clk: Clock, out_adj: CellPtr, arg_adj: &mut [CellPtr]) -> Result<FutharkThunkAdj, ThunkAdjErr> {
    arg_adj[0] += -out_adj;
    Ok(FutharkThunkAdj::Spec)
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct SquareFutThunkSpec;

impl FutharkThunkSpec for SquareFutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.square")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn arity(&self) -> Option<(u16, u16)> {
    Some((1, 1))
  }

  /*fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }*/

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(Dim{ndim: arg[0].ndim, dtype: arg[0].dtype})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(CellType{shape: arg[0].shape.clone(), dtype: arg[0].dtype})
  }

  fn gen_futhark(&self, /*abi: &mut FutAbi,*/ arg: &[Dim], _out: &[Dim]) -> Result<FutharkThunkGenCode, FutharkGenErr> {
    FutharkThunkGenCode::flat_map(arg[0], r"\u -> u * u")
  }

  fn pop_adj(&self, arg: &[(CellPtr, Clock)], _out: CellPtr, _out_clk: Clock, out_adj: CellPtr, arg_adj: &mut [CellPtr]) -> Result<FutharkThunkAdj, ThunkAdjErr> {
    // FIXME: scalar 2.
    arg_adj[0] += out_adj * (arg[0].0 + arg[0].0);
    /*
    let dtype = _;
    arg_adj[0] += out_adj * arg[0].0 * ScalarVal_::two(dtype);
    */
    Ok(FutharkThunkAdj::Spec)
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct RecipFutThunkSpec;

impl FutharkThunkSpec for RecipFutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.recip")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn arity(&self) -> Option<(u16, u16)> {
    Some((1, 1))
  }

  /*fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }*/

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(Dim{ndim: arg[0].ndim, dtype: arg[0].dtype})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(CellType{shape: arg[0].shape.clone(), dtype: arg[0].dtype})
  }

  fn gen_futhark(&self, /*abi: &mut FutAbi,*/ arg: &[Dim], _out: &[Dim]) -> Result<FutharkThunkGenCode, FutharkGenErr> {
    FutharkThunkGenCode::flat_map(arg[0], format!(r"{}.recip", arg[0].dtype.format_futhark()))
  }

  fn pop_adj(&self, arg: &[(CellPtr, Clock)], _out: CellPtr, _out_clk: Clock, out_adj: CellPtr, arg_adj: &mut [CellPtr]) -> Result<FutharkThunkAdj, ThunkAdjErr> {
    arg_adj[0] += out_adj / -arg[0].0.square();
    Ok(FutharkThunkAdj::Spec)
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct SqrtFutThunkSpec;

impl FutharkThunkSpec for SqrtFutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.sqrt")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn arity(&self) -> Option<(u16, u16)> {
    Some((1, 1))
  }

  /*fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }*/

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(Dim{ndim: arg[0].ndim, dtype: arg[0].dtype})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(CellType{shape: arg[0].shape.clone(), dtype: arg[0].dtype})
  }

  fn gen_futhark(&self, /*abi: &mut FutAbi,*/ arg: &[Dim], _out: &[Dim]) -> Result<FutharkThunkGenCode, FutharkGenErr> {
    FutharkThunkGenCode::flat_map(arg[0], format!(r"{}.sqrt", arg[0].dtype.format_futhark()))
  }

  fn pop_adj(&self, arg: &[(CellPtr, Clock)], _out: CellPtr, _out_clk: Clock, out_adj: CellPtr, arg_adj: &mut [CellPtr]) -> Result<FutharkThunkAdj, ThunkAdjErr> {
    let y = arg[0].0.sqrt();
    // FIXME: scalar 2.
    arg_adj[0] += out_adj / (y + y);
    Ok(FutharkThunkAdj::Spec)
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct RsqrtFutThunkSpec;

impl FutharkThunkSpec for RsqrtFutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.rsqrt")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn arity(&self) -> Option<(u16, u16)> {
    Some((1, 1))
  }

  /*fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }*/

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(Dim{ndim: arg[0].ndim, dtype: arg[0].dtype})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(CellType{shape: arg[0].shape.clone(), dtype: arg[0].dtype})
  }

  fn gen_futhark(&self, /*abi: &mut FutAbi,*/ arg: &[Dim], _out: &[Dim]) -> Result<FutharkThunkGenCode, FutharkGenErr> {
    // FIXME FIXME
    FutharkThunkGenCode::flat_map(arg[0], format!(r"\u -> {}.recip ({}.sqrt u)",
        arg[0].dtype.format_futhark(),
        arg[0].dtype.format_futhark(),
    ))
    //FutharkThunkGenCode::flat_map(arg[0], r"\u -> rsqrt u")
  }

  fn pop_adj(&self, arg: &[(CellPtr, Clock)], _out: CellPtr, _out_clk: Clock, out_adj: CellPtr, arg_adj: &mut [CellPtr]) -> Result<FutharkThunkAdj, ThunkAdjErr> {
    // FIXME FIXME
    unimplemented!();
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct CosFutThunkSpec;

impl FutharkThunkSpec for CosFutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.cos")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn arity(&self) -> Option<(u16, u16)> {
    Some((1, 1))
  }

  /*fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }*/

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(Dim{ndim: arg[0].ndim, dtype: arg[0].dtype})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(CellType{shape: arg[0].shape.clone(), dtype: arg[0].dtype})
  }

  fn gen_futhark(&self, /*abi: &mut FutAbi,*/ arg: &[Dim], _out: &[Dim]) -> Result<FutharkThunkGenCode, FutharkGenErr> {
    FutharkThunkGenCode::flat_map(arg[0], format!(r"{}.cos", arg[0].dtype.format_futhark()))
  }

  fn pop_adj(&self, arg: &[(CellPtr, Clock)], _out: CellPtr, _out_clk: Clock, out_adj: CellPtr, arg_adj: &mut [CellPtr]) -> Result<FutharkThunkAdj, ThunkAdjErr> {
    arg_adj[0] += out_adj * -arg[0].0.sin();
    Ok(FutharkThunkAdj::Spec)
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct SinFutThunkSpec;

impl FutharkThunkSpec for SinFutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.cos")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn arity(&self) -> Option<(u16, u16)> {
    Some((1, 1))
  }

  /*fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }*/

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(Dim{ndim: arg[0].ndim, dtype: arg[0].dtype})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(CellType{shape: arg[0].shape.clone(), dtype: arg[0].dtype})
  }

  fn gen_futhark(&self, /*abi: &mut FutAbi,*/ arg: &[Dim], _out: &[Dim]) -> Result<FutharkThunkGenCode, FutharkGenErr> {
    FutharkThunkGenCode::flat_map(arg[0], format!(r"{}.sin", arg[0].dtype.format_futhark()))
  }

  fn pop_adj(&self, arg: &[(CellPtr, Clock)], _out: CellPtr, _out_clk: Clock, out_adj: CellPtr, arg_adj: &mut [CellPtr]) -> Result<FutharkThunkAdj, ThunkAdjErr> {
    arg_adj[0] += out_adj * arg[0].0.cos();
    Ok(FutharkThunkAdj::Spec)
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct ExpFutThunkSpec;

impl FutharkThunkSpec for ExpFutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.exp")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn arity(&self) -> Option<(u16, u16)> {
    Some((1, 1))
  }

  /*fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }*/

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(Dim{ndim: arg[0].ndim, dtype: arg[0].dtype})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(CellType{shape: arg[0].shape.clone(), dtype: arg[0].dtype})
  }

  fn gen_futhark(&self, /*abi: &mut FutAbi,*/ arg: &[Dim], _out: &[Dim]) -> Result<FutharkThunkGenCode, FutharkGenErr> {
    FutharkThunkGenCode::flat_map(arg[0], format!(r"{}.exp", arg[0].dtype.format_futhark()))
  }

  fn pop_adj(&self, arg: &[(CellPtr, Clock)], _out: CellPtr, _out_clk: Clock, out_adj: CellPtr, arg_adj: &mut [CellPtr]) -> Result<FutharkThunkAdj, ThunkAdjErr> {
    arg_adj[0] += out_adj * arg[0].0.exp();
    Ok(FutharkThunkAdj::Spec)
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct LogisticFutThunkSpec;

impl FutharkThunkSpec for LogisticFutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.logistic")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn arity(&self) -> Option<(u16, u16)> {
    Some((1, 1))
  }

  /*fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }*/

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(arg[0])
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(arg[0].clone())
  }

  fn gen_futhark(&self, /*abi: &mut FutAbi,*/ arg: &[Dim], _out: &[Dim]) -> Result<FutharkThunkGenCode, FutharkGenErr> {
    FutharkThunkGenCode::flat_map(arg[0], format!(r"\u -> 1 / (1 + ({}.exp (-u)))",
        arg[0].dtype.format_futhark(),
    ))
  }

  fn pop_adj(&self, arg: &[(CellPtr, Clock)], _out: CellPtr, _out_clk: Clock, out_adj: CellPtr, arg_adj: &mut [CellPtr]) -> Result<FutharkThunkAdj, ThunkAdjErr> {
    let x = arg[0].0;
    let x_ty = x.type_();
    let y = x.logistic();
    arg_adj[0] += out_adj * (y * (ScalarVal_::one(x_ty.dtype) - y));
    Ok(FutharkThunkAdj::Spec)
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct StandardSiluFutThunkSpec;

impl FutharkThunkSpec for StandardSiluFutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.standard_silu")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn arity(&self) -> Option<(u16, u16)> {
    Some((1, 1))
  }

  /*fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }*/

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(arg[0])
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(arg[0].clone())
  }

  fn gen_futhark(&self, /*abi: &mut FutAbi,*/ arg: &[Dim], _out: &[Dim]) -> Result<FutharkThunkGenCode, FutharkGenErr> {
    FutharkThunkGenCode::flat_map(arg[0], format!(r"\u -> u / (1 + ({}.exp (-u)))", arg[0].dtype.format_futhark()))
  }

  fn pop_adj(&self, arg: &[(CellPtr, Clock)], _out: CellPtr, _out_clk: Clock, out_adj: CellPtr, arg_adj: &mut [CellPtr]) -> Result<FutharkThunkAdj, ThunkAdjErr> {
    let x = arg[0].0;
    let x_ty = x.type_();
    let y = x.standard_silu();
    let t = x.logistic();
    arg_adj[0] += out_adj * (y * (ScalarVal_::one(x_ty.dtype) - t) + t);
    Ok(FutharkThunkAdj::Spec)
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct TanhFutThunkSpec;

impl FutharkThunkSpec for TanhFutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.tanh")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn arity(&self) -> Option<(u16, u16)> {
    Some((1, 1))
  }

  /*fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }*/

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(Dim{ndim: arg[0].ndim, dtype: arg[0].dtype})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(CellType{shape: arg[0].shape.clone(), dtype: arg[0].dtype})
  }

  fn gen_futhark(&self, /*abi: &mut FutAbi,*/ arg: &[Dim], _out: &[Dim]) -> Result<FutharkThunkGenCode, FutharkGenErr> {
    FutharkThunkGenCode::flat_map(arg[0], format!(r"\u -> (({}.exp u) - ({}.exp (-u))) / (({}.exp u) + ({}.exp (-u)))",
        arg[0].dtype.format_futhark(),
        arg[0].dtype.format_futhark(),
        arg[0].dtype.format_futhark(),
        arg[0].dtype.format_futhark(),
    ))
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct PowiF32FutThunkSpec { pub exp: i64 }

impl FutharkThunkSpec for PowiF32FutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.f32.powi")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn arity(&self) -> Option<(u16, u16)> {
    Some((1, 1))
  }

  /*fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }*/

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(Dim{ndim: arg[0].ndim, dtype: f32::dtype_()})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(CellType{shape: arg[0].shape.clone(), dtype: arg[0].dtype})
  }

  fn gen_futhark(&self, /*abi: &mut FutAbi,*/ arg: &[Dim], _out: &[Dim]) -> Result<FutharkThunkGenCode, FutharkGenErr> {
    // FIXME FIXME
    FutharkThunkGenCode::flat_map(arg[0], format!(r"\u -> u ** (f32.i64 {})", self.exp))
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct LogFutThunkSpec;

impl FutharkThunkSpec for LogFutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.log")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn arity(&self) -> Option<(u16, u16)> {
    Some((1, 1))
  }

  /*fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }*/

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(Dim{ndim: arg[0].ndim, dtype: arg[0].dtype})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(CellType{shape: arg[0].shape.clone(), dtype: arg[0].dtype})
  }

  fn gen_futhark(&self, /*abi: &mut FutAbi,*/ arg: &[Dim], _out: &[Dim]) -> Result<FutharkThunkGenCode, FutharkGenErr> {
    FutharkThunkGenCode::flat_map(arg[0], format!(r"{}.log", arg[0].dtype.format_futhark()))
  }

  fn pop_adj(&self, arg: &[(CellPtr, Clock)], _out: CellPtr, _out_clk: Clock, out_adj: CellPtr, arg_adj: &mut [CellPtr]) -> Result<FutharkThunkAdj, ThunkAdjErr> {
    arg_adj[0] += out_adj / arg[0].0;
    Ok(FutharkThunkAdj::Spec)
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct FlatSumFutThunkSpec;

impl FutharkThunkSpec for FlatSumFutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.flat_sum")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn arity(&self) -> Option<(u16, u16)> {
    Some((1, 1))
  }

  /*fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }*/

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(Dim{ndim: 0, dtype: arg[0].dtype})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(CellType{shape: Vec::new(), dtype: arg[0].dtype})
  }

  fn gen_futhark(&self, /*abi: &mut FutAbi,*/ arg: &[Dim], _out: &[Dim]) -> Result<FutharkThunkGenCode, FutharkGenErr> {
    //abi.set_out_arr(0, AbiOutput::Pure, AbiArrayRepr::Nd, AbiScalarType::Unspec);
    //abi.set_arg_arr(0, AbiInput::Shared, AbiArrayRepr::Nd, AbiScalarType::Unspec);
    /*match arg[0].ndim {
      0 => {
        let mut code = FutharkThunkGenCode::default();
        code.append(format!(r"let {{%1}} = {{%0}} in"));
        code.into()
      }
      1 => {
        let mut code = FutharkThunkGenCode::default();
        code.append(format!(r"let {{%1}} = reduce (+) 0 {{%0}} in"));
        code.into()
      }
      2 => {
        let mut code = FutharkThunkGenCode::default();
        code.append(format!(r"let t0 = flatten {{%0}} in"));
        code.append(format!(r"let t1 = reduce (+) 0 t0 in"));
        code.append(format!(r"let {{%1}} = t1 in"));
        code.into()
      }
      3 => {
        let mut code = FutharkThunkGenCode::default();
        code.append(format!(r"let t0 = flatten_3d {{%0}} in"));
        code.append(format!(r"let t1 = reduce (+) 0 t0 in"));
        code.append(format!(r"let {{%1}} = t1 in"));
        code.into()
      }
      4 => {
        let mut code = FutharkThunkGenCode::default();
        code.append(format!(r"let t0 = flatten_4d {{%0}} in"));
        code.append(format!(r"let t1 = reduce (+) 0 t0 in"));
        code.append(format!(r"let {{%1}} = t1 in"));
        code.into()
      }
      _ => {
        unimplemented!();
      }
    }*/
    let mut code = FutharkThunkGenCode::default();
    code.abi.arityout = 1;
    code.abi.set_out(0, FutharkArrayRepr::Flat);
    code.abi.arityin = 1;
    code.abi.set_arg(0, FutharkArrayRepr::Flat);
    if arg[0].ndim == 0 {
      code.append(format!(r"let {{%1}} = {{%0}} in"));
    } else {
      code.append(format!(r"let {{%1}} = reduce (+) 0 {{%0}} in"));
    }
    code.into()
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct InnerMeanFutThunkSpec;

impl FutharkThunkSpec for InnerMeanFutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.inner_mean")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn arity(&self) -> Option<(u16, u16)> {
    Some((1, 1))
  }

  /*fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }*/

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    let mut ndim = arg[0].ndim();
    if ndim <= 0 {
      return Err(ThunkDimErr::_Bot);
    }
    ndim -= 1;
    let dtype = arg[0].dtype;
    Ok(Dim{ndim, dtype})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    let ndim = arg[0].ndim();
    if ndim <= 0 {
      return Err(ThunkTypeErr::_Bot);
    }
    let mut shape = arg[0].shape.clone();
    let _ = shape.pop();
    let dtype = arg[0].dtype;
    Ok(CellType{shape, dtype})
  }

  fn gen_futhark(&self, /*abi: &mut FutAbi,*/ arg: &[Dim], out: &[Dim]) -> Result<FutharkThunkGenCode, FutharkGenErr> {
    //abi.set_out_arr(0, AbiOutput::Pure, AbiArrayRepr::Nd, AbiScalarType::Unspec);
    //abi.set_arg_arr(0, AbiInput::Shared, AbiArrayRepr::Nd, AbiScalarType::Unspec);
    //let out = FutharkThunkSpec::out_dim(self, arg).map_err(|e| e.into_gen())?;
    let mut code = FutharkThunkGenCode::default();
    code.abi.arityout = 1;
    code.abi.set_out(0, FutharkArrayRepr::Nd);
    code.abi.arityin = 1;
    code.abi.set_arg(0, FutharkArrayRepr::Nd);
    match out[0].ndim {
      0 => {
        code.append(format!(r"let {{%1}} = {{%0}} in"));
      }
      1 => {
        code.cfg.emit_arg_shapes = true;
        code.append(format!(r"let a_suf = {{%0.s[1]}} in"));
        code.append(format!(r"let t0 = {{%0}} in"));
        code.append(format!(r"let t1 = map (\t -> (reduce (+) 0 t) / ({}.i64 a_suf)) t0 in",
            arg[0].dtype.format_futhark(),
        ));
        code.append(format!(r"let {{%1}} = t1 in"));
      }
      2 => {
        code.cfg.emit_arg_shapes = true;
        code.append(format!(r"let a_pre = {{%0.s[0]}} * {{%0.s[1]}} in"));
        code.append(format!(r"let a_suf = {{%0.s[2]}} in"));
        code.append(format!(r"let t0 = flatten_3d {{%0}} :> [a_pre * a_suf]{} in",
            arg[0].dtype.format_futhark(),
        ));
        code.append(format!(r"let t0 = unflatten t0 in"));
        code.append(format!(r"let t1 = map (\t -> (reduce (+) 0 t) / ({}.i64 a_suf)) t0 in",
            arg[0].dtype.format_futhark(),
        ));
        code.append(format!(r"let {{%1}} = unflatten (t1 :> [{{%0.s[0]}} * {{%0.s[1]}}]{}) in",
            arg[0].dtype.format_futhark(),
        ));
      }
      3 => {
        code.cfg.emit_arg_shapes = true;
        code.append(format!(r"let a_pre = {{%0.s[0]}} * {{%0.s[1]}} * {{%0.s[2]}} in"));
        code.append(format!(r"let a_suf = {{%0.s[3]}} in"));
        code.append(format!(r"let t0 = flatten_4d {{%0}} :> [a_pre * a_suf]{} in",
            arg[0].dtype.format_futhark(),
        ));
        //code.append(format!(r"let t0 = unflatten a_pre a_suf t0 in"));
        code.append(format!(r"let t0 = unflatten t0 in"));
        code.append(format!(r"let t1 = map (\t -> (reduce (+) 0 t) / ({}.i64 a_suf)) t0 in",
            arg[0].dtype.format_futhark(),
        ));
        //code.append(format!(r"let {{%1}} = unflatten_3d {{%0.s[0]}} {{%0.s[1]}} {{%0.s[2]}} t1 in"));
        code.append(format!(r"let {{%1}} = unflatten_3d (t1 :> [{{%0.s[0]}} * {{%0.s[1]}} * {{%0.s[2]}}]{}) in",
            arg[0].dtype.format_futhark(),
        ));
      }
      _ => {
        unimplemented!();
      }
    }
    code.into()
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct InnerSoftmaxFutThunkSpec;

impl FutharkThunkSpec for InnerSoftmaxFutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.inner_softmax")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn arity(&self) -> Option<(u16, u16)> {
    Some((1, 1))
  }

  /*fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }*/

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(arg[0].clone())
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(arg[0].clone())
  }

  fn gen_futhark(&self, /*abi: &mut FutAbi,*/ arg: &[Dim], out: &[Dim]) -> Result<FutharkThunkGenCode, FutharkGenErr> {
    //abi.set_out_arr(0, AbiOutput::Pure, AbiArrayRepr::Nd, AbiScalarType::Unspec);
    //abi.set_arg_arr(0, AbiInput::Shared, AbiArrayRepr::Nd, AbiScalarType::Unspec);
    //let out = FutharkThunkSpec::out_dim(self, arg).map_err(|e| e.into_gen())?;
    let mut code = FutharkThunkGenCode::default();
    code.abi.arityout = 1;
    code.abi.set_out(0, FutharkArrayRepr::Nd);
    code.abi.arityin = 1;
    code.abi.set_arg(0, FutharkArrayRepr::Nd);
    match out[0].ndim {
      0 => {
        unimplemented!();
      }
      1 => {
        unimplemented!();
      }
      2 => {
        code.cfg.emit_arg_shapes = true;
        code.append(format!(r"let a_pre = {{%0.s[0]}} in"));
        code.append(format!(r"let a_suf = {{%0.s[1]}} in"));
        code.append(format!(r"let t0 = flatten {{%0}} :> [a_pre * a_suf]{} in",
            arg[0].dtype.format_futhark(),
        ));
        code.append(format!(r"let t0 = unflatten t0 in"));
        code.append(format!(r"let t0_max = map (\t -> reduce ({}.max) (-{}.inf) t) t0 in",
            arg[0].dtype.format_futhark(),
            arg[0].dtype.format_futhark(),
        ));
        code.append(format!(r"let t1 = map2 (\t t_max -> map (\u -> ({}.exp) (u - t_max)) t) t0 t0_max in",
            arg[0].dtype.format_futhark(),
        ));
        code.append(format!(r"let t1_sum = map (\t -> reduce (+) 0 t) t1 in"));
        code.append(format!(r"let t2 = map2 (\t t_sum -> map (/ t_sum) t) t1 t1_sum in"));
        code.append(format!(r"let t2 = flatten t2 in"));
        code.append(format!(r"let {{%1}} = unflatten (t2 :> [{{%0.s[0]}} * {{%0.s[1]}}]{}) in",
            arg[0].dtype.format_futhark(),
        ));
      }
      3 => {
        code.cfg.emit_arg_shapes = true;
        code.append(format!(r"let a_pre = {{%0.s[0]}} * {{%0.s[1]}} in"));
        code.append(format!(r"let a_suf = {{%0.s[2]}} in"));
        code.append(format!(r"let t0 = flatten_3d {{%0}} :> [a_pre * a_suf]{} in",
            arg[0].dtype.format_futhark(),
        ));
        code.append(format!(r"let t0 = unflatten t0 in"));
        code.append(format!(r"let t0_max = map (\t -> reduce ({}.max) (-{}.inf) t) t0 in",
            arg[0].dtype.format_futhark(),
            arg[0].dtype.format_futhark(),
        ));
        code.append(format!(r"let t1 = map2 (\t t_max -> map (\u -> ({}.exp) (u - t_max)) t) t0 t0_max in",
            arg[0].dtype.format_futhark(),
        ));
        code.append(format!(r"let t1_sum = map (\t -> reduce (+) 0 t) t1 in"));
        code.append(format!(r"let t2 = map2 (\t t_sum -> map (/ t_sum) t) t1 t1_sum in"));
        code.append(format!(r"let t2 = flatten t2 in"));
        code.append(format!(r"let {{%1}} = unflatten_3d (t2 :> [{{%0.s[0]}} * {{%0.s[1]}} * {{%0.s[2]}}]{}) in",
            arg[0].dtype.format_futhark(),
        ));
      }
      4 => {
        code.cfg.emit_arg_shapes = true;
        code.append(format!(r"let a_pre = {{%0.s[0]}} * {{%0.s[1]}} * {{%0.s[2]}} in"));
        code.append(format!(r"let a_suf = {{%0.s[3]}} in"));
        code.append(format!(r"let t0 = flatten_4d {{%0}} :> [a_pre * a_suf]{} in",
            arg[0].dtype.format_futhark(),
        ));
        code.append(format!(r"let t0 = unflatten t0 in"));
        code.append(format!(r"let t0_max = map (\t -> reduce ({}.max) (-{}.inf) t) t0 in",
            arg[0].dtype.format_futhark(),
            arg[0].dtype.format_futhark(),
        ));
        code.append(format!(r"let t1 = map2 (\t t_max -> map (\u -> ({}.exp) (u - t_max)) t) t0 t0_max in",
            arg[0].dtype.format_futhark(),
        ));
        code.append(format!(r"let t1_sum = map (\t -> reduce (+) 0 t) t1 in"));
        code.append(format!(r"let t2 = map2 (\t t_sum -> map (/ t_sum) t) t1 t1_sum in"));
        code.append(format!(r"let t2 = flatten t2 in"));
        code.append(format!(r"let {{%1}} = unflatten_4d (t2 :> [{{%0.s[0]}} * {{%0.s[1]}} * {{%0.s[2]}} * {{%0.s[3]}}]{}) in",
            arg[0].dtype.format_futhark(),
        ));
      }
      _ => {
        unimplemented!();
      }
    }
    code.into()
  }

  fn pop_adj(&self, arg: &[(CellPtr, Clock)], _out: CellPtr, _out_clk: Clock, out_adj: CellPtr, arg_adj: &mut [CellPtr]) -> Result<FutharkThunkAdj, ThunkAdjErr> {
    // FIXME
    let x = arg[0].0;
    let y = x.inner_softmax();
    arg_adj[0] += inner_softmax_post_adj(y, out_adj);
    Ok(FutharkThunkAdj::Spec)
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct InnerSoftmaxPostAdjFutThunkSpec;

impl FutharkThunkSpec for InnerSoftmaxPostAdjFutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.inner_softmax_post_adj")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn arity(&self) -> Option<(u16, u16)> {
    Some((2, 1))
  }

  /*fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 2;
    abi.arityout = 1;
    abi
  }*/

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    // FIXME
    Ok(arg[0].clone())
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    // FIXME
    Ok(arg[0].clone())
  }

  fn gen_futhark(&self, /*abi: &mut FutAbi,*/ arg: &[Dim], out: &[Dim]) -> Result<FutharkThunkGenCode, FutharkGenErr> {
    //abi.set_out_arr(0, AbiOutput::Pure, AbiArrayRepr::Nd, AbiScalarType::Unspec);
    //abi.set_arg_arr(0, AbiInput::Shared, AbiArrayRepr::Nd, AbiScalarType::Unspec);
    //abi.set_arg_arr(1, AbiInput::Shared, AbiArrayRepr::Nd, AbiScalarType::Unspec);
    //let out = FutharkThunkSpec::out_dim(self, arg).map_err(|e| e.into_gen())?;
    let mut code = FutharkThunkGenCode::default();
    code.abi.arityout = 1;
    code.abi.set_out(0, FutharkArrayRepr::Nd);
    code.abi.arityin = 2;
    code.abi.set_arg(0, FutharkArrayRepr::Nd);
    code.abi.set_arg(1, FutharkArrayRepr::Nd);
    match out[0].ndim {
      // TODO
      4 => {
        code.cfg.emit_arg_shapes = true;
        code.append(format!(r"let a_pre = {{%0.s[0]}} * {{%0.s[1]}} * {{%0.s[2]}} in"));
        code.append(format!(r"let a_suf = {{%0.s[3]}} in"));
        code.append(format!(r"let t0 = flatten_4d {{%0}} :> [a_pre * a_suf]{} in",
            arg[0].dtype.format_futhark(),
        ));
        code.append(format!(r"let t0 = unflatten t0 in"));
        code.append(format!(r"let t1 = flatten_4d {{%1}} :> [a_pre * a_suf]{} in",
            arg[1].dtype.format_futhark(),
        ));
        code.append(format!(r"let t1 = unflatten t1 in"));
        code.append(format!(r"let t2 = map2 (\y dy -> map2 (*) y dy) t0 t1 in"));
        code.append(format!(r"let t2_sum = map (\t -> reduce (+) 0 t) t2 in"));
        code.append(format!(r"let t3 = map3 (\t t_sum y -> map2 (\u v -> u - t_sum * v) t y) t2 t2_sum t0 in"));
        code.append(format!(r"let t3 = flatten t3 :> [{{%0.s[0]}} * {{%0.s[1]}} * {{%0.s[2]}} * {{%0.s[3]}}]{} in",
            arg[0].dtype.format_futhark(),
        ));
        code.append(format!(r"let {{%2}} = unflatten_4d t3 in"));
      }
      _ => {
        unimplemented!();
      }
    }
    code.into()
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct InnerSoftmaxCategoricalNLLFutThunkSpec;

impl FutharkThunkSpec for InnerSoftmaxCategoricalNLLFutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.inner_softmax_categorical_nll")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn arity(&self) -> Option<(u16, u16)> {
    Some((2, 1))
  }

  /*fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 2;
    abi.arityout = 1;
    abi
  }*/

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    if !arg[1].dtype.is_uint() {
      return Err(ThunkDimErr::_Bot);
    }
    if arg[0].ndim() != arg[1].ndim() + 1 {
      return Err(ThunkDimErr::_Bot);
    }
    Ok(Dim{ndim: arg[1].ndim(), dtype: arg[0].dtype})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    if !arg[1].dtype.is_uint() {
      return Err(ThunkTypeErr::_Bot);
    }
    if arg[0].ndim() != arg[1].ndim() + 1 {
      return Err(ThunkTypeErr::_Bot);
    }
    Ok(CellType{shape: arg[1].shape.clone(), dtype: arg[0].dtype})
  }

  fn gen_futhark(&self, /*abi: &mut FutAbi,*/ arg: &[Dim], out: &[Dim]) -> Result<FutharkThunkGenCode, FutharkGenErr> {
    //abi.set_out_arr(0, AbiOutput::Pure, AbiArrayRepr::Nd, AbiScalarType::Unspec);
    //abi.set_arg_arr(0, AbiInput::Shared, AbiArrayRepr::Nd, AbiScalarType::Unspec);
    //abi.set_arg_arr(1, AbiInput::Shared, AbiArrayRepr::Nd, AbiScalarType::Unspec);
    //let out = FutharkThunkSpec::out_dim(self, arg).map_err(|e| e.into_gen())?;
    let mut code = FutharkThunkGenCode::default();
    code.abi.arityout = 1;
    code.abi.set_out(0, FutharkArrayRepr::Nd);
    code.abi.arityin = 2;
    code.abi.set_arg(0, FutharkArrayRepr::Nd);
    code.abi.set_arg(1, FutharkArrayRepr::Nd);
    match out[0].ndim {
      0 => {
        unimplemented!();
      }
      1 => {
        unimplemented!();
      }
      2 => {
        code.cfg.emit_arg_shapes = true;
        code.append(format!(r"let a_pre = {{%0.s[0]}} * {{%0.s[1]}} in"));
        code.append(format!(r"let a_suf = {{%0.s[2]}} in"));
        code.append(format!(r"let t0 = flatten_3d {{%0}} :> [a_pre * a_suf]{} in",
            arg[0].dtype.format_futhark(),
        ));
        code.append(format!(r"let t0 = unflatten t0 in"));
        code.append(format!(r"let t0_max = map (\t -> reduce ({}.max) (-{}.inf) t) t0 in",
            arg[0].dtype.format_futhark(),
            arg[0].dtype.format_futhark(),
        ));
        code.append(format!(r"let t1 = map2 (\t t_max -> map (\u -> ({}.exp) (u - t_max)) t) t0 t0_max in",
            arg[0].dtype.format_futhark(),
        ));
        code.append(format!(r"let t1_sum = map (\t -> reduce (+) 0 t) t1 in"));
        code.append(format!(r"let t_val = map2 (\t t_sum -> map (/ t_sum) t) t1 t1_sum in"));
        code.append(format!(r"let t_key = flatten {{%1}} :> [a_pre]{} in", arg[1].dtype.format_futhark()));
        code.append(format!(r"let t_key = map (\k -> let k = (i64.{} k) in (assert (k >= 0 && k < a_suf) k)) t_key in",
            arg[1].dtype.format_futhark(),
        ));
        code.append(format!(r"let t2 = map2 (\k v -> -({}.log v[k])) t_key t_val in",
            arg[0].dtype.format_futhark(),
        ));
        code.append(format!(r"let {{%2}} = unflatten (t2 :> [{{%0.s[0]}} * {{%0.s[1]}}]{}) in",
            arg[0].dtype.format_futhark(),
        ));
      }
      _ => {
        unimplemented!();
      }
    }
    code.into()
  }

  fn pop_adj(&self, arg: &[(CellPtr, Clock)], _out: CellPtr, _out_clk: Clock, out_adj: CellPtr, arg_adj: &mut [CellPtr]) -> Result<FutharkThunkAdj, ThunkAdjErr> {
    let x = arg[0].0;
    let x_ty = x.type_();
    let x_nd = x_ty.ndim() as usize;
    let inner_len = x_ty.shape[x_nd - 1];
    let mut dy_shape = x_ty.shape.clone();
    dy_shape[x_nd - 1] = 1;
    let dtype = x_ty.dtype;
    let y = x.inner_softmax();
    let rank = arg[1].0;
    let target = rank.inner_one_hot(inner_len, dtype);
    arg_adj[0] += out_adj.new_shape(dy_shape) * (y - target);
    Ok(FutharkThunkAdj::Spec)
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct InnerConcatFutThunkSpec;

impl FutharkThunkSpec for InnerConcatFutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.inner_concat")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn arity(&self) -> Option<(u16, u16)> {
    Some((2, 1))
  }

  /*fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 2;
    abi.arityout = 1;
    abi
  }*/

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    if arg[0].ndim() != arg[1].ndim() {
      return Err(ThunkDimErr::_Bot);
    }
    if arg[0].dtype != arg[1].dtype {
      return Err(ThunkDimErr::_Bot);
    }
    let ndim = max(1, arg[0].ndim());
    let dtype = arg[0].dtype;
    Ok(Dim{ndim, dtype})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    if arg[0].ndim() != arg[1].ndim() {
      return Err(ThunkTypeErr::_Bot);
    }
    if arg[0].dtype != arg[1].dtype {
      return Err(ThunkTypeErr::_Bot);
    }
    let nd = arg[0].ndim() as usize;
    let mut shape = arg[0].shape.clone();
    if nd == 0 {
      shape.push(2);
    } else {
      shape[nd - 1] += arg[1].shape[nd - 1];
    }
    let dtype = arg[0].dtype;
    Ok(CellType{shape, dtype})
  }

  fn gen_futhark(&self, /*abi: &mut FutAbi,*/ arg: &[Dim], _out: &[Dim]) -> Result<FutharkThunkGenCode, FutharkGenErr> {
    //abi.set_out_arr(0, AbiOutput::Pure, AbiArrayRepr::Nd, AbiScalarType::Unspec);
    //abi.set_arg_arr(0, AbiInput::Shared, AbiArrayRepr::Nd, AbiScalarType::Unspec);
    //abi.set_arg_arr(1, AbiInput::Shared, AbiArrayRepr::Nd, AbiScalarType::Unspec);
    let mut code = FutharkThunkGenCode::default();
    code.abi.arityout = 1;
    code.abi.set_out(0, FutharkArrayRepr::Nd);
    code.abi.arityin = 2;
    code.abi.set_arg(0, FutharkArrayRepr::Nd);
    code.abi.set_arg(1, FutharkArrayRepr::Nd);
    match (arg[0].ndim(), arg[1].ndim()) {
      (0, 0) => {
        code.append(format!(r"let {{%2}} = [{{%0}}, {{%1}}] in"));
      }
      (1, 1) => {
        code.append(format!(r"let {{%2}} = {{%0}} ++ {{%1}} in"));
      }
      (2, 2) => {
        code.append(format!(r"let {{%2}} = map2 (\tl tr -> tl ++ tr) {{%0}} {{%1}} in"));
      }
      (3, 3) => {
        code.append(format!(r"let {{%2}} = map2 (\tl1 tr1 -> map2 (\tl2 tr2 -> tl2 ++ tr2) tl1 tr1) {{%0}} {{%1}} in"));
      }
      (4, 4) => {
        code.append(format!(r"let {{%2}} = map2 (\tl1 tr1 -> map2 (\tl2 tr2 -> map2 (\tl3 tr3 -> tl3 ++ tr3) tl2 tr2) tl1 tr1) {{%0}} {{%1}} in"));
      }
      _ => {
        unimplemented!();
      }
    }
    code.into()
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct InnerTransposeFutThunkSpec;

impl FutharkThunkSpec for InnerTransposeFutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.inner_transpose")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn arity(&self) -> Option<(u16, u16)> {
    Some((1, 1))
  }

  /*fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }*/

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    if arg[0].ndim() < 2 {
      return Err(ThunkDimErr::_Bot);
    }
    Ok(Dim{ndim: arg[0].ndim, dtype: arg[0].dtype})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    if arg[0].ndim() < 2 {
      return Err(ThunkTypeErr::_Bot);
    }
    let nd = arg[0].ndim() as usize;
    let mut shape = arg[0].shape.clone();
    shape.swap(nd - 2, nd - 1);
    Ok(CellType{shape, dtype: arg[0].dtype})
  }

  fn gen_futhark(&self, /*abi: &mut FutAbi,*/ arg: &[Dim], _out: &[Dim]) -> Result<FutharkThunkGenCode, FutharkGenErr> {
    //abi.set_out_arr(0, AbiOutput::Pure, AbiArrayRepr::Nd, AbiScalarType::Unspec);
    //abi.set_arg_arr(0, AbiInput::Shared, AbiArrayRepr::Nd, AbiScalarType::Unspec);
    let mut code = FutharkThunkGenCode::default();
    code.abi.arityout = 1;
    code.abi.set_out(0, FutharkArrayRepr::Nd);
    code.abi.arityin = 1;
    code.abi.set_arg(0, FutharkArrayRepr::Nd);
    match arg[0].ndim() {
      0 | 1 => panic!("bug"),
      2 => {
        code.append(format!(r"let {{%1}} = transpose {{%0}} in"));
      }
      3 => {
        code.append(format!(r"let {{%1}} = map (\t -> transpose t) {{%0}} in"));
      }
      4 => {
        code.append(format!(r"let {{%1}} = map (\t1 -> map (\t2 -> transpose t2) t1) {{%0}} in"));
      }
      _ => {
        unimplemented!();
      }
    }
    code.into()
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct OnlineAddScale2InitFutThunkSpec {
  pub src_scale: ScalarVal_,
  pub dst_scale: ScalarVal_,
}

impl FutharkThunkSpec for OnlineAddScale2InitFutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.online_add_scale2.init")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn arity(&self) -> Option<(u16, u16)> {
    Some((1, 1))
  }

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    //Ok(arg[0])
    Err(ThunkDimErr::Nondeterm)
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    //Ok(arg[0].clone())
    Err(ThunkTypeErr::Nondeterm)
  }

  fn gen_futhark(&self, arg: &[Dim], out: &[Dim]) -> Result<FutharkThunkGenCode, FutharkGenErr> {
    FutharkThunkGenCode::flat_map2_(r"{%0}", arg[0], r"{%1}", out[0], r"{%1}",
        format!(r"\u v -> {} * v + {} * ({}.{} u)",
            self.dst_scale.format_futhark(),
            self.src_scale.format_futhark(),
            out[0].dtype.format_futhark(),
            arg[0].dtype.format_futhark(),
        )
    )
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct OnlineAddSquareScale2InitFutThunkSpec {
  pub src_scale: ScalarVal_,
  pub dst_scale: ScalarVal_,
}

impl FutharkThunkSpec for OnlineAddSquareScale2InitFutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.online_add_square_scale2.init")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn arity(&self) -> Option<(u16, u16)> {
    Some((1, 1))
  }

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    //Ok(arg[0])
    Err(ThunkDimErr::Nondeterm)
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    //Ok(arg[0].clone())
    Err(ThunkTypeErr::Nondeterm)
  }

  fn gen_futhark(&self, arg: &[Dim], out: &[Dim]) -> Result<FutharkThunkGenCode, FutharkGenErr> {
    if arg[0].ndim() != out[0].ndim() {
      return Err(ThunkDimErr::_Bot.into_gen());
    }
    FutharkThunkGenCode::flat_map2_(r"{%0}", arg[0], r"{%1}", out[0], r"{%1}",
        format!(r"\u v -> {} * v + ({} * ({}.{} u)) * ({} * ({}.{} u))",
            self.dst_scale.format_futhark(),
            self.src_scale.format_futhark(),
            out[0].dtype.format_futhark(),
            arg[0].dtype.format_futhark(),
            self.src_scale.format_futhark(),
            out[0].dtype.format_futhark(),
            arg[0].dtype.format_futhark(),
        )
    )
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct OnlineAverageScaleInitFutThunkSpec {
  pub src_scale: ScalarVal_,
  pub rate: ScalarVal_,
}

impl FutharkThunkSpec for OnlineAverageScaleInitFutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.online_average_scale.init")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn arity(&self) -> Option<(u16, u16)> {
    Some((1, 1))
  }

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    //Ok(arg[0])
    Err(ThunkDimErr::Nondeterm)
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    //Ok(arg[0].clone())
    Err(ThunkTypeErr::Nondeterm)
  }

  fn gen_futhark(&self, /*abi: &mut FutAbi,*/ arg: &[Dim], out: &[Dim]) -> Result<FutharkThunkGenCode, FutharkGenErr> {
    if arg[0].ndim() != out[0].ndim() {
      return Err(ThunkDimErr::_Bot.into_gen());
    }
    FutharkThunkGenCode::flat_map2_(r"{%0}", arg[0], r"{%1}", out[0], r"{%1}",
        format!(r"\u v -> v + {} * ({} * ({}.{} u) - v)",
            self.rate.format_futhark(),
            self.src_scale.format_futhark(),
            out[0].dtype.format_futhark(),
            arg[0].dtype.format_futhark(),
        )
    )
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct OnlineAverageSquareScaleInitFutThunkSpec {
  pub src_scale: ScalarVal_,
  pub rate: ScalarVal_,
}

impl FutharkThunkSpec for OnlineAverageSquareScaleInitFutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.online_average_square_scale.init")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn arity(&self) -> Option<(u16, u16)> {
    Some((1, 1))
  }

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    //Ok(arg[0])
    Err(ThunkDimErr::Nondeterm)
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    //Ok(arg[0].clone())
    Err(ThunkTypeErr::Nondeterm)
  }

  fn gen_futhark(&self, /*abi: &mut FutAbi,*/ arg: &[Dim], out: &[Dim]) -> Result<FutharkThunkGenCode, FutharkGenErr> {
    if arg[0].ndim() != out[0].ndim() {
      return Err(ThunkDimErr::_Bot.into_gen());
    }
    FutharkThunkGenCode::flat_map2_(r"{%0}", arg[0], r"{%1}", out[0], r"{%1}",
        format!(r"\u v -> v + {} * (({} * ({}.{} u)) * ({} * ({}.{} u)) - v)",
            self.rate.format_futhark(),
            self.src_scale.format_futhark(),
            out[0].dtype.format_futhark(),
            arg[0].dtype.format_futhark(),
            self.src_scale.format_futhark(),
            out[0].dtype.format_futhark(),
            arg[0].dtype.format_futhark(),
        )
    )
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct OnlineAdamWUpdateInitFutThunkSpec {
  pub signed_lr: ScalarVal_,
  pub lamda: ScalarVal_,
  pub eps: ScalarVal_,
}

impl FutharkThunkSpec for OnlineAdamWUpdateInitFutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.online_adamw_update.init")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn arity(&self) -> Option<(u16, u16)> {
    Some((2, 1))
  }

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(arg[0])
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(arg[0].clone())
  }

  fn gen_futhark(&self, /*abi: &mut FutAbi,*/ arg: &[Dim], out: &[Dim]) -> Result<FutharkThunkGenCode, FutharkGenErr> {
    //abi.set_out_arr(0, AbiOutput::Pure, AbiArrayRepr::Nd, AbiScalarType::Unspec);
    //abi.set_out_arr(0, AbiOutput::ImplicitInPlace, AbiArrayRepr::Nd, AbiScalarType::Unspec);
    //abi.set_arg_arr(0, AbiInput::Shared, AbiArrayRepr::Nd, AbiScalarType::Unspec);
    //abi.set_arg_arr(1, AbiInput::Shared, AbiArrayRepr::Nd, AbiScalarType::Unspec);
    // FIXME: params.
    FutharkThunkGenCode::flat_map3_(r"{%0}", arg[0], r"{%1}", arg[1], r"{%2}", out[0], r"{%2}",
        format!(r"\u v w -> (1 - {}) * w + {} * (u / (({}.sqrt v) + {}))",
            self.lamda.format_futhark(),
            self.signed_lr.format_futhark(),
            arg[1].dtype.format_futhark(),
            self.eps.format_futhark(),
        )
    )
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct BlockPadFutThunkSpec {
  pub org_block: [i64; 2],
  pub new_block: [i64; 2],
  pub pad_val: ScalarVal_,
}

impl FutharkThunkSpec for BlockPadFutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.block_pad")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn arity(&self) -> Option<(u16, u16)> {
    Some((1, 1))
  }

  /*fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }*/

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    if arg[0].ndim() < 3 {
      return Err(ThunkDimErr::_Bot);
    }
    if arg[0].dtype != self.pad_val.dtype() && !self.pad_val.is_bot() {
      return Err(ThunkDimErr::_Bot);
    }
    Ok(arg[0])
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    if arg[0].ndim() < 3 {
      return Err(ThunkTypeErr::_Bot);
    }
    if arg[0].dtype != self.pad_val.dtype() && !self.pad_val.is_bot() {
      return Err(ThunkTypeErr::_Bot);
    }
    let nd = arg[0].ndim() as usize;
    let mut shape = arg[0].shape.clone();
    assert_eq!(shape[nd - 3], self.org_block[0]);
    assert_eq!(shape[nd - 1], self.org_block[1]);
    shape[nd - 3] = self.new_block[0];
    shape[nd - 1] = self.new_block[1];
    Ok(CellType{shape, dtype: arg[0].dtype})
  }

  fn gen_futhark(&self, /*abi: &mut FutAbi,*/ arg: &[Dim], out: &[Dim]) -> Result<FutharkThunkGenCode, FutharkGenErr> {
    //abi.set_out_arr(0, AbiOutput::Pure, AbiArrayRepr::Nd, AbiScalarType::Unspec);
    //abi.set_arg_arr(0, AbiInput::Shared, AbiArrayRepr::Nd, AbiScalarType::Unspec);
    //let out = FutharkThunkSpec::out_dim(self, arg).map_err(|e| e.into_gen())?;
    let pad_outer = self.new_block[0] - self.org_block[0];
    let pad_inner = self.new_block[1] - self.org_block[1];
    let mut code = FutharkThunkGenCode::default();
    code.abi.arityout = 1;
    code.abi.set_out(0, FutharkArrayRepr::Nd);
    code.abi.arityin = 1;
    code.abi.set_arg(0, FutharkArrayRepr::Nd);
    match out[0].ndim {
      4 => {
        if pad_outer == 0 && pad_inner == 0 {
          panic!("bug");
        } else if pad_outer == 0 && pad_inner > 0 {
          code.cfg.emit_arg_shapes = true;
          code.append(format!(r"let a_inner = {{%0.s[3]}} + {} in",
              pad_inner,
          ));
          code.append(r"let t0 = {%0} in");
          code.append(format!(r"let t1 = map (\t_blk_row -> map (\t_row -> map (\t_inner -> (t_inner ++ (replicate {} {})) :> [a_inner]{}) t_row) t_blk_row) t0 in",
              pad_inner,
              self.pad_val.format_futhark(),
              arg[0].dtype.format_futhark(),
          ));
          code.append(r"let {%1} = t1 in");
        } else if pad_outer == 0 && pad_inner < 0 {
          code.cfg.emit_arg_shapes = true;
          code.append(format!(r"let a_inner = {{%0.s[3]}} + {} in",
              pad_inner,
          ));
          code.append(r"let t0 = {%0} in");
          code.append(r"let t1 = map (\t_blk_row -> map (\t_row -> map (\t_inner -> t_inner[0:a_inner]) t_row) t_blk_row) t0 in");
          code.append(r"let {%1} = t1 in");
        } else {
          println!("DEBUG: BlockPadFutThunkSpec::gen_futhark: ndim={} org block={:?} new block={:?}",
              arg[0].ndim(), self.org_block, self.new_block);
          unimplemented!();
        }
      }
      _ => {
        unimplemented!();
      }
    }
    code.into()
  }

  fn pop_adj(&self, arg: &[(CellPtr, Clock)], out: CellPtr, out_clk: Clock, out_adj: CellPtr, arg_adj: &mut [CellPtr]) -> Result<FutharkThunkAdj, ThunkAdjErr> {
    let arg_ty_ = arg[0].0.type_();
    let nd = arg_ty_.ndim() as usize;
    assert_eq!(arg_ty_.shape[nd - 3], self.org_block[0]);
    assert_eq!(arg_ty_.shape[nd - 1], self.org_block[1]);
    let pad_outer = self.new_block[0] - self.org_block[0];
    let pad_inner = self.new_block[1] - self.org_block[1];
    if pad_outer == 0 && pad_inner == 0 {
      panic!("bug");
    } else if pad_outer == 0 && pad_inner > 0 {
      arg_adj[0] += out_adj.block_unpad(self.org_block);
    } else if pad_outer == 0 && pad_inner < 0 {
      arg_adj[0] += out_adj.block_pad(self.org_block, ScalarVal_::zero(arg_ty_.dtype));
    } else {
      println!("DEBUG: BlockPadFutThunkSpec::pop_adj: ndim={} org block={:?} new block={:?}",
          arg_ty_.ndim(), self.org_block, self.new_block);
      unimplemented!();
    }
    Ok(FutharkThunkAdj::Spec)
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct BlockTriElemAffineFutThunkSpec {
  pub diag_scale: ScalarVal_,
  pub diag_shift: ScalarVal_,
  pub lo_scale: ScalarVal_,
  pub lo_shift: ScalarVal_,
  pub up_scale: ScalarVal_,
  pub up_shift: ScalarVal_,
}

impl FutharkThunkSpec for BlockTriElemAffineFutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.block_tri_elem_affine")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn arity(&self) -> Option<(u16, u16)> {
    Some((1, 1))
  }

  /*fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }*/

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    // FIXME
    Ok(arg[0])
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    // FIXME
    Ok(arg[0].clone())
  }

  fn gen_futhark(&self, /*abi: &mut FutAbi,*/ arg: &[Dim], out: &[Dim]) -> Result<FutharkThunkGenCode, FutharkGenErr> {
    //abi.set_out_arr(0, AbiOutput::Pure, AbiArrayRepr::Nd, AbiScalarType::Unspec);
    //abi.set_arg_arr(0, AbiInput::Shared, AbiArrayRepr::Nd, AbiScalarType::Unspec);
    //let out = FutharkThunkSpec::out_dim(self, arg).map_err(|e| e.into_gen())?;
    let mut code = FutharkThunkGenCode::default();
    code.abi.arityout = 1;
    code.abi.set_out(0, FutharkArrayRepr::Nd);
    code.abi.arityin = 1;
    code.abi.set_arg(0, FutharkArrayRepr::Nd);
    match out[0].ndim {
      4 => {
        code.cfg.emit_arg_shapes = true;
        //code.append(r"let a_inner = {%0.s[3]} in");
        //code.append(r"let b_inner = {%0.s[1]} in");
        //code.append(r"let t_iota_row = iota b_inner in");
        //code.append(r"let t_iota_col = iota a_inner in");
        //code.append(format!(r"let t0 = {{%0}} :> [][b_inner][][a_inner]{} in", arg[0].dtype.format_futhark()));
        code.append(r"let t_iota_row = iota {%0.s[1]} in");
        code.append(r"let t_iota_col = iota {%0.s[3]} in");
        code.append(r"let t0 = {%0} in");
        code.append(format!(r"let t1 = map (\t_blk_row -> map2 (\idx_row t_row -> map (\t_inner -> map2 (\idx_col u -> if idx_row < idx_col then ({} * u + {}) else if idx_row > idx_col then ({} * u + {}) else ({} * u + {})) t_iota_col t_inner) t_row) t_iota_row t_blk_row) t0 in",
            self.up_scale.format_futhark(),
            self.up_shift.format_futhark(),
            self.lo_scale.format_futhark(),
            self.lo_shift.format_futhark(),
            self.diag_scale.format_futhark(),
            self.diag_shift.format_futhark(),
        ));
        code.append(r"let {%1} = t1 in");
      }
      _ => {
        unimplemented!();
      }
    }
    code.into()
  }

  fn pop_adj(&self, arg: &[(CellPtr, Clock)], _out: CellPtr, _out_clk: Clock, out_adj: CellPtr, arg_adj: &mut [CellPtr]) -> Result<FutharkThunkAdj, ThunkAdjErr> {
    let arg0_ty = arg[0].0.type_();
    let dtype = arg0_ty.dtype;
    arg_adj[0] += out_adj.block_tri_elem_affine(
        self.diag_scale,
        ScalarVal_::zero(dtype),
        self.lo_scale,
        ScalarVal_::zero(dtype),
        self.up_scale,
        ScalarVal_::zero(dtype),
    );
    Ok(FutharkThunkAdj::Spec)
  }
}

/*#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct DotThunkSpec;*/

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct MatrixMulThunkSpec {
  pub l_t:      bool,
  pub r_t:      bool,
  //pub l_dtype:  Dtype,
  //pub r_dtype:  Dtype,
  pub o_dtype:  Dtype,
  pub o_scale:  ScalarVal_,
}

impl ThunkSpec for MatrixMulThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("matmul")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Time)
  }

  fn arity(&self) -> Option<(u16, u16)> {
    Some((2, 1))
  }

  fn param_count(&self) -> u16 {
    0
  }

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    if !(arg[0].ndim() == 2 && arg[1].ndim() == 2) {
      return Err(ThunkDimErr::_Bot);
    }
    Ok(Dim{ndim: 2, dtype: self.o_dtype})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    let tys = self._calculate_out_ty(arg)?;
    Ok(tys.o_ty)
  }

  fn gen_impl_(&self, spec_dim: Vec<Dim>, pmach: PMach) -> Option<Rc<dyn ThunkImpl_>> {
    match pmach {
      #[cfg(feature = "nvgpu")]
      PMach::NvGpu => {
        Some(Rc::new(MatrixMulF16F32GpuThunkImpl::default()))
      }
      _ => {
        println!("WARNING:MatrixMulThunkSpec::gen_impl_: no impl for pmach={:?}", pmach);
        None
      }
    }
  }

  fn pop_adj(&self, arg: &[(CellPtr, Clock)], out: CellPtr, _out_clk: Clock, _out_mode: ThunkMode, out_adj: CellPtr, arg_adj: &mut [CellPtr]) -> Result<(), ThunkAdjErr> {
    match (self.l_t, self.r_t) {
      (false, false) => {
        // (O-R, O-C) = (a-R, a-C) x (b-R, b-C)     = (a-R, b-C)
        // (a-R, a-C) = (O-R, O-C) x (O-C, a-C)     = (O-R, O-C)   x (b-R, b-C)^T
        // (b-R, b-C) = (b-R, O-R) x (O-R, O-C)     = (a-R, a-C)^T x (O-R, O-C)
        arg_adj[1] += arg[0].0.matmul_scale(true, out_adj, false, self.o_scale);
        arg_adj[0] += out_adj.matmul_scale(false, arg[1].0, true, self.o_scale);
      }
      (false, true) => {
        // (O-R, O-C) = (a-R, a-C)   x (b-R, b-C)^T = (a-R, b-R)
        // (a-R, a-C) = (O-R, O-C)   x (O-R, a-C)   = (O-R, O-C)   x (b-R, b-C)
        // (b-R, b-C) = (O-R, O-C)^T x (O-R, b-C)   = (O-R, O-C)^T x (a-R, a-C)
        arg_adj[1] += out_adj.matmul_scale(true, arg[0].0, false, self.o_scale);
        arg_adj[0] += out_adj.matmul_scale(false, arg[1].0, false, self.o_scale);
      }
      (true, false) => {
        // (O-R, O-C) = (a-R, a-C)^T x (b-R, b-C)   = (a-C, b-C)
        // (a-R, a-C) = (a-R, O-C)   x (O-R, O-C)^T = (b-R, b-C)   x (O-R, O-C)^T
        // (b-R, b-C) = (b-R, O-R)   x (O-R, O-C)   = (a-R, a-C)   x (O-R, O-C)
        arg_adj[1] += arg[0].0.matmul_scale(false, out_adj, false, self.o_scale);
        arg_adj[0] += arg[1].0.matmul_scale(false, out_adj, true, self.o_scale);
      }
      (true, true) => {
        // (O-R, O-C) = (a-R, a-C)^T x (b-R, b-C)^T = (a-C, b-R)
        // (a-R, a-C) = (a-R, O-C)   x (O-R, O-C)^T = (b-R, b-C)^T x (O-R, O-C)^T
        // (b-R, b-C) = (O-R, O-C)^T x (O-C, b-C)   = (O-R, O-C)^T x (a-R, a-C)^T
        arg_adj[1] += out_adj.matmul_scale(true, arg[0].0, true, self.o_scale);
        arg_adj[0] += arg[1].0.matmul_scale(true, out_adj, true, self.o_scale);
      }
    }
    Ok(())
  }
}

#[derive(Clone, Debug)]
pub struct MatrixMulTypes {
  pub l_blk_outer: i64,
  pub l_blk_inner: i64,
  pub r_blk_inner: i64,
  pub r_blk_outer: i64,
  pub o_ty:   CellType,
}

impl MatrixMulThunkSpec {
  pub fn _calculate_out_ty(&self, arg: &[CellType]) -> Result<MatrixMulTypes, ThunkTypeErr> {
    if !(arg[0].ndim() == 2 && arg[1].ndim() == 2) {
      return Err(ThunkTypeErr::_Bot);
    }
    let l_dtype = arg[0].dtype;
    let r_dtype = arg[1].dtype;
    let l_block = [arg[0].shape[0], arg[0].shape[1]];
    let r_block = [arg[1].shape[0], arg[1].shape[1]];
    let [l_blk_outer, l_blk_inner] = if self.l_t { [l_block[1], l_block[0]] } else { l_block };
    let [r_blk_inner, r_blk_outer] = if self.r_t { [r_block[1], r_block[0]] } else { r_block };
    if l_blk_inner != r_blk_inner {
      return Err(ThunkTypeErr::_Bot);
    }
    let m = l_blk_outer;
    let n = r_blk_outer;
    let o_ty = CellType{shape: vec![m, n], dtype: self.o_dtype};
    if cfg_debug() {
    println!("DEBUG: BlockMatrixMulThunkSpec::_calculate_out_ty: {:?}{} x {:?}{} = {:?}",
        &arg[0].shape, if self.l_t { " T" } else { "" },
        &arg[1].shape, if self.r_t { " T" } else { "" },
        &o_ty.shape);
    }
    let tys = MatrixMulTypes{
      l_blk_outer,
      l_blk_inner,
      r_blk_inner,
      r_blk_outer,
      o_ty,
    };
    Ok(tys)
  }
}

#[cfg(feature = "nvgpu")]
#[derive(Default)]
pub struct MatrixMulF16F32GpuThunkImpl {
  // TODO
  alpha: Cell<f32>,
  beta: Cell<f32>,
}

#[cfg(feature = "nvgpu")]
impl MatrixMulF16F32GpuThunkImpl {
  pub fn _enter(&self, ctr: &CtxCtr, env: &mut CtxEnv, spec_: &dyn ThunkSpec_, _param: &[ScalarVal_], arg: &[(CellPtr, Clock)], th: ThunkPtr, out: CellPtr, prev_oclk: Clock, oclk: Clock, mode: ThunkMode) -> ThunkResult {
    if cfg_debug() { println!("DEBUG: MatrixMulF16F32GpuThunkImpl::_enter"); }
    TL_PCTX.with(|pctx| {
      let gpu = pctx.nvgpu.as_ref().unwrap();
      let ret = gpu.compute.sync();
      match ret {
        Err(e) => {
          println!("DEBUG: MatrixMulF16F32GpuThunkImpl::_enter: gemm pre sync error: {:?}", e);
          Err(ThunkErr::Failure)
        }
        Ok(_) => Ok(())
      }
    })?;
    let t0 = Stopwatch::tl_stamp();
    let spec = spec_.as_any().downcast_ref::<MatrixMulThunkSpec>().unwrap();
    match &spec.o_scale {
      &ScalarVal_::F32(ref val) => {
        self.alpha.set(*val.borrow());
      }
      _ => unimplemented!()
    }
    match mode {
      ThunkMode::Apply => {
        self.beta.set(0.0);
      }
      ThunkMode::Accumulate => {
        self.beta.set(1.0);
      }
      _ => unimplemented!()
    }
    let mut arg_ty_ = Vec::with_capacity(arg.len());
    for &(x, _) in arg.iter() {
      match env._lookup_ref_(x) {
        Err(_) => panic!("bug"),
        Ok(e) => {
          arg_ty_.push(e.ty.clone());
        }
      }
    }
    let tys = spec._calculate_out_ty(&arg_ty_).unwrap();
    if cfg_debug() {
    println!("DEBUG: MatrixMulF16F32GpuThunkImpl::_enter: arg_ty_={:?}", &arg_ty_);
    println!("DEBUG: MatrixMulF16F32GpuThunkImpl::_enter: gemmtys={:?}", &tys);
    }
    let out_ty_ = tys.o_ty.clone();
    let colmajor_bt = spec.l_t;
    let colmajor_at = spec.r_t;
    let colmajor_n = tys.l_blk_outer;
    let colmajor_m = tys.r_blk_outer;
    let inner_len = tys.l_blk_inner;
    if cfg_debug() {
    println!("DEBUG: MatrixMulF16F32GpuThunkImpl::_enter: m={}", colmajor_m);
    println!("DEBUG: MatrixMulF16F32GpuThunkImpl::_enter: n={}", colmajor_n);
    println!("DEBUG: MatrixMulF16F32GpuThunkImpl::_enter: k={}", inner_len);
    }
    assert_eq!(inner_len, tys.r_blk_inner);
    assert!(colmajor_m >= 0);
    assert!(colmajor_m <= i32::max_value() as _);
    assert!(colmajor_n >= 0);
    assert!(colmajor_n <= i32::max_value() as _);
    assert!(inner_len >= 0);
    assert!(inner_len <= i32::max_value() as _);
    // FIXME: does ldx need multiplying by elem size?
    let (lda, ldb, ldc) = if arg_ty_[0].ndim() == 2 && arg_ty_[1].ndim() == 2 && out_ty_.ndim() == 2 {
      let ldb = arg_ty_[0].shape[1];
      let lda = arg_ty_[1].shape[1];
      let ldc = out_ty_.shape[1];
      (lda, ldb, ldc)
    } else {
      panic!("bug");
    };
    if cfg_debug() {
    println!("DEBUG: MatrixMulF16F32GpuThunkImpl::_enter: lda={}", lda);
    println!("DEBUG: MatrixMulF16F32GpuThunkImpl::_enter: ldb={}", ldb);
    println!("DEBUG: MatrixMulF16F32GpuThunkImpl::_enter: ldc={}", ldc);
    }
    assert!(lda >= 0);
    assert!(lda <= i32::max_value() as _);
    assert!(ldb >= 0);
    assert!(ldb <= i32::max_value() as _);
    assert!(ldc >= 0);
    assert!(ldc <= i32::max_value() as _);
    let loc = TL_PCTX.with(|pctx| {
      let gpu = pctx.nvgpu.as_ref().unwrap();
      gpu.device_locus()
    });
    if cfg_debug() { println!("DEBUG: MatrixMulF16F32GpuThunkImpl::_enter: read arg[0]..."); }
    let b_dptr = match env.pread_view(arg[0].0, arg[0].1, loc) {
      Err(_) => panic!("bug"),
      Ok(e) => {
        assert_eq!(&e.ty, &arg_ty_[0]);
        let v_ty = match e.view().eval_contiguous(&e.root_ty) {
          Err(_) => {
            println!("ERROR: MatrixMulF16F32GpuThunkImpl::_enter: left arg is not a zero-copy (contiguous) view");
            panic!();
          }
          Ok(ty) => ty
        };
        assert_eq!(&e.ty, v_ty.as_ref());
        match e.cel_ {
          &mut Cell_::Phy(ref _state, ref _clo, ref mut pcel) => {
            let pcel_addr = match pcel.lookup(loc, PMach::NvGpu) {
              None => panic!("bug"),
              Some(rep) => rep.addr.get()
            };
            let base = TL_PCTX.with(|pctx| {
              let (dptr, _) = pctx.nvgpu.as_ref().unwrap().lookup_dev(pcel_addr).unwrap();
              dptr
            });
            base + v_ty.pointer_offset()
          }
          _ => panic!("bug")
        }
      }
    };
    if cfg_debug() { println!("DEBUG: MatrixMulF16F32GpuThunkImpl::_enter: read arg[1]..."); }
    let a_dptr = match env.pread_view(arg[1].0, arg[1].1, loc) {
      Err(_) => panic!("bug"),
      Ok(e) => {
        assert_eq!(&e.ty, &arg_ty_[1]);
        let v_ty = match e.view().eval_contiguous(&e.root_ty) {
          Err(_) => {
            println!("ERROR: MatrixMulF16F32GpuThunkImpl::_enter: right arg is not a zero-copy (contiguous) view");
            panic!();
          }
          Ok(ty) => ty
        };
        assert_eq!(&e.ty, v_ty.as_ref());
        match e.cel_ {
          &mut Cell_::Phy(ref _state, ref _clo, ref mut pcel) => {
            let pcel_addr = match pcel.lookup(loc, PMach::NvGpu) {
              None => panic!("bug"),
              Some(rep) => rep.addr.get()
            };
            let base = TL_PCTX.with(|pctx| {
              let (dptr, _) = pctx.nvgpu.as_ref().unwrap().lookup_dev(pcel_addr).unwrap();
              dptr
            });
            base + v_ty.pointer_offset()
          }
          _ => panic!("bug")
        }
      }
    };
    if cfg_debug() { println!("DEBUG: MatrixMulF16F32GpuThunkImpl::_enter: write out..."); }
    let c_dptr = match match mode {
      ThunkMode::Apply => env.pwrite_view(out, oclk, loc),
      ThunkMode::Accumulate |
      ThunkMode::Initialize => env.prewrite_view(out, prev_oclk, oclk, loc)
    } {
      Err(_) => panic!("bug"),
      Ok(e) => {
        assert_eq!(&e.ty, &out_ty_);
        let v_ty = match e.view().eval_contiguous(&e.root_ty) {
          Err(_) => {
            println!("ERROR: MatrixMulF16F32GpuThunkImpl::_enter: output is not a zero-copy (contiguous) view");
            panic!();
          }
          Ok(ty) => ty
        };
        assert_eq!(&e.ty, v_ty.as_ref());
        match e.cel_ {
          &mut Cell_::Phy(ref _state, ref _clo, ref mut pcel) => {
            let pcel_addr = match pcel.lookup(loc, PMach::NvGpu) {
              None => panic!("bug"),
              Some(rep) => rep.addr.get()
            };
            let base = TL_PCTX.with(|pctx| {
              let (dptr, _) = pctx.nvgpu.as_ref().unwrap().lookup_dev(pcel_addr).unwrap();
              dptr
            });
            base + v_ty.pointer_offset()
          }
          _ => panic!("bug")
        }
      }
    };
    let b_gputy = match arg_ty_[0].dtype {
      Dtype::F32 => CUDA_R_32F,
      Dtype::F16 => CUDA_R_16F,
      Dtype::Bf16 => CUDA_R_16BF,
      _ => unimplemented!()
    };
    let a_gputy = match arg_ty_[1].dtype {
      Dtype::F32 => CUDA_R_32F,
      Dtype::F16 => CUDA_R_16F,
      Dtype::Bf16 => CUDA_R_16BF,
      _ => unimplemented!()
    };
    let c_gputy = match out_ty_.dtype {
      Dtype::F32 => CUDA_R_32F,
      Dtype::F16 => CUDA_R_16F,
      Dtype::Bf16 => CUDA_R_16BF,
      _ => unimplemented!()
    };
    TL_PCTX.with(|pctx| {
      let gpu = pctx.nvgpu.as_ref().unwrap();
      if cfg_debug() { println!("DEBUG: MatrixMulF16F32GpuThunkImpl::_enter: gemm..."); }
      let ret = gpu.compute.sync();
      match ret {
        Err(e) => {
          println!("DEBUG: MatrixMulF16F32GpuThunkImpl::_enter: pre gemm sync error: {:?}", e);
          Err(ThunkErr::Failure)
        }
        Ok(_) => Ok(())
      }?;
      let t1 = Stopwatch::tl_stamp();
      if cfg_debug() { println!("DEBUG: MatrixMulF16F32GpuThunkImpl::_enter: pre gemm elapsed: {:.06} s", t1 - t0); }
      TL_CTX.with(|ctx| {
        if oclk.rst <= 0 {
          panic!("bug");
        } else if oclk.rst == 1 {
          ctx.timing.pregemm1.borrow_mut().push(t1 - t0);
        } else {
          ctx.timing.pregemm.borrow_mut().push(t1 - t0);
        }
      });
      let t0 = t1;
      if cfg_debug() {
      let alpha_ptr = self.alpha.as_ptr() as usize;
      let beta_ptr = self.beta.as_ptr() as usize;
      println!("DEBUG: MatrixMulF16F32GpuThunkImpl::_enter: alpha ptr=0x{:016x}", alpha_ptr);
      println!("DEBUG: MatrixMulF16F32GpuThunkImpl::_enter: beta ptr =0x{:016x}", beta_ptr);
      }
      let ret = cublas_gemm(
          &gpu.blas_ctx,
          colmajor_at, colmajor_bt,
          colmajor_m as _, colmajor_n as _, inner_len as _,
          self.alpha.as_ptr() as *const f32 as *const c_void,
          a_dptr, a_gputy, lda as _,
          b_dptr, b_gputy, ldb as _,
          self.beta.as_ptr() as *const f32 as *const c_void,
          c_dptr, c_gputy, ldc as _,
          &gpu.compute,
      );
      match ret {
        Err(e) => {
          println!("DEBUG: MatrixMulF16F32GpuThunkImpl::_enter: gemm error: {:?}", e);
          Err(ThunkErr::Failure)
        }
        Ok(_) => Ok(())
      }?;
      let ret = gpu.compute.sync();
      match ret {
        Err(e) => {
          println!("DEBUG: MatrixMulF16F32GpuThunkImpl::_enter: gemm sync error: {:?}", e);
          Err(ThunkErr::Failure)
        }
        Ok(_) => Ok(())
      }?;
      let t1 = Stopwatch::tl_stamp();
      if cfg_debug() { println!("DEBUG: MatrixMulF16F32GpuThunkImpl::_enter: gemm OK elapsed: {:.06} s", t1 - t0); }
      TL_CTX.with(|ctx| {
        if oclk.rst <= 0 {
          panic!("bug");
        } else if oclk.rst == 1 {
          ctx.timing.gemm1.borrow_mut().push(t1 - t0);
        } else {
          ctx.timing.gemm.borrow_mut().push(t1 - t0);
        }
      });
      Ok(())
    })
  }
}

#[cfg(feature = "nvgpu")]
impl ThunkImpl for MatrixMulF16F32GpuThunkImpl {
  fn apply(&self, ctr: &CtxCtr, env: &mut CtxEnv, spec_: &dyn ThunkSpec_, param: &[ScalarVal_], arg: &[(CellPtr, Clock)], th: ThunkPtr, out: CellPtr, prev_oclk: Clock, oclk: Clock) -> ThunkResult {
    if cfg_debug() { println!("DEBUG: MatrixMulF16F32GpuThunkImpl::apply"); }
    let mode = ThunkMode::Apply;
    self._enter(ctr, env, spec_, param, arg, th, out, prev_oclk, oclk, mode)
  }

  fn accumulate(&self, ctr: &CtxCtr, env: &mut CtxEnv, spec_: &dyn ThunkSpec_, param: &[ScalarVal_], arg: &[(CellPtr, Clock)], th: ThunkPtr, out: CellPtr, prev_oclk: Clock, oclk: Clock) -> ThunkResult {
    if cfg_debug() { println!("DEBUG: MatrixMulF16F32GpuThunkImpl::accumulate"); }
    let mode = ThunkMode::Accumulate;
    self._enter(ctr, env, spec_, param, arg, th, out, prev_oclk, oclk, mode)
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct BlockMatrixMulThunkSpec {
  //pub l_shape:  [i64; 2],
  //pub r_shape:  [i64; 2],
  pub l_block:  [i64; 2],
  pub r_block:  [i64; 2],
  //pub l_nblock: [i64; 2],
  //pub r_nblock: [i64; 2],
  pub l_blk_t:  bool,
  pub r_blk_t:  bool,
  pub l_dtype:  Dtype,
  pub r_dtype:  Dtype,
  pub o_dtype:  Dtype,
  pub o_scale:  ScalarVal_,
}

impl ThunkSpec for BlockMatrixMulThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("block_matmul")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Time)
  }

  fn arity(&self) -> Option<(u16, u16)> {
    Some((2, 1))
  }

  fn param_count(&self) -> u16 {
    0
  }

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    if !((arg[0].ndim() == 2 && arg[1].ndim() == 2) ||
         (arg[0].ndim() == 4 && arg[1].ndim() == 4))
    {
      return Err(ThunkDimErr::_Bot);
    }
    if self.l_dtype != arg[0].dtype {
      return Err(ThunkDimErr::_Bot);
    }
    if self.r_dtype != arg[1].dtype {
      return Err(ThunkDimErr::_Bot);
    }
    if arg[0].ndim() == 2 && arg[1].ndim() == 2 {
      Ok(Dim{ndim: 2, dtype: self.o_dtype})
    } else if arg[0].ndim() == 4 && arg[1].ndim() == 4 {
      Ok(Dim{ndim: 4, dtype: self.o_dtype})
    } else {
      unreachable!();
    }
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    let tys = self._calculate_out_ty(arg)?;
    Ok(tys.o_ty)
    /*if arg[0].ndim() != 2 {
      return Err(ThunkTypeErr::_Bot);
    }
    if arg[1].ndim() != 2 {
      return Err(ThunkTypeErr::_Bot);
    }
    if self.l_dtype != arg[0].dtype {
      return Err(ThunkTypeErr::_Bot);
    }
    if self.r_dtype != arg[1].dtype {
      return Err(ThunkTypeErr::_Bot);
    }
    if arg[0].shape[0] % self.l_block[0] != 0 {
      return Err(ThunkTypeErr::_Bot);
    }
    if arg[0].shape[1] % self.l_block[1] != 0 {
      return Err(ThunkTypeErr::_Bot);
    }
    if arg[1].shape[0] % self.r_block[0] != 0 {
      return Err(ThunkTypeErr::_Bot);
    }
    if arg[1].shape[1] % self.r_block[1] != 0 {
      return Err(ThunkTypeErr::_Bot);
    }
    let l_nrow = arg[0].shape[0] / self.l_block[0];
    let l_ncol = arg[0].shape[1] / self.l_block[1];
    let r_nrow = arg[1].shape[0] / self.r_block[0];
    let r_ncol = arg[1].shape[1] / self.r_block[1];
    println!("DEBUG: BlockMatrixMulThunkSpec::out_ty_: l nrow={} r nrow={} l ncol={} r ncol={}",
        l_nrow, r_nrow,
        l_ncol, r_ncol);
    if !(l_nrow == r_nrow || l_nrow == 1 || r_nrow == 1) {
      return Err(ThunkTypeErr::_Bot);
    }
    if !(l_ncol == r_ncol || l_ncol == 1 || r_ncol == 1) {
      return Err(ThunkTypeErr::_Bot);
    }
    let nrow = max(l_nrow, r_nrow);
    let ncol = max(l_ncol, r_ncol);
    let [l_blk_outer, l_blk_inner] = if self.l_blk_t { [self.l_block[1], self.l_block[0]] } else { self.l_block };
    let [r_blk_inner, r_blk_outer] = if self.r_blk_t { [self.r_block[1], self.r_block[0]] } else { self.r_block };
    if l_blk_inner != r_blk_inner {
      return Err(ThunkTypeErr::_Bot);
    }
    let m = l_blk_outer * nrow;
    let n = r_blk_outer * ncol;
    println!("DEBUG: BlockMatrixMulThunkSpec::out_ty_: ({:?} / {:?}{}) x ({:?} / {:?}{}) = [{}, {}]",
        &arg[0].shape, self.l_block, if self.l_blk_t { " T" } else { "" },
        &arg[1].shape, self.r_block, if self.r_blk_t { " T" } else { "" },
        m, n);
    let o_ty = CellType{shape: vec![m, n], dtype: self.o_dtype};
    Ok(o_ty)*/
  }

  fn gen_impl_(&self, spec_dim: Vec<Dim>, pmach: PMach) -> Option<Rc<dyn ThunkImpl_>> {
    match pmach {
      #[cfg(feature = "nvgpu")]
      PMach::NvGpu => {
        Some(Rc::new(BlockMatrixMulF16F32GpuThunkImpl::default()))
      }
      _ => {
        println!("WARNING:BlockMatrixMulThunkSpec::gen_impl_: no impl for pmach={:?}", pmach);
        None
      }
    }
  }

  fn pop_adj(&self, arg: &[(CellPtr, Clock)], out: CellPtr, _out_clk: Clock, _out_mode: ThunkMode, out_adj: CellPtr, arg_adj: &mut [CellPtr]) -> Result<(), ThunkAdjErr> {
    let l_ty = arg[0].0.type_();
    let r_ty = arg[1].0.type_();
    let y_ty = out.type_();
    let dy_ty = out_adj.type_();
    /*if l_ty.ndim() == 2 && r_ty.ndim() == 2 && y_ty.ndim() == 2 && dy_ty.ndim() == 2 {
    let [l_blk_outer, l_blk_inner] = if self.l_blk_t { [self.l_block[1], self.l_block[0]] } else { self.l_block };
    let [r_blk_inner, r_blk_outer] = if self.r_blk_t { [self.r_block[1], self.r_block[0]] } else { self.r_block };
    assert_eq!(l_blk_inner, r_blk_inner);
    let out_block = [l_blk_outer, r_blk_outer];
    match (self.l_blk_t, self.r_blk_t) {
      (false, false) => {
        // (O-R, O-C) = (a-R, a-C) x (b-R, b-C)     = (a-R, b-C)
        // (a-R, a-C) = (O-R, O-C) x (O-C, a-C)     = (O-R, O-C)   x (b-R, b-C)^T
        // (b-R, b-C) = (b-R, O-R) x (O-R, O-C)     = (a-R, a-C)^T x (O-R, O-C)
        arg_adj[0] += out_adj.block_mm_scale(out_block, false, arg[1].0, self.r_block, true, self.o_scale);
        arg_adj[1] += arg[0].0.block_mm_scale(self.l_block, true, out_adj, out_block, false, self.o_scale);
      }
      (false, true) => {
        // (O-R, O-C) = (a-R, a-C)   x (b-R, b-C)^T = (a-R, b-R)
        // (a-R, a-C) = (O-R, O-C)   x (O-R, a-C)   = (O-R, O-C)   x (b-R, b-C)
        // (b-R, b-C) = (O-R, O-C)^T x (O-R, b-C)   = (O-R, O-C)^T x (a-R, a-C)
        arg_adj[0] += out_adj.block_mm_scale(out_block, false, arg[1].0, self.r_block, false, self.o_scale);
        arg_adj[1] += out_adj.block_mm_scale(out_block, true, arg[0].0, self.l_block, false, self.o_scale);
      }
      (true, false) => {
        // (O-R, O-C) = (a-R, a-C)^T x (b-R, b-C)   = (a-C, b-C)
        // (a-R, a-C) = (a-R, O-C)   x (O-R, O-C)^T = (b-R, b-C)   x (O-R, O-C)^T
        // (b-R, b-C) = (b-R, O-R)   x (O-R, O-C)   = (a-R, a-C)   x (O-R, O-C)
        arg_adj[0] += arg[1].0.block_mm_scale(self.r_block, false, out_adj, out_block, true, self.o_scale);
        arg_adj[1] += arg[0].0.block_mm_scale(self.l_block, false, out_adj, out_block, false, self.o_scale);
      }
      (true, true) => {
        // (O-R, O-C) = (a-R, a-C)^T x (b-R, b-C)^T = (a-C, b-R)
        // (a-R, a-C) = (a-R, O-C)   x (O-R, O-C)^T = (b-R, b-C)^T x (O-R, O-C)^T
        // (b-R, b-C) = (O-R, O-C)^T x (O-C, b-C)   = (O-R, O-C)^T x (a-R, a-C)^T
        arg_adj[0] += arg[1].0.block_mm_scale(self.r_block, true, out_adj, out_block, true, self.o_scale);
        arg_adj[1] += out_adj.block_mm_scale(out_block, true, arg[0].0, self.l_block, true, self.o_scale);
      }
    }
    } else */if l_ty.ndim() == 4 && r_ty.ndim() == 4 && y_ty.ndim() == 4 && dy_ty.ndim() == 4 {
      match (self.l_blk_t, self.r_blk_t) {
        (false, false) => {
          // (O-R, O-C) = (a-R, a-C) x (b-R, b-C)     = (a-R, b-C)
          // (a-R, a-C) = (O-R, O-C) x (O-C, a-C)     = (O-R, O-C)   x (b-R, b-C)^T
          // (b-R, b-C) = (b-R, O-R) x (O-R, O-C)     = (a-R, a-C)^T x (O-R, O-C)
          arg_adj[1] += arg[0].0.block_matmul_scale(true, out_adj, false, self.o_scale);
          arg_adj[0] += out_adj.block_matmul_scale(false, arg[1].0, true, self.o_scale);
        }
        (false, true) => {
          // (O-R, O-C) = (a-R, a-C)   x (b-R, b-C)^T = (a-R, b-R)
          // (a-R, a-C) = (O-R, O-C)   x (O-R, a-C)   = (O-R, O-C)   x (b-R, b-C)
          // (b-R, b-C) = (O-R, O-C)^T x (O-R, b-C)   = (O-R, O-C)^T x (a-R, a-C)
          arg_adj[1] += out_adj.block_matmul_scale(true, arg[0].0, false, self.o_scale);
          arg_adj[0] += out_adj.block_matmul_scale(false, arg[1].0, false, self.o_scale);
        }
        (true, false) => {
          // (O-R, O-C) = (a-R, a-C)^T x (b-R, b-C)   = (a-C, b-C)
          // (a-R, a-C) = (a-R, O-C)   x (O-R, O-C)^T = (b-R, b-C)   x (O-R, O-C)^T
          // (b-R, b-C) = (b-R, O-R)   x (O-R, O-C)   = (a-R, a-C)   x (O-R, O-C)
          arg_adj[1] += arg[0].0.block_matmul_scale(false, out_adj, false, self.o_scale);
          arg_adj[0] += arg[1].0.block_matmul_scale(false, out_adj, true, self.o_scale);
        }
        (true, true) => {
          // (O-R, O-C) = (a-R, a-C)^T x (b-R, b-C)^T = (a-C, b-R)
          // (a-R, a-C) = (a-R, O-C)   x (O-R, O-C)^T = (b-R, b-C)^T x (O-R, O-C)^T
          // (b-R, b-C) = (O-R, O-C)^T x (O-C, b-C)   = (O-R, O-C)^T x (a-R, a-C)^T
          arg_adj[1] += out_adj.block_matmul_scale(true, arg[0].0, true, self.o_scale);
          arg_adj[0] += arg[1].0.block_matmul_scale(true, out_adj, true, self.o_scale);
        }
      }
    } else {
      panic!("bug");
    }
    Ok(())
  }
}

#[derive(Clone, Debug)]
pub struct BlockMatrixMulTypes {
  pub l_nrow: i64,
  pub l_ncol: i64,
  pub r_nrow: i64,
  pub r_ncol: i64,
  pub nrow:   i64,
  pub ncol:   i64,
  pub l_blk_outer: i64,
  pub l_blk_inner: i64,
  pub r_blk_inner: i64,
  pub r_blk_outer: i64,
  pub o_ty:   CellType,
}

impl BlockMatrixMulThunkSpec {
  pub fn _calculate_out_ty(&self, arg: &[CellType]) -> Result<BlockMatrixMulTypes, ThunkTypeErr> {
    if !((arg[0].ndim() == 2 && arg[1].ndim() == 2) ||
         (arg[0].ndim() == 4 && arg[1].ndim() == 4))
    {
      return Err(ThunkTypeErr::_Bot);
    }
    if self.l_dtype != arg[0].dtype {
      return Err(ThunkTypeErr::_Bot);
    }
    if self.r_dtype != arg[1].dtype {
      return Err(ThunkTypeErr::_Bot);
    }
    let (l_nrow, l_ncol, r_nrow, r_ncol) = if arg[0].ndim() == 2 && arg[1].ndim() == 2 {
      if arg[0].shape[0] % self.l_block[0] != 0 {
        return Err(ThunkTypeErr::_Bot);
      }
      if arg[0].shape[1] % self.l_block[1] != 0 {
        return Err(ThunkTypeErr::_Bot);
      }
      if arg[1].shape[0] % self.r_block[0] != 0 {
        return Err(ThunkTypeErr::_Bot);
      }
      if arg[1].shape[1] % self.r_block[1] != 0 {
        return Err(ThunkTypeErr::_Bot);
      }
      let l_nrow = arg[0].shape[0] / self.l_block[0];
      let l_ncol = arg[0].shape[1] / self.l_block[1];
      let r_nrow = arg[1].shape[0] / self.r_block[0];
      let r_ncol = arg[1].shape[1] / self.r_block[1];
      (l_nrow, l_ncol, r_nrow, r_ncol)
    } else if arg[0].ndim() == 4 && arg[1].ndim() == 4 {
      let l_nrow = arg[0].shape[0];
      let l_ncol = arg[0].shape[2];
      let r_nrow = arg[1].shape[0];
      let r_ncol = arg[1].shape[2];
      (l_nrow, l_ncol, r_nrow, r_ncol)
    } else {
      unreachable!();
    };
    if cfg_debug() {
    println!("DEBUG: BlockMatrixMulThunkSpec::_calculate_out_ty: l nrow={} r nrow={} l ncol={} r ncol={}",
        l_nrow, r_nrow,
        l_ncol, r_ncol);
    }
    if !(l_nrow == r_nrow || l_nrow == 1 || r_nrow == 1) {
      return Err(ThunkTypeErr::_Bot);
    }
    if !(l_ncol == r_ncol || l_ncol == 1 || r_ncol == 1) {
      return Err(ThunkTypeErr::_Bot);
    }
    let nrow = max(l_nrow, r_nrow);
    let ncol = max(l_ncol, r_ncol);
    let [l_blk_outer, l_blk_inner] = if self.l_blk_t { [self.l_block[1], self.l_block[0]] } else { self.l_block };
    let [r_blk_inner, r_blk_outer] = if self.r_blk_t { [self.r_block[1], self.r_block[0]] } else { self.r_block };
    if l_blk_inner != r_blk_inner {
      return Err(ThunkTypeErr::_Bot);
    }
    let o_ty = if arg[0].ndim() == 2 && arg[1].ndim() == 2 {
      let m = nrow * l_blk_outer;
      let n = ncol * r_blk_outer;
      CellType{shape: vec![m, n], dtype: self.o_dtype}
    } else if arg[0].ndim() == 4 && arg[1].ndim() == 4 {
      CellType{shape: vec![nrow, l_blk_outer, ncol, r_blk_outer], dtype: self.o_dtype}
    } else {
      unreachable!();
    };
    if cfg_debug() {
    println!("DEBUG: BlockMatrixMulThunkSpec::_calculate_out_ty: ({:?} / {:?}{}) x ({:?} / {:?}{}) = {:?}",
        &arg[0].shape, self.l_block, if self.l_blk_t { " T" } else { "" },
        &arg[1].shape, self.r_block, if self.r_blk_t { " T" } else { "" },
        &o_ty.shape);
    }
    let tys = BlockMatrixMulTypes{
      l_nrow,
      l_ncol,
      r_nrow,
      r_ncol,
      nrow,
      ncol,
      l_blk_outer,
      l_blk_inner,
      r_blk_inner,
      r_blk_outer,
      o_ty,
    };
    Ok(tys)
  }
}

#[cfg(feature = "nvgpu")]
#[derive(Default)]
pub struct BlockMatrixMulF16F32GpuThunkImpl {
  // TODO
  alpha: Cell<f32>,
  beta: Cell<f32>,
  tmp_a: RefCell<Vec<u64>>,
  tmp_b: RefCell<Vec<u64>>,
  tmp_c: RefCell<Vec<u64>>,
}

#[cfg(feature = "nvgpu")]
impl BlockMatrixMulF16F32GpuThunkImpl {
  pub fn _enter(&self, ctr: &CtxCtr, env: &mut CtxEnv, spec_: &dyn ThunkSpec_, _param: &[ScalarVal_], arg: &[(CellPtr, Clock)], th: ThunkPtr, out: CellPtr, prev_oclk: Clock, oclk: Clock, mode: ThunkMode) -> ThunkResult {
    if cfg_debug() { println!("DEBUG: BlockMatrixMulF16F32GpuThunkImpl::_enter"); }
    TL_PCTX.with(|pctx| {
      let gpu = pctx.nvgpu.as_ref().unwrap();
      let ret = gpu.compute.sync();
      match ret {
        Err(e) => {
          println!("DEBUG: BlockMatrixMulF16F32GpuThunkImpl::_enter: gemm pre sync error: {:?}", e);
          Err(ThunkErr::Failure)
        }
        Ok(_) => Ok(())
      }
    })?;
    let t0 = Stopwatch::tl_stamp();
    let spec = spec_.as_any().downcast_ref::<BlockMatrixMulThunkSpec>().unwrap();
    match &spec.o_scale {
      &ScalarVal_::F32(ref val) => {
        self.alpha.set(*val.borrow());
      }
      _ => unimplemented!()
    }
    match mode {
      ThunkMode::Apply => {
        self.beta.set(0.0);
      }
      ThunkMode::Accumulate => {
        self.beta.set(1.0);
      }
      _ => unimplemented!()
    }
    let mut arg_ty_ = Vec::with_capacity(arg.len());
    for &(x, _) in arg.iter() {
      match env._lookup_ref_(x) {
        Err(_) => panic!("bug"),
        Ok(e) => {
          arg_ty_.push(e.ty.clone());
        }
      }
    }
    /*let out_ty_ = ThunkSpec::out_ty_(spec, &arg_ty_).unwrap();*/
    let tys = spec._calculate_out_ty(&arg_ty_).unwrap();
    if cfg_debug() {
    println!("DEBUG: BlockMatrixMulF16F32GpuThunkImpl::_enter: arg_ty_={:?}", &arg_ty_);
    println!("DEBUG: BlockMatrixMulF16F32GpuThunkImpl::_enter: gemmtys={:?}", &tys);
    println!("DEBUG: BlockMatrixMulF16F32GpuThunkImpl::_enter: nrow={}", tys.nrow);
    println!("DEBUG: BlockMatrixMulF16F32GpuThunkImpl::_enter: ncol={}", tys.ncol);
    println!("DEBUG: BlockMatrixMulF16F32GpuThunkImpl::_enter: nblk={}", tys.nrow * tys.ncol);
    }
    // FIXME: out type.
    let out_ty_ = tys.o_ty.clone();
    /*let out_ty_ = match env.lookup_ref(out) {
      None => panic!("bug"),
      Some(e) => {
        assert_eq!(&tys.o_ty.shape, &e.ty.shape);
        tys.o_ty.cast(e.ty.dtype)
      }
    };*/
    // FIXME FIXME: correct transposes, shapes, arg order for row major v col major.
    let colmajor_bt = spec.l_blk_t;
    let colmajor_at = spec.r_blk_t;
    let colmajor_n = tys.l_blk_outer;
    let colmajor_m = tys.r_blk_outer;
    let inner_len = tys.l_blk_inner;
    if cfg_debug() {
    println!("DEBUG: BlockMatrixMulF16F32GpuThunkImpl::_enter: m={}", colmajor_m);
    println!("DEBUG: BlockMatrixMulF16F32GpuThunkImpl::_enter: n={}", colmajor_n);
    println!("DEBUG: BlockMatrixMulF16F32GpuThunkImpl::_enter: k={}", inner_len);
    }
    assert_eq!(inner_len, tys.r_blk_inner);
    assert!(colmajor_m >= 0);
    assert!(colmajor_m <= i32::max_value() as _);
    assert!(colmajor_n >= 0);
    assert!(colmajor_n <= i32::max_value() as _);
    assert!(inner_len >= 0);
    assert!(inner_len <= i32::max_value() as _);
    // FIXME: does ldx need multiplying by elem size?
    let (lda, ldb, ldc) = if arg_ty_[0].ndim() == 2 && arg_ty_[1].ndim() == 2 && out_ty_.ndim() == 2 {
      let ldb = arg_ty_[0].shape[1];
      let lda = arg_ty_[1].shape[1];
      let ldc = out_ty_.shape[1];
      (lda, ldb, ldc)
    } else if arg_ty_[0].ndim() == 4 && arg_ty_[1].ndim() == 4 && out_ty_.ndim() == 4 {
      let ldb = arg_ty_[0].shape[2] * arg_ty_[0].shape[3];
      let lda = arg_ty_[1].shape[2] * arg_ty_[1].shape[3];
      let ldc = out_ty_.shape[2] * out_ty_.shape[3];
      (lda, ldb, ldc)
    } else {
      panic!("bug");
    };
    if cfg_debug() {
    println!("DEBUG: BlockMatrixMulF16F32GpuThunkImpl::_enter: lda={}", lda);
    println!("DEBUG: BlockMatrixMulF16F32GpuThunkImpl::_enter: ldb={}", ldb);
    println!("DEBUG: BlockMatrixMulF16F32GpuThunkImpl::_enter: ldc={}", ldc);
    }
    assert!(lda >= 0);
    assert!(lda <= i32::max_value() as _);
    assert!(ldb >= 0);
    assert!(ldb <= i32::max_value() as _);
    assert!(ldc >= 0);
    assert!(ldc <= i32::max_value() as _);
    let loc = TL_PCTX.with(|pctx| {
      let gpu = pctx.nvgpu.as_ref().unwrap();
      gpu.device_locus()
    });
    if cfg_debug() { println!("DEBUG: BlockMatrixMulF16F32GpuThunkImpl::_enter: read arg[0]..."); }
    //match env.pread_ref_(arg[0].0, arg[0].1, loc) {}
    match env.pread_view(arg[0].0, arg[0].1, loc) {
      Err(_) => panic!("bug"),
      Ok(e) => {
        assert_eq!(&e.ty, &arg_ty_[0]);
        let v_ty = match e.view().eval_contiguous(&e.root_ty) {
          Err(_) => {
            println!("ERROR: BlockMatrixMulF16F32GpuThunkImpl::_enter: left arg is not a zero-copy (contiguous) view");
            panic!();
          }
          Ok(ty) => ty
        };
        assert_eq!(&e.ty, v_ty.as_ref());
        //assert_eq!(v_ty.as_ref(), &arg_ty_[0]);
        match e.cel_ {
          &mut Cell_::Phy(ref _state, ref _clo, ref mut pcel) => {
            //let pcel_addr = pcel.get(arg[0].0, arg[0].1, &arg_ty_[0], loc, PMach::NvGpu);
            let pcel_addr = match pcel.lookup(loc, PMach::NvGpu) {
              None => panic!("bug"),
              Some(rep) => rep.addr.get()
            };
            let base0 = TL_PCTX.with(|pctx| {
              let (dptr, _) = pctx.nvgpu.as_ref().unwrap().lookup_dev(pcel_addr).unwrap();
              dptr
            });
            let base = base0 + v_ty.pointer_offset();
            let inc = spec.l_dtype.size_bytes() as u64;
            let blk_row_len = spec.l_block[0] as u64;
            let blk_col_len = spec.l_block[1] as u64;
            let stride = if arg_ty_[0].ndim() == 2 && arg_ty_[1].ndim() == 2 && out_ty_.ndim() == 2 {
              arg_ty_[0].shape[1] as u64
            } else if arg_ty_[0].ndim() == 4 && arg_ty_[1].ndim() == 4 && out_ty_.ndim() == 4 {
              (arg_ty_[0].shape[2] * arg_ty_[0].shape[3]) as u64
            } else {
              panic!("bug");
            };
            let mult_row = if tys.l_nrow == 1 && tys.l_nrow != tys.nrow { 0 } else { 1 };
            let mult_col = if tys.l_ncol == 1 && tys.l_ncol != tys.ncol { 0 } else { 1 };
            let mut tmp = self.tmp_b.borrow_mut();
            tmp.clear();
            for j in 0 .. tys.nrow as u64 {
              for i in 0 .. tys.ncol as u64 {
                tmp.push(base + inc * (blk_col_len * mult_col * i + stride * blk_row_len * j));
              }
            }
          }
          _ => panic!("bug")
        }
      }
    }
    if cfg_debug() { println!("DEBUG: BlockMatrixMulF16F32GpuThunkImpl::_enter: read arg[1]..."); }
    //match env.pread_ref_(arg[1].0, arg[1].1, loc) {}
    match env.pread_view(arg[1].0, arg[1].1, loc) {
      Err(_) => panic!("bug"),
      Ok(e) => {
        assert_eq!(&e.ty, &arg_ty_[1]);
        let v_ty = match e.view().eval_contiguous(&e.root_ty) {
          Err(_) => {
            println!("ERROR: BlockMatrixMulF16F32GpuThunkImpl::_enter: right arg is not a zero-copy (contiguous) view");
            panic!();
          }
          Ok(ty) => ty
        };
        assert_eq!(&e.ty, v_ty.as_ref());
        //assert_eq!(v_ty.as_ref(), &arg_ty_[1]);
        match e.cel_ {
          &mut Cell_::Phy(ref _state, ref _clo, ref mut pcel) => {
            //let pcel_addr = pcel.get(arg[1].0, arg[1].1, &arg_ty_[1], loc, PMach::NvGpu);
            let pcel_addr = match pcel.lookup(loc, PMach::NvGpu) {
              None => panic!("bug"),
              Some(rep) => rep.addr.get()
            };
            let base0 = TL_PCTX.with(|pctx| {
              let (dptr, _) = pctx.nvgpu.as_ref().unwrap().lookup_dev(pcel_addr).unwrap();
              dptr
            });
            let base = base0 + v_ty.pointer_offset();
            let inc = spec.r_dtype.size_bytes() as u64;
            let blk_row_len = spec.r_block[0] as u64;
            let blk_col_len = spec.r_block[1] as u64;
            let stride = if arg_ty_[0].ndim() == 2 && arg_ty_[1].ndim() == 2 && out_ty_.ndim() == 2 {
              arg_ty_[1].shape[1] as u64
            } else if arg_ty_[0].ndim() == 4 && arg_ty_[1].ndim() == 4 && out_ty_.ndim() == 4 {
              (arg_ty_[1].shape[2] * arg_ty_[1].shape[3]) as u64
            } else {
              panic!("bug");
            };
            let mult_row = if tys.r_nrow == 1 && tys.r_nrow != tys.nrow { 0 } else { 1 };
            let mult_col = if tys.r_ncol == 1 && tys.r_ncol != tys.ncol { 0 } else { 1 };
            let mut tmp = self.tmp_a.borrow_mut();
            tmp.clear();
            for j in 0 .. tys.nrow as u64 {
              for i in 0 .. tys.ncol as u64 {
                tmp.push(base + inc * (blk_col_len * mult_col * i + stride * blk_row_len * mult_row * j));
              }
            }
          }
          _ => panic!("bug")
        }
      }
    }
    if cfg_debug() { println!("DEBUG: BlockMatrixMulF16F32GpuThunkImpl::_enter: write out..."); }
    /*match match mode {
      ThunkMode::Apply => env.pwrite_ref_(out, oclk, loc),
      ThunkMode::Accumulate |
      ThunkMode::Initialize => env.prewrite_ref_(out, prev_oclk, oclk, loc)
    } {*/
    match match mode {
      ThunkMode::Apply => env.pwrite_view(out, oclk, loc),
      ThunkMode::Accumulate |
      ThunkMode::Initialize => env.prewrite_view(out, prev_oclk, oclk, loc)
    } {
      Err(_) => panic!("bug"),
      Ok(e) => {
        assert_eq!(&e.ty, &out_ty_);
        let v_ty = match e.view().eval_contiguous(&e.root_ty) {
          Err(_) => {
            println!("ERROR: BlockMatrixMulF16F32GpuThunkImpl::_enter: output is not a zero-copy (contiguous) view");
            panic!();
          }
          Ok(ty) => ty
        };
        assert_eq!(&e.ty, v_ty.as_ref());
        //assert_eq!(v_ty.as_ref(), &out_ty_);
        match e.cel_ {
          &mut Cell_::Phy(ref _state, ref _clo, ref mut pcel) => {
            //let pcel_addr = pcel.fresh(out, oclk, &out_ty_, loc, PMach::NvGpu);
            let pcel_addr = match pcel.lookup(loc, PMach::NvGpu) {
              None => panic!("bug"),
              Some(rep) => rep.addr.get()
            };
            let base0 = TL_PCTX.with(|pctx| {
              let (dptr, _) = pctx.nvgpu.as_ref().unwrap().lookup_dev(pcel_addr).unwrap();
              dptr
            });
            let base = base0 + v_ty.pointer_offset();
            let inc = spec.o_dtype.size_bytes() as u64;
            let blk_row_len = tys.l_blk_outer as u64;
            let blk_col_len = tys.r_blk_outer as u64;
            let stride = if arg_ty_[0].ndim() == 2 && arg_ty_[1].ndim() == 2 && out_ty_.ndim() == 2 {
              out_ty_.shape[1] as u64
            } else if arg_ty_[0].ndim() == 4 && arg_ty_[1].ndim() == 4 && out_ty_.ndim() == 4 {
              (out_ty_.shape[2] * out_ty_.shape[3]) as u64
            } else {
              panic!("bug");
            };
            let mut tmp = self.tmp_c.borrow_mut();
            tmp.clear();
            for j in 0 .. tys.nrow as u64 {
              for i in 0 .. tys.ncol as u64 {
                tmp.push(base + inc * (blk_col_len * i + stride * blk_row_len * j));
              }
            }
          }
          _ => panic!("bug")
        }
      }
    }
    //println!("DEBUG: BlockMatrixMulF16F32GpuThunkImpl::_enter: tmp_a=[0x{:016x}]", self.tmp_a.borrow()[0]);
    //println!("DEBUG: BlockMatrixMulF16F32GpuThunkImpl::_enter: tmp_b=[0x{:016x}]", self.tmp_b.borrow()[0]);
    //println!("DEBUG: BlockMatrixMulF16F32GpuThunkImpl::_enter: tmp_c=[0x{:016x}]", self.tmp_c.borrow()[0]);
    let b_gputy = match spec.l_dtype {
      Dtype::F32 => CUDA_R_32F,
      Dtype::F16 => CUDA_R_16F,
      Dtype::Bf16 => CUDA_R_16BF,
      _ => unimplemented!()
    };
    let a_gputy = match spec.r_dtype {
      Dtype::F32 => CUDA_R_32F,
      Dtype::F16 => CUDA_R_16F,
      Dtype::Bf16 => CUDA_R_16BF,
      _ => unimplemented!()
    };
    let c_gputy = match spec.o_dtype {
      Dtype::F32 => CUDA_R_32F,
      Dtype::F16 => CUDA_R_16F,
      Dtype::Bf16 => CUDA_R_16BF,
      _ => unimplemented!()
    };
    TL_PCTX.with(|pctx| {
      let gpu = pctx.nvgpu.as_ref().unwrap();
      if cfg_debug() { println!("DEBUG: BlockMatrixMulF16F32GpuThunkImpl::_enter: gemm..."); }
      let ret = gpu.compute.sync();
      match ret {
        Err(e) => {
          println!("DEBUG: BlockMatrixMulF16F32GpuThunkImpl::_enter: pre gemm sync error: {:?}", e);
          Err(ThunkErr::Failure)
        }
        Ok(_) => Ok(())
      }?;
      let t1 = Stopwatch::tl_stamp();
      if cfg_debug() { println!("DEBUG: BlockMatrixMulF16F32GpuThunkImpl::_enter: pre gemm elapsed: {:.06} s", t1 - t0); }
      TL_CTX.with(|ctx| {
        if oclk.rst <= 0 {
          panic!("bug");
        } else if oclk.rst == 1 {
          ctx.timing.pregemm1.borrow_mut().push(t1 - t0);
        } else {
          ctx.timing.pregemm.borrow_mut().push(t1 - t0);
        }
      });
      let t0 = t1;
      if cfg_debug() {
      //let tmp_a = self.tmp_a.borrow();
      //let tmp_b = self.tmp_b.borrow();
      //let tmp_c = self.tmp_c.borrow();
      let alpha_ptr = self.alpha.as_ptr() as usize;
      let beta_ptr = self.beta.as_ptr() as usize;
      //let tmp_a_ptr = tmp_a.as_ptr() as usize;
      //let tmp_b_ptr = tmp_b.as_ptr() as usize;
      //let tmp_c_ptr = tmp_c.as_ptr() as usize;
      println!("DEBUG: BlockMatrixMulF16F32GpuThunkImpl::_enter: alpha ptr=0x{:016x}", alpha_ptr);
      println!("DEBUG: BlockMatrixMulF16F32GpuThunkImpl::_enter: beta ptr =0x{:016x}", beta_ptr);
      //println!("DEBUG: BlockMatrixMulF16F32GpuThunkImpl::_enter: tmp a ptr=0x{:016x}", tmp_a_ptr);
      //println!("DEBUG: BlockMatrixMulF16F32GpuThunkImpl::_enter: tmp b ptr=0x{:016x}", tmp_b_ptr);
      //println!("DEBUG: BlockMatrixMulF16F32GpuThunkImpl::_enter: tmp c ptr=0x{:016x}", tmp_c_ptr);
      }
      // FIXME FIXME: the arrays to blocks have to be in vmem...
      let (nblk, tmp_a_dptr, tmp_b_dptr, tmp_c_dptr) = {
        let nblk = self.tmp_c.borrow().len();
        assert_eq!(nblk, self.tmp_a.borrow().len());
        assert_eq!(nblk, self.tmp_b.borrow().len());
        assert!(nblk <= i32::max_value() as _);
        let ntxn = (nblk + 16 - 1) / 16;
        let staging_len = ntxn * 16 * 3;
        let staging_sz = staging_len * 8;
        assert!(staging_sz + 128 <= (1 << 16));
        let staging_buf = unsafe { from_raw_parts_mut(gpu.page_map.extrabuf.ptr as *mut u64, staging_len) };
        let tmp = self.tmp_a.borrow();
        assert_eq!(tmp.len(), nblk);
        for blk_idx in 0 .. nblk {
          staging_buf[blk_idx] = tmp[blk_idx];
        }
        for blk_idx in nblk .. ntxn * 16 {
          staging_buf[blk_idx] = 0;
        }
        drop(tmp);
        let tmp = self.tmp_b.borrow();
        assert_eq!(tmp.len(), nblk);
        for blk_idx in 0 .. nblk {
          staging_buf[ntxn * 16 + blk_idx] = tmp[blk_idx];
        }
        for blk_idx in nblk .. ntxn * 16 {
          staging_buf[ntxn * 16 + blk_idx] = 0;
        }
        drop(tmp);
        let tmp = self.tmp_c.borrow();
        for blk_idx in 0 .. nblk {
          staging_buf[ntxn * 16 * 2 + blk_idx] = tmp[blk_idx];
        }
        for blk_idx in nblk .. ntxn * 16 {
          staging_buf[ntxn * 16 * 2 + blk_idx] = 0;
        }
        drop(tmp);
        drop(staging_buf);
        gpu.hard_copy_raw_mem_to_vmem(gpu.mem_pool.extra_base, gpu.page_map.extrabuf.ptr, staging_sz);
        let a = gpu.mem_pool.extra_base;
        let b = gpu.mem_pool.extra_base + (ntxn * 128) as u64;
        let c = gpu.mem_pool.extra_base + (ntxn * 128 * 2) as u64;
        (nblk, a, b, c)
      };
      assert_eq!(nblk, (tys.nrow * tys.ncol) as usize);
      let ret = cublas_gemm_batched(
          &gpu.blas_ctx,
          colmajor_at, colmajor_bt,
          colmajor_m as _, colmajor_n as _, inner_len as _,
          self.alpha.as_ptr() as *const f32 as *const c_void,
          tmp_a_dptr, a_gputy, lda as _,
          tmp_b_dptr, b_gputy, ldb as _,
          self.beta.as_ptr() as *const f32 as *const c_void,
          tmp_c_dptr, c_gputy, ldc as _,
          nblk as _,
          &gpu.compute,
      );
      match ret {
        Err(e) => {
          println!("DEBUG: BlockMatrixMulF16F32GpuThunkImpl::_enter: gemm error: {:?}", e);
          Err(ThunkErr::Failure)
        }
        Ok(_) => Ok(())
      }?;
      let ret = gpu.compute.sync();
      match ret {
        Err(e) => {
          println!("DEBUG: BlockMatrixMulF16F32GpuThunkImpl::_enter: gemm sync error: {:?}", e);
          Err(ThunkErr::Failure)
        }
        Ok(_) => Ok(())
      }?;
      let t1 = Stopwatch::tl_stamp();
      //drop(tmp_c);
      //drop(tmp_b);
      //drop(tmp_a);
      if cfg_debug() { println!("DEBUG: BlockMatrixMulF16F32GpuThunkImpl::_enter: gemm OK elapsed: {:.06} s", t1 - t0); }
      TL_CTX.with(|ctx| {
        if oclk.rst <= 0 {
          panic!("bug");
        } else if oclk.rst == 1 {
          ctx.timing.gemm1.borrow_mut().push(t1 - t0);
        } else {
          ctx.timing.gemm.borrow_mut().push(t1 - t0);
        }
      });
      Ok(())
    })
  }
}

#[cfg(feature = "nvgpu")]
impl ThunkImpl for BlockMatrixMulF16F32GpuThunkImpl {
  fn apply(&self, ctr: &CtxCtr, env: &mut CtxEnv, spec_: &dyn ThunkSpec_, param: &[ScalarVal_], arg: &[(CellPtr, Clock)], th: ThunkPtr, out: CellPtr, prev_oclk: Clock, oclk: Clock) -> ThunkResult {
    if cfg_debug() { println!("DEBUG: BlockMatrixMulF16F32GpuThunkImpl::apply"); }
    let mode = ThunkMode::Apply;
    self._enter(ctr, env, spec_, param, arg, th, out, prev_oclk, oclk, mode)
  }

  fn accumulate(&self, ctr: &CtxCtr, env: &mut CtxEnv, spec_: &dyn ThunkSpec_, param: &[ScalarVal_], arg: &[(CellPtr, Clock)], th: ThunkPtr, out: CellPtr, prev_oclk: Clock, oclk: Clock) -> ThunkResult {
    if cfg_debug() { println!("DEBUG: BlockMatrixMulF16F32GpuThunkImpl::accumulate"); }
    let mode = ThunkMode::Accumulate;
    self._enter(ctr, env, spec_, param, arg, th, out, prev_oclk, oclk, mode)
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct MemcpyThunkSpec;

impl ThunkSpec for MemcpyThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("memcpy")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn arity(&self) -> Option<(u16, u16)> {
    Some((1, 1))
  }

  fn param_count(&self) -> u16 {
    0
  }

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(arg[0])
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(arg[0].clone())
  }

  fn gen_impl_(&self, spec_dim: Vec<Dim>, pmach: PMach) -> Option<Rc<dyn ThunkImpl_>> {
    match pmach {
      #[cfg(feature = "nvgpu")]
      PMach::NvGpu => {
        Some(Rc::new(MemcpyNvgpuThunkImpl::default()))
      }
      _ => {
        println!("WARNING:MemcpyThunkSpec::gen_impl_: no impl for pmach={:?}", pmach);
        None
      }
    }
  }
}

#[cfg(feature = "nvgpu")]
#[derive(Default)]
pub struct MemcpyNvgpuThunkImpl;

#[cfg(feature = "nvgpu")]
impl ThunkImpl for MemcpyNvgpuThunkImpl {
  fn apply(&self, ctr: &CtxCtr, env: &mut CtxEnv, spec_: &dyn ThunkSpec_, _param: &[ScalarVal_], arg: &[(CellPtr, Clock)], th: ThunkPtr, out: CellPtr, prev_oclk: Clock, oclk: Clock) -> ThunkResult {
    //if cfg_debug() { println!("DEBUG: MemcpyNvgpuThunkImpl::apply"); }
    if cfg_debug() { println!("DEBUG: MemcpyNvgpuThunkImpl::apply: arg={:?} out={:?} oclk={:?}", arg, out, oclk); }
    let spec = spec_.as_any().downcast_ref::<MemcpyThunkSpec>().unwrap();
    let mut arg_ty_ = Vec::with_capacity(arg.len());
    for &(x, _) in arg.iter() {
      match env._lookup_ref_(x) {
        Err(_) => panic!("bug"),
        Ok(e) => {
          arg_ty_.push(e.ty.clone());
        }
      }
    }
    let out_ty_ = ThunkSpec::out_ty_(spec, &arg_ty_).unwrap();
    let arg_sz = arg_ty_[0].packed_span_bytes();
    let out_sz = out_ty_.packed_span_bytes();
    assert_eq!(arg_sz, out_sz);
    let sz = out_sz as usize;
    TL_PCTX.with(|pctx| {
      let gpu = pctx.nvgpu.as_ref().unwrap();
      let ret = gpu.compute.sync();
      match ret {
        Err(e) => {
          println!("DEBUG: MemcpyNvgpuThunkImpl::apply: pre sync error: {:?}", e);
          Err(ThunkErr::Failure)
        }
        Ok(_) => Ok(())
      }?;
      let loc = gpu.device_locus();
      let src_dptr = match env.pread_ref_(arg[0].0, arg[0].1, loc) {
        Err(_) => panic!("bug"),
        Ok(e) => {
          assert_eq!(&e.ty, &arg_ty_[0]);
          match e.cel_ {
            &mut Cell_::Phy(.., ref pcel) => {
              //let pcel_addr = pcel.get(arg[0].0, arg[0].1, &arg_ty_[0], loc, PMach::NvGpu);
              let pcel_addr = match pcel.lookup(loc, PMach::NvGpu) {
                None => panic!("bug"),
                Some(rep) => rep.addr.get()
              };
              //let (dptr, _) = gpu.lookup_dev(pcel_addr).unwrap();
              let dptr = match gpu.lookup_dev(pcel_addr) {
                None => {
                  println!("DEBUG: MemcpyNvgpuThunkImpl::apply: no dptr for addr={:?}", pcel_addr);
                  panic!("bug");
                }
                Some((dptr, _)) => dptr
              };
              dptr
              // FIXME
            }
            _ => panic!("bug")
          }
        }
      };
      let dst_dptr = match env.pwrite_ref_(out, oclk, loc) {
        Err(_) => panic!("bug"),
        Ok(e) => {
          assert_eq!(&e.ty, &out_ty_);
          match e.cel_ {
            &mut Cell_::Phy(.., ref pcel) => {
              //let pcel_addr = pcel.fresh(out, oclk, &out_ty_, loc, PMach::NvGpu);
              let pcel_addr = match pcel.lookup(loc, PMach::NvGpu) {
                None => panic!("bug"),
                Some(rep) => rep.addr.get()
              };
              let (dptr, _) = gpu.lookup_dev(pcel_addr).unwrap();
              dptr
              // FIXME
            }
            _ => panic!("bug")
          }
        }
      };
      let ret = cuda_memcpy_async(
          dst_dptr,
          src_dptr,
          sz,
          &gpu.compute,
      );
      match ret {
        Err(e) => {
          println!("DEBUG: MemcpyNvgpuThunkImpl::apply: error: {:?}", e);
          Err(ThunkErr::Failure)
        }
        Ok(_) => Ok(())
      }?;
      let ret = gpu.compute.sync();
      match ret {
        Err(e) => {
          println!("DEBUG: MemcpyNvgpuThunkImpl::apply: sync error: {:?}", e);
          Err(ThunkErr::Failure)
        }
        Ok(_) => Ok(())
      }?;
      //let t1 = Stopwatch::tl_stamp();
      //println!("DEBUG: MemcpyNvpuThunkImpl::apply: OK elapsed: {:.06} s", t1 - t0);
      Ok(())
    })
  }
}
