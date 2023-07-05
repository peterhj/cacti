use super::*;
use crate::algo::fp::{TotalOrd};
use crate::cell::{DtypeExt, Dim, ScalarVal_};
use crate::op::*;
use cacti_gpu_cu_ffi::{cublas_gemm_batched};
use cacti_gpu_cu_ffi::types::{CUDA_R_32F, CUDA_R_16F, CUDA_R_16BF};

use futhark_ffi::{Abi};
use futhark_syntax::{Exp};

use std::borrow::{Cow};
use std::cell::{Cell};
use std::ffi::{c_void};
//use std::io::{Write};
use std::rc::{Weak};
use std::slice::{from_raw_parts, from_raw_parts_mut};

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
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
    //Ok(Dim{ndim: 0, dtype: T::dtype()})
    unimplemented!();
  }

  fn out_ty_(&self, _arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    // FIXME FIXME
    //Ok(CellType{shape: Vec::new(), dtype: T::dtype()})
    unimplemented!();
  }

  fn gen_futhark(&self, _arg: &[Dim]) -> Result<FutharkThunkCode, FutharkGenErr> {
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
    let mut code = FutharkThunkCode::default();
    code.body.push(s);
    code.into()
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct NopFutThunkSpec;

impl FutharkThunkSpec for NopFutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.nop")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(arg[0])
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(arg[0].clone())
  }

  fn gen_futhark(&self, _arg: &[Dim]) -> Result<FutharkThunkCode, FutharkGenErr> {
    let mut code = FutharkThunkCode::default();
    code.append(r"let {%1} = {%0} in");
    code.into()
  }

  fn pop_adj(&self, _arg: &[(CellPtr, Clock)], _out: CellPtr, _out_clk: Clock, out_adj: CellPtr, arg_adj: &mut [CellPtr]) -> Option<Result<(), ThunkAdjErr>> {
    arg_adj[0] += out_adj;
    Some(Ok(()))
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
//pub struct SetScalarFutThunkSpec<T> { pub val: T }
pub struct SetScalarFutThunkSpec { pub val: ScalarVal_ }

//impl<T: DtypeExt + Copy + Eq + Any> FutharkThunkSpec for SetScalarFutThunkSpec<T> {}
impl FutharkThunkSpec for SetScalarFutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.set_scalar")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 0;
    abi.arityout = 1;
    abi
  }

  fn out_dim(&self, _arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    //Ok(Dim{ndim: 0, dtype: T::dtype()})
    let dtype = self.val.dtype();
    Ok(Dim{ndim: 0, dtype})
  }

  fn out_ty_(&self, _arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    //Ok(CellType{shape: Vec::new(), dtype: T::dtype()})
    let dtype = self.val.dtype();
    Ok(CellType{shape: Vec::new(), dtype})
  }

  /*fn scalar_val(&self) -> Option<&dyn DtypeExt> {
    Some(&self.val)
  }*/

  fn gen_futhark(&self, _arg: &[Dim]) -> Result<FutharkThunkCode, FutharkGenErr> {
    //let fmt = FutharkNumFormatter::default();
    let mut code = FutharkThunkCode::default();
    // FIXME FIXME: futhark treats actual scalars as simply pointers to cpu mem.
    /*body:     vec![format!("let {{%0}} = [{}] in", fmt.format(&self.val))],*/
    //code.body.push(format!("let {{%0}} = {} in", fmt.format(&self.val)));
    code.append(format!(r"let {{%0}} = {} in", self.val.format_futhark()));
    code.into()
  }
}

/*#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SetScalar1dFutThunkSpec<T> { pub val: T }

impl<T: DtypeExt + Copy + Eq + Any> FutharkThunkSpec for SetScalar1dFutThunkSpec<T> {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.set_scalar_1d")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 0;
    abi.arityout = 1;
    abi
  }

  fn out_dim(&self, _arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(Dim{ndim: 1, dtype: T::dtype()})
  }

  fn out_ty_(&self, _arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Err(ThunkTypeErr::Nondeterministic)
  }

  fn scalar_val(&self) -> Option<&dyn DtypeExt> {
    Some(&self.val)
  }

  fn gen_futhark(&self, _arg: &[Dim]) -> Result<FutharkThunkCode, FutharkGenErr> {
    let fmt = FutharkNumFormatter::default();
    let mut code = FutharkThunkCode::default();
    code.body.push(format!("let {{%0}} = replicate {{%0.s[0]}} {} in", fmt.format(&self.val)));
    code.into()
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SetScalar2dFutThunkSpec<T> { pub val: T }

impl<T: DtypeExt + Copy + Eq + Any> FutharkThunkSpec for SetScalar2dFutThunkSpec<T> {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.set_scalar_2d")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 0;
    abi.arityout = 1;
    abi
  }

  fn out_dim(&self, _arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(Dim{ndim: 2, dtype: T::dtype()})
  }

  fn out_ty_(&self, _arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Err(ThunkTypeErr::Nondeterministic)
  }

  fn scalar_val(&self) -> Option<&dyn DtypeExt> {
    Some(&self.val)
  }

  fn gen_futhark(&self, _arg: &[Dim]) -> Result<FutharkThunkCode, FutharkGenErr> {
    let fmt = FutharkNumFormatter::default();
    let mut code = FutharkThunkCode::default();
    code.body.push(format!("let t0 = replicate ({{%0.s[0]}} * {{%0.s[1]}}) {} in", fmt.format(&self.val)));
    code.body.push(format!("let {{%0}} = unflatten {{%0.s[0]}} {{%0.s[1]}} t0 in"));
    code.into()
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SetScalar3dFutThunkSpec<T> { pub val: T }

impl<T: DtypeExt + Copy + Eq + Any> FutharkThunkSpec for SetScalar3dFutThunkSpec<T> {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.set_scalar_3d")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 0;
    abi.arityout = 1;
    abi
  }

  fn out_dim(&self, _arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(Dim{ndim: 3, dtype: T::dtype()})
  }

  fn out_ty_(&self, _arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Err(ThunkTypeErr::Nondeterministic)
  }

  fn scalar_val(&self) -> Option<&dyn DtypeExt> {
    Some(&self.val)
  }

  fn gen_futhark(&self, _arg: &[Dim]) -> Result<FutharkThunkCode, FutharkGenErr> {
    let fmt = FutharkNumFormatter::default();
    let mut code = FutharkThunkCode::default();
    code.body.push(format!("let t0 = replicate ({{%0.s[0]}} * {{%0.s[1]}} * {{%0.s[2]}}) {} in", fmt.format(&self.val)));
    code.body.push(format!("let {{%0}} = unflatten_3d {{%0.s[0]}} {{%0.s[1]}} {{%0.s[2]}} t0 in"));
    code.into()
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SetScalar4dFutThunkSpec<T> { pub val: T }

impl<T: DtypeExt + Copy + Eq + Any> FutharkThunkSpec for SetScalar4dFutThunkSpec<T> {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.set_scalar_4d")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 0;
    abi.arityout = 1;
    abi
  }

  fn out_dim(&self, _arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(Dim{ndim: 4, dtype: T::dtype()})
  }

  fn out_ty_(&self, _arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Err(ThunkTypeErr::Nondeterministic)
  }

  fn scalar_val(&self) -> Option<&dyn DtypeExt> {
    Some(&self.val)
  }

  fn gen_futhark(&self, _arg: &[Dim]) -> Result<FutharkThunkCode, FutharkGenErr> {
    let fmt = FutharkNumFormatter::default();
    let mut code = FutharkThunkCode::default();
    code.body.push(format!("let t0 = replicate ({{%0.s[0]}} * {{%0.s[1]}} * {{%0.s[2]}} * {{%0.s[3]}}) {} in", fmt.format(&self.val)));
    code.body.push(format!("let {{%0}} = unflatten_4d {{%0.s[0]}} {{%0.s[1]}} {{%0.s[2]}} {{%0.s[3]}} t0 in"));
    code.into()
  }
}*/

// FIXME: only need new dtype.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct CastFutThunkSpec { pub org_dtype: Dtype, pub new_dtype: Dtype }

impl FutharkThunkSpec for CastFutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.cast")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(Dim{ndim: arg[0].ndim, dtype: self.new_dtype})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(CellType{shape: arg[0].shape.clone(), dtype: self.new_dtype})
  }

  fn gen_futhark(&self, arg: &[Dim]) -> Result<FutharkThunkCode, FutharkGenErr> {
    FutharkThunkCode::map_nd(arg[0], format!(r"\u -> {}.{} u",
                                             self.new_dtype.format_futhark(),
                                             self.org_dtype.format_futhark()))
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

  fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    if arg[0].dtype != Dtype::Bfloat16 {
      return Err(ThunkDimErr::_Bot);
    }
    Ok(Dim{ndim: arg[0].ndim, dtype: Dtype::Fp16})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    if arg[0].dtype != Dtype::Bfloat16 {
      return Err(ThunkTypeErr::_Bot);
    }
    Ok(CellType{shape: arg[0].shape.clone(), dtype: Dtype::Fp16})
  }

  fn gen_futhark(&self, arg: &[Dim]) -> Result<FutharkThunkCode, FutharkGenErr> {
    FutharkThunkCode::map_nd(arg[0], r"\u -> f16.f32 (f32.from_bits ((u32.u16 u) << 16))")
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

  fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    if arg[0].dtype != Dtype::Bfloat16 {
      return Err(ThunkDimErr::_Bot);
    }
    Ok(Dim{ndim: arg[0].ndim, dtype: Dtype::Fp32})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    if arg[0].dtype != Dtype::Bfloat16 {
      return Err(ThunkTypeErr::_Bot);
    }
    Ok(CellType{shape: arg[0].shape.clone(), dtype: Dtype::Fp32})
  }

  fn gen_futhark(&self, arg: &[Dim]) -> Result<FutharkThunkCode, FutharkGenErr> {
    FutharkThunkCode::map_nd(arg[0], r"\u -> f32.from_bits ((u32.u16 u) << 16)")
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

  fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    if arg[0].dtype != Dtype::Fp32 {
      return Err(ThunkDimErr::_Bot);
    }
    Ok(Dim{ndim: arg[0].ndim, dtype: Dtype::Bfloat16})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    if arg[0].dtype != Dtype::Fp32 {
      return Err(ThunkTypeErr::_Bot);
    }
    Ok(CellType{shape: arg[0].shape.clone(), dtype: Dtype::Bfloat16})
  }

  fn gen_futhark(&self, arg: &[Dim]) -> Result<FutharkThunkCode, FutharkGenErr> {
    FutharkThunkCode::map_nd(arg[0], r"\u -> u16.u32 ((f32.to_bits u) >> 16)")
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

  fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }

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

  fn gen_futhark(&self, arg: &[Dim]) -> Result<FutharkThunkCode, FutharkGenErr> {
    let out = FutharkThunkSpec::out_dim(self, arg).map_err(|e| e.into_gen())?;
    //let fmt = FutharkNumFormatter::default();
    match out.ndim {
      0 => {
        unimplemented!();
      }
      1 => {
        unimplemented!();
      }
      2 => {
        let mut code = FutharkThunkCode::default();
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
        code.body.push(format!(r"let {{%1}} = unflatten {{%0.s[0]}} {} t1 in",
            self.inner_len,
        ));
        code.into()
      }
      3 => {
        let mut code = FutharkThunkCode::default();
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
        code.body.push(format!(r"let {{%1}} = unflatten_3d {{%0.s[0]}} {{%0.s[1]}} {} t1 in",
            self.inner_len,
        ));
        code.into()
      }
      4 => {
        unimplemented!();
      }
      _ => unimplemented!()
    }
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

  fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 2;
    abi.arityout = 1;
    abi
  }

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

  fn gen_futhark(&self, arg: &[Dim]) -> Result<FutharkThunkCode, FutharkGenErr> {
    let out = FutharkThunkSpec::out_dim(self, arg).map_err(|e| e.into_gen())?;
    match out.ndim {
      0 => {
        unimplemented!();
      }
      1 => {
        unimplemented!();
      }
      2 => {
        let mut code = FutharkThunkCode::default();
        code.cfg.emit_arg_shapes = true;
        code.append(format!(r"let a_pre = {{%0.s[0]}} * {{%0.s[1]}} in"));
        code.append(format!(r"let a_suf = {{%0.s[2]}} in"));
        code.append(format!(r"let a = a_pre * a_suf in"));
        code.append(format!(r"let t_val = flatten_3d {{%0}} :> [a]{} in", arg[0].dtype.format_futhark()));
        code.append(format!(r"let t_val = unflatten a_pre a_suf t_val in"));
        /*code.append(format!(r"let t_key = flatten {{%1}} :> [a_pre]{} in", arg[1].dtype.format_futhark()));*/
        code.append(format!(r"let t1 = flatten {{%1}} :> [a_pre]{} in", arg[1].dtype.format_futhark()));
        code.append(format!(r"let t_iota = iota a_pre in"));
        code.append(format!(r"let t_key = map2 (\idx k -> let k = (i64.{} k) in (assert (k >= 0 && k < a_suf) k) + a_suf * idx) t_iota t1 in",
            arg[1].dtype.format_futhark(),
        ));
        code.append(format!(r"let t2 = map2 (\k v -> v[k]) t_key t_val in"));
        code.append(format!(r"let {{%2}} = unflatten {{%0.s[0]}} {{%0.s[1]}} t2 in"));
        code.into()
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
  }

  fn pop_adj(&self, arg: &[(CellPtr, Clock)], out: CellPtr, out_clk: Clock, out_adj: CellPtr, arg_adj: &mut [CellPtr]) -> Option<Result<(), ThunkAdjErr>> {
    let arg0_ty = arg[0].0.type_();
    let arg0_nd = arg0_ty.ndim() as usize;
    let inner_len = arg0_ty.shape[arg0_nd - 1];
    arg_adj[0] += out_adj.inner_inv_select(arg[1].0, inner_len);
    Some(Ok(()))
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

  fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 2;
    abi.arityout = 1;
    abi
  }

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

  fn gen_futhark(&self, arg: &[Dim]) -> Result<FutharkThunkCode, FutharkGenErr> {
    let out = FutharkThunkSpec::out_dim(self, arg).map_err(|e| e.into_gen())?;
    match out.ndim {
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
        let mut code = FutharkThunkCode::default();
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
        code.append(format!(r"let {{%2}} = unflatten_3d {{%0.s[0]}} {{%0.s[1]}} {} t2 in",
            self.inner_len,
        ));
        code.into()
      }
      4 => {
        unimplemented!();
      }
      _ => {
        unimplemented!();
      }
    }
  }

  fn pop_adj(&self, arg: &[(CellPtr, Clock)], out: CellPtr, out_clk: Clock, out_adj: CellPtr, arg_adj: &mut [CellPtr]) -> Option<Result<(), ThunkAdjErr>> {
    arg_adj[0] += out_adj.inner_select(arg[1].0);
    Some(Ok(()))
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct AddScalarF32FutThunkSpec { pub val: TotalOrd<f32> }

impl FutharkThunkSpec for AddScalarF32FutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.f32.add_scalar")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(Dim{ndim: arg[0].ndim, dtype: f32::dtype()})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(CellType{shape: arg[0].shape.clone(), dtype: f32::dtype()})
  }

  fn gen_futhark(&self, arg: &[Dim]) -> Result<FutharkThunkCode, FutharkGenErr> {
    let fmt = FutharkNumFormatter::default();
    FutharkThunkCode::map_nd(arg[0], format!(r"\u -> u + {}", fmt.format(&self.val)))
  }

  fn pop_adj(&self, _arg: &[(CellPtr, Clock)], _out: CellPtr, _out_clk: Clock, out_adj: CellPtr, arg_adj: &mut [CellPtr]) -> Option<Result<(), ThunkAdjErr>> {
    arg_adj[0] += out_adj;
    Some(Ok(()))
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct AddFutThunkSpec;

impl FutharkThunkSpec for AddFutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.add")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 2;
    abi.arityout = 1;
    abi
  }

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

  fn gen_futhark(&self, arg: &[Dim]) -> Result<FutharkThunkCode, FutharkGenErr> {
    FutharkThunkCode::map2_nd(arg[0], arg[1], r"+")
  }

  fn pop_adj(&self, _arg: &[(CellPtr, Clock)], _out: CellPtr, _out_clk: Clock, out_adj: CellPtr, arg_adj: &mut [CellPtr]) -> Option<Result<(), ThunkAdjErr>> {
    arg_adj[0] += out_adj;
    arg_adj[1] += out_adj;
    Some(Ok(()))
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SubScalarF32FutThunkSpec { pub val: TotalOrd<f32> }

impl FutharkThunkSpec for SubScalarF32FutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.f32.sub_scalar")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(Dim{ndim: arg[0].ndim, dtype: f32::dtype()})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(CellType{shape: arg[0].shape.clone(), dtype: f32::dtype()})
  }

  fn gen_futhark(&self, arg: &[Dim]) -> Result<FutharkThunkCode, FutharkGenErr> {
    let fmt = FutharkNumFormatter::default();
    FutharkThunkCode::map_nd(arg[0], format!(r"\u -> u - {}", fmt.format(&self.val)))
  }

  fn pop_adj(&self, _arg: &[(CellPtr, Clock)], _out: CellPtr, _out_clk: Clock, out_adj: CellPtr, arg_adj: &mut [CellPtr]) -> Option<Result<(), ThunkAdjErr>> {
    arg_adj[0] += out_adj;
    Some(Ok(()))
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct SubFutThunkSpec;

impl FutharkThunkSpec for SubFutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.sub")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 2;
    abi.arityout = 1;
    abi
  }

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

  fn gen_futhark(&self, arg: &[Dim]) -> Result<FutharkThunkCode, FutharkGenErr> {
    FutharkThunkCode::map2_nd(arg[0], arg[1], r"-")
  }

  fn pop_adj(&self, _arg: &[(CellPtr, Clock)], _out: CellPtr, _out_clk: Clock, out_adj: CellPtr, arg_adj: &mut [CellPtr]) -> Option<Result<(), ThunkAdjErr>> {
    arg_adj[0] += out_adj;
    arg_adj[1] += -out_adj;
    Some(Ok(()))
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct MulScalarF32FutThunkSpec { pub val: TotalOrd<f32> }

impl FutharkThunkSpec for MulScalarF32FutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.f32.mul_scalar")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(Dim{ndim: arg[0].ndim, dtype: f32::dtype()})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(CellType{shape: arg[0].shape.clone(), dtype: arg[0].dtype})
  }

  fn gen_futhark(&self, arg: &[Dim]) -> Result<FutharkThunkCode, FutharkGenErr> {
    // FIXME: param.
    let fmt = FutharkNumFormatter::default();
    FutharkThunkCode::map_nd(arg[0], format!(r"\u -> u * {}", fmt.format(&self.val)))
  }

  fn pop_adj(&self, _arg: &[(CellPtr, Clock)], _out: CellPtr, _out_clk: Clock, out_adj: CellPtr, arg_adj: &mut [CellPtr]) -> Option<Result<(), ThunkAdjErr>> {
    arg_adj[0] += out_adj * self.val.0;
    Some(Ok(()))
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct MulFutThunkSpec;

impl FutharkThunkSpec for MulFutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.mul")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 2;
    abi.arityout = 1;
    abi
  }

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

  fn gen_futhark(&self, arg: &[Dim]) -> Result<FutharkThunkCode, FutharkGenErr> {
    FutharkThunkCode::map2_nd(arg[0], arg[1], r"*")
  }

  fn pop_adj(&self, arg: &[(CellPtr, Clock)], _out: CellPtr, _out_clk: Clock, out_adj: CellPtr, arg_adj: &mut [CellPtr]) -> Option<Result<(), ThunkAdjErr>> {
    arg_adj[0] += out_adj * arg[1].0;
    arg_adj[1] += arg[0].0 * out_adj;
    Some(Ok(()))
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct DivScalarF32FutThunkSpec { pub val: TotalOrd<f32> }

impl FutharkThunkSpec for DivScalarF32FutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.f32.div_scalar")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(Dim{ndim: arg[0].ndim, dtype: f32::dtype()})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(CellType{shape: arg[0].shape.clone(), dtype: f32::dtype()})
  }

  fn gen_futhark(&self, arg: &[Dim]) -> Result<FutharkThunkCode, FutharkGenErr> {
    // FIXME: param.
    let fmt = FutharkNumFormatter::default();
    FutharkThunkCode::map_nd(arg[0], format!(r"\u -> u / {}", fmt.format(&self.val)))
  }

  fn pop_adj(&self, _arg: &[(CellPtr, Clock)], _out: CellPtr, _out_clk: Clock, out_adj: CellPtr, arg_adj: &mut [CellPtr]) -> Option<Result<(), ThunkAdjErr>> {
    arg_adj[0] += out_adj / self.val.0;
    Some(Ok(()))
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct DivFutThunkSpec;

impl FutharkThunkSpec for DivFutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.div")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 2;
    abi.arityout = 1;
    abi
  }

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

  fn gen_futhark(&self, arg: &[Dim]) -> Result<FutharkThunkCode, FutharkGenErr> {
    FutharkThunkCode::map2_nd(arg[0], arg[1], r"/")
  }

  fn pop_adj(&self, arg: &[(CellPtr, Clock)], _out: CellPtr, _out_clk: Clock, out_adj: CellPtr, arg_adj: &mut [CellPtr]) -> Option<Result<(), ThunkAdjErr>> {
    arg_adj[0] += out_adj / arg[1].0;
    arg_adj[1] += (out_adj * arg[0].0) / -arg[1].0.square();
    Some(Ok(()))
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

  fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(Dim{ndim: arg[0].ndim, dtype: arg[0].dtype})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(CellType{shape: arg[0].shape.clone(), dtype: arg[0].dtype})
  }

  fn gen_futhark(&self, arg: &[Dim]) -> Result<FutharkThunkCode, FutharkGenErr> {
    FutharkThunkCode::map_nd(arg[0], r"\u -> -u")
    //FutharkThunkCode::map_nd(arg[0], format!(r"\u -> ({}.neg u)", arg[0].dtype.format_futhark()))
  }

  fn pop_adj(&self, _arg: &[(CellPtr, Clock)], _out: CellPtr, _out_clk: Clock, out_adj: CellPtr, arg_adj: &mut [CellPtr]) -> Option<Result<(), ThunkAdjErr>> {
    arg_adj[0] += -out_adj;
    Some(Ok(()))
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

  fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    assert_eq!(arg[0].dtype, Dtype::Fp16);
    Ok(Dim{ndim: arg[0].ndim, dtype: arg[0].dtype})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    assert_eq!(arg[0].dtype, Dtype::Fp16);
    Ok(CellType{shape: arg[0].shape.clone(), dtype: arg[0].dtype})
  }

  fn gen_futhark(&self, arg: &[Dim]) -> Result<FutharkThunkCode, FutharkGenErr> {
    //FutharkThunkCode::map_nd(arg[0], r"\u -> -u")
    //FutharkThunkCode::map_nd(arg[0], r"\u -> 0.0f16 - u")
    /*let mut code = FutharkThunkCode::default();
    code.append(r"let c = f16.from_bits 0xbc00u16 in");
    code.append_map_nd(arg[0], r"\u -> c * u")?;
    code.into()*/
    FutharkThunkCode::map_nd(arg[0], r"\u -> f16.from_bits ((f16.to_bits u) ^ 0x8000u16)")
  }

  fn pop_adj(&self, _arg: &[(CellPtr, Clock)], _out: CellPtr, _out_clk: Clock, out_adj: CellPtr, arg_adj: &mut [CellPtr]) -> Option<Result<(), ThunkAdjErr>> {
    arg_adj[0] += -out_adj;
    Some(Ok(()))
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

  fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(Dim{ndim: arg[0].ndim, dtype: arg[0].dtype})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(CellType{shape: arg[0].shape.clone(), dtype: arg[0].dtype})
  }

  fn gen_futhark(&self, arg: &[Dim]) -> Result<FutharkThunkCode, FutharkGenErr> {
    FutharkThunkCode::map_nd(arg[0], r"\u -> u * u")
  }

  fn pop_adj(&self, arg: &[(CellPtr, Clock)], _out: CellPtr, _out_clk: Clock, out_adj: CellPtr, arg_adj: &mut [CellPtr]) -> Option<Result<(), ThunkAdjErr>> {
    // FIXME FIXME
    unimplemented!();
    /*
    let dtype = _;
    arg_adj[0] += out_adj * arg[0].0 * ScalarVal_::two(dtype);
    Some(Ok(()))
    */
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

  fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(Dim{ndim: arg[0].ndim, dtype: arg[0].dtype})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(CellType{shape: arg[0].shape.clone(), dtype: arg[0].dtype})
  }

  fn gen_futhark(&self, arg: &[Dim]) -> Result<FutharkThunkCode, FutharkGenErr> {
    FutharkThunkCode::map_nd(arg[0], r"recip")
  }

  fn pop_adj(&self, arg: &[(CellPtr, Clock)], _out: CellPtr, _out_clk: Clock, out_adj: CellPtr, arg_adj: &mut [CellPtr]) -> Option<Result<(), ThunkAdjErr>> {
    arg_adj[0] += out_adj / -arg[0].0.square();
    Some(Ok(()))
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

  fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(Dim{ndim: arg[0].ndim, dtype: arg[0].dtype})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(CellType{shape: arg[0].shape.clone(), dtype: arg[0].dtype})
  }

  fn gen_futhark(&self, arg: &[Dim]) -> Result<FutharkThunkCode, FutharkGenErr> {
    FutharkThunkCode::map_nd(arg[0], format!(r"{}.sqrt", arg[0].dtype.format_futhark()))
  }

  fn pop_adj(&self, arg: &[(CellPtr, Clock)], _out: CellPtr, _out_clk: Clock, out_adj: CellPtr, arg_adj: &mut [CellPtr]) -> Option<Result<(), ThunkAdjErr>> {
    // FIXME FIXME
    unimplemented!();
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

  fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(Dim{ndim: arg[0].ndim, dtype: arg[0].dtype})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(CellType{shape: arg[0].shape.clone(), dtype: arg[0].dtype})
  }

  fn gen_futhark(&self, arg: &[Dim]) -> Result<FutharkThunkCode, FutharkGenErr> {
    // FIXME FIXME
    FutharkThunkCode::map_nd(arg[0], format!(r"\u -> recip ({}.sqrt u)", arg[0].dtype.format_futhark()))
    //FutharkThunkCode::map_nd(arg[0], r"\u -> rsqrt u")
  }

  fn pop_adj(&self, arg: &[(CellPtr, Clock)], _out: CellPtr, _out_clk: Clock, out_adj: CellPtr, arg_adj: &mut [CellPtr]) -> Option<Result<(), ThunkAdjErr>> {
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

  fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(Dim{ndim: arg[0].ndim, dtype: arg[0].dtype})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(CellType{shape: arg[0].shape.clone(), dtype: arg[0].dtype})
  }

  fn gen_futhark(&self, arg: &[Dim]) -> Result<FutharkThunkCode, FutharkGenErr> {
    FutharkThunkCode::map_nd(arg[0], format!(r"{}.cos", arg[0].dtype.format_futhark()))
  }

  fn pop_adj(&self, arg: &[(CellPtr, Clock)], _out: CellPtr, _out_clk: Clock, out_adj: CellPtr, arg_adj: &mut [CellPtr]) -> Option<Result<(), ThunkAdjErr>> {
    arg_adj[0] += out_adj * -arg[0].0.sin();
    Some(Ok(()))
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

  fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(Dim{ndim: arg[0].ndim, dtype: arg[0].dtype})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(CellType{shape: arg[0].shape.clone(), dtype: arg[0].dtype})
  }

  fn gen_futhark(&self, arg: &[Dim]) -> Result<FutharkThunkCode, FutharkGenErr> {
    FutharkThunkCode::map_nd(arg[0], format!(r"{}.sin", arg[0].dtype.format_futhark()))
  }

  fn pop_adj(&self, arg: &[(CellPtr, Clock)], _out: CellPtr, _out_clk: Clock, out_adj: CellPtr, arg_adj: &mut [CellPtr]) -> Option<Result<(), ThunkAdjErr>> {
    arg_adj[0] += out_adj * arg[0].0.cos();
    Some(Ok(()))
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

  fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(Dim{ndim: arg[0].ndim, dtype: arg[0].dtype})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(CellType{shape: arg[0].shape.clone(), dtype: arg[0].dtype})
  }

  fn gen_futhark(&self, arg: &[Dim]) -> Result<FutharkThunkCode, FutharkGenErr> {
    FutharkThunkCode::map_nd(arg[0], format!(r"{}.exp", arg[0].dtype.format_futhark()))
  }

  fn pop_adj(&self, arg: &[(CellPtr, Clock)], _out: CellPtr, _out_clk: Clock, out_adj: CellPtr, arg_adj: &mut [CellPtr]) -> Option<Result<(), ThunkAdjErr>> {
    arg_adj[0] += out_adj * arg[0].0.exp();
    Some(Ok(()))
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

  fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(Dim{ndim: arg[0].ndim, dtype: arg[0].dtype})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(CellType{shape: arg[0].shape.clone(), dtype: arg[0].dtype})
  }

  fn gen_futhark(&self, arg: &[Dim]) -> Result<FutharkThunkCode, FutharkGenErr> {
    FutharkThunkCode::map_nd(arg[0], format!(r"\u -> (({}.exp u) - ({}.exp (-u))) / (({}.exp u) + ({}.exp (-u)))",
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

  fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(Dim{ndim: arg[0].ndim, dtype: f32::dtype()})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(CellType{shape: arg[0].shape.clone(), dtype: arg[0].dtype})
  }

  fn gen_futhark(&self, arg: &[Dim]) -> Result<FutharkThunkCode, FutharkGenErr> {
    // FIXME FIXME
    FutharkThunkCode::map_nd(arg[0], format!(r"\u -> u ** (f32.i64 {})", self.exp))
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

  fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(Dim{ndim: arg[0].ndim, dtype: arg[0].dtype})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(CellType{shape: arg[0].shape.clone(), dtype: arg[0].dtype})
  }

  fn gen_futhark(&self, arg: &[Dim]) -> Result<FutharkThunkCode, FutharkGenErr> {
    FutharkThunkCode::map_nd(arg[0], format!(r"{}.log", arg[0].dtype.format_futhark()))
  }

  fn pop_adj(&self, arg: &[(CellPtr, Clock)], _out: CellPtr, _out_clk: Clock, out_adj: CellPtr, arg_adj: &mut [CellPtr]) -> Option<Result<(), ThunkAdjErr>> {
    arg_adj[0] += out_adj / arg[0].0;
    Some(Ok(()))
  }
}

/*#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct InnerMax3dThunkSpec;

impl FutharkThunk_ for InnerMax3dThunkSpec {
  fn _arg_count(&self) -> u8 {
    1
  }

  fn _body(&self) -> Vec<u8> {
    b"map (\t1 -> map (\t2 -> [reduce max -inf t2]) t1) {%0}\n".to_owned()
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct InnerMean3dThunkSpec;

impl FutharkThunk_ for InnerSum3dThunkSpec {
  fn _arg_count(&self) -> u8 {
    1
  }

  fn _body(&self) -> Vec<u8> {
    b"map (\t1 -> map (\t2 -> [(reduce (+) 0 t2) / ({%t.0}.i64 (length t2))]) t1) {%0}\n".to_owned()
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct InnerSum3dThunkSpec;

impl FutharkThunk_ for InnerSum3dThunkSpec {
  fn _arg_count(&self) -> u8 {
    1
  }

  fn _body(&self) -> Vec<u8> {
    b"map (\t1 -> map (\t2 -> [reduce (+) 0 t2]) t1) {%0}\n".to_owned()
  }
}*/

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct FlatSumFutThunkSpec;

impl FutharkThunkSpec for FlatSumFutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.flat_sum")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(Dim{ndim: 0, dtype: arg[0].dtype})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(CellType{shape: Vec::new(), dtype: arg[0].dtype})
  }

  fn gen_futhark(&self, arg: &[Dim]) -> Result<FutharkThunkCode, FutharkGenErr> {
    match arg[0].ndim {
      0 => {
        let mut code = FutharkThunkCode::default();
        code.append(format!(r"let {{%1}} = {{%0}} in"));
        code.into()
      }
      1 => {
        let mut code = FutharkThunkCode::default();
        code.append(format!(r"let {{%1}} = reduce (+) 0 {{%0}} in"));
        code.into()
      }
      2 => {
        let mut code = FutharkThunkCode::default();
        code.append(format!(r"let t0 = flatten {{%0}} in"));
        code.append(format!(r"let t1 = reduce (+) 0 t0 in"));
        code.append(format!(r"let {{%1}} = t1 in"));
        code.into()
      }
      3 => {
        let mut code = FutharkThunkCode::default();
        code.append(format!(r"let t0 = flatten_3d {{%0}} in"));
        code.append(format!(r"let t1 = reduce (+) 0 t0 in"));
        code.append(format!(r"let {{%1}} = t1 in"));
        code.into()
      }
      4 => {
        let mut code = FutharkThunkCode::default();
        code.append(format!(r"let t0 = flatten_4d {{%0}} in"));
        code.append(format!(r"let t1 = reduce (+) 0 t0 in"));
        code.append(format!(r"let {{%1}} = t1 in"));
        code.into()
      }
      _ => {
        unimplemented!();
      }
    }
  }
}

/*#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct Sum1dFutThunkSpec;

impl FutharkThunkSpec for Sum1dFutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.sum1d")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    if arg[0].ndim() != 1 {
      return Err(ThunkDimErr::default());
    }
    Ok(Dim{ndim: 0, dtype: arg[0].dtype})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    if arg[0].ndim() != 1 {
      return Err(ThunkTypeErr::default());
    }
    Ok(CellType{shape: Vec::new(), dtype: arg[0].dtype})
  }

  fn gen_futhark(&self, _arg: &[Dim]) -> Result<FutharkThunkCode, FutharkGenErr> {
    let mut code = FutharkThunkCode::default();
    code.body.push(format!("let {{%1}} = reduce (+) 0 {{%0}} in"));
    code.into()
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct Sum2dFutThunkSpec;

impl FutharkThunkSpec for Sum2dFutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.sum2d")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    if arg[0].ndim() != 2 {
      return Err(ThunkDimErr::default());
    }
    Ok(Dim{ndim: 0, dtype: arg[0].dtype})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    if arg[0].ndim() != 2 {
      return Err(ThunkTypeErr::default());
    }
    Ok(CellType{shape: Vec::new(), dtype: arg[0].dtype})
  }

  fn gen_futhark(&self, _arg: &[Dim]) -> Result<FutharkThunkCode, FutharkGenErr> {
    let mut code = FutharkThunkCode::default();
    code.cfg.emit_arg_shapes = true;
    // FIXME: instead, could reshape and reduce once.
    /*body:     vec![format!("let {{%1}} = reduce (\t1 -> reduce (+) 0 t1) 0 {{%0}} in")],*/
    code.body.push(format!("let t0 = flatten {{%0}} in"));
    code.body.push(format!("let t1 = reduce (+) 0 t0 in"));
    code.body.push(format!("let {{%1}} = unflatten {{%0.s[0]}} {{%0.s[1]}} t1 in"));
    code.into()
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct Sum3dFutThunkSpec;

impl FutharkThunkSpec for Sum3dFutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.sum3d")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    if arg[0].ndim() != 3 {
      return Err(ThunkDimErr::default());
    }
    Ok(Dim{ndim: 0, dtype: arg[0].dtype})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    if arg[0].ndim() != 3 {
      return Err(ThunkTypeErr::default());
    }
    Ok(CellType{shape: Vec::new(), dtype: arg[0].dtype})
  }

  fn gen_futhark(&self, _arg: &[Dim]) -> Result<FutharkThunkCode, FutharkGenErr> {
    let mut code = FutharkThunkCode::default();
    code.cfg.emit_arg_shapes = true;
    // FIXME: instead, could reshape and reduce once.
    /*body:     vec![format!("let {{%1}} = reduce (\t2 -> reduce (\t1 -> reduce (+) 0 t1) 0 t2) 0 {{%0}} in")],*/
    code.body.push(format!("let t0 = flatten_3d {{%0}} in"));
    code.body.push(format!("let t1 = reduce (+) 0 t0 in"));
    code.body.push(format!("let {{%1}} = unflatten_3d {{%0.s[0]}} {{%0.s[1]}} {{%0.s[2]}} t1 in"));
    code.into()
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct Sum4dFutThunkSpec;

impl FutharkThunkSpec for Sum4dFutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.sum4d")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    if arg[0].ndim() != 4 {
      return Err(ThunkDimErr::default());
    }
    Ok(Dim{ndim: 0, dtype: arg[0].dtype})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    if arg[0].ndim() != 4 {
      return Err(ThunkTypeErr::default());
    }
    Ok(CellType{shape: Vec::new(), dtype: arg[0].dtype})
  }

  fn gen_futhark(&self, _arg: &[Dim]) -> Result<FutharkThunkCode, FutharkGenErr> {
    let mut code = FutharkThunkCode::default();
    code.cfg.emit_arg_shapes = true;
    // FIXME: instead, could reshape and reduce once.
    /*body:     vec![format!("let {{%1}} = reduce (\t3 -> reduce (\t2 -> reduce (\t1 -> reduce (+) 0 t1) 0 t2) 0 t3) 0 {{%0}} in")],*/
    code.body.push(format!("let t0 = flatten_4d {{%0}} in"));
    code.body.push(format!("let t1 = reduce (+) 0 t0 in"));
    code.body.push(format!("let {{%1}} = unflatten_4d {{%0.s[0]}} {{%0.s[1]}} {{%0.s[2]}} {{%0.s[3]}} t1 in"));
    code.into()
  }
}*/

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct InnerMeanFutThunkSpec;

impl FutharkThunkSpec for InnerMeanFutThunkSpec {
  fn debug_name(&self) -> Option<&'static str> {
    Some("futhark.inner_mean")
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Space)
  }

  fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }

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

  fn gen_futhark(&self, arg: &[Dim]) -> Result<FutharkThunkCode, FutharkGenErr> {
    let out = FutharkThunkSpec::out_dim(self, arg).map_err(|e| e.into_gen())?;
    match out.ndim {
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
        let mut code = FutharkThunkCode::default();
        code.cfg.emit_arg_shapes = true;
        code.append(format!(r"let a_pre = {{%0.s[0]}} * {{%0.s[1]}} * {{%0.s[2]}} in"));
        code.append(format!(r"let a_suf = {{%0.s[3]}} in"));
        code.append(format!(r"let a = a_pre * a_suf in"));
        code.append(format!(r"let t0 = flatten_4d {{%0}} :> [a]{} in",
            arg[0].dtype.format_futhark(),
        ));
        code.append(format!(r"let t0 = unflatten a_pre a_suf t0 in"));
        code.append(format!(r"let t1 = map (\t -> (reduce (+) 0 t) / ({}.i64 a_suf)) t0 in",
            arg[0].dtype.format_futhark(),
        ));
        code.append(format!(r"let {{%1}} = unflatten_3d {{%0.s[0]}} {{%0.s[1]}} {{%0.s[2]}} t1 in"));
        code.into()
      }
      _ => {
        unimplemented!();
      }
    }
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

  fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(arg[0].clone())
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(arg[0].clone())
  }

  fn gen_futhark(&self, arg: &[Dim]) -> Result<FutharkThunkCode, FutharkGenErr> {
    let out = FutharkThunkSpec::out_dim(self, arg).map_err(|e| e.into_gen())?;
    match out.ndim {
      0 => {
        unimplemented!();
      }
      1 => {
        unimplemented!();
      }
      2 => {
        let mut code = FutharkThunkCode::default();
        code.cfg.emit_arg_shapes = true;
        code.body.push(format!(r"let a_pre = {{%0.s[0]}} in"));
        code.body.push(format!(r"let a_suf = {{%0.s[1]}} in"));
        code.body.push(format!(r"let a = a_pre * a_suf in"));
        code.body.push(format!(r"let t0 = flatten {{%0}} :> [a]{} in", arg[0].dtype.format_futhark()));
        code.body.push(format!(r"let t0 = unflatten a_pre a_suf t0 in"));
        code.body.push(format!(r"let t0_max = map (\t -> reduce ({}.max) (-{}.inf) t) t0 in", arg[0].dtype.format_futhark(), arg[0].dtype.format_futhark()));
        code.body.push(format!(r"let t1 = map2 (\t t_max -> map (\u -> ({}.exp) (u - t_max)) t) t0 t0_max in", arg[0].dtype.format_futhark()));
        code.body.push(format!(r"let t1_sum = map (\t -> reduce (+) 0 t) t1 in"));
        code.body.push(format!(r"let t2 = map2 (\t t_sum -> map (/ t_sum) t) t1 t1_sum in"));
        code.body.push(format!(r"let t2 = flatten t2 :> [a]{} in", arg[0].dtype.format_futhark()));
        code.body.push(format!(r"let {{%1}} = unflatten {{%0.s[0]}} {{%0.s[1]}} t2 in"));
        code.into()
      }
      3 => {
        let mut code = FutharkThunkCode::default();
        code.cfg.emit_arg_shapes = true;
        code.body.push(format!(r"let a_pre = {{%0.s[0]}} * {{%0.s[1]}} in"));
        code.body.push(format!(r"let a_suf = {{%0.s[2]}} in"));
        code.body.push(format!(r"let a = a_pre * a_suf in"));
        code.body.push(format!(r"let t0 = flatten_3d {{%0}} :> [a]{} in", arg[0].dtype.format_futhark()));
        code.body.push(format!(r"let t0 = unflatten a_pre a_suf t0 in"));
        code.body.push(format!(r"let t0_max = map (\t -> reduce ({}.max) (-{}.inf) t) t0 in", arg[0].dtype.format_futhark(), arg[0].dtype.format_futhark()));
        code.body.push(format!(r"let t1 = map2 (\t t_max -> map (\u -> ({}.exp) (u - t_max)) t) t0 t0_max in", arg[0].dtype.format_futhark()));
        code.body.push(format!(r"let t1_sum = map (\t -> reduce (+) 0 t) t1 in"));
        code.body.push(format!(r"let t2 = map2 (\t t_sum -> map (/ t_sum) t) t1 t1_sum in"));
        code.body.push(format!(r"let t2 = flatten t2 :> [a]{} in", arg[0].dtype.format_futhark()));
        code.body.push(format!(r"let {{%1}} = unflatten_3d {{%0.s[0]}} {{%0.s[1]}} {{%0.s[2]}} t2 in"));
        code.into()
      }
      4 => {
        let mut code = FutharkThunkCode::default();
        code.cfg.emit_arg_shapes = true;
        code.body.push(format!(r"let a_pre = {{%0.s[0]}} * {{%0.s[1]}} * {{%0.s[2]}} in"));
        code.body.push(format!(r"let a_suf = {{%0.s[3]}} in"));
        code.body.push(format!(r"let a = a_pre * a_suf in"));
        code.body.push(format!(r"let t0 = flatten_4d {{%0}} :> [a]{} in", arg[0].dtype.format_futhark()));
        code.body.push(format!(r"let t0 = unflatten a_pre a_suf t0 in"));
        code.body.push(format!(r"let t0_max = map (\t -> reduce ({}.max) (-{}.inf) t) t0 in", arg[0].dtype.format_futhark(), arg[0].dtype.format_futhark()));
        code.body.push(format!(r"let t1 = map2 (\t t_max -> map (\u -> ({}.exp) (u - t_max)) t) t0 t0_max in", arg[0].dtype.format_futhark()));
        code.body.push(format!(r"let t1_sum = map (\t -> reduce (+) 0 t) t1 in"));
        code.body.push(format!(r"let t2 = map2 (\t t_sum -> map (/ t_sum) t) t1 t1_sum in"));
        code.body.push(format!(r"let t2 = flatten t2 :> [a]{} in", arg[0].dtype.format_futhark()));
        code.body.push(format!(r"let {{%1}} = unflatten_4d {{%0.s[0]}} {{%0.s[1]}} {{%0.s[2]}} {{%0.s[3]}} t2 in"));
        code.into()
      }
      _ => {
        unimplemented!();
      }
    }
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

  fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 2;
    abi.arityout = 1;
    abi
  }

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

  fn gen_futhark(&self, arg: &[Dim]) -> Result<FutharkThunkCode, FutharkGenErr> {
    let out = FutharkThunkSpec::out_dim(self, arg).map_err(|e| e.into_gen())?;
    match out.ndim {
      0 => {
        unimplemented!();
      }
      1 => {
        unimplemented!();
      }
      2 => {
        let mut code = FutharkThunkCode::default();
        code.cfg.emit_arg_shapes = true;
        code.append(format!(r"let a_pre = {{%0.s[0]}} * {{%0.s[1]}} in"));
        code.append(format!(r"let a_suf = {{%0.s[2]}} in"));
        code.append(format!(r"let a = a_pre * a_suf in"));
        code.append(format!(r"let t0 = flatten_3d {{%0}} :> [a]{} in", arg[0].dtype.format_futhark()));
        code.append(format!(r"let t0 = unflatten a_pre a_suf t0 in"));
        code.append(format!(r"let t0_max = map (\t -> reduce ({}.max) (-{}.inf) t) t0 in", arg[0].dtype.format_futhark(), arg[0].dtype.format_futhark()));
        code.append(format!(r"let t1 = map2 (\t t_max -> map (\u -> ({}.exp) (u - t_max)) t) t0 t0_max in", arg[0].dtype.format_futhark()));
        code.append(format!(r"let t1_sum = map (\t -> reduce (+) 0 t) t1 in"));
        code.append(format!(r"let t_val = map2 (\t t_sum -> map (/ t_sum) t) t1 t1_sum in"));
        code.append(format!(r"let t_iota = iota a_pre in"));
        code.append(format!(r"let t_key = flatten {{%1}} :> [a_pre]{} in", arg[1].dtype.format_futhark()));
        code.append(format!(r"let t_key = map2 (\idx k -> let k = (i64.{} k) in (assert (k >= 0 && k < a_suf) k) + a_suf * idx) t_iota t_key in",
            arg[1].dtype.format_futhark(),
        ));
        code.append(format!(r"let t2 = map2 (\k v -> -({}.log v[k])) t_key t_val in",
            arg[0].dtype.format_futhark(),
        ));
        code.append(format!(r"let {{%2}} = unflatten {{%0.s[0]}} {{%0.s[1]}} t2 in"));
        code.into()
      }
      _ => {
        unimplemented!();
      }
    }
  }

  fn pop_adj(&self, arg: &[(CellPtr, Clock)], _out: CellPtr, _out_clk: Clock, out_adj: CellPtr, arg_adj: &mut [CellPtr]) -> Option<Result<(), ThunkAdjErr>> {
    let x = arg[0].0;
    let x_ty = x.type_();
    let x_nd = x_ty.ndim() as usize;
    let inner_len = x_ty.shape[x_nd - 1];
    let dtype = x_ty.dtype;
    let y = x.inner_softmax();
    let rank = arg[1].0;
    let target = rank.inner_one_hot(inner_len, dtype);
    arg_adj[0] += out_adj * (y - target);
    Some(Ok(()))
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

  fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }

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

  fn gen_futhark(&self, arg: &[Dim]) -> Result<FutharkThunkCode, FutharkGenErr> {
    let out = FutharkThunkSpec::out_dim(self, arg).map_err(|e| e.into_gen())?;
    let pad_outer = self.new_block[0] - self.org_block[0];
    let pad_inner = self.new_block[1] - self.org_block[1];
    match out.ndim {
      4 => {
        if pad_outer == 0 && pad_inner == 0 {
          panic!("bug");
        } else if pad_outer == 0 && pad_inner > 0 {
          let mut code = FutharkThunkCode::default();
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
          code.into()
        } else if pad_outer == 0 && pad_inner < 0 {
          let mut code = FutharkThunkCode::default();
          code.cfg.emit_arg_shapes = true;
          code.append(format!(r"let a_inner = {{%0.s[3]}} + {} in",
              pad_inner,
          ));
          code.append(r"let t0 = {%0} in");
          code.append(r"let t1 = map (\t_blk_row -> map (\t_row -> map (\t_inner -> t_inner[0:a_inner]) t_row) t_blk_row) t0 in");
          code.append(r"let {%1} = t1 in");
          code.into()
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
  }

  fn pop_adj(&self, arg: &[(CellPtr, Clock)], out: CellPtr, out_clk: Clock, out_adj: CellPtr, arg_adj: &mut [CellPtr]) -> Option<Result<(), ThunkAdjErr>> {
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
    Some(Ok(()))
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

  fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    // FIXME
    Ok(arg[0])
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    // FIXME
    Ok(arg[0].clone())
  }

  fn gen_futhark(&self, arg: &[Dim]) -> Result<FutharkThunkCode, FutharkGenErr> {
    let out = FutharkThunkSpec::out_dim(self, arg).map_err(|e| e.into_gen())?;
    match out.ndim {
      4 => {
        let mut code = FutharkThunkCode::default();
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
        code.into()
      }
      _ => {
        unimplemented!();
      }
    }
  }

  fn pop_adj(&self, arg: &[(CellPtr, Clock)], _out: CellPtr, _out_clk: Clock, out_adj: CellPtr, arg_adj: &mut [CellPtr]) -> Option<Result<(), ThunkAdjErr>> {
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
    Some(Ok(()))
  }
}

/*#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct DotThunkSpec;*/

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
    Some("block_mm")
  }

  fn arity(&self) -> (u16, u16) {
    (2, 1)
  }

  fn cost_r0(&self) -> Option<ThunkCostR0> {
    Some(ThunkCostR0::Time)
  }

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    if arg[0].ndim() != 2 {
      return Err(ThunkDimErr::_Bot);
    }
    if arg[1].ndim() != 2 {
      return Err(ThunkDimErr::_Bot);
    }
    if self.l_dtype != arg[0].dtype {
      return Err(ThunkDimErr::_Bot);
    }
    if self.r_dtype != arg[1].dtype {
      return Err(ThunkDimErr::_Bot);
    }
    Ok(Dim{ndim: 2, dtype: self.o_dtype})
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
        println!("WARNING: BlockMatrixMulThunkSpec::gen_impl_: no impl for pmach={:?}", pmach);
        None
      }
    }
  }

  fn pop_adj(&self, arg: &[(CellPtr, Clock)], _out: CellPtr, _out_clk: Clock, out_adj: CellPtr, arg_adj: &mut [CellPtr]) -> Result<(), ThunkAdjErr> {
    let [l_blk_outer, l_blk_inner] = if self.l_blk_t { [self.l_block[1], self.l_block[0]] } else { self.l_block };
    let [r_blk_inner, r_blk_outer] = if self.r_blk_t { [self.r_block[1], self.r_block[0]] } else { self.r_block };
    assert_eq!(l_blk_inner, r_blk_inner);
    match (self.l_blk_t, self.r_blk_t) {
      (false, false) => {
        // (O-R, O-C) = (a-R, a-C) x (b-R, b-C)     = (a-R, b-C)
        // (a-R, a-C) = (O-R, O-C) x (O-C, a-C)     = (O-R, O-C)   x (b-R, b-C)^T
        // (b-R, b-C) = (b-R, O-R) x (O-R, O-C)     = (a-R, a-C)^T x (O-R, O-C)
        arg_adj[0] += out_adj.block_mm([l_blk_outer, r_blk_outer], false, arg[1].0, self.r_block, true);
        arg_adj[1] += arg[0].0.block_mm(self.l_block, true, out_adj, [l_blk_outer, r_blk_outer], false);
      }
      (false, true) => {
        // (O-R, O-C) = (a-R, a-C)   x (b-R, b-C)^T = (a-R, b-R)
        // (a-R, a-C) = (O-R, O-C)   x (O-R, a-C)   = (O-R, O-C)   x (b-R, b-C)
        // (b-R, b-C) = (O-R, O-C)^T x (O-R, b-C)   = (O-R, O-C)^T x (a-R, a-C)
        arg_adj[0] += out_adj.block_mm([l_blk_outer, r_blk_outer], false, arg[1].0, self.r_block, false);
        arg_adj[1] += out_adj.block_mm([l_blk_outer, r_blk_outer], true, arg[0].0, self.r_block, false);
      }
      (true, false) => {
        // (O-R, O-C) = (a-R, a-C)^T x (b-R, b-C)   = (a-C, b-C)
        // (a-R, a-C) = (a-R, O-C)   x (O-R, O-C)^T = (b-R, b-C)   x (O-R, O-C)^T
        // (b-R, b-C) = (b-R, O-R)   x (O-R, O-C)   = (a-R, a-C)   x (O-R, O-C)
        arg_adj[0] += arg[1].0.block_mm(self.r_block, false, out_adj, [l_blk_outer, r_blk_outer], true);
        arg_adj[1] += arg[0].0.block_mm(self.l_block, false, out_adj, [l_blk_outer, r_blk_outer], false);
      }
      (true, true) => {
        // (O-R, O-C) = (a-R, a-C)^T x (b-R, b-C)^T = (a-C, b-R)
        // (a-R, a-C) = (a-R, O-C)   x (O-R, O-C)^T = (b-R, b-C)^T x (O-R, O-C)^T
        // (b-R, b-C) = (O-R, O-C)^T x (O-C, b-C)   = (O-R, O-C)^T x (a-R, a-C)^T
        arg_adj[0] += arg[1].0.block_mm(self.r_block, true, out_adj, [l_blk_outer, r_blk_outer], true);
        arg_adj[1] += out_adj.block_mm([l_blk_outer, r_blk_outer], true, arg[0].0, self.r_block, true);
      }
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
    if arg[0].ndim() != 2 {
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
    println!("DEBUG: BlockMatrixMulThunkSpec::_calculate_out_ty: l nrow={} r nrow={} l ncol={} r ncol={}",
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
    println!("DEBUG: BlockMatrixMulThunkSpec::_calculate_out_ty: ({:?} / {:?}{}) x ({:?} / {:?}{}) = [{}, {}]",
        &arg[0].shape, self.l_block, if self.l_blk_t { " T" } else { "" },
        &arg[1].shape, self.r_block, if self.r_blk_t { " T" } else { "" },
        m, n);
    let o_ty = CellType{shape: vec![m, n], dtype: self.o_dtype};
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
impl ThunkImpl for BlockMatrixMulF16F32GpuThunkImpl {
  fn apply(&self, ctr: &CtxCtr, env: &mut CtxEnv, spec_: &dyn ThunkSpec_, arg: &[(CellPtr, Clock)], th: ThunkPtr, out: CellPtr, oclk: Clock) -> ThunkRet {
    println!("DEBUG: BlockMatrixMulF16F32GpuThunkImpl::apply");
    let spec = spec_.as_any().downcast_ref::<BlockMatrixMulThunkSpec>().unwrap();
    /*self.alpha.set(1.0);*/
    match &spec.o_scale {
      &ScalarVal_::F32(ref val) => {
        self.alpha.set(*val.borrow());
      }
      _ => unimplemented!()
    }
    self.beta.set(0.0);
    let mut arg_ty_ = Vec::with_capacity(arg.len());
    for &(x, _) in arg.iter() {
      match env.lookup_ref(x) {
        None => panic!("bug"),
        Some(e) => {
          arg_ty_.push(e.ty.clone());
        }
      }
    }
    /*let out_ty_ = ThunkSpec::out_ty_(spec, &arg_ty_).unwrap();*/
    let tys = spec._calculate_out_ty(&arg_ty_).unwrap();
    println!("DEBUG: BlockMatrixMulF16F32GpuThunkImpl::apply: arg_ty_={:?}", &arg_ty_);
    println!("DEBUG: BlockMatrixMulF16F32GpuThunkImpl::apply: gemmtys={:?}", &tys);
    println!("DEBUG: BlockMatrixMulF16F32GpuThunkImpl::apply: nrow={}", tys.nrow);
    println!("DEBUG: BlockMatrixMulF16F32GpuThunkImpl::apply: ncol={}", tys.ncol);
    println!("DEBUG: BlockMatrixMulF16F32GpuThunkImpl::apply: nblk={}", tys.nrow * tys.ncol);
    let out_ty_ = tys.o_ty.clone();
    // FIXME FIXME: correct transposes, shapes, arg order for row major v col major.
    let colmajor_bt = spec.l_blk_t;
    let colmajor_at = spec.r_blk_t;
    let colmajor_n = tys.l_blk_outer;
    let colmajor_m = tys.r_blk_outer;
    let inner_len = tys.l_blk_inner;
    println!("DEBUG: BlockMatrixMulF16F32GpuThunkImpl::apply: m={}", colmajor_m);
    println!("DEBUG: BlockMatrixMulF16F32GpuThunkImpl::apply: n={}", colmajor_n);
    println!("DEBUG: BlockMatrixMulF16F32GpuThunkImpl::apply: k={}", inner_len);
    assert_eq!(inner_len, tys.r_blk_inner);
    assert!(colmajor_m >= 0);
    assert!(colmajor_m <= i32::max_value() as _);
    assert!(colmajor_n >= 0);
    assert!(colmajor_n <= i32::max_value() as _);
    assert!(inner_len >= 0);
    assert!(inner_len <= i32::max_value() as _);
    // FIXME: does ldx need multiplying by elem size?
    let ldb = arg_ty_[0].shape[1];
    let lda = arg_ty_[1].shape[1];
    let ldc = out_ty_.shape[1];
    println!("DEBUG: BlockMatrixMulF16F32GpuThunkImpl::apply: lda={}", lda);
    println!("DEBUG: BlockMatrixMulF16F32GpuThunkImpl::apply: ldb={}", ldb);
    println!("DEBUG: BlockMatrixMulF16F32GpuThunkImpl::apply: ldc={}", ldc);
    assert!(ldb >= 0);
    assert!(ldb <= i32::max_value() as _);
    assert!(lda >= 0);
    assert!(lda <= i32::max_value() as _);
    assert!(ldc >= 0);
    assert!(ldc <= i32::max_value() as _);
    let loc = TL_PCTX.with(|pctx| {
      let gpu = pctx.nvgpu.as_ref().unwrap();
      gpu.device_locus()
    });
    println!("DEBUG: BlockMatrixMulF16F32GpuThunkImpl::apply: read arg[0]...");
    match env.pread_ref(arg[0].0, arg[0].1, /*CellEMode::Read,*/) {
      None => panic!("bug"),
      Some(e) => {
        match e.cel_ {
          &mut Cell_::Phy(ref _state, ref _clo, ref mut pcel) => {
            let pcel_addr = pcel.get(arg[0].1, &arg_ty_[0], loc, PMach::NvGpu);
            let base = TL_PCTX.with(|pctx| {
              let (dptr, _) = pctx.nvgpu.as_ref().unwrap().lookup_dev(pcel_addr).unwrap();
              dptr
            });
            let inc = spec.l_dtype.size_bytes() as u64;
            let blk_row_len = spec.l_block[0] as u64;
            let blk_col_len = spec.l_block[1] as u64;
            let stride = arg_ty_[0].shape[1] as u64;
            // FIXME FIXME
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
    println!("DEBUG: BlockMatrixMulF16F32GpuThunkImpl::apply: read arg[1]...");
    match env.pread_ref(arg[1].0, arg[1].1, /*CellEMode::Read,*/) {
      None => panic!("bug"),
      Some(e) => {
        match e.cel_ {
          &mut Cell_::Phy(ref _state, ref _clo, ref mut pcel) => {
            let pcel_addr = pcel.get(arg[1].1, &arg_ty_[1], loc, PMach::NvGpu);
            let base = TL_PCTX.with(|pctx| {
              let (dptr, _) = pctx.nvgpu.as_ref().unwrap().lookup_dev(pcel_addr).unwrap();
              dptr
            });
            let inc = spec.r_dtype.size_bytes() as u64;
            let blk_row_len = spec.r_block[0] as u64;
            let blk_col_len = spec.r_block[1] as u64;
            let stride = arg_ty_[1].shape[1] as u64;
            // FIXME FIXME
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
    println!("DEBUG: BlockMatrixMulF16F32GpuThunkImpl::apply: write out...");
    match env.pwrite_ref(out, oclk, /*CellEMode::Mutex,*/) {
      None => panic!("bug"),
      Some(e) => {
        match e.cel_ {
          &mut Cell_::Phy(ref _state, ref _clo, ref mut pcel) => {
            let pcel_addr = pcel.fresh(oclk, &out_ty_, loc, PMach::NvGpu);
            let base = TL_PCTX.with(|pctx| {
              let (dptr, _) = pctx.nvgpu.as_ref().unwrap().lookup_dev(pcel_addr).unwrap();
              dptr
            });
            let inc = spec.o_dtype.size_bytes() as u64;
            let blk_row_len = tys.l_blk_outer as u64;
            let blk_col_len = tys.r_blk_outer as u64;
            let stride = out_ty_.shape[1] as u64;
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
    //println!("DEBUG: BlockMatrixMulF16F32GpuThunkImpl::apply: tmp_a=[0x{:016x}]", self.tmp_a.borrow()[0]);
    //println!("DEBUG: BlockMatrixMulF16F32GpuThunkImpl::apply: tmp_b=[0x{:016x}]", self.tmp_b.borrow()[0]);
    //println!("DEBUG: BlockMatrixMulF16F32GpuThunkImpl::apply: tmp_c=[0x{:016x}]", self.tmp_c.borrow()[0]);
    let b_gputy = match spec.l_dtype {
      Dtype::Fp32 => CUDA_R_32F,
      Dtype::Fp16 => CUDA_R_16F,
      Dtype::Bfloat16 => CUDA_R_16BF,
      _ => unimplemented!()
    };
    let a_gputy = match spec.r_dtype {
      Dtype::Fp32 => CUDA_R_32F,
      Dtype::Fp16 => CUDA_R_16F,
      Dtype::Bfloat16 => CUDA_R_16BF,
      _ => unimplemented!()
    };
    let c_gputy = match spec.o_dtype {
      Dtype::Fp32 => CUDA_R_32F,
      Dtype::Fp16 => CUDA_R_16F,
      Dtype::Bfloat16 => CUDA_R_16BF,
      _ => unimplemented!()
    };
    TL_PCTX.with(|pctx| {
      let gpu = &pctx.nvgpu.as_ref().unwrap();
      println!("DEBUG: BlockMatrixMulF16F32GpuThunkImpl::apply: gemm...");
      let ret = gpu.compute.sync();
      match ret {
        Err(e) => {
          println!("DEBUG: BlockMatrixMulF16F32GpuThunkImpl::apply: gemm pre sync error: {:?}", e);
          return ThunkRet::Failure;
        }
        Ok(_) => {}
      }
      //let tmp_a = self.tmp_a.borrow();
      //let tmp_b = self.tmp_b.borrow();
      //let tmp_c = self.tmp_c.borrow();
      let alpha_ptr = self.alpha.as_ptr() as usize;
      let beta_ptr = self.beta.as_ptr() as usize;
      //let tmp_a_ptr = tmp_a.as_ptr() as usize;
      //let tmp_b_ptr = tmp_b.as_ptr() as usize;
      //let tmp_c_ptr = tmp_c.as_ptr() as usize;
      println!("DEBUG: BlockMatrixMulF16F32GpuThunkImpl::apply: alpha ptr=0x{:016x}", alpha_ptr);
      println!("DEBUG: BlockMatrixMulF16F32GpuThunkImpl::apply: beta ptr =0x{:016x}", beta_ptr);
      //println!("DEBUG: BlockMatrixMulF16F32GpuThunkImpl::apply: tmp a ptr=0x{:016x}", tmp_a_ptr);
      //println!("DEBUG: BlockMatrixMulF16F32GpuThunkImpl::apply: tmp b ptr=0x{:016x}", tmp_b_ptr);
      //println!("DEBUG: BlockMatrixMulF16F32GpuThunkImpl::apply: tmp c ptr=0x{:016x}", tmp_c_ptr);
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
        let staging_buf = unsafe { from_raw_parts_mut(gpu.page_map.back_buf.ptr as *mut u64, staging_len) };
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
        gpu.hard_copy_raw_mem_to_vmem(gpu.mem_pool.back_base, gpu.page_map.back_buf.ptr, staging_sz);
        let a = gpu.mem_pool.back_base;
        let b = gpu.mem_pool.back_base + (ntxn * 128) as u64;
        let c = gpu.mem_pool.back_base + (ntxn * 128 * 2) as u64;
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
          println!("DEBUG: BlockMatrixMulF16F32GpuThunkImpl::apply: gemm error: {:?}", e);
          return ThunkRet::Failure;
        }
        Ok(_) => {}
      }
      let ret = gpu.compute.sync();
      match ret {
        Err(e) => {
          println!("DEBUG: BlockMatrixMulF16F32GpuThunkImpl::apply: gemm sync error: {:?}", e);
          return ThunkRet::Failure;
        }
        Ok(_) => {}
      }
      //drop(tmp_c);
      //drop(tmp_b);
      //drop(tmp_a);
      println!("DEBUG: BlockMatrixMulF16F32GpuThunkImpl::apply: gemm OK");
      ThunkRet::Success
    })
  }

  fn accumulate(&self, ctr: &CtxCtr, env: &mut CtxEnv, spec_: &dyn ThunkSpec_, arg: &[(CellPtr, Clock)], th: ThunkPtr, out: CellPtr, oclk: Clock) -> ThunkRet {
    let spec = spec_.as_any().downcast_ref::<BlockMatrixMulThunkSpec>().unwrap();
    /*self.alpha.set(1.0);*/
    match &spec.o_scale {
      &ScalarVal_::F32(ref val) => {
        self.alpha.set(*val.borrow());
      }
      _ => unimplemented!()
    }
    self.beta.set(1.0);
    unimplemented!();
    /*
    TL_PCTX.with(|pctx| {
      // FIXME FIXME
    })
    */
  }
}
