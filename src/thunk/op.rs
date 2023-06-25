use super::*;
use crate::algo::fp::{TotalOrd};
use crate::cell::{DtypeExt, Dim, ScalarVal_};
use cacti_gpu_cu_ffi::{cublas_gemm_batched};
use cacti_gpu_cu_ffi::types::{CUDA_R_32F, CUDA_R_16F, CUDA_R_16BF};

use futhark_ffi::{Abi};
use futhark_syntax::{Exp};

use std::borrow::{Cow};
use std::cell::{Cell};
use std::ffi::{c_void};
//use std::io::{Write};
use std::rc::{Weak};

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
pub struct SetScalarFutThunkSpec<T> { pub val: T }

impl<T: DtypeExt + Copy + Eq + Any> FutharkThunkSpec for SetScalarFutThunkSpec<T> {
  fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 0;
    abi.arityout = 1;
    abi
  }

  fn out_dim(&self, _arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(Dim{ndim: 0, dtype: T::dtype()})
  }

  fn out_ty_(&self, _arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(CellType{shape: Vec::new(), dtype: T::dtype()})
  }

  fn scalar_val(&self) -> Option<&dyn DtypeExt> {
    Some(&self.val)
  }

  fn gen_futhark(&self, _arg: &[Dim]) -> Result<FutharkThunkCode, FutharkGenErr> {
    let fmt = FutharkNumFormatter::default();
    let mut code = FutharkThunkCode::default();
    // FIXME FIXME: futhark treats actual scalars as simply pointers to cpu mem.
    /*body:     vec![format!("let {{%0}} = [{}] in", fmt.format(&self.val))],*/
    code.body.push(format!("let {{%0}} = {} in", fmt.format(&self.val)));
    code.into()
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SetScalar1dFutThunkSpec<T> { pub val: T }

impl<T: DtypeExt + Copy + Eq + Any> FutharkThunkSpec for SetScalar1dFutThunkSpec<T> {
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
}

// FIXME: only need new dtype.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct CastFutThunkSpec { pub org_dtype: Dtype, pub new_dtype: Dtype }

impl FutharkThunkSpec for CastFutThunkSpec {
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
    FutharkThunkCode::map_nd(arg[0], format!(r"\t -> {}.{} t",
                                             self.new_dtype.format_futhark(),
                                             self.org_dtype.format_futhark()))
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct CastBf16F16FutThunkSpec;

impl FutharkThunkSpec for CastBf16F16FutThunkSpec {
  fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    if arg[0].dtype != Dtype::BFloat16 {
      return Err(ThunkDimErr::_Bot);
    }
    Ok(Dim{ndim: arg[0].ndim, dtype: Dtype::Float16})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    if arg[0].dtype != Dtype::BFloat16 {
      return Err(ThunkTypeErr::_Bot);
    }
    Ok(CellType{shape: arg[0].shape.clone(), dtype: Dtype::Float16})
  }

  fn gen_futhark(&self, arg: &[Dim]) -> Result<FutharkThunkCode, FutharkGenErr> {
    FutharkThunkCode::map_nd(arg[0], r"\t -> f16.f32 (f32.from_bits ((u32.u16 t) << 16))")
    /*match arg[0].ndim() {
      0 => {
        FutharkThunkCode{
          body:     vec![
                        format!("let {{%1}} = f16.f32 (f32.from_bits ((u32.u16 {{%0}}) << 16)) in"),
                    ],
        }.into()
      }
      1 => {
        FutharkThunkCode{
          body:     vec![
                        format!("let {{%1}} = map (\t -> f16.f32 (f32.from_bits ((u32.u16 t) << 16))) {{%0}} in"),
                    ],
        }.into()
      }
      2 => {
        FutharkThunkCode{
          body:     vec![
                        format!("let t0 = flatten {{%0}} in"),
                        format!("let t1 = map (\t -> f16.f32 (f32.from_bits ((u32.u16 t) << 16))) t0 in"),
                        format!("let {{%1}} = unflatten {{%0.s[0]}} {{%0.s[1]}} t1 in"),
                    ],
        }.into()
      }
      3 => {
        FutharkThunkCode{
          body:     vec![
                        format!("let t0 = flatten_3d {{%0}} in"),
                        format!("let t1 = map (\t -> f16.f32 (f32.from_bits ((u32.u16 t) << 16))) t0 in"),
                        format!("let {{%1}} = unflatten_3d {{%0.s[0]}} {{%0.s[1]}} {{%0.s[2]}} t1 in"),
                    ],
        }.into()
      }
      4 => {
        FutharkThunkCode{
          body:     vec![
                        format!("let t0 = flatten_4d {{%0}} in"),
                        format!("let t1 = map (\t -> f16.f32 (f32.from_bits ((u32.u16 t) << 16))) t0 in"),
                        format!("let {{%1}} = unflatten_4d {{%0.s[0]}} {{%0.s[1]}} {{%0.s[2]}} {{%0.s[3]}} t1 in"),
                    ],
        }.into()
      }
      _ => {
        unimplemented!();
      }
    }*/
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct CastBf16F32FutThunkSpec;

impl FutharkThunkSpec for CastBf16F32FutThunkSpec {
  fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    if arg[0].dtype != Dtype::BFloat16 {
      return Err(ThunkDimErr::_Bot);
    }
    Ok(Dim{ndim: arg[0].ndim, dtype: Dtype::Float32})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    if arg[0].dtype != Dtype::BFloat16 {
      return Err(ThunkTypeErr::_Bot);
    }
    Ok(CellType{shape: arg[0].shape.clone(), dtype: Dtype::Float32})
  }

  fn gen_futhark(&self, arg: &[Dim]) -> Result<FutharkThunkCode, FutharkGenErr> {
    FutharkThunkCode::map_nd(arg[0], r"\t -> f32.from_bits ((u32.u16 t) << 16)")
    /*match arg[0].ndim() {
      0 => {
        FutharkThunkCode{
          body:     vec![
                        format!("let {{%1}} = f32.from_bits ((u32.u16 {{%0}}) << 16) in"),
                    ],
        }.into()
      }
      1 => {
        FutharkThunkCode{
          body:     vec![
                        format!("let {{%1}} = map (\t -> f32.from_bits ((u32.u16 t) << 16)) {{%0}} in"),
                    ],
        }.into()
      }
      2 => {
        FutharkThunkCode{
          body:     vec![
                        format!("let t0 = flatten {{%0}} in"),
                        format!("let t1 = map (\t -> f32.from_bits ((u32.u16 t) << 16)) t0 in"),
                        format!("let {{%1}} = unflatten {{%0.s[0]}} {{%0.s[1]}} t1 in"),
                    ],
        }.into()
      }
      3 => {
        FutharkThunkCode{
          body:     vec![
                        format!("let t0 = flatten_3d {{%0}} in"),
                        format!("let t1 = map (\t -> f32.from_bits ((u32.u16 t) << 16)) t0 in"),
                        format!("let {{%1}} = unflatten_3d {{%0.s[0]}} {{%0.s[1]}} {{%0.s[2]}} t1 in"),
                    ],
        }.into()
      }
      4 => {
        FutharkThunkCode{
          body:     vec![
                        format!("let t0 = flatten_4d {{%0}} in"),
                        format!("let t1 = map (\t -> f32.from_bits ((u32.u16 t) << 16)) t0 in"),
                        format!("let {{%1}} = unflatten_4d {{%0.s[0]}} {{%0.s[1]}} {{%0.s[2]}} {{%0.s[3]}} t1 in"),
                    ],
        }.into()
      }
      _ => {
        unimplemented!();
      }
    }*/
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct CastF32Bf16FutThunkSpec;

impl FutharkThunkSpec for CastF32Bf16FutThunkSpec {
  fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 1;
    abi.arityout = 1;
    abi
  }

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    if arg[0].dtype != Dtype::Float32 {
      return Err(ThunkDimErr::_Bot);
    }
    Ok(Dim{ndim: arg[0].ndim, dtype: Dtype::BFloat16})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    if arg[0].dtype != Dtype::Float32 {
      return Err(ThunkTypeErr::_Bot);
    }
    Ok(CellType{shape: arg[0].shape.clone(), dtype: Dtype::BFloat16})
  }

  fn gen_futhark(&self, arg: &[Dim]) -> Result<FutharkThunkCode, FutharkGenErr> {
    FutharkThunkCode::map_nd(arg[0], r"\t -> u16.u32 ((f32.to_bits t) >> 16)")
    /*match arg[0].ndim() {
      0 => {
        FutharkThunkCode{
          body:     vec![
                        format!("let {{%1}} = u16.u32 ((f32.to_bits {{%0}}) >> 16) in"),
                    ],
        }.into()
      }
      1 => {
        FutharkThunkCode{
          body:     vec![
                        format!("let {{%1}} = map (\t -> u16.u32 ((f32.to_bits t) >> 16)) {{%0}} in"),
                    ],
        }.into()
      }
      2 => {
        FutharkThunkCode{
          body:     vec![
                        format!("let t0 = flatten {{%0}} in"),
                        format!("let t1 = map (\t -> u16.u32 ((f32.to_bits t) >> 16)) t0 in"),
                        format!("let {{%1}} = unflatten {{%0.s[0]}} {{%0.s[1]}} t1 in"),
                    ],
        }.into()
      }
      3 => {
        FutharkThunkCode{
          body:     vec![
                        format!("let t0 = flatten_3d {{%0}} in"),
                        format!("let t1 = map (\t -> u16.u32 ((f32.to_bits t) >> 16)) t0 in"),
                        format!("let {{%1}} = unflatten_3d {{%0.s[0]}} {{%0.s[1]}} {{%0.s[2]}} t1 in"),
                    ],
        }.into()
      }
      4 => {
        FutharkThunkCode{
          body:     vec![
                        format!("let t0 = flatten_4d {{%0}} in"),
                        format!("let t1 = map (\t -> u16.u32 ((f32.to_bits t) >> 16)) t0 in"),
                        format!("let {{%1}} = unflatten_4d {{%0.s[0]}} {{%0.s[1]}} {{%0.s[2]}} {{%0.s[3]}} t1 in"),
                    ],
        }.into()
      }
      _ => {
        unimplemented!();
      }
    }*/
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct InnerOneHotFutThunkSpec { pub inner_len: i64, /*pub org_dtype: Dtype,*/ pub new_dtype: Dtype }

impl FutharkThunkSpec for InnerOneHotFutThunkSpec {
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
    Ok(Dim{ndim: arg[0].ndim + 1, dtype: self.new_dtype})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    /*if arg[0].dtype != self.org_dtype {
      return Err(ThunkTypeErr::_Bot);
    }*/
    let mut shape = arg[0].shape.clone();
    shape.push(self.inner_len);
    Ok(CellType{shape, dtype: self.new_dtype})
  }

  fn gen_futhark(&self, arg: &[Dim]) -> Result<FutharkThunkCode, FutharkGenErr> {
    let out = FutharkThunkSpec::out_dim(self, arg).map_err(|e| e.into_gen())?;
    let fmt = FutharkNumFormatter::default();
    //match (out.ndim, out.dtype) {}
    match out.ndim {
      0 => {
        unimplemented!();
      }
      //(1, Dtype::Float32) => {}
      1 => {
        unimplemented!();
      }
      //(2, Dtype::Float32) => {}
      2 => {
        let mut code = FutharkThunkCode::default();
        code.cfg.emit_arg_shapes = true;
        code.body.push(format!("let t_oidx = {{%0}} in"));
        code.body.push(format!("let t_iota = indices t_oidx in"));
        code.body.push(format!("let t_key = map (\\(i,k) -> ({}.{} k) + {} * i) (zip t_iota t_oidx) in",
            Dtype::Int64.format_futhark(),
            arg[0].dtype.format_futhark(),
            self.inner_len,
        ));
        code.body.push(format!("let t_val = replicate {{%0.s[0]}} 1.0{} in",
            //fmt.format(&TotalOrd::from(1.0_f32)),
            out.dtype.format_futhark(),
        ));
        /*code.body.push(format!("let t0 = replicate ({{%0.s[0]}} * {}) 0.0{} in",
            self.inner_len,
            //fmt.format(&TotalOrd::from(0.0_f32)),
            out.dtype.format_futhark(),
        ));
        code.body.push(format!("let t1 = scatter t0 t_key t_val in"));*/
        code.body.push(format!("let t1 = spread ({{%0.s[0]}} * {}) 0.0{} t_key t_val in",
            self.inner_len,
            out.dtype.format_futhark(),
        ));
        code.body.push(format!("let {{%1}} = unflatten {{%0.s[0]}} {} t1 in",
            self.inner_len,
        ));
        code.into()
      }
      //(3, Dtype::Float32) => {}
      3 => {
        let mut code = FutharkThunkCode::default();
        code.cfg.emit_arg_shapes = true;
        code.body.push(format!("let a = {{%0.s[0]}} * {{%0.s[1]}} in"));
        code.body.push(format!("let t_oidx = flatten {{%0}} :> [a]{} in",
            arg[0].dtype.format_futhark(),
        ));
        code.body.push(format!("let t_iota = indices t_oidx in"));
        // FIXME FIXME: debugging.
        //code.body.push(format!("let t_key = map (\\(i,k) -> ({}.{} 0) + {} * i) (zip t_iota t_oidx) in",
        code.body.push(format!("let t_key = map (\\(i,k) -> ({}.{} k) + {} * i) (zip t_iota t_oidx) in",
            Dtype::Int64.format_futhark(),
            arg[0].dtype.format_futhark(),
            self.inner_len,
        ));
        /*code.body.push(format!("let t_val = replicate ({{%0.s[0]}} * {{%0.s[1]}}) 1.0{} in",
            //fmt.format(&TotalOrd::from(1.0_f32)),
            out.dtype.format_futhark(),
        ));
        code.body.push(format!("let t0 = replicate ({{%0.s[0]}} * {{%0.s[1]}} * {}) 0.0{} in",
            self.inner_len,
            //fmt.format(&TotalOrd::from(0.0_f32)),
            out.dtype.format_futhark(),
        ));
        code.body.push(format!("let t1 = scatter t0 t_key t_val in"));*/
        code.body.push(format!("let t_val = replicate a 1.0{} in",
            out.dtype.format_futhark(),
        ));
        code.body.push(format!("let t1 = spread (a * {}) 0.0{} t_key t_val in",
            self.inner_len,
            out.dtype.format_futhark(),
        ));
        code.body.push(format!("let {{%1}} = unflatten_3d {{%0.s[0]}} {{%0.s[1]}} {} t1 in",
            self.inner_len,
        ));
        code.into()
      }
      //(4, Dtype::Float32) => {}
      4 => {
        unimplemented!();
      }
      _ => unimplemented!()
    }
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct AddScalarF32FutThunkSpec { pub val: TotalOrd<f32> }

impl FutharkThunkSpec for AddScalarF32FutThunkSpec {
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
    FutharkThunkCode::map_nd(arg[0], format!(r"\t -> t + {}", fmt.format(&self.val)))
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct AddFutThunkSpec;

impl FutharkThunkSpec for AddFutThunkSpec {
  fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 2;
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
    FutharkThunkCode::map2_nd(arg[0], arg[1], r"\u, v -> u + v")
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SubScalarF32FutThunkSpec { pub val: TotalOrd<f32> }

impl FutharkThunkSpec for SubScalarF32FutThunkSpec {
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
    FutharkThunkCode::map_nd(arg[0], format!(r"\t -> t - {}", fmt.format(&self.val)))
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct SubFutThunkSpec;

impl FutharkThunkSpec for SubFutThunkSpec {
  fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 2;
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
    FutharkThunkCode::map2_nd(arg[0], arg[1], r"\u, v -> u - v")
    /*FutharkThunkCode{
      body:     vec![format!("let {{%2}} = {{%0}} - {{%1}} in")],
    }.into()*/
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct MulScalarF32FutThunkSpec { pub val: TotalOrd<f32> }

impl FutharkThunkSpec for MulScalarF32FutThunkSpec {
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
    FutharkThunkCode::map_nd(arg[0], format!(r"\t -> t * {}", fmt.format(&self.val)))
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct MulFutThunkSpec;

impl FutharkThunkSpec for MulFutThunkSpec {
  fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 2;
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
    FutharkThunkCode::map2_nd(arg[0], arg[1], r"\u, v -> u * v")
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct DivScalarF32FutThunkSpec { pub val: TotalOrd<f32> }

impl FutharkThunkSpec for DivScalarF32FutThunkSpec {
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
    FutharkThunkCode::map_nd(arg[0], format!(r"\t -> t / {}", fmt.format(&self.val)))
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct DivFutThunkSpec;

impl FutharkThunkSpec for DivFutThunkSpec {
  fn abi(&self) -> Abi {
    let mut abi = Abi::default();
    abi.arityin = 2;
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
    FutharkThunkCode::map2_nd(arg[0], arg[1], r"\u, v -> u / v")
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct SqrtFutThunkSpec;

impl FutharkThunkSpec for SqrtFutThunkSpec {
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
    FutharkThunkCode::map_nd(arg[0], r"\t -> sqrt t")
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct RsqrtFutThunkSpec;

impl FutharkThunkSpec for RsqrtFutThunkSpec {
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
    FutharkThunkCode::map_nd(arg[0], r"\t -> recip (sqrt t)")
    //FutharkThunkCode::map_nd(arg[0], r"\t -> rsqrt t")
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct CosFutThunkSpec;

impl FutharkThunkSpec for CosFutThunkSpec {
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
    FutharkThunkCode::map_nd(arg[0], r"\t -> cos t")
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct SinFutThunkSpec;

impl FutharkThunkSpec for SinFutThunkSpec {
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
    FutharkThunkCode::map_nd(arg[0], r"\t -> sin t")
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct ExpFutThunkSpec;

impl FutharkThunkSpec for ExpFutThunkSpec {
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
    FutharkThunkCode::map_nd(arg[0], r"\t -> exp t")
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct TanhFutThunkSpec;

impl FutharkThunkSpec for TanhFutThunkSpec {
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
    FutharkThunkCode::map_nd(arg[0], r"\t -> ((exp t) - (exp (-t))) / ((exp t) + (exp (-t)))")
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct PowiF32FutThunkSpec { pub exp: i64 }

impl FutharkThunkSpec for PowiF32FutThunkSpec {
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
    FutharkThunkCode::map_nd(arg[0], format!(r"\t -> t ** (f32.i64 {})", self.exp))
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct InnerMax3dThunkSpec;

/*impl FutharkThunk_ for InnerMax3dThunkSpec {
  fn _arg_count(&self) -> u8 {
    1
  }

  fn _body(&self) -> Vec<u8> {
    b"map (\t1 -> map (\t2 -> [reduce max -inf t2]) t1) {%0}\n".to_owned()
  }
}*/

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct InnerMean3dThunkSpec;

/*impl FutharkThunk_ for InnerSum3dThunkSpec {
  fn _arg_count(&self) -> u8 {
    1
  }

  fn _body(&self) -> Vec<u8> {
    b"map (\t1 -> map (\t2 -> [(reduce (+) 0 t2) / ({%t.0}.i64 (length t2))]) t1) {%0}\n".to_owned()
  }
}*/

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct InnerSum3dThunkSpec;

/*impl FutharkThunk_ for InnerSum3dThunkSpec {
  fn _arg_count(&self) -> u8 {
    1
  }

  fn _body(&self) -> Vec<u8> {
    b"map (\t1 -> map (\t2 -> [reduce (+) 0 t2]) t1) {%0}\n".to_owned()
  }
}*/

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct Sum1dFutThunkSpec;

impl FutharkThunkSpec for Sum1dFutThunkSpec {
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
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct InnerSoftmaxFutThunkSpec;

impl FutharkThunkSpec for InnerSoftmaxFutThunkSpec {
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
    unimplemented!();
  }
}

/*#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct DotThunkSpec;*/

/*impl CustomThunk_ for AddScalarF32ThunkSpec {
}*/

/*#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct BlockLMatrixMulThunkSpec {
  pub l_block:  [i64; 2],
  pub lt:       bool,
  pub rt:       bool,
  pub l_dtype:  Dtype,
  pub r_dtype:  Dtype,
  pub o_dtype:  Dtype,
}*/

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
  fn arity(&self) -> (u16, u16) {
    (2, 1)
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
    // FIXME FIXME
    /*let (m, n) = match (self.l_blk_t, self.r_blk_t) {
      (false, false) => {
        if arg[0].shape[1] != arg[1].shape[0] {
          return Err(ThunkTypeErr::_Bot);
        }
        (arg[0].shape[0], arg[1].shape[1])
      }
      (true, false) => {
        if arg[0].shape[0] != arg[1].shape[0] {
          return Err(ThunkTypeErr::_Bot);
        }
        (arg[0].shape[1], arg[1].shape[1])
      }
      (false, true) => {
        if arg[0].shape[1] != arg[1].shape[1] {
          return Err(ThunkTypeErr::_Bot);
        }
        (arg[0].shape[0], arg[1].shape[0])
      }
      (true, true) => {
        if arg[0].shape[0] != arg[1].shape[1] {
          return Err(ThunkTypeErr::_Bot);
        }
        (arg[0].shape[1], arg[1].shape[0])
      }
    };*/
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
    Ok(CellType{shape: vec![m, n], dtype: self.o_dtype})
  }

  fn gen_impl_(&self, spec_dim: Vec<Dim>, pmach: PMach) -> Option<Rc<dyn ThunkImpl_>> {
    // FIXME FIXME
    unimplemented!();
  }
}

pub struct BlockMatrixMulF16F32GpuThunkImpl {
  // TODO
  alpha: Cell<f32>,
  beta: Cell<f32>,
  tmp_a: RefCell<Vec<u64>>,
  tmp_b: RefCell<Vec<u64>>,
  tmp_c: RefCell<Vec<u64>>,
}

impl ThunkImpl for BlockMatrixMulF16F32GpuThunkImpl {
  fn apply(&self, ctr: &CtxCtr, env: &mut CtxEnv, spec_: &dyn ThunkSpec_, arg: &[(CellPtr, Clock)], th: ThunkPtr, out: CellPtr, oclk: Clock) -> ThunkRet {
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
    let out_ty_ = ThunkSpec::out_ty_(spec, &arg_ty_).unwrap();
    // FIXME FIXME: correct transposes, shapes, arg order for row major v col major.
    let colmajor_at = spec.r_blk_t;
    let colmajor_bt = spec.l_blk_t;
    /*let o_nrow = out_ty_.shape[0];
    assert!(o_nrow <= i32::max_value() as _);
    let o_ncol = out_ty_.shape[1];
    assert!(o_ncol <= i32::max_value() as _);*/
    // FIXME FIXME: should be the block inner len.
    /*let inner_len = if spec.l_blk_t { arg_ty_[0].shape[0] } else { arg_ty_[0].shape[1] };
    assert_eq!(inner_len, if spec.r_blk_t { arg_ty_[1].shape[1] } else { arg_ty_[1].shape[0] });
    assert!(inner_len <= i32::max_value() as _);*/
    let inner_len = if spec.l_blk_t { spec.l_block[0] } else { spec.l_block[1] };
    assert_eq!(inner_len, if spec.r_blk_t { spec.r_block[1] } else { spec.r_block[0] });
    assert!(inner_len <= i32::max_value() as _);
    // FIXME FIXME: m, n should be the block size.
    let o_blknrow = if spec.l_blk_t { spec.l_block[1] } else { spec.l_block[0] };
    assert!(o_blknrow <= i32::max_value() as _);
    let o_blkncol = if spec.l_blk_t { spec.r_block[1] } else { spec.r_block[0] };
    assert!(o_blkncol <= i32::max_value() as _);
    let colmajor_m = o_blkncol;
    let colmajor_n = o_blknrow;
    let ldb = arg_ty_[0].shape[1];
    assert!(ldb <= i32::max_value() as _);
    let lda = arg_ty_[1].shape[1];
    assert!(lda <= i32::max_value() as _);
    let ldc = out_ty_.shape[1];
    assert!(ldc <= i32::max_value() as _);
    // FIXME FIXME: load dptrs to blocks.
    /*let b_nrowblk = arg_ty_[0].shape[0] / spec.l_block[0];
    assert_eq!(0, arg_ty_[0].shape[0] % spec.l_block[0]);
    let b_ncolblk = arg_ty_[0].shape[1] / spec.l_block[1];
    assert_eq!(0, arg_ty_[0].shape[1] % spec.l_block[1]);
    let a_nrowblk = arg_ty_[1].shape[0] / spec.r_block[0];
    assert_eq!(0, arg_ty_[1].shape[0] % spec.r_block[0]);
    let a_ncolblk = arg_ty_[1].shape[1] / spec.r_block[1];
    assert_eq!(0, arg_ty_[1].shape[1] % spec.r_block[1]);*/
    let nrowblk = arg_ty_[0].shape[0] / spec.l_block[0];
    assert_eq!(0, arg_ty_[0].shape[0] % spec.l_block[0]);
    assert_eq!(nrowblk, arg_ty_[1].shape[0] / spec.r_block[0]);
    assert_eq!(0, arg_ty_[1].shape[0] % spec.r_block[0]);
    let ncolblk = arg_ty_[0].shape[1] / spec.l_block[1];
    assert_eq!(0, arg_ty_[0].shape[1] % spec.l_block[1]);
    assert_eq!(ncolblk, arg_ty_[1].shape[1] / spec.r_block[1]);
    assert_eq!(0, arg_ty_[1].shape[1] % spec.r_block[1]);
    match env.pread_ref(arg[0].0, arg[0].1, /*CellEMode::Read,*/) {
      None => panic!("bug"),
      Some(e) => {
        match e.cel_ {
          &mut Cell_::Phy(ref _state, ref _clo, ref mut pcel) => {
            let (_, pcel_addr) = pcel.get_pm(arg[0].1, PMach::NvGpu);
            /*let pcel_ = Weak::upgrade(pcel_).unwrap();
            let gpu_cel = pcel_.as_any().downcast_ref::<GpuInnerCell>().unwrap();
            let base = gpu_cel.dptr;*/
            let base = TL_PCTX.with(|pctx| {
              let gpu_cel = pctx.nvgpu.as_ref().unwrap().lookup(pcel_addr).unwrap();
              gpu_cel.dptr
            });
            let inc = spec.l_dtype.size_bytes() as u64;
            /*let nrowblk = arg_ty_[0].shape[0] / spec.l_block[0];
            assert_eq!(0, arg_ty_[0].shape[0] % spec.l_block[0]);
            let ncolblk = arg_ty_[0].shape[1] / spec.l_block[1];
            assert_eq!(0, arg_ty_[0].shape[1] % spec.l_block[1]);*/
            let blknrow = spec.l_block[0] as u64;
            let blkncol = spec.l_block[1] as u64;
            let stride = arg_ty_[0].shape[1] as u64;
            let mut tmp = self.tmp_b.borrow_mut();
            tmp.clear();
            for j in 0 .. nrowblk as u64 {
              for i in 0 .. ncolblk as u64 {
                tmp.push(base + inc * (blkncol * i + stride * blknrow * j));
              }
            }
          }
          _ => panic!("bug")
        }
      }
    }
    match env.pread_ref(arg[1].0, arg[1].1, /*CellEMode::Read,*/) {
      None => panic!("bug"),
      Some(e) => {
        match e.cel_ {
          &mut Cell_::Phy(ref _state, ref _clo, ref mut pcel) => {
            let (_, pcel_addr) = pcel.get_pm(arg[1].1, PMach::NvGpu);
            /*let pcel_ = Weak::upgrade(pcel_).unwrap();
            let gpu_cel = pcel_.as_any().downcast_ref::<GpuInnerCell>().unwrap();
            let base = gpu_cel.dptr;*/
            let base = TL_PCTX.with(|pctx| {
              let gpu_cel = pctx.nvgpu.as_ref().unwrap().lookup(pcel_addr).unwrap();
              gpu_cel.dptr
            });
            let inc = spec.r_dtype.size_bytes() as u64;
            /*let nrowblk = arg_ty_[1].shape[0] / spec.r_block[0];
            assert_eq!(0, arg_ty_[1].shape[0] % spec.r_block[0]);
            let ncolblk = arg_ty_[1].shape[1] / spec.r_block[1];
            assert_eq!(0, arg_ty_[1].shape[1] % spec.r_block[1]);*/
            let blknrow = spec.r_block[0] as u64;
            let blkncol = spec.r_block[1] as u64;
            let stride = arg_ty_[1].shape[1] as u64;
            let mut tmp = self.tmp_a.borrow_mut();
            tmp.clear();
            for j in 0 .. nrowblk as u64 {
              for i in 0 .. ncolblk as u64 {
                tmp.push(base + inc * (blkncol * i + stride * blknrow * j));
              }
            }
          }
          _ => panic!("bug")
        }
      }
    }
    match env.pwrite_ref(out, oclk, /*CellEMode::Mutex,*/) {
      None => panic!("bug"),
      Some(e) => {
        match e.cel_ {
          &mut Cell_::Phy(ref _state, ref _clo, ref mut pcel) => {
            /*clo.thunk_.push(th);
            assert_eq!(clo.thunk.len(), oclk.up as usize);*/
            let (_, pcel_addr) = pcel.get_pm(oclk, PMach::NvGpu);
            /*let pcel_ = Weak::upgrade(pcel_).unwrap();
            let gpu_cel = pcel_.as_any().downcast_ref::<GpuInnerCell>().unwrap();
            let base = gpu_cel.dptr;*/
            let base = TL_PCTX.with(|pctx| {
              let gpu_cel = pctx.nvgpu.as_ref().unwrap().lookup(pcel_addr).unwrap();
              gpu_cel.dptr
            });
            let inc = spec.o_dtype.size_bytes() as u64;
            /*let o_nrowblk = if spec.l_blk_t {
              let ncolblk = arg_ty_[0].shape[1] / spec.l_block[1];
              assert_eq!(0, arg_ty_[0].shape[1] % spec.l_block[1]);
              ncolblk
            } else {
              let nrowblk = arg_ty_[0].shape[0] / spec.l_block[0];
              assert_eq!(0, arg_ty_[0].shape[0] % spec.l_block[0]);
              nrowblk
            };
            let o_ncolblk = if spec.r_blk_t {
              let nrowblk = arg_ty_[1].shape[0] / spec.r_block[0];
              assert_eq!(0, arg_ty_[1].shape[0] % spec.r_block[0]);
              nrowblk
            } else {
              let ncolblk = arg_ty_[1].shape[1] / spec.r_block[1];
              assert_eq!(0, arg_ty_[1].shape[1] % spec.r_block[1]);
              ncolblk
            };*/
            let o_blknrow = o_blknrow as u64;
            let o_blkncol = o_blkncol as u64;
            let stride = out_ty_.shape[1] as u64;
            let mut tmp = self.tmp_c.borrow_mut();
            tmp.clear();
            for j in 0 .. nrowblk as u64 {
              for i in 0 .. ncolblk as u64 {
                tmp.push(base + inc * (o_blkncol * i + stride * o_blknrow * j));
              }
            }
          }
          _ => panic!("bug")
        }
      }
    }
    let b_gputy = match spec.l_dtype {
      Dtype::Float32 => CUDA_R_32F,
      Dtype::Float16 => CUDA_R_16F,
      Dtype::BFloat16 => CUDA_R_16BF,
      _ => unimplemented!()
    };
    let a_gputy = match spec.r_dtype {
      Dtype::Float32 => CUDA_R_32F,
      Dtype::Float16 => CUDA_R_16F,
      Dtype::BFloat16 => CUDA_R_16BF,
      _ => unimplemented!()
    };
    let c_gputy = match spec.o_dtype {
      Dtype::Float32 => CUDA_R_32F,
      Dtype::Float16 => CUDA_R_16F,
      Dtype::BFloat16 => CUDA_R_16BF,
      _ => unimplemented!()
    };
    TL_PCTX.with(|pctx| {
      let gpu = &pctx.nvgpu.as_ref().unwrap();
      let ret = cublas_gemm_batched(
          &gpu.blas_ctx,
          colmajor_at, colmajor_bt,
          colmajor_m as _, colmajor_n as _, inner_len as _,
          self.alpha.as_ptr() as *const f32 as *const c_void,
          &*self.tmp_a.borrow(), a_gputy, lda as _,
          &*self.tmp_b.borrow(), b_gputy, ldb as _,
          self.beta.as_ptr() as *const f32 as *const c_void,
          &*self.tmp_c.borrow(), c_gputy, ldc as _,
          &gpu.compute,
      );
      match ret {
        Err(_) => ThunkRet::Failure,
        Ok(_) => ThunkRet::Success
      }
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
