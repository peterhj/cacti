use super::*;
use crate::algo::fp::{TotalOrd};
use crate::cell::{DtypeExt, Dim};
use cacti_gpu_cu_ffi::{cublas_gemm_batched};
use cacti_gpu_cu_ffi::types::{CUDA_R_32F, CUDA_R_16F};

use futhark_syntax::{Exp as FutExp};

use std::borrow::{Cow};
use std::cell::{Cell};
//use std::io::{Write};

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct LamFutExpThunkSpec {
  pub lam_src: Cow<'static, str>,
  pub lam_exp: FutExp,
  pub ar_in: u16,
  pub ar_out: u16,
  // FIXME FIXME
  //pub arg_dim: Vec<Dim>,
  //pub out_dim: RefCell<Option<Vec<Dim>>>,
  pub wrap_parens: bool,
}

impl FutharkThunkSpec for LamFutExpThunkSpec {
  fn arity(&self) -> (u16, u16) {
    (self.ar_in, self.ar_out)
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

  fn gen_futhark(&self, ) -> FutharkThunkCode {
    let mut s = String::new();
    if self.wrap_parens {
      write!(&mut s, "({})", &self.lam_src).unwrap();
    } else {
      s.push_str(&self.lam_src);
    }
    for k in 0 .. self.ar_in {
      write!(&mut s, " {{%{}}}", k).unwrap();
    }
    FutharkThunkCode{
      body: vec![s],
    }
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SetScalarFutThunkSpec<T> { pub val: T }

impl<T: DtypeExt + Copy + Eq + Any> FutharkThunkSpec for SetScalarFutThunkSpec<T> {
  fn arity(&self) -> (u16, u16) {
    (0, 1)
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

  fn gen_futhark(&self, ) -> FutharkThunkCode {
    let fmt = FutharkNumFormatter::default();
    FutharkThunkCode{
      // FIXME FIXME: futhark treats actual scalars as simply pointers to cpu mem.
      /*body:     vec![format!("let {{%0}} = [{}] in", fmt.format(&self.val))],*/
      body:     vec![format!("let {{%0}} = {} in", fmt.format(&self.val))],
    }
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SetScalar1dFutThunkSpec<T> { pub val: T }

impl<T: DtypeExt + Copy + Eq + Any> FutharkThunkSpec for SetScalar1dFutThunkSpec<T> {
  fn arity(&self) -> (u16, u16) {
    (0, 1)
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

  fn gen_futhark(&self, ) -> FutharkThunkCode {
    let fmt = FutharkNumFormatter::default();
    FutharkThunkCode{
      body:     vec![format!("let {{%0}} = replicate {{%0.s[0]}} {} in", fmt.format(&self.val))],
    }
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SetScalar2dFutThunkSpec<T> { pub val: T }

impl<T: DtypeExt + Copy + Eq + Any> FutharkThunkSpec for SetScalar2dFutThunkSpec<T> {
  fn arity(&self) -> (u16, u16) {
    (0, 1)
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

  fn gen_futhark(&self, ) -> FutharkThunkCode {
    let fmt = FutharkNumFormatter::default();
    FutharkThunkCode{
      body:     vec![
                    format!("let t0 = replicate ({{%0.s[0]}} * {{%0.s[1]}}) {} in", fmt.format(&self.val)),
                    format!("let {{%0}} = unflatten {{%0.s[0]}} {{%0.s[1]}} t0 in"),
                ],
    }
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SetScalar3dFutThunkSpec<T> { pub val: T }

impl<T: DtypeExt + Copy + Eq + Any> FutharkThunkSpec for SetScalar3dFutThunkSpec<T> {
  fn arity(&self) -> (u16, u16) {
    (0, 1)
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

  fn gen_futhark(&self, ) -> FutharkThunkCode {
    let fmt = FutharkNumFormatter::default();
    FutharkThunkCode{
      body:     vec![
                    format!("let t0 = replicate ({{%0.s[0]}} * {{%0.s[1]}} * {{%0.s[2]}}) {} in", fmt.format(&self.val)),
                    format!("let {{%0}} = unflatten_3d {{%0.s[0]}} {{%0.s[1]}} {{%0.s[2]}} t0 in"),
                ],
    }
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SetScalar4dFutThunkSpec<T> { pub val: T }

impl<T: DtypeExt + Copy + Eq + Any> FutharkThunkSpec for SetScalar4dFutThunkSpec<T> {
  fn arity(&self) -> (u16, u16) {
    (0, 1)
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

  fn gen_futhark(&self, ) -> FutharkThunkCode {
    let fmt = FutharkNumFormatter::default();
    FutharkThunkCode{
      body:     vec![
                    format!("let t0 = replicate ({{%0.s[0]}} * {{%0.s[1]}} * {{%0.s[2]}} * {{%0.s[3]}}) {} in", fmt.format(&self.val)),
                    format!("let {{%0}} = unflatten_4d {{%0.s[0]}} {{%0.s[1]}} {{%0.s[2]}} {{%0.s[3]}} t0 in"),
                ],
    }
  }
}

/*#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct DowncastF32F16FutThunkSpec;

impl FutharkThunkSpec for DowncastF32F16FutThunkSpec {
  fn gen_futhark(&self, ) -> FutharkThunkCode {
    FutharkThunkCode{
      arityin:  1,
      arityout: 1,
      body:     vec![format!("let {{%1}} = f16.f32 {{%0}} in")],
    }
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct UpcastF16F32FutThunkSpec;

impl FutharkThunkSpec for UpcastF16F32FutThunkSpec {
  fn gen_futhark(&self, ) -> FutharkThunkCode {
    FutharkThunkCode{
      arityin:  1,
      arityout: 1,
      body:     vec![format!("let {{%1}} = f32.f16 {{%0}} in")],
    }
  }
}*/

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct CastFutThunkSpec { pub org_dtype: Dtype, pub new_dtype: Dtype }

impl FutharkThunkSpec for CastFutThunkSpec {
  fn arity(&self) -> (u16, u16) {
    (1, 1)
  }

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(Dim{ndim: arg[0].ndim, dtype: self.new_dtype})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(CellType{shape: arg[0].shape.clone(), dtype: self.new_dtype})
  }

  fn gen_futhark(&self, ) -> FutharkThunkCode {
    FutharkThunkCode{
      body:     vec![format!("let {{%1}} = {}.{} {{%0}} in",
                    self.new_dtype.format_futhark(),
                    self.org_dtype.format_futhark(),
                )],
    }
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct AddScalarF32FutThunkSpec { pub val: TotalOrd<f32> }

impl FutharkThunkSpec for AddScalarF32FutThunkSpec {
  fn arity(&self) -> (u16, u16) {
    (1, 1)
  }

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(Dim{ndim: arg[0].ndim, dtype: f32::dtype()})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(CellType{shape: arg[0].shape.clone(), dtype: f32::dtype()})
  }

  fn gen_futhark(&self, ) -> FutharkThunkCode {
    let fmt = FutharkNumFormatter::default();
    FutharkThunkCode{
      body:     vec![format!("let {{%1}} = {{%0}} + {} in", fmt.format(&self.val))],
    }
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct AddFutThunkSpec;

impl FutharkThunkSpec for AddFutThunkSpec {
  fn arity(&self) -> (u16, u16) {
    (2, 1)
  }

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(Dim{ndim: arg[0].ndim, dtype: arg[0].dtype})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(CellType{shape: arg[0].shape.clone(), dtype: arg[0].dtype})
  }

  fn gen_futhark(&self, ) -> FutharkThunkCode {
    FutharkThunkCode{
      body:     vec![format!("let {{%2}} = {{%0}} + {{%1}} in")],
    }
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SubScalarF32FutThunkSpec { pub val: TotalOrd<f32> }

impl FutharkThunkSpec for SubScalarF32FutThunkSpec {
  fn arity(&self) -> (u16, u16) {
    (1, 1)
  }

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(Dim{ndim: arg[0].ndim, dtype: f32::dtype()})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(CellType{shape: arg[0].shape.clone(), dtype: f32::dtype()})
  }

  fn gen_futhark(&self, ) -> FutharkThunkCode {
    let fmt = FutharkNumFormatter::default();
    FutharkThunkCode{
      body:     vec![format!("let {{%1}} = {{%0}} - {} in", fmt.format(&self.val))],
    }
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct SubFutThunkSpec;

impl FutharkThunkSpec for SubFutThunkSpec {
  fn arity(&self) -> (u16, u16) {
    (2, 1)
  }

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(Dim{ndim: arg[0].ndim, dtype: arg[0].dtype})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(CellType{shape: arg[0].shape.clone(), dtype: arg[0].dtype})
  }

  fn gen_futhark(&self, ) -> FutharkThunkCode {
    FutharkThunkCode{
      body:     vec![format!("let {{%2}} = {{%0}} - {{%1}} in")],
    }
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct MulScalarF32FutThunkSpec { pub val: TotalOrd<f32> }

impl FutharkThunkSpec for MulScalarF32FutThunkSpec {
  fn arity(&self) -> (u16, u16) {
    (1, 1)
  }

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(Dim{ndim: arg[0].ndim, dtype: f32::dtype()})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(CellType{shape: arg[0].shape.clone(), dtype: arg[0].dtype})
  }

  fn gen_futhark(&self, ) -> FutharkThunkCode {
    let fmt = FutharkNumFormatter::default();
    FutharkThunkCode{
      body:     vec![format!("let {{%1}} = {{%0}} * {} in", fmt.format(&self.val))],
    }
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct MulFutThunkSpec;

impl FutharkThunkSpec for MulFutThunkSpec {
  fn arity(&self) -> (u16, u16) {
    (2, 1)
  }

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(Dim{ndim: arg[0].ndim, dtype: arg[0].dtype})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(CellType{shape: arg[0].shape.clone(), dtype: arg[0].dtype})
  }

  fn gen_futhark(&self, ) -> FutharkThunkCode {
    FutharkThunkCode{
      body:     vec![format!("let {{%2}} = {{%0}} * {{%1}} in")],
    }
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct DivScalarF32FutThunkSpec { pub val: TotalOrd<f32> }

impl FutharkThunkSpec for DivScalarF32FutThunkSpec {
  fn arity(&self) -> (u16, u16) {
    (1, 1)
  }

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(Dim{ndim: arg[0].ndim, dtype: f32::dtype()})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(CellType{shape: arg[0].shape.clone(), dtype: f32::dtype()})
  }

  fn gen_futhark(&self, ) -> FutharkThunkCode {
    let fmt = FutharkNumFormatter::default();
    FutharkThunkCode{
      body:     vec![format!("let {{%1}} = {{%0}} / {} in", fmt.format(&self.val))],
    }
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct DivFutThunkSpec;

impl FutharkThunkSpec for DivFutThunkSpec {
  fn arity(&self) -> (u16, u16) {
    (2, 1)
  }

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(Dim{ndim: arg[0].ndim, dtype: arg[0].dtype})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(CellType{shape: arg[0].shape.clone(), dtype: arg[0].dtype})
  }

  fn gen_futhark(&self, ) -> FutharkThunkCode {
    FutharkThunkCode{
      body:     vec![format!("let {{%2}} = {{%0}} / {{%1}} in")],
    }
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct SqrtFutThunkSpec;

impl FutharkThunkSpec for SqrtFutThunkSpec {
  fn arity(&self) -> (u16, u16) {
    (1, 1)
  }

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(Dim{ndim: arg[0].ndim, dtype: arg[0].dtype})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(CellType{shape: arg[0].shape.clone(), dtype: arg[0].dtype})
  }

  fn gen_futhark(&self, ) -> FutharkThunkCode {
    FutharkThunkCode{
      body:     vec![format!("let {{%1}} = sqrt {{%0}} in")],
    }
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct RsqrtFutThunkSpec;

impl FutharkThunkSpec for RsqrtFutThunkSpec {
  fn arity(&self) -> (u16, u16) {
    (1, 1)
  }

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(Dim{ndim: arg[0].ndim, dtype: arg[0].dtype})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(CellType{shape: arg[0].shape.clone(), dtype: arg[0].dtype})
  }

  fn gen_futhark(&self, ) -> FutharkThunkCode {
    FutharkThunkCode{
      // FIXME FIXME
      body:     vec![format!("let {{%1}} = recip (sqrt {{%0}}) in")],
      //body:     vec![format!("let {{%1}} = rsqrt {{%0}} in")],
    }
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct CosFutThunkSpec;

impl FutharkThunkSpec for CosFutThunkSpec {
  fn arity(&self) -> (u16, u16) {
    (1, 1)
  }

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(Dim{ndim: arg[0].ndim, dtype: arg[0].dtype})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(CellType{shape: arg[0].shape.clone(), dtype: arg[0].dtype})
  }

  fn gen_futhark(&self, ) -> FutharkThunkCode {
    FutharkThunkCode{
      body:     vec![format!("let {{%1}} = cos {{%0}} in")],
    }
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct SinFutThunkSpec;

impl FutharkThunkSpec for SinFutThunkSpec {
  fn arity(&self) -> (u16, u16) {
    (1, 1)
  }

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(Dim{ndim: arg[0].ndim, dtype: arg[0].dtype})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(CellType{shape: arg[0].shape.clone(), dtype: arg[0].dtype})
  }

  fn gen_futhark(&self, ) -> FutharkThunkCode {
    FutharkThunkCode{
      body:     vec![format!("let {{%1}} = sin {{%0}} in")],
    }
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct ExpFutThunkSpec;

impl FutharkThunkSpec for ExpFutThunkSpec {
  fn arity(&self) -> (u16, u16) {
    (1, 1)
  }

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(Dim{ndim: arg[0].ndim, dtype: arg[0].dtype})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(CellType{shape: arg[0].shape.clone(), dtype: arg[0].dtype})
  }

  fn gen_futhark(&self, ) -> FutharkThunkCode {
    FutharkThunkCode{
      body:     vec![format!("let {{%1}} = exp {{%0}} in")],
    }
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct TanhFutThunkSpec;

impl FutharkThunkSpec for TanhFutThunkSpec {
  fn arity(&self) -> (u16, u16) {
    (1, 1)
  }

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(Dim{ndim: arg[0].ndim, dtype: arg[0].dtype})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(CellType{shape: arg[0].shape.clone(), dtype: arg[0].dtype})
  }

  fn gen_futhark(&self, ) -> FutharkThunkCode {
    FutharkThunkCode{
      body:     vec![format!("let {{%1}} = ((exp {{%0}}) - (exp -{{%0}})) / ((exp {{%0}}) + (exp -{{%0}})) in")],
    }
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct PowiF32FutThunkSpec { pub exp: i64 }

impl FutharkThunkSpec for PowiF32FutThunkSpec {
  fn arity(&self) -> (u16, u16) {
    (1, 1)
  }

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(Dim{ndim: arg[0].ndim, dtype: f32::dtype()})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(CellType{shape: arg[0].shape.clone(), dtype: arg[0].dtype})
  }

  fn gen_futhark(&self, ) -> FutharkThunkCode {
    FutharkThunkCode{
      // FIXME FIXME
      body:     vec![format!("let {{%1}} = {{%0}} ** (f32.i64 {}) in", self.exp)],
    }
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
  fn arity(&self) -> (u16, u16) {
    (1, 1)
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

  fn gen_futhark(&self, ) -> FutharkThunkCode {
    FutharkThunkCode{
      body:     vec![format!("let {{%1}} = reduce (+) 0 {{%0}} in")],
    }
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct Sum2dFutThunkSpec;

impl FutharkThunkSpec for Sum2dFutThunkSpec {
  fn arity(&self) -> (u16, u16) {
    (1, 1)
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

  fn gen_futhark(&self, ) -> FutharkThunkCode {
    FutharkThunkCode{
      // FIXME: instead, could reshape and reduce once.
      /*body:     vec![format!("let {{%1}} = reduce (\t1 -> reduce (+) 0 t1) 0 {{%0}} in")],*/
      body:     vec![
                    format!("let t0 = flatten {{%0}} in"),
                    format!("let t1 = reduce (+) 0 t0 in"),
                    format!("let {{%1}} = unflatten {{%0.s[0]}} {{%0.s[1]}} t1 in"),
                ],
    }
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct Sum3dFutThunkSpec;

impl FutharkThunkSpec for Sum3dFutThunkSpec {
  fn arity(&self) -> (u16, u16) {
    (1, 1)
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

  fn gen_futhark(&self, ) -> FutharkThunkCode {
    FutharkThunkCode{
      // FIXME: instead, could reshape and reduce once.
      /*body:     vec![format!("let {{%1}} = reduce (\t2 -> reduce (\t1 -> reduce (+) 0 t1) 0 t2) 0 {{%0}} in")],*/
      body:     vec![
                    format!("let t0 = flatten_3d {{%0}} in"),
                    format!("let t1 = reduce (+) 0 t0 in"),
                    format!("let {{%1}} = unflatten_3d {{%0.s[0]}} {{%0.s[1]}} {{%0.s[2]}} t1 in"),
                ],
    }
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct Sum4dFutThunkSpec;

impl FutharkThunkSpec for Sum4dFutThunkSpec {
  fn arity(&self) -> (u16, u16) {
    (1, 1)
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

  fn gen_futhark(&self, ) -> FutharkThunkCode {
    FutharkThunkCode{
      // FIXME: instead, could reshape and reduce once.
      /*body:     vec![format!("let {{%1}} = reduce (\t3 -> reduce (\t2 -> reduce (\t1 -> reduce (+) 0 t1) 0 t2) 0 t3) 0 {{%0}} in")],*/
      body:     vec![
                    format!("let t0 = flatten_4d {{%0}} in"),
                    format!("let t1 = reduce (+) 0 t0 in"),
                    format!("let {{%1}} = unflatten_4d {{%0.s[0]}} {{%0.s[1]}} {{%0.s[2]}} {{%0.s[3]}} t1 in"),
                ],
    }
  }
}

/*#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct DotThunkSpec;*/

/*impl CustomThunk_ for AddScalarF32ThunkSpec {
}*/

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct BlockMulMatrixThunkSpec {
  //pub l_shape:  [i64; 2],
  //pub r_shape:  [i64; 2],
  pub l_block:  [i64; 2],
  pub r_block:  [i64; 2],
  //pub l_nblock: [i64; 2],
  //pub r_nblock: [i64; 2],
  pub lt:       bool,
  pub rt:       bool,
  pub l_dtype:  Dtype,
  pub r_dtype:  Dtype,
  pub o_dtype:  Dtype,
}

impl ThunkSpec for BlockMulMatrixThunkSpec {
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
    let (m, n) = match (self.lt, self.rt) {
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
    };
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
    if self.l_dtype != arg[0].dtype {
      return Err(ThunkTypeErr::_Bot);
    }
    if self.r_dtype != arg[1].dtype {
      return Err(ThunkTypeErr::_Bot);
    }
    Ok(CellType{shape: vec![m, n], dtype: self.o_dtype})
  }

  fn gen_impl_(&self, spec_dim: Vec<Dim>, pmach: PMach) -> Option<Rc<dyn ThunkImpl_>> {
    // FIXME FIXME
    unimplemented!();
  }
}

pub struct BlockMulMatrixGpuThunkImpl {
  // TODO
  alpha: Cell<f32>,
  beta: Cell<f32>,
  tmp_a: RefCell<Vec<u64>>,
  tmp_b: RefCell<Vec<u64>>,
  tmp_c: RefCell<Vec<u64>>,
}

impl ThunkImpl for BlockMulMatrixGpuThunkImpl {
  fn apply(&self, ctr: &CtxCtr, env: &mut CtxEnv, spec_: &dyn ThunkSpec_, arg: &[(CellPtr, Clock)], out: CellPtr) -> ThunkRet {
    let spec = spec_.as_any().downcast_ref::<BlockMulMatrixThunkSpec>().unwrap();
    self.alpha.set(1.0);
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
    let m = out_ty_.shape[0];
    assert!(m <= i32::max_value() as _);
    let n = out_ty_.shape[1];
    assert!(n <= i32::max_value() as _);
    let inner_len = if spec.lt { arg_ty_[0].shape[0] } else { arg_ty_[0].shape[1] };
    assert_eq!(inner_len, if spec.rt { arg_ty_[1].shape[1] } else { arg_ty_[1].shape[0] });
    assert!(inner_len <= i32::max_value() as _);
    // FIXME FIXME: load dptrs to blocks.
    /*
    let a_nrowblk = arg_ty_[0].shape[0] / self.l_block[0];
    let a_ncolblk = arg_ty_[0].shape[1] / self.l_block[1];
    match env.pread_ref(arg[0]) {
      None => panic!("bug"),
      Some(e) => {
        match e.cel_ {
          &mut Cell::Phy(ref _state, ref mut pcel) => {
            let pcel_ = pcel.get(PMach::NvGpu).unwrap();
            let gpu_cel = pcel_.as_any().downcast_ref::<GpuInnerCell>().unwrap();
            //gpu_cel.dptr
            // FIXME FIXME
            //self.tmp_a.borrow_mut()...;
          }
          _ => panic!("bug")
        }
      }
    }
    // TODO TODO
    match env.pwrite_ref(out) {
      None => panic!("bug"),
      Some(e) => {
        match e.cel_ {
          &mut Cell::Phy(ref _state, ref mut pcel) => {
            let pcel_ = pcel.get(PMach::NvGpu).unwrap();
            let gpu_cel = pcel_.as_any().downcast_ref::<GpuInnerCell>().unwrap();
            //gpu_cel.dptr
            // FIXME FIXME
            //self.tmp_c.borrow_mut()...;
          }
          _ => panic!("bug")
        }
      }
    }
    */
    let lda = arg_ty_[0].shape[1];
    assert!(lda <= i32::max_value() as _);
    let ldb = arg_ty_[1].shape[1];
    assert!(ldb <= i32::max_value() as _);
    let ldc = out_ty_.shape[1];
    assert!(ldc <= i32::max_value() as _);
    let a_gputy = match spec.l_dtype {
      Dtype::Float32 => CUDA_R_32F,
      Dtype::Float16 => CUDA_R_16F,
      _ => unimplemented!()
    };
    let b_gputy = match spec.r_dtype {
      Dtype::Float32 => CUDA_R_32F,
      Dtype::Float16 => CUDA_R_16F,
      _ => unimplemented!()
    };
    let c_gputy = match spec.o_dtype {
      Dtype::Float32 => CUDA_R_32F,
      Dtype::Float16 => CUDA_R_16F,
      _ => unimplemented!()
    };
    TL_PCTX.with(|pctx| {
      // FIXME FIXME
      let gpu = &pctx.nvgpu.as_ref().unwrap();
      let ret = cublas_gemm_batched(
          &gpu.blas_ctx,
          spec.lt, spec.rt,
          m as _, n as _, inner_len as _,
          self.alpha.as_ptr() as *const _,
          &*self.tmp_a.borrow(), a_gputy, lda as _,
          &*self.tmp_b.borrow(), b_gputy, ldb as _,
          self.beta.as_ptr() as *const _,
          &*self.tmp_c.borrow(), c_gputy, ldc as _,
          &gpu.main,
      );
      match ret {
        Err(_) => ThunkRet::Failure,
        Ok(_) => ThunkRet::Success
      }
    })
  }

  fn accumulate(&self, ctr: &CtxCtr, env: &mut CtxEnv, spec_: &dyn ThunkSpec_, arg: &[(CellPtr, Clock)], out: CellPtr) -> ThunkRet {
    let spec = spec_.as_any().downcast_ref::<BlockMulMatrixThunkSpec>().unwrap();
    self.alpha.set(1.0);
    self.beta.set(1.0);
    unimplemented!();
    /*
    TL_PCTX.with(|pctx| {
      // FIXME FIXME
    })
    */
  }
}
