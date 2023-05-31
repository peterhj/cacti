use super::*;
use crate::algo::fp::{TotalOrd};
use crate::cell::{DtypeExt, Dim};

//use std::io::{Write};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SetScalarFutThunkSpec<T> { pub val: T }

impl<T: DtypeExt + Eq + Any> FutharkThunkSpec for SetScalarFutThunkSpec<T> {
  fn arity(&self) -> (u16, u16) {
    (0, 1)
  }

  fn out_dim(&self, _arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    Ok(Dim{ndim: 0, dtype: T::dtype()})
  }

  fn out_ty_(&self, _arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    Ok(CellType{shape: Vec::new(), dtype: T::dtype()})
  }

  fn gen_futhark(&self, ) -> FutharkThunkCode {
    let fmt = FutharkNumFormatter::default();
    FutharkThunkCode{
      // FIXME FIXME: futhark treats actual scalars as simply pointers to cpu mem.
      body:     vec![format!("let {{%0}} = [{}] in", fmt.format(&self.val))],
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
pub struct DotThunkSpec;

/*impl CustomThunk_ for AddScalarF32ThunkSpec {
}*/

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct BlockMulMatrixThunkSpec {
  pub dtype:    Dtype,
  pub lt:       bool,
  pub rt:       bool,
  // FIXME FIXME
  pub l_shape:  [i64; 2],
  pub r_shape:  [i64; 2],
  pub l_block:  [i64; 2],
  pub r_block:  [i64; 2],
  pub l_nblock: [i64; 2],
  pub r_nblock: [i64; 2],
}

impl ThunkSpec for BlockMulMatrixThunkSpec {
  fn arity(&self) -> (u16, u16) {
    (2, 1)
  }

  fn out_dim(&self, arg: &[Dim]) -> Result<Dim, ThunkDimErr> {
    // FIXME FIXME
    if arg[0].ndim != 2 {
      return Err(ThunkDimErr::_Bot);
    }
    if arg[1].ndim != 2 {
      return Err(ThunkDimErr::_Bot);
    }
    if arg[0].dtype != arg[1].dtype {
      return Err(ThunkDimErr::_Bot);
    }
    Ok(Dim{ndim: 2, dtype: arg[0].dtype})
  }

  fn out_ty_(&self, arg: &[CellType]) -> Result<CellType, ThunkTypeErr> {
    // FIXME FIXME
    if arg[0].shape.len() != 2 {
      return Err(ThunkTypeErr::_Bot);
    }
    if arg[1].shape.len() != 2 {
      return Err(ThunkTypeErr::_Bot);
    }
    if arg[0].shape[1] != arg[1].shape[0] {
      return Err(ThunkTypeErr::_Bot);
    }
    if arg[0].dtype != arg[1].dtype {
      return Err(ThunkTypeErr::_Bot);
    }
    Ok(CellType{shape: vec![arg[0].shape[0], arg[1].shape[1]], dtype: arg[0].dtype})
  }

  // FIXME FIXME
}

pub struct BlockMulMatrixGpuThunkImpl {
}
