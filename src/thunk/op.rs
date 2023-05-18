use super::*;
use crate::algo::fp::{NonNan};
use crate::cell::{DtypeExt};
use crate::cell::gpu::*;
use crate::cell::smp::*;

//use std::io::{Write};

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct CopyScalarThunkSpec<T> { pub val: T }

/*impl<T: DtypeExt + Eq> ThunkSpec for CopyScalarThunkSpec<T> {
  fn gen_impl_smp(&self, ) -> Option<Box<dyn ThunkImpl_<Cel=SmpInnerCell>>> {
    unimplemented!();
  }

  fn gen_impl_gpu(&self, ) -> Option<Box<dyn ThunkImpl_<Cel=GpuInnerCell>>> {
    unimplemented!();
  }
}*/

impl<T: DtypeExt + Eq + Any> FutharkThunkSpec for CopyScalarThunkSpec<T> {
  fn gen_futhark(&self, ) -> FutharkThunkCode {
    let fmt = FutharkNumFormatter::default();
    FutharkThunkCode{
      arityin:  0,
      arityout: 1,
      body:     format!("let {{%0}} = {} in\n", fmt.format(&self.val)),
    }
  }
}

#[derive(Clone, Copy, Default)] pub struct DowncastF32F16ThunkSpec {}

/*impl FutharkThunk_ for DowncastF32F16ThunkSpec {
  fn _arg_count(&self) -> u8 {
    1
  }

  fn _body(&self) -> Vec<u8> {
    b"f16.f32 {%0}\n".to_owned()
  }
}*/

#[derive(Clone, Copy, Default)] pub struct UpcastF16F32ThunkSpec {}

/*impl FutharkThunk_ for UpcastF16F32ThunkSpec {
  fn _arg_count(&self) -> u8 {
    1
  }

  fn _body(&self) -> Vec<u8> {
    b"f32.f16 {%0}\n".to_owned()
  }
}*/

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct AddScalarF32ThunkSpec { pub scalar: NonNan<f32> }
impl ThunkSpec for AddScalarF32ThunkSpec {}

/*impl FutharkThunk_ for AddScalarF32ThunkSpec {
  fn _arg_count(&self) -> u8 {
    1
  }

  fn _body(&self) -> Vec<u8> {
    let mut buf = Vec::new();
    writeln!(&mut buf, b"{{%0}} + {}f32\n", self.scalar).unwrap();
    buf
  }
}*/

#[derive(Clone, Copy, Default)] pub struct AddThunkSpec {}

/*impl FutharkThunk_ for AddThunkSpec {
  fn _arg_count(&self) -> u8 {
    2
  }

  fn _body(&self) -> Vec<u8> {
    b"{%0} + {%1}\n".to_owned()
  }
}*/

impl FutharkThunkSpec for AddThunkSpec {
  fn gen_futhark(&self, ) -> FutharkThunkCode {
    FutharkThunkCode{
      arityin:  2,
      arityout: 1,
      body:     format!("let {{%2}} = {{%0}} + {{%1}} in\n"),
    }
  }
}

#[derive(Clone, Copy, Default)] pub struct SubThunkSpec {}

impl FutharkThunkSpec for SubThunkSpec {
  fn gen_futhark(&self, ) -> FutharkThunkCode {
    FutharkThunkCode{
      arityin:  2,
      arityout: 1,
      body:     format!("let {{%2}} = {{%0}} - {{%1}} in\n"),
    }
  }
}

#[derive(Clone, Copy)] pub struct MulScalarF32ThunkSpec { pub scalar: f32 }

#[derive(Clone, Copy, Default)] pub struct MulThunkSpec {}

impl FutharkThunkSpec for MulThunkSpec {
  fn gen_futhark(&self, ) -> FutharkThunkCode {
    FutharkThunkCode{
      arityin:  2,
      arityout: 1,
      body:     format!("let {{%2}} = {{%0}} / {{%1}} in\n"),
    }
  }
}

#[derive(Clone, Copy)] pub struct DivScalarF32ThunkSpec { pub scalar: f32 }

#[derive(Clone, Copy, Default)] pub struct DivThunkSpec {}

/*impl FutharkThunk_ for DivThunkSpec {
  fn _arg_count(&self) -> u8 {
    2
  }

  fn _body(&self) -> Vec<u8> {
    b"{%0} / {%1}\n".to_owned()
  }
}*/

impl FutharkThunkSpec for DivThunkSpec {
  fn gen_futhark(&self, ) -> FutharkThunkCode {
    FutharkThunkCode{
      arityin:  2,
      arityout: 1,
      body:     format!("let {{%2}} = {{%0}} / {{%1}} in\n"),
    }
  }
}

#[derive(Clone, Copy, Default)] pub struct SqrtThunkSpec {}

/*impl FutharkThunk_ for SqrtThunkSpec {
  fn _arg_count(&self) -> u8 {
    1
  }

  fn _body(&self) -> Vec<u8> {
    b"sqrt {%0}\n".to_owned()
  }
}*/

impl FutharkThunkSpec for SqrtThunkSpec {
  fn gen_futhark(&self, ) -> FutharkThunkCode {
    FutharkThunkCode{
      arityin:  1,
      arityout: 1,
      body:     format!("let {{%1}} = sqrt {{%0}} in\n"),
    }
  }
}

#[derive(Clone, Copy, Default)] pub struct RsqrtThunkSpec {}

impl FutharkThunkSpec for RsqrtThunkSpec {
  fn gen_futhark(&self, ) -> FutharkThunkCode {
    FutharkThunkCode{
      arityin:  1,
      arityout: 1,
      body:     format!("let {{%1}} = 1.0 / sqrt {{%0}} in\n"),
      //body:     format!("let {{%1}} = rsqrt {{%0}} in\n"),
    }
  }
}

#[derive(Clone, Copy)] pub struct PowiThunkSpec { pub exp: i64 }

#[derive(Clone, Copy, Default)] pub struct InnerMax3dThunkSpec {}

/*impl FutharkThunk_ for InnerMax3dThunkSpec {
  fn _arg_count(&self) -> u8 {
    1
  }

  fn _body(&self) -> Vec<u8> {
    b"map (\t1 -> map (\t2 -> [reduce max -inf t2]) t1) {%0}\n".to_owned()
  }
}*/

#[derive(Clone, Copy, Default)] pub struct InnerMean3dThunkSpec {}

/*impl FutharkThunk_ for InnerSum3dThunkSpec {
  fn _arg_count(&self) -> u8 {
    1
  }

  fn _body(&self) -> Vec<u8> {
    b"map (\t1 -> map (\t2 -> [(reduce (+) 0 t2) / ({%t.0}.i64 (length t2))]) t1) {%0}\n".to_owned()
  }
}*/

#[derive(Clone, Copy, Default)] pub struct InnerSum3dThunkSpec {}

/*impl FutharkThunk_ for InnerSum3dThunkSpec {
  fn _arg_count(&self) -> u8 {
    1
  }

  fn _body(&self) -> Vec<u8> {
    b"map (\t1 -> map (\t2 -> [reduce (+) 0 t2]) t1) {%0}\n".to_owned()
  }
}*/

#[derive(Clone, Copy, Default)] pub struct DotThunkSpec {}

/*impl CustomThunk_ for AddScalarF32ThunkSpec {
}*/
