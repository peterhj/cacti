use super::*;

//use std::io::{Write};

#[derive(Clone, Copy, Default)] pub struct DowncastF32F16ThunkOp {}

/*impl FutharkThunk_ for DowncastF32F16ThunkOp {
  fn _arg_count(&self) -> u8 {
    1
  }

  fn _body(&self) -> Vec<u8> {
    b"f16.f32 {%0}\n".to_owned()
  }
}*/

#[derive(Clone, Copy, Default)] pub struct UpcastF16F32ThunkOp {}

/*impl FutharkThunk_ for UpcastF16F32ThunkOp {
  fn _arg_count(&self) -> u8 {
    1
  }

  fn _body(&self) -> Vec<u8> {
    b"f32.f16 {%0}\n".to_owned()
  }
}*/

#[derive(Clone, Copy)] pub struct AddScalarF32ThunkOp { pub scalar: f32 }

/*impl FutharkThunk_ for AddScalarF32ThunkOp {
  fn _arg_count(&self) -> u8 {
    1
  }

  fn _body(&self) -> Vec<u8> {
    let mut buf = Vec::new();
    writeln!(&mut buf, b"{{%0}} + {}f32\n", self.scalar).unwrap();
    buf
  }
}*/

#[derive(Clone, Copy, Default)] pub struct AddThunkOp {}

/*impl FutharkThunk_ for AddThunkOp {
  fn _arg_count(&self) -> u8 {
    2
  }

  fn _body(&self) -> Vec<u8> {
    b"{%0} + {%1}\n".to_owned()
  }
}*/

#[derive(Clone, Copy, Default)] pub struct SubThunkOp {}

#[derive(Clone, Copy)] pub struct MulScalarF32ThunkOp { pub scalar: f32 }

#[derive(Clone, Copy)] pub struct DivScalarF32ThunkOp { pub scalar: f32 }

#[derive(Clone, Copy, Default)] pub struct DivThunkOp {}

/*impl FutharkThunk_ for DivThunkOp {
  fn _arg_count(&self) -> u8 {
    2
  }

  fn _body(&self) -> Vec<u8> {
    b"{%0} / {%1}\n".to_owned()
  }
}*/

#[derive(Clone, Copy, Default)] pub struct SqrtThunkOp {}

/*impl FutharkThunk_ for DivThunkOp {
  fn _arg_count(&self) -> u8 {
    1
  }

  fn _body(&self) -> Vec<u8> {
    b"sqrt {%0}\n".to_owned()
  }
}*/

#[derive(Clone, Copy, Default)] pub struct RsqrtThunkOp {}

#[derive(Clone, Copy)] pub struct PowiThunkOp { pub exp: i64 }

#[derive(Clone, Copy, Default)] pub struct InnerMax3dThunkOp {}

/*impl FutharkThunk_ for InnerMax3dThunkOp {
  fn _arg_count(&self) -> u8 {
    1
  }

  fn _body(&self) -> Vec<u8> {
    b"map (\t1 -> map (\t2 -> [reduce max -inf t2]) t1) {%0}\n".to_owned()
  }
}*/

#[derive(Clone, Copy, Default)] pub struct InnerMean3dThunkOp {}

/*impl FutharkThunk_ for InnerSum3dThunkOp {
  fn _arg_count(&self) -> u8 {
    1
  }

  fn _body(&self) -> Vec<u8> {
    b"map (\t1 -> map (\t2 -> [(reduce (+) 0 t2) / ({%t.0}.i64 (length t2))]) t1) {%0}\n".to_owned()
  }
}*/

#[derive(Clone, Copy, Default)] pub struct InnerSum3dThunkOp {}

/*impl FutharkThunk_ for InnerSum3dThunkOp {
  fn _arg_count(&self) -> u8 {
    1
  }

  fn _body(&self) -> Vec<u8> {
    b"map (\t1 -> map (\t2 -> [reduce (+) 0 t2]) t1) {%0}\n".to_owned()
  }
}*/

#[derive(Clone, Copy, Default)] pub struct DotThunkOp {}

/*impl CustomThunk_ for AddScalarF32ThunkOp {
}*/
