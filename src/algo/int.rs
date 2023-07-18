pub trait UintConstExt: Sized {
  fn zero()     -> Self;
  fn one()      -> Self;
}

pub trait IntConstExt: UintConstExt {
  fn neg_one()  -> Self;
}

impl UintConstExt for i8 {
  fn zero()     -> i8 { 0_i8 }
  fn one()      -> i8 { 1_i8 }
}

impl IntConstExt for i8 {
  fn neg_one()  -> i8 { -1_i8 }
}

impl UintConstExt for i16 {
  fn zero()     -> i16 { 0_i16 }
  fn one()      -> i16 { 1_i16 }
}

impl IntConstExt for i16 {
  fn neg_one()  -> i16 { -1_i16 }
}

impl UintConstExt for i32 {
  fn zero()     -> i32 { 0_i32 }
  fn one()      -> i32 { 1_i32 }
}

impl IntConstExt for i32 {
  fn neg_one()  -> i32 { -1_i32 }
}

impl UintConstExt for i64 {
  fn zero()     -> i64 { 0_i64 }
  fn one()      -> i64 { 1_i64 }
}

impl IntConstExt for i64 {
  fn neg_one()  -> i64 { -1_i64 }
}

impl UintConstExt for u8 {
  fn zero()     -> u8 { 0_u8 }
  fn one()      -> u8 { 1_u8 }
}

impl UintConstExt for u16 {
  fn zero()     -> u16 { 0_u16 }
  fn one()      -> u16 { 1_u16 }
}

impl UintConstExt for u32 {
  fn zero()     -> u32 { 0_u32 }
  fn one()      -> u32 { 1_u32 }
}

impl UintConstExt for u64 {
  fn zero()     -> u64 { 0_u64 }
  fn one()      -> u64 { 1_u64 }
}
