pub use half::{f16, bf16};

use std::borrow::{Borrow};
use std::cmp::{Ordering};
use std::convert::{TryFrom};
use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::hash::{Hash, Hasher};

/* `TotalOrd` is derived from the implementation in libcore:

Short version for non-lawyers:

The Rust Project is dual-licensed under Apache 2.0 and MIT
terms.

Longer version:

Copyrights in the Rust project are retained by their contributors. No
copyright assignment is required to contribute to the Rust project.

Some files include explicit copyright notices and/or license notices.
For full authorship information, see the version control history or
https://thanks.rust-lang.org

Except as otherwise noted (below and/or in individual files), Rust is
licensed under the Apache License, Version 2.0 <LICENSE-APACHE> or
<http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
<LICENSE-MIT> or <http://opensource.org/licenses/MIT>, at your option. */

#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct TotalOrd<F: Copy>(pub F);

impl<F: Copy + Debug> Debug for TotalOrd<F> {
  fn fmt(&self, f: &mut Formatter) -> FmtResult {
    (self.0).fmt(f)
  }
}

impl<F: Copy> AsRef<F> for TotalOrd<F> {
  fn as_ref(&self) -> &F {
    &self.0
  }
}

impl<F: Copy> Borrow<F> for TotalOrd<F> {
  fn borrow(&self) -> &F {
    &self.0
  }
}

impl<'a, F: Copy> Borrow<F> for &'a TotalOrd<F> {
  fn borrow(&self) -> &F {
    &self.0
  }
}

impl From<f16> for TotalOrd<f16> {
  fn from(x: f16) -> TotalOrd<f16> {
    TotalOrd(x)
  }
}

impl PartialEq for TotalOrd<f16> {
  fn eq(&self, other: &TotalOrd<f16>) -> bool {
    match self.cmp(other) {
      Ordering::Equal => true,
      _ => false
    }
  }
}

impl Eq for TotalOrd<f16> {}

impl PartialOrd for TotalOrd<f16> {
  fn partial_cmp(&self, other: &TotalOrd<f16>) -> Option<Ordering> {
    Some(self.cmp(other))
  }
}

impl Ord for TotalOrd<f16> {
  fn cmp(&self, other: &TotalOrd<f16>) -> Ordering {
    self.to_signed_bits().cmp(&other.to_signed_bits())
  }
}

impl Hash for TotalOrd<f16> {
  fn hash<H: Hasher>(&self, hasher: &mut H) {
    self.to_bits().hash(hasher);
  }
}

impl TotalOrd<f16> {
  #[inline]
  pub fn to_bits(&self) -> u16 {
    // NB: This should be the same 1-to-1 mapping as done by `total_cmp`
    // in libcore.
    let mut bits = (self.0).to_bits();
    bits ^= ((((bits as i16) >> 15) as u16) >> 1);
    bits
  }

  #[inline]
  pub fn to_signed_bits(&self) -> i16 {
    self.to_bits() as i16
  }
}

impl From<f32> for TotalOrd<f32> {
  fn from(x: f32) -> TotalOrd<f32> {
    TotalOrd(x)
  }
}

impl PartialEq for TotalOrd<f32> {
  fn eq(&self, other: &TotalOrd<f32>) -> bool {
    match self.cmp(other) {
      Ordering::Equal => true,
      _ => false
    }
  }
}

impl Eq for TotalOrd<f32> {}

impl PartialOrd for TotalOrd<f32> {
  fn partial_cmp(&self, other: &TotalOrd<f32>) -> Option<Ordering> {
    Some(self.cmp(other))
  }
}

impl Ord for TotalOrd<f32> {
  fn cmp(&self, other: &TotalOrd<f32>) -> Ordering {
    self.to_signed_bits().cmp(&other.to_signed_bits())
  }
}

impl Hash for TotalOrd<f32> {
  fn hash<H: Hasher>(&self, hasher: &mut H) {
    self.to_bits().hash(hasher);
  }
}

impl TotalOrd<f32> {
  #[inline]
  pub fn to_bits(&self) -> u32 {
    // NB: This should be the same 1-to-1 mapping as done by `total_cmp`
    // in libcore.
    let mut bits = (self.0).to_bits();
    bits ^= ((((bits as i32) >> 31) as u32) >> 1);
    bits
  }

  #[inline]
  pub fn to_signed_bits(&self) -> i32 {
    self.to_bits() as i32
  }
}

impl From<f64> for TotalOrd<f64> {
  fn from(x: f64) -> TotalOrd<f64> {
    TotalOrd(x)
  }
}

impl PartialEq for TotalOrd<f64> {
  fn eq(&self, other: &TotalOrd<f64>) -> bool {
    match self.cmp(other) {
      Ordering::Equal => true,
      _ => false
    }
  }
}

impl Eq for TotalOrd<f64> {}

impl PartialOrd for TotalOrd<f64> {
  fn partial_cmp(&self, other: &TotalOrd<f64>) -> Option<Ordering> {
    Some(self.cmp(other))
  }
}

impl Ord for TotalOrd<f64> {
  fn cmp(&self, other: &TotalOrd<f64>) -> Ordering {
    self.to_signed_bits().cmp(&other.to_signed_bits())
  }
}

impl Hash for TotalOrd<f64> {
  fn hash<H: Hasher>(&self, hasher: &mut H) {
    self.to_bits().hash(hasher);
  }
}

impl TotalOrd<f64> {
  #[inline]
  pub fn to_bits(&self) -> u64 {
    // NB: This should be the same 1-to-1 mapping as done by `total_cmp`
    // in libcore.
    let mut bits = (self.0).to_bits();
    bits ^= ((((bits as i64) >> 63) as u64) >> 1);
    bits
  }

  #[inline]
  pub fn to_signed_bits(&self) -> i64 {
    self.to_bits() as i64
  }
}

/*#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Nan<F: Copy>(F);

impl<F: Copy + Debug> Debug for Nan<F> {
  fn fmt(&self, f: &mut Formatter) -> FmtResult {
    (self.0).fmt(f)
  }
}

impl<F: Copy> AsRef<F> for Nan<F> {
  fn as_ref(&self) -> &F {
    &self.0
  }
}

#[derive(Clone, Copy, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct NonNan<F: Copy>(F);

impl<F: Copy + Debug> Debug for NonNan<F> {
  fn fmt(&self, f: &mut Formatter) -> FmtResult {
    (self.0).fmt(f)
  }
}

impl<F: Copy> AsRef<F> for NonNan<F> {
  fn as_ref(&self) -> &F {
    &self.0
  }
}

impl TryFrom<f32> for NonNan<f32> {
  type Error = Nan<f32>;

  fn try_from(x: f32) -> Result<NonNan<f32>, Nan<f32>> {
    if x.is_nan() {
      return Err(Nan(x));
    }
    Ok(NonNan(x))
  }
}

impl Eq for NonNan<f32> {}

impl Ord for NonNan<f32> {
  fn cmp(&self, other: &NonNan<f32>) -> Ordering {
    match self.partial_cmp(other) {
      None => panic!("bug"),
      Some(o) => o
    }
  }
}

/*impl Hash for NonNan<f32> {
  fn hash<H: Hasher>(&self, hasher: &mut H) {
    (self.0).to_bits().hash(hasher)
  }
}*/*/

pub trait FpConstExt: Sized {
  fn zero()     -> Self;
  fn neg_zero() -> Self;
  fn one()      -> Self;
  fn neg_one()  -> Self;
  fn inf()      -> Self;
  fn neg_inf()  -> Self;
  fn some_nan() -> Self;
}

impl FpConstExt for f16 {
  fn zero()     -> f16 { f16::from_bits(0) }
  fn neg_zero() -> f16 { f16::from_bits(0x8000) }
  fn one()      -> f16 { f16::from_bits(0x3c00) }
  fn neg_one()  -> f16 { f16::from_bits(0xbc00) }
  fn inf()      -> f16 { f16::INFINITY }
  fn neg_inf()  -> f16 { f16::NEG_INFINITY }
  fn some_nan() -> f16 { f16::NAN }
}

impl FpConstExt for f32 {
  fn zero()     -> f32 { 0.0_f32 }
  fn neg_zero() -> f32 { -0.0_f32 }
  fn one()      -> f32 { 1.0_f32 }
  fn neg_one()  -> f32 { -1.0_f32 }
  fn inf()      -> f32 { f32::INFINITY }
  fn neg_inf()  -> f32 { f32::NEG_INFINITY }
  fn some_nan() -> f32 { f32::NAN }
}

impl FpConstExt for f64 {
  fn zero()     -> f64 { 0.0_f64 }
  fn neg_zero() -> f64 { -0.0_f64 }
  fn one()      -> f64 { 1.0_f64 }
  fn neg_one()  -> f64 { -1.0_f64 }
  fn inf()      -> f64 { f64::INFINITY }
  fn neg_inf()  -> f64 { f64::NEG_INFINITY }
  fn some_nan() -> f64 { f64::NAN }
}
