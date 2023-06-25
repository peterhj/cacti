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
