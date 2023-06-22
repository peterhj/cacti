pub use half::{f16, bf16};

use std::cmp::{Ordering};
use std::convert::{TryFrom};
use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::hash::{Hash, Hasher};

/* `TotalOrd` is derived from the implementation in libcore:

Copyright (c) 2014 The Rust Project Developers

Permission is hereby granted, free of charge, to any
person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the
Software without restriction, including without
limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software
is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice
shall be included in all copies or substantial portions
of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF
ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE. */

#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct TotalOrd<F: Copy>(F);

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

#[derive(Clone, Copy)]
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
}*/
