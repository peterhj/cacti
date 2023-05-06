use std::cmp::{Ordering};
use std::convert::{TryFrom};
use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::hash::{Hash, Hasher};

#[derive(Clone, Copy, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct NonNan<F>(F);

impl TryFrom<f32> for NonNan<f32> {
  type Error = ();

  fn try_from(x: f32) -> Result<NonNan<f32>, ()> {
    if x.is_nan() {
      return Err(());
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

impl Hash for NonNan<f32> {
  fn hash<H: Hasher>(&self, hasher: &mut H) {
    (self.0).to_bits().hash(hasher)
  }
}

impl<F: Debug> Debug for NonNan<F> {
  fn fmt(&self, f: &mut Formatter) -> FmtResult {
    (self.0).fmt(f)
  }
}
