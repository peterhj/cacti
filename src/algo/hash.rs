use std::hash::{Hasher};

#[repr(transparent)]
pub struct Hasher_<'a> {
  pub inner: &'a mut dyn Hasher,
}

impl<'a> Hasher for Hasher_<'a> {
  fn finish(&self) -> u64 {
    self.inner.finish()
  }

  fn write(&mut self, bytes: &[u8]) {
    self.inner.write(bytes);
  }
}
