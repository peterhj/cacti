pub mod fp;
pub mod sync;

#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Bitmask64 {
  bits: u64,
}

impl Default for Bitmask64 {
  fn default() -> Bitmask64 {
    Bitmask64::new()
  }
}

impl Bitmask64 {
  pub fn new() -> Bitmask64 {
    Bitmask64{
      bits: 0,
    }
  }

  pub fn cardinality(&self) -> usize {
    unimplemented!();
  }

  pub fn clear(&mut self) {
    self.bits = 0;
  }

  pub fn insert(&mut self, p: usize) -> bool {
    let (w, o) = (p / 64, p % 64);
    if w >= 1 {
      panic!("bug");
    }
    let prev_bit = ((self.bits >> o) & 1) != 0;
    self.bits |= (1 << o);
    prev_bit
  }

  pub fn remove(&mut self, p: usize) -> bool {
    let (w, o) = (p / 64, p % 64);
    if w >= 1 {
      panic!("bug");
    }
    let prev_bit = ((self.bits >> o) & 1) != 0;
    self.bits &= !(1 << o);
    prev_bit
  }
}

#[derive(Clone)]
pub struct Bitvec64 {
  bits: Vec<u64>,
  card: usize,
}

impl Default for Bitvec64 {
  fn default() -> Bitvec64 {
    Bitvec64::new()
  }
}

impl Bitvec64 {
  pub fn new() -> Bitvec64 {
    Bitvec64{
      bits: Vec::new(),
      card: 0,
    }
  }

  pub fn cardinality(&self) -> usize {
    self.card
  }

  pub fn clear(&mut self) {
    self.bits.clear();
    self.card = 0;
  }

  pub fn insert(&mut self, p: usize) -> bool {
    let (w, o) = (p / 64, p % 64);
    if w >= self.bits.len() {
      self.bits.resize(w + 1, 0);
    }
    let prev_bit = ((self.bits[w] >> o) & 1) != 0;
    self.bits[w] |= (1 << o);
    prev_bit
  }

  pub fn remove(&mut self, p: usize) -> bool {
    let (w, o) = (p / 64, p % 64);
    if w >= self.bits.len() {
      panic!("bug");
    }
    let prev_bit = ((self.bits[w] >> o) & 1) != 0;
    self.bits[w] &= !(1 << o);
    prev_bit
  }
}
