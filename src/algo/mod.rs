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

  #[allow(unused_parens)]
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

  #[allow(unused_parens)]
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

#[derive(Clone)]
pub struct MergeVecDeque<T> {
  buf:      Vec<T>,
  front:    usize,
}

impl<T> Default for MergeVecDeque<T> {
  fn default() -> MergeVecDeque<T> {
    MergeVecDeque::new()
  }
}

impl<T> MergeVecDeque<T> {
  pub fn new() -> MergeVecDeque<T> {
    MergeVecDeque{
      buf:      Vec::new(),
      front:    0,
    }
  }

  pub fn len(&self) -> usize {
    self.buf.len() - self.front
  }

  pub fn front_offset(&self) -> usize {
    self.front
  }

  pub fn last(&self) -> Option<&T> {
    if self.buf.len() <= self.front {
      return None;
    }
    self.buf.last()
  }

  pub fn at_offset_unchecked(&self, off: usize) -> &T {
    &self.buf[off]
  }

  pub fn compact(&mut self) {
    if self.front <= 0 {
      return;
    }
    let len = self.len();
    let front = self.front;
    for i in 0 .. len {
      self.buf.swap(i, front + i);
    }
    self.buf.resize_with(len, || unreachable!());
    self.front = 0;
  }

  pub fn push_front(&mut self, val: T) {
    if self.front <= 0 {
      panic!("bug");
    }
    self.front -= 1;
    self.buf[self.front] = val;
  }

  pub fn push_back(&mut self, val: T) {
    self.buf.push(val);
  }

  pub fn pop_back(&mut self) -> Option<T> {
    if self.buf.len() <= self.front {
      return None;
    }
    self.buf.pop()
  }
}

impl<T: Clone> MergeVecDeque<T> {
  pub fn pop_front(&mut self) -> Option<T> {
    if self.buf.len() <= self.front {
      return None;
    }
    let val = self.buf[self.front].clone();
    self.front += 1;
    Some(val)
  }
}
