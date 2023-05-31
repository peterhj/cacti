pub mod fp;
//pub mod int;
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

  pub fn first(&self) -> Option<&T> {
    if self.buf.len() <= self.front {
      return None;
    }
    Some(&self.buf[self.front])
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

#[derive(Clone, Copy)]
pub struct Extent {
  pub offset:   usize,
  pub mask:     usize,
}

impl Extent {
  pub fn new_alloc(offset: usize, size: usize) -> Extent {
    Extent{
      offset,
      mask: size,
    }
  }

  pub fn size_bytes(&self) -> usize {
    self.mask & (isize::max_value() as usize)
  }

  pub fn free(&self) -> bool {
    (self.mask & (isize::min_value() as usize)) != 0
  }

  pub fn set_free(&mut self, val: bool) {
    if val {
      self.mask |= (isize::min_value() as usize);
    } else {
      self.mask &= (isize::max_value() as usize);
    }
  }

  pub fn merge_free_unchecked(&self, rhs: Extent) -> Extent {
    assert_eq!(self.offset + self.size_bytes(), rhs.offset);
    let mut e = Extent{
      offset: self.offset,
      mask:   self.size_bytes() + rhs.size_bytes(),
    };
    e.set_free(true);
    e
  }
}

#[derive(Clone, Default)]
pub struct ExtentVecList {
  pub list: MergeVecDeque<Extent>,
}

impl ExtentVecList {
  pub fn fresh_alloc(&mut self, size: usize) -> (usize, Extent) {
    let offset = self.list.last().map(|e| e.offset + e.size_bytes()).unwrap_or(0);
    let e = Extent::new_alloc(offset, size);
    let key = self.list.buf.len();
    self.list.push_back(e);
    (key, e)
  }

  pub fn mark_free(&mut self, key: usize) {
    self.list.buf[key].set_free(true);
  }

  pub fn merge_free(&mut self) {
    if self.list.first().map(|e| !e.free()).unwrap_or(true) {
      return;
    }
    let mut e0 = self.list.pop_front().unwrap();
    loop {
      if self.list.first().map(|e| !e.free()).unwrap_or(true) {
        break;
      }
      let e = self.list.pop_front().unwrap();
      e0 = e0.merge_free_unchecked(e);
    }
    self.list.push_front(e0);
  }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Region {
  pub off:  usize,
  pub sz:   usize,
}

/*impl PartialOrd for Region {
  fn partial_cmp(&self, rhs: &Region) -> Option<Ordering> {
    Some(self.cmp(rhs))
  }
}

impl Ord for Region {
  fn cmp(&self, rhs: &Region) -> Ordering {
    let ret = self.off.cmp(&rhs.off);
    if ret == Ordering::Equal {
      assert_eq!(self.sz, rhs.sz);
    }
    ret
  }
}*/

impl Region {
  pub fn merge(&self, rhs: &Region) -> Region {
    assert_eq!(self.off + self.sz, rhs.off);
    Region{
      off:  self.off,
      sz:   self.sz + rhs.sz,
    }
  }
}
