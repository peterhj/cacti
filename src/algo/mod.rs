pub use fxhash::{FxHashMap as HashMap, FxHashSet as HashSet};

use std::cell::{Cell};
//pub use std::collections::{HashMap, HashSet};
pub use std::collections::{BTreeMap, BTreeSet};
use std::cmp::{Ordering};
use std::fmt::{Debug};
use std::mem::{swap};

pub mod fp;
pub mod hash;
pub mod int;
//pub mod nd;
pub mod str;
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

/*#[derive(Clone)]
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
}*/

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Region {
  pub off:  usize,
  pub sz:   usize,
}

impl PartialOrd for Region {
  fn partial_cmp(&self, rhs: &Region) -> Option<Ordering> {
    Some(self.cmp(rhs))
  }
}

impl Ord for Region {
  fn cmp(&self, rhs: &Region) -> Ordering {
    let ret = self.off.cmp(&rhs.off);
    if ret == Ordering::Equal {
      return self.sz.cmp(&rhs.sz);
    }
    ret
  }
}

impl Region {
  pub fn merge(&self, rhs: Region) -> Region {
    assert_eq!(self.off + self.sz, rhs.off);
    Region{
      off:  self.off,
      sz:   self.sz + rhs.sz,
    }
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(transparent)]
pub struct RevOrd<T>(pub T);

impl<'a, T: Copy> From<&'a RevOrd<T>> for RevOrd<T> {
  fn from(this: &'a RevOrd<T>) -> RevOrd<T> {
    *this
  }
}

impl<T> From<T> for RevOrd<T> {
  fn from(inner: T) -> RevOrd<T> {
    RevOrd(inner)
  }
}

impl<T> AsRef<T> for RevOrd<T> {
  fn as_ref(&self) -> &T {
    &(self.0)
  }
}

impl<T: PartialOrd> PartialOrd for RevOrd<T> {
  fn partial_cmp(&self, rhs: &RevOrd<T>) -> Option<Ordering> {
    match (&(self.0)).partial_cmp(&(rhs.0)) {
      Some(Ordering::Greater) => Some(Ordering::Less),
      Some(Ordering::Less) => Some(Ordering::Greater),
      Some(Ordering::Equal) => Some(Ordering::Equal),
      None => None,
    }
  }
}

impl<T: Ord> Ord for RevOrd<T> {
  fn cmp(&self, rhs: &RevOrd<T>) -> Ordering {
    match (self.0).cmp(&(rhs.0)) {
      Ordering::Greater => Ordering::Less,
      Ordering::Less => Ordering::Greater,
      Ordering::Equal => Ordering::Equal
    }
  }
}

pub type RevSortKey8<K> = SortKey8<RevOrd<K>>;
pub type RevSortMap8<K, V> = SortMap8<RevOrd<K>, V>;

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct SortKey8<K> {
  pub key: K,
  pub probe: Cell<u8>,
}

impl<K> AsRef<SortKey8<K>> for SortKey8<K> {
  fn as_ref(&self) -> &SortKey8<K> {
    self
  }
}

impl<K> AsRef<K> for SortKey8<K> {
  fn as_ref(&self) -> &K {
    &self.key
  }
}

#[derive(Clone, Debug)]
pub struct SortMap8<K, V> {
  pub buf:  Vec<(K, V)>,
}

impl<K: Copy, V> Default for SortMap8<K, V> {
  fn default() -> SortMap8<K, V> {
    SortMap8::new()
  }
}

impl<K: Copy, V> SortMap8<K, V> {
  pub fn new() -> SortMap8<K, V> {
    SortMap8{buf: Vec::new()}
  }

  pub fn clear(&mut self) {
    self.buf.clear();
  }
}

impl<K: Copy + Ord + Debug, V> SortMap8<K, V> {
  pub fn iter(&self) -> impl Iterator<Item=(&K, &V)> {
    self.buf.iter().map(|&(ref k, ref v)| (k, v))
  }

  pub fn probe(&self, query: &SortKey8<K>) -> u8 {
    let mut p = query.probe.get();
    let len = self.buf.len();
    if (p as usize) < len &&
       &query.key == &self.buf[p as usize].0
    {
      return p;
    }
    let mut q = p;
    p += 1;
    loop {
      let mut t = false;
      if (p as usize) < len {
        t = true;
        if &query.key == &self.buf[p as usize].0 {
          query.probe.set(p);
          return p;
        }
        p += 1;
      }
      if q > 0 {
        t = true;
        q -= 1;
        if &query.key == &self.buf[q as usize].0 {
          query.probe.set(q);
          return q;
        }
      }
      if !t {
        panic!("bug: SortMap8::probe: not found: key={:?}", &query.key);
      }
    }
  }

  pub fn find<K2: Into<K>>(&self, key: K2) -> Option<(SortKey8<K>, &V)> {
    let key = key.into();
    for (i, (k, v)) in self.buf.iter().enumerate() {
      if &key == k {
        assert!(i <= 255);
        return Some((SortKey8{key: *k, probe: Cell::new(i as u8)}, v));
      }
    }
    None
  }

  pub fn find_mut<K2: Into<K>>(&mut self, key: K2) -> Option<(SortKey8<K>, &mut V)> {
    let key = key.into();
    for (i, (k, v)) in self.buf.iter_mut().enumerate() {
      if &key == k {
        assert!(i <= 255);
        return Some((SortKey8{key: *k, probe: Cell::new(i as u8)}, v));
      }
    }
    None
  }

  pub fn find_lub<K2: Into<K>>(&self, key: K2) -> Option<(SortKey8<K>, &V)> {
    let key = key.into();
    for (i, (k, v)) in self.buf.iter().enumerate() {
      if &key <= k {
        assert!(i <= 255);
        return Some((SortKey8{key: *k, probe: Cell::new(i as u8)}, v));
      }
    }
    None
  }

  pub fn find_lub_mut<K2: Into<K>>(&mut self, key: K2) -> Option<(SortKey8<K>, &mut V)> {
    let key = key.into();
    for (i, (k, v)) in self.buf.iter_mut().enumerate() {
      if &key <= k {
        assert!(i <= 255);
        return Some((SortKey8{key: *k, probe: Cell::new(i as u8)}, v));
      }
    }
    None
  }

  pub fn get<Q: AsRef<SortKey8<K>>>(&self, query: Q) -> &V {
    let p = self.probe(query.as_ref());
    &(self.buf[p as usize].1)
  }

  pub fn get_mut<Q: AsRef<SortKey8<K>>>(&mut self, query: Q) -> &mut V {
    let p = self.probe(query.as_ref());
    &mut (self.buf[p as usize].1)
  }

  pub fn swap<Q: AsRef<SortKey8<K>>>(&mut self, query: Q, val: &mut V) {
    let p = self.probe(query.as_ref());
    swap(&mut (self.buf[p as usize].1), val);
  }

  /*pub fn remove<Q: AsRef<SortKey8<K>>>(&mut self, query: Q) -> (K, V) {
    let p = self.probe(query.as_ref()) as usize;
    for i in p + 1 .. self.buf.len() {
      self.buf.swap(i - 1, i);
    }
    self.buf.pop().unwrap()
  }*/

  pub fn remove<K2: Into<K>>(&mut self, query: K2) -> (K, V) {
    let key = query.into();
    //let p = self.probe(query.as_ref()) as usize;
    let mut p = None;
    for i in 0 .. self.buf.len() {
      if &key == &self.buf[i].0 {
        p = Some(i);
        break;
      }
    }
    if p.is_none() {
      panic!("bug: SortMap8::remove: not found: key={:?}", &key);
    }
    let p = p.unwrap();
    for i in p + 1 .. self.buf.len() {
      self.buf.swap(i - 1, i);
    }
    self.buf.pop().unwrap()
  }

  pub fn insert<K2: Into<K>>(&mut self, key: K2, val: V) -> SortKey8<K> {
    let olen = self.buf.len();
    if olen > 255 {
      panic!("bug: SortMap8::insert: overcapacity: len={}", olen);
    }
    let key = key.into();
    let mut p = olen as u8;
    self.buf.push((key, val));
    loop {
      if p > 0 {
        let q = p - 1;
        match key.cmp(&self.buf[q as usize].0) {
          Ordering::Equal => {
            panic!("bug: SortMap8::insert: duplicate: key={:?}", &key);
          }
          Ordering::Less => {
            self.buf.swap(p as usize, q as usize);
            p = q;
          }
          Ordering::Greater => {
            return SortKey8{key, probe: Cell::new(p)};
          }
        }
        continue;
      }
      return SortKey8{key, probe: Cell::new(p)};
    }
  }
}

pub trait StdCellExt<T> {
  fn fetch_add(&self, val: T) -> T;
  fn fetch_sub(&self, val: T) -> T;
}

impl StdCellExt<isize> for Cell<isize> {
  fn fetch_add(&self, val: isize) -> isize {
    let old_val = self.get();
    let new_val = old_val + val;
    self.set(new_val);
    old_val
  }

  fn fetch_sub(&self, val: isize) -> isize {
    let old_val = self.get();
    let new_val = old_val - val;
    self.set(new_val);
    old_val
  }
}

impl StdCellExt<i64> for Cell<i64> {
  fn fetch_add(&self, val: i64) -> i64 {
    let old_val = self.get();
    let new_val = old_val + val;
    self.set(new_val);
    old_val
  }

  fn fetch_sub(&self, val: i64) -> i64 {
    let old_val = self.get();
    let new_val = old_val - val;
    self.set(new_val);
    old_val
  }
}

impl StdCellExt<i32> for Cell<i32> {
  fn fetch_add(&self, val: i32) -> i32 {
    let old_val = self.get();
    let new_val = old_val + val;
    self.set(new_val);
    old_val
  }

  fn fetch_sub(&self, val: i32) -> i32 {
    let old_val = self.get();
    let new_val = old_val - val;
    self.set(new_val);
    old_val
  }
}

impl StdCellExt<i16> for Cell<i16> {
  fn fetch_add(&self, val: i16) -> i16 {
    let old_val = self.get();
    let new_val = old_val + val;
    self.set(new_val);
    old_val
  }

  fn fetch_sub(&self, val: i16) -> i16 {
    let old_val = self.get();
    let new_val = old_val - val;
    self.set(new_val);
    old_val
  }
}

impl StdCellExt<i8> for Cell<i8> {
  fn fetch_add(&self, val: i8) -> i8 {
    let old_val = self.get();
    let new_val = old_val + val;
    self.set(new_val);
    old_val
  }

  fn fetch_sub(&self, val: i8) -> i8 {
    let old_val = self.get();
    let new_val = old_val - val;
    self.set(new_val);
    old_val
  }
}

impl StdCellExt<usize> for Cell<usize> {
  fn fetch_add(&self, val: usize) -> usize {
    let old_val = self.get();
    let new_val = old_val + val;
    self.set(new_val);
    old_val
  }

  fn fetch_sub(&self, val: usize) -> usize {
    let old_val = self.get();
    let new_val = old_val - val;
    self.set(new_val);
    old_val
  }
}

impl StdCellExt<u64> for Cell<u64> {
  fn fetch_add(&self, val: u64) -> u64 {
    let old_val = self.get();
    let new_val = old_val + val;
    self.set(new_val);
    old_val
  }

  fn fetch_sub(&self, val: u64) -> u64 {
    let old_val = self.get();
    let new_val = old_val - val;
    self.set(new_val);
    old_val
  }
}

impl StdCellExt<u32> for Cell<u32> {
  fn fetch_add(&self, val: u32) -> u32 {
    let old_val = self.get();
    let new_val = old_val + val;
    self.set(new_val);
    old_val
  }

  fn fetch_sub(&self, val: u32) -> u32 {
    let old_val = self.get();
    let new_val = old_val - val;
    self.set(new_val);
    old_val
  }
}

impl StdCellExt<u16> for Cell<u16> {
  fn fetch_add(&self, val: u16) -> u16 {
    let old_val = self.get();
    let new_val = old_val + val;
    self.set(new_val);
    old_val
  }

  fn fetch_sub(&self, val: u16) -> u16 {
    let old_val = self.get();
    let new_val = old_val - val;
    self.set(new_val);
    old_val
  }
}

impl StdCellExt<u8> for Cell<u8> {
  fn fetch_add(&self, val: u8) -> u8 {
    let old_val = self.get();
    let new_val = old_val + val;
    self.set(new_val);
    old_val
  }

  fn fetch_sub(&self, val: u8) -> u8 {
    let old_val = self.get();
    let new_val = old_val - val;
    self.set(new_val);
    old_val
  }
}
