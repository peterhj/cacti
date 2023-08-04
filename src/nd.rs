use crate::cell::{CellPtr, CellViewHandle};
use crate::ctx::{TL_CTX, Ctx};
use crate::panick::{panick_wrap};

use std::ops::{Index, IndexMut, Range, RangeTo, RangeFrom, RangeFull};

pub type IRange = Range<i64>;
pub type IRangeTo = RangeTo<i64>;
pub type IRangeFrom = RangeFrom<i64>;
pub type IRangeFull = RangeFull;

/*#[derive(Clone, Copy, Debug)]
pub enum IRange_ {
  Full,
  //From{start: i64},
  //To{end: i64},
  Range(Range<i64>),
}*/

pub trait I_ {
  fn _convert(self, len: i64) -> IRange where Self: Sized;
}

impl I_ for IRange {
  fn _convert(self, _len: i64) -> IRange {
    self
  }
}

impl I_ for IRangeTo {
  fn _convert(self: IRangeTo, _len: i64) -> IRange {
    IRange{start: 0, end: self.end}
  }
}

impl I_ for IRangeFrom {
  fn _convert(self: IRangeFrom, len: i64) -> IRange {
    IRange{start: self.start, end: len}
  }
}

impl I_ for IRangeFull {
  fn _convert(self: RangeFull, len: i64) -> IRange {
    IRange{start: 0, end: len}
  }
}

pub trait Ix_<Idx, Ret> {
  fn _convert_index(&self, og: CellPtr, idx: Idx) -> Ret;
}

pub trait I2_ {
  fn _convert2(self, len: i64) -> (IRange, bool) where Self: Sized;
}

impl I2_ for i64 {
  fn _convert2(self, _len: i64) -> (IRange, bool) {
    (IRange{start: self, end: self + 1}, true)
  }
}

impl I2_ for IRange {
  fn _convert2(self, _len: i64) -> (IRange, bool) {
    (self, false)
  }
}

impl I2_ for IRangeTo {
  fn _convert2(self: IRangeTo, _len: i64) -> (IRange, bool) {
    (IRange{start: 0, end: self.end}, false)
  }
}

impl I2_ for IRangeFrom {
  fn _convert2(self: IRangeFrom, len: i64) -> (IRange, bool) {
    (IRange{start: self.start, end: len}, false)
  }
}

impl I2_ for IRangeFull {
  fn _convert2(self: RangeFull, len: i64) -> (IRange, bool) {
    (IRange{start: 0, end: len}, false)
  }
}

pub trait Ix2_<Idx, Ret, Mask> {
  fn _convert_index2(&self, og: CellPtr, idx: Idx) -> (Ret, Mask);
}

macro_rules! convert_index {
  (1, $idx:tt) => {
    impl Ix_<$idx, [IRange; 1]> for Ctx {
      #[track_caller]
      fn _convert_index(&self, og: CellPtr, idx: $idx) -> [IRange; 1] {
        panick_wrap(|| {
          let env = self.env.borrow();
          match env._lookup_view(og) {
            Err(_) => panic!("bug"),
            Ok(e) => {
              if e.ty.shape.len() != 1 {
                println!("ERROR: Ctx::_convert_index: shape is {}-d but index is {}-d",
                    e.ty.shape.len(), 1);
                panic!();
              }
              [I_::_convert(idx, e.ty.shape[0])]
            }
          }
        })
      }
    }
  };
  (2, $idx:tt) => {
    impl Ix_<$idx, [IRange; 2]> for Ctx {
      #[track_caller]
      fn _convert_index(&self, og: CellPtr, idx: $idx) -> [IRange; 2] {
        panick_wrap(|| {
          let env = self.env.borrow();
          match env._lookup_view(og) {
            Err(_) => panic!("bug"),
            Ok(e) => {
              if e.ty.shape.len() != 2 {
                println!("ERROR: Ctx::_convert_index: shape is {}-d but index is {}-d",
                    e.ty.shape.len(), 2);
                panic!();
              }
              [I_::_convert(idx.0, e.ty.shape[0])
              ,I_::_convert(idx.1, e.ty.shape[1])
              ]
            }
          }
        })
      }
    }
  };
  // TODO
}

convert_index!(1, IRange);
convert_index!(1, IRangeTo);
convert_index!(1, IRangeFrom);
convert_index!(1, IRangeFull);

convert_index!(2, (IRange, IRange));
convert_index!(2, (IRange, IRangeTo));
convert_index!(2, (IRange, IRangeFrom));
convert_index!(2, (IRange, IRangeFull));
convert_index!(2, (IRangeTo, IRange));
convert_index!(2, (IRangeTo, IRangeTo));
convert_index!(2, (IRangeTo, IRangeFrom));
convert_index!(2, (IRangeTo, IRangeFull));
convert_index!(2, (IRangeFrom, IRange));
convert_index!(2, (IRangeFrom, IRangeTo));
convert_index!(2, (IRangeFrom, IRangeFrom));
convert_index!(2, (IRangeFrom, IRangeFull));
convert_index!(2, (IRangeFull, IRange));
convert_index!(2, (IRangeFull, IRangeTo));
convert_index!(2, (IRangeFull, IRangeFrom));
convert_index!(2, (IRangeFull, IRangeFull));

// TODO

macro_rules! index_full {
  ($idxty:tt) => {
    impl Index<$idxty> for CellPtr {
      type Output = CellPtr;

      #[track_caller]
      fn index(&self, _: $idxty) -> &CellPtr {
        self
      }
    }

    impl IndexMut<$idxty> for CellPtr {
      #[track_caller]
      fn index_mut(&mut self, _: $idxty) -> &mut CellPtr {
        self
      }
    }
  };
}

macro_rules! index {
  ($idxty:tt) => {
    impl Index<$idxty> for CellPtr {
      type Output = CellViewHandle;

      #[track_caller]
      fn index(&self, idx: $idxty) -> &CellViewHandle {
        let this = *self;
        let view = panick_wrap(|| TL_CTX.with(|ctx| {
          let idx = ctx._convert_index(this, idx);
          ctx.alias_view_slice(this, &idx as &[_])
        }));
        CellViewHandle::_from2(self, view)
      }
    }

    impl IndexMut<$idxty> for CellPtr {
      #[track_caller]
      fn index_mut(&mut self, idx: $idxty) -> &mut CellViewHandle {
        let this = *self;
        let view = panick_wrap(|| TL_CTX.with(|ctx| {
          let idx = ctx._convert_index(this, idx);
          ctx.alias_view_slice(this, &idx as &[_])
        }));
        CellViewHandle::_from2_mut(self, view)
      }
    }
  };
}

index!(IRange);
index!(IRangeTo);
index!(IRangeFrom);
index_full!(IRangeFull);

index!((IRange, IRange));
index!((IRange, IRangeTo));
index!((IRange, IRangeFrom));
index!((IRange, IRangeFull));
index!((IRangeTo, IRange));
index!((IRangeTo, IRangeTo));
index!((IRangeTo, IRangeFrom));
index!((IRangeTo, IRangeFull));
index!((IRangeFrom, IRange));
index!((IRangeFrom, IRangeTo));
index!((IRangeFrom, IRangeFrom));
index!((IRangeFrom, IRangeFull));
index!((IRangeFull, IRange));
index!((IRangeFull, IRangeTo));
index!((IRangeFull, IRangeFrom));
index_full!((IRangeFull, IRangeFull));

// TODO

index_full!((IRangeFull, IRangeFull, IRangeFull));
index_full!((IRangeFull, IRangeFull, IRangeFull, IRangeFull));
