use crate::cell::{CellPtr, StableCell, CellDeref, CellViewHandle};
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
  #[inline]
  fn _convert(self, _len: i64) -> IRange {
    self
  }
}

impl I_ for IRangeTo {
  #[inline]
  fn _convert(self: IRangeTo, _len: i64) -> IRange {
    IRange{start: 0, end: self.end}
  }
}

impl I_ for IRangeFrom {
  #[inline]
  fn _convert(self: IRangeFrom, len: i64) -> IRange {
    IRange{start: self.start, end: len}
  }
}

impl I_ for IRangeFull {
  #[inline]
  fn _convert(self: RangeFull, len: i64) -> IRange {
    IRange{start: 0, end: len}
  }
}

pub trait Ix_<Idx, Ret> {
  fn _convert_index(&self, og: CellPtr, idx: Idx) -> Ret;
}

pub trait Ip_ {
  fn _convert_proj(self, len: i64) -> (IRange, bool) where Self: Sized;
}

impl Ip_ for i64 {
  #[inline]
  fn _convert_proj(self, _len: i64) -> (IRange, bool) {
    (IRange{start: self, end: self + 1}, true)
  }
}

impl Ip_ for IRange {
  #[inline]
  fn _convert_proj(self, _len: i64) -> (IRange, bool) {
    (self, false)
  }
}

impl Ip_ for IRangeTo {
  #[inline]
  fn _convert_proj(self: IRangeTo, _len: i64) -> (IRange, bool) {
    (IRange{start: 0, end: self.end}, false)
  }
}

impl Ip_ for IRangeFrom {
  #[inline]
  fn _convert_proj(self: IRangeFrom, len: i64) -> (IRange, bool) {
    (IRange{start: self.start, end: len}, false)
  }
}

impl Ip_ for IRangeFull {
  #[inline]
  fn _convert_proj(self: RangeFull, len: i64) -> (IRange, bool) {
    (IRange{start: 0, end: len}, false)
  }
}

pub trait Ixp_<Idx, Ret, Mask> {
  fn _convert_iproj(&self, og: CellPtr, idx: Idx) -> (Ret, Mask);
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
  (3, $idx:tt) => {
    impl Ix_<$idx, [IRange; 3]> for Ctx {
      #[track_caller]
      fn _convert_index(&self, og: CellPtr, idx: $idx) -> [IRange; 3] {
        panick_wrap(|| {
          let env = self.env.borrow();
          match env._lookup_view(og) {
            Err(_) => panic!("bug"),
            Ok(e) => {
              if e.ty.shape.len() != 3 {
                println!("ERROR: Ctx::_convert_index: shape is {}-d but index is {}-d",
                    e.ty.shape.len(), 3);
                panic!();
              }
              [I_::_convert(idx.0, e.ty.shape[0])
              ,I_::_convert(idx.1, e.ty.shape[1])
              ,I_::_convert(idx.2, e.ty.shape[2])
              ]
            }
          }
        })
      }
    }
  };
  (4, $idx:tt) => {
    impl Ix_<$idx, [IRange; 4]> for Ctx {
      #[track_caller]
      fn _convert_index(&self, og: CellPtr, idx: $idx) -> [IRange; 4] {
        panick_wrap(|| {
          let env = self.env.borrow();
          match env._lookup_view(og) {
            Err(_) => panic!("bug"),
            Ok(e) => {
              if e.ty.shape.len() != 4 {
                println!("ERROR: Ctx::_convert_index: shape is {}-d but index is {}-d",
                    e.ty.shape.len(), 4);
                panic!();
              }
              [I_::_convert(idx.0, e.ty.shape[0])
              ,I_::_convert(idx.1, e.ty.shape[1])
              ,I_::_convert(idx.2, e.ty.shape[2])
              ,I_::_convert(idx.3, e.ty.shape[3])
              ]
            }
          }
        })
      }
    }
  };
  // TODO
}

macro_rules! convert_iproj {
  (1, $idx:tt) => {
    impl Ixp_<$idx, [IRange; 1], [bool; 1]> for Ctx {
      #[track_caller]
      fn _convert_iproj(&self, og: CellPtr, idx: $idx) -> ([IRange; 1], [bool; 1]) {
        panick_wrap(|| {
          let env = self.env.borrow();
          match env._lookup_view(og) {
            Err(_) => panic!("bug"),
            Ok(e) => {
              if e.ty.shape.len() != 1 {
                println!("ERROR: Ctx::_convert_iproj: shape is {}-d but index is {}-d",
                    e.ty.shape.len(), 1);
                panic!();
              }
              let [(idx0, mask0)] =
                  [Ip_::_convert_proj(idx, e.ty.shape[0])];
              ([idx0], [mask0])
            }
          }
        })
      }
    }
  };
  (2, $idx:tt) => {
    impl Ixp_<$idx, [IRange; 2], [bool; 2]> for Ctx {
      #[track_caller]
      fn _convert_iproj(&self, og: CellPtr, idx: $idx) -> ([IRange; 2], [bool; 2]) {
        panick_wrap(|| {
          let env = self.env.borrow();
          match env._lookup_view(og) {
            Err(_) => panic!("bug"),
            Ok(e) => {
              if e.ty.shape.len() != 2 {
                println!("ERROR: Ctx::_convert_iproj: shape is {}-d but index is {}-d",
                    e.ty.shape.len(), 2);
                panic!();
              }
              let [(idx0, mask0), (idx1, mask1)] =
                  [Ip_::_convert_proj(idx.0, e.ty.shape[0])
                  ,Ip_::_convert_proj(idx.1, e.ty.shape[1])
                  ];
              ([idx0, idx1], [mask0, mask1])
            }
          }
        })
      }
    }
  };
  (3, $idx:tt) => {
    impl Ixp_<$idx, [IRange; 3], [bool; 3]> for Ctx {
      #[track_caller]
      fn _convert_iproj(&self, og: CellPtr, idx: $idx) -> ([IRange; 3], [bool; 3]) {
        panick_wrap(|| {
          let env = self.env.borrow();
          match env._lookup_view(og) {
            Err(_) => panic!("bug"),
            Ok(e) => {
              if e.ty.shape.len() != 3 {
                println!("ERROR: Ctx::_convert_iproj: shape is {}-d but index is {}-d",
                    e.ty.shape.len(), 3);
                panic!();
              }
              let [(idx0, mask0), (idx1, mask1), (idx2, mask2)] =
                  [Ip_::_convert_proj(idx.0, e.ty.shape[0])
                  ,Ip_::_convert_proj(idx.1, e.ty.shape[1])
                  ,Ip_::_convert_proj(idx.2, e.ty.shape[2])
                  ];
              ([idx0, idx1, idx2], [mask0, mask1, mask2])
            }
          }
        })
      }
    }
  };
  (4, $idx:tt) => {
    impl Ixp_<$idx, [IRange; 4], [bool; 4]> for Ctx {
      #[track_caller]
      fn _convert_iproj(&self, og: CellPtr, idx: $idx) -> ([IRange; 4], [bool; 4]) {
        panick_wrap(|| {
          let env = self.env.borrow();
          match env._lookup_view(og) {
            Err(_) => panic!("bug"),
            Ok(e) => {
              if e.ty.shape.len() != 4 {
                println!("ERROR: Ctx::_convert_iproj: shape is {}-d but index is {}-d",
                    e.ty.shape.len(), 4);
                panic!();
              }
              let [(idx0, mask0), (idx1, mask1), (idx2, mask2), (idx3, mask3)] =
                  [Ip_::_convert_proj(idx.0, e.ty.shape[0])
                  ,Ip_::_convert_proj(idx.1, e.ty.shape[1])
                  ,Ip_::_convert_proj(idx.2, e.ty.shape[2])
                  ,Ip_::_convert_proj(idx.3, e.ty.shape[3])
                  ];
              ([idx0, idx1, idx2, idx3], [mask0, mask1, mask2, mask3])
            }
          }
        })
      }
    }
  };
  // TODO
}

/*convert_index!(1, IRange);
convert_index!(1, IRangeTo);
convert_index!(1, IRangeFrom);
convert_index!(1, IRangeFull);

convert_iproj!(1, i64);

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

convert_iproj!(2, (i64, IRange));
convert_iproj!(2, (i64, IRangeTo));
convert_iproj!(2, (i64, IRangeFrom));
convert_iproj!(2, (i64, IRangeFull));
convert_iproj!(2, (IRange, i64));
convert_iproj!(2, (IRangeTo, i64));
convert_iproj!(2, (IRangeFrom, i64));
convert_iproj!(2, (IRangeFull, i64));*/

// TODO

macro_rules! ifull {
  ($nd:tt, $idxty:tt) => {
    convert_index!($nd, $idxty);

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
  ($nd:tt, $idxty:tt) => {
    convert_index!($nd, $idxty);

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

    impl Index<$idxty> for StableCell {
      type Output = CellViewHandle;

      #[track_caller]
      fn index(&self, idx: $idxty) -> &CellViewHandle {
        let this = self._deref();
        let view = panick_wrap(|| TL_CTX.with(|ctx| {
          let idx = ctx._convert_index(this, idx);
          ctx.alias_view_slice(this, &idx as &[_])
        }));
        CellViewHandle::_from2(self.as_ptr_ref(), view)
      }
    }

    impl IndexMut<$idxty> for StableCell {
      #[track_caller]
      fn index_mut(&mut self, idx: $idxty) -> &mut CellViewHandle {
        let this = self._deref();
        let view = panick_wrap(|| TL_CTX.with(|ctx| {
          let idx = ctx._convert_index(this, idx);
          ctx.alias_view_slice(this, &idx as &[_])
        }));
        CellViewHandle::_from2_mut(self.as_ptr_mut(), view)
      }
    }
  };
}

macro_rules! iproj {
  ($nd:tt, $idxty:tt) => {
    convert_iproj!($nd, $idxty);

    impl Index<$idxty> for CellPtr {
      type Output = CellViewHandle;

      #[track_caller]
      fn index(&self, idx: $idxty) -> &CellViewHandle {
        let this = *self;
        let view = panick_wrap(|| TL_CTX.with(|ctx| {
          let (idx, mask) = ctx._convert_iproj(this, idx);
          let slice = ctx.alias_view_slice(this, &idx as &[_]);
          let proj = ctx.alias_view_proj(slice, &mask as &[_]);
          proj
        }));
        CellViewHandle::_from2(self, view)
      }
    }

    impl IndexMut<$idxty> for CellPtr {
      #[track_caller]
      fn index_mut(&mut self, idx: $idxty) -> &mut CellViewHandle {
        let this = *self;
        let view = panick_wrap(|| TL_CTX.with(|ctx| {
          let (idx, mask) = ctx._convert_iproj(this, idx);
          let slice = ctx.alias_view_slice(this, &idx as &[_]);
          let proj = ctx.alias_view_proj(slice, &mask as &[_]);
          proj
        }));
        CellViewHandle::_from2_mut(self, view)
      }
    }
  };
}

index!(1, IRange);
index!(1, IRangeTo);
index!(1, IRangeFrom);
ifull!(1, IRangeFull);
iproj!(1, i64);

index!(2, (IRange, IRange));
index!(2, (IRange, IRangeTo));
index!(2, (IRange, IRangeFrom));
index!(2, (IRange, IRangeFull));
index!(2, (IRangeTo, IRange));
index!(2, (IRangeTo, IRangeTo));
index!(2, (IRangeTo, IRangeFrom));
index!(2, (IRangeTo, IRangeFull));
index!(2, (IRangeFrom, IRange));
index!(2, (IRangeFrom, IRangeTo));
index!(2, (IRangeFrom, IRangeFrom));
index!(2, (IRangeFrom, IRangeFull));
index!(2, (IRangeFull, IRange));
index!(2, (IRangeFull, IRangeTo));
index!(2, (IRangeFull, IRangeFrom));
ifull!(2, (IRangeFull, IRangeFull));
iproj!(2, (i64, i64));
iproj!(2, (i64, IRange));
iproj!(2, (i64, IRangeTo));
iproj!(2, (i64, IRangeFrom));
iproj!(2, (i64, IRangeFull));
iproj!(2, (IRange, i64));
iproj!(2, (IRangeTo, i64));
iproj!(2, (IRangeFrom, i64));
iproj!(2, (IRangeFull, i64));

index!(3, (IRange, IRange, IRange));
index!(3, (IRange, IRange, IRangeTo));
index!(3, (IRange, IRange, IRangeFrom));
index!(3, (IRange, IRange, IRangeFull));
index!(3, (IRange, IRangeTo, IRange));
index!(3, (IRange, IRangeTo, IRangeTo));
index!(3, (IRange, IRangeTo, IRangeFrom));
index!(3, (IRange, IRangeTo, IRangeFull));
index!(3, (IRange, IRangeFrom, IRange));
index!(3, (IRange, IRangeFrom, IRangeTo));
index!(3, (IRange, IRangeFrom, IRangeFrom));
index!(3, (IRange, IRangeFrom, IRangeFull));
index!(3, (IRange, IRangeFull, IRange));
index!(3, (IRange, IRangeFull, IRangeTo));
index!(3, (IRange, IRangeFull, IRangeFrom));
index!(3, (IRange, IRangeFull, IRangeFull));
index!(3, (IRangeTo, IRange, IRange));
index!(3, (IRangeTo, IRange, IRangeTo));
index!(3, (IRangeTo, IRange, IRangeFrom));
index!(3, (IRangeTo, IRange, IRangeFull));
index!(3, (IRangeTo, IRangeTo, IRange));
index!(3, (IRangeTo, IRangeTo, IRangeTo));
index!(3, (IRangeTo, IRangeTo, IRangeFrom));
index!(3, (IRangeTo, IRangeTo, IRangeFull));
index!(3, (IRangeTo, IRangeFrom, IRange));
index!(3, (IRangeTo, IRangeFrom, IRangeTo));
index!(3, (IRangeTo, IRangeFrom, IRangeFrom));
index!(3, (IRangeTo, IRangeFrom, IRangeFull));
index!(3, (IRangeTo, IRangeFull, IRange));
index!(3, (IRangeTo, IRangeFull, IRangeTo));
index!(3, (IRangeTo, IRangeFull, IRangeFrom));
index!(3, (IRangeTo, IRangeFull, IRangeFull));
index!(3, (IRangeFrom, IRange, IRange));
index!(3, (IRangeFrom, IRange, IRangeTo));
index!(3, (IRangeFrom, IRange, IRangeFrom));
index!(3, (IRangeFrom, IRange, IRangeFull));
index!(3, (IRangeFrom, IRangeTo, IRange));
index!(3, (IRangeFrom, IRangeTo, IRangeTo));
index!(3, (IRangeFrom, IRangeTo, IRangeFrom));
index!(3, (IRangeFrom, IRangeTo, IRangeFull));
index!(3, (IRangeFrom, IRangeFrom, IRange));
index!(3, (IRangeFrom, IRangeFrom, IRangeTo));
index!(3, (IRangeFrom, IRangeFrom, IRangeFrom));
index!(3, (IRangeFrom, IRangeFrom, IRangeFull));
index!(3, (IRangeFrom, IRangeFull, IRange));
index!(3, (IRangeFrom, IRangeFull, IRangeTo));
index!(3, (IRangeFrom, IRangeFull, IRangeFrom));
index!(3, (IRangeFrom, IRangeFull, IRangeFull));
index!(3, (IRangeFull, IRange, IRange));
index!(3, (IRangeFull, IRange, IRangeTo));
index!(3, (IRangeFull, IRange, IRangeFrom));
index!(3, (IRangeFull, IRange, IRangeFull));
index!(3, (IRangeFull, IRangeTo, IRange));
index!(3, (IRangeFull, IRangeTo, IRangeTo));
index!(3, (IRangeFull, IRangeTo, IRangeFrom));
index!(3, (IRangeFull, IRangeTo, IRangeFull));
index!(3, (IRangeFull, IRangeFrom, IRange));
index!(3, (IRangeFull, IRangeFrom, IRangeTo));
index!(3, (IRangeFull, IRangeFrom, IRangeFrom));
index!(3, (IRangeFull, IRangeFrom, IRangeFull));
index!(3, (IRangeFull, IRangeFull, IRange));
index!(3, (IRangeFull, IRangeFull, IRangeTo));
index!(3, (IRangeFull, IRangeFull, IRangeFrom));
ifull!(3, (IRangeFull, IRangeFull, IRangeFull));
iproj!(3, (i64, i64, i64));
iproj!(3, (i64, i64, IRange));
iproj!(3, (i64, i64, IRangeTo));
iproj!(3, (i64, i64, IRangeFrom));
iproj!(3, (i64, i64, IRangeFull));
iproj!(3, (i64, IRange, i64));
iproj!(3, (i64, IRange, IRange));
iproj!(3, (i64, IRange, IRangeTo));
iproj!(3, (i64, IRange, IRangeFrom));
iproj!(3, (i64, IRange, IRangeFull));
iproj!(3, (i64, IRangeTo, i64));
iproj!(3, (i64, IRangeTo, IRange));
iproj!(3, (i64, IRangeTo, IRangeTo));
iproj!(3, (i64, IRangeTo, IRangeFrom));
iproj!(3, (i64, IRangeTo, IRangeFull));
iproj!(3, (i64, IRangeFrom, i64));
iproj!(3, (i64, IRangeFrom, IRange));
iproj!(3, (i64, IRangeFrom, IRangeTo));
iproj!(3, (i64, IRangeFrom, IRangeFrom));
iproj!(3, (i64, IRangeFrom, IRangeFull));
iproj!(3, (i64, IRangeFull, i64));
iproj!(3, (i64, IRangeFull, IRange));
iproj!(3, (i64, IRangeFull, IRangeTo));
iproj!(3, (i64, IRangeFull, IRangeFrom));
iproj!(3, (i64, IRangeFull, IRangeFull));
iproj!(3, (IRange, i64, i64));
iproj!(3, (IRange, i64, IRange));
iproj!(3, (IRange, i64, IRangeTo));
iproj!(3, (IRange, i64, IRangeFrom));
iproj!(3, (IRange, i64, IRangeFull));
iproj!(3, (IRange, IRange, i64));
iproj!(3, (IRange, IRangeTo, i64));
iproj!(3, (IRange, IRangeFrom, i64));
iproj!(3, (IRange, IRangeFull, i64));
iproj!(3, (IRangeTo, i64, i64));
iproj!(3, (IRangeTo, i64, IRange));
iproj!(3, (IRangeTo, i64, IRangeTo));
iproj!(3, (IRangeTo, i64, IRangeFrom));
iproj!(3, (IRangeTo, i64, IRangeFull));
iproj!(3, (IRangeTo, IRange, i64));
iproj!(3, (IRangeTo, IRangeTo, i64));
iproj!(3, (IRangeTo, IRangeFrom, i64));
iproj!(3, (IRangeTo, IRangeFull, i64));
iproj!(3, (IRangeFrom, i64, i64));
iproj!(3, (IRangeFrom, i64, IRange));
iproj!(3, (IRangeFrom, i64, IRangeTo));
iproj!(3, (IRangeFrom, i64, IRangeFrom));
iproj!(3, (IRangeFrom, i64, IRangeFull));
iproj!(3, (IRangeFrom, IRange, i64));
iproj!(3, (IRangeFrom, IRangeTo, i64));
iproj!(3, (IRangeFrom, IRangeFrom, i64));
iproj!(3, (IRangeFrom, IRangeFull, i64));
iproj!(3, (IRangeFull, i64, i64));
iproj!(3, (IRangeFull, i64, IRange));
iproj!(3, (IRangeFull, i64, IRangeTo));
iproj!(3, (IRangeFull, i64, IRangeFrom));
iproj!(3, (IRangeFull, i64, IRangeFull));
iproj!(3, (IRangeFull, IRange, i64));
iproj!(3, (IRangeFull, IRangeTo, i64));
iproj!(3, (IRangeFull, IRangeFrom, i64));
iproj!(3, (IRangeFull, IRangeFull, i64));

index!(4, (IRange, IRange, IRange, IRange));
index!(4, (IRange, IRange, IRange, IRangeTo));
index!(4, (IRange, IRange, IRange, IRangeFrom));
index!(4, (IRange, IRange, IRange, IRangeFull));
index!(4, (IRange, IRange, IRangeTo, IRange));
index!(4, (IRange, IRange, IRangeTo, IRangeTo));
index!(4, (IRange, IRange, IRangeTo, IRangeFrom));
index!(4, (IRange, IRange, IRangeTo, IRangeFull));
index!(4, (IRange, IRange, IRangeFrom, IRange));
index!(4, (IRange, IRange, IRangeFrom, IRangeTo));
index!(4, (IRange, IRange, IRangeFrom, IRangeFrom));
index!(4, (IRange, IRange, IRangeFrom, IRangeFull));
index!(4, (IRange, IRange, IRangeFull, IRange));
index!(4, (IRange, IRange, IRangeFull, IRangeTo));
index!(4, (IRange, IRange, IRangeFull, IRangeFrom));
index!(4, (IRange, IRange, IRangeFull, IRangeFull));
index!(4, (IRange, IRangeTo, IRange, IRange));
index!(4, (IRange, IRangeTo, IRange, IRangeTo));
index!(4, (IRange, IRangeTo, IRange, IRangeFrom));
index!(4, (IRange, IRangeTo, IRange, IRangeFull));
index!(4, (IRange, IRangeTo, IRangeTo, IRange));
index!(4, (IRange, IRangeTo, IRangeTo, IRangeTo));
index!(4, (IRange, IRangeTo, IRangeTo, IRangeFrom));
index!(4, (IRange, IRangeTo, IRangeTo, IRangeFull));
index!(4, (IRange, IRangeTo, IRangeFrom, IRange));
index!(4, (IRange, IRangeTo, IRangeFrom, IRangeTo));
index!(4, (IRange, IRangeTo, IRangeFrom, IRangeFrom));
index!(4, (IRange, IRangeTo, IRangeFrom, IRangeFull));
index!(4, (IRange, IRangeTo, IRangeFull, IRange));
index!(4, (IRange, IRangeTo, IRangeFull, IRangeTo));
index!(4, (IRange, IRangeTo, IRangeFull, IRangeFrom));
index!(4, (IRange, IRangeTo, IRangeFull, IRangeFull));
index!(4, (IRange, IRangeFrom, IRange, IRange));
index!(4, (IRange, IRangeFrom, IRange, IRangeTo));
index!(4, (IRange, IRangeFrom, IRange, IRangeFrom));
index!(4, (IRange, IRangeFrom, IRange, IRangeFull));
index!(4, (IRange, IRangeFrom, IRangeTo, IRange));
index!(4, (IRange, IRangeFrom, IRangeTo, IRangeTo));
index!(4, (IRange, IRangeFrom, IRangeTo, IRangeFrom));
index!(4, (IRange, IRangeFrom, IRangeTo, IRangeFull));
index!(4, (IRange, IRangeFrom, IRangeFrom, IRange));
index!(4, (IRange, IRangeFrom, IRangeFrom, IRangeTo));
index!(4, (IRange, IRangeFrom, IRangeFrom, IRangeFrom));
index!(4, (IRange, IRangeFrom, IRangeFrom, IRangeFull));
index!(4, (IRange, IRangeFrom, IRangeFull, IRange));
index!(4, (IRange, IRangeFrom, IRangeFull, IRangeTo));
index!(4, (IRange, IRangeFrom, IRangeFull, IRangeFrom));
index!(4, (IRange, IRangeFrom, IRangeFull, IRangeFull));
index!(4, (IRange, IRangeFull, IRange, IRange));
index!(4, (IRange, IRangeFull, IRange, IRangeTo));
index!(4, (IRange, IRangeFull, IRange, IRangeFrom));
index!(4, (IRange, IRangeFull, IRange, IRangeFull));
index!(4, (IRange, IRangeFull, IRangeTo, IRange));
index!(4, (IRange, IRangeFull, IRangeTo, IRangeTo));
index!(4, (IRange, IRangeFull, IRangeTo, IRangeFrom));
index!(4, (IRange, IRangeFull, IRangeTo, IRangeFull));
index!(4, (IRange, IRangeFull, IRangeFrom, IRange));
index!(4, (IRange, IRangeFull, IRangeFrom, IRangeTo));
index!(4, (IRange, IRangeFull, IRangeFrom, IRangeFrom));
index!(4, (IRange, IRangeFull, IRangeFrom, IRangeFull));
index!(4, (IRange, IRangeFull, IRangeFull, IRange));
index!(4, (IRange, IRangeFull, IRangeFull, IRangeTo));
index!(4, (IRange, IRangeFull, IRangeFull, IRangeFrom));
index!(4, (IRange, IRangeFull, IRangeFull, IRangeFull));
index!(4, (IRangeTo, IRange, IRange, IRange));
index!(4, (IRangeTo, IRange, IRange, IRangeTo));
index!(4, (IRangeTo, IRange, IRange, IRangeFrom));
index!(4, (IRangeTo, IRange, IRange, IRangeFull));
index!(4, (IRangeTo, IRange, IRangeTo, IRange));
index!(4, (IRangeTo, IRange, IRangeTo, IRangeTo));
index!(4, (IRangeTo, IRange, IRangeTo, IRangeFrom));
index!(4, (IRangeTo, IRange, IRangeTo, IRangeFull));
index!(4, (IRangeTo, IRange, IRangeFrom, IRange));
index!(4, (IRangeTo, IRange, IRangeFrom, IRangeTo));
index!(4, (IRangeTo, IRange, IRangeFrom, IRangeFrom));
index!(4, (IRangeTo, IRange, IRangeFrom, IRangeFull));
index!(4, (IRangeTo, IRange, IRangeFull, IRange));
index!(4, (IRangeTo, IRange, IRangeFull, IRangeTo));
index!(4, (IRangeTo, IRange, IRangeFull, IRangeFrom));
index!(4, (IRangeTo, IRange, IRangeFull, IRangeFull));
index!(4, (IRangeTo, IRangeTo, IRange, IRange));
index!(4, (IRangeTo, IRangeTo, IRange, IRangeTo));
index!(4, (IRangeTo, IRangeTo, IRange, IRangeFrom));
index!(4, (IRangeTo, IRangeTo, IRange, IRangeFull));
index!(4, (IRangeTo, IRangeTo, IRangeTo, IRange));
index!(4, (IRangeTo, IRangeTo, IRangeTo, IRangeTo));
index!(4, (IRangeTo, IRangeTo, IRangeTo, IRangeFrom));
index!(4, (IRangeTo, IRangeTo, IRangeTo, IRangeFull));
index!(4, (IRangeTo, IRangeTo, IRangeFrom, IRange));
index!(4, (IRangeTo, IRangeTo, IRangeFrom, IRangeTo));
index!(4, (IRangeTo, IRangeTo, IRangeFrom, IRangeFrom));
index!(4, (IRangeTo, IRangeTo, IRangeFrom, IRangeFull));
index!(4, (IRangeTo, IRangeTo, IRangeFull, IRange));
index!(4, (IRangeTo, IRangeTo, IRangeFull, IRangeTo));
index!(4, (IRangeTo, IRangeTo, IRangeFull, IRangeFrom));
index!(4, (IRangeTo, IRangeTo, IRangeFull, IRangeFull));
index!(4, (IRangeTo, IRangeFrom, IRange, IRange));
index!(4, (IRangeTo, IRangeFrom, IRange, IRangeTo));
index!(4, (IRangeTo, IRangeFrom, IRange, IRangeFrom));
index!(4, (IRangeTo, IRangeFrom, IRange, IRangeFull));
index!(4, (IRangeTo, IRangeFrom, IRangeTo, IRange));
index!(4, (IRangeTo, IRangeFrom, IRangeTo, IRangeTo));
index!(4, (IRangeTo, IRangeFrom, IRangeTo, IRangeFrom));
index!(4, (IRangeTo, IRangeFrom, IRangeTo, IRangeFull));
index!(4, (IRangeTo, IRangeFrom, IRangeFrom, IRange));
index!(4, (IRangeTo, IRangeFrom, IRangeFrom, IRangeTo));
index!(4, (IRangeTo, IRangeFrom, IRangeFrom, IRangeFrom));
index!(4, (IRangeTo, IRangeFrom, IRangeFrom, IRangeFull));
index!(4, (IRangeTo, IRangeFrom, IRangeFull, IRange));
index!(4, (IRangeTo, IRangeFrom, IRangeFull, IRangeTo));
index!(4, (IRangeTo, IRangeFrom, IRangeFull, IRangeFrom));
index!(4, (IRangeTo, IRangeFrom, IRangeFull, IRangeFull));
index!(4, (IRangeTo, IRangeFull, IRange, IRange));
index!(4, (IRangeTo, IRangeFull, IRange, IRangeTo));
index!(4, (IRangeTo, IRangeFull, IRange, IRangeFrom));
index!(4, (IRangeTo, IRangeFull, IRange, IRangeFull));
index!(4, (IRangeTo, IRangeFull, IRangeTo, IRange));
index!(4, (IRangeTo, IRangeFull, IRangeTo, IRangeTo));
index!(4, (IRangeTo, IRangeFull, IRangeTo, IRangeFrom));
index!(4, (IRangeTo, IRangeFull, IRangeTo, IRangeFull));
index!(4, (IRangeTo, IRangeFull, IRangeFrom, IRange));
index!(4, (IRangeTo, IRangeFull, IRangeFrom, IRangeTo));
index!(4, (IRangeTo, IRangeFull, IRangeFrom, IRangeFrom));
index!(4, (IRangeTo, IRangeFull, IRangeFrom, IRangeFull));
index!(4, (IRangeTo, IRangeFull, IRangeFull, IRange));
index!(4, (IRangeTo, IRangeFull, IRangeFull, IRangeTo));
index!(4, (IRangeTo, IRangeFull, IRangeFull, IRangeFrom));
index!(4, (IRangeTo, IRangeFull, IRangeFull, IRangeFull));
index!(4, (IRangeFrom, IRange, IRange, IRange));
index!(4, (IRangeFrom, IRange, IRange, IRangeTo));
index!(4, (IRangeFrom, IRange, IRange, IRangeFrom));
index!(4, (IRangeFrom, IRange, IRange, IRangeFull));
index!(4, (IRangeFrom, IRange, IRangeTo, IRange));
index!(4, (IRangeFrom, IRange, IRangeTo, IRangeTo));
index!(4, (IRangeFrom, IRange, IRangeTo, IRangeFrom));
index!(4, (IRangeFrom, IRange, IRangeTo, IRangeFull));
index!(4, (IRangeFrom, IRange, IRangeFrom, IRange));
index!(4, (IRangeFrom, IRange, IRangeFrom, IRangeTo));
index!(4, (IRangeFrom, IRange, IRangeFrom, IRangeFrom));
index!(4, (IRangeFrom, IRange, IRangeFrom, IRangeFull));
index!(4, (IRangeFrom, IRange, IRangeFull, IRange));
index!(4, (IRangeFrom, IRange, IRangeFull, IRangeTo));
index!(4, (IRangeFrom, IRange, IRangeFull, IRangeFrom));
index!(4, (IRangeFrom, IRange, IRangeFull, IRangeFull));
index!(4, (IRangeFrom, IRangeTo, IRange, IRange));
index!(4, (IRangeFrom, IRangeTo, IRange, IRangeTo));
index!(4, (IRangeFrom, IRangeTo, IRange, IRangeFrom));
index!(4, (IRangeFrom, IRangeTo, IRange, IRangeFull));
index!(4, (IRangeFrom, IRangeTo, IRangeTo, IRange));
index!(4, (IRangeFrom, IRangeTo, IRangeTo, IRangeTo));
index!(4, (IRangeFrom, IRangeTo, IRangeTo, IRangeFrom));
index!(4, (IRangeFrom, IRangeTo, IRangeTo, IRangeFull));
index!(4, (IRangeFrom, IRangeTo, IRangeFrom, IRange));
index!(4, (IRangeFrom, IRangeTo, IRangeFrom, IRangeTo));
index!(4, (IRangeFrom, IRangeTo, IRangeFrom, IRangeFrom));
index!(4, (IRangeFrom, IRangeTo, IRangeFrom, IRangeFull));
index!(4, (IRangeFrom, IRangeTo, IRangeFull, IRange));
index!(4, (IRangeFrom, IRangeTo, IRangeFull, IRangeTo));
index!(4, (IRangeFrom, IRangeTo, IRangeFull, IRangeFrom));
index!(4, (IRangeFrom, IRangeTo, IRangeFull, IRangeFull));
index!(4, (IRangeFrom, IRangeFrom, IRange, IRange));
index!(4, (IRangeFrom, IRangeFrom, IRange, IRangeTo));
index!(4, (IRangeFrom, IRangeFrom, IRange, IRangeFrom));
index!(4, (IRangeFrom, IRangeFrom, IRange, IRangeFull));
index!(4, (IRangeFrom, IRangeFrom, IRangeTo, IRange));
index!(4, (IRangeFrom, IRangeFrom, IRangeTo, IRangeTo));
index!(4, (IRangeFrom, IRangeFrom, IRangeTo, IRangeFrom));
index!(4, (IRangeFrom, IRangeFrom, IRangeTo, IRangeFull));
index!(4, (IRangeFrom, IRangeFrom, IRangeFrom, IRange));
index!(4, (IRangeFrom, IRangeFrom, IRangeFrom, IRangeTo));
index!(4, (IRangeFrom, IRangeFrom, IRangeFrom, IRangeFrom));
index!(4, (IRangeFrom, IRangeFrom, IRangeFrom, IRangeFull));
index!(4, (IRangeFrom, IRangeFrom, IRangeFull, IRange));
index!(4, (IRangeFrom, IRangeFrom, IRangeFull, IRangeTo));
index!(4, (IRangeFrom, IRangeFrom, IRangeFull, IRangeFrom));
index!(4, (IRangeFrom, IRangeFrom, IRangeFull, IRangeFull));
index!(4, (IRangeFrom, IRangeFull, IRange, IRange));
index!(4, (IRangeFrom, IRangeFull, IRange, IRangeTo));
index!(4, (IRangeFrom, IRangeFull, IRange, IRangeFrom));
index!(4, (IRangeFrom, IRangeFull, IRange, IRangeFull));
index!(4, (IRangeFrom, IRangeFull, IRangeTo, IRange));
index!(4, (IRangeFrom, IRangeFull, IRangeTo, IRangeTo));
index!(4, (IRangeFrom, IRangeFull, IRangeTo, IRangeFrom));
index!(4, (IRangeFrom, IRangeFull, IRangeTo, IRangeFull));
index!(4, (IRangeFrom, IRangeFull, IRangeFrom, IRange));
index!(4, (IRangeFrom, IRangeFull, IRangeFrom, IRangeTo));
index!(4, (IRangeFrom, IRangeFull, IRangeFrom, IRangeFrom));
index!(4, (IRangeFrom, IRangeFull, IRangeFrom, IRangeFull));
index!(4, (IRangeFrom, IRangeFull, IRangeFull, IRange));
index!(4, (IRangeFrom, IRangeFull, IRangeFull, IRangeTo));
index!(4, (IRangeFrom, IRangeFull, IRangeFull, IRangeFrom));
index!(4, (IRangeFrom, IRangeFull, IRangeFull, IRangeFull));
index!(4, (IRangeFull, IRange, IRange, IRange));
index!(4, (IRangeFull, IRange, IRange, IRangeTo));
index!(4, (IRangeFull, IRange, IRange, IRangeFrom));
index!(4, (IRangeFull, IRange, IRange, IRangeFull));
index!(4, (IRangeFull, IRange, IRangeTo, IRange));
index!(4, (IRangeFull, IRange, IRangeTo, IRangeTo));
index!(4, (IRangeFull, IRange, IRangeTo, IRangeFrom));
index!(4, (IRangeFull, IRange, IRangeTo, IRangeFull));
index!(4, (IRangeFull, IRange, IRangeFrom, IRange));
index!(4, (IRangeFull, IRange, IRangeFrom, IRangeTo));
index!(4, (IRangeFull, IRange, IRangeFrom, IRangeFrom));
index!(4, (IRangeFull, IRange, IRangeFrom, IRangeFull));
index!(4, (IRangeFull, IRange, IRangeFull, IRange));
index!(4, (IRangeFull, IRange, IRangeFull, IRangeTo));
index!(4, (IRangeFull, IRange, IRangeFull, IRangeFrom));
index!(4, (IRangeFull, IRange, IRangeFull, IRangeFull));
index!(4, (IRangeFull, IRangeTo, IRange, IRange));
index!(4, (IRangeFull, IRangeTo, IRange, IRangeTo));
index!(4, (IRangeFull, IRangeTo, IRange, IRangeFrom));
index!(4, (IRangeFull, IRangeTo, IRange, IRangeFull));
index!(4, (IRangeFull, IRangeTo, IRangeTo, IRange));
index!(4, (IRangeFull, IRangeTo, IRangeTo, IRangeTo));
index!(4, (IRangeFull, IRangeTo, IRangeTo, IRangeFrom));
index!(4, (IRangeFull, IRangeTo, IRangeTo, IRangeFull));
index!(4, (IRangeFull, IRangeTo, IRangeFrom, IRange));
index!(4, (IRangeFull, IRangeTo, IRangeFrom, IRangeTo));
index!(4, (IRangeFull, IRangeTo, IRangeFrom, IRangeFrom));
index!(4, (IRangeFull, IRangeTo, IRangeFrom, IRangeFull));
index!(4, (IRangeFull, IRangeTo, IRangeFull, IRange));
index!(4, (IRangeFull, IRangeTo, IRangeFull, IRangeTo));
index!(4, (IRangeFull, IRangeTo, IRangeFull, IRangeFrom));
index!(4, (IRangeFull, IRangeTo, IRangeFull, IRangeFull));
index!(4, (IRangeFull, IRangeFrom, IRange, IRange));
index!(4, (IRangeFull, IRangeFrom, IRange, IRangeTo));
index!(4, (IRangeFull, IRangeFrom, IRange, IRangeFrom));
index!(4, (IRangeFull, IRangeFrom, IRange, IRangeFull));
index!(4, (IRangeFull, IRangeFrom, IRangeTo, IRange));
index!(4, (IRangeFull, IRangeFrom, IRangeTo, IRangeTo));
index!(4, (IRangeFull, IRangeFrom, IRangeTo, IRangeFrom));
index!(4, (IRangeFull, IRangeFrom, IRangeTo, IRangeFull));
index!(4, (IRangeFull, IRangeFrom, IRangeFrom, IRange));
index!(4, (IRangeFull, IRangeFrom, IRangeFrom, IRangeTo));
index!(4, (IRangeFull, IRangeFrom, IRangeFrom, IRangeFrom));
index!(4, (IRangeFull, IRangeFrom, IRangeFrom, IRangeFull));
index!(4, (IRangeFull, IRangeFrom, IRangeFull, IRange));
index!(4, (IRangeFull, IRangeFrom, IRangeFull, IRangeTo));
index!(4, (IRangeFull, IRangeFrom, IRangeFull, IRangeFrom));
index!(4, (IRangeFull, IRangeFrom, IRangeFull, IRangeFull));
index!(4, (IRangeFull, IRangeFull, IRange, IRange));
index!(4, (IRangeFull, IRangeFull, IRange, IRangeTo));
index!(4, (IRangeFull, IRangeFull, IRange, IRangeFrom));
index!(4, (IRangeFull, IRangeFull, IRange, IRangeFull));
index!(4, (IRangeFull, IRangeFull, IRangeTo, IRange));
index!(4, (IRangeFull, IRangeFull, IRangeTo, IRangeTo));
index!(4, (IRangeFull, IRangeFull, IRangeTo, IRangeFrom));
index!(4, (IRangeFull, IRangeFull, IRangeTo, IRangeFull));
index!(4, (IRangeFull, IRangeFull, IRangeFrom, IRange));
index!(4, (IRangeFull, IRangeFull, IRangeFrom, IRangeTo));
index!(4, (IRangeFull, IRangeFull, IRangeFrom, IRangeFrom));
index!(4, (IRangeFull, IRangeFull, IRangeFrom, IRangeFull));
index!(4, (IRangeFull, IRangeFull, IRangeFull, IRange));
index!(4, (IRangeFull, IRangeFull, IRangeFull, IRangeTo));
index!(4, (IRangeFull, IRangeFull, IRangeFull, IRangeFrom));
ifull!(4, (IRangeFull, IRangeFull, IRangeFull, IRangeFull));
iproj!(4, (i64, i64, i64, i64));
iproj!(4, (i64, i64, i64, IRange));
iproj!(4, (i64, i64, i64, IRangeTo));
iproj!(4, (i64, i64, i64, IRangeFrom));
iproj!(4, (i64, i64, i64, IRangeFull));
iproj!(4, (i64, i64, IRange, i64));
iproj!(4, (i64, i64, IRange, IRange));
iproj!(4, (i64, i64, IRange, IRangeTo));
iproj!(4, (i64, i64, IRange, IRangeFrom));
iproj!(4, (i64, i64, IRange, IRangeFull));
iproj!(4, (i64, i64, IRangeTo, i64));
iproj!(4, (i64, i64, IRangeTo, IRange));
iproj!(4, (i64, i64, IRangeTo, IRangeTo));
iproj!(4, (i64, i64, IRangeTo, IRangeFrom));
iproj!(4, (i64, i64, IRangeTo, IRangeFull));
iproj!(4, (i64, i64, IRangeFrom, i64));
iproj!(4, (i64, i64, IRangeFrom, IRange));
iproj!(4, (i64, i64, IRangeFrom, IRangeTo));
iproj!(4, (i64, i64, IRangeFrom, IRangeFrom));
iproj!(4, (i64, i64, IRangeFrom, IRangeFull));
iproj!(4, (i64, i64, IRangeFull, i64));
iproj!(4, (i64, i64, IRangeFull, IRange));
iproj!(4, (i64, i64, IRangeFull, IRangeTo));
iproj!(4, (i64, i64, IRangeFull, IRangeFrom));
iproj!(4, (i64, i64, IRangeFull, IRangeFull));
iproj!(4, (i64, IRange, i64, i64));
iproj!(4, (i64, IRange, i64, IRange));
iproj!(4, (i64, IRange, i64, IRangeTo));
iproj!(4, (i64, IRange, i64, IRangeFrom));
iproj!(4, (i64, IRange, i64, IRangeFull));
iproj!(4, (i64, IRange, IRange, i64));
iproj!(4, (i64, IRange, IRange, IRange));
iproj!(4, (i64, IRange, IRange, IRangeTo));
iproj!(4, (i64, IRange, IRange, IRangeFrom));
iproj!(4, (i64, IRange, IRange, IRangeFull));
iproj!(4, (i64, IRange, IRangeTo, i64));
iproj!(4, (i64, IRange, IRangeTo, IRange));
iproj!(4, (i64, IRange, IRangeTo, IRangeTo));
iproj!(4, (i64, IRange, IRangeTo, IRangeFrom));
iproj!(4, (i64, IRange, IRangeTo, IRangeFull));
iproj!(4, (i64, IRange, IRangeFrom, i64));
iproj!(4, (i64, IRange, IRangeFrom, IRange));
iproj!(4, (i64, IRange, IRangeFrom, IRangeTo));
iproj!(4, (i64, IRange, IRangeFrom, IRangeFrom));
iproj!(4, (i64, IRange, IRangeFrom, IRangeFull));
iproj!(4, (i64, IRange, IRangeFull, i64));
iproj!(4, (i64, IRange, IRangeFull, IRange));
iproj!(4, (i64, IRange, IRangeFull, IRangeTo));
iproj!(4, (i64, IRange, IRangeFull, IRangeFrom));
iproj!(4, (i64, IRange, IRangeFull, IRangeFull));
iproj!(4, (i64, IRangeTo, i64, i64));
iproj!(4, (i64, IRangeTo, i64, IRange));
iproj!(4, (i64, IRangeTo, i64, IRangeTo));
iproj!(4, (i64, IRangeTo, i64, IRangeFrom));
iproj!(4, (i64, IRangeTo, i64, IRangeFull));
iproj!(4, (i64, IRangeTo, IRange, i64));
iproj!(4, (i64, IRangeTo, IRange, IRange));
iproj!(4, (i64, IRangeTo, IRange, IRangeTo));
iproj!(4, (i64, IRangeTo, IRange, IRangeFrom));
iproj!(4, (i64, IRangeTo, IRange, IRangeFull));
iproj!(4, (i64, IRangeTo, IRangeTo, i64));
iproj!(4, (i64, IRangeTo, IRangeTo, IRange));
iproj!(4, (i64, IRangeTo, IRangeTo, IRangeTo));
iproj!(4, (i64, IRangeTo, IRangeTo, IRangeFrom));
iproj!(4, (i64, IRangeTo, IRangeTo, IRangeFull));
iproj!(4, (i64, IRangeTo, IRangeFrom, i64));
iproj!(4, (i64, IRangeTo, IRangeFrom, IRange));
iproj!(4, (i64, IRangeTo, IRangeFrom, IRangeTo));
iproj!(4, (i64, IRangeTo, IRangeFrom, IRangeFrom));
iproj!(4, (i64, IRangeTo, IRangeFrom, IRangeFull));
iproj!(4, (i64, IRangeTo, IRangeFull, i64));
iproj!(4, (i64, IRangeTo, IRangeFull, IRange));
iproj!(4, (i64, IRangeTo, IRangeFull, IRangeTo));
iproj!(4, (i64, IRangeTo, IRangeFull, IRangeFrom));
iproj!(4, (i64, IRangeTo, IRangeFull, IRangeFull));
iproj!(4, (i64, IRangeFrom, i64, i64));
iproj!(4, (i64, IRangeFrom, i64, IRange));
iproj!(4, (i64, IRangeFrom, i64, IRangeTo));
iproj!(4, (i64, IRangeFrom, i64, IRangeFrom));
iproj!(4, (i64, IRangeFrom, i64, IRangeFull));
iproj!(4, (i64, IRangeFrom, IRange, i64));
iproj!(4, (i64, IRangeFrom, IRange, IRange));
iproj!(4, (i64, IRangeFrom, IRange, IRangeTo));
iproj!(4, (i64, IRangeFrom, IRange, IRangeFrom));
iproj!(4, (i64, IRangeFrom, IRange, IRangeFull));
iproj!(4, (i64, IRangeFrom, IRangeTo, i64));
iproj!(4, (i64, IRangeFrom, IRangeTo, IRange));
iproj!(4, (i64, IRangeFrom, IRangeTo, IRangeTo));
iproj!(4, (i64, IRangeFrom, IRangeTo, IRangeFrom));
iproj!(4, (i64, IRangeFrom, IRangeTo, IRangeFull));
iproj!(4, (i64, IRangeFrom, IRangeFrom, i64));
iproj!(4, (i64, IRangeFrom, IRangeFrom, IRange));
iproj!(4, (i64, IRangeFrom, IRangeFrom, IRangeTo));
iproj!(4, (i64, IRangeFrom, IRangeFrom, IRangeFrom));
iproj!(4, (i64, IRangeFrom, IRangeFrom, IRangeFull));
iproj!(4, (i64, IRangeFrom, IRangeFull, i64));
iproj!(4, (i64, IRangeFrom, IRangeFull, IRange));
iproj!(4, (i64, IRangeFrom, IRangeFull, IRangeTo));
iproj!(4, (i64, IRangeFrom, IRangeFull, IRangeFrom));
iproj!(4, (i64, IRangeFrom, IRangeFull, IRangeFull));
iproj!(4, (i64, IRangeFull, i64, i64));
iproj!(4, (i64, IRangeFull, i64, IRange));
iproj!(4, (i64, IRangeFull, i64, IRangeTo));
iproj!(4, (i64, IRangeFull, i64, IRangeFrom));
iproj!(4, (i64, IRangeFull, i64, IRangeFull));
iproj!(4, (i64, IRangeFull, IRange, i64));
iproj!(4, (i64, IRangeFull, IRange, IRange));
iproj!(4, (i64, IRangeFull, IRange, IRangeTo));
iproj!(4, (i64, IRangeFull, IRange, IRangeFrom));
iproj!(4, (i64, IRangeFull, IRange, IRangeFull));
iproj!(4, (i64, IRangeFull, IRangeTo, i64));
iproj!(4, (i64, IRangeFull, IRangeTo, IRange));
iproj!(4, (i64, IRangeFull, IRangeTo, IRangeTo));
iproj!(4, (i64, IRangeFull, IRangeTo, IRangeFrom));
iproj!(4, (i64, IRangeFull, IRangeTo, IRangeFull));
iproj!(4, (i64, IRangeFull, IRangeFrom, i64));
iproj!(4, (i64, IRangeFull, IRangeFrom, IRange));
iproj!(4, (i64, IRangeFull, IRangeFrom, IRangeTo));
iproj!(4, (i64, IRangeFull, IRangeFrom, IRangeFrom));
iproj!(4, (i64, IRangeFull, IRangeFrom, IRangeFull));
iproj!(4, (i64, IRangeFull, IRangeFull, i64));
iproj!(4, (i64, IRangeFull, IRangeFull, IRange));
iproj!(4, (i64, IRangeFull, IRangeFull, IRangeTo));
iproj!(4, (i64, IRangeFull, IRangeFull, IRangeFrom));
iproj!(4, (i64, IRangeFull, IRangeFull, IRangeFull));
iproj!(4, (IRange, i64, i64, i64));
iproj!(4, (IRange, i64, i64, IRange));
iproj!(4, (IRange, i64, i64, IRangeTo));
iproj!(4, (IRange, i64, i64, IRangeFrom));
iproj!(4, (IRange, i64, i64, IRangeFull));
iproj!(4, (IRange, i64, IRange, i64));
iproj!(4, (IRange, i64, IRange, IRange));
iproj!(4, (IRange, i64, IRange, IRangeTo));
iproj!(4, (IRange, i64, IRange, IRangeFrom));
iproj!(4, (IRange, i64, IRange, IRangeFull));
iproj!(4, (IRange, i64, IRangeTo, i64));
iproj!(4, (IRange, i64, IRangeTo, IRange));
iproj!(4, (IRange, i64, IRangeTo, IRangeTo));
iproj!(4, (IRange, i64, IRangeTo, IRangeFrom));
iproj!(4, (IRange, i64, IRangeTo, IRangeFull));
iproj!(4, (IRange, i64, IRangeFrom, i64));
iproj!(4, (IRange, i64, IRangeFrom, IRange));
iproj!(4, (IRange, i64, IRangeFrom, IRangeTo));
iproj!(4, (IRange, i64, IRangeFrom, IRangeFrom));
iproj!(4, (IRange, i64, IRangeFrom, IRangeFull));
iproj!(4, (IRange, i64, IRangeFull, i64));
iproj!(4, (IRange, i64, IRangeFull, IRange));
iproj!(4, (IRange, i64, IRangeFull, IRangeTo));
iproj!(4, (IRange, i64, IRangeFull, IRangeFrom));
iproj!(4, (IRange, i64, IRangeFull, IRangeFull));
iproj!(4, (IRange, IRange, i64, i64));
iproj!(4, (IRange, IRange, i64, IRange));
iproj!(4, (IRange, IRange, i64, IRangeTo));
iproj!(4, (IRange, IRange, i64, IRangeFrom));
iproj!(4, (IRange, IRange, i64, IRangeFull));
iproj!(4, (IRange, IRange, IRange, i64));
iproj!(4, (IRange, IRange, IRangeTo, i64));
iproj!(4, (IRange, IRange, IRangeFrom, i64));
iproj!(4, (IRange, IRange, IRangeFull, i64));
iproj!(4, (IRange, IRangeTo, i64, i64));
iproj!(4, (IRange, IRangeTo, i64, IRange));
iproj!(4, (IRange, IRangeTo, i64, IRangeTo));
iproj!(4, (IRange, IRangeTo, i64, IRangeFrom));
iproj!(4, (IRange, IRangeTo, i64, IRangeFull));
iproj!(4, (IRange, IRangeTo, IRange, i64));
iproj!(4, (IRange, IRangeTo, IRangeTo, i64));
iproj!(4, (IRange, IRangeTo, IRangeFrom, i64));
iproj!(4, (IRange, IRangeTo, IRangeFull, i64));
iproj!(4, (IRange, IRangeFrom, i64, i64));
iproj!(4, (IRange, IRangeFrom, i64, IRange));
iproj!(4, (IRange, IRangeFrom, i64, IRangeTo));
iproj!(4, (IRange, IRangeFrom, i64, IRangeFrom));
iproj!(4, (IRange, IRangeFrom, i64, IRangeFull));
iproj!(4, (IRange, IRangeFrom, IRange, i64));
iproj!(4, (IRange, IRangeFrom, IRangeTo, i64));
iproj!(4, (IRange, IRangeFrom, IRangeFrom, i64));
iproj!(4, (IRange, IRangeFrom, IRangeFull, i64));
iproj!(4, (IRange, IRangeFull, i64, i64));
iproj!(4, (IRange, IRangeFull, i64, IRange));
iproj!(4, (IRange, IRangeFull, i64, IRangeTo));
iproj!(4, (IRange, IRangeFull, i64, IRangeFrom));
iproj!(4, (IRange, IRangeFull, i64, IRangeFull));
iproj!(4, (IRange, IRangeFull, IRange, i64));
iproj!(4, (IRange, IRangeFull, IRangeTo, i64));
iproj!(4, (IRange, IRangeFull, IRangeFrom, i64));
iproj!(4, (IRange, IRangeFull, IRangeFull, i64));
iproj!(4, (IRangeTo, i64, i64, i64));
iproj!(4, (IRangeTo, i64, i64, IRange));
iproj!(4, (IRangeTo, i64, i64, IRangeTo));
iproj!(4, (IRangeTo, i64, i64, IRangeFrom));
iproj!(4, (IRangeTo, i64, i64, IRangeFull));
iproj!(4, (IRangeTo, i64, IRange, i64));
iproj!(4, (IRangeTo, i64, IRange, IRange));
iproj!(4, (IRangeTo, i64, IRange, IRangeTo));
iproj!(4, (IRangeTo, i64, IRange, IRangeFrom));
iproj!(4, (IRangeTo, i64, IRange, IRangeFull));
iproj!(4, (IRangeTo, i64, IRangeTo, i64));
iproj!(4, (IRangeTo, i64, IRangeTo, IRange));
iproj!(4, (IRangeTo, i64, IRangeTo, IRangeTo));
iproj!(4, (IRangeTo, i64, IRangeTo, IRangeFrom));
iproj!(4, (IRangeTo, i64, IRangeTo, IRangeFull));
iproj!(4, (IRangeTo, i64, IRangeFrom, i64));
iproj!(4, (IRangeTo, i64, IRangeFrom, IRange));
iproj!(4, (IRangeTo, i64, IRangeFrom, IRangeTo));
iproj!(4, (IRangeTo, i64, IRangeFrom, IRangeFrom));
iproj!(4, (IRangeTo, i64, IRangeFrom, IRangeFull));
iproj!(4, (IRangeTo, i64, IRangeFull, i64));
iproj!(4, (IRangeTo, i64, IRangeFull, IRange));
iproj!(4, (IRangeTo, i64, IRangeFull, IRangeTo));
iproj!(4, (IRangeTo, i64, IRangeFull, IRangeFrom));
iproj!(4, (IRangeTo, i64, IRangeFull, IRangeFull));
iproj!(4, (IRangeTo, IRange, i64, i64));
iproj!(4, (IRangeTo, IRange, i64, IRange));
iproj!(4, (IRangeTo, IRange, i64, IRangeTo));
iproj!(4, (IRangeTo, IRange, i64, IRangeFrom));
iproj!(4, (IRangeTo, IRange, i64, IRangeFull));
iproj!(4, (IRangeTo, IRange, IRange, i64));
iproj!(4, (IRangeTo, IRange, IRangeTo, i64));
iproj!(4, (IRangeTo, IRange, IRangeFrom, i64));
iproj!(4, (IRangeTo, IRange, IRangeFull, i64));
iproj!(4, (IRangeTo, IRangeTo, i64, i64));
iproj!(4, (IRangeTo, IRangeTo, i64, IRange));
iproj!(4, (IRangeTo, IRangeTo, i64, IRangeTo));
iproj!(4, (IRangeTo, IRangeTo, i64, IRangeFrom));
iproj!(4, (IRangeTo, IRangeTo, i64, IRangeFull));
iproj!(4, (IRangeTo, IRangeTo, IRange, i64));
iproj!(4, (IRangeTo, IRangeTo, IRangeTo, i64));
iproj!(4, (IRangeTo, IRangeTo, IRangeFrom, i64));
iproj!(4, (IRangeTo, IRangeTo, IRangeFull, i64));
iproj!(4, (IRangeTo, IRangeFrom, i64, i64));
iproj!(4, (IRangeTo, IRangeFrom, i64, IRange));
iproj!(4, (IRangeTo, IRangeFrom, i64, IRangeTo));
iproj!(4, (IRangeTo, IRangeFrom, i64, IRangeFrom));
iproj!(4, (IRangeTo, IRangeFrom, i64, IRangeFull));
iproj!(4, (IRangeTo, IRangeFrom, IRange, i64));
iproj!(4, (IRangeTo, IRangeFrom, IRangeTo, i64));
iproj!(4, (IRangeTo, IRangeFrom, IRangeFrom, i64));
iproj!(4, (IRangeTo, IRangeFrom, IRangeFull, i64));
iproj!(4, (IRangeTo, IRangeFull, i64, i64));
iproj!(4, (IRangeTo, IRangeFull, i64, IRange));
iproj!(4, (IRangeTo, IRangeFull, i64, IRangeTo));
iproj!(4, (IRangeTo, IRangeFull, i64, IRangeFrom));
iproj!(4, (IRangeTo, IRangeFull, i64, IRangeFull));
iproj!(4, (IRangeTo, IRangeFull, IRange, i64));
iproj!(4, (IRangeTo, IRangeFull, IRangeTo, i64));
iproj!(4, (IRangeTo, IRangeFull, IRangeFrom, i64));
iproj!(4, (IRangeTo, IRangeFull, IRangeFull, i64));
iproj!(4, (IRangeFrom, i64, i64, i64));
iproj!(4, (IRangeFrom, i64, i64, IRange));
iproj!(4, (IRangeFrom, i64, i64, IRangeTo));
iproj!(4, (IRangeFrom, i64, i64, IRangeFrom));
iproj!(4, (IRangeFrom, i64, i64, IRangeFull));
iproj!(4, (IRangeFrom, i64, IRange, i64));
iproj!(4, (IRangeFrom, i64, IRange, IRange));
iproj!(4, (IRangeFrom, i64, IRange, IRangeTo));
iproj!(4, (IRangeFrom, i64, IRange, IRangeFrom));
iproj!(4, (IRangeFrom, i64, IRange, IRangeFull));
iproj!(4, (IRangeFrom, i64, IRangeTo, i64));
iproj!(4, (IRangeFrom, i64, IRangeTo, IRange));
iproj!(4, (IRangeFrom, i64, IRangeTo, IRangeTo));
iproj!(4, (IRangeFrom, i64, IRangeTo, IRangeFrom));
iproj!(4, (IRangeFrom, i64, IRangeTo, IRangeFull));
iproj!(4, (IRangeFrom, i64, IRangeFrom, i64));
iproj!(4, (IRangeFrom, i64, IRangeFrom, IRange));
iproj!(4, (IRangeFrom, i64, IRangeFrom, IRangeTo));
iproj!(4, (IRangeFrom, i64, IRangeFrom, IRangeFrom));
iproj!(4, (IRangeFrom, i64, IRangeFrom, IRangeFull));
iproj!(4, (IRangeFrom, i64, IRangeFull, i64));
iproj!(4, (IRangeFrom, i64, IRangeFull, IRange));
iproj!(4, (IRangeFrom, i64, IRangeFull, IRangeTo));
iproj!(4, (IRangeFrom, i64, IRangeFull, IRangeFrom));
iproj!(4, (IRangeFrom, i64, IRangeFull, IRangeFull));
iproj!(4, (IRangeFrom, IRange, i64, i64));
iproj!(4, (IRangeFrom, IRange, i64, IRange));
iproj!(4, (IRangeFrom, IRange, i64, IRangeTo));
iproj!(4, (IRangeFrom, IRange, i64, IRangeFrom));
iproj!(4, (IRangeFrom, IRange, i64, IRangeFull));
iproj!(4, (IRangeFrom, IRange, IRange, i64));
iproj!(4, (IRangeFrom, IRange, IRangeTo, i64));
iproj!(4, (IRangeFrom, IRange, IRangeFrom, i64));
iproj!(4, (IRangeFrom, IRange, IRangeFull, i64));
iproj!(4, (IRangeFrom, IRangeTo, i64, i64));
iproj!(4, (IRangeFrom, IRangeTo, i64, IRange));
iproj!(4, (IRangeFrom, IRangeTo, i64, IRangeTo));
iproj!(4, (IRangeFrom, IRangeTo, i64, IRangeFrom));
iproj!(4, (IRangeFrom, IRangeTo, i64, IRangeFull));
iproj!(4, (IRangeFrom, IRangeTo, IRange, i64));
iproj!(4, (IRangeFrom, IRangeTo, IRangeTo, i64));
iproj!(4, (IRangeFrom, IRangeTo, IRangeFrom, i64));
iproj!(4, (IRangeFrom, IRangeTo, IRangeFull, i64));
iproj!(4, (IRangeFrom, IRangeFrom, i64, i64));
iproj!(4, (IRangeFrom, IRangeFrom, i64, IRange));
iproj!(4, (IRangeFrom, IRangeFrom, i64, IRangeTo));
iproj!(4, (IRangeFrom, IRangeFrom, i64, IRangeFrom));
iproj!(4, (IRangeFrom, IRangeFrom, i64, IRangeFull));
iproj!(4, (IRangeFrom, IRangeFrom, IRange, i64));
iproj!(4, (IRangeFrom, IRangeFrom, IRangeTo, i64));
iproj!(4, (IRangeFrom, IRangeFrom, IRangeFrom, i64));
iproj!(4, (IRangeFrom, IRangeFrom, IRangeFull, i64));
iproj!(4, (IRangeFrom, IRangeFull, i64, i64));
iproj!(4, (IRangeFrom, IRangeFull, i64, IRange));
iproj!(4, (IRangeFrom, IRangeFull, i64, IRangeTo));
iproj!(4, (IRangeFrom, IRangeFull, i64, IRangeFrom));
iproj!(4, (IRangeFrom, IRangeFull, i64, IRangeFull));
iproj!(4, (IRangeFrom, IRangeFull, IRange, i64));
iproj!(4, (IRangeFrom, IRangeFull, IRangeTo, i64));
iproj!(4, (IRangeFrom, IRangeFull, IRangeFrom, i64));
iproj!(4, (IRangeFrom, IRangeFull, IRangeFull, i64));
iproj!(4, (IRangeFull, i64, i64, i64));
iproj!(4, (IRangeFull, i64, i64, IRange));
iproj!(4, (IRangeFull, i64, i64, IRangeTo));
iproj!(4, (IRangeFull, i64, i64, IRangeFrom));
iproj!(4, (IRangeFull, i64, i64, IRangeFull));
iproj!(4, (IRangeFull, i64, IRange, i64));
iproj!(4, (IRangeFull, i64, IRange, IRange));
iproj!(4, (IRangeFull, i64, IRange, IRangeTo));
iproj!(4, (IRangeFull, i64, IRange, IRangeFrom));
iproj!(4, (IRangeFull, i64, IRange, IRangeFull));
iproj!(4, (IRangeFull, i64, IRangeTo, i64));
iproj!(4, (IRangeFull, i64, IRangeTo, IRange));
iproj!(4, (IRangeFull, i64, IRangeTo, IRangeTo));
iproj!(4, (IRangeFull, i64, IRangeTo, IRangeFrom));
iproj!(4, (IRangeFull, i64, IRangeTo, IRangeFull));
iproj!(4, (IRangeFull, i64, IRangeFrom, i64));
iproj!(4, (IRangeFull, i64, IRangeFrom, IRange));
iproj!(4, (IRangeFull, i64, IRangeFrom, IRangeTo));
iproj!(4, (IRangeFull, i64, IRangeFrom, IRangeFrom));
iproj!(4, (IRangeFull, i64, IRangeFrom, IRangeFull));
iproj!(4, (IRangeFull, i64, IRangeFull, i64));
iproj!(4, (IRangeFull, i64, IRangeFull, IRange));
iproj!(4, (IRangeFull, i64, IRangeFull, IRangeTo));
iproj!(4, (IRangeFull, i64, IRangeFull, IRangeFrom));
iproj!(4, (IRangeFull, i64, IRangeFull, IRangeFull));
iproj!(4, (IRangeFull, IRange, i64, i64));
iproj!(4, (IRangeFull, IRange, i64, IRange));
iproj!(4, (IRangeFull, IRange, i64, IRangeTo));
iproj!(4, (IRangeFull, IRange, i64, IRangeFrom));
iproj!(4, (IRangeFull, IRange, i64, IRangeFull));
iproj!(4, (IRangeFull, IRange, IRange, i64));
iproj!(4, (IRangeFull, IRange, IRangeTo, i64));
iproj!(4, (IRangeFull, IRange, IRangeFrom, i64));
iproj!(4, (IRangeFull, IRange, IRangeFull, i64));
iproj!(4, (IRangeFull, IRangeTo, i64, i64));
iproj!(4, (IRangeFull, IRangeTo, i64, IRange));
iproj!(4, (IRangeFull, IRangeTo, i64, IRangeTo));
iproj!(4, (IRangeFull, IRangeTo, i64, IRangeFrom));
iproj!(4, (IRangeFull, IRangeTo, i64, IRangeFull));
iproj!(4, (IRangeFull, IRangeTo, IRange, i64));
iproj!(4, (IRangeFull, IRangeTo, IRangeTo, i64));
iproj!(4, (IRangeFull, IRangeTo, IRangeFrom, i64));
iproj!(4, (IRangeFull, IRangeTo, IRangeFull, i64));
iproj!(4, (IRangeFull, IRangeFrom, i64, i64));
iproj!(4, (IRangeFull, IRangeFrom, i64, IRange));
iproj!(4, (IRangeFull, IRangeFrom, i64, IRangeTo));
iproj!(4, (IRangeFull, IRangeFrom, i64, IRangeFrom));
iproj!(4, (IRangeFull, IRangeFrom, i64, IRangeFull));
iproj!(4, (IRangeFull, IRangeFrom, IRange, i64));
iproj!(4, (IRangeFull, IRangeFrom, IRangeTo, i64));
iproj!(4, (IRangeFull, IRangeFrom, IRangeFrom, i64));
iproj!(4, (IRangeFull, IRangeFrom, IRangeFull, i64));
iproj!(4, (IRangeFull, IRangeFull, i64, i64));
iproj!(4, (IRangeFull, IRangeFull, i64, IRange));
iproj!(4, (IRangeFull, IRangeFull, i64, IRangeTo));
iproj!(4, (IRangeFull, IRangeFull, i64, IRangeFrom));
iproj!(4, (IRangeFull, IRangeFull, i64, IRangeFull));
iproj!(4, (IRangeFull, IRangeFull, IRange, i64));
iproj!(4, (IRangeFull, IRangeFull, IRangeTo, i64));
iproj!(4, (IRangeFull, IRangeFull, IRangeFrom, i64));
iproj!(4, (IRangeFull, IRangeFull, IRangeFull, i64));
