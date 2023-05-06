use crate::cell::{CellMode};
use crate::clock::*;
use crate::ptr::*;

use std::any::{Any, TypeId};
use std::hash::{Hash};
use std::mem::{size_of};
use std::rc::{Rc};
use std::slice::{from_raw_parts};

pub mod ops;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct ThunkPtr(i32);

/*#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ThunkKey {
  // FIXME
  //tyid: TypeId,
}*/

#[derive(Clone, Copy)]
#[repr(u8)]
pub enum ThunkArgMode {
  _Top,
  I,
  O,
  IO,
}

#[derive(Clone, Copy)]
pub struct ThunkArg {
  pub ptr:  CellPtr,
  pub mode: CellMode,
  pub amod: ThunkArgMode,
  // TODO
}

pub trait ThunkSpec {}

pub trait Thunk {
  fn as_any(&self) -> &dyn Any;
  fn as_bytes_repr(&self) -> &[u8];
  fn thunk_eq(&self, other: &dyn Thunk) -> Option<bool>;
}

impl<T: Any + Copy + Eq + ThunkSpec> Thunk for T {
  fn as_any(&self) -> &dyn Any {
    self
  }

  fn as_bytes_repr(&self) -> &[u8] {
    // SAFETY: This should be safe as the type is `Copy`.
    let ptr = (self as *const T) as *const u8;
    let len = size_of::<T>();
    unsafe { from_raw_parts(ptr, len) }
  }

  fn thunk_eq(&self, other: &dyn Thunk) -> Option<bool> {
    other.as_any().downcast_ref::<T>().map(|other| self == other)
  }
}

pub struct PThunk {
  // FIXME
  pub ptr:      ThunkPtr,
  //pub spec:     Rc<dyn ThunkSpec>,
  pub clk:      Clock,
  //pub localsub: Vec<CellPtr>,
  pub localsub: Vec<ThunkArg>,
  pub inner:    InnerThunk,
}

pub enum InnerThunk {
  _Top,
  Futhark(FutharkThunk),
  Custom(Rc<dyn CustomThunk>),
}

pub struct FutharkThunk {
  // TODO
  pub f_decl:   Vec<u8>,
  pub f_body:   Vec<u8>,
  //pub f_hash:   _,
  pub object:   Option<futhark_ffi::Object>,
}

pub trait CustomThunk: Any {
  // FIXME FIXME
}

// TODO
