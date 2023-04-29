use crate::cell::{CellMode};
use crate::clock::*;
use crate::ptr::*;

use std::any::{Any, TypeId};
use std::rc::{Rc};

pub mod ops;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct ThunkPtr(i32);

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ThunkKey {
  // FIXME
  //tyid: TypeId,
}

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

pub trait ThunkSpec: Any + Copy {
  // FIXME FIXME
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
