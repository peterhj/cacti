use self::nvgpu::{NvGpuPCtx};
use self::smp::{SmpPCtx};
use crate::algo::{HashMap, RevSortMap8};
use crate::algo::fp::*;
use crate::cell::{CellPtr, CellType, DtypeConstExt, InnerCell, InnerCell_};
use crate::panick::*;

use std::borrow::{Borrow};
use std::cell::{Cell, RefCell};
use std::cmp::{max, min};
use std::ffi::{c_void};
use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::io::{Read};
use std::mem::{align_of};
use std::rc::{Rc};
use std::slice::{from_raw_parts, from_raw_parts_mut};

pub mod nvgpu;
pub mod smp;
pub mod swap;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[repr(u8)]
pub enum Locus {
  // TODO TODO
  _Top = 0,
  Swap = 31,
  Mem  = 63,
  VMem = 127,
  _Bot = 255,
}

impl Locus {
  /*pub fn fastest() -> Locus {
    TL_PCTX.with(|pctx| pctx.fastest_locus())
  }*/

  pub fn max(self, rhs: Locus) -> Locus {
    max(self, rhs)
  }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[repr(u8)]
pub enum PMach {
  // TODO TODO
  _Top = 0,
  Smp,
  NvGpu,
  _Bot = 255,
}

#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct PMachSet {
  bits: u8,
}

impl Default for PMachSet {
  fn default() -> PMachSet {
    PMachSet{bits: 0}
  }
}

impl PMachSet {
  pub fn insert(&mut self, pm: PMach) {
    match pm {
      PMach::_Top => {}
      PMach::Smp => {
        self.bits |= 1;
      }
      PMach::NvGpu => {
        self.bits |= 2;
      }
      PMach::_Bot => panic!("bug")
    }
  }

  pub fn contains(&self, pm: PMach) -> bool {
    match pm {
      PMach::_Top => true,
      PMach::Smp => {
        (self.bits & 1) != 0
      }
      PMach::NvGpu => {
        (self.bits & 2) != 0
      }
      PMach::_Bot => panic!("bug")
    }
  }
}

#[derive(Clone, Copy, Debug)]
#[repr(u8)]
pub enum PMemErr {
  Oom,
  Bot,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct PAddr {
  pub bits: u64,
}

impl Debug for PAddr {
  fn fmt(&self, f: &mut Formatter) -> FmtResult {
    write!(f, "PAddr(0x{:x})", self.bits)
  }
}

impl PAddr {
  pub fn nil() -> PAddr {
    PAddr{bits: 0}
  }

  pub fn from_unchecked(bits: u64) -> PAddr {
    PAddr{bits}
  }

  pub fn to_unchecked(&self) -> u64 {
    self.bits
  }
}

/*#[derive(Clone, Copy)]
pub struct PAddrTabEntry {
  pub loc:  Locus,
  pub pm:   PMach,
  pub pin:  bool,
}*/

thread_local! {
  pub static TL_PCTX: PCtx = PCtx::new();
}

#[derive(Default)]
pub struct PCtxCtr {
  pub addr: Cell<u64>,
}

impl PCtxCtr {
  pub fn fresh_addr(&self) -> PAddr {
    let next = self.addr.get() + 1;
    assert!(next > 0);
    self.addr.set(next);
    PAddr{bits: next}
  }

  pub fn next_addr(&self) -> PAddr {
    let next = self.addr.get() + 1;
    assert!(next > 0);
    PAddr{bits: next}
  }

  pub fn peek_addr(&self) -> PAddr {
    let cur = self.addr.get();
    PAddr{bits: cur}
  }
}

#[derive(Clone, Default)]
pub struct TagUnifier {
  root: HashMap<u32, u32>,
}

impl TagUnifier {
  pub fn parse_tag(tag_s: &[u8]) -> Result<u32, ()> {
    for i in (0 .. tag_s.len()).rev() {
      let c = tag_s[i];
      if !(c >= b'0' && c <= b'9') {
        assert_eq!(c, b'_');
        return String::from_utf8_lossy(&tag_s[i + 1 .. ]).parse().map_err(|_| ());
      }
    }
    panic!("bug");
  }

  pub fn reset(&mut self) {
    self.root.clear();
  }

  pub fn find(&mut self, tag: u32) -> u32 {
    if !self.root.contains_key(&tag) {
      self.root.insert(tag, tag);
      return tag;
    }
    let mut cursor = tag;
    loop {
      let next = match self.root.get(&cursor) {
        None => panic!("bug"),
        Some(&t) => t
      };
      if cursor == next {
        return cursor;
      }
      cursor = next;
    }
  }

  pub fn unify(&mut self, ltag: u32, rtag: u32) {
    let ltag = self.find(ltag);
    if ltag == rtag {
      return;
    }
    let rtag = self.find(rtag);
    if ltag == rtag {
      return;
    }
    assert_eq!(self.root.insert(ltag, rtag), Some(ltag));
  }
}

pub struct PCtx {
  // TODO TODO
  pub ctr:      PCtxCtr,
  pub pmset:    PMachSet,
  pub lpmatrix: RevSortMap8<(Locus, PMach), ()>,
  pub plmatrix: RevSortMap8<(PMach, Locus), ()>,
  pub tagunify: RefCell<TagUnifier>,
  //pub addrtab:  RefCell<HashMap<PAddr, PAddrTabEntry>>,
  pub smp:      SmpPCtx,
  #[cfg(feature = "nvgpu")]
  pub nvgpu_ct: i32,
  #[cfg(feature = "nvgpu")]
  pub nvgpu:    Option<NvGpuPCtx>,
  /*pub nvgpu:    Vec<NvGpuPCtx>,*/
}

impl PCtx {
  pub fn new() -> PCtx {
    println!("DEBUG: PCtx::new");
    let mut pctx = PCtx{
      ctr:      PCtxCtr::default(),
      pmset:    PMachSet::default(),
      lpmatrix: RevSortMap8::default(),
      plmatrix: RevSortMap8::default(),
      tagunify: RefCell::new(TagUnifier::default()),
      smp:      SmpPCtx::new(),
      #[cfg(feature = "nvgpu")]
      nvgpu_ct: 0,
      #[cfg(feature = "nvgpu")]
      nvgpu:    None,
    };
    #[cfg(feature = "nvgpu")]
    {
      let gpu_ct = NvGpuPCtx::dev_count();
      pctx.nvgpu_ct = gpu_ct;
      for dev in 0 .. gpu_ct {
        pctx.nvgpu = NvGpuPCtx::new(dev);
        break;
      }
    }
    pctx.smp.append_matrix(&mut pctx.lpmatrix, &mut pctx.plmatrix);
    pctx.pmset.insert(PMach::Smp);
    #[cfg(feature = "nvgpu")]
    if let Some(gpu) = pctx.nvgpu.as_ref() {
      gpu.append_matrix(&mut pctx.lpmatrix, &mut pctx.plmatrix);
      pctx.pmset.insert(PMach::NvGpu);
    }
    pctx
  }

  pub fn fastest_locus(&self) -> Locus {
    let mut max_locus = Locus::_Top;
    max_locus = max_locus.max(self.smp.fastest_locus());
    if let Some(locus) = self.nvgpu.as_ref().map(|gpu| gpu.fastest_locus()) {
      max_locus = max_locus.max(locus);
    }
    max_locus
  }

  /*//pub fn try_alloc(&self, x: CellPtr, sz: usize, locus: Locus) -> Result<Option<Weak<dyn InnerCell_>>, PMemErr> {}
  pub fn try_alloc(&self, x: CellPtr, sz: usize, locus: Locus) -> Result<Option<Rc<dyn InnerCell_>>, PMemErr> {
    match self.lpmatrix.find_lub((locus, PMach::_Top)) {
      None => Ok(None),
      Some((key, _)) => {
        let pmach = key.key.as_ref().1;
        let ret = match pmach {
          PMach::Smp => {
            self.smp.try_alloc(x, sz, locus)
              //.map(|r| Some(Rc::downgrade(&r) as _))
              .map(|r| Some(r as _))
          }
          PMach::NvGpu => {
            self.nvgpu.as_ref().unwrap().try_alloc(x, sz, locus)
              //.map(|r| Some(Rc::downgrade(&r) as _))
              .map(|r| Some(r as _))
          }
          _ => panic!("bug")
        };
        ret
      }
    }
  }*/

  pub fn alloc(&self, ty: &CellType, locus: Locus, pmach: PMach) -> PAddr {
    match pmach {
      #[cfg(not(feature = "nvgpu"))]
      PMach::NvGpu => {
        unimplemented!();
      }
      #[cfg(feature = "nvgpu")]
      PMach::NvGpu => {
        let addr = match self.nvgpu.as_ref().unwrap().try_alloc(&self.ctr, locus, ty) {
          Err(e) => {
            println!("BUG:   PCtx::alloc: unimplemented error: {:?}", e);
            panic!();
          }
          Ok(addr) => addr
        };
        addr
      }
      _ => {
        println!("DEBUG: PCtx::alloc: {:?}", &self.lpmatrix);
        println!("DEBUG: PCtx::alloc: {:?}", &self.plmatrix);
        println!("BUG:   PCtx::alloc: unimplemented: locus={:?} pmach={:?}", locus, pmach);
        panic!();
      }
    }
  }

  pub fn alloc_loc(&self, ty: &CellType, locus: Locus) -> (PMach, PAddr) {
    match self.lpmatrix.find_lub((locus, PMach::_Bot)) {
      None => {
        println!("BUG:   PCtx::alloc_loc: failed to alloc: locus={:?}", locus);
        panic!();
      }
      Some((key, _)) => {
        let pmach = key.key.as_ref().1;
        let addr = self.alloc(ty, locus, pmach);
        (pmach, addr)
      }
    }
  }

  pub fn lookup(&self, addr: PAddr) -> Option<(Locus, PMach, Rc<dyn InnerCell_>)> {
    // FIXME
    #[cfg(feature = "nvgpu")]
    if let Some(gpu) = self.nvgpu.as_ref() {
      let pm = PMach::NvGpu;
      match gpu.lookup(addr) {
        None => {}
        Some((loc, icel)) => {
          return Some((loc, pm, icel));
        }
      }
    }
    // TODO
    None
  }

  pub fn lookup_pm(&self, pmach: PMach, addr: PAddr) -> Option<(Locus, Rc<dyn InnerCell_>)> {
    match pmach {
      #[cfg(not(feature = "nvgpu"))]
      PMach::NvGpu => {
        unimplemented!();
      }
      #[cfg(feature = "nvgpu")]
      PMach::NvGpu => {
        if let Some(gpu) = self.nvgpu.as_ref() {
          let pm = PMach::NvGpu;
          match gpu.lookup(addr) {
            None => {}
            Some((loc, icel)) => {
              return Some((loc, icel));
            }
          }
        }
      }
      _ => {
        unimplemented!();
      }
    }
    None
  }

  pub fn hard_copy(&self, dst_loc: Locus, dst_pm: PMach, dst: PAddr, src_loc: Locus, src_pm: PMach, src: PAddr) {
    match (dst_pm, src_pm) {
      #[cfg(feature = "nvgpu")]
      (PMach::NvGpu, PMach::NvGpu) => {
        self.nvgpu.as_ref().unwrap().hard_copy(dst_loc, dst, src_loc, src)
      }
      _ => {
        panic!("bug: PCtx::hard_copy: unimplemented: dst pm={:?} src pm={:?}", dst_pm, src_pm)
      }
    }
  }

  pub fn set_root(&self, addr: PAddr, new_root: CellPtr) -> Option<CellPtr> {
    // FIXME
    match self.lookup(addr) {
      None => panic!("bug"),
      Some((_, _, icel)) => {
        let oroot = InnerCell_::root(&*icel);
        InnerCell_::set_root(&*icel, Some(new_root));
        oroot
      }
    }
  }

  pub fn unset_root(&self, addr: PAddr) -> Option<CellPtr> {
    // FIXME
    match self.lookup(addr) {
      None => panic!("bug"),
      Some((_, _, icel)) => {
        let oroot = InnerCell_::root(&*icel);
        InnerCell_::set_root(&*icel, None);
        oroot
      }
    }
  }

  pub fn set_pin(&self, addr: PAddr) -> bool {
    // FIXME
    match self.lookup(addr) {
      None => panic!("bug"),
      Some((_, _, icel)) => {
        let opin = InnerCell_::pin(&*icel);
        InnerCell_::set_pin(&*icel, true);
        opin
      }
    }
  }

  pub fn unset_pin(&self, addr: PAddr) -> bool {
    // FIXME
    match self.lookup(addr) {
      None => panic!("bug"),
      Some((_, _, icel)) => {
        let opin = InnerCell_::pin(&*icel);
        InnerCell_::set_pin(&*icel, false);
        opin
      }
    }
  }
}

pub trait PCtxImpl {
  // FIXME: don't need this associated type.
  //type ICel: InnerCell;

  fn pmach(&self) -> PMach;
  fn fastest_locus(&self) -> Locus;
  fn append_matrix(&self, lp: &mut RevSortMap8<(Locus, PMach), ()>, pl: &mut RevSortMap8<(PMach, Locus), ()>);
  //fn try_alloc(&self, x: CellPtr, sz: usize, pmset: PMachSet) -> Result<Rc<Self::ICel>, PMemErr>;
  //fn try_alloc(&self, x: CellPtr, sz: usize, locus: Locus) -> Result<Rc<dyn InnerCell_>, PMemErr>;
  fn try_alloc(&self, pctr: &PCtxCtr, locus: Locus, ty: &CellType) -> Result<PAddr, PMemErr>;
  //fn try_alloc(&self, pctr: &PCtxCtr, ty: &CellType, locus: Locus) -> Result<PAddr, PMemErr>;
  //fn lookup(&self, addr: PAddr) -> Option<()>;
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct MemReg {
  pub ptr:  *mut c_void,
  pub sz:   usize,
}

impl MemReg {
  #[track_caller]
  pub fn copy_from_slice<T: DtypeConstExt + Copy/*, Buf: Borrow<[T]>*/>(&self, src_buf: &[T]) {
    panick_wrap(|| self._copy_from_slice(src_buf))
  }

  pub fn _copy_from_slice<T: DtypeConstExt + Copy/*, Buf: Borrow<[T]>*/>(&self, src_buf: &[T]) {
    //let src_buf = src_buf.borrow();
    let src_len = src_buf.len();
    let dsz = <T as DtypeConstExt>::dtype().size_bytes();
    let src_sz = dsz * src_len;
    assert_eq!(self.sz, src_sz);
    let src_start = src_buf.as_ptr() as usize;
    let src_end = src_start + src_sz;
    let dst_start = self.ptr as usize;
    let dst_end = dst_start + self.sz;
    if !(src_end <= dst_start || dst_end <= src_start) {
      panic!("bug: MemReg::_copy_from: overlapping src and dst");
    }
    unsafe {
      std::intrinsics::copy_nonoverlapping(src_buf.as_ptr() as *const u8, self.ptr as *mut u8, self.sz);
    }
  }

  #[track_caller]
  pub fn copy_from_reader<R: Read>(&self, src: R) {
    panick_wrap(|| self._copy_from_reader(src))
  }

  pub fn _copy_from_reader<R: Read>(&self, mut src: R) {
    let dst_buf = unsafe { from_raw_parts_mut(self.ptr as *mut u8, self.sz) };
    let mut dst_off = 0;
    loop {
      match src.read(&mut dst_buf[dst_off .. ]) {
        Err(_) => panic!("ERROR: I/O error"),
        Ok(0) => break,
        Ok(n) => {
          dst_off += n;
        }
      }
    }
  }

  pub fn _debug_dump_f32(&self) {
    let len = self.sz / 4;
    assert_eq!(0, self.sz % 4);
    assert_eq!(0, (self.ptr as usize) % align_of::<f32>());
    let buf = unsafe { from_raw_parts(self.ptr as *mut u8 as *const u8 as *const f32, len) };
    let start = 0;
    print!("DEBUG: MemReg: {:08x} :", start * 4);
    for i in start .. min(start + 8, len) {
      let x = buf[i];
      print!(" {}", x);
    }
    println!();
    if len <= 0 {
      return;
    }
    let start = (len - 1) - ((len - 1) & (8 - 1));
    print!("DEBUG: MemReg: {:08x} :", start * 4);
    for i in start .. min(start + 8, len) {
      let x = buf[i];
      print!(" {}", x);
    }
    println!();
  }

  pub fn _debug_dump_f16(&self) {
    let len = self.sz / 2;
    assert_eq!(0, self.sz % 2);
    assert_eq!(0, (self.ptr as usize) % align_of::<u16>());
    let buf = unsafe { from_raw_parts(self.ptr as *mut u8 as *const u8 as *const u16, len) };
    let start = 0;
    print!("DEBUG: MemReg: {:08x} :", start * 2);
    for i in start .. min(start + 8, len) {
      let x = f16::from_bits(buf[i]);
      print!(" {}", x);
    }
    println!();
    if len <= 0 {
      return;
    }
    let start = (len - 1) - ((len - 1) & (8 - 1));
    print!("DEBUG: MemReg: {:08x} :", start * 2);
    for i in start .. min(start + 8, len) {
      let x = f16::from_bits(buf[i]);
      print!(" {}", x);
    }
    println!();
  }
}
