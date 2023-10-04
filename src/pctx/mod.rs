use self::nvgpu::{NvGpuPCtx};
use self::smp::{SmpPCtx};
use self::swap::{SwapPCtx};
use crate::algo::{HashMap, RevSortMap8};
use crate::algo::fp::*;
use crate::cell::{CellPtr, CellType, DtypeConstExt};
use crate::panick::*;
use cacti_cfg_env::*;

use smol_str::{SmolStr};

use std::any::{Any};
use std::borrow::{Borrow};
use std::cell::{Cell, RefCell};
use std::cmp::{max, min};
use std::convert::{TryFrom, TryInto};
use std::ffi::{c_void};
use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::io::{Read};
use std::mem::{align_of};
use std::ops::{Deref, DerefMut};
use std::rc::{Rc};
use std::slice::{from_raw_parts, from_raw_parts_mut};
use std::str::{FromStr};

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

impl<'a> TryFrom<&'a str> for Locus {
  type Error = SmolStr;

  fn try_from(s: &'a str) -> Result<Locus, SmolStr> {
    Locus::from_str(s)
  }
}

impl FromStr for Locus {
  type Err = SmolStr;

  fn from_str(s: &str) -> Result<Locus, SmolStr> {
    Ok(match s {
      "mem" => Locus::Mem,
      "vmem" => Locus::VMem,
      _ => return Err(s.into())
    })
  }
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
  Swap,
  Smp,
  NvGpu,
  _Bot = 255,
}

impl<'a> TryFrom<&'a str> for PMach {
  type Error = SmolStr;

  fn try_from(s: &'a str) -> Result<PMach, SmolStr> {
    PMach::from_str(s)
  }
}

impl FromStr for PMach {
  type Err = SmolStr;

  fn from_str(s: &str) -> Result<PMach, SmolStr> {
    Ok(match s {
      "swap" => PMach::Swap,
      "smp" => PMach::Smp,
      "gpu" => PMach::NvGpu,
      "nvgpu" => PMach::NvGpu,
      _ => return Err(s.into())
    })
  }
}

/*#[derive(Clone, Copy)]
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
}*/

#[derive(Clone, Copy, Debug)]
#[repr(u8)]
pub enum PMemErr {
  Oom = 2,
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

  pub fn is_nil(&self) -> bool {
    self.bits == 0
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
  //pub pmset:    PMachSet,
  pub lpmatrix: RevSortMap8<(Locus, PMach), ()>,
  pub plmatrix: RevSortMap8<(PMach, Locus), ()>,
  pub tagunify: RefCell<TagUnifier>,
  //pub addrtab:  RefCell<HashMap<PAddr, PAddrTabEntry>>,
  pub swap:     SwapPCtx,
  pub smp:      SmpPCtx,
  #[cfg(feature = "nvgpu")]
  pub nvgpu_ct: i32,
  #[cfg(feature = "nvgpu")]
  pub nvgpu:    Option<NvGpuPCtx>,
  /*pub nvgpu:    Vec<NvGpuPCtx>,*/
}

impl PCtx {
  pub fn new() -> PCtx {
    if cfg_debug() { println!("DEBUG: PCtx::new"); }
    let mut pctx = PCtx{
      ctr:      PCtxCtr::default(),
      //pmset:    PMachSet::default(),
      lpmatrix: RevSortMap8::default(),
      plmatrix: RevSortMap8::default(),
      tagunify: RefCell::new(TagUnifier::default()),
      swap:     SwapPCtx::new(),
      smp:      SmpPCtx::new(),
      #[cfg(feature = "nvgpu")]
      nvgpu_ct: 0,
      #[cfg(feature = "nvgpu")]
      nvgpu:    None,
    };
    #[cfg(feature = "nvgpu")]
    {
      // FIXME: multi-gpu.
      let gpu_ct = NvGpuPCtx::dev_count();
      pctx.nvgpu_ct = gpu_ct;
      for dev in 0 .. gpu_ct {
        pctx.nvgpu = NvGpuPCtx::new(dev);
        break;
      }
    }
    pctx.swap.append_matrix(&mut pctx.lpmatrix, &mut pctx.plmatrix);
    pctx.smp.append_matrix(&mut pctx.lpmatrix, &mut pctx.plmatrix);
    //pctx.pmset.insert(PMach::Smp);
    #[cfg(feature = "nvgpu")]
    if let Some(gpu) = pctx.nvgpu.as_ref() {
      gpu.append_matrix(&mut pctx.lpmatrix, &mut pctx.plmatrix);
      //pctx.pmset.insert(PMach::NvGpu);
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

  #[cfg(not(feature = "nvgpu"))]
  pub fn fastest_pmach(&self) -> PMach {
    PMach::Smp
  }

  #[cfg(feature = "nvgpu")]
  pub fn fastest_pmach(&self) -> PMach {
    PMach::NvGpu
  }

  pub fn alloc(&self, ty: &CellType, locus: Locus, pmach: PMach) -> PAddr {
    match pmach {
      PMach::Swap => {
        // FIXME: this can only alloc cow.
        unimplemented!();
        /*let addr = match self.swap.try_alloc(&self.ctr, ty, locus) {
          Err(e) => {
            println!("BUG:   PCtx::alloc: unimplemented error: {:?}", e);
            panic!();
          }
          Ok(addr) => addr
        };
        addr*/
      }
      PMach::Smp => {
        // FIXME
        unimplemented!();
      }
      #[cfg(not(feature = "nvgpu"))]
      PMach::NvGpu => {
        unimplemented!();
      }
      #[cfg(feature = "nvgpu")]
      PMach::NvGpu => {
        let gpu = self.nvgpu.as_ref().unwrap();
        let mut retry = false;
        'retry: loop {
          let addr = match gpu.try_alloc(&self.ctr, ty, locus) {
            Err(PMemErr::Oom) => {
              if retry {
                println!("ERROR:  PCtx::alloc: unrecoverable out-of-memory failure (nvgpu device memory)");
                panic!();
              }
              if locus == Locus::VMem {
                let req_sz: usize = ty.packed_span_bytes().try_into().unwrap();
                if gpu.mem_pool._try_soft_oom(req_sz).is_none() {
                  println!("ERROR:  PCtx::alloc: out-of-memory, soft oom failure (nvgpu device memory): req sz={:?}", req_sz);
                  panic!();
                }
              }
              retry = true;
              continue 'retry;
            }
            Err(e) => {
              println!("ERROR:  PCtx::alloc: unimplemented error: {:?} (locus={:?} pmach={:?})",
                  e, locus, PMach::NvGpu);
              panic!();
            }
            Ok(addr) => addr
          };
          return addr
        }
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

  pub fn live(&self, addr: PAddr) -> bool {
    if self.swap.live(addr) {
      return true;
    }
    #[cfg(feature = "nvgpu")]
    if let Some(gpu) = self.nvgpu.as_ref() {
      if gpu.live(addr) {
        return true;
      }
    }
    // TODO
    false
  }

  pub fn get_ref(&self, addr: PAddr) -> Option<u32> {
    match self.lookup(addr) {
      Some((_, _, icel)) => Some(icel.get_ref()),
      _ => None
    }
  }

  pub fn retain(&self, addr: PAddr) {
    self.swap.retain(addr);
    #[cfg(feature = "nvgpu")]
    if let Some(gpu) = self.nvgpu.as_ref() {
      gpu.retain(addr);
    }
    // TODO
  }

  pub fn get_pin(&self, addr: PAddr) -> Option<u16> {
    match self.lookup(addr) {
      Some((_, _, icel)) => Some(icel.get_pin()),
      _ => None
    }
  }

  pub fn pin(&self, addr: PAddr) {
    self.swap.pin(addr);
    #[cfg(feature = "nvgpu")]
    if let Some(gpu) = self.nvgpu.as_ref() {
      gpu.pin(addr);
    }
    // TODO
  }

  pub fn pinned(&self, addr: PAddr) -> bool {
    if self.swap.pinned(addr) {
      return true;
    }
    #[cfg(feature = "nvgpu")]
    if let Some(gpu) = self.nvgpu.as_ref() {
      if gpu.pinned(addr) {
        return true;
      }
    }
    // TODO
    false
  }

  pub fn release(&self, addr: PAddr) -> Option<(Locus, PMach, Rc<dyn InnerCell_>)> {
    {
      let pm = PMach::Swap;
      match self.swap.release(addr) {
        None => {}
        Some((loc, icel)) => {
          return Some((loc, pm, icel));
        }
      }
    }
    #[cfg(feature = "nvgpu")]
    if let Some(gpu) = self.nvgpu.as_ref() {
      let pm = PMach::NvGpu;
      match gpu.release(addr) {
        None => {}
        Some((loc, icel)) => {
          return Some((loc, pm, icel));
        }
      }
    }
    // TODO
    None
  }

  pub fn unpin(&self, addr: PAddr) -> Option<(Locus, PMach, Rc<dyn InnerCell_>)> {
    {
      let pm = PMach::Swap;
      match self.swap.unpin(addr) {
        None => {}
        Some((loc, icel)) => {
          return Some((loc, pm, icel));
        }
      }
    }
    #[cfg(feature = "nvgpu")]
    if let Some(gpu) = self.nvgpu.as_ref() {
      let pm = PMach::NvGpu;
      match gpu.unpin(addr) {
        None => {}
        Some((loc, icel)) => {
          return Some((loc, pm, icel));
        }
      }
    }
    // TODO
    None
  }

  pub fn yeet(&self, addr: PAddr) -> Option<(Locus, PMach, Rc<dyn InnerCell_>)> {
    if self.pinned(addr) {
      return self.release(addr);
    }
    self.force_yeet(addr)
  }

  pub fn force_yeet(&self, addr: PAddr) -> Option<(Locus, PMach, Rc<dyn InnerCell_>)> {
    {
      let pm = PMach::Swap;
      match self.swap.yeet(addr) {
        None => {}
        Some((loc, icel)) => {
          //assert!(icel.invalid());
          return Some((loc, pm, icel));
        }
      }
    }
    #[cfg(feature = "nvgpu")]
    if let Some(gpu) = self.nvgpu.as_ref() {
      let pm = PMach::NvGpu;
      match gpu.yeet(addr) {
        None => {}
        Some((loc, icel)) => {
          //assert!(icel.invalid());
          return Some((loc, pm, icel));
        }
      }
    }
    // TODO
    None
  }

  pub fn lookup(&self, addr: PAddr) -> Option<(Locus, PMach, Rc<dyn InnerCell_>)> {
    {
      let pm = PMach::Swap;
      match self.swap.lookup(addr) {
        None => {}
        Some((loc, icel)) => {
          assert!(!InnerCell_::invalid(&*icel));
          return Some((loc, pm, icel));
        }
      }
    }
    #[cfg(feature = "nvgpu")]
    if let Some(gpu) = self.nvgpu.as_ref() {
      let pm = PMach::NvGpu;
      match gpu.lookup(addr) {
        None => {}
        Some((loc, icel)) => {
          assert!(!InnerCell_::invalid(&*icel));
          return Some((loc, pm, icel));
        }
      }
    }
    // TODO
    None
  }

  pub fn lookup_pm(&self, pmach: PMach, addr: PAddr) -> Option<(Locus, Rc<dyn InnerCell_>)> {
    match pmach {
      PMach::Swap => {
        match self.swap.lookup(addr) {
          None => {}
          Some((loc, icel)) => {
            assert!(!InnerCell_::invalid(&*icel));
            return Some((loc, icel));
          }
        }
      }
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
              assert!(!InnerCell_::invalid(&*icel));
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

  pub fn lookup_mem_reg_(&self, addr: PAddr) -> Option<(MemReg, PMach, Rc<dyn InnerCell_>)> {
    {
      let pm = PMach::Swap;
      match self.swap.lookup_mem_reg_(addr) {
        None => {}
        Some((reg, icel)) => {
          assert!(!InnerCell_::invalid(&*icel));
          return Some((reg, pm, icel));
        }
      }
    }
    #[cfg(feature = "nvgpu")]
    if let Some(gpu) = self.nvgpu.as_ref() {
      let pm = PMach::NvGpu;
      match gpu.lookup_mem_reg_(addr) {
        None => {}
        Some((reg, icel)) => {
          assert!(!InnerCell_::invalid(&*icel));
          return Some((reg, pm, icel));
        }
      }
    }
    // TODO
    None
  }

  pub fn hard_copy(&self, dst_loc: Locus, dst_pm: PMach, dst: PAddr, src_loc: Locus, src_pm: PMach, src: PAddr, sz: usize) {
    match (dst_loc, dst_pm, src_loc, src_pm) {
      #[cfg(feature = "nvgpu")]
      (Locus::Mem, PMach::NvGpu, Locus::Mem, PMach::Swap) => {
        let gpu = self.nvgpu.as_ref().unwrap();
        let dst_reg = gpu.lookup_mem_reg(dst).unwrap();
        assert!(sz <= dst_reg.sz);
        let src_reg = self.swap.lookup_mem_reg(src).unwrap();
        assert!(sz <= src_reg.sz);
        // FIXME: check we can copy nonoverlapping here.
        /*unsafe {
          std::intrinsics::copy_nonoverlapping(src_reg.ptr as *const u8, dst_reg.ptr as *mut u8, sz);
        }*/
        self.smp.th_pool.memcpy(dst_reg.ptr as *mut u8, src_reg.ptr as *const u8, sz);
      }
      #[cfg(feature = "nvgpu")]
      (Locus::VMem, PMach::NvGpu, Locus::Mem, PMach::Swap) => {
        let gpu = self.nvgpu.as_ref().unwrap();
        let (dst_dptr, dst_sz) = gpu.lookup_dev(dst).unwrap();
        assert!(sz <= dst_sz);
        let src_reg = self.swap.lookup_mem_reg(src).unwrap();
        assert!(sz <= src_reg.sz);
        self.nvgpu.as_ref().unwrap().hard_copy_raw_mem_to_vmem(dst_dptr, src_reg.ptr, sz)
      }
      #[cfg(feature = "nvgpu")]
      (_, PMach::NvGpu, _, PMach::NvGpu) => {
        self.nvgpu.as_ref().unwrap().hard_copy(dst_loc, dst, src_loc, src, sz)
      }
      _ => {
        panic!("bug: PCtx::hard_copy: unimplemented: dst loc={:?} pm={:?} src loc={:?} pm={:?}",
            dst_loc, dst_pm, src_loc, src_pm)
      }
    }
  }

  pub fn lookup_root(&self, addr: PAddr) -> Option<CellPtr> {
    // FIXME
    match self.lookup(addr) {
      None => panic!("bug"),
      Some((_, _, icel)) => {
        InnerCell_::root(&*icel)
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
      None => {
        println!("WARNING:PCtx::unset_root: failed to lookup addr={:?}", addr);
        //panic!("bug");
        None
      }
      Some((_, _, icel)) => {
        let oroot = InnerCell_::root(&*icel);
        InnerCell_::set_root(&*icel, None);
        oroot
      }
    }
  }

  pub fn try_lookup_cow(&self, addr: PAddr) -> Option<bool> {
    // FIXME
    match self.lookup(addr) {
      None => None,
      Some((_, _, icel)) => {
        let ocow = InnerCell_::cow(&*icel);
        Some(ocow)
      }
    }
  }

  pub fn lookup_cow(&self, addr: PAddr) -> bool {
    // FIXME
    match self.lookup(addr) {
      None => panic!("bug"),
      Some((_, _, icel)) => {
        let ocow = InnerCell_::cow(&*icel);
        ocow
      }
    }
  }

  pub fn set_cow(&self, addr: PAddr) -> bool {
    // FIXME
    match self.lookup(addr) {
      None => panic!("bug"),
      Some((_, _, icel)) => {
        let ocow = InnerCell_::cow(&*icel);
        InnerCell_::set_cow(&*icel, true);
        ocow
      }
    }
  }

  pub fn unset_cow(&self, addr: PAddr) -> bool {
    // FIXME
    match self.lookup(addr) {
      None => panic!("bug"),
      Some((_, _, icel)) => {
        let ocow = InnerCell_::cow(&*icel);
        InnerCell_::set_cow(&*icel, false);
        ocow
      }
    }
  }

  /*pub fn lookup_pin(&self, addr: PAddr) -> bool {
    // FIXME
    match self.lookup(addr) {
      None => panic!("bug"),
      Some((_, _, icel)) => {
        let opin = InnerCell_::pin(&*icel);
        opin
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
  }*/
}

pub trait PCtxImpl {
  fn pmach(&self) -> PMach;
  fn fastest_locus(&self) -> Locus;
  fn append_matrix(&self, lp: &mut RevSortMap8<(Locus, PMach), ()>, pl: &mut RevSortMap8<(PMach, Locus), ()>);
  //fn try_alloc(&self, x: CellPtr, sz: usize, pmset: PMachSet) -> Result<Rc<Self::ICel>, PMemErr>;
  //fn try_alloc(&self, x: CellPtr, sz: usize, locus: Locus) -> Result<Rc<dyn InnerCell_>, PMemErr>;
  //fn try_alloc(&self, pctr: &PCtxCtr, locus: Locus, ty: &CellType) -> Result<PAddr, PMemErr>;
  fn try_alloc(&self, pctr: &PCtxCtr, ty: &CellType, locus: Locus) -> Result<PAddr, PMemErr>;
  //fn lookup(&self, addr: PAddr) -> Option<()>;
}

pub type MemReg = UnsafeMemReg;

#[derive(Clone, Copy)]
#[repr(C)]
pub struct UnsafeMemReg {
  ptr:  *mut c_void,
  sz:   usize,
}

impl UnsafeMemReg {
  pub unsafe fn _from_raw_parts(ptr: *mut c_void, sz: usize) -> UnsafeMemReg {
    UnsafeMemReg{ptr, sz}
  }

  pub unsafe fn _as_bytes<'b>(self) -> &'b [u8] {
    assert!(!self.ptr.is_null());
    unsafe { from_raw_parts(self.ptr as *const u8, self.sz) }
  }

  pub fn slice(self, start_bytes: usize, end_bytes: usize) -> UnsafeMemReg {
    let base = self.ptr as usize;
    let end_base = base + self.sz;
    let slice_base = base + start_bytes;
    let slice_end_base = base + end_bytes;
    let slice_sz = slice_end_base - slice_base;
    let slice = UnsafeMemReg{ptr: slice_base as *mut _, sz: slice_sz};
    assert!(slice.is_subregion(&self));
    slice
  }

  pub fn as_ptr(&self) -> *mut c_void {
    self.ptr
  }

  pub fn size_bytes(&self) -> usize {
    self.sz
  }

  pub fn is_subregion(&self, other: &UnsafeMemReg) -> bool {
    let src = self.ptr as usize;
    let end_src = src + self.sz;
    assert!(src <= end_src);
    let dst = other.ptr as usize;
    let end_dst = dst + other.sz;
    assert!(dst <= end_dst);
    dst <= src && end_src <= end_dst
  }

  #[track_caller]
  pub fn copy_from_slice<T: DtypeConstExt + Copy>(&self, src_buf: &[T]) {
    panick_wrap(|| self._copy_from_slice(src_buf))
  }

  pub fn _copy_from_slice<T: DtypeConstExt + Copy>(&self, src_buf: &[T]) {
    //let src_buf = src_buf.borrow();
    let src_len = src_buf.len();
    let dsz = <T as DtypeConstExt>::dtype_().size_bytes();
    let src_sz = dsz * src_len;
    assert_eq!(self.sz, src_sz);
    let src_start = src_buf.as_ptr() as usize;
    let src_end = src_start + src_sz;
    let dst_start = self.ptr as usize;
    let dst_end = dst_start + self.sz;
    if !(src_end <= dst_start || dst_end <= src_start) {
      panic!("bug: UnsafeMemReg::_copy_from_slice: overlapping src and dst");
    }
    assert!(!self.ptr.is_null());
    unsafe {
      std::intrinsics::copy_nonoverlapping(src_buf.as_ptr() as *const u8, self.ptr as *mut u8, self.sz);
    }
  }

  #[track_caller]
  pub fn copy_from_bytes(&self, src_buf: &[u8]) {
    panick_wrap(|| self._copy_from_bytes(src_buf))
  }

  pub fn _copy_from_bytes(&self, src_buf: &[u8]) {
    let src_sz = src_buf.len();
    assert_eq!(self.sz, src_sz);
    let src_start = src_buf.as_ptr() as usize;
    let src_end = src_start + src_sz;
    let dst_start = self.ptr as usize;
    let dst_end = dst_start + self.sz;
    if !(src_end <= dst_start || dst_end <= src_start) {
      panic!("bug: UnsafeMemReg::_copy_from_bytes: overlapping src and dst");
    }
    assert!(!self.ptr.is_null());
    unsafe {
      std::intrinsics::copy_nonoverlapping(src_buf.as_ptr() as *const u8, self.ptr as *mut u8, self.sz);
    }
  }

  #[track_caller]
  pub fn copy_from_reader<R: Read>(&self, src: R) {
    panick_wrap(|| self._copy_from_reader(src))
  }

  pub fn _copy_from_reader<R: Read>(&self, mut src: R) {
    assert!(!self.ptr.is_null());
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

  pub fn _as_slice_f32(&self) -> &[f32] {
    assert!(!self.ptr.is_null());
    assert_eq!(self.ptr as usize % 4, 0);
    unsafe { from_raw_parts(self.ptr as *const f32, self.sz / 4) }
  }

  pub fn _as_slice_i64(&self) -> &[i64] {
    assert!(!self.ptr.is_null());
    assert_eq!(self.ptr as usize % 8, 0);
    unsafe { from_raw_parts(self.ptr as *const i64, self.sz / 8) }
  }

  pub fn _as_slice_u16(&self) -> &[u16] {
    assert!(!self.ptr.is_null());
    assert_eq!(self.ptr as usize % 2, 0);
    unsafe { from_raw_parts(self.ptr as *const u16, self.sz / 2) }
  }

  /*pub fn _debug_dump_f32(&self) {
    let len = self.sz / 4;
    assert_eq!(0, self.sz % 4);
    assert_eq!(0, (self.ptr as usize) % align_of::<f32>());
    let buf = unsafe { from_raw_parts(self.ptr as *mut u8 as *const u8 as *const f32, len) };
    let start = 0;
    print!("DEBUG: UnsafeMemReg: {:08x} :", start * 4);
    for i in start .. min(start + 8, len) {
      let x = buf[i];
      print!(" {:+e}", x);
    }
    println!();
    if len <= 0 {
      return;
    }
    let start = (len - 1) - ((len - 1) & (8 - 1));
    print!("DEBUG: UnsafeMemReg: {:08x} :", start * 4);
    for i in start .. min(start + 8, len) {
      let x = buf[i];
      print!(" {:+e}", x);
    }
    println!();
  }

  pub fn _debug_dump_f16(&self) {
    let len = self.sz / 2;
    assert_eq!(0, self.sz % 2);
    assert_eq!(0, (self.ptr as usize) % align_of::<u16>());
    let buf = unsafe { from_raw_parts(self.ptr as *mut u8 as *const u8 as *const u16, len) };
    let start = 0;
    print!("DEBUG: UnsafeMemReg: {:08x} :", start * 2);
    for i in start .. min(start + 8, len) {
      let x = f16::from_bits(buf[i]);
      print!(" {:+e}", x);
    }
    println!();
    if len <= 0 {
      return;
    }
    let start = (len - 1) - ((len - 1) & (8 - 1));
    print!("DEBUG: UnsafeMemReg: {:08x} :", start * 2);
    for i in start .. min(start + 8, len) {
      let x = f16::from_bits(buf[i]);
      print!(" {:+e}", x);
    }
    println!();
  }*/
}

pub trait InnerCell {
  // TODO
  fn invalid(&self) -> bool { unimplemented!(); }
  fn invalidate(&self) { unimplemented!(); }
  //fn try_borrow(&self) -> () { unimplemented!(); }
  //fn try_borrow_mut(&self) -> () { unimplemented!(); }
  unsafe fn as_unsafe_mem_reg(&self) -> Option<UnsafeMemReg> { None }
  //fn as_reg(&self) -> Option<MemReg> { self.as_mem_reg() }
  fn size(&self) -> usize { unimplemented!(); }
  fn root(&self) -> Option<CellPtr> { unimplemented!(); }
  fn set_root(&self, _root: Option<CellPtr>) { unimplemented!(); }
  fn cow(&self) -> bool { unimplemented!(); }
  fn set_cow(&self, _flag: bool) { unimplemented!(); }
  //fn pin(&self) -> bool { unimplemented!(); }
  //fn set_pin(&self, _flag: bool) { unimplemented!(); }
  fn tag(&self) -> Option<u32> { unimplemented!(); }
  fn set_tag(&self, _tag: Option<u32>) { unimplemented!(); }
  fn get_ref(&self) -> u32 { unimplemented!(); }
  fn live(&self) -> bool { unimplemented!(); }
  fn retain(&self) { unimplemented!(); }
  fn release(&self) { unimplemented!(); }
  fn get_pin(&self) -> u16 { unimplemented!(); }
  fn pinned(&self) -> bool { unimplemented!(); }
  fn pin(&self) { unimplemented!(); }
  fn unpin(&self) { unimplemented!(); }
  fn locked(&self) -> bool { unimplemented!(); }
  fn lock(&self) { unimplemented!(); }
  fn unlock(&self) { unimplemented!(); }
  fn _try_borrow(&self) -> Result<(), BorrowErr> { Err(BorrowErr::NotImpl) }
  fn _try_borrow_unsafe(&self) -> Result<(), BorrowErr> { Err(BorrowErr::NotImpl) }
  fn _unborrow(&self) { unimplemented!(); }
  fn _try_borrow_mut(&self) -> Result<(), BorrowErr> { Err(BorrowErr::NotImpl) }
  fn _unborrow_mut(&self) { unimplemented!(); }
  fn mem_borrow(&self) -> Result<BorrowRef<[u8]>, BorrowErr> { Err(BorrowErr::NotImpl) }
  fn mem_borrow_mut(&self) -> Result<BorrowRefMut<[u8]>, BorrowErr> { Err(BorrowErr::NotImpl) }
}

pub trait InnerCell_ {
  fn as_any(&self) -> &dyn Any;
  // TODO
  fn invalid(&self) -> bool;
  fn invalidate(&self);
  unsafe fn as_unsafe_mem_reg(&self) -> Option<UnsafeMemReg>;
  //fn as_reg(&self) -> Option<MemReg> { self.as_mem_reg() }
  fn size(&self) -> usize;
  fn root(&self) -> Option<CellPtr>;
  fn set_root(&self, _root: Option<CellPtr>);
  fn cow(&self) -> bool;
  fn set_cow(&self, _flag: bool);
  //fn pin(&self) -> bool;
  //fn set_pin(&self, _flag: bool);
  fn tag(&self) -> Option<u32>;
  fn set_tag(&self, _tag: Option<u32>);
  fn live(&self) -> bool;
  fn get_ref(&self) -> u32;
  fn retain(&self);
  fn release(&self);
  fn get_pin(&self) -> u16;
  fn pinned(&self) -> bool;
  fn pin(&self);
  fn unpin(&self);
  fn _try_borrow(&self) -> Result<(), BorrowErr>;
  fn _try_borrow_unsafe(&self) -> Result<(), BorrowErr>;
  fn _unborrow(&self);
  fn mem_borrow(&self) -> Result<BorrowRef<[u8]>, BorrowErr>;
  fn mem_borrow_mut(&self) -> Result<BorrowRefMut<[u8]>, BorrowErr>;
}

impl<C: InnerCell + Any> InnerCell_ for C {
  fn as_any(&self) -> &dyn Any {
    self
  }

  fn invalid(&self) -> bool {
    InnerCell::invalid(self)
  }

  fn invalidate(&self) {
    InnerCell::invalidate(self)
  }

  unsafe fn as_unsafe_mem_reg(&self) -> Option<UnsafeMemReg> {
    InnerCell::as_unsafe_mem_reg(self)
  }

  fn size(&self) -> usize {
    InnerCell::size(self)
  }

  fn root(&self) -> Option<CellPtr> {
    InnerCell::root(self)
  }

  fn set_root(&self, root: Option<CellPtr>) {
    InnerCell::set_root(self, root)
  }

  fn cow(&self) -> bool {
    InnerCell::cow(self)
  }

  fn set_cow(&self, flag: bool) {
    InnerCell::set_cow(self, flag)
  }

  /*fn pin(&self) -> bool {
    InnerCell::pin(self)
  }

  fn set_pin(&self, flag: bool) {
    InnerCell::set_pin(self, flag)
  }*/

  fn tag(&self) -> Option<u32> {
    InnerCell::tag(self)
  }

  fn set_tag(&self, tag: Option<u32>) {
    InnerCell::set_tag(self, tag)
  }

  fn live(&self) -> bool {
    InnerCell::live(self)
  }

  fn get_ref(&self) -> u32 {
    InnerCell::get_ref(self)
  }

  fn retain(&self) {
    InnerCell::retain(self)
  }

  fn release(&self) {
    InnerCell::release(self)
  }

  fn get_pin(&self) -> u16 {
    InnerCell::get_pin(self)
  }

  fn pinned(&self) -> bool {
    InnerCell::pinned(self)
  }

  fn pin(&self) {
    InnerCell::pin(self)
  }

  fn unpin(&self) {
    InnerCell::unpin(self)
  }

  fn _try_borrow(&self) -> Result<(), BorrowErr> {
    InnerCell::_try_borrow(self)
  }

  fn _try_borrow_unsafe(&self) -> Result<(), BorrowErr> {
    InnerCell::_try_borrow_unsafe(self)
  }

  fn _unborrow(&self) {
    InnerCell::_unborrow(self)
  }

  fn mem_borrow(&self) -> Result<BorrowRef<[u8]>, BorrowErr> {
    InnerCell::mem_borrow(self)
  }

  fn mem_borrow_mut(&self) -> Result<BorrowRefMut<[u8]>, BorrowErr> {
    InnerCell::mem_borrow_mut(self)
  }
}

#[derive(Clone, Copy, Debug)]
#[repr(u8)]
pub enum BorrowErr {
  NotImpl = 1,
  Unsafe,
  Immutable,
  AlreadyMutablyBorrowed,
  AlreadyBorrowed,
  TooManyBorrows,
}

#[repr(transparent)]
pub struct BorrowCell {
  ctr:  Cell<i8>,
}

impl Drop for BorrowCell {
  fn drop(&mut self) {
    let c = self.ctr.get();
    if c != 0 {
      panic!("bug");
    }
  }
}

impl BorrowCell {
  pub fn new() -> BorrowCell {
    BorrowCell{ctr: Cell::new(0)}
  }

  pub fn _borrowed(&self) -> bool {
    self.ctr.get() != 0
  }

  pub fn _try_borrow(&self) -> Result<(), BorrowErr> {
    let c = self.ctr.get();
    if c < 0 {
      return Err(BorrowErr::AlreadyMutablyBorrowed);
    }
    if c >= i8::max_value() {
      return Err(BorrowErr::TooManyBorrows);
    }
    self.ctr.set(c + 1);
    Ok(())
  }

  pub fn _unborrow(&self) {
    let c = self.ctr.get();
    if c <= 0 {
      panic!("bug");
    }
    self.ctr.set(c - 1);
  }

  pub fn _try_borrow_mut(&self) -> Result<(), BorrowErr> {
    let c = self.ctr.get();
    if c < 0 {
      return Err(BorrowErr::AlreadyMutablyBorrowed);
    } else if c > 0 {
      return Err(BorrowErr::AlreadyBorrowed);
    }
    self.ctr.set(-1);
    Ok(())
  }

  pub fn _unborrow_mut(&self) {
    let c = self.ctr.get();
    if c != -1 {
      panic!("bug");
    }
    self.ctr.set(0);
  }
}

pub struct BorrowRef<'a, T: ?Sized> {
  borc: &'a BorrowCell,
  val:  Option<&'a T>,
}

impl<'a, T: ?Sized> Drop for BorrowRef<'a, T> {
  fn drop(&mut self) {
    self.val = None;
    self.borc._unborrow();
  }
}

impl<'a, T: ?Sized> Deref for BorrowRef<'a, T> {
  type Target = T;

  fn deref(&self) -> &T {
    self.val.as_ref().unwrap()
  }
}

impl<'a, T: ?Sized> BorrowRef<'a, T> {
  pub fn map<F: FnOnce(&T) -> &T>(mut this: BorrowRef<'a, T>, f: F) -> BorrowRef<'a, T> {
    this.val = Some((f)(this.val.take().unwrap()));
    this
  }
}

pub struct BorrowRefMut<'a, T: ?Sized> {
  borc: &'a BorrowCell,
  val:  Option<&'a mut T>,
}

impl<'a, T: ?Sized> Drop for BorrowRefMut<'a, T> {
  fn drop(&mut self) {
    self.val = None;
    self.borc._unborrow_mut();
  }
}

impl<'a, T: ?Sized> Deref for BorrowRefMut<'a, T> {
  type Target = T;

  fn deref(&self) -> &T {
    self.val.as_ref().unwrap()
  }
}

impl<'a, T: ?Sized> DerefMut for BorrowRefMut<'a, T> {
  fn deref_mut(&mut self) -> &mut T {
    self.val.as_mut().unwrap()
  }
}

impl<'a, T: ?Sized> BorrowRefMut<'a, T> {
  pub fn map_mut<F: FnOnce(&mut T) -> &mut T>(this: &mut BorrowRefMut<'a, T>, f: F) {
    this.val = Some((f)(this.val.take().unwrap()));
  }
}
