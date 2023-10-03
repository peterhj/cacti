use super::*;
use crate::algo::{BTreeSet, RevSortMap8};
use crate::cell::*;
use crate::clock::*;
use cacti_cfg_env::*;
use cacti_smp_c_ffi::*;

#[cfg(target_os = "linux")]
use libc::{__errno_location};
#[cfg(not(target_os = "linux"))]
use libc::{__errno as __errno_location};
use libc::{
  ENOMEM, _SC_PAGESIZE, _SC_NPROCESSORS_ONLN,
  free, malloc, sysconf,
  c_char, c_int, c_void,
};
use once_cell::sync::{Lazy};

use std::cell::{Cell};
use std::io::{Error as IoError};
use std::process::{Command, Stdio};
//use std::rc::{Rc};
use std::str::{from_utf8};
use std::sync::{Arc};
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::sync::mpsc::{SyncSender, Receiver, sync_channel};
use std::thread::{JoinHandle, Thread, park, spawn};

pub static ONCE_SMP_INFO: Lazy<SmpInfo> = Lazy::new(|| SmpInfo::new());
thread_local! {
  pub static TL_SMP_INFO: SmpInfo = SmpInfo::tl_clone();
}

#[derive(Clone)]
pub struct SmpInfo {
  pub sc_page_sz: Option<usize>,
  pub lscpu: Option<Arc<LscpuParse>>,
}

impl SmpInfo {
  pub fn new() -> SmpInfo {
    let ret = unsafe { sysconf(_SC_PAGESIZE) };
    let sc_page_sz = if ret <= 0 {
      None
    } else {
      ret.try_into().ok()
    };
    SmpInfo{
      sc_page_sz,
      lscpu: LscpuParse::open().ok().map(|inner| inner.into()),
    }
  }

  pub fn tl_clone() -> SmpInfo {
    ONCE_SMP_INFO.clone()
  }
}

#[derive(Clone, Copy, Default, Debug)]
pub struct LscpuEntry {
  pub cpu:  u16,
  pub core: u16,
  pub sock: u8,
  pub node: u8,
}

#[derive(Debug)]
pub struct LscpuParse {
  pub entries: Vec<LscpuEntry>,
  pub core_ct: AtomicUsize,
}

#[derive(Clone, Copy, Debug)]
enum LscpuParseState {
  Field(u8),
  Skip,
}

impl LscpuParse {
  pub fn open() -> Result<LscpuParse, ()> {
    let out = Command::new("lscpu")
        .arg("-p")
        .stdout(Stdio::piped())
        .output()
        .map_err(|_| ())?;
    if !out.status.success() {
      return Err(());
    }
    LscpuParse::parse(out.stdout)
  }

  pub fn parse<O: AsRef<[u8]>>(out: O) -> Result<LscpuParse, ()> {
    let out = out.as_ref();
    let mut entries = Vec::new();
    let mut e = LscpuEntry::default();
    let mut save = 0;
    let mut cursor = 0;
    let mut state = LscpuParseState::Field(0);
    loop {
      match state {
        LscpuParseState::Field(col) => {
          if cursor >= out.len() {
            if save == cursor && col == 0 {
              break;
            } else {
              return Err(());
            }
          }
          let x = out[cursor];
          if x == b'#' && save == cursor && col == 0 {
            cursor += 1;
            save = usize::max_value();
            state = LscpuParseState::Skip;
          } else if x >= b'0' && x <= b'9' {
            cursor += 1;
          } else if x == b',' {
            let s = from_utf8(&out[save .. cursor]).map_err(|_| ())?;
            match col {
              0 => e.cpu  = s.parse().map_err(|_| ())?,
              1 => e.core = s.parse().map_err(|_| ())?,
              2 => e.sock = s.parse().map_err(|_| ())?,
              3 => e.node = s.parse().map_err(|_| ())?,
              _ => unreachable!()
            }
            cursor += 1;
            if col < 3 {
              save = cursor;
              state = LscpuParseState::Field(col + 1);
            } else {
              entries.push(e);
              e = LscpuEntry::default();
              save = usize::max_value();
              state = LscpuParseState::Skip;
            }
          } else {
            return Err(());
          }
        }
        LscpuParseState::Skip => {
          if cursor >= out.len() {
            break;
          }
          let x = out[cursor];
          cursor += 1;
          if x == b'\n' {
            save = cursor;
            state = LscpuParseState::Field(0);
          }
        }
      }
    }
    Ok(LscpuParse{
      entries,
      core_ct: AtomicUsize::new(0),
    })
  }

  pub fn physical_core_count(&self) -> Option<u16> {
    match self.core_ct.load(AtomicOrdering::Relaxed) {
      0 => {}
      c => {
        assert!(c <= u16::max_value() as _);
        return Some(c as _);
      }
    }
    let mut core = BTreeSet::new();
    for e in self.entries.iter() {
      core.insert(e.core);
    }
    if core.len() == 0 {
      return None;
    }
    assert!(core.len() <= u16::max_value() as _);
    match self.core_ct.compare_exchange(0, core.len(), AtomicOrdering::Relaxed, AtomicOrdering::Relaxed) {
      Ok(0) | Err(0) => {}
      Ok(c) | Err(c) => {
        assert_eq!(c, core.len());
      }
    }
    Some(core.len() as _)
  }
}

#[repr(C)]
pub struct MemCell {
  pub ptr:  *mut c_void,
  pub sz:   usize,
}

impl Drop for MemCell {
  fn drop(&mut self) {
    assert!(!self.ptr.is_null());
    unsafe {
      free(self.ptr);
    }
  }
}

impl MemCell {
  pub fn try_alloc(sz: usize) -> Result<MemCell, PMemErr> {
    unsafe {
      let ptr = malloc(sz);
      if ptr.is_null() {
        let e = *(__errno_location)();
        if e == ENOMEM {
          return Err(PMemErr::Oom);
        } else {
          return Err(PMemErr::Bot);
        }
      }
      Ok(MemCell{ptr, sz})
    }
  }

  pub fn as_reg(&self) -> MemReg {
    MemReg{
      ptr:  self.ptr,
      sz:   self.size_bytes(),
    }
  }

  pub fn size_bytes(&self) -> usize {
    self.sz
  }
}

/*pub struct SmpInnerCell {
  pub clk:  Cell<Clock>,
  pub mem:  MemCell,
  // FIXME
  //#[cfg(feature = "nvgpu")]
  //pub gpu:  Option<GpuOuterCell>,
  // TODO
}

impl SmpInnerCell {
  /*pub fn wait_gpu(&self) {
    match self.gpu.as_ref() {
      None => {}
      Some(cel) => {
        // FIXME FIXME: query spin wait.
        cel.write.event.sync().unwrap();
      }
    }
  }*/
}

impl InnerCell for SmpInnerCell {}*/

#[derive(Clone, Copy)]
pub enum SmpCtl2Thread {
  Shutdown,
  Memcpy(usize, usize, usize),
}

#[derive(Clone, Copy)]
pub enum SmpThread2Ctl {
  Complete(u16),
}

pub struct SmpThreadPool {
  ctl2th:   Vec<(SyncSender<SmpCtl2Thread>, JoinHandle<()>, )>,
  th2ctl:   Receiver<SmpThread2Ctl>,
}

impl Drop for SmpThreadPool {
  fn drop(&mut self) {
    for &(ref ctl2th, ref h) in self.ctl2th.iter() {
      ctl2th.send(SmpCtl2Thread::Shutdown).unwrap();
      h.thread().unpark();
    }
    for (_, h) in self.ctl2th.drain(..) {
      h.join().unwrap();
    }
  }
}

impl SmpThreadPool {
  pub fn new(thct: u16) -> SmpThreadPool {
    if cfg_info() { println!("INFO:   SmpThreadPool::new: thread count={}", thct); }
    let mut ctl2th = Vec::with_capacity(thct as _);
    let (th2ctl_tx, th2ctl) = sync_channel(2 * thct as usize);
    for rank in 0 .. thct {
      let (ctl2th_tx, ctl2th_rx) = sync_channel(4);
      let th2ctl_tx = th2ctl_tx.clone();
      let h = spawn(move || {
        loop {
          park();
          match ctl2th_rx.recv() {
            Ok(SmpCtl2Thread::Shutdown) => {
              break;
            }
            Ok(SmpCtl2Thread::Memcpy(dst_ptr, src_ptr, sz)) => {
              unsafe {
                let dst_ptr = dst_ptr as *mut u8;
                let src_ptr = src_ptr as *const u8;
                std::intrinsics::copy_nonoverlapping(src_ptr, dst_ptr, sz);
              }
              th2ctl_tx.send(SmpThread2Ctl::Complete(rank)).unwrap();
            }
            _ => break
          }
        }
      });
      ctl2th.push((ctl2th_tx, h));
    }
    SmpThreadPool{
      ctl2th,
      th2ctl,
    }
  }

  #[inline]
  pub fn num_threads(&self) -> usize {
    self.ctl2th.len()
  }

  pub fn memcpy(&self, dst_ptr: *mut u8, src_ptr: *const u8, sz: usize) {
    assert!((dst_ptr as usize + sz <= src_ptr as usize) ||
            (src_ptr as usize + sz <= dst_ptr as usize));
    let thsz = self.num_threads();
    let mut chunks = Vec::with_capacity(thsz);
    for r in 0 .. thsz {
      // FIXME: try to align on dst cacheline.
      let start = (sz * r) / thsz;
      let end = (sz * (r + 1)) / thsz;
      let ch_sz = end - start;
      let ch_src = unsafe { src_ptr.offset(start as _) as usize };
      let ch_dst = unsafe { dst_ptr.offset(start as _) as usize };
      chunks.push((ch_dst, ch_src, ch_sz));
    }
    for (r, &(ref ctl2th, ref h)) in self.ctl2th.iter().enumerate() {
      let (dst_ptr, src_ptr, sz) = chunks[r];
      ctl2th.send(SmpCtl2Thread::Memcpy(dst_ptr, src_ptr, sz)).unwrap();
      h.thread().unpark();
    }
    // FIXME: bitset.
    let mut recv = BTreeSet::new();
    for _ in 0 .. thsz {
      match self.th2ctl.recv() {
        Ok(SmpThread2Ctl::Complete(rank)) => {
          recv.insert(rank);
        }
        _ => panic!("bug")
      }
    }
    assert_eq!(recv.len(), thsz);
  }
}

pub struct SmpPCtx {
  pub page_sz:  usize,
  //pub lcore_ct: u16,
  pub pcore_ct: u16,
  pub th_pool:  SmpThreadPool,
}

impl PCtxImpl for SmpPCtx {
  //type ICel = SmpInnerCell;

  fn pmach(&self) -> PMach {
    PMach::Smp
  }

  fn fastest_locus(&self) -> Locus {
    Locus::Mem
  }

  fn append_matrix(&self, lp: &mut RevSortMap8<(Locus, PMach), ()>, pl: &mut RevSortMap8<(PMach, Locus), ()>) {
    lp.insert((Locus::Mem, PMach::Smp), ());
    pl.insert((PMach::Smp, Locus::Mem), ());
  }

  //fn try_alloc(&self, x: CellPtr, sz: usize, /*pmset: PMachSet,*/ locus: Locus) -> Result<Rc<dyn InnerCell_>, PMemErr> {}
  //fn try_alloc(&self, pctr: &PCtxCtr, /*pmset: PMachSet,*/ locus: Locus, ty: &CellType) -> Result<PAddr, PMemErr> {}
  fn try_alloc(&self, pctr: &PCtxCtr, ty: &CellType, locus: Locus) -> Result<PAddr, PMemErr> {
    unimplemented!();
  }
}

impl SmpPCtx {
  pub fn new() -> SmpPCtx {
    let page_sz = TL_SMP_INFO.with(|smp| {
      if smp.sc_page_sz.is_none() {
        println!("ERROR:  SmpPCtx::new: failed to get the page size");
        panic!();
      }
      smp.sc_page_sz.unwrap()
    });
    if cfg_info() { println!("INFO:   SmpPCtx::new: page size={}", page_sz); }
    let pcore_ct = TL_SMP_INFO.with(|smp| {
      smp.lscpu.as_ref().and_then(|parse| {
        //if cfg_info() { println!("INFO:   SmpPCtx::new: lscpu={:?}", parse); }
        parse.physical_core_count()
      })
    }).unwrap_or_else(|| {
      if cfg_info() { println!("WARNING:SmpPCtx::new: lscpu failure, try fallback"); }
      let ret = unsafe { sysconf(_SC_NPROCESSORS_ONLN) };
      if ret <= 0 {
        println!("ERROR:  SmpPCtx::new: failed to get the logical core count");
        panic!();
      }
      ret.try_into().unwrap()
    });
    if cfg_info() { println!("INFO:   SmpPCtx::new: core count={}", pcore_ct); }
    /*let n: u32 = if LIBCBLAS.openblas.get_num_threads.is_some() {
      let n = (LIBCBLAS.openblas.get_num_threads.as_ref().unwrap())();
      if cfg_info() { println!("INFO:   SmpPCtx::new: blas num threads={}", n); }
      assert!(n >= 1);
      /*// FIXME FIXME: debugging.
      let n = 1;
      (LIBCBLAS.openblas.set_num_threads.as_ref().unwrap())(n);
      if cfg_info() { println!("INFO:   SmpPCtx::new: blas set num threads={}", n); }
      let n = (LIBCBLAS.openblas.get_num_threads.as_ref().unwrap())();
      if cfg_info() { println!("INFO:   SmpPCtx::new: blas num threads={}", n); }*/
      n as _
    } else {
      1
    };*/
    let th_pool = SmpThreadPool::new(pcore_ct);
    SmpPCtx{
      page_sz,
      //lcore_ct,
      pcore_ct,
      th_pool,
    }
  }

  pub fn page_size(&self) -> usize {
    self.page_sz
  }

  /*pub fn logical_core_count(&self) -> u16 {
    self.lcore_ct
  }*/

  pub fn phy_core_ct(&self) -> u16 {
    self.physical_core_count()
  }

  pub fn physical_core_count(&self) -> u16 {
    self.pcore_ct
  }

  /*pub fn append_matrix(&self, lp: &mut Vec<(Locus, PMach)>, pl: &mut Vec<(PMach, Locus)>) {
    lp.push((Locus::Mem, PMach::Smp));
    pl.push((PMach::Smp, Locus::Mem));
  }*/

  /*pub fn try_mem_alloc(&self, sz: usize, pmset: PMachSet) -> Result<MemCell, PMemErr> {
    if pmset.contains(PMach::NvGpu) {
      MemCell::try_alloc_page_locked(sz)
    } else {
      MemCell::try_alloc(sz)
    }
  }*/
}

pub extern "C" fn tl_pctx_smp_mem_alloc_hook(ptr: *mut *mut c_void, sz: usize, raw_tag: *const c_char) -> c_int {
  // FIXME
  unimplemented!();
  /*
  assert!(!ptr.is_null());
  unsafe {
    let mem = malloc(sz);
    if mem.is_null() {
      return 1;
    }
    write(ptr, mem);
    0
  }
  */
}

pub extern "C" fn tl_pctx_smp_mem_free_hook(ptr: *mut c_void) -> c_int {
  // FIXME
  unimplemented!();
  /*
  assert!(!ptr.is_null());
  unsafe {
    free(ptr);
  }
  */
}

pub extern "C" fn tl_pctx_smp_mem_unify_hook(lhs_raw_tag: *mut c_char, rhs_raw_tag: *mut c_char) {
  // FIXME
}
