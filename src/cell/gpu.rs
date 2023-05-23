use super::*;
use crate::algo::{Bitvec64, MergeVecDeque, ExtentVecList, Extent};
use crate::algo::sync::{SpinWait};
use crate::clock::*;

use cacti_gpu_cu_ffi::*;
use cacti_gpu_cu_ffi::types::{cudaErrorCudartUnloading, cudaErrorNotReady};

//use std::alloc::{Layout};
use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, HashMap};
use std::rc::{Rc, Weak};

//#[derive(Clone)]
pub struct GpuSnapshot {
  pub record:   Cell<bool>,
  pub dev:      i32,
  pub event:    CudartEvent,
  //pub stream:   CudartStream,
}

impl GpuSnapshot {
  pub fn fresh(dev: i32) -> Rc<GpuSnapshot> {
    cudart_set_cur_dev(dev).unwrap();
    Rc::new(GpuSnapshot{
      record: Cell::new(false),
      dev,
      event: CudartEvent::create_fastest().unwrap(),
    })
  }

  pub fn set_record(&self) {
    self.record.set(true);
  }

  #[allow(non_upper_case_globals)]
  pub fn wait(&self, sw: &mut SpinWait) {
    if !self.record.get() {
      return;
    }
    loop {
      match self.event.query() {
        Ok(_) => break,
        Err(cudaErrorNotReady) => {
          sw.spin();
        }
        _ => panic!("bug")
      }
    }
    self.record.set(false);
  }
}

//#[derive(Clone)]
pub struct GpuOuterCell {
  pub write:    Rc<GpuSnapshot>,
  pub lastuse:  Rc<GpuSnapshot>,
  //pub lastcopy: Rc<GpuSnapshot>,
  //pub smp_dep:  Option<Rc<SmpInnerCell>>,
  // TODO
}

pub struct GpuInnerCell {
  pub ptr:      Cell<CellPtr>,
  pub clk:      Cell<Clock>,
  pub ref_:     GpuInnerRef,
  pub dev:      i32,
  pub dptr:     *mut u8,
  pub off:      usize,
  pub sz:       usize,
  //pub write:    Rc<CudartEvent>,
  //pub lastuse:  Rc<CudartEvent>,
  pub write:    Rc<GpuSnapshot>,
  pub lastuse:  Rc<GpuSnapshot>,
  //pub lastcopy: Rc<GpuSnapshot>,
  // FIXME
  //pub smp_dep:  Option<Rc<SmpInnerCell>>,
  // TODO
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct GpuInnerRef(u32);

impl GpuInnerRef {
  pub fn free() -> GpuInnerRef {
    GpuInnerRef(0)
  }

  pub fn is_free(&self) -> bool {
    self.0 == 0
  }
}

/*thread_local! {
  static TL_GPU_CTX: GpuCtx = GpuCtx::new();
}*/

pub struct GpuCtx {
  // FIXME FIXME: work threads for copying.
  pub iref_ctr:     Cell<u32>,
  pub dev:          Cell<i32>,
  pub main:         CudartStream,
  pub copy_to:      CudartStream,
  pub copy_from:    CudartStream,
  pub mem_pool:     GpuMemPool,
  pub cel_map:      RefCell<HashMap<GpuInnerRef, Rc<GpuInnerCell>>>,
  // TODO
}

impl GpuCtx {
  pub fn new() -> GpuCtx {
    println!("DEBUG: GpuCtx::new");
    let dev = 0;
    cudart_set_cur_dev(dev).unwrap();
    let main = CudartStream::null();
    let copy_to = CudartStream::create_nonblocking().unwrap();
    let copy_from = CudartStream::create_nonblocking().unwrap();
    /*let main = CudartStream::create().unwrap();
    let copy_to = CudartStream::create().unwrap();
    let copy_from = CudartStream::create().unwrap();*/
    let mem_pool = GpuMemPool::new(dev);
    GpuCtx{
      iref_ctr:     Cell::new(0),
      dev:          Cell::new(dev),
      main,
      copy_to,
      copy_from,
      mem_pool,
      cel_map:      RefCell::new(HashMap::default()),
      // TODO
    }
  }

  fn _fresh_inner_ref(&self) -> GpuInnerRef {
    let next = self.iref_ctr.get() + 1;
    assert!(next > 0);
    assert!(next < u32::max_value());
    self.iref_ctr.set(next);
    GpuInnerRef(next)
  }

  pub fn find_ptr(&self, r: GpuInnerRef) -> Option<*mut u8> {
    unimplemented!();
  }

  pub fn fresh_outer(&self) -> Rc<GpuOuterCell> {
    unimplemented!();
  }

  pub fn copy_mem_to_gpu(&self, cel: &GpuInnerCell, src_mem: *const u8, mem_sz: usize) {
    // FIXME
    cudart_memcpy(cel.dptr as _, src_mem as _, mem_sz, &self.copy_to).unwrap();
    cel.write.set_record();
    cel.write.event.record(&self.copy_to).unwrap();
  }

  pub fn copy_mem_from_gpu(&self, cel: &GpuInnerCell, dst_mem: *mut u8, mem_sz: usize) {
    // FIXME
    self.copy_from.wait_event(&cel.write.event).unwrap();
    cudart_memcpy(dst_mem as _, cel.dptr as _, mem_sz, &self.copy_from).unwrap();
  }

  pub fn copy_swap_to_gpu(&self, cel: &GpuInnerCell, src: (), mem_sz: usize) {
    unimplemented!();
  }

  pub fn copy_swap_from_gpu(&self, cel: &GpuInnerCell, dst: (), mem_sz: usize) {
    unimplemented!();
  }

  pub fn try_front_pre_alloc(&self, query_sz: usize) -> Option<(usize, usize, usize)> {
    self.mem_pool.try_front_pre_alloc(query_sz)
  }

  pub fn front_alloc(&self, ptr: CellPtr, offset: usize, req_sz: usize, next_offset: usize) -> Weak<GpuInnerCell> {
    let r = self._fresh_inner_ref();
    self.mem_pool.front_set.borrow_mut().insert(r, GpuInnerMemDesc{off: offset, sz: req_sz});
    self.mem_pool.front_cursor.set(next_offset);
    let dptr = unsafe { self.mem_pool.reserve_base.offset(offset as isize) };
    /*cudart_set_cur_dev(self.dev).unwrap();
    let write = Rc::new(CudartEvent::create_fastest().unwrap());
    let lastuse = Rc::new(CudartEvent::create_fastest().unwrap());*/
    let write = GpuSnapshot::fresh(self.dev.get());
    let lastuse = GpuSnapshot::fresh(self.dev.get());
    let cel = Rc::new(GpuInnerCell{
      ptr: Cell::new(ptr),
      clk: Cell::new(Clock::default()),
      ref_: r,
      dev: self.dev.get(),
      dptr,
      off: offset,
      sz: req_sz,
      write,
      lastuse,
      // FIXME
    });
    let xcel = Rc::downgrade(&cel);
    self.cel_map.borrow_mut().insert(r, cel);
    xcel
  }

  pub fn try_front_free(&self, r: GpuInnerRef) -> Option<()> {
    match self.cel_map.borrow().get(&r) {
      None => {}
      Some(cel) => {
        // FIXME FIXME: think about this ref count check.
        let c = Rc::strong_count(cel);
        if c > 1 {
          return None;
        }
        let mut sw = SpinWait::default();
        cel.write.wait(&mut sw);
        cel.lastuse.wait(&mut sw);
        // FIXME FIXME: remove from front.
        drop(cel);
        self.mem_pool.front_set.borrow_mut().remove(&r);
        self.cel_map.borrow_mut().remove(&r);
      }
    }
    Some(())
  }
}

#[derive(Clone)]
pub struct GpuInnerMemCell {
  pub off:  usize,
  pub sz:   usize,
  pub prevfree: Cell<usize>,
  pub nextfree: Cell<usize>,
}

impl GpuInnerMemCell {
  pub fn new(off: usize, sz: usize) -> GpuInnerMemCell {
    GpuInnerMemCell{
      off,
      sz,
      prevfree: Cell::new(usize::max_value()),
      nextfree: Cell::new(usize::max_value()),
    }
  }
}

#[derive(Clone, Copy)]
pub struct GpuInnerMemDesc {
  pub off:  usize,
  pub sz:   usize,
}

pub struct GpuMemPool {
  pub dev:          i32,
  pub reserve_base: *mut u8,
  pub reserve_sz:   usize,
  pub front_pad:    usize,
  pub front_sz:     usize,
  pub boundary_pad: usize,
  pub back_sz:      usize,
  pub back_pad:     usize,
  pub front_list:   RefCell<MergeVecDeque<(GpuInnerRef, GpuInnerMemCell)>>,
  pub front_set:    RefCell<BTreeMap<GpuInnerRef, GpuInnerMemDesc>>,
  pub front_cursor: Cell<usize>,
  pub back_bitmap:  RefCell<Bitvec64>,
  pub back_cursor:  Cell<usize>,
  pub tmp_freelist: ExtentVecList,
  // TODO
}

impl Drop for GpuMemPool {
  #[allow(non_upper_case_globals)]
  fn drop(&mut self) {
    // FIXME FIXME: wait for nonblocking transfers.
    match CudartStream::null().sync() {
      Ok(_) | Err(cudaErrorCudartUnloading) => {}
      Err(_) => panic!("bug")
    }
    match cudart_free(self.reserve_base as _) {
      Ok(_) | Err(cudaErrorCudartUnloading) => {}
      Err(_) => panic!("bug")
    }
  }
}

impl GpuMemPool {
  #[allow(unused_parens)]
  pub fn new(dev: i32) -> GpuMemPool {
    println!("DEBUG: GpuMemPool::new");
    cudart_set_cur_dev(dev).unwrap();
    let (free_sz, total_sz) = cudart_get_mem_info().unwrap();
    assert!(free_sz <= total_sz);
    let reserve_bp = ctx_cfg_get_gpu_reserve_mem_per_10k();
    let unrounded_reserve_sz = (total_sz * reserve_bp as usize + 10000 - 1) / 10000;
    // NB: assuming page size is 64 KiB.
    let reserve_sz = ((unrounded_reserve_sz + (1 << 16) - 1) >> 16) << 16;
    assert!(reserve_sz >= unrounded_reserve_sz);
    if reserve_sz > free_sz {
      panic!("FAIL: GpuCtx::new: gpu oom: tried to reserve {} bytes on gpu {}, but only found {} bytes free (out of {} bytes total)",
          reserve_sz, dev, free_sz, total_sz);
    }
    println!("DEBUG: GpuMemPool::new: reserve sz={}", reserve_sz);
    let reserve_base = cudart_malloc(reserve_sz).unwrap() as *mut u8;
    CudartStream::null().sync().unwrap();
    let reserve_warp_offset = reserve_base.align_offset(128);
    if reserve_warp_offset != 0 {
      panic!("FAIL: GpuCtx::new: gpu bug: misaligned alloc, offset by {} bytes (expected alignment {} bytes)",
          reserve_warp_offset, 128);
    }
    let front_pad = (1 << 16);
    let boundary_pad = (1 << 16);
    let back_pad = (1 << 16);
    let back_sz = (1 << 24);
    let front_sz = reserve_sz - (front_pad + boundary_pad + back_sz + back_pad);
    assert!(front_sz >= (1 << 24));
    println!("DEBUG: GpuMemPool::new: front sz={}", front_sz);
    println!("DEBUG: GpuMemPool::new: back sz={}", back_sz);
    GpuMemPool{
      dev,
      reserve_base,
      reserve_sz,
      front_pad,
      front_sz,
      boundary_pad,
      back_sz,
      back_pad,
      front_list:   RefCell::new(MergeVecDeque::new()),
      front_set:    RefCell::new(BTreeMap::new()),
      front_cursor: Cell::new(0),
      back_bitmap:  RefCell::new(Bitvec64::new()),
      back_cursor:  Cell::new(back_sz),
      tmp_freelist: ExtentVecList::default(),
      // TODO
    }
  }

  pub fn try_front_pre_alloc(&self, query_sz: usize) -> Option<(usize, usize, usize)> {
    let req_sz = ((query_sz + 128 - 1) / 128) * 128;
    assert!(query_sz <= req_sz);
    let offset = self.front_cursor.get();
    let next_offset = offset + req_sz;
    if next_offset >= self.front_sz {
      return None;
    }
    Some((offset, req_sz, next_offset))
  }

  pub fn find_front_lru_match(&self, query_sz: usize) -> Option<GpuInnerRef> {
    let mut mat = None;
    for (&r, desc) in self.front_set.borrow().iter() {
      if query_sz == desc.sz {
        return Some(r);
      }
      if query_sz <= desc.sz {
        match mat {
          None => {
            mat = Some((r, desc.sz));
          }
          Some((_, prev_sz)) => if desc.sz < prev_sz {
            mat = Some((r, desc.sz));
          }
        }
      }
    }
    mat.map(|(r, _)| r)
  }

  pub fn try_back_alloc(&self, sz: usize) -> (Option<GpuInnerRef>, Option<Weak<GpuInnerCell>>) {
    unimplemented!();
  }

  pub fn back_free_all(&self) {
    unimplemented!();
  }
}

pub extern "C" fn tl_ctx_gpu_alloc_hook(dptr: *mut u64, sz: usize) -> i32 {
  TL_CTX.with(|ctx| {
    // FIXME FIXME
    0
  })
}

pub extern "C" fn tl_ctx_gpu_free_hook(dptr: u64) -> i32 {
  TL_CTX.with(|ctx| {
    // FIXME FIXME
    0
  })
}

pub struct GpuDryCtx {
  // FIXME
}
