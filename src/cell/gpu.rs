use super::*;
use crate::algo::{Bitvec64};
use crate::algo::sync::{SpinWait};
use crate::clock::*;

use cacti_cuffi::*;
use cacti_cuffi::types::{cudaErrorCudartUnloading, cudaErrorNotReady};

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

thread_local! {
  static TL_GPU_CTX: GpuCtx = GpuCtx::new();
}

pub struct GpuCtx {
  // FIXME FIXME: work threads for copying.
  dev:          i32,
  main:         CudartStream,
  copy_to:      CudartStream,
  copy_from:    CudartStream,
  iref_ctr:     Cell<u32>,
  mem_pool:     GpuMemPool,
  cel_map:      RefCell<HashMap<GpuInnerRef, Rc<GpuInnerCell>>>,
  // TODO
}

impl GpuCtx {
  pub fn new() -> GpuCtx {
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
      dev,
      main,
      copy_to,
      copy_from,
      iref_ctr:     Cell::new(0),
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

  pub fn try_front_alloc(&self, query_sz: usize) -> Option<(usize, usize, usize)> {
    self.mem_pool.try_front_alloc(query_sz)
  }

  pub fn front_alloc(&self, ptr: CellPtr, offset: usize, req_sz: usize, next_offset: usize) -> Option<Weak<GpuInnerCell>> {
    let r = self._fresh_inner_ref();
    self.mem_pool.front_set.borrow_mut().insert(r, GpuInnerCellMemDesc{off: offset, sz: req_sz});
    self.mem_pool.front_cursor.set(next_offset);
    let dptr = unsafe { self.mem_pool.reserve_base.offset(offset as isize) };
    /*cudart_set_cur_dev(self.dev).unwrap();
    let write = Rc::new(CudartEvent::create_fastest().unwrap());
    let lastuse = Rc::new(CudartEvent::create_fastest().unwrap());*/
    let write = GpuSnapshot::fresh(self.dev);
    let lastuse = GpuSnapshot::fresh(self.dev);
    let cel = Rc::new(GpuInnerCell{
      ptr: Cell::new(ptr),
      clk: Cell::new(Clock::default()),
      ref_: r,
      dev: self.dev,
      dptr,
      off: offset,
      sz: req_sz,
      write,
      lastuse,
      // FIXME
    });
    let xcel = Rc::downgrade(&cel);
    self.cel_map.borrow_mut().insert(r, cel);
    Some(xcel)
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
        let mut spin = SpinWait::default();
        cel.write.wait(&mut spin);
        cel.lastuse.wait(&mut spin);
        // FIXME FIXME: remove from front.
        drop(cel);
        self.mem_pool.front_set.borrow_mut().remove(&r);
        self.cel_map.borrow_mut().remove(&r);
      }
    }
    Some(())
  }
}

#[derive(Clone, Copy)]
pub struct GpuInnerCellMemDesc {
  off:  usize,
  sz:   usize,
}

pub struct GpuMemPool {
  dev:          i32,
  reserve_base: *mut u8,
  reserve_sz:   usize,
  front_pad:    usize,
  front_sz:     usize,
  boundary_pad: usize,
  back_sz:      usize,
  back_pad:     usize,
  front_set:    RefCell<BTreeMap<GpuInnerRef, GpuInnerCellMemDesc>>,
  front_cursor: Cell<usize>,
  back_bitmap:  RefCell<Bitvec64>,
  back_cursor:  Cell<usize>,
  // TODO
}

impl Drop for GpuMemPool {
  fn drop(&mut self) {
    // FIXME
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
  pub fn new(dev: i32) -> GpuMemPool {
    cudart_set_cur_dev(dev).unwrap();
    let (free_sz, total_sz) = cudart_get_mem_info().unwrap();
    assert!(free_sz <= total_sz);
    let reserve_bp = ctx_get_gpu_reserve_mem_per_10k();
    let unrounded_reserve_sz = (total_sz * reserve_bp as usize + 10000 - 1) / 10000;
    // NB: assuming page size is 64 KiB.
    let reserve_sz = ((unrounded_reserve_sz + (1 << 16) - 1) >> 16) << 16;
    assert!(reserve_sz >= unrounded_reserve_sz);
    if reserve_sz > free_sz {
      panic!("FAIL: GpuCtx::new: gpu oom: tried to reserve {} bytes on gpu {}, but only found {} bytes free (out of {} bytes total)",
          reserve_sz, dev, free_sz, total_sz);
    }
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
    GpuMemPool{
      dev,
      reserve_base,
      reserve_sz,
      front_pad,
      front_sz,
      boundary_pad,
      back_sz,
      back_pad,
      front_set:    RefCell::new(BTreeMap::new()),
      front_cursor: Cell::new(0),
      back_bitmap:  RefCell::new(Bitvec64::new()),
      back_cursor:  Cell::new(back_sz),
      // TODO
    }
  }

  pub fn try_front_alloc(&self, query_sz: usize) -> Option<(usize, usize, usize)> {
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

pub struct GpuDryCtx {
  // FIXME
}
