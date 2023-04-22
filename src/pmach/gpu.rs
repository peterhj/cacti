use super::*;

use cacti_cuffi::*;

//use std::alloc::{Layout};
use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, HashMap};
use std::rc::{Rc, Weak};

/*
pub struct GpuPCell {
  dtype:        Dtype,
  //pitch:        Vec<u64>,
  //shape:        Vec<usize>,
  laststate:    PCellState,
  // FIXME
  inner_ref:    Option<GpuInnerRef>,
  inner:        Option<Weak<GpuInnerCell>>,
}

impl Default for GpuPCell {
  fn default() -> GpuPCell {
    GpuPCell{
      dtype:        Dtype::_Top,
      laststate:    PCellState::Unalloc,
      inner_ref:    None,
      inner:        None,
    }
  }
}
*/

#[derive(Clone, Copy)]
pub struct GpuInnerCellDesc {
  offset:   usize,
  sz:       usize,
}

pub struct GpuInnerCell {
  //stableptr:  StablePtr,
  ptr:      *mut u8,
  offset:   usize,
  sz:       usize,
  write:    CudartEvent,
  read_:    CudartEvent,
  copy_:    CudartEvent,
  // FIXME
  //cpu_dep:  Option<Rc<CpuInnerCell>>,
  // TODO
}

pub struct GpuOuterCell {
  //stableptr:  StablePtr,
  // FIXME
  //write:    CudartEvent,
  pub copy_:    CudartEvent,
  //cpu_dep:  Option<Rc<CpuInnerCell>>,
  // TODO
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct GpuInnerRef(u32);

thread_local! {
  static TL_GPU_CTX: TlsGpuCtx = TlsGpuCtx::new();
}

pub struct TlsGpuCtx {
  main:         CudartStream,
  copy_to:      CudartStream,
  copy_from:    CudartStream,
  reserve_ptr:  *mut u8,
  reserve_sz:   usize,
  iref_ctr:     Cell<u32>,
  //front_sz:     Cell<usize>,
  front_end:    Cell<usize>,
  //front:        RefCell<Vec<(GpuInnerRef, GpuInnerCellDesc)>>,
  front_set:    RefCell<BTreeMap<GpuInnerRef, GpuInnerCellDesc>>,
  back_sz:      Cell<usize>,
  //back_start:   Cell<usize>,
  back:         RefCell<Vec<(GpuInnerRef, GpuInnerCellDesc)>>,
  //back_set:     RefCell<BTreeSet<GpuInnerRef>>,
  //back_gen:     u32,
  cel_map:      RefCell<HashMap<GpuInnerRef, Rc<GpuInnerCell>>>,
  // TODO
}

impl TlsGpuCtx {
  pub fn new() -> TlsGpuCtx {
    let dev = 0;
    cudart_set_cur_dev(dev).unwrap();
    let (free_sz, total_sz) = cudart_get_mem_info().unwrap();
    assert!(free_sz <= total_sz);
    let reserve_bp = tl_get_gpu_reserve_mem_per_10k();
    let unrounded_reserve_sz = (total_sz * reserve_bp as usize + 10000 - 1) / 10000;
    // NB: assuming page size is 64 KiB.
    let reserve_sz = ((unrounded_reserve_sz + (1 << 16) - 1) >> 16) << 16;
    assert!(reserve_sz >= unrounded_reserve_sz);
    if reserve_sz > free_sz {
      panic!("FAIL: TlsGpuCtx::new: gpu oom: tried to reserve {} bytes on gpu {}, but only found {} bytes free (out of {} bytes total)",
          reserve_sz, dev, free_sz, total_sz);
    }
    let reserve_ptr = cudart_malloc(reserve_sz).unwrap() as *mut u8;
    CudartStream::null().sync().unwrap();
    let reserve_warp_offset = reserve_ptr.align_offset(128);
    if reserve_warp_offset != 0 {
      panic!("FAIL: TlsGpuCtx::new: gpu bug: misaligned alloc, offset by {} bytes (expected alignment {} bytes)",
          reserve_warp_offset, 128);
    }
    TlsGpuCtx{
      /*main:         CudartStream::null(),
      copy_to:      CudartStream::create_nonblocking().unwrap(),
      copy_from:    CudartStream::create_nonblocking().unwrap(),*/
      main:         CudartStream::create().unwrap(),
      copy_to:      CudartStream::create().unwrap(),
      copy_from:    CudartStream::create().unwrap(),
      reserve_ptr,
      reserve_sz,
      iref_ctr:     Cell::new(0),
      //front_sz:     Cell::new(0),
      front_end:    Cell::new(0),
      //front:        RefCell::new(Vec::new()),
      front_set:    RefCell::new(BTreeMap::default()),
      back_sz:      Cell::new(0),
      back:         RefCell::new(Vec::new()),
      //back_set:     RefCell::new(BTreeSet::default()),
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

  pub fn try_front_alloc(&self, stableptr: StablePtr, query_sz: usize) -> (Option<GpuInnerRef>, Option<Weak<GpuInnerCell>>) {
    let req_sz = ((query_sz + 128 - 1) / 128) * 128;
    assert!(query_sz <= req_sz);
    let offset = self.front_end.get();
    let next_offset = offset + req_sz;
    if next_offset + self.back_sz.get() > self.reserve_sz {
      return (None, None);
    }
    let r = self._fresh_inner_ref();
    self.front_end.set(next_offset);
    self.front_set.borrow_mut().insert(r, GpuInnerCellDesc{offset, sz: req_sz});
    let ptr = unsafe { self.reserve_ptr.offset(offset as isize) };
    let dev = 0;
    cudart_set_cur_dev(dev).unwrap();
    let write = CudartEvent::create_fastest().unwrap();
    let read_ = CudartEvent::create_fastest().unwrap();
    let copy_ = CudartEvent::create_fastest().unwrap();
    let cel = Rc::new(GpuInnerCell{
      //stableptr,
      ptr,
      offset,
      sz: req_sz,
      write,
      read_,
      copy_,
    });
    let xcel = Rc::downgrade(&cel);
    self.cel_map.borrow_mut().insert(r, cel);
    (Some(r), Some(xcel))
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

  pub fn try_front_free(&self, r: GpuInnerRef) -> Option<()> {
    match self.cel_map.borrow().get(&r) {
      None => {}
      Some(cel) => {
        let c = Rc::strong_count(cel);
        if c > 1 {
          return None;
        }
        cel.read_.sync();
        cel.copy_.sync();
        // FIXME FIXME: remove from front.
        self.front_set.borrow_mut().remove(&r);
        self.cel_map.borrow_mut().remove(&r);
      }
    }
    Some(())
  }

  pub fn try_back_alloc(&self, sz: usize) -> (Option<GpuInnerRef>, Option<Weak<GpuInnerCell>>) {
    unimplemented!();
  }

  pub fn back_free_all(&self) {
    unimplemented!();
  }

  pub fn find_ptr(&self, r: GpuInnerRef) -> Option<*mut u8> {
    unimplemented!();
  }

  pub fn fresh_outer(&self) -> Rc<GpuOuterCell> {
    unimplemented!();
  }

  pub fn copy_mem_to_gpu(&self, cel: &GpuInnerCell, src_mem: *const u8, mem_sz: usize) {
    // FIXME
    cudart_memcpy(cel.ptr as _, src_mem as _, mem_sz, &self.copy_to);
    cel.write.record(&self.copy_to);
  }

  pub fn copy_mem_from_gpu(&self, cel: &GpuInnerCell, dst_mem: *mut u8, mem_sz: usize) {
    // FIXME
    self.copy_from.wait_event(&cel.write);
    cudart_memcpy(dst_mem as _, cel.ptr as _, mem_sz, &self.copy_from);
  }

  pub fn copy_swap_to_gpu(&self, cel: &GpuInnerCell, src: (), mem_sz: usize) {
    unimplemented!();
  }

  pub fn copy_swap_from_gpu(&self, cel: &GpuInnerCell, src: (), mem_sz: usize) {
    unimplemented!();
  }
}
