use super::{TL_PCTX, Locus, PMach};
use crate::algo::{MergeVecDeque, Region};
use crate::algo::sync::{SpinWait};
use crate::cell::*;
use crate::clock::*;
use crate::ctx::*;

use cacti_gpu_cu_ffi::*;
//use cacti_gpu_cu_ffi::types::{CU_CTX_SCHED_YIELD, cudaErrorCudartUnloading, cudaErrorNotReady};
use cacti_gpu_cu_ffi::types::*;

//use std::alloc::{Layout};
use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::convert::{TryInto};
use std::rc::{Rc, Weak};
use std::ptr::{write};

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
  //pub ref_:     GpuInnerRef,
  pub dev:      i32,
  pub dptr:     u64,
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

impl InnerCell for GpuInnerCell {
}

/*#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct GpuInnerRef(u32);

impl GpuInnerRef {
  pub fn free() -> GpuInnerRef {
    GpuInnerRef(0)
  }

  pub fn is_free(&self) -> bool {
    self.0 == 0
  }
}*/

#[derive(Clone, Copy, Debug)]
pub struct NvGpuInfo {
  // NB: futhark needs the following.
  pub capability_major: i32,
  pub capability_minor: i32,
  pub compute_mode:     CudaComputeMode,
  pub mp_count:         i32,
  pub max_mp_threads:   i32,
  pub max_blk_threads:  i32,
  pub max_sharedmem:    i32,
  pub max_grid_dim_x:   i32,
  pub warp_size:        i32,
  // NB: the rest is extra.
  pub pci_bus_id:       i32,
  pub pci_device_id:    i32,
  pub pci_domain_id:    i32,
  pub max_pitch:        i32,
  pub integrated:       bool,
  pub unified_address:  bool,
  pub managed_memory:   bool,
}

#[inline]
fn try_i32_to_bool(v: i32) -> Result<bool, i32> {
  Ok(match v {
    0 => false,
    1 => true,
    _ => return Err(v)
  })
}

impl NvGpuInfo {
  pub fn new(dev: i32) -> NvGpuInfo {
    let capability_major = cuda_device_attribute_get(dev, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR).unwrap();
    let capability_minor = cuda_device_attribute_get(dev, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR).unwrap();
    let compute_mode = cuda_device_attribute_get(dev, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE).unwrap().try_into().unwrap();
    let mp_count = cuda_device_attribute_get(dev, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT).unwrap();
    let max_mp_threads = cuda_device_attribute_get(dev, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR).unwrap();
    let max_blk_threads = cuda_device_attribute_get(dev, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK).unwrap();
    let max_sharedmem = cuda_device_attribute_get(dev, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK).unwrap();
    let max_grid_dim_x = cuda_device_attribute_get(dev, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X).unwrap();
    let warp_size = cuda_device_attribute_get(dev, CU_DEVICE_ATTRIBUTE_WARP_SIZE).unwrap();
    let pci_bus_id = cuda_device_attribute_get(dev, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID).unwrap();
    let pci_device_id = cuda_device_attribute_get(dev, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID).unwrap();
    let pci_domain_id = cuda_device_attribute_get(dev, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID).unwrap();
    let max_pitch = cuda_device_attribute_get(dev, CU_DEVICE_ATTRIBUTE_MAX_PITCH).unwrap();
    let integrated = try_i32_to_bool(cuda_device_attribute_get(dev, CU_DEVICE_ATTRIBUTE_INTEGRATED).unwrap()).unwrap();
    let unified_address = try_i32_to_bool(cuda_device_attribute_get(dev, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING).unwrap()).unwrap();
    let managed_memory = try_i32_to_bool(cuda_device_attribute_get(dev, CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY).unwrap()).unwrap();
    NvGpuInfo{
      capability_major,
      capability_minor,
      compute_mode,
      mp_count,
      max_mp_threads,
      max_blk_threads,
      max_sharedmem,
      max_grid_dim_x,
      warp_size,
      pci_bus_id,
      pci_device_id,
      pci_domain_id,
      max_pitch,
      integrated,
      unified_address,
      managed_memory,
    }
  }
}

pub type NvGpuPCtx = GpuPCtx;

pub struct GpuPCtx {
  // FIXME FIXME: work threads for copying.
  //pub iref_ctr:     Cell<u32>,
  //pub dev:          i32,
  pub pctx:         CudaPrimaryCtx,
  pub info:         NvGpuInfo,
  pub main:         CudartStream,
  pub copy_to:      CudartStream,
  pub copy_from:    CudartStream,
  pub mem_pool:     GpuMemPool,
  //pub cel_map:      RefCell<HashMap<GpuInnerRef, Rc<GpuInnerCell>>>,
  pub cel_map:      RefCell<HashMap<CellPtr, Rc<GpuInnerCell>>>,
  // TODO
}

impl GpuPCtx {
  pub fn new(dev: i32) -> Option<GpuPCtx> {
    println!("DEBUG: NvGpuPCtx::new: dev={}", dev);
    if LIBCUDA._inner.is_none() {
      return None;
    }
    cudart_set_cur_dev(dev).unwrap();
    let pctx = CudaPrimaryCtx::retain(dev).unwrap();
    // FIXME: confirm that SCHED_YIELD is what we really want.
    pctx.set_flags(CU_CTX_SCHED_YIELD).unwrap();
    let info = NvGpuInfo::new(dev);
    println!("DEBUG: NvGpuPCtx::new: info={:?}", &info);
    let main = CudartStream::null();
    let copy_to = CudartStream::create_nonblocking().unwrap();
    let copy_from = CudartStream::create_nonblocking().unwrap();
    /*let main = CudartStream::create().unwrap();
    let copy_to = CudartStream::create().unwrap();
    let copy_from = CudartStream::create().unwrap();*/
    let mem_pool = GpuMemPool::new(dev);
    let cel_map = RefCell::new(HashMap::default());
    Some(GpuPCtx{
      //iref_ctr:     Cell::new(0),
      //dev,
      pctx,
      info,
      main,
      copy_to,
      copy_from,
      mem_pool,
      cel_map,
      // TODO
    })
  }

  pub fn dev(&self) -> i32 {
    self.pctx.device()
  }

  pub fn append_matrix(&self, lp: &mut Vec<(Locus, PMach)>, pl: &mut Vec<(PMach, Locus)>) {
    // FIXME: query device/driver capabilities.
    lp.push((Locus::VMem, PMach::NvGpu));
    pl.push((PMach::NvGpu, Locus::VMem));
  }

  pub fn fastest_locus(&self) -> Locus {
    // FIXME: query device/driver capabilities.
    Locus::VMem
  }

  /*fn _fresh_inner_ref(&self) -> GpuInnerRef {
    let next = self.iref_ctr.get() + 1;
    assert!(next > 0);
    assert!(next < u32::max_value());
    self.iref_ctr.set(next);
    GpuInnerRef(next)
  }*/

  pub fn find_dptr(&self, p: CellPtr) -> Option<u64> {
    unimplemented!();
  }

  pub fn fresh_outer(&self) -> Rc<GpuOuterCell> {
    unimplemented!();
  }

  pub fn copy_mem_to_gpu(&self, cel: &GpuInnerCell, src_mem: *const u8, mem_sz: usize) {
    // FIXME
    cuda_memcpy_h2d_async(cel.dptr, src_mem as _, mem_sz, &self.copy_to).unwrap();
    cel.write.set_record();
    cel.write.event.record(&self.copy_to).unwrap();
  }

  pub fn copy_mem_from_gpu(&self, cel: &GpuInnerCell, dst_mem: *mut u8, mem_sz: usize) {
    // FIXME
    self.copy_from.wait_event(&cel.write.event).unwrap();
    cuda_memcpy_d2h_async(dst_mem as _, cel.dptr, mem_sz, &self.copy_from).unwrap();
  }

  pub fn copy_swap_to_gpu(&self, cel: &GpuInnerCell, src: (), mem_sz: usize) {
    unimplemented!();
  }

  pub fn copy_swap_from_gpu(&self, cel: &GpuInnerCell, dst: (), mem_sz: usize) {
    unimplemented!();
  }

  pub fn try_pre_alloc(&self, query_sz: usize) -> Option<(usize, usize, usize)> {
    self.mem_pool.try_front_pre_alloc(query_sz)
  }

  pub fn alloc(&self, ptr: CellPtr, offset: usize, req_sz: usize, next_offset: usize) -> Weak<GpuInnerCell> {
    //let r = self._fresh_inner_ref();
    self.mem_pool.front_set.borrow_mut().insert(ptr, Region{off: offset, sz: req_sz});
    self.mem_pool.front_cursor.set(next_offset);
    let dptr = self.mem_pool.reserve_base + offset as u64;
    /*cudart_set_cur_dev(self.dev()).unwrap();
    let write = Rc::new(CudartEvent::create_fastest().unwrap());
    let lastuse = Rc::new(CudartEvent::create_fastest().unwrap());*/
    let write = GpuSnapshot::fresh(self.dev());
    let lastuse = GpuSnapshot::fresh(self.dev());
    let cel = Rc::new(GpuInnerCell{
      ptr: Cell::new(ptr),
      clk: Cell::new(Clock::default()),
      //ref_: r,
      dev: self.dev(),
      dptr,
      off: offset,
      sz: req_sz,
      write,
      lastuse,
      // FIXME
    });
    println!("DEBUG: GpuPCtx::alloc: p={:?} dptr=0x{:016x} sz={} off={}", ptr, dptr, req_sz, offset);
    assert!(self.mem_pool.alloc_map.borrow_mut().insert(Region{off: offset, sz: req_sz}, ptr).is_none());
    let xcel = Rc::downgrade(&cel);
    self.cel_map.borrow_mut().insert(ptr, cel);
    xcel
  }

  //pub fn try_free(&self, r: GpuInnerRef) -> Option<()> {}
  pub fn try_free(&self, ptr: CellPtr) -> Option<()> {
    match self.cel_map.borrow().get(&ptr) {
      None => {}
      Some(cel) => {
        // FIXME FIXME: think about this ref count check.
        /*let c = Rc::strong_count(cel);
        if c > 1 {
          return None;
        }*/
        let mut sw = SpinWait::default();
        cel.write.wait(&mut sw);
        cel.lastuse.wait(&mut sw);
        // FIXME FIXME: remove from front.
        drop(cel);
        self.mem_pool.front_set.borrow_mut().remove(&ptr);
        assert_eq!(self.mem_pool.alloc_map.borrow_mut().remove(&Region{off: cel.off, sz: cel.sz}), Some(ptr));
        self.cel_map.borrow_mut().remove(&ptr);
      }
    }
    Some(())
  }

  pub fn unify(&self, x: CellPtr, y: CellPtr) {
    let mut cel_map = self.cel_map.borrow_mut();
    match cel_map.remove(&x) {
      None => panic!("bug"),
      Some(mut cel) => {
        assert_eq!(cel.ptr.get(), x);
        cel.ptr.set(y);
        let mut alloc_map = self.mem_pool.alloc_map.borrow_mut();
        match alloc_map.remove(&Region{off: cel.off, sz: cel.sz}) {
          None => panic!("bug"),
          Some(ox) => {
            assert_eq!(ox, x);
          }
        }
        alloc_map.insert(Region{off: cel.off, sz: cel.sz}, y);
        cel_map.insert(y, cel);
      }
    }
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

pub struct GpuMemPool {
  pub dev:          i32,
  pub reserve_base: u64,
  pub reserve_sz:   usize,
  pub front_pad:    usize,
  pub front_base:   u64,
  pub front_sz:     usize,
  pub boundary_pad: usize,
  pub back_base:    u64,
  pub back_sz:      usize,
  pub back_pad:     usize,
  //pub front_list:   RefCell<MergeVecDeque<(GpuInnerRef, GpuInnerMemCell)>>,
  //pub front_set:    RefCell<BTreeMap<GpuInnerRef, Region>>,
  pub front_set:    RefCell<HashMap<CellPtr, Region>>,
  pub front_cursor: Cell<usize>,
  //pub back_bitmap:  RefCell<Bitvec64>,
  pub back_cursor:  Cell<usize>,
  //pub tmp_freelist: ExtentVecList,
  //pub alloc_map:    RefCell<BTreeMap<Region, GpuInnerRef>>,
  pub alloc_map:    RefCell<BTreeMap<Region, CellPtr>>,
  pub free_set:     RefCell<BTreeSet<Region>>,
  pub free_merge:   RefCell<MergeVecDeque<Region>>,
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
    match cuda_mem_free(self.reserve_base) {
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
      panic!("ERROR: GpuPCtx::new: gpu oom: tried to reserve {} bytes on gpu {}, but only found {} bytes free (out of {} bytes total)",
          reserve_sz, dev, free_sz, total_sz);
    }
    println!("DEBUG: GpuMemPool::new: reserve sz={}", reserve_sz);
    let reserve_base = cuda_mem_alloc(reserve_sz).unwrap();
    CudartStream::null().sync().unwrap();
    println!("DEBUG: GpuMemPool::new: reserve base=0x{:016x}", reserve_base);
    let reserve_warp_offset = reserve_base & (128 - 1);
    if reserve_warp_offset != 0 {
      panic!("ERROR: GpuPCtx::new: gpu bug: misaligned alloc, offset by {} bytes (expected alignment {} bytes)",
          reserve_warp_offset, 128);
    }
    let front_pad = (1 << 16);
    let boundary_pad = (1 << 16);
    let back_pad = (1 << 16);
    let back_sz = (1 << 22);
    let front_sz = reserve_sz - (front_pad + boundary_pad + back_sz + back_pad);
    let front_base = reserve_base + front_pad as u64;
    let back_base = reserve_base + (front_pad + front_sz + boundary_pad) as u64;
    assert!(front_sz >= (1 << 26));
    println!("DEBUG: GpuMemPool::new: front sz={}", front_sz);
    println!("DEBUG: GpuMemPool::new: back sz={}", back_sz);
    GpuMemPool{
      dev,
      reserve_base,
      reserve_sz,
      front_pad,
      front_base,
      front_sz,
      boundary_pad,
      back_base,
      back_sz,
      back_pad,
      //front_list:   RefCell::new(MergeVecDeque::new()),
      //front_set:    RefCell::new(BTreeMap::new()),
      front_set:    RefCell::new(HashMap::new()),
      front_cursor: Cell::new(0),
      //back_bitmap:  RefCell::new(Bitvec64::new()),
      back_cursor:  Cell::new(back_sz),
      //tmp_freelist: ExtentVecList::default(),
      alloc_map:    RefCell::new(BTreeMap::new()),
      free_set:     RefCell::new(BTreeSet::new()),
      free_merge:   RefCell::new(MergeVecDeque::new()),
      // TODO
    }
  }

  //pub fn lookup_dptr(&self, query_dptr: u64) -> Option<(Region, Option<GpuInnerRef>)> {}
  pub fn lookup_dptr(&self, query_dptr: u64) -> Option<(Region, Option<CellPtr>)> {
    println!("DEBUG: GpuMemPool::lookup_dptr: query dptr=0x{:016x}", query_dptr);
    println!("DEBUG: GpuMemPool::lookup_dptr:   res base=0x{:016x}", self.reserve_base);
    if query_dptr < self.reserve_base {
      return None;
    } else if query_dptr >= self.reserve_base + self.reserve_sz as u64 {
      return None;
    }
    println!("DEBUG: GpuMemPool::lookup_dptr:   allocmap={:?}", &self.alloc_map);
    println!("DEBUG: GpuMemPool::lookup_dptr:   free set={:?}", &self.free_set);
    let query_off = (query_dptr - self.reserve_base) as usize;
    let q = Region{off: query_off, sz: 0};
    if self.alloc_map.borrow().len() <= self.free_set.borrow().len() {
      match self.alloc_map.borrow().range(q .. ).next() {
        None => {}
        Some((&k, &v)) => if k.off == query_off {
          assert!(k.sz > 0);
          return Some((k, Some(v)));
        }
      }
      match self.free_set.borrow().range(q .. ).next() {
        None => {}
        Some(&k) => if k.off == query_off {
          assert!(k.sz > 0);
          return Some((k, None));
        }
      }
    } else {
      match self.free_set.borrow().range(q .. ).next() {
        None => {}
        Some(&k) => if k.off == query_off {
          assert!(k.sz > 0);
          return Some((k, None));
        }
      }
      match self.alloc_map.borrow().range(q .. ).next() {
        None => {}
        Some((&k, &v)) => if k.off == query_off {
          assert!(k.sz > 0);
          return Some((k, Some(v)));
        }
      }
    }
    None
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

  //pub fn find_front_lru_match(&self, query_sz: usize) -> Option<GpuInnerRef> {}
  pub fn find_front_lru_match(&self, query_sz: usize) -> Option<CellPtr> {
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

  pub fn back_alloc(&self, sz: usize) -> u64 {
  //pub fn try_back_alloc(&self, sz: usize) -> (Option<GpuInnerRef>, Option<Weak<GpuInnerCell>>) {}
    // FIXME: currently, we only use this to alloc int32's for futhark error vars.
    assert_eq!(sz, 4);
    let prev_curs = self.back_cursor.get();
    let next_curs = prev_curs - 4;
    let dptr = self.back_base + next_curs as u64;
    self.back_cursor.set(next_curs);
    println!("DEBUG: GpuMemPool::back_alloc: dptr=0x{:016x} sz={}", dptr, sz);
    dptr
  }

  pub fn back_free_all(&self) {
    // FIXME FIXME
    //unimplemented!();
  }
}

pub extern "C" fn tl_pctx_gpu_alloc_hook(dptr: *mut u64, sz: usize) -> i32 {
  assert!(!dptr.is_null());
  TL_PCTX.with(|pctx| {
    match pctx.nvgpu.as_ref().unwrap().try_pre_alloc(sz) {
      None => panic!("bug"),
      Some((offset, req_sz, next_offset)) => {
        // FIXME FIXME
        let p = ctx_fresh_tmp();
        let x = pctx.nvgpu.as_ref().unwrap().alloc(p, offset, req_sz, next_offset);
        let y = Weak::upgrade(&x).unwrap();
        unsafe {
          write(dptr, y.dptr);
        }
      }
    }
    0
  })
}

pub extern "C" fn tl_pctx_gpu_free_hook(dptr: u64) -> i32 {
  TL_PCTX.try_with(|_pctx| {
    // FIXME FIXME
    0
  }).unwrap_or_else(|_| 0)
}

pub extern "C" fn tl_pctx_gpu_back_alloc_hook(dptr: *mut u64, sz: usize) -> i32 {
  assert!(!dptr.is_null());
  TL_PCTX.with(|pctx| {
    // FIXME FIXME
    let x = pctx.nvgpu.as_ref().unwrap().mem_pool.back_alloc(sz);
    unsafe {
      write(dptr, x);
    }
    0
  })
}

pub extern "C" fn tl_pctx_gpu_back_free_hook(dptr: u64) -> i32 {
  TL_PCTX.try_with(|_pctx| {
    // FIXME FIXME
    0
  }).unwrap_or_else(|_| 0)
}

pub struct GpuDryCtx {
  // FIXME
}