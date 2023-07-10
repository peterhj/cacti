use super::*;
use crate::algo::{BTreeMap, BTreeSet, HashMap, Region, RevSortMap8};
use crate::algo::str::*;
use crate::algo::sync::{SpinWait};
use crate::cell::*;
use crate::clock::*;
use crate::ctx::*;
use cacti_gpu_cu_ffi::*;
use cacti_gpu_cu_ffi::types::*;

use libc::{c_char};

use std::cell::{Cell, RefCell};
use std::convert::{TryInto};
use std::ffi::{CStr, c_void};
use std::rc::{Rc, Weak};
use std::ptr::{write};

pub const ALLOC_ALIGN: usize = 256;

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

#[derive(Clone, Copy, Debug)]
pub enum NvGpuInnerReg {
  Mem{ptr: *mut c_void, size: usize},
  VMem{dptr: u64, size: usize},
}

impl NvGpuInnerReg {
  pub fn locus(&self) -> Locus {
    match self {
      &NvGpuInnerReg::Mem{..} => Locus::Mem,
      &NvGpuInnerReg::VMem{..} => Locus::VMem,
    }
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

pub struct GpuOuterCell_ {
  // FIXME FIXME
  //pub ptr:      Cell<CellPtr>,
  pub ptr:      Cell<PAddr>,
  pub clk:      Cell<Clock>,
  pub mem:      MemReg,
  pub write:    Rc<GpuSnapshot>,
  pub lastuse:  Rc<GpuSnapshot>,
}

pub type GpuInnerCell = NvGpuInnerCell;

pub struct NvGpuInnerCell {
  pub addr: PAddr,
  pub root: Cell<CellPtr>,
  pub flag: Cell<u8>,
  pub tag:  Cell<u32>,
  pub dev:  i32,
  pub dptr: Cell<u64>,
  pub sz:   usize,
  //pub write:    Rc<CudartEvent>,
  //pub lastuse:  Rc<CudartEvent>,
  //pub write:    Rc<GpuSnapshot>,
  //pub lastuse:  Rc<GpuSnapshot>,
  //pub lastcopy: Rc<GpuSnapshot>,
  // FIXME
  //pub smp_dep:  Option<Rc<SmpInnerCell>>,
  // TODO
}

impl NvGpuInnerCell {
  pub fn front(&self) -> bool {
    (self.flag.get() & 1) != 0
  }

  pub fn set_front(&self, flag: bool) {
    if flag {
      self.flag.set(self.flag.get() | 1);
    } else {
      self.flag.set(self.flag.get() & !1);
    }
  }

  pub fn back(&self) -> bool {
    (self.flag.get() & 2) != 0
  }

  pub fn set_back(&self, flag: bool) {
    if flag {
      self.flag.set(self.flag.get() | 2);
    } else {
      self.flag.set(self.flag.get() & !2);
    }
  }
}

impl InnerCell for NvGpuInnerCell {
  fn size(&self) -> usize {
    self.sz
  }

  fn root(&self) -> Option<CellPtr> {
    let x = self.root.get();
    if x.is_nil() {
      None
    } else {
      Some(x)
    }
  }

  fn set_root(&self, x: Option<CellPtr>) {
    let x = x.unwrap_or(CellPtr::nil());
    self.root.set(x);
  }

  fn pin(&self) -> bool {
    (self.flag.get() & 4) != 0
  }

  fn set_pin(&self, flag: bool) {
    if flag {
      self.flag.set(self.flag.get() | 4);
    } else {
      self.flag.set(self.flag.get() & !4);
    }
  }

  fn tag(&self) -> Option<u32> {
    if (self.flag.get() & 8) != 0 {
      Some(self.tag.get())
    } else {
      None
    }
  }

  fn set_tag(&self, tag: Option<u32>) {
    if let Some(val) = tag {
      self.flag.set(self.flag.get() | 8);
      self.tag.set(val);
    } else {
      self.flag.set(self.flag.get() & !8);
      self.tag.set(0);
    }
  }
}

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

pub type GpuPCtx = NvGpuPCtx;

pub struct NvGpuPCtx {
  // FIXME FIXME: work threads for copying.
  //pub iref_ctr:     Cell<u32>,
  //pub dev:          i32,
  pub pctx:         CudaPrimaryCtx,
  pub info:         NvGpuInfo,
  pub blas_ctx:     CublasContext,
  pub compute:      CudartStream,
  pub copy_to:      CudartStream,
  pub copy_from:    CudartStream,
  pub page_map:     NvGpuPageMap,
  pub mem_pool:     NvGpuMemPool,
  //pub cel_map:      RefCell<HashMap<GpuInnerRef, Rc<GpuInnerCell>>>,
  //pub cel_map:      RefCell<HashMap<CellPtr, Rc<GpuInnerCell>>>,
  //pub cel_map:      RefCell<HashMap<PAddr, Rc<GpuInnerCell>>>,
  // TODO
}

impl PCtxImpl for NvGpuPCtx {
  fn pmach(&self) -> PMach {
    PMach::NvGpu
  }

  fn fastest_locus(&self) -> Locus {
    self.device_locus()
  }

  fn append_matrix(&self, lp: &mut RevSortMap8<(Locus, PMach), ()>, pl: &mut RevSortMap8<(PMach, Locus), ()>) {
    if self.info.integrated {
      println!("DEBUG: NvGpuPCtx::append_matrix: integrated: capability={}.{}",
          self.info.capability_major, self.info.capability_minor);
      if !self.info.unified_address {
        println!("WARNING: NvGpuPCtx::append_matrix: integrated but not unified address space");
      }
      if !self.info.managed_memory {
        println!("WARNING: NvGpuPCtx::append_matrix: integrated but not managed memory");
      }
    } else {
      lp.insert((Locus::VMem, PMach::NvGpu), ());
      pl.insert((PMach::NvGpu, Locus::VMem), ());
    }
    lp.insert((Locus::Mem, PMach::NvGpu), ());
    pl.insert((PMach::NvGpu, Locus::Mem), ());
  }

  fn try_alloc(&self, pctr: &PCtxCtr, /*pmset: PMachSet,*/ locus: Locus, ty: &CellType) -> Result<PAddr, PMemErr> {
    let sz = ty.packed_span_bytes() as usize;
    match locus {
      Locus::Mem => {
        let mem = NvGpuMemCell::try_alloc(sz)?;
        let addr = pctr.fresh_addr();
        assert!(self.page_map.page_tab.borrow_mut().insert(addr, Rc::new(mem)).is_none());
        Ok(addr)
      }
      Locus::VMem => {
        let query_sz = ty.packed_span_bytes() as usize;
        let req = match self.mem_pool.try_pre_alloc(query_sz) {
          NvGpuMemPoolReq::Oom => {
            // FIXME FIXME: oom.
            unimplemented!();
          }
          req => req
        };
        let addr = pctr.fresh_addr();
        let _ = self.mem_pool.alloc(addr, req);
        Ok(addr)
      }
      _ => unimplemented!()
    }
  }
}

impl NvGpuPCtx {
  pub fn dev_count() -> i32 {
    let n = cuda_device_get_count().unwrap_or(0);
    for i in 0 .. n {
      assert_eq!(Ok(i), cuda_device_get(i));
    }
    n
  }

  pub fn new(dev: i32) -> Option<NvGpuPCtx> {
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
    let blas_ctx = CublasContext::create().unwrap();
    let compute = CudartStream::null();
    let copy_to = CudartStream::create_nonblocking().unwrap();
    let copy_from = CudartStream::create_nonblocking().unwrap();
    /*let compute = CudartStream::create().unwrap();
    let copy_to = CudartStream::create().unwrap();
    let copy_from = CudartStream::create().unwrap();*/
    let page_map = NvGpuPageMap::new();
    let mem_pool = NvGpuMemPool::new(dev);
    //let cel_map = RefCell::new(HashMap::default());
    Some(NvGpuPCtx{
      //iref_ctr:     Cell::new(0),
      //dev,
      pctx,
      info,
      blas_ctx,
      compute,
      copy_to,
      copy_from,
      page_map,
      mem_pool,
      //cel_map,
      // TODO
    })
  }

  pub fn dev(&self) -> i32 {
    self.pctx.device()
  }

  pub fn device_locus(&self) -> Locus {
    if self.info.integrated {
      println!("DEBUG: NvGpuPCtx::device_locus: integrated: capability={}.{}",
          self.info.capability_major, self.info.capability_minor);
      if !self.info.unified_address {
        println!("WARNING: NvGpuPCtx::device_locus: integrated but not unified address space");
      }
      if !self.info.managed_memory {
        println!("WARNING: NvGpuPCtx::device_locus: integrated but not managed memory");
      }
      Locus::Mem
    } else {
      Locus::VMem
    }
  }

  /*fn _fresh_inner_ref(&self) -> GpuInnerRef {
    let next = self.iref_ctr.get() + 1;
    assert!(next > 0);
    assert!(next < u32::max_value());
    self.iref_ctr.set(next);
    GpuInnerRef(next)
  }*/

  /*//pub fn find_dptr(&self, p: CellPtr) -> Option<u64> {}
  pub fn find_dptr(&self, p: PAddr) -> Option<u64> {
    unimplemented!();
  }*/

  pub fn fresh_outer(&self) -> Rc<GpuOuterCell> {
    unimplemented!();
  }

  pub fn hard_copy(&self, dst_loc: Locus, dst: PAddr, src_loc: Locus, src: PAddr) {
    match (dst_loc, src_loc) {
      (Locus::VMem, Locus::Mem) => {
        let (dst_dptr, dst_sz) = match self.lookup_reg(dst) {
          Some(NvGpuInnerReg::VMem{dptr, size}) => (dptr, size),
          _ => panic!("bug")
        };
        let (src_ptr, src_sz) = match self.lookup_reg(src) {
          Some(NvGpuInnerReg::Mem{ptr, size}) => (ptr, size),
          _ => panic!("bug")
        };
        assert_eq!(dst_sz, src_sz);
        println!("DEBUG: NvGpuPCtx::hard_copy: dst dptr=0x{:016x} src ptr=0x{:016x} sz={}",
            dst_dptr, src_ptr as usize, src_sz);
        self.compute.sync().unwrap();
        cuda_memcpy_h2d_async(dst_dptr, src_ptr, src_sz, &self.compute).unwrap();
        self.compute.sync().unwrap();
      }
      _ => unimplemented!()
    }
  }

  pub fn hard_copy_raw_mem_to_vmem(&self, dst_dptr: u64, src_ptr: *const c_void, sz: usize) {
    println!("DEBUG: NvGpuPCtx::hard_copy_raw_mem_to_vmem: dst dptr=0x{:016x} src ptr=0x{:016x} sz={}",
        dst_dptr, src_ptr as usize, sz);
    self.compute.sync().unwrap();
    cuda_memcpy_h2d_async(dst_dptr, src_ptr, sz, &self.compute).unwrap();
    self.compute.sync().unwrap();
  }

  pub fn hard_copy_nb_raw_mem_to_vmem(&self, dst_dptr: u64, src_ptr: *const c_void, sz: usize) {
    println!("DEBUG: NvGpuPCtx::hard_copy_nb_raw_mem_to_vmem: dst dptr=0x{:016x} src ptr=0x{:016x} sz={}",
        dst_dptr, src_ptr as usize, sz);
    cuda_memcpy_h2d_async(dst_dptr, src_ptr, sz, &self.compute).unwrap();
  }

  /*pub fn copy_mem_to_gpu(&self, cel: &GpuInnerCell, src_mem: *const u8, mem_sz: usize) {
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
  }*/

  /*pub fn try_pre_alloc(&self, query_sz: usize) -> Option<(usize, usize, usize)> {
    self.mem_pool.try_pre_alloc(query_sz)
  }*/

  /*pub fn alloc(&self, addr: PAddr, offset: usize, req_sz: usize, next_offset: usize) /*-> Weak<GpuInnerCell> */{
    //let r = self._fresh_inner_ref();
    self.mem_pool.alloc_map.borrow_mut().insert(ptr, Region{off: offset, sz: req_sz});
    self.mem_pool.front_cursor.set(next_offset);
    let dptr = self.mem_pool.front_base + offset as u64;
    /*cudart_set_cur_dev(self.dev()).unwrap();
    let write = Rc::new(CudartEvent::create_fastest().unwrap());
    let lastuse = Rc::new(CudartEvent::create_fastest().unwrap());*/
    //let write = GpuSnapshot::fresh(self.dev());
    //let lastuse = GpuSnapshot::fresh(self.dev());
    let cel = Rc::new(GpuInnerCell{
      ptr: Cell::new(ptr),
      clk: Cell::new(Clock::default()),
      //ref_: r,
      dev: self.dev(),
      dptr,
      off: offset,
      sz: req_sz,
      //write,
      //lastuse,
      // FIXME
    });
    println!("DEBUG: NvGpuPCtx::alloc: p={:?} dptr=0x{:016x} sz={} off={}", ptr, dptr, req_sz, offset);
    //println!("DEBUG: NvGpuPCtx::alloc:   front={}", &self.mem_pool.front_cursor.get());
    assert!(self.mem_pool.alloc_index.borrow_mut().insert(Region{off: offset, sz: req_sz}, ptr).is_none());
    let xcel = Rc::downgrade(&cel);
    self.mem_pool.cel_map.borrow_mut().insert(ptr, cel);
    xcel
  }*/

  /*//pub fn try_free(&self, r: GpuInnerRef) -> Option<()> {}
  //pub fn try_free(&self, ptr: CellPtr) -> Option<()> {}
  pub fn try_free(&self, ptr: PAddr) -> Option<()> {
    match self.mem_pool.cel_map.borrow().get(&ptr) {
      None => {}
      Some(cel) => {
        // FIXME FIXME: think about this ref count check.
        /*let c = Rc::strong_count(cel);
        if c > 1 {
          return None;
        }*/
        /*let mut sw = SpinWait::default();
        cel.write.wait(&mut sw);
        cel.lastuse.wait(&mut sw);*/
        // FIXME FIXME: remove from front.
        drop(cel);
        self.mem_pool.alloc_map.borrow_mut().remove(&ptr);
        assert_eq!(self.mem_pool.alloc_index.borrow_mut().remove(&Region{off: cel.off, sz: cel.sz}), Some(ptr));
        self.mem_pool.cel_map.borrow_mut().remove(&ptr);
      }
    }
    Some(())
  }*/

  /*// FIXME FIXME
  //pub fn unify(&self, x: CellPtr, y: CellPtr) {}
  pub fn unify(&self, x: PAddr, y: PAddr) {
    let mut cel_map = self.cel_map.borrow_mut();
    match cel_map.remove(&x) {
      None => panic!("bug"),
      Some(cel) => {
        assert_eq!(cel.ptr.get(), x);
        cel.ptr.set(y);
        let mut alloc_index = self.mem_pool.alloc_index.borrow_mut();
        match alloc_index.remove(&Region{off: cel.off, sz: cel.sz}) {
          None => panic!("bug"),
          Some(ox) => {
            assert_eq!(ox, x);
          }
        }
        alloc_index.insert(Region{off: cel.off, sz: cel.sz}, y);
        cel_map.insert(y, cel);
      }
    }
  }*/

  /*pub fn lookup(&self, x: PAddr) -> Option<Rc<NvGpuInnerCell>> {
    match self.cel_map.borrow().get(&x) {
      None => None,
      Some(icel) => Some(icel.clone())
    }
  }*/

  pub fn lookup_(&self, x: PAddr) -> Option<(Locus, Rc<dyn InnerCell_>)> {
    self.lookup(x)
  }

  pub fn lookup(&self, x: PAddr) -> Option<(Locus, Rc<dyn InnerCell_>)> {
    let page_tab = self.page_map.page_tab.borrow();
    let cel_map = self.mem_pool.cel_map.borrow();
    if page_tab.len() <= cel_map.len() {
      match page_tab.get(&x) {
        None => {}
        Some(icel) => {
          return Some((Locus::Mem, icel.clone()));
        }
      }
      match cel_map.get(&x) {
        None => {}
        Some(icel) => {
          return Some((Locus::VMem, icel.clone()));
        }
      }
    } else {
      match cel_map.get(&x) {
        None => {}
        Some(icel) => {
          return Some((Locus::VMem, icel.clone()));
        }
      }
      match page_tab.get(&x) {
        None => {}
        Some(icel) => {
          return Some((Locus::Mem, icel.clone()));
        }
      }
    }
    None
  }

  /*pub fn find_reg(&self, p: PAddr) -> Option<NvGpuInnerReg> {
    self.lookup_reg(p)
  }*/

  pub fn lookup_reg(&self, p: PAddr) -> Option<NvGpuInnerReg> {
    let page_tab = self.page_map.page_tab.borrow();
    let cel_map = self.mem_pool.cel_map.borrow();
    if page_tab.len() <= cel_map.len() {
      match page_tab.get(&p) {
        None => {}
        Some(icel) => {
          return Some(NvGpuInnerReg::Mem{
            ptr:  icel.ptr,
            size: icel.sz,
          });
        }
      }
      match cel_map.get(&p) {
        None => {}
        Some(icel) => {
          return Some(NvGpuInnerReg::VMem{
            dptr: icel.dptr.get(),
            size: icel.sz,
          });
        }
      }
    } else {
      match cel_map.get(&p) {
        None => {}
        Some(icel) => {
          return Some(NvGpuInnerReg::VMem{
            dptr: icel.dptr.get(),
            size: icel.sz,
          });
        }
      }
      match page_tab.get(&p) {
        None => {}
        Some(icel) => {
          return Some(NvGpuInnerReg::Mem{
            ptr:  icel.ptr,
            size: icel.sz,
          });
        }
      }
    }
    None
  }

  pub fn lookup_dev(&self, p: PAddr) -> Option<(u64, usize)> {
    let page_tab = self.page_map.page_tab.borrow();
    let cel_map = self.mem_pool.cel_map.borrow();
    if page_tab.len() <= cel_map.len() {
      match page_tab.get(&p) {
        None => {}
        Some(icel) => {
          assert!(self.info.unified_address);
          #[cfg(not(target_pointer_width = "64"))]
          unimplemented!();
          #[cfg(target_pointer_width = "64")]
          return Some((icel.ptr as usize as u64, icel.sz));
        }
      }
      match cel_map.get(&p) {
        None => {}
        Some(icel) => {
          return Some((icel.dptr.get(), icel.sz));
        }
      }
    } else {
      match cel_map.get(&p) {
        None => {}
        Some(icel) => {
          return Some((icel.dptr.get(), icel.sz));
        }
      }
      match page_tab.get(&p) {
        None => {}
        Some(icel) => {
          assert!(self.info.unified_address);
          #[cfg(not(target_pointer_width = "64"))]
          unimplemented!();
          #[cfg(target_pointer_width = "64")]
          return Some((icel.ptr as usize as u64, icel.sz));
        }
      }
    }
    None
  }
}

#[repr(C)]
pub struct NvGpuMemCell {
  //pub addr: PAddr,
  pub root: Cell<CellPtr>,
  pub flag: Cell<u8>,
  pub ptr:  *mut c_void,
  pub sz:   usize,
}

impl Drop for NvGpuMemCell {
  fn drop(&mut self) {
    if self.ptr.is_null() {
      return;
    }
    unsafe {
      match cuda_mem_free_host(self.ptr) {
        Ok(_) => {}
        Err(CUDA_ERROR_DEINITIALIZED) => {}
        Err(_) => panic!("bug"),
      }
    }
  }
}

impl NvGpuMemCell {
  pub fn try_alloc(sz: usize) -> Result<NvGpuMemCell, PMemErr> {
    let ptr = match cuda_mem_alloc_host(sz) {
      Err(CUDA_ERROR_OUT_OF_MEMORY) => {
        return Err(PMemErr::Oom);
      }
      Err(_) => {
        return Err(PMemErr::Bot);
      }
      Ok(ptr) => ptr
    };
    if ptr.is_null() {
      return Err(PMemErr::Bot);
    }
    println!("DEBUG: NvGpuMemCell::try_alloc: ptr=0x{:016x} sz={}", ptr as usize, sz);
    Ok(NvGpuMemCell{
      root: Cell::new(CellPtr::nil()),
      flag: Cell::new(0),
      ptr,
      sz,
    })
  }
}

impl InnerCell for NvGpuMemCell {
  fn as_mem_reg(&self) -> Option<MemReg> {
    Some(MemReg{ptr: self.ptr, sz: self.sz})
  }

  fn size(&self) -> usize {
    self.sz
  }

  fn root(&self) -> Option<CellPtr> {
    let x = self.root.get();
    if x.is_nil() {
      None
    } else {
      Some(x)
    }
  }

  fn set_root(&self, x: Option<CellPtr>) {
    let x = x.unwrap_or(CellPtr::nil());
    self.root.set(x);
  }

  fn pin(&self) -> bool {
    (self.flag.get() & 4) != 0
  }

  fn set_pin(&self, flag: bool) {
    if flag {
      self.flag.set(self.flag.get() | 4);
    } else {
      self.flag.set(self.flag.get() & !4);
    }
  }
}

pub struct NvGpuPageMap {
  pub page_tab: RefCell<HashMap<PAddr, Rc<NvGpuMemCell>>>,
  pub extrabuf: Rc<NvGpuMemCell>,
}

impl NvGpuPageMap {
  pub fn new() -> NvGpuPageMap {
    let mem = match NvGpuMemCell::try_alloc(1 << 16) {
      Err(_) => {
        println!("ERROR: NvGpuPageMap: failed to allocate shadow back buffer");
        panic!();
      }
      Ok(mem) => mem
    };
    NvGpuPageMap{
      page_tab: RefCell::new(HashMap::new()),
      extrabuf: Rc::new(mem),
    }
  }

  /*pub fn try_alloc(&mut self, sz: usize) -> Result<PAddr, PMemErr> {
  }*/
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

#[derive(Clone, Copy, Debug)]
pub enum NvGpuMemPoolReq {
  Oom,
  Front{offset: usize, next_offset: usize},
  Back{offset: usize, next_offset: usize},
}

impl NvGpuMemPoolReq {
  pub fn size(&self) -> usize {
    match self {
      NvGpuMemPoolReq::Oom => panic!("bug"),
      NvGpuMemPoolReq::Front{offset, next_offset} |
      NvGpuMemPoolReq::Back{offset, next_offset} => {
        next_offset - offset
      }
    }
  }
}

#[derive(Clone)]
pub struct NvGpuAlloc {
  pub reg:  Region,
}

pub type GpuMemPool = NvGpuMemPool;

pub struct NvGpuMemPool {
  pub dev:          i32,
  pub reserve_base: u64,
  pub reserve_sz:   usize,
  pub front_pad:    usize,
  pub front_base:   u64,
  pub front_sz:     usize,
  pub boundary_pad: usize,
  pub extra_base:   u64,
  pub extra_sz:     usize,
  pub extra_pad:    usize,
  pub front_cursor: Cell<usize>,
  pub back_cursor:  Cell<usize>,
  pub back_alloc:   Cell<bool>,
  pub alloc_pin:    Cell<bool>,
  pub front_tag:    RefCell<Option<u32>>,
  pub free_list:    RefCell<Vec<PAddr>>,
  pub free_index:   RefCell<BTreeSet<Region>>,
  pub size_index:   RefCell<HashMap<usize, BTreeSet<PAddr>>>,
  pub alloc_index:  RefCell<BTreeMap<Region, PAddr>>,
  pub alloc_map:    RefCell<HashMap<PAddr, NvGpuAlloc>>,
  pub cel_map:      RefCell<HashMap<PAddr, Rc<NvGpuInnerCell>>>,
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
    println!("DEBUG: NvGpuMemPool::new");
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
    let reserve_warp_offset = reserve_base & (ALLOC_ALIGN as u64 - 1);
    if reserve_warp_offset != 0 {
      panic!("ERROR: GpuPCtx::new: gpu bug: misaligned alloc, offset by {} bytes (expected alignment {} bytes)",
          reserve_warp_offset, ALLOC_ALIGN);
    }
    let front_pad = (1 << 16);
    let boundary_pad = (1 << 16);
    let extra_pad = (1 << 16);
    let extra_sz = (1 << 16);
    let front_sz = reserve_sz - (front_pad + boundary_pad + extra_sz + extra_pad);
    let front_base = reserve_base + front_pad as u64;
    let extra_base = reserve_base + (front_pad + front_sz + boundary_pad) as u64;
    assert!(front_sz >= (1 << 26));
    println!("DEBUG: NvGpuMemPool::new: front sz={}", front_sz);
    println!("DEBUG: NvGpuMemPool::new: back sz={}", extra_sz);
    GpuMemPool{
      dev,
      reserve_base,
      reserve_sz,
      front_pad,
      front_base,
      front_sz,
      boundary_pad,
      extra_base,
      extra_sz,
      extra_pad,
      front_cursor: Cell::new(0),
      back_cursor:  Cell::new(0),
      back_alloc:   Cell::new(false),
      alloc_pin:    Cell::new(false),
      front_tag:    RefCell::new(None),
      free_list:    RefCell::new(Vec::new()),
      free_index:   RefCell::new(BTreeSet::new()),
      size_index:   RefCell::new(HashMap::new()),
      alloc_index:  RefCell::new(BTreeMap::new()),
      alloc_map:    RefCell::new(HashMap::new()),
      cel_map:      RefCell::new(HashMap::new()),
      // TODO
    }
  }

  pub fn front_dptr(&self) -> u64 {
    self.front_base + self.front_cursor.get() as u64
  }

  pub fn front_offset(&self) -> usize {
    self.front_cursor.get()
  }

  pub fn back_offset(&self) -> usize {
    self.back_cursor.get()
  }

  pub fn set_back_offset(&self, off: usize) {
    self.back_cursor.set(off);
  }

  pub fn set_back_alloc(&self, flag: bool) {
    self.back_alloc.set(flag);
  }

  pub fn set_alloc_pin(&self, flag: bool) {
    self.alloc_pin.set(flag);
  }

  //pub fn set_front_tag(&self, tag: Option<&[u8]>) {}
  pub fn set_front_tag(&self, tag: Option<u32>) {
    match tag {
      None => {
        *self.front_tag.borrow_mut() = None;
      }
      Some(tag) => {
        // *self.front_tag.borrow_mut() = Some(buf.to_owned());
        *self.front_tag.borrow_mut() = Some(tag);
      }
    }
  }

  //pub fn lookup_dptr(&self, query_dptr: u64) -> Option<(Region, Option<GpuInnerRef>)> {}
  //pub fn lookup_dptr(&self, query_dptr: u64) -> Option<(Region, Option<CellPtr>)> {}
  pub fn lookup_dptr(&self, query_dptr: u64) -> Option<(Region, Option<PAddr>)> {
    //println!("DEBUG: NvGpuMemPool::lookup_dptr: query dptr=0x{:016x}", query_dptr);
    //println!("DEBUG: NvGpuMemPool::lookup_dptr: front base=0x{:016x}", self.front_base);
    if query_dptr < self.front_base {
      return None;
    } else if query_dptr >= self.front_base + self.front_sz as u64 {
      return None;
    }
    //println!("DEBUG: GpuMemPool::lookup_dptr:   allocmap={:?}", &self.alloc_index);
    //println!("DEBUG: GpuMemPool::lookup_dptr:   free set={:?}", &self.free_set);
    let query_off = (query_dptr - self.front_base) as usize;
    let q = Region{off: query_off, sz: 0};
    match self.alloc_index.borrow().range(q .. ).next() {
      None => {}
      Some((&k, &v)) => if k.off == query_off {
        assert!(k.sz > 0);
        return Some((k, Some(v)));
      }
    }
    /*if self.alloc_index.borrow().len() <= self.free_set.borrow().len() {
      match self.alloc_index.borrow().range(q .. ).next() {
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
      match self.alloc_index.borrow().range(q .. ).next() {
        None => {}
        Some((&k, &v)) => if k.off == query_off {
          assert!(k.sz > 0);
          return Some((k, Some(v)));
        }
      }
    }*/
    None
  }

  //pub fn try_pre_alloc_with_tag(&self, query_sz: usize, query_tag: &[u8]) -> NvGpuMemPoolReq {}
  pub fn try_pre_alloc_with_tag(&self, query_sz: usize, query_tag: u32, unify: &mut TagUnifier) -> NvGpuMemPoolReq {
    if let Some(&tag) = self.front_tag.borrow().as_ref() {
      if unify.find(query_tag) == unify.find(tag) {
        return self._try_front_pre_alloc(query_sz);
      }
    }
    if self.back_alloc.get() {
      self._try_back_pre_alloc(query_sz)
    } else {
      self._try_front_pre_alloc(query_sz)
    }
  }

  pub fn try_pre_alloc(&self, query_sz: usize) -> NvGpuMemPoolReq {
    if self.back_alloc.get() {
      self._try_back_pre_alloc(query_sz)
    } else {
      self._try_front_pre_alloc(query_sz)
    }
  }

  pub fn _try_back_pre_alloc(&self, query_sz: usize) -> NvGpuMemPoolReq {
    let req_sz = ((query_sz + ALLOC_ALIGN - 1) / ALLOC_ALIGN) * ALLOC_ALIGN;
    assert!(query_sz <= req_sz);
    let offset = self.back_cursor.get();
    let next_offset = offset + req_sz;
    if next_offset + self.front_cursor.get() > self.front_sz {
      return NvGpuMemPoolReq::Oom;
    }
    NvGpuMemPoolReq::Back{offset, next_offset}
  }

  pub fn _try_front_pre_alloc(&self, query_sz: usize) -> NvGpuMemPoolReq {
    let req_sz = ((query_sz + ALLOC_ALIGN - 1) / ALLOC_ALIGN) * ALLOC_ALIGN;
    assert!(query_sz <= req_sz);
    let offset = self.front_cursor.get();
    let next_offset = offset + req_sz;
    if next_offset + self.back_cursor.get() > self.front_sz {
      return NvGpuMemPoolReq::Oom;
    }
    NvGpuMemPoolReq::Front{offset, next_offset}
  }

  pub fn alloc(&self, addr: PAddr, req: NvGpuMemPoolReq) -> Rc<NvGpuInnerCell> {
    match req {
      NvGpuMemPoolReq::Oom => panic!("bug"),
      NvGpuMemPoolReq::Front{offset, next_offset} => {
        let req_sz = next_offset - offset;
        self._front_alloc(addr, offset, req_sz, next_offset)
      }
      NvGpuMemPoolReq::Back{offset, next_offset} => {
        let req_sz = next_offset - offset;
        self._back_alloc(addr, offset, req_sz, next_offset)
      }
    }
  }

  pub fn _back_alloc(&self, addr: PAddr, backoffset: usize, req_sz: usize, next_backoffset: usize) -> Rc<NvGpuInnerCell> {
    let offset = self.front_sz - next_backoffset;
    /*let next_offset = self.front_sz - backoffset;*/
    let reg = Region{off: offset, sz: req_sz};
    let a = NvGpuAlloc{reg};
    self.alloc_map.borrow_mut().insert(addr, a);
    self.back_cursor.set(next_backoffset);
    let dptr = self.front_base + offset as u64;
    let cel = Rc::new(NvGpuInnerCell{
      addr,
      root: Cell::new(CellPtr::nil()),
      flag: Cell::new(0),
      tag:  Cell::new(0),
      dev:  self.dev,
      dptr: Cell::new(dptr),
      sz:   req_sz,
    });
    cel.set_back(true);
    println!("DEBUG: NvGpuMemPool::_back_alloc: addr={:?} dptr=0x{:016x} size={} off={}", addr, dptr, req_sz, offset);
    let mut size_index = self.size_index.borrow_mut();
    match size_index.get_mut(&req_sz) {
      None => {
        let mut addr_set = BTreeSet::new();
        addr_set.insert(addr);
        println!("DEBUG: NvGpuMemPool::_back_alloc:   sz set={:?}", &addr_set);
        size_index.insert(req_sz, addr_set);
      }
      Some(addr_set) => {
        addr_set.insert(addr);
        println!("DEBUG: NvGpuMemPool::_back_alloc:   sz set={:?}", &addr_set);
      }
    }
    assert!(self.alloc_index.borrow_mut().insert(Region{off: offset, sz: req_sz}, addr).is_none());
    assert!(self.cel_map.borrow_mut().insert(addr, cel.clone()).is_none());
    //dptr
    cel
  }

  pub fn _front_alloc(&self, addr: PAddr, offset: usize, req_sz: usize, next_offset: usize) -> Rc<NvGpuInnerCell> {
    let reg = Region{off: offset, sz: req_sz};
    let a = NvGpuAlloc{reg};
    self.alloc_map.borrow_mut().insert(addr, a);
    self.front_cursor.set(next_offset);
    let dptr = self.front_base + offset as u64;
    //let write = GpuSnapshot::fresh(self.dev());
    //let lastuse = GpuSnapshot::fresh(self.dev());
    let cel = Rc::new(NvGpuInnerCell{
      addr,
      root: Cell::new(CellPtr::nil()),
      flag: Cell::new(0),
      tag:  Cell::new(0),
      dev:  self.dev,
      dptr: Cell::new(dptr),
      sz:   req_sz,
      //write,
      //lastuse,
      // FIXME
    });
    cel.set_front(true);
    println!("DEBUG: NvGpuMemPool::_front_alloc: addr={:?} dptr=0x{:016x} size={} off={}", addr, dptr, req_sz, offset);
    let mut size_index = self.size_index.borrow_mut();
    match size_index.get_mut(&req_sz) {
      None => {
        let mut addr_set = BTreeSet::new();
        addr_set.insert(addr);
        println!("DEBUG: NvGpuMemPool::_front_alloc:   sz set={:?}", &addr_set);
        size_index.insert(req_sz, addr_set);
      }
      Some(addr_set) => {
        addr_set.insert(addr);
        println!("DEBUG: NvGpuMemPool::_front_alloc:   sz set={:?}", &addr_set);
      }
    }
    assert!(self.alloc_index.borrow_mut().insert(Region{off: offset, sz: req_sz}, addr).is_none());
    assert!(self.cel_map.borrow_mut().insert(addr, cel.clone()).is_none());
    //dptr
    cel
  }

  /*//pub fn find_front_lru_match(&self, query_sz: usize) -> Option<GpuInnerRef> {}
  //pub fn find_front_lru_match(&self, query_sz: usize) -> Option<CellPtr> {}
  pub fn find_front_lru_match(&self, query_sz: usize) -> Option<PAddr> {
    let mut mat = None;
    for (&r, desc) in self.alloc_map.borrow().iter() {
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
  }*/

  pub fn _try_front_relocate(&self, addr: PAddr, new_dptr: u64) -> Option<u64> {
    let old_reg = match self.alloc_map.borrow().get(&addr) {
      None => return None,
      Some(a) => a.reg
    };
    assert_eq!(self.alloc_index.borrow_mut().remove(&old_reg), Some(addr));
    let new_off = (new_dptr - self.front_base) as usize;
    let new_reg = Region{off: new_off, sz: old_reg.sz};
    assert_eq!(self.front_cursor.get(), new_reg.off);
    //assert!(self.alloc_index.borrow_mut().insert(new_reg, addr).is_none());
    if let Some(addr2) = self.alloc_index.borrow_mut().insert(new_reg, addr) {
      println!("DEBUG: NvGpuMemPool::try_front_relocate: addr={:?} new dptr=0x{:016x} old reg={:?} new reg={:?} addr2={:?}",
          addr, new_dptr, old_reg, new_reg, addr2);
      panic!("bug");
    }
    match self.cel_map.borrow().get(&addr) {
      None => panic!("bug"),
      Some(icel) => {
        icel.dptr.set(new_dptr);
      }
    }
    match self.alloc_map.borrow_mut().get_mut(&addr) {
      None => panic!("bug"),
      Some(a) => {
        a.reg.off = new_reg.off;
      }
    }
    self.front_cursor.set(new_reg.off + new_reg.sz);
    let old_dptr = self.front_base + old_reg.off as u64;
    Some(old_dptr)
  }

  pub fn front_relocate(&self, src: PAddr, dst_dptr: u64, stream: &CudartStream) {
    /*let (src_dptr, src_sz) = match self.lookup_reg(src) {
      Some(NvGpuInnerReg::VMem{dptr, size}) => (dptr, size),
      _ => panic!("bug")
    };*/
    let (src_dptr, src_sz) = match self.cel_map.borrow().get(&src) {
      None => panic!("bug"),
      Some(icel) => {
        (icel.dptr.get(), icel.sz)
      }
    };
    assert_eq!(self._try_front_relocate(src, dst_dptr), Some(src_dptr));
    if dst_dptr == src_dptr {
    } else if (dst_dptr + src_sz as u64 <= src_dptr) ||
              (src_dptr + src_sz as u64 <= dst_dptr)
    {
      stream.sync().unwrap();
      cuda_memcpy_async(dst_dptr, src_dptr, src_sz, stream).unwrap();
      stream.sync().unwrap();
    } else {
      // FIXME FIXME: overlapping copy.
      unimplemented!();
    }
  }

  pub fn try_free(&self, addr: PAddr) -> Option<Rc<NvGpuInnerCell>> {
    let mut front_prefix = self.front_offset();
    let old_alloc = match self.alloc_map.borrow_mut().remove(&addr) {
      None => return None,
      Some(a) => a
    };
    let old_reg = old_alloc.reg;
    if front_prefix <= old_reg.off {
    } else if old_reg.off < front_prefix && front_prefix < old_reg.off + old_reg.sz {
      panic!("bug");
    } else if old_reg.off + old_reg.sz == front_prefix {
      // FIXME FIXME
      front_prefix = old_reg.off;
      self.front_cursor.set(front_prefix);
      let mut free_index = self.free_index.borrow_mut();
      let pivot = Region{off: front_prefix, sz: 0};
      match free_index.range( .. pivot).rev().next() {
        None => {}
        Some(&l_reg) => {
          if l_reg.off + l_reg.sz > front_prefix {
            panic!("bug");
          } else if l_reg.off + l_reg.sz == front_prefix {
            front_prefix = l_reg.off;
            self.front_cursor.set(front_prefix);
            free_index.remove(&l_reg);
          }
        }
      }
    } else if old_reg.off + old_reg.sz < front_prefix {
      let mut free_index = self.free_index.borrow_mut();
      let mut merge_reg = old_reg;
      let pivot = Region{off: old_reg.off + old_reg.sz, sz: 0};
      match free_index.range( .. pivot).rev().next() {
        None => {}
        Some(&l_reg) => {
          if l_reg.off + l_reg.sz > old_reg.off {
            panic!("bug");
          } else if l_reg.off + l_reg.sz == old_reg.off {
            free_index.remove(&l_reg);
            merge_reg = l_reg.merge(merge_reg);
          }
        }
      }
      let pivot = Region{off: old_reg.off, sz: 0};
      match free_index.range(pivot .. ).next() {
        None => {}
        Some(&r_reg) => {
          if merge_reg.off + merge_reg.sz > r_reg.off {
            panic!("bug");
          } else if merge_reg.off + merge_reg.sz == r_reg.off {
            free_index.remove(&r_reg);
            merge_reg = merge_reg.merge(r_reg);
          }
        }
      }
      assert!(merge_reg.off + merge_reg.sz < front_prefix);
      free_index.insert(merge_reg);
    } else {
      unreachable!();
    }
    match self.size_index.borrow_mut().get_mut(&old_reg.sz) {
      None => panic!("bug"),
      Some(addr_set) => {
        addr_set.remove(&addr);
      }
    }
    assert_eq!(self.alloc_index.borrow_mut().remove(&old_reg), Some(addr));
    let icel = self.cel_map.borrow_mut().remove(&addr).unwrap();
    let old_dptr = self.front_base + old_reg.off as u64;
    assert_eq!(old_dptr, icel.dptr.get());
    Some(icel)
  }

  /*pub fn back_alloc(&self, sz: usize) -> u64 {
  //pub fn try_back_alloc(&self, sz: usize) -> (Option<GpuInnerRef>, Option<Weak<GpuInnerCell>>) {}
    // FIXME: currently, we only use this to alloc int32's for futhark error vars.
    assert_eq!(sz, 4);
    let prev_curs = self.back_cursor.get();
    let next_curs = prev_curs - 4;
    let dptr = self.extra_base + next_curs as u64;
    self.back_cursor.set(next_curs);
    println!("DEBUG: GpuMemPool::back_alloc: dptr=0x{:016x} sz={}", dptr, sz);
    dptr
  }*/

  /*pub fn back_free_all(&self) {
    // FIXME FIXME
    //unimplemented!();
  }*/
}

pub extern "C" fn tl_pctx_gpu_alloc_hook(dptr: *mut u64, sz: usize, raw_tag: *const c_char) -> i32 {
  assert!(!dptr.is_null());
  TL_PCTX.with(|pctx| {
    /*println!("DEBUG: tl_pctx_gpu_alloc_hook: sz={} raw tag=0x{:016x}",
        sz,
        raw_tag as usize,
    );*/
    let gpu = pctx.nvgpu.as_ref().unwrap();
    let (req, tag) = if raw_tag.is_null() {
      println!("DEBUG: tl_pctx_gpu_alloc_hook: sz={} ctag=(null)",
          sz,
      );
      let req = gpu.mem_pool.try_pre_alloc(sz);
      (req, None)
    } else {
      unsafe {
        let ctag = CStr::from_ptr(raw_tag);
        println!("DEBUG: tl_pctx_gpu_alloc_hook: sz={} ctag=\"{}\"",
            sz,
            sane_ascii(ctag.to_bytes()),
        );
        //let tag = ctag.to_bytes();
        let tag = TagUnifier::parse_tag(ctag.to_bytes()).unwrap();
        let mut unify = &mut *pctx.tagunify.borrow_mut();
        unify.find(tag);
        let req = gpu.mem_pool.try_pre_alloc_with_tag(sz, tag, unify);
        (req, Some(tag))
      }
    };
    match req {
      NvGpuMemPoolReq::Oom => {
        // FIXME FIXME: oom.
        unimplemented!();
      }
      _ => {
        let p = pctx.ctr.fresh_addr();
        let cel = gpu.mem_pool.alloc(p, req);
        if gpu.mem_pool.alloc_pin.get() {
          InnerCell::set_pin(&*cel, true);
        }
        InnerCell::set_tag(&*cel, tag);
        unsafe {
          write(dptr, cel.dptr.get());
        }
        println!("DEBUG: tl_pctx_gpu_alloc_hook:   addr={:?} dptr=0x{:016x} size={} tag={:?}",
            p, cel.dptr.get(), req.size(), tag,
        );
      }
    }
    0
  })
}

pub extern "C" fn tl_pctx_gpu_free_hook(dptr: u64) -> i32 {
  TL_PCTX.try_with(|pctx| {
    println!("DEBUG: tl_pctx_gpu_free_hook: dptr=0x{:016x}",
        dptr,
    );
    let gpu = pctx.nvgpu.as_ref().unwrap();
    let p = match gpu.mem_pool.lookup_dptr(dptr) {
      Some((_, Some(p))) => p,
      _ => panic!("bug"),
    };
    gpu.mem_pool.free_list.borrow_mut().push(p);
    match gpu.mem_pool.cel_map.borrow().get(&p) {
      None => panic!("bug"),
      Some(cel) => {
        println!("DEBUG: tl_pctx_gpu_free_hook:   addr={:?} dptr=0x{:016x} size={} tag={:?}",
            p, cel.dptr.get(), InnerCell::size(&**cel), InnerCell::tag(&**cel),
        );
      }
    }
    0
  }).unwrap_or_else(|_| {
    println!("DEBUG: tl_pctx_gpu_free_hook: dptr=0x{:016x} (pctx deinit)",
        dptr,
    );
    0
  })
}

pub extern "C" fn tl_pctx_gpu_unify_hook(lhs_raw_tag: *const c_char, rhs_raw_tag: *const c_char) {
  TL_PCTX.try_with(|pctx| {
    /*println!("DEBUG: tl_pctx_gpu_unify_hook: raw ltag=0x{:016x} raw rtag=0x{:016x}",
        lhs_raw_tag as usize,
        rhs_raw_tag as usize,
    );*/
    unsafe {
      let lhs_ctag = if lhs_raw_tag.is_null() {
        None
      } else {
        Some(CStr::from_ptr(lhs_raw_tag))
      };
      let rhs_ctag = if rhs_raw_tag.is_null() {
        None
      } else {
        Some(CStr::from_ptr(rhs_raw_tag))
      };
      println!("DEBUG: tl_pctx_gpu_unify_hook: ltag={:?} rtag={:?}",
          lhs_ctag.map(|c| sane_ascii(c.to_bytes())),
          rhs_ctag.map(|c| sane_ascii(c.to_bytes())),
      );
      match (lhs_ctag.map(|c| c.to_bytes()), rhs_ctag.map(|c| c.to_bytes())) {
        (None, None) => {}
        (None, Some(tag)) |
        (Some(tag), None) => {
          let tag = TagUnifier::parse_tag(tag).unwrap();
          let mut unify = pctx.tagunify.borrow_mut();
          unify.find(tag);
        }
        (Some(ltag), Some(rtag)) => {
          let ltag = TagUnifier::parse_tag(ltag).unwrap();
          let rtag = TagUnifier::parse_tag(rtag).unwrap();
          let mut unify = pctx.tagunify.borrow_mut();
          unify.unify(ltag, rtag);
        }
      }
    }
  }).unwrap_or_else(|_| ())
}

pub extern "C" fn tl_pctx_gpu_failarg_alloc_hook(dptr: *mut u64, sz: usize) -> i32 {
  assert!(!dptr.is_null());
  TL_PCTX.with(|pctx| {
    if sz > ALLOC_ALIGN {
      println!("ERROR: tl_pctx_gpu_failarg_alloc_hook: requested size={} greater than max size={}", sz, ALLOC_ALIGN);
      panic!();
    }
    let mem_pool = &pctx.nvgpu.as_ref().unwrap().mem_pool;
    let x = mem_pool.extra_base + mem_pool.extra_sz as u64 - ALLOC_ALIGN as u64;
    unsafe {
      write(dptr, x);
    }
    0
  })
}

pub extern "C" fn tl_pctx_gpu_failarg_free_hook(_dptr: u64) -> i32 {
  TL_PCTX.try_with(|_pctx| {
    // FIXME FIXME
    0
  }).unwrap_or_else(|_| 0)
}

pub struct GpuDryCtx {
  // FIXME
}
