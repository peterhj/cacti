use super::*;
use crate::algo::{BTreeMap, BTreeSet, HashMap, Region, RevSortMap8, StdCellExt};
use crate::algo::str::*;
use crate::algo::sync::{SpinWait};
use crate::cell::*;
use crate::clock::*;
use crate::ctx::*;
use cacti_cfg_env::*;
use cacti_gpu_cu_ffi::*;
use cacti_gpu_cu_ffi::types::*;

use libc::{c_char, c_int, c_void};

use std::cell::{Cell, RefCell, UnsafeCell};
use std::convert::{TryInto};
use std::ffi::{CStr};
use std::fs::{OpenOptions};
use std::io::{Write};
use std::rc::{Rc, Weak};
use std::path::{PathBuf};
use std::ptr::{write};
use std::str::{from_utf8};

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

/*//#[derive(Clone)]
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
}*/

#[derive(Clone, Copy, Debug)]
pub struct NvGpuDeviceInfo {
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

impl NvGpuDeviceInfo {
  pub fn new(dev: i32) -> NvGpuDeviceInfo {
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
    NvGpuDeviceInfo{
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
  pub pctx:         CudaPrimaryCtx,
  pub dev_info:     NvGpuDeviceInfo,
  pub blas_ctx:     CublasContext,
  pub compute:      CudartStream,
  //pub copy_to:      CudartStream,
  //pub copy_from:    CudartStream,
  pub page_map:     NvGpuPageMap,
  pub mem_pool:     NvGpuMemPool,
  pub kernels:      NvGpuCopyKernels,
}

impl Drop for NvGpuPCtx {
  fn drop(&mut self) {
    self._dump_info();
  }
}

impl PCtxImpl for NvGpuPCtx {
  fn pmach(&self) -> PMach {
    PMach::NvGpu
  }

  fn fastest_locus(&self) -> Locus {
    self.device_locus()
  }

  fn append_matrix(&self, lp: &mut RevSortMap8<(Locus, PMach), ()>, pl: &mut RevSortMap8<(PMach, Locus), ()>) {
    if self.dev_info.integrated {
      if cfg_debug() {
      println!("DEBUG: NvGpuPCtx::append_matrix: integrated: capability={}.{}",
          self.dev_info.capability_major, self.dev_info.capability_minor);
      }
      if !self.dev_info.unified_address {
        println!("WARNING: NvGpuPCtx::append_matrix: integrated but not unified address space");
      }
      if !self.dev_info.managed_memory {
        println!("WARNING: NvGpuPCtx::append_matrix: integrated but not managed memory");
      }
    } else {
      lp.insert((Locus::VMem, PMach::NvGpu), ());
      pl.insert((PMach::NvGpu, Locus::VMem), ());
    }
    lp.insert((Locus::Mem, PMach::NvGpu), ());
    pl.insert((PMach::NvGpu, Locus::Mem), ());
  }

  //fn try_alloc(&self, pctr: &PCtxCtr, /*pmset: PMachSet,*/ locus: Locus, ty: &CellType) -> Result<PAddr, PMemErr> {}
  fn try_alloc(&self, pctr: &PCtxCtr, ty: &CellType, locus: Locus) -> Result<PAddr, PMemErr> {
    let sz = ty.packed_span_bytes() as usize;
    match locus {
      Locus::Mem => {
        let addr = pctr.fresh_addr();
        //let mem = NvGpuMemCell::try_alloc(sz)?;
        //assert!(self.page_map.page_tab.borrow_mut().insert(addr, Rc::new(mem)).is_none());
        match self.page_map.try_alloc(addr, sz) {
          Err(_) => {
            // FIXME FIXME: oom.
            unimplemented!();
          }
          Ok(_) => {}
        }
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
    if cfg_info() { println!("INFO:   NvGpuPCtx::new: dev={}", dev); }
    if LIBCUDA._inner.is_none() {
      return None;
    }
    cudart_set_cur_dev(dev).unwrap();
    let pctx = CudaPrimaryCtx::retain(dev).unwrap();
    // FIXME: confirm that SCHED_YIELD is what we really want.
    pctx.set_flags(CU_CTX_SCHED_YIELD).unwrap();
    let dev_info = NvGpuDeviceInfo::new(dev);
    if cfg_info() { println!("INFO:   NvGpuPCtx::new: dev info={:?}", &dev_info); }
    let blas_ctx = CublasContext::create().unwrap();
    let compute = CudartStream::null();
    //let copy_to = CudartStream::create_nonblocking().unwrap();
    //let copy_from = CudartStream::create_nonblocking().unwrap();
    /*let compute = CudartStream::create().unwrap();
    let copy_to = CudartStream::create().unwrap();
    let copy_from = CudartStream::create().unwrap();*/
    let page_map = NvGpuPageMap::new();
    let mem_pool = NvGpuMemPool::new(dev);
    let kernels = NvGpuCopyKernels::new();
    Some(NvGpuPCtx{
      pctx,
      dev_info,
      blas_ctx,
      compute,
      //copy_to,
      //copy_from,
      page_map,
      mem_pool,
      kernels,
    })
  }

  pub fn dev(&self) -> i32 {
    self.pctx.device()
  }

  pub fn device_locus(&self) -> Locus {
    if self.dev_info.integrated {
      if cfg_debug() {
      println!("DEBUG: NvGpuPCtx::device_locus: integrated: capability={}.{}",
          self.dev_info.capability_major, self.dev_info.capability_minor);
      }
      if !self.dev_info.unified_address {
        println!("WARNING: NvGpuPCtx::device_locus: integrated but not unified address space");
      }
      if !self.dev_info.managed_memory {
        println!("WARNING: NvGpuPCtx::device_locus: integrated but not managed memory");
      }
      Locus::Mem
    } else {
      Locus::VMem
    }
  }

  /*pub fn fresh_outer(&self) -> Rc<GpuOuterCell> {
    unimplemented!();
  }*/

  pub fn hard_copy(&self, dst_loc: Locus, dst: PAddr, src_loc: Locus, src: PAddr, sz: usize) {
    match (dst_loc, src_loc) {
      (Locus::VMem, Locus::VMem) => {
        if dst == src {
          let (dst_dptr, dst_sz) = match self.lookup_reg(dst) {
            Some(NvGpuInnerReg::VMem{dptr, size}) => (dptr, size),
            _ => panic!("bug")
          };
          let (src_dptr, src_sz) = match self.lookup_reg(src) {
            Some(NvGpuInnerReg::VMem{dptr, size}) => (dptr, size),
            _ => panic!("bug")
          };
          if cfg_debug() {
          println!("WARNING:NvGpuPCtx::hard_copy: VMem <- VMem: dst addr={:?} dptr=0x{:016x} sz={} src addr={:?} ptr=0x{:016x} sz={} copy sz={}",
              dst, dst_dptr, dst_sz, src, src_dptr, src_sz, sz);
          }
        } else {
          unimplemented!();
        }
      }
      (Locus::Mem, Locus::Mem) => {
        if dst == src {
          let (dst_ptr, dst_sz) = match self.lookup_reg(dst) {
            Some(NvGpuInnerReg::Mem{ptr, size}) => (ptr, size),
            _ => panic!("bug")
          };
          let (src_ptr, src_sz) = match self.lookup_reg(src) {
            Some(NvGpuInnerReg::Mem{ptr, size}) => (ptr, size),
            _ => panic!("bug")
          };
          if cfg_debug() {
          println!("WARNING:NvGpuPCtx::hard_copy: Mem <- Mem: dst addr={:?} ptr=0x{:016x} sz={} src addr={:?} ptr=0x{:016x} sz={} copy sz={}",
              dst, dst_ptr as usize, dst_sz, src, src_ptr as usize, src_sz, sz);
          }
        } else {
          unimplemented!();
        }
      }
      (Locus::VMem, Locus::Mem) => {
        let (dst_dptr, dst_sz) = match self.lookup_reg(dst) {
          Some(NvGpuInnerReg::VMem{dptr, size}) => (dptr, size),
          _ => panic!("bug")
        };
        let (src_ptr, src_sz) = match self.lookup_reg(src) {
          Some(NvGpuInnerReg::Mem{ptr, size}) => (ptr, size),
          _ => panic!("bug")
        };
        assert!(dst_sz >= sz);
        assert!(src_sz >= sz);
        if cfg_debug() {
        println!("DEBUG: NvGpuPCtx::hard_copy: VMem <- Mem: dst addr={:?} dptr=0x{:016x} sz={} src addr={:?} ptr=0x{:016x} sz={} copy sz={}",
            dst, dst_dptr, dst_sz, src, src_ptr as usize, src_sz, sz);
        }
        self.compute.sync().unwrap();
        cuda_memcpy_h2d_async(dst_dptr, src_ptr, sz, &self.compute).unwrap();
        self.compute.sync().unwrap();
      }
      (Locus::Mem, Locus::VMem) => {
        let (dst_ptr, dst_sz) = match self.lookup_reg(dst) {
          Some(NvGpuInnerReg::Mem{ptr, size}) => (ptr, size),
          _ => panic!("bug")
        };
        let (src_dptr, src_sz) = match self.lookup_reg(src) {
          Some(NvGpuInnerReg::VMem{dptr, size}) => (dptr, size),
          _ => panic!("bug")
        };
        assert!(dst_sz >= sz);
        assert!(src_sz >= sz);
        if cfg_debug() {
        println!("DEBUG: NvGpuPCtx::hard_copy: Mem <- VMem: dst addr={:?} ptr=0x{:016x} sz={} src addr={:?} dptr=0x{:016x} sz={} copy sz={}",
            dst, dst_ptr as usize, dst_sz, src, src_dptr, src_sz, sz);
        }
        self.compute.sync().unwrap();
        cuda_memcpy_d2h_async(dst_ptr, src_dptr, sz, &self.compute).unwrap();
        self.compute.sync().unwrap();
      }
      _ => unimplemented!()
    }
  }

  pub fn hard_copy_raw_mem_to_vmem(&self, dst_dptr: u64, src_ptr: *const c_void, sz: usize) {
    if cfg_debug() {
    println!("DEBUG: NvGpuPCtx::hard_copy_raw_mem_to_vmem: dst dptr=0x{:016x} src ptr=0x{:016x} sz={}",
        dst_dptr, src_ptr as usize, sz);
    }
    self.compute.sync().unwrap();
    cuda_memcpy_h2d_async(dst_dptr, src_ptr, sz, &self.compute).unwrap();
    self.compute.sync().unwrap();
  }

  pub fn hard_copy_nb_raw_mem_to_vmem(&self, dst_dptr: u64, src_ptr: *const c_void, sz: usize) {
    if cfg_debug() {
    println!("DEBUG: NvGpuPCtx::hard_copy_nb_raw_mem_to_vmem: dst dptr=0x{:016x} src ptr=0x{:016x} sz={}",
        dst_dptr, src_ptr as usize, sz);
    }
    cuda_memcpy_h2d_async(dst_dptr, src_ptr, sz, &self.compute).unwrap();
  }

  pub fn release(&self, addr: PAddr) -> Option<(Locus, Rc<dyn InnerCell_>)> {
    //let page_tab = self.page_map.page_tab.borrow();
    //let cel_map = self.mem_pool.cel_map.borrow();
    if self.page_map.page_tab.borrow().len() <= self.mem_pool.cel_map.borrow().len() {
      match self.page_map.release(addr) {
        None => {}
        Some(icel) => {
          return Some((Locus::Mem, icel));
        }
      }
      match self.mem_pool.try_free(addr) {
        None => {}
        Some(icel) => {
          return Some((Locus::VMem, icel));
        }
      }
    } else {
      match self.mem_pool.try_free(addr) {
        None => {}
        Some(icel) => {
          return Some((Locus::VMem, icel));
        }
      }
      match self.page_map.release(addr) {
        None => {}
        Some(icel) => {
          return Some((Locus::Mem, icel));
        }
      }
    }
    None
  }

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
          assert!(self.dev_info.unified_address);
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
          assert!(self.dev_info.unified_address);
          #[cfg(not(target_pointer_width = "64"))]
          unimplemented!();
          #[cfg(target_pointer_width = "64")]
          return Some((icel.ptr as usize as u64, icel.sz));
        }
      }
    }
    None
  }

  pub fn _dump_info(&self) {
    if cfg_info() {
    println!("INFO:   NvGpuPCtx::_dump_info: mem pool usage: front={} free={} back={} out of total reserve={}",
        self.mem_pool.front_cursor.get(),
        self.mem_pool.free_size.get(),
        self.mem_pool.back_cursor.get(),
        self.mem_pool.reserve_sz,
    );
    }
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

#[repr(C)]
pub struct NvGpuMemCell {
  //pub addr: PAddr,
  pub root: Cell<CellPtr>,
  // FIXME
  //pub refc: Cell<u32>,
  pub flag: Cell<u8>,
  pub ptr:  *mut c_void,
  pub sz:   usize,
}

impl Drop for NvGpuMemCell {
  fn drop(&mut self) {
    if self.ptr.is_null() {
      return;
    }
    match cuda_mem_free_host(self.ptr) {
      Ok(_) => {}
      Err(CUDA_ERROR_DEINITIALIZED) => {}
      Err(_) => panic!("bug"),
    }
  }
}

impl NvGpuMemCell {
  pub fn _try_alloc(sz: usize) -> Result<NvGpuMemCell, PMemErr> {
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
    if cfg_debug() { println!("DEBUG: NvGpuMemCell::_try_alloc: ptr=0x{:016x} sz={}", ptr as usize, sz); }
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

  fn cow(&self) -> bool {
    (self.flag.get() & 0x10) != 0
  }

  fn set_cow(&self, flag: bool) {
    if flag {
      self.flag.set(self.flag.get() | 0x10);
    } else {
      self.flag.set(self.flag.get() & !0x10);
    }
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
  pub page_idx: RefCell<HashMap<*mut c_void, PAddr>>,
  pub extrabuf: Rc<NvGpuMemCell>,
}

impl NvGpuPageMap {
  pub fn new() -> NvGpuPageMap {
    let mem = match NvGpuMemCell::_try_alloc(1 << 16) {
      Err(_) => {
        println!("ERROR: NvGpuPageMap: failed to allocate shadow back buffer");
        panic!();
      }
      Ok(mem) => mem
    };
    NvGpuPageMap{
      page_idx: RefCell::new(HashMap::new()),
      page_tab: RefCell::new(HashMap::new()),
      extrabuf: Rc::new(mem),
    }
  }

  pub fn lookup(&self, addr: PAddr) -> Option<Rc<NvGpuMemCell>> {
    self.page_tab.borrow().get(&addr).map(|cel| cel.clone())
  }

  pub fn rev_lookup(&self, ptr: *mut c_void) -> Option<PAddr> {
    self.page_idx.borrow().get(&ptr).map(|&addr| addr)
  }

  pub fn try_alloc(&self, addr: PAddr, sz: usize) -> Result<Rc<NvGpuMemCell>, PMemErr> {
    if cfg_debug() {
      println!("DEBUG: NvGpuPageMap::try_alloc: addr={:?} sz={:?}", addr, sz);
    }
    let cel = Rc::new(NvGpuMemCell::_try_alloc(sz)?);
    assert!(self.page_tab.borrow_mut().insert(addr, cel.clone()).is_none());
    assert!(self.page_idx.borrow_mut().insert(cel.ptr, addr).is_none());
    Ok(cel)
  }

  pub fn release(&self, addr: PAddr) -> Option<Rc<NvGpuMemCell>> {
    let cel = self.page_tab.borrow_mut().remove(&addr);
    match cel.as_ref() {
      None => {}
      Some(cel) => {
        let oaddr = self.page_idx.borrow_mut().remove(&cel.ptr);
        if !(oaddr == Some(addr)) {
          println!("WARNING: NvGpuPageMap::release: addr={:?} cel.ptr={:?} oaddr={:?}", addr, cel.ptr, oaddr);
        }
        assert_eq!(oaddr, Some(addr));
      }
    }
    cel
  }
}

/*#[derive(Clone)]
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
}*/

pub type GpuInnerCell = NvGpuInnerCell;

pub struct NvGpuInnerCell {
  //pub addr: PAddr,
  pub root: Cell<CellPtr>,
  // FIXME
  //pub refc: Cell<u32>,
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

  fn cow(&self) -> bool {
    (self.flag.get() & 0x10) != 0
  }

  fn set_cow(&self, flag: bool) {
    if flag {
      self.flag.set(self.flag.get() | 0x10);
    } else {
      self.flag.set(self.flag.get() & !0x10);
    }
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
  pub free_size:    Cell<usize>,
  pub back_cursor:  Cell<usize>,
  pub back_alloc:   Cell<bool>,
  pub alloc_pin:    Cell<bool>,
  pub front_tag:    RefCell<Option<u32>>,
  pub tmp_freelist: RefCell<Vec<PAddr>>,
  pub free_index:   RefCell<BTreeSet<Region>>,
  pub size_index:   RefCell<HashMap<usize, BTreeSet<PAddr>>>,
  pub alloc_index:  RefCell<BTreeMap<Region, PAddr>>,
  pub alloc_map:    RefCell<HashMap<PAddr, NvGpuAlloc>>,
  pub cel_map:      RefCell<HashMap<PAddr, Rc<NvGpuInnerCell>>>,
  // TODO
}

impl Drop for NvGpuMemPool {
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

impl NvGpuMemPool {
  #[allow(unused_parens)]
  pub fn new(dev: i32) -> NvGpuMemPool {
    if cfg_debug() { println!("DEBUG: NvGpuMemPool::new"); }
    cudart_set_cur_dev(dev).unwrap();
    let (free_sz, total_sz) = cudart_get_mem_info().unwrap();
    assert!(free_sz <= total_sz);
    let reserve_bp = ctx_cfg_get_gpu_reserve_mem_per_10k();
    let unrounded_reserve_sz = (total_sz * reserve_bp as usize + 10000 - 1) / 10000;
    // NB: assuming page size is 64 KiB.
    let reserve_sz = ((unrounded_reserve_sz + (1 << 16) - 1) >> 16) << 16;
    assert!(reserve_sz >= unrounded_reserve_sz);
    if reserve_sz > free_sz {
      panic!("ERROR: NvGpuMemPool::new: gpu oom: tried to reserve {} bytes on gpu {}, but only found {} bytes free (out of {} bytes total)",
          reserve_sz, dev, free_sz, total_sz);
    }
    if cfg_info() { println!("INFO:   NvGpuMemPool::new: reserve sz={}", reserve_sz); }
    let reserve_base = cuda_mem_alloc(reserve_sz).unwrap();
    CudartStream::null().sync().unwrap();
    if cfg_debug() { println!("DEBUG: NvGpuMemPool::new: reserve base=0x{:016x}", reserve_base); }
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
    if cfg_debug() { println!("DEBUG: NvGpuMemPool::new: front sz={}", front_sz); }
    if cfg_debug() { println!("DEBUG: NvGpuMemPool::new: back sz={}", extra_sz); }
    NvGpuMemPool{
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
      free_size:    Cell::new(0),
      back_cursor:  Cell::new(0),
      back_alloc:   Cell::new(false),
      alloc_pin:    Cell::new(false),
      front_tag:    RefCell::new(None),
      tmp_freelist: RefCell::new(Vec::new()),
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

  pub fn rev_lookup(&self, query_dptr: u64) -> Option<(Region, Option<PAddr>)> {
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
      Some((&k, &v)) => if k.off <= query_off && query_off < k.off + k.sz {
        assert!(k.sz > 0);
        return Some((k, Some(v)));
      }
    }
    match self.alloc_index.borrow().range( .. q).rev().next() {
      None => {}
      Some((&k, &v)) => if k.off <= query_off && query_off < k.off + k.sz {
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
    if req_sz <= self.free_size.get() {
      let free_index = self.free_index.borrow();
      for old_reg in free_index.iter() {
        if req_sz <= old_reg.sz {
          let offset = old_reg.off;
          let next_offset = offset + req_sz;
          assert!(next_offset <= self.front_cursor.get());
          return NvGpuMemPoolReq::Front{offset, next_offset};
        }
      }
    }
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
      //addr,
      root: Cell::new(CellPtr::nil()),
      flag: Cell::new(0),
      tag:  Cell::new(0),
      dev:  self.dev,
      dptr: Cell::new(dptr),
      sz:   req_sz,
    });
    cel.set_back(true);
    if cfg_debug() { println!("DEBUG: NvGpuMemPool::_back_alloc: addr={:?} dptr=0x{:016x} size={} off={}", addr, dptr, req_sz, offset); }
    let mut size_index = self.size_index.borrow_mut();
    match size_index.get_mut(&req_sz) {
      None => {
        let mut addr_set = BTreeSet::new();
        addr_set.insert(addr);
        //println!("DEBUG: NvGpuMemPool::_back_alloc:   sz set={:?}", &addr_set);
        size_index.insert(req_sz, addr_set);
      }
      Some(addr_set) => {
        addr_set.insert(addr);
        //println!("DEBUG: NvGpuMemPool::_back_alloc:   sz set={:?}", &addr_set);
      }
    }
    assert!(self.alloc_index.borrow_mut().insert(Region{off: offset, sz: req_sz}, addr).is_none());
    assert!(self.cel_map.borrow_mut().insert(addr, cel.clone()).is_none());
    //dptr
    cel
  }

  pub fn _front_alloc(&self, addr: PAddr, offset: usize, req_sz: usize, next_offset: usize) -> Rc<NvGpuInnerCell> {
    if offset < self.front_cursor.get() {
      assert!(offset + req_sz <= self.front_cursor.get());
      let mut free_index = self.free_index.borrow_mut();
      let mut free_idx_iter = free_index.iter();
      for &old_reg in &mut free_idx_iter {
        if old_reg.off == offset {
          drop(free_idx_iter);
          if req_sz == old_reg.sz {
            free_index.remove(&old_reg);
            self.free_size.fetch_sub(old_reg.sz);
          } else if req_sz < old_reg.sz {
            free_index.remove(&old_reg);
            self.free_size.fetch_sub(old_reg.sz);
            let mut split_reg = old_reg;
            split_reg.off += req_sz;
            split_reg.sz -= req_sz;
            assert_eq!(split_reg.off, next_offset);
            assert_eq!(split_reg.off + split_reg.sz, old_reg.off + old_reg.sz);
            self.free_size.fetch_add(split_reg.sz);
            free_index.insert(split_reg);
          } else {
            unreachable!();
          }
          break;
        }
      }
    } else {
      self.front_cursor.set(next_offset);
    }
    let dptr = self.front_base + offset as u64;
    //let write = GpuSnapshot::fresh(self.dev());
    //let lastuse = GpuSnapshot::fresh(self.dev());
    let cel = Rc::new(NvGpuInnerCell{
      //addr,
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
    if cfg_debug() { println!("DEBUG: NvGpuMemPool::_front_alloc: addr={:?} dptr=0x{:016x} size={} off={}", addr, dptr, req_sz, offset); }
    let mut size_index = self.size_index.borrow_mut();
    match size_index.get_mut(&req_sz) {
      None => {
        let mut addr_set = BTreeSet::new();
        addr_set.insert(addr);
        //println!("DEBUG: NvGpuMemPool::_front_alloc:   sz set={:?}", &addr_set);
        size_index.insert(req_sz, addr_set);
      }
      Some(addr_set) => {
        addr_set.insert(addr);
        //println!("DEBUG: NvGpuMemPool::_front_alloc:   sz set={:?}", &addr_set);
      }
    }
    let reg = Region{off: offset, sz: req_sz};
    let a = NvGpuAlloc{reg};
    assert!(self.alloc_map.borrow_mut().insert(addr, a).is_none());
    assert!(self.alloc_index.borrow_mut().insert(reg, addr).is_none());
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
    if !(self.front_cursor.get() == new_reg.off) {
      println!("DEBUG: NvGpuMemPool::_try_front_relocate: front cursor=0x{:016x}", self.front_cursor.get());
      println!("DEBUG: NvGpuMemPool::_try_front_relocate: old reg off =0x{:016x}", old_reg.off);
      println!("DEBUG: NvGpuMemPool::_try_front_relocate: old reg sz  =0x{:016x}", old_reg.sz);
      println!("DEBUG: NvGpuMemPool::_try_front_relocate: new reg off =0x{:016x}", new_reg.off);
      println!("DEBUG: NvGpuMemPool::_try_front_relocate: new reg sz  =0x{:016x}", new_reg.sz);
    }
    assert_eq!(self.front_cursor.get(), new_reg.off);
    //assert!(self.alloc_index.borrow_mut().insert(new_reg, addr).is_none());
    if let Some(addr2) = self.alloc_index.borrow_mut().insert(new_reg, addr) {
      println!("DEBUG: NvGpuMemPool::_try_front_relocate: addr={:?} new dptr=0x{:016x} old reg={:?} new reg={:?} addr2={:?}",
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
    if cfg_debug() {
    println!("DEBUG: NvGpuMemPool::try_free: addr={:?}", addr);
    println!("DEBUG: NvGpuMemPool::try_free: old front cursor=0x{:016x}", self.front_cursor.get());
    }
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
            self.free_size.fetch_sub(l_reg.sz);
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
            self.free_size.fetch_sub(l_reg.sz);
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
            self.free_size.fetch_sub(r_reg.sz);
            merge_reg = merge_reg.merge(r_reg);
          }
        }
      }
      assert!(merge_reg.off + merge_reg.sz < front_prefix);
      self.free_size.fetch_add(merge_reg.sz);
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
    if cfg_debug() {
    println!("DEBUG: NvGpuMemPool::try_free: OK");
    println!("DEBUG: NvGpuMemPool::try_free: new front cursor=0x{:016x}", self.front_cursor.get());
    }
    Some(icel)
  }
}

pub extern "C" fn tl_pctx_nvgpu_mem_alloc_hook(ptr: *mut *mut c_void, sz: usize, raw_tag: *const c_char) -> c_int {
  assert!(!ptr.is_null());
  let ctag = unsafe {
    if raw_tag.is_null() {
      None
    } else {
      Some(CStr::from_ptr(raw_tag))
    }
  };
  if cfg_debug() { println!("DEBUG: tl_pctx_nvgpu_mem_alloc_hook: sz={} ctag={:?}",
      sz,
      ctag.map(|ctag| safe_ascii(ctag.to_bytes())),
  ); }
  TL_PCTX.with(|pctx| {
    let gpu = pctx.nvgpu.as_ref().unwrap();
    let addr = pctx.ctr.fresh_addr();
    match gpu.page_map.try_alloc(addr, sz) {
      Err(PMemErr::Oom) => {
        // FIXME FIXME: oom.
        unimplemented!();
      }
      Err(_) => panic!("bug"),
      Ok(cel) => {
        unsafe {
          write(ptr, cel.ptr);
        }
      }
    }
    0
  })
}

pub extern "C" fn tl_pctx_nvgpu_mem_free_hook(ptr: *mut c_void) -> c_int {
  TL_PCTX.try_with(|pctx| {
    if cfg_debug() { println!("DEBUG: tl_pctx_nvgpu_mem_free_hook: ptr=0x{:016x}",
        ptr as usize,
    ); }
    let gpu = pctx.nvgpu.as_ref().unwrap();
    match gpu.page_map.rev_lookup(ptr) {
      None => panic!("bug"),
      Some(addr) => {
        let cel = gpu.page_map.release(addr).unwrap();
        if cfg_debug() { println!("DEBUG: tl_pctx_nvgpu_mem_free_hook:   addr={:?} ptr=0x{:016x} size={}",
            addr, cel.ptr as usize, InnerCell::size(&*cel),
        ); }
        assert_eq!(cel.ptr, ptr);
      }
    }
    0
  }).unwrap_or_else(|_| {
    if cfg_debug() { println!("DEBUG: tl_pctx_nvgpu_mem_free_hook: ptr=0x{:016x} (pctx deinit)",
        ptr as usize,
    ); }
    0
  })
}

pub extern "C" fn tl_pctx_nvgpu_mem_unify_hook(lhs_raw_tag: *mut c_char, rhs_raw_tag: *mut c_char) {
  let (lhs_ctag, rhs_ctag) = unsafe {
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
    (lhs_ctag, rhs_ctag)
  };
  if cfg_debug() { println!("DEBUG: tl_pctx_nvgpu_mem_unify_hook: ltag={:?} rtag={:?}",
      lhs_ctag.map(|c| safe_ascii(c.to_bytes())),
      rhs_ctag.map(|c| safe_ascii(c.to_bytes())),
  ); }
}

pub extern "C" fn tl_pctx_gpu_alloc_hook(dptr: *mut u64, sz: usize, raw_tag: *const c_char) -> c_int {
  assert!(!dptr.is_null());
  TL_PCTX.with(|pctx| {
    /*println!("DEBUG: tl_pctx_gpu_alloc_hook: sz={} raw tag=0x{:016x}",
        sz,
        raw_tag as usize,
    );*/
    let gpu = pctx.nvgpu.as_ref().unwrap();
    let (req, tag) = if raw_tag.is_null() {
      if cfg_debug() { println!("DEBUG: tl_pctx_gpu_alloc_hook: sz={} ctag=(null)",
          sz,
      ); }
      let req = gpu.mem_pool.try_pre_alloc(sz);
      (req, None)
    } else {
      unsafe {
        let ctag = CStr::from_ptr(raw_tag);
        if cfg_debug() { println!("DEBUG: tl_pctx_gpu_alloc_hook: sz={} ctag=\"{}\"",
            sz,
            safe_ascii(ctag.to_bytes()),
        ); }
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
        if cfg_debug() { println!("DEBUG: tl_pctx_gpu_alloc_hook:   addr={:?} dptr=0x{:016x} size={} tag={:?}",
            p, cel.dptr.get(), req.size(), tag,
        ); }
      }
    }
    0
  })
}

pub extern "C" fn tl_pctx_gpu_free_hook(dptr: u64) -> c_int {
  TL_PCTX.try_with(|pctx| {
    if cfg_debug() { println!("DEBUG: tl_pctx_gpu_free_hook: dptr=0x{:016x}",
        dptr,
    ); }
    let gpu = pctx.nvgpu.as_ref().unwrap();
    let p = match gpu.mem_pool.rev_lookup(dptr) {
      Some((_, Some(p))) => p,
      _ => panic!("bug"),
    };
    gpu.mem_pool.tmp_freelist.borrow_mut().push(p);
    match gpu.mem_pool.cel_map.borrow().get(&p) {
      None => panic!("bug"),
      Some(cel) => {
        if cfg_debug() { println!("DEBUG: tl_pctx_gpu_free_hook:   addr={:?} dptr=0x{:016x} size={} tag={:?}",
            p, cel.dptr.get(), InnerCell::size(&**cel), InnerCell::tag(&**cel),
        ); }
      }
    }
    0
  }).unwrap_or_else(|_| {
    if cfg_debug() { println!("DEBUG: tl_pctx_gpu_free_hook: dptr=0x{:016x} (pctx deinit)",
        dptr,
    ); }
    0
  })
}

pub extern "C" fn tl_pctx_gpu_unify_hook(lhs_raw_tag: *const c_char, rhs_raw_tag: *const c_char) {
  let (lhs_ctag, rhs_ctag) = unsafe {
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
    (lhs_ctag, rhs_ctag)
  };
  if cfg_debug() { println!("DEBUG: tl_pctx_gpu_unify_hook: ltag={:?} rtag={:?}",
      lhs_ctag.map(|c| safe_ascii(c.to_bytes())),
      rhs_ctag.map(|c| safe_ascii(c.to_bytes())),
  ); }
  TL_PCTX.try_with(|pctx| {
    /*println!("DEBUG: tl_pctx_gpu_unify_hook: raw ltag=0x{:016x} raw rtag=0x{:016x}",
        lhs_raw_tag as usize,
        rhs_raw_tag as usize,
    );*/
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
  }).unwrap_or_else(|_| ())
}

pub extern "C" fn tl_pctx_gpu_failarg_alloc_hook(dptr: *mut u64, sz: usize) -> c_int {
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

pub extern "C" fn tl_pctx_gpu_failarg_free_hook(_dptr: u64) -> c_int {
  TL_PCTX.try_with(|_pctx| {
    // FIXME FIXME
    0
  }).unwrap_or_else(|_| 0)
}

pub fn is_subregion_dev(src_dptr: u64, src_sz: usize, dst_dptr: u64, dst_sz: usize) -> bool {
  let end_src_dptr = src_dptr + src_sz as u64;
  let end_dst_dptr = dst_dptr + dst_sz as u64;
  assert!(src_dptr <= end_src_dptr);
  assert!(dst_dptr <= end_dst_dptr);
  dst_dptr <= src_dptr && end_src_dptr <= end_dst_dptr
}

static ACCUMULATE_1D_F32_FUNCTIONNAME: &'static [u8] = b"cacti_rts_accumulate_1d_f32\0";
static ACCUMULATE_1D_F32_IDX32_SOURCE: &'static [u8] =
b"
typedef int i32;
extern \"C\" __global__ __launch_bounds__(MAX_THREADS_PER_BLOCK) void kernel(float *dst, const float *src, i32 n) {
  i32 idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < n) {
    dst[idx] = dst[idx] + src[idx];
  }
  return;
}
\0";

static ACCUMULATE_1D_F16_FUNCTIONNAME: &'static [u8] = b"cacti_rts_accumulate_1d_f16\0";
static ACCUMULATE_1D_F16_IDX32_SOURCE: &'static [u8] =
b"
#include <cuda_fp16.h>
typedef int i32;
typedef unsigned short u16;
typedef __half f16;
extern \"C\" __global__ __launch_bounds__(MAX_THREADS_PER_BLOCK) void kernel(u16 *dst, const u16 *src, i32 n) {
  i32 idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < n) {
    f16 x_0 = __ushort_as_half(src[idx]);
    f16 x_1 = __ushort_as_half(dst[idx]);
    f16 y_0 = x_0 + x_1;
    dst[idx] = __half_as_ushort(y_0);
  }
  return;
}
\0";

/*static ACCUMULATE_1D_F16_V2_FUNCTIONNAME: &'static [u8] = b"cacti_rts_accumulate_1d_f16_v2\0";
static ACCUMULATE_1D_F16_V2_IDX32_SOURCE: &'static [u8] =
b"
#include <cuda_fp16.h>
typedef int i32;
typedef unsigned short u16;
typedef __half f16;
typedef __half2 f16x2;
extern \"C\" __global__ __launch_bounds__(MAX_THREADS_PER_BLOCK) void kernel(u16 *dst, const u16 *src, i32 n) {
  i32 idx = threadIdx.x + blockDim.x * blockIdx.x;
  // FIXME: pointer alignment.
  if (idx * 2 + 1 < n) {
    f16x2 x_0 = __ldg((const f16x2 *)(src + idx));
    f16x2 x_1 = __ldg((const f16x2 *)(dst + idx));
    f16x2 y_0 = x_0 + x_1;
    __stwb((f16x2 *)(dst + idx), y_0);
  } else if (idx * 2 < n) {
    f16 x_0 = __ushort_as_half(src[idx]);
    f16 x_1 = __ushort_as_half(dst[idx]);
    f16 y_0 = x_0 + x_1;
    dst[idx] = __half_as_ushort(y_0);
  }
  return;
}
\0";*/

static ACCUMULATE_1D_U16_FUNCTIONNAME: &'static [u8] = b"cacti_rts_accumulate_1d_u16\0";
static ACCUMULATE_1D_U16_IDX32_SOURCE: &'static [u8] =
b"
typedef int i32;
typedef unsigned short u16;
extern \"C\" __global__ __launch_bounds__(MAX_THREADS_PER_BLOCK) void kernel(u16 *dst, const u16 *src, i32 n) {
  i32 idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < n) {
    dst[idx] = dst[idx] + src[idx];
  }
  return;
}
\0";

pub struct NvGpuCopyKernel {
  pub ptx:  Box<[u8]>,
  //pub prog: NvrtcProgram,
  pub mod_: CudaModule,
  pub func: CudaFunction,
}

impl NvGpuCopyKernel {
  pub fn from_source(fname_buf: &[u8], src_buf: &[u8]) -> NvGpuCopyKernel {
    assert_eq!(fname_buf[fname_buf.len() - 1], 0);
    if cfg_devel_dump() {
      println!("DEBUG: NvGpuCopyKernel: src sz={}", src_buf.len());
      let path = PathBuf::from(&format!(".tmp.{}.cu", from_utf8(&fname_buf[ .. fname_buf.len() - 1]).unwrap()));
      let mut tmp = OpenOptions::new().read(false).write(true).create(true).truncate(true).open(&path).unwrap();
      tmp.write_all(&src_buf[ .. src_buf.len() - 1]).unwrap();
      //tmp.write_all(src_buf).unwrap();
    }
    let prog = NvrtcProgram::create(src_buf).unwrap();
    // FIXME
    let mut opts = Vec::new();
    opts.push(b"-arch\0" as &[_]);
    opts.push(b"compute_75\0" as &[_]);
    opts.push(b"-default-device\0" as &[_]);
    opts.push(b"--disable-warnings\0" as &[_]);
    opts.push(b"-I/usr/local/cuda/include\0" as &[_]);
    opts.push(b"-I/usr/include\0" as &[_]);
    opts.push(b"-DMAX_THREADS_PER_BLOCK=1024\0" as &[_]);
    let res = prog.compile(&opts);
    let log_sz = prog.get_log_size().unwrap();
    let mut log = Vec::with_capacity(log_sz + 1);
    log.resize(log_sz + 1, 0);
    prog.get_log(&mut log).unwrap();
    if cfg_devel_dump() {
      println!("DEBUG: NvGpuCopyKernel: log sz={}", log_sz);
      let path = PathBuf::from(&format!(".tmp.{}.log", from_utf8(&fname_buf[ .. fname_buf.len() - 1]).unwrap()));
      let mut tmp = OpenOptions::new().read(false).write(true).create(true).truncate(true).open(&path).unwrap();
      tmp.write_all(&log[ .. log_sz - 1]).unwrap();
    }
    assert!(res.is_ok());
    let ptx_sz = prog.get_ptx_size().unwrap();
    let mut ptx = Vec::with_capacity(ptx_sz + 1);
    ptx.resize(ptx_sz + 1, 0);
    prog.get_ptx(&mut ptx).unwrap();
    if cfg_devel_dump() {
      println!("DEBUG: NvGpuCopyKernel: ptx sz={}", ptx_sz);
      let path = PathBuf::from(&format!(".tmp.{}.ptx", from_utf8(&fname_buf[ .. fname_buf.len() - 1]).unwrap()));
      let mut tmp = OpenOptions::new().read(false).write(true).create(true).truncate(true).open(&path).unwrap();
      tmp.write_all(&ptx[ .. ptx_sz - 1]).unwrap();
      //tmp.write_all(&ptx[ .. ptx_sz]).unwrap();
    }
    drop(prog);
    NvGpuCopyKernel::from_ptx(ptx.into())
  }

  pub fn from_ptx(ptx: Box<[u8]>) -> NvGpuCopyKernel {
    let mod_ = CudaModule::load_data(ptx.as_ptr() as *const _).unwrap();
    let func = mod_.get_function(b"kernel\0").unwrap();
    NvGpuCopyKernel{ptx, mod_, func}
  }

  pub fn launch32(&self, dst_dptr: u64, src_dptr: u64, len: i32, stream: &CudartStream) -> () {
    let dst_arg = UnsafeCell::new(dst_dptr);
    let src_arg = UnsafeCell::new(src_dptr);
    let len_arg = UnsafeCell::new(len);
    let args = UnsafeCell::new([
        dst_arg.get() as *mut _,
        src_arg.get() as *mut _,
        len_arg.get() as *mut _,
    ]);
    let gridx = (len as u32 + 256 - 1) / 256;
    self.func.launch_kernel([gridx, 1, 1], [256, 1, 1], 0, &args, stream).unwrap();
  }
}

pub struct NvGpuCopyKernels {
  pub accumulate_1d_f32_idx32: Option<NvGpuCopyKernel>,
  pub accumulate_1d_f16_idx32: Option<NvGpuCopyKernel>,
  pub accumulate_1d_u16_idx32: Option<NvGpuCopyKernel>,
  //pub strided_copy_2d_u32_idx32: Option<NvGpuCopyKernel>,
  //pub strided_accumulate_2d_f32_idx32: Option<NvGpuCopyKernel>,
  // TODO
}

impl NvGpuCopyKernels {
  pub fn new() -> NvGpuCopyKernels {
    NvGpuCopyKernels{
      accumulate_1d_f32_idx32:  Some(NvGpuCopyKernel::from_source(
          ACCUMULATE_1D_F32_FUNCTIONNAME,
          ACCUMULATE_1D_F32_IDX32_SOURCE
      )),
      accumulate_1d_f16_idx32:  Some(NvGpuCopyKernel::from_source(
          ACCUMULATE_1D_F16_FUNCTIONNAME,
          ACCUMULATE_1D_F16_IDX32_SOURCE
      )),
      accumulate_1d_u16_idx32:  Some(NvGpuCopyKernel::from_source(
          ACCUMULATE_1D_U16_FUNCTIONNAME,
          ACCUMULATE_1D_U16_IDX32_SOURCE
      )),
    }
  }
}

/*pub struct GpuDryCtx {
  // FIXME
}*/
