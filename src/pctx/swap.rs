use super::*;
use crate::algo::{HashMap, Region, RevSortMap8, StdCellExt};
use crate::cell::*;
use crate::clock::*;
use crate::util::mmap::*;

use std::cell::{Cell};

/*pub struct SwapInnerCell {
  pub clk:  Cell<Clock>,
}*/

#[repr(C)]
pub struct SwapCowMemCell {
  pub root: Cell<CellPtr>,
  // FIXME
  //pub refc: Cell<u32>,
  pub flag: Cell<u8>,
  pub mem:  MmapFileSlice,
}

impl Drop for SwapCowMemCell {
  fn drop(&mut self) {
    //let _ = self.buf.take();
  }
}

impl SwapCowMemCell {
  // TODO
}

impl InnerCell for SwapCowMemCell {
  fn as_mem_reg(&self) -> Option<MemReg> {
    if self.mem.as_ptr().is_null() {
      return None;
    }
    Some(MemReg{ptr: self.mem.as_ptr(), sz: self.mem.size_bytes()})
  }

  fn size(&self) -> usize {
    self.mem.size_bytes()
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

  // TODO
}

pub struct SwapPCtx {
  pub page_tab: RefCell<HashMap<PAddr, Rc<SwapCowMemCell>>>,
  //pub page_idx: RefCell<HashMap<*mut c_void, PAddr>>,
}

impl PCtxImpl for SwapPCtx {
  fn pmach(&self) -> PMach {
    PMach::Swap
  }

  fn fastest_locus(&self) -> Locus {
    Locus::Mem
  }

  fn append_matrix(&self, lp: &mut RevSortMap8<(Locus, PMach), ()>, pl: &mut RevSortMap8<(PMach, Locus), ()>) {
    lp.insert((Locus::Mem, PMach::Swap), ());
    pl.insert((PMach::Swap, Locus::Mem), ());
  }

  //fn try_alloc(&self, pctr: &PCtxCtr, locus: Locus, ty: &CellType) -> Result<PAddr, PMemErr> {}
  fn try_alloc(&self, pctr: &PCtxCtr, ty: &CellType, locus: Locus) -> Result<PAddr, PMemErr> {
    unimplemented!();
  }
}

impl SwapPCtx {
  pub fn new() -> SwapPCtx {
    SwapPCtx{
      page_tab: RefCell::new(HashMap::new()),
    }
  }

  pub fn lookup(&self, addr: PAddr) -> Option<(Locus, Rc<SwapCowMemCell>)> {
    match self.page_tab.borrow().get(&addr) {
      None => None,
      Some(icel) => Some((Locus::Mem, icel.clone()))
    }
  }

  pub fn lookup_mem_reg(&self, addr: PAddr) -> Option<MemReg> {
    match self.page_tab.borrow().get(&addr) {
      None => None,
      Some(icel) => Some(MemReg{
        ptr:  icel.mem.as_ptr(),
        sz:   icel.mem.size_bytes(),
      })
    }
  }

  pub fn _try_alloc_cow(&self, addr: PAddr, ty: &CellType, mem: MmapFileSlice) -> Result<Rc<SwapCowMemCell>, PMemErr> {
    if cfg_debug() {
      println!("DEBUG: SwapPCtx::_try_alloc: addr={:?} ty={:?} mem sz={}",
          addr, ty, mem.size_bytes());
    }
    let cel = Rc::new(SwapCowMemCell{
      root: Cell::new(CellPtr::nil()),
      //refc: Cell::new(1),
      flag: Cell::new(0),
      mem,
    });
    InnerCell_::set_cow(&*cel, true);
    assert!(self.page_tab.borrow_mut().insert(addr, cel.clone()).is_none());
    //assert!(self.page_idx.borrow_mut().insert(cel.ptr, addr).is_none());
    Ok(cel)
  }

  pub fn release(&self, addr: PAddr) -> Option<(Locus, Rc<SwapCowMemCell>)> {
    if cfg_debug() {
      println!("DEBUG: SwapPCtx::release: addr={:?}", addr);
    }
    match self.page_tab.borrow_mut().remove(&addr) {
      None => None,
      Some(icel) => Some((Locus::Mem, icel))
    }
  }
}
