#[cfg(unix)]
use libc::{
  PROT_NONE, PROT_READ, PROT_WRITE,
  MAP_FAILED, MAP_ANON, MAP_PRIVATE, MAP_SHARED,
  MAP_NORESERVE, MAP_HUGETLB,
  mmap, munmap,
};

use std::ffi::{c_void};
use std::fs::{File};
#[cfg(unix)] use std::os::unix::fs::{MetadataExt};
#[cfg(unix)] use std::os::unix::io::{AsRawFd};
use std::ptr::{null_mut};
use std::slice::{from_raw_parts};

pub struct MmapBuf {
  ptr:  *mut c_void,
  size: usize,
}

impl Drop for MmapBuf {
  #[cfg(not(unix))]
  fn drop(&mut self) {
    unimplemented!();
  }

  #[cfg(unix)]
  fn drop(&mut self) {
    assert!(!self.ptr.is_null());
    let ret = unsafe { munmap(self.ptr, self.size) };
    assert_eq!(ret, 0);
  }
}

impl MmapBuf {
  #[cfg(not(unix))]
  pub fn new_anon(_size: usize) -> Result<MmapBuf, ()> {
    unimplemented!();
  }

  #[cfg(not(unix))]
  pub fn from_file(_f: &File) -> Result<MmapBuf, ()> {
    unimplemented!();
  }

  #[cfg(unix)]
  pub fn new_anon(size: usize) -> Result<MmapBuf, ()> {
    let ptr = unsafe { mmap(null_mut(), size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0) };
    if ptr == MAP_FAILED {
      return Err(());
    }
    assert!(!ptr.is_null());
    Ok(MmapBuf{ptr, size})
  }

  #[cfg(unix)]
  pub fn new_noderef(size: usize) -> Result<MmapBuf, ()> {
    let ptr = unsafe { mmap(null_mut(), size, PROT_NONE, MAP_PRIVATE | MAP_ANON | MAP_NORESERVE | MAP_HUGETLB, -1, 0) };
    if ptr == MAP_FAILED {
      return Err(());
    }
    assert!(!ptr.is_null());
    Ok(MmapBuf{ptr, size})
  }

  #[cfg(unix)]
  pub fn from_file(f: &File) -> Result<MmapBuf, ()> {
    let size = f.metadata().unwrap().size();
    if size > usize::max_value() as u64 {
      return Err(());
    }
    let size = size as usize;
    let fd = f.as_raw_fd();
    let ptr = unsafe { mmap(null_mut(), size, PROT_READ, MAP_SHARED, fd, 0) };
    if ptr == MAP_FAILED {
      return Err(());
    }
    assert!(!ptr.is_null());
    Ok(MmapBuf{ptr, size})
  }

  pub fn as_ptr(&self) -> *mut c_void {
    self.ptr
  }

  pub fn size_bytes(&self) -> usize {
    self.size
  }

  pub fn as_bytes(&self) -> &[u8] {
    self.bytes()
  }

  pub fn bytes(&self) -> &[u8] {
    unsafe { from_raw_parts(self.ptr as *mut u8 as *const u8, self.size) }
  }
}
