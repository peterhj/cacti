#[cfg(unix)]
use libc::{
  PROT_NONE, PROT_READ, PROT_WRITE,
  MAP_FAILED, MAP_ANON, MAP_PRIVATE, MAP_SHARED,
  mmap, munmap,
};
#[cfg(target_os = "linux")]
use libc::{
  MAP_NORESERVE, MAP_HUGETLB,
};

use std::convert::{TryInto};
use std::ffi::{c_void};
use std::fs::{File};
use std::mem::{swap};
#[cfg(unix)] use std::os::unix::fs::{MetadataExt};
#[cfg(unix)] use std::os::unix::io::{AsRawFd};
use std::ptr::{null_mut};
use std::slice::{from_raw_parts};
use std::sync::{Arc};

#[derive(Clone)]
pub struct MmapFileSlice {
  pub file: MmapFile,
  pub off:  usize,
  pub sz:   usize,
}

impl From<MmapFile> for MmapFileSlice {
  fn from(file: MmapFile) -> MmapFileSlice {
    let off = 0;
    let sz = file.size;
    MmapFileSlice{file, off, sz}
  }
}

impl MmapFileSlice {
  pub fn as_ptr(&self) -> *mut c_void {
    unsafe { (self.file.ptr as *mut u8).offset(self.off.try_into().unwrap()) as *mut c_void }
  }

  pub fn size_bytes(&self) -> usize {
    self.sz
  }

  pub unsafe fn as_unsafe_bytes(&self) -> &[u8] {
    let ptr = self.as_ptr();
    from_raw_parts(ptr as *mut u8 as *const u8, self.sz)
  }
}

#[derive(Clone)]
pub struct MmapFile {
  pub file: Arc<File>,
  pub ptr:  *mut c_void,
  pub size: usize,
}

impl Drop for MmapFile {
  #[cfg(not(unix))]
  fn drop(&mut self) {
    unimplemented!();
  }

  #[cfg(unix)]
  fn drop(&mut self) {
    // FIXME: weak count?
    if Arc::strong_count(&self.file) == 1 {
      assert!(!self.ptr.is_null());
      let ret = unsafe { munmap(self.ptr, self.size) };
      assert_eq!(ret, 0);
    }
  }
}

impl MmapFile {
  #[cfg(not(unix))]
  pub fn from_file(_f: &Arc<File>) -> Result<MmapFile, ()> {
    unimplemented!();
  }

  #[cfg(unix)]
  pub fn from_file(f: &Arc<File>) -> Result<MmapFile, ()> {
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
    Ok(MmapFile{ptr, size, file: f.clone()})
  }

  pub fn as_ptr(&self) -> *mut c_void {
    self.ptr
  }

  pub fn size_bytes(&self) -> usize {
    self.size
  }

  pub unsafe fn as_unsafe_bytes(&self) -> &[u8] {
    from_raw_parts(self.ptr as *mut u8 as *const u8, self.size)
  }

  pub fn to_ref(&self) -> UnsafeMmapRef {
    UnsafeMmapRef {
      ptr:  self.ptr,
      size: self.size,
    }
  }

  pub fn slice(&self, start: usize, end: usize) -> MmapFileSlice {
    let off = start;
    let sz = end - start;
    MmapFileSlice{file: self.clone(), off, sz}
  }
}

pub struct MmapBuf {
  pub ptr:  *mut c_void,
  pub size: usize,
}

impl Drop for MmapBuf {
  #[cfg(not(unix))]
  fn drop(&mut self) {
    unimplemented!();
  }

  #[cfg(unix)]
  fn drop(&mut self) {
    if self.ptr.is_null() {
      return;
    }
    let ret = unsafe { munmap(self.ptr, self.size) };
    assert_eq!(ret, 0);
  }
}

impl MmapBuf {
  pub fn null() -> MmapBuf {
    MmapBuf{
      ptr:  null_mut(),
      size: 0,
    }
  }

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

  #[cfg(target_os = "linux")]
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

  pub fn take(&mut self) -> MmapBuf {
    let mut new_ptr = null_mut();
    let mut new_size = 0;
    swap(&mut self.ptr, &mut new_ptr);
    swap(&mut self.size, &mut new_size);
    MmapBuf{
      ptr:  new_ptr,
      size: new_size,
    }
  }

  pub fn as_ptr(&self) -> *mut c_void {
    self.ptr
  }

  pub fn size_bytes(&self) -> usize {
    self.size
  }

  pub unsafe fn as_unsafe_bytes(&self) -> &[u8] {
    from_raw_parts(self.ptr as *mut u8 as *const u8, self.size)
  }

  pub fn to_ref(&self) -> UnsafeMmapRef {
    UnsafeMmapRef {
      ptr:  self.ptr,
      size: self.size,
    }
  }
}

#[derive(Clone, Copy)]
pub struct UnsafeMmapRef {
  pub ptr:  *mut c_void,
  pub size: usize,
}

impl UnsafeMmapRef {
  pub fn as_ptr(&self) -> *mut c_void {
    self.ptr
  }

  pub fn size_bytes(&self) -> usize {
    self.size
  }

  pub unsafe fn as_unsafe_bytes(&self) -> &[u8] {
    from_raw_parts(self.ptr as *mut u8 as *const u8, self.size)
  }
}
