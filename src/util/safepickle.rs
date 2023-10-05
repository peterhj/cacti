use crate::algo::{BTreeMap, BTreeSet};
use crate::algo::str::*;
use crate::cell::{CellType, Dtype};
use crate::util::{FileOrPath};
use crate::util::mmap::*;
use cacti_cfg_env::*;

use glob::{glob};
pub use repugnant_pickle::torch::{
    TensorType as TorchDtype,
    RepugnantTorchTensor as PickleTensor,
    //RepugnantTorchTensorsIter as PickleTensorsIter,
    RepugnantTorchFile as PickleFile,
};
use smol_str::{SmolStr};

use std::cell::{RefCell};
use std::cmp::{min};
use std::convert::{TryFrom, TryInto};
use std::fs::{File};
//use std::io::{Read, Seek, SeekFrom};
use std::io::{Error as IoError};
use std::ops::{Deref};
use std::path::{PathBuf, Path};
use std::sync::{Arc};

#[derive(Debug)]
pub enum PickleDirErr {
  Glob,
  Path,
  File,
  PickleFile,
  DuplicateName(SmolStr),
}

pub struct PickleDir {
  pub dir_path: PathBuf,
  pub model_paths: Vec<PathBuf>,
  pub serial_key: Vec<SmolStr>,
  pub tensor_key: BTreeSet<SmolStr>,
  pub tensor_map: BTreeMap<SmolStr, (usize, PickleTensor)>,
  pub model_files: Vec<RefCell<Option<MmapFile>>>,
}

impl AsRef<Path> for PickleDir {
  fn as_ref(&self) -> &Path {
    self.dir_path.as_ref()
  }
}

impl PickleDir {
  pub fn open<P: Into<PathBuf>>(p: P) -> Result<PickleDir, PickleDirErr> {
    let mut this = PickleDir{
      dir_path: p.into(),
      model_paths: Vec::new(),
      serial_key: Vec::new(),
      tensor_key: BTreeSet::default(),
      tensor_map: BTreeMap::default(),
      model_files: Vec::new(),
    };
    this._reopen()?;
    Ok(this)
  }

  pub fn _reopen(&mut self) -> Result<(), PickleDirErr> {
    self.model_paths.clear();
    self.serial_key.clear();
    self.tensor_key.clear();
    self.tensor_map.clear();
    let mut p = self.dir_path.clone();
    p.push("pytorch_model.bin");
    let files: Vec<FileOrPath> = match File::open(&p) {
      Ok(f) => vec![(p, f).into()],
      Err(_) => {
        let mut p = self.dir_path.clone();
        p.push("pytorch_model-*-of-*.bin");
        let mut files: Vec<FileOrPath> = Vec::new();
        for e in glob(p.to_str().unwrap()).map_err(|_| PickleDirErr::Glob)? {
          match e {
            Err(_) => return Err(PickleDirErr::Path),
            Ok(p) => {
              files.push(p.into());
            }
          }
        }
        if files.is_empty() {
          return Err(PickleDirErr::Path);
        }
        files.sort_by(|lx, rx| lx.as_path().cmp(rx.as_path()));
        files
      }
    };
    for file_or_p in files.into_iter() {
      let (p, file) = file_or_p.try_open().map_err(|_| PickleDirErr::File)?;
      if cfg_debug() { println!("DEBUG: PickleDir::_reopen: open \"{}\"...", safe_ascii(p.to_str().unwrap().as_bytes())); }
      let file = PickleFile::new(file).map_err(|_| PickleDirErr::PickleFile)?;
      let model_idx = self.model_paths.len();
      self.model_paths.push(p);
      self.model_files.push(RefCell::new(None));
      for t in file.tensors().iter() {
        if self.tensor_key.contains(&t.name) {
          return Err(PickleDirErr::DuplicateName(t.name.clone()));
        }
        if cfg_debug() { println!("DEBUG: PickleDir::_reopen:   name=\"{}\"", safe_ascii(t.name.as_bytes())); }
        self.serial_key.push(t.name.clone());
        self.tensor_key.insert(t.name.clone());
        self.tensor_map.insert(t.name.clone(), (model_idx, t.clone()));
      }
      if cfg_debug() { println!("DEBUG: PickleDir::_reopen:   done"); }
    }
    Ok(())
  }

  pub fn clone_keys(&self) -> BTreeSet<SmolStr> {
    self.tensor_key.clone()
  }

  pub fn get<K: AsRef<str>>(&self, key: K) -> (CellType, PickleSlice) {
    let key = key.as_ref();
    match self.tensor_map.get(key) {
      None => panic!("bug"),
      Some(&(model_idx, ref t)) => {
        let dtype = match t.type_.as_ref() {
          Err(s) => {
            println!("ERROR:  PickleDir::get: unknown tensor dtype: \"{}\"", safe_ascii(s.as_bytes()));
            panic!();
          }
          Ok(&tt) => Dtype::try_from(tt).unwrap()
        };
        let shape = t.shape.clone().into();
        let ty = CellType{shape, dtype};
        // FIXME: strided layout.
        //let span = ty.packed_span_bytes();
        let (stride, span) = ty.packed_stride_and_span_bytes();
        if !(stride.len() <= t.stride.len()) {
          println!("ERROR:  PickleDir::get: unexpected tensor stride: key=\"{}\" shape={:?} packed stride={:?} actual stride={:?}",
              safe_ascii(key.as_bytes()), &ty.shape, &stride, &t.stride);
          panic!();
        }
        if &stride != &t.stride[t.stride.len() - stride.len() .. ] {
          println!("ERROR:  PickleDir::get: loading strided (non-packed) tensors is not supported: key=\"{}\" shape={:?} packed stride={:?} actual stride={:?}",
              safe_ascii(key.as_bytes()), &ty.shape, &stride, &t.stride);
          panic!();
        }
        let offset = t.storage_offset + t.storage_start * dtype.size_bytes() as u64;
        let rem_size = (t.storage_end - t.storage_start) * dtype.size_bytes() as u64;
        if cfg_debug() {
          println!("DEBUG:  PickleDir::open: key=\"{}\" ty={:?}", safe_ascii(key.as_bytes()), ty);
          println!("DEBUG:  PickleDir::open: stride={:?} span={:?}", stride, span);
          println!("DEBUG:  PickleDir::open: data offset={} start={} end={} dty sz={}",
              t.storage_offset, t.storage_start, t.storage_end, dtype.size_bytes());
          println!("DEBUG:  PickleDir::open: offset={:?} rem size={:?}", offset, rem_size);
        }
        /*if span != rem_size {
          println!("WARNING:PickleDir::open: span != rem_size, but continuing...");
        }*/
        assert!(span <= rem_size);
        assert!(offset <= t.storage_offset + t.storage_size);
        assert_eq!(offset + rem_size, t.storage_offset + t.storage_size);
        let mut align = 1;
        for shift in 1 ..= 12 {
          let a = 1 << shift;
          if offset & (a - 1) == 0 {
            align = a;
          } else {
            break;
          }
        }
        if cfg_debug() { println!("DEBUG:  PickleDir::open: align?={:?}", align); }
        assert!(align >= 64);
        let file = if self.model_files[model_idx].borrow().is_none() {
          let f = Arc::new(File::open(&self.model_paths[model_idx]).unwrap());
          let file = MmapFile::from_file(&f).unwrap();
          if cfg_debug() { println!("DEBUG:  PickleDir::open: file idx={} size={}", model_idx, file.size_bytes()); }
          *self.model_files[model_idx].borrow_mut() = Some(file.clone());
          file
        } else {
          self.model_files[model_idx].borrow().as_ref().unwrap().clone()
        };
        let off: usize = offset.try_into().unwrap();
        let sz: usize = span.try_into().unwrap();
        let eoff = off + sz;
        let mem = file.slice(off .. eoff);
        let slice = PickleSlice{mem};
        (ty, slice)
      }
    }
  }
}

#[derive(Clone)]
pub struct PickleSlice {
  pub mem:  MmapFileSlice,
  /*pub file:     File,
  pub pos:      u64,*/
}

/*impl Read for PickleSlice {
  fn read(&mut self, buf: &mut [u8]) -> Result<usize, IoError> {
    if self.pos >= self.size {
      return Ok(0);
    }
    let rem = buf.len() as u64;
    let buf_end = min(self.pos + rem, self.size);
    match self.file.read(&mut buf[ .. buf_end as usize]) {
      Err(e) => Err(e),
      Ok(n) => {
        self.pos += n as u64;
        Ok(n)
      }
    }
  }
}*/

impl AsRef<MmapFileSlice> for PickleSlice {
  fn as_ref(&self) -> &MmapFileSlice {
    &self.mem
  }
}

impl Deref for PickleSlice {
  type Target = MmapFileSlice;

  fn deref(&self) -> &MmapFileSlice {
    &self.mem
  }
}

impl PickleSlice {
  pub fn mmap(&self) -> &MmapFileSlice {
    &self.mem
  }
}

/*impl PickleSlice {
  pub fn size_bytes(&self) -> usize {
    self.mem.size_bytes()
  }

  pub fn as_bytes(&self) -> &[u8] {
    self.mem.as_bytes()
  }
}*/
