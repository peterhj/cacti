use crate::algo::{HashMap, HashSet};
use crate::cell::{CellType, Dtype};
use crate::util::{FileOrPath};
use crate::util::mmap::{MmapFile, MmapFileSlice};
use cacti_cfg_env::*;

//pub use cell_split::{Dtype as TensorDtype};
pub use cell_split::safetensor::{TensorsDict, TensorDtype};
use glob::{glob};
use smol_str::{SmolStr};

use std::cell::{RefCell};
use std::convert::{TryFrom, TryInto};
use std::fs::{File};
use std::path::{PathBuf, Path};
use std::sync::{Arc};

pub type SafeTensorsDir = TensorsDir;
pub type SafeTensorMem = TensorMem;

#[derive(Debug)]
pub enum TensorsDirErr {
  Glob,
  Path,
  File,
  TensorsDict,
  DuplicateName(SmolStr),
}

pub struct TensorsDir {
  pub dir_path: PathBuf,
  pub model_paths: Vec<PathBuf>,
  pub serial_key: Vec<SmolStr>,
  pub tensor_key: HashSet<SmolStr>,
  pub tensor_map: HashMap<SmolStr, (usize, TensorEntry)>,
  pub model_files: Vec<RefCell<Option<MmapFile>>>,
}

impl AsRef<Path> for TensorsDir {
  fn as_ref(&self) -> &Path {
    self.dir_path.as_ref()
  }
}

impl TensorsDir {
  pub fn open<P: Into<PathBuf>>(p: P) -> Result<TensorsDir, TensorsDirErr> {
    let mut this = TensorsDir{
      dir_path: p.into(),
      model_paths: Vec::new(),
      serial_key: Vec::new(),
      tensor_key: HashSet::new(),
      tensor_map: HashMap::new(),
      model_files: Vec::new(),
    };
    this._reopen()?;
    Ok(this)
  }

  pub fn _reopen(&mut self) -> Result<(), TensorsDirErr> {
    self.model_paths.clear();
    self.serial_key.clear();
    self.tensor_key.clear();
    self.tensor_map.clear();
    let mut p = self.dir_path.clone();
    p.push("model.safetensors");
    let files: Vec<FileOrPath> = match File::open(&p) {
      Ok(f) => vec![(p, f).into()],
      Err(_) => {
        let mut p = self.dir_path.clone();
        p.push("model-*-of-*.safetensors");
        let mut files: Vec<FileOrPath> = Vec::new();
        for e in glob(p.to_str().unwrap()).map_err(|_| TensorsDirErr::Glob)? {
          match e {
            Err(_) => return Err(TensorsDirErr::Path),
            Ok(p) => {
              files.push(p.into());
            }
          }
        }
        if files.is_empty() {
          return Err(TensorsDirErr::Path);
        }
        files.sort_by(|lx, rx| lx.as_path().cmp(rx.as_path()));
        files
      }
    };
    for file_or_p in files.into_iter() {
      let (p, file) = file_or_p.try_open().map_err(|_| TensorsDirErr::File)?;
      //if cfg_debug() { println!("DEBUG: TensorsDir::_reopen: open \"{}\"...", safe_ascii(p.to_str().unwrap().as_bytes())); }
      let dict = TensorsDict::from_reader(file).map_err(|_| TensorsDirErr::TensorsDict)?;
      let model_idx = self.model_paths.len();
      self.model_paths.push(p);
      self.model_files.push(RefCell::new(None));
      for (name, entry) in dict.entries.iter() {
        if self.tensor_key.contains(name) {
          return Err(TensorsDirErr::DuplicateName(name.clone()));
        }
        let off = dict.buf_start + entry.data_offsets[0];
        let eoff = dict.buf_start + entry.data_offsets[1];
        let shape = entry.shape.clone();
        let dtype = Dtype::try_from(entry.dtype).unwrap();
        let ty = CellType{shape, dtype};
        let e = TensorEntry{ty, off, eoff};
        //if cfg_debug() { println!("DEBUG: TensorsDir::_reopen:   name=\"{}\"", safe_ascii(t.name.as_bytes())); }
        self.serial_key.push(name.clone());
        self.tensor_key.insert(name.clone());
        self.tensor_map.insert(name.clone(), (model_idx, e));
      }
      if cfg_debug() { println!("DEBUG: TensorsDir::_reopen:   done"); }
    }
    Ok(())
  }

  pub fn clone_keys(&self) -> HashSet<SmolStr> {
    self.tensor_key.clone()
  }

  pub fn get(&self, key: &SmolStr) -> (CellType, TensorMem) {
    match self.tensor_map.get(key) {
      None => panic!("bug"),
      Some(&(model_idx, ref e)) => {
        let span = e.ty.packed_span_bytes();
        /*let (_stride, span) = e.ty.packed_stride_and_span_bytes();*/
        let offset = e.off;
        let size = e.eoff - e.off;
        assert_eq!(span, size);
        if cfg_debug() { println!("DEBUG: TensorsDir::open: offset={} size={}", offset, size); }
        let file = if self.model_files[model_idx].borrow().is_none() {
          let f = Arc::new(File::open(&self.model_paths[model_idx]).unwrap());
          let file = MmapFile::from_file(&f).unwrap();
          *self.model_files[model_idx].borrow_mut() = Some(file.clone());
          file
        } else {
          self.model_files[model_idx].borrow().as_ref().unwrap().clone()
        };
        let off: usize = e.off.try_into().unwrap();
        let eoff: usize = e.eoff.try_into().unwrap();
        let mmap = file.slice(off, eoff);
        let ten = TensorMem{mmap};
        (e.ty.clone(), ten)
      }
    }
  }
}

pub struct TensorEntry {
  pub ty:   CellType,
  pub off:  u64,
  pub eoff: u64,
}

#[derive(Clone)]
pub struct TensorMem {
  pub mmap: MmapFileSlice,
}

impl TensorMem {
  pub fn mmap(&self) -> &MmapFileSlice {
    &self.mmap
  }
}
