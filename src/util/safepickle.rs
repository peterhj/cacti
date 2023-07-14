use crate::algo::{HashMap, HashSet};
use crate::algo::str::*;
use crate::cell::{CellType, Dtype};

use glob::{glob};
pub use repugnant_pickle::torch::{
    TensorType as TorchDtype,
    RepugnantTorchTensor as PickleTensor,
    RepugnantTorchTensorsIter as PickleTensorsIter,
    RepugnantTorchFile as PickleFile,
};
use smol_str::{SmolStr};

use std::cmp::{min};
use std::convert::{TryFrom};
use std::fs::{File};
use std::io::{Read, Seek, SeekFrom, Error as IoError};
use std::path::{PathBuf, Path};

pub enum FileOrPath {
  Path(PathBuf),
  File(PathBuf, File),
}

impl From<PathBuf> for FileOrPath {
  fn from(p: PathBuf) -> FileOrPath {
    FileOrPath::Path(p)
  }
}

impl From<(PathBuf, File)> for FileOrPath {
  fn from((p, f): (PathBuf, File)) -> FileOrPath {
    FileOrPath::File(p, f)
  }
}

impl FileOrPath {
  pub fn try_open(self) -> Result<(PathBuf, File), IoError> {
    match self {
      FileOrPath::Path(p) => {
        File::open(&p).map(|f| (p, f))
      }
      FileOrPath::File(p, f) => Ok((p, f))
    }
  }

  pub fn as_path(&self) -> &Path {
    match self {
      &FileOrPath::Path(ref p) => p,
      &FileOrPath::File(ref p, _) => p,
    }
  }
}

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
  pub tensor_key: HashSet<SmolStr>,
  pub tensor_map: HashMap<SmolStr, (usize, PickleTensor)>,
}

impl PickleDir {
  pub fn from<P: Into<PathBuf>>(p: P) -> Result<PickleDir, PickleDirErr> {
    PickleDir::open(p)
  }

  pub fn open<P: Into<PathBuf>>(p: P) -> Result<PickleDir, PickleDirErr> {
    let mut this = PickleDir{
      dir_path: p.into(),
      model_paths: Vec::new(),
      tensor_key: HashSet::new(),
      tensor_map: HashMap::new(),
    };
    this._reopen()?;
    Ok(this)
  }

  pub fn _reopen(&mut self) -> Result<(), PickleDirErr> {
    self.model_paths.clear();
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
      println!("DEBUG: PickleDir::_reopen: open \"{}\"...", safe_ascii(p.to_str().unwrap().as_bytes()));
      let mut file = PickleFile::new(file).map_err(|_| PickleDirErr::PickleFile)?;
      let mut iter = file.iter_tensors_data();
      iter._fixup_offsets();
      drop(iter);
      let model_idx = self.model_paths.len();
      self.model_paths.push(p);
      for t in file.tensors().iter() {
        if self.tensor_key.contains(&t.name) {
          return Err(PickleDirErr::DuplicateName(t.name.clone()));
        }
        println!("DEBUG: PickleDir::_reopen:   name=\"{}\"", safe_ascii(t.name.as_bytes()));
        self.tensor_key.insert(t.name.clone());
        self.tensor_map.insert(t.name.clone(), (model_idx, t.clone()));
      }
      println!("DEBUG: PickleDir::_reopen:   done");
    }
    Ok(())
  }

  pub fn clone_keys(&self) -> HashSet<SmolStr> {
    self.tensor_key.clone()
  }

  pub fn get(&self, key: &SmolStr) -> (CellType, /*CellLayout,*/ PickleSlice) {
    match self.tensor_map.get(key) {
      None => panic!("bug"),
      Some(&(model_idx, ref t)) => {
        let dtype = match t.tensor_type {
          Err(_) => Dtype::top(),
          Ok(tt) => Dtype::try_from(tt).unwrap()
        };
        let shape = t.shape.clone();
        // FIXME FIXME: layout.
        //let stride = ;
        let ty = CellType{shape, dtype};
        let span = ty.packed_span_bytes();
        let offset = t.absolute_offset;
        let size = t.storage_len * dtype.size_bytes() as u64;
        assert_eq!(span, size);
        // FIXME: mmap.
        let mut file = File::open(&self.model_paths[model_idx]).unwrap();
        file.seek(SeekFrom::Start(offset)).unwrap();
        println!("DEBUG: PickleDir::open: offset={} size={}", offset, size);
        let slice = PickleSlice{
          offset,
          size,
          file,
          pos:    0,
        };
        (ty, slice)
      }
    }
  }
}

pub struct PickleSlice {
  pub offset:   u64,
  pub size:     u64,
  pub file:     File,
  pub pos:      u64,
}

impl Read for PickleSlice {
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
}
