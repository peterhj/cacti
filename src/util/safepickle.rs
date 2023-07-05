use crate::algo::{HashMap, HashSet};
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
use std::path::{PathBuf};

#[derive(Debug)]
pub enum PickleDirErr {
  Path,
  DuplicateName(SmolStr),
  _Bot,
}

pub struct PickleDir {
  pub dir_path: PathBuf,
  pub model_paths: Vec<PathBuf>,
  pub tensor_key: HashSet<SmolStr>,
  pub tensor_map: HashMap<SmolStr, (usize, PickleTensor)>,
}

impl PickleDir {
  pub fn from<P: Into<PathBuf>>(p: P) -> PickleDir {
    PickleDir{
      dir_path: p.into(),
      model_paths: Vec::new(),
      tensor_key: HashSet::new(),
      tensor_map: HashMap::new(),
    }
  }

  pub fn _reload(&mut self) -> Result<(), PickleDirErr> {
    self.model_paths.clear();
    self.tensor_key.clear();
    self.tensor_map.clear();
    let mut p = self.dir_path.clone();
    let filename = format!("pytorch_model.bin");
    p.push(filename);
    //let prefix1 = format!("pytorch_model-00001-of-*");
    let mut file = match PickleFile::open(&p) {
      Err(_) => return Err(PickleDirErr::Path),
      Ok(f) => f
    };
    let mut iter = file.iter_tensors_data();
    /*loop {
      let r = iter.next_tensor();
      if r.is_none() {
        break;
      }
    }*/
    iter._fixup_offsets();
    drop(iter);
    self.model_paths.push(p);
    for t in file.tensors().iter() {
      if self.tensor_key.contains(&t.name) {
        return Err(PickleDirErr::DuplicateName(t.name.clone()));
      }
      self.tensor_key.insert(t.name.clone());
      self.tensor_map.insert(t.name.clone(), (0, t.clone()));
    }
    Ok(())
  }

  pub fn clone_keys(&self) -> HashSet<SmolStr> {
    self.tensor_key.clone()
  }

  pub fn get(&self, key: &SmolStr) -> (CellType, /*CellLayout,*/ PickleSlice) {
    match self.tensor_map.get(key) {
      None => panic!("bug"),
      Some(&(_, ref t)) => {
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
        let mut file = File::open(&self.model_paths[0]).unwrap();
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
