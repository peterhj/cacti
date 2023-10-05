use std::fs::{File};
use std::io::{Error as IoError};
use std::path::{PathBuf, Path};

pub mod cell;
pub mod mmap;
pub mod pickle { pub use super::safepickle::*; }
pub mod safepickle;
pub mod safetensor;
pub mod safetensors { pub use super::safetensor::*; }
pub mod stat;
pub mod time;

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
