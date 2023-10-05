use crate::algo::{BTreeMap, BTreeSet, HashMap};
use crate::cell::{StableCell, CellType, Dtype};
use crate::util::mmap::{MmapFile, MmapFileSlice};

pub use cell_split::{CellSplit, CellRepr};
use cell_split::{CellType as ExtCellType, Dtype as ExtDtype};
use smol_str::{SmolStr};

use std::cell::{RefCell};
use std::convert::{TryInto};
use std::fs::{File};
use std::path::{Path};
use std::sync::{Arc};

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Pat {
  Int(i32),
  Str(SmolStr),
}

impl From<i32> for Pat {
  fn from(x: i32) -> Pat {
    Pat::Int(x)
  }
}

impl From<usize> for Pat {
  fn from(x: usize) -> Pat {
    assert!(x < i32::max_value() as usize);
    Pat::Int(x as _)
  }
}

impl<'a> From<&'a str> for Pat {
  fn from(x: &'a str) -> Pat {
    Pat::Str(x.into())
  }
}

/*impl<S: Borrow<str>> From<S> for Pat {
  fn from(x: S) -> Pat {
    Pat::Str(x.borrow().into())
  }
}*/

pub struct CellMatcher {
  pub rule: Vec<(Box<[Pat]>, StableCell)>,
  pub map:  BTreeMap<Box<[Pat]>, usize>,
}

impl CellMatcher {
  pub fn new() -> CellMatcher {
    CellMatcher{
      rule: Vec::new(),
      map:  BTreeMap::default(),
    }
  }

  pub fn len(&self) -> usize {
    self.rule.len()
  }

  pub fn match_(&self, mut keys: BTreeSet<SmolStr>) -> CellMatches {
    let mut mat = Vec::new();
    let mut map = BTreeMap::default();
    for &(ref pat, ref cel) in self.rule.iter() {
      //println!("DEBUG: CellMatcher: pat={:?} cel={:?}", pat, cel);
      let mut kmat = None;
      let mut idx = 0;
      for key in keys.iter() {
        let k = key.as_str();
        //println!("DEBUG: CellMatcher:   key={}", k);
        idx = 0;
        for tok in k.split(".") {
          //println!("DEBUG: CellMatcher:   idx={} tok={}", idx, tok);
          match &pat[idx] {
            &Pat::Int(x) => {
              if tok == &format!("{}", x) {
                idx += 1;
              }
            }
            &Pat::Str(ref x) => {
              if tok == x {
                idx += 1;
              } else if tok.find(x.as_str()).is_some() {
                idx += 1;
              }
            }
          }
          if idx == pat.len() {
            //println!("DEBUG: CellMatcher:   match idx={}", idx);
            break;
          }
        }
        if idx == pat.len() {
          kmat = Some(key.clone());
          break;
        }
      }
      if idx == pat.len() {
        let key = kmat.unwrap();
        keys.remove(&key);
        map.insert(key.clone(), mat.len());
        mat.push((key, cel.clone()));
      }
    }
    CellMatches{mat, map}
  }
}

pub trait CellMatcherExt<K> {
  fn insert(&mut self, key: K, cel: StableCell);
}

impl<K0: Into<Pat>> CellMatcherExt<K0> for CellMatcher {
  fn insert(&mut self, key: K0, cel: StableCell) {
    let mut pat = Vec::with_capacity(1);
    pat.push(key.into());
    let pat: Box<[_]> = pat.into();
    assert!(self.map.insert(pat.clone(), self.rule.len()).is_none());
    self.rule.push((pat, cel))
  }
}

impl<K0: Into<Pat>> CellMatcherExt<(K0,)> for CellMatcher {
  fn insert(&mut self, key: (K0,), cel: StableCell) {
    let mut pat = Vec::with_capacity(1);
    pat.push(key.0.into());
    let pat: Box<[_]> = pat.into();
    assert!(self.map.insert(pat.clone(), self.rule.len()).is_none());
    self.rule.push((pat, cel))
  }
}

impl<K0: Into<Pat>, K1: Into<Pat>> CellMatcherExt<(K0, K1)> for CellMatcher {
  fn insert(&mut self, key: (K0, K1), cel: StableCell) {
    let mut pat = Vec::with_capacity(2);
    pat.push(key.0.into());
    pat.push(key.1.into());
    let pat: Box<[_]> = pat.into();
    assert!(self.map.insert(pat.clone(), self.rule.len()).is_none());
    self.rule.push((pat, cel))
  }
}

impl<K0: Into<Pat>, K1: Into<Pat>, K2: Into<Pat>> CellMatcherExt<(K0, K1, K2)> for CellMatcher {
  fn insert(&mut self, key: (K0, K1, K2), cel: StableCell) {
    let mut pat = Vec::with_capacity(3);
    pat.push(key.0.into());
    pat.push(key.1.into());
    pat.push(key.2.into());
    let pat: Box<[_]> = pat.into();
    assert!(self.map.insert(pat.clone(), self.rule.len()).is_none());
    self.rule.push((pat, cel))
  }
}

impl<K0: Into<Pat>, K1: Into<Pat>, K2: Into<Pat>, K3: Into<Pat>> CellMatcherExt<(K0, K1, K2, K3)> for CellMatcher {
  fn insert(&mut self, key: (K0, K1, K2, K3), cel: StableCell) {
    let mut pat = Vec::with_capacity(4);
    pat.push(key.0.into());
    pat.push(key.1.into());
    pat.push(key.2.into());
    pat.push(key.3.into());
    let pat: Box<[_]> = pat.into();
    assert!(self.map.insert(pat.clone(), self.rule.len()).is_none());
    self.rule.push((pat, cel))
  }
}

impl<K0: Into<Pat>, K1: Into<Pat>, K2: Into<Pat>, K3: Into<Pat>, K4: Into<Pat>> CellMatcherExt<(K0, K1, K2, K3, K4)> for CellMatcher {
  fn insert(&mut self, key: (K0, K1, K2, K3, K4), cel: StableCell) {
    let mut pat = Vec::with_capacity(4);
    pat.push(key.0.into());
    pat.push(key.1.into());
    pat.push(key.2.into());
    pat.push(key.3.into());
    pat.push(key.4.into());
    let pat: Box<[_]> = pat.into();
    assert!(self.map.insert(pat.clone(), self.rule.len()).is_none());
    self.rule.push((pat, cel))
  }
}

pub struct CellMatches {
  pub mat:  Vec<(SmolStr, StableCell)>,
  pub map:  BTreeMap<SmolStr, usize>,
}

impl CellMatches {
  pub fn inv(&self) -> CellInvertedMatches {
    let mut mat = Vec::new();
    let mut map = HashMap::default();
    for &(ref key, ref cel) in self.mat.iter() {
      let cel = cel.clone();
      let key = key.clone();
      assert!(map.insert(cel.clone(), mat.len()).is_none());
      mat.push((cel, key));
    }
    CellInvertedMatches{mat, map}
  }

  pub fn len(&self) -> usize {
    self.mat.len()
  }

  pub fn iter(&self) -> impl Iterator<Item=(&SmolStr, &StableCell)> {
    self.mat.iter().map(|&(ref key, ref cel)| (key, cel))
  }

  pub fn get(&self, key: &SmolStr) -> StableCell {
    match self.map.get(key) {
      None => panic!("bug"),
      Some(&idx) => {
        self.mat[idx].1.clone()
      }
    }
  }
}

pub type CellInvMatches = CellInvertedMatches;

pub struct CellInvertedMatches {
  pub mat:  Vec<(StableCell, SmolStr)>,
  pub map:  HashMap<StableCell, usize>,
}

impl CellInvertedMatches {
  pub fn len(&self) -> usize {
    self.mat.len()
  }

  pub fn iter(&self) -> impl Iterator<Item=(&StableCell, &SmolStr)> {
    self.mat.iter().map(|&(ref cel, ref key)| (cel, key))
  }

  pub fn get(&self, cel: &StableCell) -> &SmolStr {
    match self.map.get(cel) {
      None => {
        println!("ERROR:  CellInvertedMatches: missing key for {:?}", cel);
        panic!();
      }
      Some(&idx) => {
        &self.mat[idx].1
      }
    }
  }
}

impl From<ExtCellType> for CellType {
  fn from(ty: ExtCellType) -> CellType {
    CellType{shape: ty.shape.into(), dtype: ty.dtype.into()}
  }
}

impl From<CellType> for ExtCellType {
  fn from(ty: CellType) -> ExtCellType {
    ExtCellType{shape: ty.shape.into(), dtype: ty.dtype.into()}
  }
}

impl From<ExtDtype> for Dtype {
  fn from(ty: ExtDtype) -> Dtype {
    match ty {
      ExtDtype::F64 => Dtype::F64,
      ExtDtype::F32 => Dtype::F32,
      ExtDtype::I64 => Dtype::I64,
      ExtDtype::I32 => Dtype::I32,
      ExtDtype::I16 => Dtype::I16,
      ExtDtype::I8 => Dtype::I8,
      ExtDtype::U64 => Dtype::U64,
      ExtDtype::U32 => Dtype::U32,
      ExtDtype::U16 => Dtype::U16,
      ExtDtype::U8 => Dtype::U8,
      ExtDtype::F16 => Dtype::F16,
      ExtDtype::Bf16 => Dtype::Bf16,
      _ => unimplemented!()
    }
  }
}

impl From<Dtype> for ExtDtype {
  fn from(ty: Dtype) -> ExtDtype {
    match ty {
      Dtype::F64 => ExtDtype::F64,
      Dtype::F32 => ExtDtype::F32,
      Dtype::I64 => ExtDtype::I64,
      Dtype::I32 => ExtDtype::I32,
      Dtype::I16 => ExtDtype::I16,
      Dtype::I8 => ExtDtype::I8,
      Dtype::U64 => ExtDtype::U64,
      Dtype::U32 => ExtDtype::U32,
      Dtype::U16 => ExtDtype::U16,
      Dtype::U8 => ExtDtype::U8,
      Dtype::F16 => ExtDtype::F16,
      Dtype::Bf16 => ExtDtype::Bf16,
      _ => unimplemented!()
    }
  }
}

pub struct CellSplitMmap {
  pub inner: RefCell<CellSplit>,
  pub data_files: Vec<MmapFile>,
}

impl CellSplitMmap {
  pub fn open<P: AsRef<Path>>(paths: &[P]) -> CellSplitMmap {
    let inner = CellSplit::new_(paths);
    let mut data_files = Vec::new();
    for p in inner.paths().iter() {
      let f = Arc::new(File::open(p).unwrap());
      let file = MmapFile::from_file(&f).unwrap();
      data_files.push(file);
    }
    CellSplitMmap{
      inner: RefCell::new(inner),
      data_files,
    }
  }

  pub fn clone_keys(&self) -> BTreeSet<SmolStr> {
    self.inner.borrow_mut().clone_keys()
  }

  pub fn get<K: AsRef<str>>(&self, key: K) -> (CellType, CellMem) {
    let (ty, rep, rank, off, eoff) = self.inner.borrow_mut().get(key);
    assert_eq!(rep, CellRepr::Nd);
    let ty: CellType = ty.into();
    assert!(ty.flat_len() <= (eoff - off) as i64);
    let file = &self.data_files[rank as usize];
    let off: usize = off.try_into().unwrap();
    let eoff: usize = eoff.try_into().unwrap();
    let mmap = file.slice(off .. eoff);
    let mem = CellMem{mmap};
    (ty, mem)
  }
}

#[derive(Clone)]
pub struct CellMem {
  pub mmap: MmapFileSlice,
}

impl CellMem {
  pub fn mmap(&self) -> &MmapFileSlice {
    &self.mmap
  }
}
