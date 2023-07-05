use crate::algo::{HashMap, HashSet};
use crate::cell::{StableCell};

use smol_str::{SmolStr};

//use std::borrow::{Borrow};

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
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
  pub rule: Vec<(Vec<Pat>, StableCell)>,
  pub map:  HashMap<Vec<Pat>, usize>,
}

impl CellMatcher {
  pub fn new() -> CellMatcher {
    CellMatcher{
      rule: Vec::new(),
      map:  HashMap::new(),
    }
  }

  pub fn len(&self) -> usize {
    self.rule.len()
  }

  pub fn match_(&self, mut keys: HashSet<SmolStr>) -> CellMatches {
    let mut mat = Vec::new();
    let mut map = HashMap::new();
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
    self.map.insert(pat.clone(), self.rule.len());
    self.rule.push((pat, cel))
  }
}

impl<K0: Into<Pat>> CellMatcherExt<(K0,)> for CellMatcher {
  fn insert(&mut self, key: (K0,), cel: StableCell) {
    let mut pat = Vec::with_capacity(1);
    pat.push(key.0.into());
    self.map.insert(pat.clone(), self.rule.len());
    self.rule.push((pat, cel))
  }
}

impl<K0: Into<Pat>, K1: Into<Pat>> CellMatcherExt<(K0, K1)> for CellMatcher {
  fn insert(&mut self, key: (K0, K1), cel: StableCell) {
    let mut pat = Vec::with_capacity(2);
    pat.push(key.0.into());
    pat.push(key.1.into());
    self.map.insert(pat.clone(), self.rule.len());
    self.rule.push((pat, cel))
  }
}

impl<K0: Into<Pat>, K1: Into<Pat>, K2: Into<Pat>> CellMatcherExt<(K0, K1, K2)> for CellMatcher {
  fn insert(&mut self, key: (K0, K1, K2), cel: StableCell) {
    let mut pat = Vec::with_capacity(3);
    pat.push(key.0.into());
    pat.push(key.1.into());
    pat.push(key.2.into());
    self.map.insert(pat.clone(), self.rule.len());
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
    self.map.insert(pat.clone(), self.rule.len());
    self.rule.push((pat, cel))
  }
}

pub struct CellMatches {
  pub mat:  Vec<(SmolStr, StableCell)>,
  pub map:  HashMap<SmolStr, usize>,
}

impl CellMatches {
  pub fn len(&self) -> usize {
    self.mat.len()
  }

  pub fn iter(&self) -> impl Iterator<Item=(&SmolStr, &StableCell)> {
    self.mat.iter().map(|&(ref key, ref cel)| (key, cel))
  }

  pub fn inv(&self) -> CellInvertedMatches {
    let mut mat = Vec::new();
    let mut map = HashMap::new();
    for &(ref key, ref cel) in self.mat.iter() {
      let cel = cel.clone();
      let key = key.clone();
      map.insert(cel.clone(), mat.len());
      mat.push((cel, key));
    }
    CellInvertedMatches{mat, map}
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
      None => panic!("bug"),
      Some(&idx) => {
        &self.mat[idx].1
      }
    }
  }
}
