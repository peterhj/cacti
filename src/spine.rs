#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct DataPtr(usize);

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct CodePtr(usize);

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct SpinePtr(u32);

impl SpinePtr {
  pub fn zero() -> SpinePtr {
    SpinePtr(0)
  }

  pub fn idx(self) -> usize {
    self.0 as _
  }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SpineEntry {
  _Top,
  Read(SpinePtr),
  Write(SpinePtr),
  Accumulate(SpinePtr),
  SyncGpu(SpinePtr),
  UnsyncGpu(SpinePtr),
  Apply(CodePtr),
  Constant(DataPtr),
  Variable(DataPtr),
}

pub struct Spine {
  //ackp: SpinePtr,
  bp:   SpinePtr,
  curp: SpinePtr,
  //bar:  Vec<SpinePtr>,
  bwd:  Vec<SpinePtr>,
  rst:  Vec<SpinePtr>,
  log:  Vec<SpineEntry>,
  //env:  HashMap<_, _>,
}

impl Spine {
  pub fn constant(&mut self, x: DataPtr) {
    self.log.push(SpineEntry::Constant(x));
    self.curp = SpinePtr(self.log.len() as _);
  }

  pub fn variable(&mut self, x: DataPtr) {
    self.log.push(SpineEntry::Variable(x));
    self.curp = SpinePtr(self.log.len() as _);
  }

  pub fn push_arg(&mut self, p: SpinePtr) {
    self.log.push(SpineEntry::Read(p));
  }

  pub fn apply(&mut self, clo: ()) -> SpinePtr {
    //self.log.push(SpineEntry::Apply(_));
    unimplemented!();
    self.curp = SpinePtr(self.log.len() as _);
  }

  pub fn backward(&mut self) {
    self.bwd.push(self.curp);
    unimplemented!();
  }

  pub fn wait(&mut self) {
    unimplemented!();
    //self.ackp = self.curp;
  }

  pub fn reset(&mut self) {
    self.rst.push(self.curp);
    if self.rst.len() >= 2 {
      let init_rst = if self.rst.len() >= 3 {
        self.rst[self.rst.len() - 3]
      } else {
        SpinePtr(0)
      };
      let prev_rst = self.rst[self.rst.len() - 2];
      let last_rst = self.rst[self.rst.len() - 1];
      if &self.log[init_rst.idx() .. prev_rst.idx()] ==
         &self.log[prev_rst.idx() .. last_rst.idx()]
      {
        // NB: tail compression.
        // FIXME: self.bp.
        // FIXME: self.bwd.
        self.curp = prev_rst;
        self.rst.pop();
        self.log.resize(prev_rst.idx(), SpineEntry::_Top);
      }
    }
    unimplemented!();
  }
}

pub struct SpineIter {
}

impl SpineIter {
}

impl Iterator for SpineIter {
  type Item = ();

  fn next(&mut self) -> Option<()> {
    None
  }
}
