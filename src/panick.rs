use cacti_cfg_env::*;

use once_cell::sync::{Lazy};

use std::cell::{RefCell};
use std::panic::{Location, set_hook, take_hook};

pub static PANICK_ONCE_CTX: Lazy<PanickOnceCtx> = Lazy::new(|| PanickOnceCtx::new());
thread_local! {
  pub static PANICK_TL_CTX: PanickCtx = PanickCtx::new();
}

pub struct PanickOnceCtx {
  pub init: bool,
}

impl PanickOnceCtx {
  pub fn new() -> PanickOnceCtx {
    if cfg_debug() { println!("DEBUG: PanickOnceCtx::new: init..."); }
    let og_hook = take_hook();
    set_hook(Box::new(move |info| {
      PANICK_TL_CTX.with(|ctx| {
        let stack = ctx.stack.borrow();
        // TODO TODO: full backtrace?
        /*match stack.first() {
          None => {}
          Some(frame) => {
            let loc = frame.loc;
            eprintln!("panick: {}:{}:{}", loc.file(), loc.line(), loc.column());
          }
        }*/
        for frame in stack.iter().rev() {
          let loc = frame.loc;
          eprintln!("panick: {}:{}:{}", loc.file(), loc.line(), loc.column());
        }
      });
      og_hook(info)
    }));
    if cfg_debug() { println!("DEBUG: PanickOnceCtx::new: init... done!"); }
    PanickOnceCtx{
      init: true,
    }
  }
}

#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct PanickFrame {
  pub loc:  &'static Location<'static>,
  //pub bits: u8,
}

/*impl PanickFrame {
  pub fn top_level(&self) -> bool {
    self.bits & 1 != 0
  }
}*/

pub struct PanickCtx {
  pub stack:    RefCell<Vec<PanickFrame>>,
  pub onceinit: bool,
}

impl PanickCtx {
  pub fn new() -> PanickCtx {
    if cfg_debug() { println!("DEBUG: PanickCtx::new: init..."); }
    let onceinit = PANICK_ONCE_CTX.init;
    assert!(onceinit);
    if cfg_debug() { println!("DEBUG: PanickCtx::new: init... done!"); }
    PanickCtx{
      stack:    RefCell::new(Vec::new()),
      onceinit,
    }
  }
}

#[track_caller]
pub fn panick_wrap<F: FnOnce() -> T, T>(f: F) -> T {
  let loc = Location::caller();
  PANICK_TL_CTX.with(|panick| {
    let frame = PanickFrame{loc};
    let mut stack = panick.stack.borrow_mut();
    stack.push(frame);
  });
  let t = (f)();
  PANICK_TL_CTX.with(|panick| {
    let mut stack = panick.stack.borrow_mut();
    stack.pop()
  });
  t
}
