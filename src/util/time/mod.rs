#[cfg(not(target_os = "linux"))]
pub use self::default::{Stopwatch, Timestamp};
#[cfg(target_os = "linux")]
pub use self::linux::{Stopwatch, Timestamp};

use std::cell::{RefCell};

thread_local! {
  pub static TL_WATCH: RefCell<Stopwatch> = RefCell::new(Stopwatch::new());
}

#[cfg(not(target_os = "linux"))]
mod default;
#[cfg(target_os = "linux")]
mod linux;

impl Stopwatch {
  pub fn tl_clone() -> Stopwatch {
    TL_WATCH.with(|watch| {
      *watch.borrow()
    })
  }

  pub fn tl_stamp() -> Timestamp {
    TL_WATCH.with(|watch| {
      watch.borrow_mut().stamp()
    })
  }

  pub fn tl_lap() -> f64 {
    TL_WATCH.with(|watch| {
      watch.borrow_mut().lap()
    })
  }
}
