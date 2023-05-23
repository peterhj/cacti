#[cfg(not(target_os = "linux"))]
pub use self::default::Stopwatch;
#[cfg(target_os = "linux")]
pub use self::linux::Stopwatch;

mod default;
mod linux;
