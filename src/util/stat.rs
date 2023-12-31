use crate::algo::fp::{TotalOrd};

#[derive(Debug)]
pub struct StatDigest {
  pub n:    i64,
  pub median: Option<f64>,
  pub mean: Option<f64>,
  //pub std_: f64,
  pub min:  Option<f64>,
  pub max:  Option<f64>,
}

impl StatDigest {
  pub fn from(data: &[f64]) -> StatDigest {
    let n = data.len() as i64;
    let mut sum = 0.0;
    let mut min = None;
    let mut max = None;
    for &x in data.iter() {
      sum += x;
      match min {
        None => {
          min = Some(x);
        }
        Some(ox) => {
          if ox > x {
            min = Some(x);
          }
        }
      }
      match max {
        None => {
          max = Some(x);
        }
        Some(ox) => {
          if ox < x {
            max = Some(x);
          }
        }
      }
    }
    let median = if data.is_empty() {
      None
    } else {
      let mut data2 = data.to_owned();
      data2.sort_by(|&lx, &rx| TotalOrd(lx).cmp(&TotalOrd(rx)));
      // FIXME
      Some(data2[n as usize / 2])
    };
    let mean = if data.is_empty() {
      None
    } else {
      Some(sum / n as f64)
    };
    StatDigest{
      n,
      median,
      mean,
      min,
      max,
    }
  }
}

#[derive(Debug)]
pub struct LazyStatDigest {
  // TODO
  //pub last: Cell<StatDigest>,
  //pub buf:  RefCell<Vec<f64>>,
}
