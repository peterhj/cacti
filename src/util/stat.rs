#[derive(Debug)]
pub struct StatDigest {
  pub n:    i64,
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
    let mean = if data.is_empty() {
      None
    } else {
      Some(sum / n as f64)
    };
    StatDigest{
      n,
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
