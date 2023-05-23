//extern crate cacti;
extern crate dessert;

use dessert::data::mnist::*;

fn main() {
  let train_label = UbyteIdx1File::open("/home/i/data/mnist/train-labels-idx1-ubyte").unwrap();
  let train_image = UbyteIdx3File::open("/home/i/data/mnist/train-images-idx3-ubyte").unwrap();
  let test_label = UbyteIdx1File::open("/home/i/data/mnist/t10k-labels-idx1-ubyte").unwrap();
  let test_image = UbyteIdx3File::open("/home/i/data/mnist/t10k-images-idx3-ubyte").unwrap();
  println!("mnist: train label n={}", train_label.count());
  println!("mnist: train image n={} shape={:?}", train_image.count(), train_image.item_shape());
  println!("mnist: test label n={}", test_label.count());
  println!("mnist: test image n={} shape={:?}", test_image.count(), test_image.item_shape());
}
