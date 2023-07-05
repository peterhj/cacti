pub fn sane_ascii(s: &[u8]) -> String {
  let mut buf = Vec::new();
  for &u in s.iter() {
    if u == b' ' || u == b'.' || u == b':' || u == b'/' || u == b'-' || u == b'_' {
      buf.push(u);
    } else if u >= b'0' && u <= b'9' {
      buf.push(u);
    } else if u >= b'A' && u <= b'Z' {
      buf.push(u);
    } else if u >= b'a' && u <= b'z' {
      buf.push(u);
    } else {
      buf.push(b'?');
    }
  }
  String::from_utf8_lossy(&buf).into()
}