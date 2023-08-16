use std::convert::{TryFrom};
use std::fmt;

// derived from `parse-size` (MIT licensed):
//
// Copyright 2021 kennytm
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

/// The system to use when parsing prefixes like "KB" and "GB".
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum UnitSystem {
    /// Use powers of 1000 for prefixes. Parsing "1 KB" returns 1000.
    Decimal,
    /// Use powers of 1024 for prefixes. Parsing "1 KB" returns 1024.
    Binary,
}

impl UnitSystem {
    /// Obtains the power factor for the given prefix character.
    ///
    /// Returns None if the input is not a valid prefix.
    ///
    /// The only valid prefixes are K, M, G, T, P and E. The higher powers Z and
    /// Y exceed the `u64` range and thus considered invalid.
    fn factor(self, prefix: u8) -> Option<u64> {
        Some(match (self, prefix) {
            (Self::Decimal, b'k') | (Self::Decimal, b'K') => 1_000,
            (Self::Decimal, b'm') | (Self::Decimal, b'M') => 1_000_000,
            (Self::Decimal, b'g') | (Self::Decimal, b'G') => 1_000_000_000,
            (Self::Decimal, b't') | (Self::Decimal, b'T') => 1_000_000_000_000,
            (Self::Decimal, b'p') | (Self::Decimal, b'P') => 1_000_000_000_000_000,
            (Self::Decimal, b'e') | (Self::Decimal, b'E') => 1_000_000_000_000_000_000,
            (Self::Binary, b'k') | (Self::Binary, b'K') => 1_u64 << 10,
            (Self::Binary, b'm') | (Self::Binary, b'M') => 1_u64 << 20,
            (Self::Binary, b'g') | (Self::Binary, b'G') => 1_u64 << 30,
            (Self::Binary, b't') | (Self::Binary, b'T') => 1_u64 << 40,
            (Self::Binary, b'p') | (Self::Binary, b'P') => 1_u64 << 50,
            (Self::Binary, b'e') | (Self::Binary, b'E') => 1_u64 << 60,
            _ => return None,
        })
    }
}

/// How to deal with the "B" suffix.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum ByteSuffix {
    /// The "B" suffix must never appear. Parsing a string with the "B" suffix
    /// causes [`Error::InvalidDigit`] error.
    Deny,
    /// The "B" suffix is optional.
    Allow,
    /// The "B" suffix is required. Parsing a string without the "B" suffix
    /// causes [`Error::InvalidDigit`] error.
    Require,
}

/// Configuration of the parser.
#[derive(Clone, Debug)]
pub struct Config {
    unit_system: UnitSystem,
    default_factor: u64,
    byte_suffix: ByteSuffix,
}

impl Config {
    /// Creates a new parser configuration.
    pub const fn new() -> Self {
        Self {
            unit_system: UnitSystem::Decimal,
            default_factor: 1,
            byte_suffix: ByteSuffix::Allow,
        }
    }

    /// Changes the configuration's unit system.
    ///
    /// The default system is decimal (powers of 1000).
    pub const fn with_unit_system(mut self, unit_system: UnitSystem) -> Self {
        self.unit_system = unit_system;
        self
    }

    /// Changes the configuration to use the binary unit system, which are
    /// defined to be powers of 1024.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use parse_size::Config;
    ///
    /// let cfg = Config::new().with_binary();
    /// assert_eq!(cfg.parse_size("1 KB"), Ok(1024));
    /// assert_eq!(cfg.parse_size("1 KiB"), Ok(1024));
    /// assert_eq!(cfg.parse_size("1 MB"), Ok(1048576));
    /// assert_eq!(cfg.parse_size("1 MiB"), Ok(1048576));
    /// ```
    pub const fn with_binary(self) -> Self {
        self.with_unit_system(UnitSystem::Binary)
    }

    /// Changes the configuration to use the decimal unit system, which are
    /// defined to be powers of 1000. This is the default setting.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use parse_size::Config;
    ///
    /// let cfg = Config::new().with_decimal();
    /// assert_eq!(cfg.parse_size("1 KB"), Ok(1000));
    /// assert_eq!(cfg.parse_size("1 KiB"), Ok(1024));
    /// assert_eq!(cfg.parse_size("1 MB"), Ok(1000000));
    /// assert_eq!(cfg.parse_size("1 MiB"), Ok(1048576));
    /// ```
    pub const fn with_decimal(self) -> Self {
        self.with_unit_system(UnitSystem::Decimal)
    }

    /// Changes the default factor when a byte unit is not provided.
    ///
    /// This is useful for keeping backward compatibility when migrating from an
    /// old user interface expecting non-byte input.
    ///
    /// The default value is 1.
    ///
    /// # Examples
    ///
    /// If the input is a pure number, we treat that as mebibytes.
    ///
    /// ```rust
    /// use parse_size::Config;
    ///
    /// let cfg = Config::new().with_default_factor(1048576);
    /// assert_eq!(cfg.parse_size("10"), Ok(10485760));
    /// assert_eq!(cfg.parse_size("0.5"), Ok(524288));
    /// assert_eq!(cfg.parse_size("128 B"), Ok(128)); // explicit units overrides the default
    /// assert_eq!(cfg.parse_size("16 KiB"), Ok(16384));
    /// ```
    pub const fn with_default_factor(mut self, factor: u64) -> Self {
        self.default_factor = factor;
        self
    }

    /// Changes the handling of the "B" suffix.
    ///
    /// Normally, the character "B" at the end of the input is optional. This
    /// can be changed to deny or require such suffix.
    ///
    /// Power prefixes (K, Ki, M, Mi, ...) are not affected.
    ///
    /// # Examples
    ///
    /// Deny the suffix.
    ///
    /// ```rust
    /// use parse_size::{ByteSuffix, Config, Error};
    ///
    /// let cfg = Config::new().with_byte_suffix(ByteSuffix::Deny);
    /// assert_eq!(cfg.parse_size("123"), Ok(123));
    /// assert_eq!(cfg.parse_size("123k"), Ok(123000));
    /// assert_eq!(cfg.parse_size("123B"), Err(Error::InvalidDigit));
    /// assert_eq!(cfg.parse_size("123KB"), Err(Error::InvalidDigit));
    /// ```
    ///
    /// Require the suffix.
    ///
    /// ```rust
    /// use parse_size::{ByteSuffix, Config, Error};
    ///
    /// let cfg = Config::new().with_byte_suffix(ByteSuffix::Require);
    /// assert_eq!(cfg.parse_size("123"), Err(Error::InvalidDigit));
    /// assert_eq!(cfg.parse_size("123k"), Err(Error::InvalidDigit));
    /// assert_eq!(cfg.parse_size("123B"), Ok(123));
    /// assert_eq!(cfg.parse_size("123KB"), Ok(123000));
    /// ```
    pub const fn with_byte_suffix(mut self, byte_suffix: ByteSuffix) -> Self {
        self.byte_suffix = byte_suffix;
        self
    }

    /// Parses the string input into the number of bytes it represents.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use parse_size::{Config, Error};
    ///
    /// let cfg = Config::new().with_binary();
    /// assert_eq!(cfg.parse_size("10 KB"), Ok(10240));
    /// assert_eq!(cfg.parse_size("20000"), Ok(20000));
    /// assert_eq!(cfg.parse_size("^_^"), Err(Error::InvalidDigit));
    /// ```
    pub fn parse_byte_size<T: AsRef<[u8]>>(&self, src: T) -> Result<u64, Error> {
        parse_size_inner(self, src.as_ref())
    }
}

impl Default for Config {
    fn default() -> Self {
        Self::new()
    }
}

// TODO: Switch to IntErrorKind once it is stable.
/// The error returned when parse failed.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[non_exhaustive]
pub enum Error {
    /// The input contains no numbers.
    Empty,
    /// An invalid character is encountered while parsing.
    InvalidDigit,
    /// The resulting number is too large to fit into a `u64`.
    PosOverflow,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Empty => "cannot parse integer from empty string",
            Self::InvalidDigit => "invalid digit found in string",
            Self::PosOverflow => "number too large to fit in target type",
        })
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}

/// Parses the string input into the number of bytes it represents using the
/// default configuration.
///
/// Equivalent to calling [`Config::parse_size()`] with the default
/// configuration ([`Config::new()`]).
///
/// # Examples
///
/// ```rust
/// use parse_size::{parse_size, Error};
///
/// assert_eq!(parse_size("10 KB"), Ok(10000));
/// assert_eq!(parse_size("20000"), Ok(20000));
/// assert_eq!(parse_size("0.2 MiB"), Ok(209715));
/// assert_eq!(parse_size("^_^"), Err(Error::InvalidDigit));
/// ```
pub fn parse_byte_size<T: AsRef<[u8]>>(src: T) -> Result<u64, Error> {
    parse_size_inner(&Config::new(), src.as_ref())
}

fn parse_size_inner(cfg: &Config, mut src: &[u8]) -> Result<u64, Error> {
    // if it ends with 'B' the default factor is always 1.
    let mut multiply = cfg.default_factor;
    match src {
        [init @ .., b'b'] | [init @ .., b'B'] => {
            if cfg.byte_suffix == ByteSuffix::Deny {
                return Err(Error::InvalidDigit);
            }
            src = init;
            multiply = 1;
        }
        _ => {
            if cfg.byte_suffix == ByteSuffix::Require {
                return Err(Error::InvalidDigit);
            }
        }
    }

    // if it ends with an 'i' we always use binary prefix.
    let mut unit_system = cfg.unit_system;
    match src {
        [init @ .., b'i'] | [init @ .., b'I'] => {
            src = init;
            unit_system = UnitSystem::Binary;
        }
        _ => {}
    }

    if let [init @ .., prefix] = src {
        if let Some(f) = unit_system.factor(*prefix) {
            multiply = f;
            src = init;
        }
    }

    #[derive(Copy, Clone, PartialEq)]
    enum Ps {
        Empty,
        Integer,
        IntegerOverflow,
        Fraction,
        FractionOverflow,
        PosExponent,
        NegExponent,
    }

    macro_rules! append_digit {
        ($before:expr, $method:ident, $digit_char:expr) => {
            $before
                .checked_mul(10)
                .and_then(|v| v.$method(($digit_char - b'0').into()))
        };
    }

    let mut mantissa = 0_u64;
    let mut fractional_exponent = 0;
    let mut exponent = 0_i32;
    let mut state = Ps::Empty;

    for b in src {
        match (state, *b) {
            (Ps::Integer, b'0'..=b'9') | (Ps::Empty, b'0'..=b'9') => {
                if let Some(m) = append_digit!(mantissa, checked_add, *b) {
                    mantissa = m;
                    state = Ps::Integer;
                } else {
                    if *b >= b'5' {
                        mantissa += 1;
                    }
                    state = Ps::IntegerOverflow;
                    fractional_exponent += 1;
                }
            }
            (Ps::IntegerOverflow, b'0'..=b'9') => {
                fractional_exponent += 1;
            }
            (Ps::Fraction, b'0'..=b'9') => {
                if let Some(m) = append_digit!(mantissa, checked_add, *b) {
                    mantissa = m;
                    fractional_exponent -= 1;
                } else {
                    if *b >= b'5' {
                        mantissa += 1;
                    }
                    state = Ps::FractionOverflow;
                }
            }
            (Ps::FractionOverflow, b'0'..=b'9') => {}
            (Ps::PosExponent, b'0'..=b'9') => {
                if let Some(e) = append_digit!(exponent, checked_add, *b) {
                    exponent = e;
                } else {
                    return Err(Error::PosOverflow);
                }
            }
            (Ps::NegExponent, b'0'..=b'9') => {
                if let Some(e) = append_digit!(exponent, checked_sub, *b) {
                    exponent = e;
                }
            }

            (_, b'_') | (_, b' ') | (Ps::PosExponent, b'+') => {}
            (Ps::Integer, b'e')
            | (Ps::Integer, b'E')
            | (Ps::Fraction, b'e')
            | (Ps::Fraction, b'E')
            | (Ps::IntegerOverflow, b'e')
            | (Ps::IntegerOverflow, b'E')
            | (Ps::FractionOverflow, b'e')
            | (Ps::FractionOverflow, b'E') => state = Ps::PosExponent,
            (Ps::PosExponent, b'-') => state = Ps::NegExponent,
            (Ps::Integer, b'.') => state = Ps::Fraction,
            (Ps::IntegerOverflow, b'.') => state = Ps::FractionOverflow,
            _ => return Err(Error::InvalidDigit),
        }
    }

    if state == Ps::Empty {
        return Err(Error::Empty);
    }

    let exponent = exponent.saturating_add(fractional_exponent);
    if exponent >= 0 {
        let power = 10_u64
            .checked_pow(exponent as u32)
            .ok_or(Error::PosOverflow)?;
        let multiply = multiply.checked_mul(power).ok_or(Error::PosOverflow)?;
        mantissa.checked_mul(multiply).ok_or(Error::PosOverflow)
    } else if exponent >= -38 {
        let power = 10_u128.pow(-exponent as u32);
        let result = (u128::from(mantissa) * u128::from(multiply) + power / 2) / power;
        u64::try_from(result).map_err(|_| Error::PosOverflow)
    } else {
        // (2^128) * 1e-39 < 1, always, and thus saturate to 0.
        Ok(0)
    }
}
