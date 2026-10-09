//! The arithmetic tool's evaluator: a hand-written parser and nothing else.
//!
//! The `calculate` tool must never be a route into code execution. Rust has no
//! `eval`, and this module keeps it that way: text is tokenised, parsed by
//! recursive descent and evaluated as it is parsed, and the only things that can
//! ever be evaluated are numbers, `+ - * / // % **`, unary `+ -` and
//! parentheses. Anything else is refused, whole.
//!
//! Semantics are Python's, so a model that knows Python is not surprised
//! (`tests/parity/registry/calc.json` is recorded from Python's own arithmetic
//! by `tools/lattice_native_parity.py`):
//! - precedence and associativity: `**` is right-associative and binds tighter
//!   than a unary minus on its left (`-2 ** 2` is `-4`) but not than one on its
//!   right (`2 ** -1` is `0.5`);
//! - an integer stays an integer under `+ - * // % **` (with a non-negative
//!   exponent), in `i128`; `/` always gives a float, correctly rounded even for
//!   integers beyond 2^53; mixed operands are floats;
//! - `//` and `%` floor (the remainder takes the sign of the divisor), for
//!   integers and for floats, by CPython's own algorithm;
//! - a float is printed as Python prints it (`repr`): `0.1 + 0.2` is
//!   `0.30000000000000004`, `1e16` is `1e+16`, `5 / 2` is `2.5`, `4 / 2` is `2.0`.
//!
//! Limits, each a refusal with a sentence the model can act on: at most 200
//! characters; nesting (parentheses, unary signs and `**` chains together) at
//! most 64 deep; an exponent between -64 and 64 and a base of at most 1e6 in
//! size; an integer that overflows `i128`; any value beyond 1e100 in size;
//! division by zero; and a negative base raised to a fractional power (Python
//! answers with a complex number; there is no such thing here).
//!
//! Deliberately more permissive than Python: an integer may have leading zeros
//! (`007` is `7`; Python refuses it), and digits with underscores are not numbers.
//!
//! Invariant: total. Any input, however malformed or long, gives a value or a
//! [`CalcError`]; nothing panics and nothing recurses beyond the depth limit.

use std::fmt;

pub const MAX_EXPRESSION_CHARS: usize = 200;
pub const MAX_DEPTH: usize = 64;
pub const MAX_EXPONENT: f64 = 64.0;
pub const MAX_BASE: f64 = 1e6;
pub const MAX_MAGNITUDE: f64 = 1e100;

/// A number: an integer or a float, as Python distinguishes them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Num {
    Int(i128),
    Float(f64),
}

impl Num {
    fn float(self) -> f64 {
        match self {
            Num::Int(i) => i as f64,
            Num::Float(f) => f,
        }
    }
}

impl fmt::Display for Num {
    /// Python's `repr`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Num::Int(i) => write!(f, "{i}"),
            Num::Float(x) => f.write_str(&float_repr(*x)),
        }
    }
}

/// Why an expression was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CalcError {
    /// A character or word that is not arithmetic.
    Disallowed,
    /// Made only of allowed pieces, but not a whole expression.
    Incomplete,
    TooLong,
    TooDeep,
    NumberTooLarge,
    ResultTooLarge,
    DivisionByZero,
    ExponentRange,
    NotReal,
}

impl fmt::Display for CalcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            CalcError::Disallowed => "Only numbers, + - * / // % ** and parentheses are allowed.",
            CalcError::Incomplete => "That is not a complete arithmetic expression.",
            CalcError::TooLong => "The expression is longer than 200 characters.",
            CalcError::TooDeep => "The expression is nested more than 64 levels deep.",
            CalcError::NumberTooLarge => "A number in the expression is too large.",
            CalcError::ResultTooLarge => "The result is too large.",
            CalcError::DivisionByZero => "division by zero",
            CalcError::ExponentRange => {
                "An exponent must be between -64 and 64, and its base at most 1000000 in size."
            }
            CalcError::NotReal => "The result is not a real number.",
        })
    }
}

impl std::error::Error for CalcError {}

/// Evaluate `expression` and print the value the way Python would.
pub fn calculate(expression: &str) -> Result<String, CalcError> {
    evaluate(expression).map(|value| value.to_string())
}

/// Evaluate `expression`.
pub fn evaluate(expression: &str) -> Result<Num, CalcError> {
    if expression.chars().count() > MAX_EXPRESSION_CHARS {
        return Err(CalcError::TooLong);
    }
    let tokens = tokenize(expression)?;
    let mut parser = Parser {
        tokens,
        position: 0,
    };
    let value = parser.expression(0)?;
    if parser.position != parser.tokens.len() {
        return Err(CalcError::Incomplete);
    }
    Ok(value)
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Token {
    Number(Num),
    Plus,
    Minus,
    Star,
    Slash,
    DoubleSlash,
    Percent,
    Power,
    Open,
    Close,
}

fn tokenize(text: &str) -> Result<Vec<Token>, CalcError> {
    let chars: Vec<char> = text.chars().collect();
    let mut tokens = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        match c {
            ' ' | '\t' | '\r' | '\n' => i += 1,
            '+' => {
                tokens.push(Token::Plus);
                i += 1;
            }
            '-' => {
                tokens.push(Token::Minus);
                i += 1;
            }
            '%' => {
                tokens.push(Token::Percent);
                i += 1;
            }
            '(' => {
                tokens.push(Token::Open);
                i += 1;
            }
            ')' => {
                tokens.push(Token::Close);
                i += 1;
            }
            '*' if chars.get(i + 1) == Some(&'*') => {
                tokens.push(Token::Power);
                i += 2;
            }
            '*' => {
                tokens.push(Token::Star);
                i += 1;
            }
            '/' if chars.get(i + 1) == Some(&'/') => {
                tokens.push(Token::DoubleSlash);
                i += 2;
            }
            '/' => {
                tokens.push(Token::Slash);
                i += 1;
            }
            '0'..='9' | '.' => {
                let (number, next) = number(&chars, i)?;
                tokens.push(Token::Number(number));
                i = next;
            }
            _ => return Err(CalcError::Disallowed),
        }
    }
    Ok(tokens)
}

/// One numeric literal starting at `start`: `digits`, `digits.digits`, `.digits`,
/// `digits.` and any of those with an exponent.
fn number(chars: &[char], start: usize) -> Result<(Num, usize), CalcError> {
    let digits = |from: usize| {
        chars[from..]
            .iter()
            .take_while(|c| c.is_ascii_digit())
            .count()
    };
    let mut i = start;
    let whole = digits(i);
    i += whole;
    let mut is_float = false;
    let mut fraction = 0;
    if chars.get(i) == Some(&'.') {
        is_float = true;
        i += 1;
        fraction = digits(i);
        i += fraction;
    }
    if whole == 0 && fraction == 0 {
        return Err(CalcError::Disallowed);
    }
    if matches!(chars.get(i), Some('e' | 'E')) {
        let mut j = i + 1;
        if matches!(chars.get(j), Some('+' | '-')) {
            j += 1;
        }
        let exponent = digits(j);
        if exponent == 0 {
            return Err(CalcError::Disallowed);
        }
        is_float = true;
        i = j + exponent;
    }
    let literal: String = chars[start..i].iter().collect();
    if is_float {
        let value: f64 = literal.parse().map_err(|_| CalcError::Disallowed)?;
        if !value.is_finite() || value.abs() > MAX_MAGNITUDE {
            return Err(CalcError::NumberTooLarge);
        }
        Ok((Num::Float(value), i))
    } else {
        let value: i128 = literal.parse().map_err(|_| CalcError::NumberTooLarge)?;
        Ok((Num::Int(value), i))
    }
}

struct Parser {
    tokens: Vec<Token>,
    position: usize,
}

impl Parser {
    fn peek(&self) -> Option<Token> {
        self.tokens.get(self.position).copied()
    }

    fn bump(&mut self) -> Option<Token> {
        let token = self.peek();
        self.position += 1;
        token
    }

    /// One level deeper, or the refusal for going past the limit.
    fn deeper(depth: usize) -> Result<usize, CalcError> {
        if depth >= MAX_DEPTH {
            Err(CalcError::TooDeep)
        } else {
            Ok(depth + 1)
        }
    }

    /// `expression := term (('+' | '-') term)*`
    fn expression(&mut self, depth: usize) -> Result<Num, CalcError> {
        let mut left = self.term(depth)?;
        while let Some(op @ (Token::Plus | Token::Minus)) = self.peek() {
            self.position += 1;
            let right = self.term(depth)?;
            left = if op == Token::Plus {
                add(left, right)?
            } else {
                subtract(left, right)?
            };
        }
        Ok(left)
    }

    /// `term := factor (('*' | '/' | '//' | '%') factor)*`
    fn term(&mut self, depth: usize) -> Result<Num, CalcError> {
        let mut left = self.factor(depth)?;
        while let Some(op @ (Token::Star | Token::Slash | Token::DoubleSlash | Token::Percent)) =
            self.peek()
        {
            self.position += 1;
            let right = self.factor(depth)?;
            left = match op {
                Token::Star => multiply(left, right)?,
                Token::Slash => divide(left, right)?,
                Token::DoubleSlash => floor_divide(left, right)?,
                _ => remainder(left, right)?,
            };
        }
        Ok(left)
    }

    /// `factor := ('+' | '-') factor | power`
    fn factor(&mut self, depth: usize) -> Result<Num, CalcError> {
        match self.peek() {
            Some(Token::Plus) => {
                self.position += 1;
                self.factor(Self::deeper(depth)?)
            }
            Some(Token::Minus) => {
                self.position += 1;
                negate(self.factor(Self::deeper(depth)?)?)
            }
            _ => self.power(depth),
        }
    }

    /// `power := primary ('**' factor)?`
    fn power(&mut self, depth: usize) -> Result<Num, CalcError> {
        let base = self.primary(depth)?;
        if self.peek() == Some(Token::Power) {
            self.position += 1;
            let exponent = self.factor(Self::deeper(depth)?)?;
            return power(base, exponent);
        }
        Ok(base)
    }

    /// `primary := number | '(' expression ')'`
    fn primary(&mut self, depth: usize) -> Result<Num, CalcError> {
        match self.bump() {
            Some(Token::Number(number)) => Ok(number),
            Some(Token::Open) => {
                let value = self.expression(Self::deeper(depth)?)?;
                match self.bump() {
                    Some(Token::Close) => Ok(value),
                    _ => Err(CalcError::Incomplete),
                }
            }
            _ => Err(CalcError::Incomplete),
        }
    }
}

/// A float result inside the limits, or the refusal.
fn checked(value: f64) -> Result<Num, CalcError> {
    if value.is_nan() {
        return Err(CalcError::NotReal);
    }
    if value.abs() > MAX_MAGNITUDE {
        return Err(CalcError::ResultTooLarge);
    }
    Ok(Num::Float(value))
}

fn negate(value: Num) -> Result<Num, CalcError> {
    match value {
        Num::Int(i) => i
            .checked_neg()
            .map(Num::Int)
            .ok_or(CalcError::ResultTooLarge),
        Num::Float(f) => Ok(Num::Float(-f)),
    }
}

fn add(a: Num, b: Num) -> Result<Num, CalcError> {
    match (a, b) {
        (Num::Int(x), Num::Int(y)) => x
            .checked_add(y)
            .map(Num::Int)
            .ok_or(CalcError::ResultTooLarge),
        _ => checked(a.float() + b.float()),
    }
}

fn subtract(a: Num, b: Num) -> Result<Num, CalcError> {
    match (a, b) {
        (Num::Int(x), Num::Int(y)) => x
            .checked_sub(y)
            .map(Num::Int)
            .ok_or(CalcError::ResultTooLarge),
        _ => checked(a.float() - b.float()),
    }
}

fn multiply(a: Num, b: Num) -> Result<Num, CalcError> {
    match (a, b) {
        (Num::Int(x), Num::Int(y)) => x
            .checked_mul(y)
            .map(Num::Int)
            .ok_or(CalcError::ResultTooLarge),
        _ => checked(a.float() * b.float()),
    }
}

/// `/`: always a float; an integer over an integer is correctly rounded.
fn divide(a: Num, b: Num) -> Result<Num, CalcError> {
    match (a, b) {
        (Num::Int(_), Num::Int(0)) => Err(CalcError::DivisionByZero),
        (Num::Int(x), Num::Int(y)) => checked(int_true_divide(x, y)),
        _ => {
            let divisor = b.float();
            if divisor == 0.0 {
                return Err(CalcError::DivisionByZero);
            }
            checked(a.float() / divisor)
        }
    }
}

/// `//`: floor division.
fn floor_divide(a: Num, b: Num) -> Result<Num, CalcError> {
    match (a, b) {
        (Num::Int(_), Num::Int(0)) => Err(CalcError::DivisionByZero),
        (Num::Int(x), Num::Int(y)) => {
            let quotient = x.checked_div(y).ok_or(CalcError::ResultTooLarge)?;
            let rest = x % y;
            let floored = if rest != 0 && ((rest < 0) != (y < 0)) {
                quotient - 1
            } else {
                quotient
            };
            Ok(Num::Int(floored))
        }
        _ => {
            let divisor = b.float();
            if divisor == 0.0 {
                return Err(CalcError::DivisionByZero);
            }
            checked(float_divmod(a.float(), divisor).0)
        }
    }
}

/// `%`: the remainder takes the sign of the divisor.
fn remainder(a: Num, b: Num) -> Result<Num, CalcError> {
    match (a, b) {
        (Num::Int(_), Num::Int(0)) => Err(CalcError::DivisionByZero),
        (Num::Int(_), Num::Int(-1)) => Ok(Num::Int(0)),
        (Num::Int(x), Num::Int(y)) => {
            let rest = x % y;
            Ok(Num::Int(if rest != 0 && ((rest < 0) != (y < 0)) {
                rest + y
            } else {
                rest
            }))
        }
        _ => {
            let divisor = b.float();
            if divisor == 0.0 {
                return Err(CalcError::DivisionByZero);
            }
            checked(float_divmod(a.float(), divisor).1)
        }
    }
}

/// `**`, within the limits.
fn power(a: Num, b: Num) -> Result<Num, CalcError> {
    let (base, exponent) = (a.float(), b.float());
    if exponent.abs() > MAX_EXPONENT || base.abs() > MAX_BASE {
        return Err(CalcError::ExponentRange);
    }
    if let (Num::Int(x), Num::Int(y)) = (a, b)
        && y >= 0
    {
        return x
            .checked_pow(y as u32)
            .map(Num::Int)
            .ok_or(CalcError::ResultTooLarge);
    }
    if base == 0.0 && exponent < 0.0 {
        return Err(CalcError::DivisionByZero);
    }
    if base < 0.0 && exponent.fract() != 0.0 {
        return Err(CalcError::NotReal);
    }
    checked(base.powf(exponent))
}

/// CPython's `float_divmod`: `(floor division, modulo)` of two floats.
fn float_divmod(x: f64, y: f64) -> (f64, f64) {
    let mut modulus = x % y;
    let mut div = (x - modulus) / y;
    if modulus != 0.0 {
        if (y < 0.0) != (modulus < 0.0) {
            modulus += y;
            div -= 1.0;
        }
    } else {
        modulus = 0.0f64.copysign(y);
    }
    let floored = if div != 0.0 {
        let floor = div.floor();
        if div - floor > 0.5 {
            floor + 1.0
        } else {
            floor
        }
    } else {
        0.0f64.copysign(x / y)
    };
    (floored, modulus)
}

/// 2^`exponent` for an exponent inside the normal range of a double, exactly.
fn power_of_two(exponent: i32) -> f64 {
    debug_assert!((-1022..=1023).contains(&exponent));
    f64::from_bits(((exponent + 1023) as u64) << 52)
}

/// `a / b` for two integers, correctly rounded (Python's `int.__truediv__`),
/// where converting each to a float first would round twice.
fn int_true_divide(a: i128, b: i128) -> f64 {
    let negative = (a < 0) != (b < 0);
    let magnitude = ratio(a.unsigned_abs(), b.unsigned_abs());
    if negative { -magnitude } else { magnitude }
}

/// The nearest double (ties to even) to `a / b`, for `b != 0`, by long division.
fn ratio(a: u128, b: u128) -> f64 {
    const TWO53: u128 = 1 << 53;
    if a == 0 {
        return 0.0;
    }
    if a < TWO53 && b < TWO53 {
        // Both operands are exact as doubles, so one division rounds once.
        return a as f64 / b as f64;
    }
    let quotient = a / b;
    let mut rest = a % b;
    if quotient >= 1 << 53 {
        // More integer bits than a double keeps: round the quotient itself, with
        // the remainder as the sticky bit.
        let bits = 128 - quotient.leading_zeros() as i32;
        let shift = bits - 53;
        let mut top = quotient >> shift;
        let dropped = quotient & ((1u128 << shift) - 1);
        let half = 1u128 << (shift - 1);
        if dropped > half || (dropped == half && (rest != 0 || top & 1 == 1)) {
            top += 1;
        }
        return top as f64 * power_of_two(shift);
    }
    // Fewer: bring down bits of the fraction until there are 54 significant
    // bits (53 and a guard bit); `rest` stays below `b`, so `rest << 1` fits.
    let (mut significand, mut scale) = (quotient, 0i32);
    if quotient == 0 {
        loop {
            rest <<= 1;
            scale += 1;
            if rest >= b {
                rest -= b;
                break;
            }
        }
        significand = 1;
    }
    while 128 - significand.leading_zeros() < 54 {
        rest <<= 1;
        significand <<= 1;
        scale += 1;
        if rest >= b {
            rest -= b;
            significand |= 1;
        }
    }
    let mut top = significand >> 1;
    if significand & 1 == 1 && (rest != 0 || top & 1 == 1) {
        top += 1;
    }
    top as f64 * power_of_two(1 - scale)
}

/// Python's `repr(float)`: the shortest text that reads back as the same
/// double, in fixed notation from `1e-4` up to (not including) `1e16` and in
/// scientific notation (`1e+16`, `1.5e-05`) outside it, with `.0` on a whole
/// number in fixed notation.
pub fn float_repr(value: f64) -> String {
    if value.is_nan() {
        return "nan".to_owned();
    }
    if value.is_infinite() {
        return if value < 0.0 { "-inf" } else { "inf" }.to_owned();
    }
    let sign = if value.is_sign_negative() { "-" } else { "" };
    // `{:e}` gives the shortest round-trip digits: `d[.ddd]e<exp>`.
    let scientific = format!("{:e}", value.abs());
    let (mantissa, exp) = scientific.split_once('e').unwrap_or((&scientific, "0"));
    let exp10: i32 = exp.parse().unwrap_or(0);
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    // Python's `decpt`: the decimal point sits after `decpt` digits.
    let decpt = exp10 + 1;
    let body = if decpt <= -4 || decpt > 16 {
        let mut text = String::new();
        text.push_str(&digits[..1]);
        if digits.len() > 1 {
            text.push('.');
            text.push_str(&digits[1..]);
        }
        text.push('e');
        text.push(if exp10 < 0 { '-' } else { '+' });
        text.push_str(&format!("{:02}", exp10.abs()));
        text
    } else if decpt <= 0 {
        format!("0.{}{digits}", "0".repeat((-decpt) as usize))
    } else if decpt as usize >= digits.len() {
        format!("{digits}{}.0", "0".repeat(decpt as usize - digits.len()))
    } else {
        format!(
            "{}.{}",
            &digits[..decpt as usize],
            &digits[decpt as usize..]
        )
    };
    format!("{sign}{body}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn calc(expression: &str) -> String {
        calculate(expression).unwrap_or_else(|e| format!("error: {e}"))
    }

    // Every expected value below was checked against CPython 3.12 (the ones that
    // are refusals here are Python answers the limits turn away).

    #[test]
    fn basic_arithmetic_keeps_integers_integers() {
        for (expression, expected) in [
            ("6 * 7", "42"),
            ("1 + 2 * 3", "7"),
            ("(1 + 2) * 3", "9"),
            ("10 - 4 - 3", "3"),
            ("2 ** 10", "1024"),
            ("7 // 2", "3"),
            ("7 % 3", "1"),
            ("  42  ", "42"),
            ("007", "7"),
            ("+5", "5"),
            ("--5", "5"),
            ("-(-3)", "3"),
            ("0", "0"),
            ("2 ** 0", "1"),
            ("0 ** 0", "1"),
            ("10 ** 30", "1000000000000000000000000000000"),
        ] {
            assert_eq!(calc(expression), expected, "{expression}");
        }
    }

    #[test]
    fn division_is_always_a_float_and_prints_like_python() {
        for (expression, expected) in [
            ("5 / 2", "2.5"),
            ("4 / 2", "2.0"),
            ("1 / 3", "0.3333333333333333"),
            ("3 / 7", "0.42857142857142855"),
            ("-1 / 3", "-0.3333333333333333"),
            ("0.1 + 0.2", "0.30000000000000004"),
            ("1e16", "1e+16"),
            ("1e15", "1000000000000000.0"),
            ("123456789.123", "123456789.123"),
            ("0.0001", "0.0001"),
            ("0.00001", "1e-05"),
            ("1.5e-7", "1.5e-07"),
            ("2 ** -1", "0.5"),
            ("2 ** 0.5", "1.4142135623730951"),
            ("1e100", "1e+100"),
            ("-0.0", "-0.0"),
            ("0 / -5", "-0.0"),
            (".5", "0.5"),
            ("5.", "5.0"),
            ("1.e2", "100.0"),
            ("2.5E+2", "250.0"),
            ("3 * 1.5", "4.5"),
            ("10 / 4 * 2", "5.0"),
        ] {
            assert_eq!(calc(expression), expected, "{expression}");
        }
    }

    #[test]
    fn power_binds_like_python() {
        for (expression, expected) in [
            ("-2 ** 2", "-4"),
            ("2 ** 3 ** 2", "512"),
            ("(2 ** 3) ** 2", "64"),
            ("2 ** -2", "0.25"),
            ("-2 ** -2", "-0.25"),
            ("(-2) ** 3", "-8"),
            ("(-2) ** 2", "4"),
            ("(-2) ** -2", "0.25"),
            ("(-8) ** 2.0", "64.0"),
            ("2 * 3 ** 2", "18"),
        ] {
            assert_eq!(calc(expression), expected, "{expression}");
        }
    }

    #[test]
    fn floor_division_and_remainder_follow_the_divisor_s_sign() {
        for (expression, expected) in [
            ("7 // 2", "3"),
            ("-7 // 2", "-4"),
            ("7 // -2", "-4"),
            ("-7 // -2", "3"),
            ("7 % 3", "1"),
            ("-7 % 3", "2"),
            ("7 % -3", "-2"),
            ("-7 % -3", "-1"),
            ("6 % -3", "0"),
            ("7.5 // 2", "3.0"),
            ("-7.5 // 2", "-4.0"),
            ("7.5 % 2", "1.5"),
            ("-7.5 % 2", "0.5"),
            ("7.5 % -2", "-0.5"),
            ("5 // 0.3", "16.0"),
            ("1 % 0.1", "0.09999999999999995"),
            ("-1 // 0.1", "-10.0"),
            ("6 % -3.0", "-0.0"),
            ("0.0 // -1", "-0.0"),
        ] {
            assert_eq!(calc(expression), expected, "{expression}");
        }
    }

    #[test]
    fn integer_division_is_correctly_rounded_beyond_two_to_the_53() {
        // 2^53 + 1 is not a double; a / b must round once, from the exact value.
        for (expression, expected) in [
            ("9007199254740993 / 1", "9007199254740992.0"),
            ("9007199254740993 / 3", "3002399751580331.0"),
            ("1 / 10000000000000000000000", "1e-22"),
            ("1 / 10000000000000000000000000000000000000", "1e-37"),
            ("100000000000000000000 / 3", "3.333333333333333e+19"),
            (
                "170141183460469231731687303715884105727 / 3",
                "5.671372782015641e+37",
            ),
            ("340282366920938463 / 7", "4.861176670299121e+16"),
            // Exactly half-way between two doubles: ties go to the even one.
            ("27021597764222979 / 3", "9007199254740992.0"),
            ("27021597764222985 / 3", "9007199254740996.0"),
            ("18014398509481985 / 2", "9007199254740992.0"),
            ("-7 / 2", "-3.5"),
            ("7 / -2", "-3.5"),
        ] {
            assert_eq!(calc(expression), expected, "{expression}");
        }
    }

    #[test]
    fn a_float_prints_as_pythons_repr() {
        for (value, expected) in [
            (0.0, "0.0"),
            (-0.0, "-0.0"),
            (1.0, "1.0"),
            (100.0, "100.0"),
            (1e15, "1000000000000000.0"),
            (1e16, "1e+16"),
            (1.2345e16, "1.2345e+16"),
            (123456789012345680.0, "1.2345678901234568e+17"),
            (0.5, "0.5"),
            (0.001, "0.001"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (0.000123, "0.000123"),
            (1.5e-10, "1.5e-10"),
            (1e100, "1e+100"),
            (2.5e-100, "2.5e-100"),
            (5e-324, "5e-324"),
            (-1.5, "-1.5"),
            (std::f64::consts::PI, "3.141592653589793"),
            (f64::INFINITY, "inf"),
            (f64::NEG_INFINITY, "-inf"),
        ] {
            assert_eq!(float_repr(value), expected, "{value:e}");
        }
    }

    #[test]
    fn anything_that_is_not_arithmetic_is_refused_whole() {
        for expression in [
            "abs(-1)",
            "__import__('os')",
            "1 + x",
            "2 ^ 3",
            "1 & 2",
            "1 << 2",
            "~1",
            "1_000",
            "0x10",
            "1,5",
            "1;2",
            "\"a\"",
            "1 if 1 else 2",
            "[1]",
            "2x",
            "e5",
            "1e",
            "1e+",
            "\u{ff11}",
            "1 \u{2212} 1",
            "$1",
            "1 = 1",
            "!1",
            "1 < 2",
        ] {
            assert_eq!(
                calc(expression),
                "error: Only numbers, + - * / // % ** and parentheses are allowed.",
                "{expression:?}"
            );
        }
    }

    #[test]
    fn incomplete_expressions_are_refused_with_their_own_sentence() {
        for expression in [
            "",
            "   ",
            "1 +",
            "(1",
            "1)",
            "()",
            "* 2",
            "1 2",
            "1 ** ",
            "1 // // 2",
            "(1 + )",
            "1.2.3",
            "2 (3)",
        ] {
            assert_eq!(
                calc(expression),
                "error: That is not a complete arithmetic expression.",
                "{expression:?}"
            );
        }
    }

    #[test]
    fn the_limits_are_refusals() {
        for (expression, expected) in [
            ("1 / 0", "error: division by zero"),
            ("1 // 0", "error: division by zero"),
            ("1 % 0", "error: division by zero"),
            ("1.5 / 0", "error: division by zero"),
            ("1 / 0.0", "error: division by zero"),
            ("1 // 0.0", "error: division by zero"),
            ("1 % 0.0", "error: division by zero"),
            ("0 ** -1", "error: division by zero"),
            ("0.0 ** -1", "error: division by zero"),
            ("(-8) ** 0.5", "error: The result is not a real number."),
            ("(-8) ** (1 / 3)", "error: The result is not a real number."),
            ("1000000 ** 7", "error: The result is too large."),
            ("1e6 ** 20", "error: The result is too large."),
            ("9.9e99 * 10", "error: The result is too large."),
            ("1e100 + 1e100", "error: The result is too large."),
            ("1e101", "error: A number in the expression is too large."),
            ("1e999", "error: A number in the expression is too large."),
            (
                "170141183460469231731687303715884105728",
                "error: A number in the expression is too large.",
            ),
            (
                "170141183460469231731687303715884105727 + 1",
                "error: The result is too large.",
            ),
            (
                "-170141183460469231731687303715884105727 - 2",
                "error: The result is too large.",
            ),
            (
                "-9223372036854775808 * 18446744073709551616 // -1",
                "error: The result is too large.",
            ),
        ] {
            assert_eq!(calc(expression), expected, "{expression}");
        }
        for (expression, expected) in [
            ("2 ** 64", "18446744073709551616"),
            ("1000000 ** 3", "1000000000000000000"),
            ("5e99 + 5e99", "1e+100"),
            (
                "170141183460469231731687303715884105727",
                "170141183460469231731687303715884105727",
            ),
            ("-9223372036854775808 * 18446744073709551616 % -1", "0"),
        ] {
            assert_eq!(calc(expression), expected, "{expression}");
        }
        let exponent =
            "error: An exponent must be between -64 and 64, and its base at most 1000000 in size.";
        for expression in [
            "2 ** 65",
            "2 ** -65",
            "1000001 ** 2",
            "-1000001 ** 2",
            "2 ** 1e99",
            "0.5 ** 65",
        ] {
            assert_eq!(calc(expression), exponent, "{expression}");
        }
    }

    #[test]
    fn length_and_nesting_are_bounded() {
        let parens = |n: usize| format!("{}1{}", "(".repeat(n), ")".repeat(n));
        assert_eq!(calc(&parens(64)), "1", "64 levels are allowed");
        assert_eq!(
            calc(&parens(65)),
            "error: The expression is nested more than 64 levels deep."
        );
        assert_eq!(
            calc(&parens(90)),
            "error: The expression is nested more than 64 levels deep."
        );
        assert_eq!(calc(&format!("{}1", "-".repeat(64))), "1");
        assert_eq!(
            calc(&format!("{}1", "-".repeat(65))),
            "error: The expression is nested more than 64 levels deep."
        );
        assert_eq!(calc(&format!("1{}", "**1".repeat(64))), "1");
        assert_eq!(
            calc(&format!("1{}", "**1".repeat(65))),
            "error: The expression is nested more than 64 levels deep."
        );
        // A long flat expression is not deep.
        assert_eq!(calc(&("1+".repeat(99) + "1")), "100");
        let long = "1+".repeat(100) + "1";
        assert_eq!(long.chars().count(), 201);
        assert_eq!(
            calc(&long),
            "error: The expression is longer than 200 characters."
        );
        let exactly = "1+".repeat(99) + "11";
        assert_eq!(exactly.chars().count(), 200);
        assert_eq!(calc(&exactly), "110");
        assert_eq!(
            calc(&"\u{1F600}".repeat(201)),
            "error: The expression is longer than 200 characters.",
            "characters, not bytes"
        );
    }

    /// A tiny deterministic generator, so the property test needs no crate.
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 33
        }

        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    /// A random integer expression, as text.
    fn generate(rng: &mut Lcg, depth: u32) -> String {
        if depth == 0 || rng.below(4) == 0 {
            return rng.below(50).to_string();
        }
        match rng.below(7) {
            0 => format!("({})", generate(rng, depth - 1)),
            1 => format!("-{}", generate(rng, depth - 1)),
            2 => format!("+{}", generate(rng, depth - 1)),
            _ => {
                let op = ["+", "-", "*", "//", "%", "**"][rng.below(6) as usize];
                format!(
                    "{} {op} {}",
                    generate(rng, depth - 1),
                    generate(rng, depth - 1)
                )
            }
        }
    }

    /// Why the reference has no value to compare with.
    #[derive(Debug, PartialEq)]
    enum Stop {
        DivisionByZero,
        /// Something the property test does not compare (an overflow, a negative
        /// power, a limit): skipped, and covered by the unit tests above.
        Skip,
    }

    /// The reference: a precedence-climbing (Pratt) parser over integers, written
    /// separately from the evaluator (a different way to parse the same grammar)
    /// with floor division and remainder defined through `div_euclid`.
    struct Reference {
        tokens: Vec<String>,
        at: usize,
    }

    impl Reference {
        fn new(source: &str) -> Self {
            let chars: Vec<char> = source.chars().collect();
            let mut tokens = Vec::new();
            let mut i = 0;
            while i < chars.len() {
                let c = chars[i];
                if c == ' ' {
                    i += 1;
                } else if c.is_ascii_digit() {
                    let start = i;
                    while i < chars.len() && chars[i].is_ascii_digit() {
                        i += 1;
                    }
                    tokens.push(chars[start..i].iter().collect());
                } else if (c == '*' || c == '/') && chars.get(i + 1) == Some(&c) {
                    tokens.push(format!("{c}{c}"));
                    i += 2;
                } else {
                    tokens.push(c.to_string());
                    i += 1;
                }
            }
            Self { tokens, at: 0 }
        }

        fn parse(&mut self, min_binding: u8) -> Result<i128, Stop> {
            let token = self.tokens.get(self.at).cloned().ok_or(Stop::Skip)?;
            self.at += 1;
            let mut left = match token.as_str() {
                "(" => {
                    let inside = self.parse(0)?;
                    self.at += 1; // the closing parenthesis
                    inside
                }
                "-" => self.parse(30)?.checked_neg().ok_or(Stop::Skip)?,
                "+" => self.parse(30)?,
                digits => digits.parse::<i128>().map_err(|_| Stop::Skip)?,
            };
            while let Some(op) = self.tokens.get(self.at).cloned() {
                let (left_binding, right_binding) = match op.as_str() {
                    "+" | "-" => (10, 11),
                    "*" | "//" | "%" => (20, 21),
                    "**" => (40, 39),
                    _ => break,
                };
                if left_binding < min_binding {
                    break;
                }
                self.at += 1;
                let right = self.parse(right_binding)?;
                left = Self::apply(&op, left, right)?;
            }
            Ok(left)
        }

        fn apply(op: &str, x: i128, y: i128) -> Result<i128, Stop> {
            match op {
                "+" => x.checked_add(y).ok_or(Stop::Skip),
                "-" => x.checked_sub(y).ok_or(Stop::Skip),
                "*" => x.checked_mul(y).ok_or(Stop::Skip),
                "//" | "%" => {
                    if y == 0 {
                        return Err(Stop::DivisionByZero);
                    }
                    // floor(x / y): for a negative divisor, negate both.
                    let (n, d) = if y > 0 {
                        (x, y)
                    } else {
                        (x.checked_neg().ok_or(Stop::Skip)?, -y)
                    };
                    let floor = n.div_euclid(d);
                    if op == "//" {
                        Ok(floor)
                    } else {
                        x.checked_sub(y.checked_mul(floor).ok_or(Stop::Skip)?)
                            .ok_or(Stop::Skip)
                    }
                }
                _ => {
                    if !(0..=64).contains(&y) || x.abs() > 1_000_000 {
                        return Err(Stop::Skip);
                    }
                    x.checked_pow(y as u32).ok_or(Stop::Skip)
                }
            }
        }
    }

    #[test]
    fn evaluation_agrees_with_an_independent_reference_on_random_expressions() {
        let mut rng = Lcg(0x5eed_1234_abcd_ef01);
        let (mut compared, mut refused_for_zero) = (0, 0);
        for _ in 0..6000 {
            let source = generate(&mut rng, 5);
            if source.chars().count() > MAX_EXPRESSION_CHARS {
                continue;
            }
            match Reference::new(&source).parse(0) {
                Ok(expected) => {
                    assert_eq!(evaluate(&source), Ok(Num::Int(expected)), "{source}");
                    compared += 1;
                }
                Err(Stop::DivisionByZero) => {
                    assert_eq!(
                        evaluate(&source),
                        Err(CalcError::DivisionByZero),
                        "{source}"
                    );
                    refused_for_zero += 1;
                }
                Err(Stop::Skip) => {}
            }
        }
        assert!(
            compared > 2000,
            "the generator must give enough comparable expressions ({compared})"
        );
        assert!(
            refused_for_zero > 50,
            "and enough zero divisors ({refused_for_zero})"
        );
    }

    #[test]
    fn random_garbage_never_panics() {
        let mut rng = Lcg(42);
        let alphabet: Vec<char> = "0123456789.eE+-*/%() \t\n\u{e9}xX_".chars().collect();
        for _ in 0..20_000 {
            let len = rng.below(40) as usize;
            let source: String = (0..len)
                .map(|_| alphabet[rng.below(alphabet.len() as u64) as usize])
                .collect();
            let _ = evaluate(&source);
        }
    }
}
