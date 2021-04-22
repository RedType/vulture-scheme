use crate::{
    error::LexError as Error,
    types as ty,
};
use lazy_static::lazy_static;
use regex::{Regex, RegexSet};
use unicode_segmentation::UnicodeSegmentation;

pub type Token = Tagged<Lexeme>;

#[derive(Debug)]
pub struct Tagged<T> {
    pub elem: T,
    pub line: usize,
    pub col: usize,
    pub bytes: usize, // number of bytes that this lexeme took up in the corpus
}

#[derive(Clone, Debug, PartialEq)]
pub enum Lexeme {
    Nothing,
    Identifier(String),
    LParen, RParen,
    Num(ty::Number),
    String_(String),
    True, False,
}

pub struct Corpus<'a> {
    pub text: &'a str,
    line: usize,
    col: usize,
}

impl<'a> Corpus<'a> {
    pub fn new(text: &'a str) -> Self {
        Self {
            text,
            line: 1,
            col: 1,
        }
    }

    pub fn advance_by(&mut self, bytes: usize) {
        let skipped_lines = &self.text[..bytes].lines();
        let lines = skipped_lines.clone().count() - 1;
        let cols = skipped_lines.clone()
                                .last()
                                .unwrap_or("")
                                .graphemes(true)
                                .count();
        self.text = &self.text[bytes..];
        self.line += lines;
        self.col = if lines == 0 {
            self.col + cols
        } else {
            cols + 1
        };
    }

    pub fn tag<T>(&self, elem: T, bytes: usize) -> Tagged<T> {
        Tagged {
            bytes, elem,
            line: self.line,
            col: self.col,
        }
    }
}

pub fn lex(corpus: &str) -> Result<Vec<Token>, Error> {
    let mut tokens: Vec<Token> = Vec::new();
    let mut head = Corpus::new(corpus);
    while head.text.len() > 0 {
        // lexers are listed in order of priority (low = better)
        let token = match ALL.matches(head.text).iter().min() {
            Some(i) => LEXERS[i](&head)?,
            None => return Err(Error::NoMatch(head.text.to_owned())),
        };
        head.advance_by(token.bytes);
        tokens.push(token);
    }
    tokens = tokens.into_iter()
                   .filter(|t| t.elem != Lexeme::Nothing)
                   .collect();
    Ok(tokens)
}

type Lexer = dyn for<'a> Fn(&'a Corpus) -> Result<Token, Error> + Sync;
macro_rules! gen_matchers {
    { $($f:ident => $re:expr),+$(,)? } => {
        lazy_static! {
            static ref ALL: RegexSet = RegexSet::new(&[$($re),+]).unwrap();
            static ref LEXERS: Vec<Box<Lexer>> =
                vec![$(Box::new(|c| {
                    let re = Regex::new($re).unwrap();
                    decode::$f(re, c)
                })),+];
        }
    };
}

gen_matchers! {
    parens => r"[()]",
    true_ => r"^#t(?:rue)?",
    false_ => r"^#f(?:alse)?",
    string => r#"^"((?:[^"\\]|\\"|\\\\|\\\s*)*)""#,
    inf_nan => r"^(\+|-)(inf|nan)\.0",
    complex_number => r"(?x)^
        ([\+-]?\d*\.?\d+  # real part
        (?:e[\+-]?\d+)?)? # real exponent
        ([\+-]?\d*\.?\d+  # imaginary part
        (?:e[\+-]?\d+)?)i # imaginary exponent (and i)",
    real_number => r"(?x)^
        ([\+-]?\d*\.?\d+) # real
        (e[\+-]?\d+)?     # exponent",
    rational_number => r"(?x)^
        ([\+-]?\d+       # numerator
        (?:e[\+-]?\d+)?) # numerator exponent
        /                # divided by
        ([\+-]\d+        # denominator
        (?:e[\+-]?\d+)?) # denominator exponent",
    integer => r"(?x)^
        ([\+-]?\d+       # integer
        (?:e[\+-]?\d+)?) # exponent",
    identifier => r"^\p{Alphabetic}\w*",
    whitespace => r"^\s+",
    line_comment => r"^;[^\r\n]*\r|\n|(?:\r\n)",
}

mod decode {
    use crate::{
        error::LexError as Error,
        types::Number,
        lexer::{
            Corpus,
            Lexeme,
            Token,
        },
    };
    use regex::Regex;
    use std::str::FromStr;
    use unicode_segmentation::UnicodeSegmentation;

    type Result<T> = std::result::Result<T, Error>;

    fn str_is_whitespace(s: &str) -> bool {
        s.chars().fold(true, |a, c| a && c.is_whitespace())
    }

    fn replace_escapes(s: &str) -> Result<String> {
        let mut gs = s.graphemes(true).peekable();
        let mut string = String::new();
        while let Some(g) = gs.next() {
            match g {
                r"\" => match gs.next() {
                    None => return Err(
                        Error::EscapeAtEndOfString(s.to_owned())),
                    Some(r"\") => string.push('\\'),
                    Some("n") => string.push_str("\n"),
                    Some("r") => string.push('\r'),
                    Some("t") => string.push('\t'),
                    Some("0") => string.push('\0'),
                    Some("\"") => string.push('"'),
                    // consume whitespace after a \
                    Some(g) if str_is_whitespace(g) => {
                        while let Some(g) = gs.peek() {
                            if str_is_whitespace(g) {
                                let _ = gs.next();
                            }
                        }
                    },
                    Some(g) => return Err(
                        Error::InvalidEscapeSequence(format!(r"\{}", g))),
                },
                g => string.push_str(g),
            }
        }
        Ok(string)
    }

    pub fn parens(re: Regex, corpus: &Corpus) -> Result<Token> {
        let m_str = re.find(&corpus.text).unwrap().as_str();
        let bytes = m_str.len();
        let lex = match m_str {
            "(" => Lexeme::LParen,
            ")" => Lexeme::RParen,
            _ => unreachable!(),
        };
        Ok(corpus.tag(lex, bytes))
    }

    pub fn true_(re: Regex, corpus: &Corpus) -> Result<Token> {
        let len = re.find(&corpus.text).unwrap().as_str().len();
        Ok(corpus.tag(Lexeme::True, len))
    }

    pub fn false_(re: Regex, corpus: &Corpus) -> Result<Token> {
        let len = re.find(&corpus.text).unwrap().as_str().len();
        Ok(corpus.tag(Lexeme::False, len))
    }

    pub fn string(re: Regex, corpus: &Corpus) -> Result<Token> {
        let cs = re.captures(&corpus.text).unwrap();
        let len = cs.get(0).unwrap().as_str().len();
        let raw = cs.get(1).unwrap().as_str();
        let text = replace_escapes(raw)?;
        Ok(corpus.tag(Lexeme::String_(text), len))
    }

    pub fn complex_number(re: Regex, corpus: &Corpus) -> Result<Token> {
        let cs = re.captures(corpus.text).unwrap();
        let len = cs.get(0).unwrap().as_str().len();
        let real_part =
            if let Some(real_match) = cs.get(1) {
                f64::from_str(real_match.as_str()).unwrap()
            } else { 0.0 };
        let imag_part =
            if let Some(imag_match) = cs.get(2) {
                f64::from_str(imag_match.as_str()).unwrap()
            } else { 1.0 };
        let lex = Lexeme::Num(Number::Complex(real_part, imag_part));
        Ok(corpus.tag(lex, len))
    }

    pub fn inf_nan(re: Regex, corpus: &Corpus) -> Result<Token> {
        let cs = re.captures(corpus.text).unwrap();
        let len = cs.get(0).unwrap().as_str().len();
        let sign = cs.get(1).unwrap().as_str();
        let inf_nan_str = cs.get(2).unwrap().as_str();
        let inf_nan = match (sign, inf_nan_str) {
            (_, "nan") => f64::NAN,
            ("+", "inf") => f64::INFINITY,
            ("-", "inf") => f64::NEG_INFINITY,
            _ => unreachable!(),
        };
        Ok(corpus.tag(Lexeme::Num(Number::Real(inf_nan)), len))
    }

    pub fn real_number(re: Regex, corpus: &Corpus) -> Result<Token> {
        let real_str = re.find(corpus.text).unwrap().as_str();
        let real = f64::from_str(&real_str).expect("Real false positive");
        Ok(corpus.tag(Lexeme::Num(Number::Real(real)), real_str.len()))
    }

    pub fn rational_number(re: Regex, corpus: &Corpus) -> Result<Token> {
        let cs = re.captures(corpus.text).unwrap();
        let len = cs.get(0).unwrap().as_str().len();
        let numerator_str = cs.get(1).unwrap().as_str();
        let denominator_str = cs.get(2).unwrap().as_str();
        let mut numerator = i64::from_str(numerator_str)
                .expect("Rational numerator false positive");
        let mut denominator = i64::from_str(denominator_str)
                .expect("Rational denominator false positive");
        if denominator < 0 {
            numerator *= -1;
            denominator *= -1;
        }
        let num = Number::Rational(numerator, denominator as u64);
        Ok(corpus.tag(Lexeme::Num(num), len))
    }

    pub fn integer(re: Regex, corpus: &Corpus) -> Result<Token> {
        let cs = re.captures(corpus.text).unwrap();
        let len = cs.get(0).unwrap().as_str().len();
        let int_str = cs.get(1).unwrap().as_str();
        let int = i64::from_str(int_str).expect("Integer false positive");
        let num = Number::Integral(int);
        Ok(corpus.tag(Lexeme::Num(num), len))
    }

    pub fn identifier(re: Regex, corpus: &Corpus) -> Result<Token> {
        let text = re.find(corpus.text).unwrap().as_str();
        let lex = Lexeme::Identifier(text.to_owned());
        Ok(corpus.tag(lex, text.len()))
    }

    pub fn whitespace(re: Regex, corpus: &Corpus) -> Result<Token> {
        let len = re.find(corpus.text).unwrap().as_str().len();
        Ok(corpus.tag(Lexeme::Nothing, len))
    }

    pub fn line_comment(re: Regex, corpus: &Corpus) -> Result<Token> {
        let len = re.find(corpus.text).unwrap().as_str().len();
        Ok(corpus.tag(Lexeme::Nothing, len))
    }
}

#[test]
fn lexeme_validation() {
    use Lexeme::*;
    let first = |v: Result<Vec<Token>, _>| v.unwrap()[0].elem.clone();
    assert_eq!(first(lex("#t")), True);
    assert_eq!(first(lex("#true")), True);
    assert_eq!(first(lex("#f")), False);
    assert_eq!(first(lex("#false")), False);
    assert_eq!(first(lex(r#""hello world""#)),
        String_("hello world".to_owned()));
    assert_eq!(first(lex(r#""\"hello\" \"world\"""#)),
        String_("\"hello\" \"world\"".to_owned()));
    assert_eq!(first(lex("(")), LParen);
    assert_eq!(first(lex(")")), RParen);
}
