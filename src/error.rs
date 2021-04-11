use crate::lexer::Fragment;

#[derive(Debug)]
pub enum Error {
    Ambiguous(String, Vec<Fragment>),
    EscapeAtEndOfString(String),
    InvalidEscapeSequence(String),
    NoMatch(String),
    ReservedSymbol(String, usize),
}
