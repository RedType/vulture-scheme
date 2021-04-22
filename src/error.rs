#[derive(Debug)]
pub enum LexError {
    EscapeAtEndOfString(String),
    InvalidEscapeSequence(String),
    NoMatch(String),
    //ReservedSymbol(String, usize),
}
