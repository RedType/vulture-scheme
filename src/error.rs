#[derive(Debug)]
pub enum LexError {
    NoToken(usize), // handled by lex() internally
    EscapeAtEndOfString(String),
    InvalidEscapeSequence(String),
    NoMatch(String),
    //ReservedSymbol(String, usize),
}
