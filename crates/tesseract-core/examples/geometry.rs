//! Prints the geometry this build of the engine uses.
//!
//! `cargo run -p tesseract-core --example geometry`
//!
//! The two constants that set it are `SIDE` and `LEVELS` at the top of
//! `src/entity.rs`. Change them, rebuild, and everything follows.

use tesseract_core::{
    BITS, DIMENSION_BITS, DOMAIN_BITS, DOMAINS, Entity, LEVELS, LOG_SIDE, SIDE, WORDS,
    WORDS_PER_DOMAIN,
};

fn main() {
    println!("side            {SIDE}  ({LOG_SIDE} bits per digit)");
    println!("axes            {LEVELS}");
    println!("entity          {BITS} bits, {} bytes", BITS / 8);
    println!("words           {WORDS} of 64 bits");
    println!("domains         {DOMAINS} of {DOMAIN_BITS} bits ({WORDS_PER_DOMAIN} words)");
    println!("dimension       {DIMENSION_BITS} bits");
    println!("entity in memory {} bytes", size_of::<Entity>());
    println!();
    println!(
        "one memory unit  {} bytes of parameters",
        4 * size_of::<Entity>()
    );
    println!(
        "                 {} bytes more while it learns",
        2 * BITS * size_of::<i16>()
    );
}
