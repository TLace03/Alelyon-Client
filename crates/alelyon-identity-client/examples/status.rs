//! Print the identity service's status: whether it is up, its notices and which sign-in providers are on.
//! It needs no account and sends no credential.
//!
//!     cargo run --example status
//!     ALELYON_IDENTITY_URL=http://127.0.0.1:8765 cargo run --example status

use alelyon_identity_client::{Client, client};

fn main() {
    let Some(url) = client::base_url() else {
        println!("ALELYON_IDENTITY_URL=off: no identity service is configured.");
        return;
    };
    let client = match Client::new(url.clone()) {
        Ok(c) => c,
        Err(why) => {
            eprintln!("{why}");
            std::process::exit(2);
        }
    };
    let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build().expect("a tokio runtime");
    match runtime.block_on(client.status()) {
        Ok(status) => {
            println!("{url}: service {}", if status.service.is_empty() { "(not stated)" } else { &status.service });
            for n in &status.notices {
                println!("  notice [{}] {} ({} to {})", n.kind, n.title, n.starts_at, n.ends_at);
            }
            for p in &status.providers {
                println!("  provider {:<12} {}", p.id, if p.enabled { "on" } else { "off" });
            }
        }
        Err(failure) => {
            eprintln!("{url}: {}", failure.words());
            std::process::exit(1);
        }
    }
}
