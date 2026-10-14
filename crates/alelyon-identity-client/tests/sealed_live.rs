//! Sealed friends chat against a running identity service. Skipped unless `ALELYON_SEALED_LIVE` names a JSON file with
//! `{"url", "ada": {"id", "token"}, "bob": {"id", "token"}}` for two signed-in friends on a test service; nothing here
//! talks to the deployed one.


use alelyon_identity_client::client::Client;
use alelyon_identity_client::sealed::DeviceKey;
use alelyon_identity_client::sealed_chat::{self, Entry, Opened};
use alelyon_identity_client::social::{self, Problem};
use serde_json::Value;

fn setup() -> Option<(String, (String, String), (String, String))> {
    let path = std::env::var("ALELYON_SEALED_LIVE").ok()?;
    let v: Value = serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()?;
    let person = |k: &str| (v[k]["id"].as_str().unwrap().to_string(), v[k]["token"].as_str().unwrap().to_string());
    Some((v["url"].as_str()?.to_string(), person("ada"), person("bob")))
}

#[test]
fn ada_seals_for_bob_over_the_service_and_bob_opens_it() {
    let Some((url, (ada_id, ada_token), (bob_id, bob_token))) = setup() else {
        eprintln!("ALELYON_SEALED_LIVE is not set: skipped");
        return;
    };
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
    rt.block_on(async {
        let c = Client::new(url).unwrap();
        let ada = DeviceKey::generate().unwrap();
        let bob = DeviceKey::generate().unwrap();

        // Ada has a key, Bob not yet: a sealed send is refused and plain text still goes.
        sealed_chat::register(&c, &ada_token, &ada).await.unwrap();
        let none = sealed_chat::send(&c, &ada_token, &ada, &ada_id, &bob_id, "too early").await;
        assert!(matches!(none, Err(Problem::Refused(ref code, _)) if code == "no-devices"), "{none:?}");
        social::send(&c, &ada_token, &bob_id, "plain, before Bob had a key").await.unwrap();

        let bob_device = sealed_chat::register(&c, &bob_token, &bob).await.unwrap();
        assert_eq!(bob_device.id, bob.id(), "the service derives the same device id");

        // Both have keys: plain text is refused, and a sealed message goes through.
        let refused = social::send(&c, &ada_token, &bob_id, "now refused").await;
        assert!(matches!(refused, Err(Problem::Refused(ref code, _)) if code == "sealed-required"), "{refused:?}");
        let sent = sealed_chat::send(&c, &ada_token, &ada, &ada_id, &bob_id, "hello Bob, sealed ✓").await.unwrap();
        assert!(matches!(sent, Entry::Sealed(ref s) if s.key.is_some()), "Ada's own copy comes back");

        // Bob reads as his device and opens it; the earlier plain message is there too.
        let (entries, _) = sealed_chat::conversation(&c, &bob_token, &ada_id, &bob.id(), None).await.unwrap();
        let theirs = sealed_chat::devices(&c, &bob_token, Some(&ada_id)).await.unwrap();
        let senders = sealed_chat::Senders::new(&theirs, &[]);
        let opened: Vec<Opened> = entries.iter().map(|e| sealed_chat::open(e, &bob, &bob_id, &ada_id, &senders)).collect();
        assert_eq!(opened, vec![Opened::Text("plain, before Bob had a key".into()), Opened::Text("hello Bob, sealed ✓".into())]);

        // Ada's own device opens her copy too.
        let (mine, _) = sealed_chat::conversation(&c, &ada_token, &bob_id, &ada.id(), None).await.unwrap();
        let own = sealed_chat::Senders { mine: [(ada.id(), ada.public())].into(), ..Default::default() };
        assert_eq!(sealed_chat::open(mine.last().unwrap(), &ada, &ada_id, &bob_id, &own), Opened::Text("hello Bob, sealed ✓".into()));

        // Bob adds a second device: Ada's next send covers it (her client re-reads the list), and it opens there.
        let bob_laptop = DeviceKey::generate().unwrap();
        sealed_chat::register(&c, &bob_token, &bob_laptop).await.unwrap();
        sealed_chat::send(&c, &ada_token, &ada, &ada_id, &bob_id, "for both of Bob's devices").await.unwrap();
        let (on_laptop, _) = sealed_chat::conversation(&c, &bob_token, &ada_id, &bob_laptop.id(), None).await.unwrap();
        let texts: Vec<Opened> = on_laptop.iter().map(|e| sealed_chat::open(e, &bob_laptop, &bob_id, &ada_id, &senders)).collect();
        assert_eq!(texts.last(), Some(&Opened::Text("for both of Bob's devices".into())));
        assert_eq!(texts[1], Opened::NotForThisDevice, "the message sealed before the laptop existed says so");
    });
}

/// The website seals, the app opens, and the app's reply opens on the website. Skipped unless `ALELYON_SEALED_CROSS`
/// names a JSON file the website's half (`www/scripts/sealed-cross.mjs`) wrote: `{"url", "ada": {"id", "token"},
/// "bob": {"id", "token"}, "bob_secret", "expect"}` (Bob's device secret, shared with that script for the test only).
#[test]
fn a_message_sealed_on_the_website_opens_in_the_app_and_the_reply_goes_back() {
    let Ok(path) = std::env::var("ALELYON_SEALED_CROSS") else {
        eprintln!("ALELYON_SEALED_CROSS is not set: skipped");
        return;
    };
    let v: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let s = |a: &str, b: &str| v[a][b].as_str().unwrap().to_string();
    let (url, ada_id, bob_id, bob_token) = (v["url"].as_str().unwrap().to_string(), s("ada", "id"), s("bob", "id"), s("bob", "token"));
    let hex = v["bob_secret"].as_str().unwrap();
    let secret: [u8; 32] = (0..64).step_by(2).map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap()).collect::<Vec<u8>>().try_into().unwrap();
    let bob = DeviceKey::from_secret(secret);
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
    rt.block_on(async {
        let c = Client::new(url).unwrap();
        let (entries, _) = sealed_chat::conversation(&c, &bob_token, &ada_id, &bob.id(), None).await.unwrap();
        let theirs = sealed_chat::devices(&c, &bob_token, Some(&ada_id)).await.unwrap();
        let senders = sealed_chat::Senders::new(&theirs, &[]);
        let last = entries.last().expect("the website's message is there");
        assert_eq!(sealed_chat::open(last, &bob, &bob_id, &ada_id, &senders), Opened::Text(v["expect"].as_str().unwrap().into()));
        sealed_chat::send(&c, &bob_token, &bob, &bob_id, &ada_id, "from the app back to the website \u{2713}").await.unwrap();
    });
}
