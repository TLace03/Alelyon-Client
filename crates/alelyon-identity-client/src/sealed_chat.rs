//! Sealed friends chat over the identity service's contract v2 (`docs/social-contract.md`, "Sealed messages"): this
//! device's key kept with DPAPI, registering it, sealing a message for every live device of both people, and opening
//! what arrives. The envelope itself is [`crate::sealed`].
//!
//! What the person should know, and an application must say: this gives no forward secrecy (a stolen device key opens
//! every past message sealed for it), the service still sees who writes to whom and when, and a device that loses its
//! key loses the messages only it could open.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use ring::rand::{SecureRandom, SystemRandom};
use serde_json::{Value, json};

use crate::client::Client;
use crate::sealed::{self, DeviceKey, Envelope, Header, WrappedKey};
use super::social::{Message, Problem, call, interpret};
use crate::vault;

/// The label the device key is sealed under, apart from the session's.
const DEVICE_ENTROPY: &[u8] = b"alelyon.friends.device-key/v1";

/// Where this device's key is kept: `~/.alelyon/alelyon/friends-device.sealed`, or the file `ALELYON_FRIENDS_DEVICE_FILE`
/// names.
pub fn device_key_path() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("ALELYON_FRIENDS_DEVICE_FILE").filter(|v| !v.is_empty()) {
        return Some(PathBuf::from(p));
    }
    std::env::var_os("USERPROFILE").map(|h| PathBuf::from(h).join(".alelyon").join("alelyon").join("friends-device.sealed"))
}

/// This device's key from `at`, or a new one kept there. `None` when it can be neither opened nor kept (no DPAPI): a
/// key that cannot be kept is not made, since messages sealed for it could never be opened again.
pub fn load_or_create_device_key(at: &Path) -> Result<DeviceKey, String> {
    if let Some(hex) = vault::open_with(at, DEVICE_ENTROPY)
        && let Some(secret) = unhex32(&hex)
    {
        return Ok(DeviceKey::from_secret(secret));
    }
    let key = DeviceKey::generate().map_err(|e| format!("{e:?}"))?;
    vault::keep_with(at, &hex(key.secret()), DEVICE_ENTROPY)?;
    Ok(key)
}

/// Make a new key in place of a revoked one (the service never takes a revoked key back).
pub fn replace_device_key(at: &Path) -> Result<DeviceKey, String> {
    vault::forget(at);
    load_or_create_device_key(at)
}

/// A device as the service lists it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Device {
    pub id: String,
    pub public_key: [u8; 32],
    pub created_at: String,
}

impl Device {
    fn from_json(v: &Value) -> Option<Device> {
        let raw = URL_SAFE_NO_PAD.decode(v.get("public_key")?.as_str()?).ok()?;
        let public_key: [u8; 32] = raw.try_into().ok()?;
        let id = v.get("id")?.as_str()?.to_string();
        // The id is derived from the key; a list that disagrees is not trusted.
        (sealed::device_id(&public_key) == id).then(|| Device {
            id,
            public_key,
            created_at: v.get("created_at").and_then(Value::as_str).unwrap_or_default().to_string(),
        })
    }
}

fn devices_from(v: &Value) -> Vec<Device> {
    v.get("devices").and_then(Value::as_array).map(|a| a.iter().filter_map(Device::from_json).collect()).unwrap_or_default()
}

/// Register this device's public key; registering it again answers the same device.
pub async fn register(client: &Client, token: &str, key: &DeviceKey) -> Result<Device, Problem> {
    let v = call(client, token, "/identity/native/social/device-register", json!({"public_key": URL_SAFE_NO_PAD.encode(key.public())})).await?;
    Device::from_json(v.get("device").unwrap_or(&Value::Null)).ok_or_else(|| Problem::Refused("bad-device".into(), "The service answered with a device that does not match its key.".into()))
}

/// This account's live devices (`friend` None), or a friend's.
pub async fn devices(client: &Client, token: &str, friend: Option<&str>) -> Result<Vec<Device>, Problem> {
    let body = match friend {
        Some(f) => json!({"friend_id": f}),
        None => json!({}),
    };
    call(client, token, "/identity/native/social/devices", body).await.map(|v| devices_from(&v))
}

pub async fn revoke(client: &Client, token: &str, device_id: &str) -> Result<(), Problem> {
    call(client, token, "/identity/native/social/device-revoke", json!({"device_id": device_id})).await.map(|_| ())
}

/// A sealed entry as one device received it.
#[derive(Clone, Debug, PartialEq)]
pub struct SealedEntry {
    pub id: String,
    pub from: String,
    pub sender_device: String,
    pub nonce: [u8; 12],
    pub ciphertext: Vec<u8>,
    /// This device's wrapped key (enc, wrapped), or None when the message was not sealed for it.
    pub key: Option<(Vec<u8>, Vec<u8>)>,
    pub sent_at: String,
}

/// One entry of a conversation as a device reads it.
#[derive(Clone, Debug, PartialEq)]
pub enum Entry {
    Text(Message),
    Sealed(SealedEntry),
}

fn b64(v: &Value, k: &str) -> Option<Vec<u8>> {
    URL_SAFE_NO_PAD.decode(v.get(k)?.as_str()?).ok()
}

impl Entry {
    pub fn from_json(v: &Value) -> Option<Entry> {
        match v.get("kind").and_then(Value::as_str) {
            Some("sealed") => {
                let key = match v.get("key") {
                    Some(k) if k.is_object() => Some((b64(k, "enc")?, b64(k, "wrapped")?)),
                    _ => None,
                };
                Some(Entry::Sealed(SealedEntry {
                    id: v.get("id")?.as_str()?.to_string(),
                    from: v.get("from")?.as_str()?.to_string(),
                    sender_device: v.get("sender_device")?.as_str()?.to_string(),
                    nonce: b64(v, "nonce")?.try_into().ok()?,
                    ciphertext: b64(v, "ciphertext")?,
                    key,
                    sent_at: v.get("sent_at").and_then(Value::as_str).unwrap_or_default().to_string(),
                }))
            }
            _ => Some(Entry::Text(Message::from_json(v))),
        }
    }

    pub fn id(&self) -> &str {
        match self {
            Entry::Text(m) => &m.id,
            Entry::Sealed(s) => &s.id,
        }
    }
}

/// The newest page of a conversation as this device reads it, oldest first, and whether older entries exist.
pub async fn conversation(client: &Client, token: &str, friend_id: &str, device_id: &str, before: Option<&str>) -> Result<(Vec<Entry>, bool), Problem> {
    let mut body = json!({"friend_id": friend_id, "device_id": device_id});
    if let Some(b) = before {
        body["before"] = json!(b);
    }
    let v = call(client, token, "/identity/native/social/messages", body).await?;
    let entries = v.get("messages").and_then(Value::as_array).map(|a| a.iter().filter_map(Entry::from_json).collect()).unwrap_or_default();
    Ok((entries, v.get("more").and_then(Value::as_bool).unwrap_or(false)))
}

/// A fresh message id: `msg_` and 22 random base64url characters (the envelope's associated data binds it).
pub fn new_message_id() -> Result<String, Problem> {
    let mut raw = [0u8; 16];
    SystemRandom::new().fill(&mut raw).map_err(|_| Problem::Refused("random".into(), "This PC's random source failed.".into()))?;
    Ok(format!("msg_{}", URL_SAFE_NO_PAD.encode(raw)))
}

fn body_for(friend_id: &str, message_id: &str, sender_device: &str, env: &Envelope) -> Value {
    json!({
        "friend_id": friend_id,
        "message_id": message_id,
        "sender_device": sender_device,
        "nonce": URL_SAFE_NO_PAD.encode(env.nonce),
        "ciphertext": URL_SAFE_NO_PAD.encode(&env.ciphertext),
        "keys": env.keys.iter().map(|k: &WrappedKey| json!({"device": k.device, "enc": URL_SAFE_NO_PAD.encode(&k.enc), "wrapped": URL_SAFE_NO_PAD.encode(&k.wrapped)})).collect::<Vec<_>>(),
    })
}

/// Seal `text` on this device for every live device of the friend and of this account, and send it. When the friend's
/// devices changed since they were read, seal again once for the list the service returned.
pub async fn send(client: &Client, token: &str, me: &DeviceKey, my_account: &str, friend_id: &str, text: &str) -> Result<Entry, Problem> {
    let mine = devices(client, token, None).await?;
    let mut theirs = devices(client, token, Some(friend_id)).await?;
    if theirs.is_empty() {
        return Err(Problem::Refused("no-devices".into(), "Your friend has no device that can read encrypted messages yet.".into()));
    }
    let message_id = new_message_id()?;
    let header = Header { sender_account: my_account.into(), recipient_account: friend_id.into(), sender_device: me.id(), message_id: message_id.clone() };
    for attempt in 0..2 {
        let targets: Vec<[u8; 32]> = theirs.iter().chain(mine.iter()).map(|d| d.public_key).collect();
        let env = sealed::seal(text, &header, me, &targets).map_err(|e| Problem::Refused("seal".into(), format!("This message could not be sealed ({e:?}).")))?;
        let (status, retry, raw) = client
            .post_raw("/identity/native/social/send-sealed", with_credentials(body_for(friend_id, &message_id, &me.id(), &env), token))
            .await
            .map_err(Problem::Unreachable)?;
        let v: Value = serde_json::from_str(&raw).unwrap_or(Value::Null);
        if status == 409 && v.get("error").and_then(Value::as_str) == Some("devices-changed") && attempt == 0 {
            theirs = devices_from(&v);
            continue;
        }
        let v = interpret(status, retry, &raw)?;
        return Entry::from_json(v.get("message").unwrap_or(&Value::Null)).ok_or_else(|| Problem::Refused("bad-message".into(), "The service answered without the message.".into()));
    }
    Err(Problem::Refused("devices-changed".into(), "Your friend's devices keep changing. Try again.".into()))
}

fn with_credentials(mut body: Value, token: &str) -> Value {
    body["client_id"] = json!(crate::client::CLIENT_ID);
    body["refresh_token"] = json!(token);
    body
}

/// What an entry says on this device.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Opened {
    Text(String),
    /// Sealed for other devices only (sent before this device existed, or to the person's other devices).
    NotForThisDevice,
    /// Its sending device is no longer listed (removed), so its seal cannot be checked.
    UnknownSender,
    /// It did not open: damaged, or not what its header says.
    Unopenable,
}

/// The public keys a sealed entry may come from, by device id: the friend's live devices and this account's, kept
/// apart so that an entry is checked against the devices of the account it says it is from.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Senders {
    pub theirs: HashMap<String, [u8; 32]>,
    pub mine: HashMap<String, [u8; 32]>,
}

impl Senders {
    pub fn new(theirs: &[Device], mine: &[Device]) -> Senders {
        let keys = |list: &[Device]| list.iter().map(|d| (d.id.clone(), d.public_key)).collect();
        Senders { theirs: keys(theirs), mine: keys(mine) }
    }
}

/// Open one entry. Its sending device must be one of the devices of the account the entry is from (`from`): the
/// friend's for the friend's messages, this account's for its own. A device of the other person is refused, so a
/// message sealed by the friend cannot be shown as this account's own, or the other way round.
pub fn open(entry: &Entry, me: &DeviceKey, my_account: &str, friend_id: &str, senders: &Senders) -> Opened {
    let s = match entry {
        Entry::Text(m) => return Opened::Text(m.text.clone()),
        Entry::Sealed(s) => s,
    };
    let Some((enc, wrapped)) = &s.key else { return Opened::NotForThisDevice };
    let pool = if s.from == my_account {
        &senders.mine
    } else if s.from == friend_id {
        &senders.theirs
    } else {
        return Opened::UnknownSender;
    };
    let Some(sender_public) = pool.get(&s.sender_device) else { return Opened::UnknownSender };
    let recipient = if s.from == my_account { friend_id } else { my_account };
    let header = Header { sender_account: s.from.clone(), recipient_account: recipient.into(), sender_device: s.sender_device.clone(), message_id: s.id.clone() };
    let env = Envelope { nonce: s.nonce, ciphertext: s.ciphertext.clone(), keys: vec![WrappedKey { device: me.id(), enc: enc.clone(), wrapped: wrapped.clone() }] };
    match sealed::open(&env, &header, me, sender_public) {
        Ok(text) => Opened::Text(text),
        Err(_) => Opened::Unopenable,
    }
}

/// Where the device ids already seen for each friend are kept (ids only, which are not secret):
/// `~/.alelyon/alelyon/friends-known-devices.json`, or the file `ALELYON_FRIENDS_KNOWN_DEVICES_FILE` names.
pub fn known_devices_path() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("ALELYON_FRIENDS_KNOWN_DEVICES_FILE").filter(|v| !v.is_empty()) {
        return Some(PathBuf::from(p));
    }
    std::env::var_os("USERPROFILE").map(|h| PathBuf::from(h).join(".alelyon").join("alelyon").join("friends-known-devices.json"))
}

/// Record a friend's listed devices and answer the ones that are new since last time. The first time a friend is seen
/// their devices are trusted as they are (trust on first use, ADR-0049 Decision 3) and nothing is answered.
pub fn remember_devices(at: &Path, friend_id: &str, listed: &[Device]) -> Vec<Device> {
    let mut all: serde_json::Map<String, Value> = std::fs::read_to_string(at).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default();
    let ids: Vec<String> = listed.iter().map(|d| d.id.clone()).collect();
    let fresh: Vec<Device> = match all.get(friend_id).and_then(Value::as_array) {
        Some(known) => {
            let known: Vec<String> = known.iter().filter_map(Value::as_str).map(str::to_string).collect();
            new_devices(listed, &known).into_iter().cloned().collect()
        }
        None => Vec::new(),
    };
    let previous: Vec<String> = all.get(friend_id).and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_str).map(str::to_string).collect()).unwrap_or_default();
    // Removed devices stay known, so one that comes back is not announced as new twice; ids only ever accumulate.
    let mut merged = previous;
    for id in ids {
        if !merged.contains(&id) {
            merged.push(id);
        }
    }
    all.insert(friend_id.into(), json!(merged));
    if let Some(dir) = at.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let _ = std::fs::write(at, serde_json::to_string_pretty(&Value::Object(all)).unwrap_or_default());
    fresh
}

/// Devices of the friend that were not there before (`known` ids), for "added a new device" notices.
pub fn new_devices<'a>(listed: &'a [Device], known: &[String]) -> Vec<&'a Device> {
    listed.iter().filter(|d| !known.contains(&d.id)).collect()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn unhex32(s: &str) -> Option<[u8; 32]> {
    if s.len() != 64 {
        return None;
    }
    let v: Option<Vec<u8>> = (0..64).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).ok()).collect();
    v?.try_into().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn people() -> (DeviceKey, DeviceKey) {
        (DeviceKey::from_secret([1u8; 32]), DeviceKey::from_secret([3u8; 32]))
    }

    fn sealed_entry(env: &Envelope, header: &Header, for_device: &str) -> Entry {
        let key = env.keys.iter().find(|k| k.device == for_device).map(|k| (k.enc.clone(), k.wrapped.clone()));
        Entry::Sealed(SealedEntry { id: header.message_id.clone(), from: header.sender_account.clone(), sender_device: header.sender_device.clone(), nonce: env.nonce, ciphertext: env.ciphertext.clone(), key, sent_at: String::new() })
    }

    #[test]
    fn a_sealed_entry_opens_for_the_recipient_and_for_the_sender_and_says_why_when_not() {
        let (ada, bob) = people();
        let header = Header { sender_account: "acct_ada".into(), recipient_account: "acct_bob".into(), sender_device: ada.id(), message_id: "msg_aaaaaaaaaaaaaaaaaaaaaa".into() };
        let env = sealed::seal("hello Bob", &header, &ada, &[bob.public(), ada.public()]).unwrap();
        let senders = Senders { theirs: [(ada.id(), ada.public())].into(), mine: [(bob.id(), bob.public())].into() };
        let as_ada = Senders { theirs: senders.mine.clone(), mine: senders.theirs.clone() };
        assert_eq!(open(&sealed_entry(&env, &header, &bob.id()), &bob, "acct_bob", "acct_ada", &senders), Opened::Text("hello Bob".into()));
        assert_eq!(open(&sealed_entry(&env, &header, &ada.id()), &ada, "acct_ada", "acct_bob", &as_ada), Opened::Text("hello Bob".into()), "the sender's own copy");
        assert_eq!(open(&sealed_entry(&env, &header, "nobody"), &bob, "acct_bob", "acct_ada", &senders), Opened::NotForThisDevice);
        assert_eq!(open(&sealed_entry(&env, &header, &bob.id()), &bob, "acct_bob", "acct_ada", &Senders::default()), Opened::UnknownSender);
        let mut wrong = sealed_entry(&env, &header, &bob.id());
        if let Entry::Sealed(s) = &mut wrong {
            s.id = "msg_bbbbbbbbbbbbbbbbbbbbbb".into(); // another message's id: the associated data no longer matches
        }
        assert_eq!(open(&wrong, &bob, "acct_bob", "acct_ada", &senders), Opened::Unopenable);
    }

    #[test]
    fn an_entry_sealed_by_the_friend_cannot_pass_as_this_accounts_own_or_the_other_way_round() {
        // The review pack's Q2: Ada seals naming Bob as the sender and herself as the recipient, and a service hands
        // it to Bob marked as from Bob. The header Bob rebuilds matches, so only the device check stops it.
        let (ada, bob) = people();
        let forged = Header { sender_account: "acct_bob".into(), recipient_account: "acct_ada".into(), sender_device: ada.id(), message_id: "msg_cccccccccccccccccccccc".into() };
        let env = sealed::seal("I never wrote this", &forged, &ada, &[bob.public()]).unwrap();
        let senders = Senders { theirs: [(ada.id(), ada.public())].into(), mine: [(bob.id(), bob.public())].into() };
        assert_eq!(open(&sealed_entry(&env, &forged, &bob.id()), &bob, "acct_bob", "acct_ada", &senders), Opened::UnknownSender);
        let merged = Senders { theirs: senders.theirs.clone(), mine: senders.theirs.clone() };
        assert_eq!(
            open(&sealed_entry(&env, &forged, &bob.id()), &bob, "acct_bob", "acct_ada", &merged),
            Opened::Text("I never wrote this".into()),
            "positive control: with Ada's device among Bob's own, as the merged map had it, it opened"
        );
        // An entry from a third account opens against nobody's devices.
        let mut stranger = sealed_entry(&env, &forged, &bob.id());
        if let Entry::Sealed(s) = &mut stranger {
            s.from = "acct_eve".into();
        }
        assert_eq!(open(&stranger, &bob, "acct_bob", "acct_ada", &merged), Opened::UnknownSender);
    }

    #[test]
    fn entries_parse_both_kinds_and_a_device_list_that_disagrees_with_its_keys_is_dropped() {
        let (ada, _) = people();
        let good = json!({"id": ada.id(), "public_key": URL_SAFE_NO_PAD.encode(ada.public()), "created_at": "t"});
        let lying = json!({"id": "0".repeat(32), "public_key": URL_SAFE_NO_PAD.encode(ada.public())});
        let list = devices_from(&json!({"devices": [good, lying]}));
        assert_eq!(list.len(), 1, "a device whose id is not its key's is not trusted");
        let text = Entry::from_json(&json!({"kind": "text", "id": "m1", "from": "a", "text": "hi", "sent_at": "t"})).unwrap();
        assert!(matches!(text, Entry::Text(ref m) if m.text == "hi"));
        let sealed = Entry::from_json(&json!({"kind": "sealed", "id": "m2", "from": "a", "sender_device": "d", "nonce": URL_SAFE_NO_PAD.encode([1u8; 12]), "ciphertext": URL_SAFE_NO_PAD.encode([2u8; 20]), "key": null, "sent_at": "t"})).unwrap();
        assert!(matches!(sealed, Entry::Sealed(ref s) if s.key.is_none() && s.ciphertext.len() == 20));
        assert!(Entry::from_json(&json!({"kind": "sealed", "id": "m3"})).is_none(), "a sealed entry missing fields is dropped");
    }

    #[test]
    fn message_ids_meet_the_contract_and_new_devices_are_told_apart() {
        let id = new_message_id().unwrap();
        assert!(id.starts_with("msg_") && id.len() == 26 && id[4..].chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'));
        assert_ne!(id, new_message_id().unwrap());
        let (ada, bob) = people();
        let listed = vec![
            Device { id: ada.id(), public_key: ada.public(), created_at: String::new() },
            Device { id: bob.id(), public_key: bob.public(), created_at: String::new() },
        ];
        let fresh = new_devices(&listed, &[ada.id()]);
        assert_eq!(fresh.len(), 1);
        assert_eq!(fresh[0].id, bob.id());
    }

    #[test]
    fn known_devices_trust_the_first_list_and_announce_only_later_additions() {
        let dir = std::env::temp_dir().join(format!("alelyon-known-devices-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let at = dir.join("known.json");
        let (ada, bob) = people();
        let one = Device { id: ada.id(), public_key: ada.public(), created_at: String::new() };
        let two = Device { id: bob.id(), public_key: bob.public(), created_at: String::new() };
        assert!(remember_devices(&at, "acct_ada", std::slice::from_ref(&one)).is_empty(), "first sight: trusted, no notice");
        assert!(remember_devices(&at, "acct_ada", std::slice::from_ref(&one)).is_empty());
        assert_eq!(remember_devices(&at, "acct_ada", &[one.clone(), two.clone()]), vec![two.clone()], "a later addition is announced");
        assert!(remember_devices(&at, "acct_ada", &[one.clone(), two.clone()]).is_empty(), "and only once");
        assert!(remember_devices(&at, "acct_ada", std::slice::from_ref(&two)).is_empty(), "a removal is not an addition");
        assert!(remember_devices(&at, "acct_bob", &[two]).is_empty(), "each friend has their own first sight");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(windows)]
    #[test]
    fn the_device_key_is_kept_sealed_and_comes_back_and_a_replacement_is_new() {
        let dir = std::env::temp_dir().join(format!("alelyon-device-key-{}", std::process::id()));
        let at = dir.join("friends-device.sealed");
        let _ = std::fs::remove_dir_all(&dir);
        let first = load_or_create_device_key(&at).unwrap();
        let again = load_or_create_device_key(&at).unwrap();
        assert_eq!(first.public(), again.public(), "the kept key comes back");
        let raw = std::fs::read(&at).unwrap();
        assert!(!raw.windows(32).any(|w| w == first.secret()), "the file does not hold the secret in the clear");
        assert!(vault::open(&at).is_none(), "the session's label does not open the device key");
        let replaced = replace_device_key(&at).unwrap();
        assert_ne!(replaced.public(), first.public());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
