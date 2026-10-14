//! Sealed friends messages, protocol `alelyon.friends.sealed/v1` (ADR-0049, accepted 2026-10-10).
//!
//! Nothing calls this yet: it is Phase 1, the envelope both clients share. The website implements the same envelope in
//! `www/src/scripts/friends/sealed.ts`; `tests/vectors/sealed-v1.json` holds envelopes each side made for the other to
//! open, and both sides check HPKE against RFC 9180's published vectors.
//!
//! A message is encrypted once with AES-256-GCM under a fresh random content key. The content key is wrapped for each
//! device that may read the message (the recipient's devices and the sender's other devices) with HPKE (RFC 9180) in
//! Auth mode: DHKEM(X25519, HKDF-SHA256), HKDF-SHA256, AES-128-GCM, single-shot. Auth mode proves to the recipient
//! which device sent the message. The associated data binds the protocol, both accounts, the sending device and the
//! message id, so an envelope cannot be moved to another conversation or replayed as another message.
//!
//! What this does NOT give (ADR-0049): forward secrecy (a stolen device key opens every past message wrapped for it),
//! hiding metadata (who, when, how long), or protection from a service that hands out a false device key.

use curve25519_dalek::montgomery::MontgomeryPoint;
use hkdf::Hkdf;
use ring::aead::{AES_128_GCM, AES_256_GCM, Aad, LessSafeKey, Nonce, UnboundKey};
use ring::rand::{SecureRandom, SystemRandom};
use sha2::{Digest, Sha256};
use zeroize::Zeroizing;

/// The protocol's name, first in every associated data and HPKE info.
pub const PROTOCOL: &str = "alelyon.friends.sealed/v1";
/// The HPKE `info` every content-key wrap uses.
pub const WRAP_INFO: &[u8] = b"alelyon.friends.sealed/v1 content key";
/// The most devices one message is wrapped for (both people's together).
pub const MAX_DEVICES: usize = 32;
/// The longest message, in bytes of UTF-8 (2000 characters of up to four bytes each).
pub const MAX_TEXT_BYTES: usize = 8000;

const KEM_ID: u16 = 0x0020;
const KDF_ID: u16 = 0x0001;
const AEAD_ID: u16 = 0x0001;
const MODE_AUTH: u8 = 0x02;
const NK: usize = 16;
const NN: usize = 12;
const NH: usize = 32;

/// Why something could not be sealed or opened. Deliberately coarse: an opener learns only that it failed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SealError {
    /// A field is empty, too long, or holds a NUL; or too many devices.
    Invalid(&'static str),
    /// No wrapped key for this device, a wrong key, changed associated data, or damaged bytes.
    Unopenable,
    /// The system's random source failed.
    Random,
}

/// A device's X25519 key pair. The secret is wiped when dropped.
pub struct DeviceKey {
    secret: Zeroizing<[u8; 32]>,
    public: [u8; 32],
}

impl std::fmt::Debug for DeviceKey {
    /// The device's id only: the secret never reaches a log.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DeviceKey({})", self.id())
    }
}

impl DeviceKey {
    /// A new random device key.
    pub fn generate() -> Result<DeviceKey, SealError> {
        let mut secret = Zeroizing::new([0u8; 32]);
        SystemRandom::new().fill(secret.as_mut()).map_err(|_| SealError::Random)?;
        Ok(DeviceKey::from_secret(*secret))
    }

    /// The key pair for a stored secret (the vault's copy).
    pub fn from_secret(secret: [u8; 32]) -> DeviceKey {
        let public = MontgomeryPoint::mul_base_clamped(secret).to_bytes();
        DeviceKey { secret: Zeroizing::new(secret), public }
    }

    pub fn public(&self) -> [u8; 32] {
        self.public
    }

    /// The secret, for the vault to seal. Handle with care.
    pub fn secret(&self) -> &[u8; 32] {
        &self.secret
    }

    /// The device's id as the service lists it: the first 16 bytes of SHA-256 over the protocol's device label and the
    /// public key, in lowercase hex.
    pub fn id(&self) -> String {
        device_id(&self.public)
    }
}

/// The id of the device whose public key this is (`DeviceKey::id`).
pub fn device_id(public: &[u8; 32]) -> String {
    let mut h = Sha256::new();
    h.update(b"alelyon.friends.device/v1\0");
    h.update(public);
    hex(&h.finalize()[..16])
}

/// What the associated data binds: who sent the message to whom, from which device, under which id.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Header {
    pub sender_account: String,
    pub recipient_account: String,
    pub sender_device: String,
    pub message_id: String,
}

impl Header {
    /// The associated data: the protocol and each field, each as a two-byte big-endian length and its UTF-8 bytes.
    pub fn aad(&self) -> Result<Vec<u8>, SealError> {
        let mut out = Vec::new();
        for field in [PROTOCOL, &self.sender_account, &self.recipient_account, &self.sender_device, &self.message_id] {
            if field.is_empty() || field.len() > 256 || field.contains('\0') {
                return Err(SealError::Invalid("a header field is empty, longer than 256 bytes, or holds a NUL"));
            }
            out.extend_from_slice(&(field.len() as u16).to_be_bytes());
            out.extend_from_slice(field.as_bytes());
        }
        Ok(out)
    }
}

/// One device's copy of the content key.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WrappedKey {
    pub device: String,
    /// HPKE's encapsulated key (the ephemeral public key), 32 bytes.
    pub enc: Vec<u8>,
    /// The content key sealed for that device, 32 + 16 bytes.
    pub wrapped: Vec<u8>,
}

/// A sealed message as it travels.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Envelope {
    pub nonce: [u8; 12],
    pub ciphertext: Vec<u8>,
    pub keys: Vec<WrappedKey>,
}

/// A source of randomness, injectable so tests can make the same envelope twice.
pub trait Random {
    fn fill(&mut self, out: &mut [u8]) -> Result<(), SealError>;
}

/// The operating system's random source.
pub struct SystemRandomSource(SystemRandom);

impl Default for SystemRandomSource {
    fn default() -> Self {
        SystemRandomSource(SystemRandom::new())
    }
}

impl Random for SystemRandomSource {
    fn fill(&mut self, out: &mut [u8]) -> Result<(), SealError> {
        self.0.fill(out).map_err(|_| SealError::Random)
    }
}

/// Seal `text` from `sender` (this device) for each device in `devices` (public keys).
pub fn seal(text: &str, header: &Header, sender: &DeviceKey, devices: &[[u8; 32]]) -> Result<Envelope, SealError> {
    seal_with(text, header, sender, devices, &mut SystemRandomSource::default())
}

pub fn seal_with(text: &str, header: &Header, sender: &DeviceKey, devices: &[[u8; 32]], random: &mut dyn Random) -> Result<Envelope, SealError> {
    if text.is_empty() || text.len() > MAX_TEXT_BYTES {
        return Err(SealError::Invalid("the text is empty or too long"));
    }
    if devices.is_empty() || devices.len() > MAX_DEVICES {
        return Err(SealError::Invalid("a message is sealed for 1 to 32 devices"));
    }
    if header.sender_device != sender.id() {
        return Err(SealError::Invalid("the header names another sending device"));
    }
    let aad = header.aad()?;
    let mut content_key = Zeroizing::new([0u8; 32]);
    random.fill(content_key.as_mut())?;
    let mut nonce = [0u8; 12];
    random.fill(&mut nonce)?;
    let ciphertext = aead_seal(&AES_256_GCM, content_key.as_ref(), &nonce, &aad, text.as_bytes())?;
    let mut keys = Vec::with_capacity(devices.len());
    for public in devices {
        let mut ephemeral = Zeroizing::new([0u8; 32]);
        random.fill(ephemeral.as_mut())?;
        let (enc, wrapped) = hpke_seal_auth(public, sender, &ephemeral, WRAP_INFO, &aad, content_key.as_ref())?;
        keys.push(WrappedKey { device: device_id(public), enc: enc.to_vec(), wrapped });
    }
    Ok(Envelope { nonce, ciphertext, keys })
}

/// Open an envelope on this device (`me`), sent by the device whose public key is `sender_public`.
pub fn open(envelope: &Envelope, header: &Header, me: &DeviceKey, sender_public: &[u8; 32]) -> Result<String, SealError> {
    if header.sender_device != device_id(sender_public) {
        return Err(SealError::Unopenable);
    }
    let aad = header.aad().map_err(|_| SealError::Unopenable)?;
    let mine = me.id();
    let key = envelope.keys.iter().find(|k| k.device == mine).ok_or(SealError::Unopenable)?;
    let enc: [u8; 32] = key.enc.as_slice().try_into().map_err(|_| SealError::Unopenable)?;
    let content_key = Zeroizing::new(hpke_open_auth(&enc, me, sender_public, WRAP_INFO, &aad, &key.wrapped)?);
    if content_key.len() != 32 {
        return Err(SealError::Unopenable);
    }
    let plain = aead_open(&AES_256_GCM, &content_key, &envelope.nonce, &aad, &envelope.ciphertext)?;
    String::from_utf8(plain).map_err(|_| SealError::Unopenable)
}

// ------------------------------------------------------------------- HPKE (RFC 9180), the one suite and mode used

fn i2osp2(v: u16) -> [u8; 2] {
    v.to_be_bytes()
}

fn kem_suite() -> Vec<u8> {
    [b"KEM".as_slice(), &i2osp2(KEM_ID)].concat()
}

fn hpke_suite() -> Vec<u8> {
    [b"HPKE".as_slice(), &i2osp2(KEM_ID), &i2osp2(KDF_ID), &i2osp2(AEAD_ID)].concat()
}

fn labeled_extract(suite: &[u8], salt: &[u8], label: &[u8], ikm: &[u8]) -> [u8; NH] {
    let labeled = [b"HPKE-v1".as_slice(), suite, label, ikm].concat();
    let (prk, _) = Hkdf::<Sha256>::extract(Some(salt), &labeled);
    prk.into()
}

fn labeled_expand(suite: &[u8], prk: &[u8; NH], label: &[u8], info: &[u8], len: usize) -> Vec<u8> {
    let labeled = [&i2osp2(len as u16)[..], b"HPKE-v1", suite, label, info].concat();
    let mut out = vec![0u8; len];
    Hkdf::<Sha256>::from_prk(prk).expect("a 32-byte PRK").expand(&labeled, &mut out).expect("a short output");
    out
}

fn dh(secret: &[u8; 32], public: &[u8; 32]) -> Result<[u8; 32], SealError> {
    let shared = MontgomeryPoint(*public).mul_clamped(*secret).to_bytes();
    // RFC 9180 7.1.4: an all-zero result (a low-order public key) is refused.
    if shared.iter().all(|b| *b == 0) {
        return Err(SealError::Unopenable);
    }
    Ok(shared)
}

fn extract_and_expand(dh: &[u8], kem_context: &[u8]) -> [u8; NH] {
    let suite = kem_suite();
    let prk = labeled_extract(&suite, b"", b"eae_prk", dh);
    labeled_expand(&suite, &prk, b"shared_secret", kem_context, NH).try_into().expect("32 bytes")
}

/// The key schedule for Auth mode (no PSK): the AEAD key and base nonce.
fn key_schedule_auth(shared_secret: &[u8; NH], info: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let suite = hpke_suite();
    let psk_id_hash = labeled_extract(&suite, b"", b"psk_id_hash", b"");
    let info_hash = labeled_extract(&suite, b"", b"info_hash", info);
    let context = [&[MODE_AUTH][..], &psk_id_hash, &info_hash].concat();
    let secret = labeled_extract(&suite, shared_secret, b"secret", b"");
    (labeled_expand(&suite, &secret, b"key", &context, NK), labeled_expand(&suite, &secret, b"base_nonce", &context, NN))
}

/// SealAuth, single-shot (sequence number 0), with the ephemeral secret given. Answers (enc, ciphertext).
fn hpke_seal_auth(recipient: &[u8; 32], sender: &DeviceKey, ephemeral: &[u8; 32], info: &[u8], aad: &[u8], pt: &[u8]) -> Result<([u8; 32], Vec<u8>), SealError> {
    let enc = MontgomeryPoint::mul_base_clamped(*ephemeral).to_bytes();
    let dh_bytes = Zeroizing::new([dh(ephemeral, recipient)?, dh(sender.secret(), recipient)?].concat());
    let kem_context = [&enc[..], recipient, &sender.public()].concat();
    let shared = Zeroizing::new(extract_and_expand(&dh_bytes, &kem_context));
    let (key, nonce) = key_schedule_auth(&shared, info);
    let nonce: [u8; NN] = nonce.try_into().expect("12 bytes");
    Ok((enc, aead_seal(&AES_128_GCM, &key, &nonce, aad, pt)?))
}

/// OpenAuth, single-shot.
fn hpke_open_auth(enc: &[u8; 32], me: &DeviceKey, sender: &[u8; 32], info: &[u8], aad: &[u8], ct: &[u8]) -> Result<Vec<u8>, SealError> {
    let dh_bytes = Zeroizing::new([dh(me.secret(), enc)?, dh(me.secret(), sender)?].concat());
    let kem_context = [&enc[..], &me.public(), sender].concat();
    let shared = Zeroizing::new(extract_and_expand(&dh_bytes, &kem_context));
    let (key, nonce) = key_schedule_auth(&shared, info);
    let nonce: [u8; NN] = nonce.try_into().expect("12 bytes");
    aead_open(&AES_128_GCM, &key, &nonce, aad, ct)
}

fn aead_seal(alg: &'static ring::aead::Algorithm, key: &[u8], nonce: &[u8; 12], aad: &[u8], pt: &[u8]) -> Result<Vec<u8>, SealError> {
    let key = LessSafeKey::new(UnboundKey::new(alg, key).map_err(|_| SealError::Invalid("an AEAD key of the wrong length"))?);
    let mut buf = pt.to_vec();
    key.seal_in_place_append_tag(Nonce::assume_unique_for_key(*nonce), Aad::from(aad), &mut buf)
        .map_err(|_| SealError::Invalid("the AEAD refused to seal"))?;
    Ok(buf)
}

fn aead_open(alg: &'static ring::aead::Algorithm, key: &[u8], nonce: &[u8; 12], aad: &[u8], ct: &[u8]) -> Result<Vec<u8>, SealError> {
    let key = LessSafeKey::new(UnboundKey::new(alg, key).map_err(|_| SealError::Unopenable)?);
    let mut buf = ct.to_vec();
    let plain = key.open_in_place(Nonce::assume_unique_for_key(*nonce), Aad::from(aad), &mut buf).map_err(|_| SealError::Unopenable)?;
    Ok(plain.to_vec())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn unhex(s: &str) -> Vec<u8> {
        (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
    }

    fn arr(s: &str) -> [u8; 32] {
        unhex(s).try_into().unwrap()
    }

    fn rfc() -> Value {
        serde_json::from_str(include_str!("../tests/vectors/rfc9180-a13-auth.json")).unwrap()
    }

    /// Randomness replayed from a list, so an envelope can be made again byte for byte.
    struct Replay(Vec<Vec<u8>>);
    impl Random for Replay {
        fn fill(&mut self, out: &mut [u8]) -> Result<(), SealError> {
            let next = self.0.remove(0);
            assert_eq!(next.len(), out.len(), "the replayed draw has the size asked for");
            out.copy_from_slice(&next);
            Ok(())
        }
    }

    #[test]
    fn hpke_auth_mode_matches_rfc_9180_appendix_a_1_3() {
        let v = rfc();
        let s = |k: &str| v[k].as_str().unwrap().to_string();
        let sender = DeviceKey::from_secret(arr(&s("skSm")));
        let recipient = DeviceKey::from_secret(arr(&s("skRm")));
        assert_eq!(hex(&sender.public()), s("pkSm"), "a secret's public key, as the RFC derives it");
        assert_eq!(hex(&recipient.public()), s("pkRm"));
        let e0 = &v["encryptions"][0];
        let (enc, ct) = hpke_seal_auth(&recipient.public(), &sender, &arr(&s("skEm")), &unhex(&s("info")), &unhex(e0["aad"].as_str().unwrap()), &unhex(e0["pt"].as_str().unwrap())).unwrap();
        assert_eq!(hex(&enc), s("enc"));
        assert_eq!(hex(&ct), e0["ct"].as_str().unwrap(), "sequence 0's ciphertext");
        let pt = hpke_open_auth(&enc, &recipient, &sender.public(), &unhex(&s("info")), &unhex(e0["aad"].as_str().unwrap()), &ct).unwrap();
        assert_eq!(hex(&pt), e0["pt"].as_str().unwrap());
        // the intermediate values, so a mismatch says where it starts
        let dh_bytes = [dh(&arr(&s("skEm")), &recipient.public()).unwrap(), dh(sender.secret(), &recipient.public()).unwrap()].concat();
        let kem_context = [&enc[..], &recipient.public(), &sender.public()].concat();
        let shared = extract_and_expand(&dh_bytes, &kem_context);
        assert_eq!(hex(&shared), s("shared_secret"));
        let (key, nonce) = key_schedule_auth(&shared, &unhex(&s("info")));
        assert_eq!((hex(&key), hex(&nonce)), (s("key"), s("base_nonce")));
    }

    #[test]
    fn an_auth_mode_wrap_opens_only_with_the_right_keys() {
        let v = rfc();
        let s = |k: &str| v[k].as_str().unwrap().to_string();
        let sender = DeviceKey::from_secret(arr(&s("skSm")));
        let recipient = DeviceKey::from_secret(arr(&s("skRm")));
        let other = DeviceKey::from_secret([7u8; 32]);
        let (enc, ct) = hpke_seal_auth(&recipient.public(), &sender, &[9u8; 32], b"info", b"aad", b"content key").unwrap();
        assert!(hpke_open_auth(&enc, &recipient, &sender.public(), b"info", b"aad", &ct).is_ok());
        assert_eq!(hpke_open_auth(&enc, &other, &sender.public(), b"info", b"aad", &ct), Err(SealError::Unopenable), "another device");
        assert_eq!(hpke_open_auth(&enc, &recipient, &other.public(), b"info", b"aad", &ct), Err(SealError::Unopenable), "another sender");
        assert_eq!(hpke_open_auth(&enc, &recipient, &sender.public(), b"info", b"aaD", &ct), Err(SealError::Unopenable), "other associated data");
        assert_eq!(hpke_open_auth(&enc, &recipient, &sender.public(), b"inFo", b"aad", &ct), Err(SealError::Unopenable), "other info");
        assert_eq!(hpke_open_auth(&[0u8; 32], &recipient, &sender.public(), b"info", b"aad", &ct), Err(SealError::Unopenable), "a low-order enc");
    }

    fn people() -> (DeviceKey, DeviceKey, DeviceKey, Header) {
        let ada_pc = DeviceKey::from_secret([1u8; 32]);
        let ada_web = DeviceKey::from_secret([2u8; 32]);
        let bob_pc = DeviceKey::from_secret([3u8; 32]);
        let header = Header {
            sender_account: "acct_ada".into(),
            recipient_account: "acct_bob".into(),
            sender_device: ada_pc.id(),
            message_id: "msg_1".into(),
        };
        (ada_pc, ada_web, bob_pc, header)
    }

    #[test]
    fn a_sealed_message_opens_on_every_device_it_was_sealed_for_and_nowhere_else() {
        let (ada_pc, ada_web, bob_pc, header) = people();
        let stranger = DeviceKey::generate().unwrap();
        let env = seal("hello, Bob ✓", &header, &ada_pc, &[bob_pc.public(), ada_web.public()]).unwrap();
        assert_eq!(open(&env, &header, &bob_pc, &ada_pc.public()).unwrap(), "hello, Bob ✓");
        assert_eq!(open(&env, &header, &ada_web, &ada_pc.public()).unwrap(), "hello, Bob ✓", "the sender's other device");
        assert_eq!(open(&env, &header, &stranger, &ada_pc.public()), Err(SealError::Unopenable));
        assert!(!env.ciphertext.windows(5).any(|w| w == b"hello"), "no plain text in the envelope");
    }

    #[test]
    fn a_changed_header_or_damaged_bytes_do_not_open() {
        let (ada_pc, _ada_web, bob_pc, header) = people();
        let env = seal("hello", &header, &ada_pc, &[bob_pc.public()]).unwrap();
        for changed in [
            Header { message_id: "msg_2".into(), ..header.clone() },
            Header { recipient_account: "acct_carol".into(), ..header.clone() },
            Header { sender_account: "acct_eve".into(), ..header.clone() },
        ] {
            assert_eq!(open(&env, &changed, &bob_pc, &ada_pc.public()), Err(SealError::Unopenable), "{changed:?}");
        }
        let mut truncated = env.clone();
        truncated.ciphertext.pop();
        assert_eq!(open(&truncated, &header, &bob_pc, &ada_pc.public()), Err(SealError::Unopenable));
        let mut flipped = env.clone();
        flipped.keys[0].wrapped[0] ^= 1;
        assert_eq!(open(&flipped, &header, &bob_pc, &ada_pc.public()), Err(SealError::Unopenable));
        let impostor = DeviceKey::from_secret([4u8; 32]);
        assert_eq!(open(&env, &header, &bob_pc, &impostor.public()), Err(SealError::Unopenable), "a sender key the header does not name");
    }

    #[test]
    fn sealing_refuses_bad_input() {
        let (ada_pc, _w, bob_pc, header) = people();
        assert!(matches!(seal("", &header, &ada_pc, &[bob_pc.public()]), Err(SealError::Invalid(_))));
        assert!(matches!(seal(&"x".repeat(MAX_TEXT_BYTES + 1), &header, &ada_pc, &[bob_pc.public()]), Err(SealError::Invalid(_))));
        assert!(matches!(seal("hi", &header, &ada_pc, &[]), Err(SealError::Invalid(_))));
        assert!(matches!(seal("hi", &header, &ada_pc, &vec![bob_pc.public(); MAX_DEVICES + 1]), Err(SealError::Invalid(_))));
        assert!(matches!(seal("hi", &Header { sender_device: "someone-else".into(), ..header.clone() }, &ada_pc, &[bob_pc.public()]), Err(SealError::Invalid(_))));
        assert!(matches!(seal("hi", &Header { message_id: "a\0b".into(), ..header }, &ada_pc, &[bob_pc.public()]), Err(SealError::Invalid(_))));
    }

    // ---------------------------------------------------------------- cross-implementation vectors
    //
    // tests/vectors/sealed-v1.json was written by the website's implementation (www/scripts/gen-sealed-vectors.mjs)
    // from fixed keys and fixed "random" draws. Making the same envelopes here from the same draws, byte for byte, and
    // opening them, is the evidence the two implementations agree.

    const VECTORS: [&str; 2] = ["sealed_a", "sealed_b"];

    fn bytes(v: &Value) -> Vec<u8> {
        unhex(v.as_str().unwrap())
    }

    fn vector_header(v: &Value) -> Header {
        Header {
            sender_account: v["sender_account"].as_str().unwrap().into(),
            recipient_account: v["recipient_account"].as_str().unwrap().into(),
            sender_device: v["sender_device"].as_str().unwrap().into(),
            message_id: v["message_id"].as_str().unwrap().into(),
        }
    }

    fn vector_envelope(v: &Value) -> Envelope {
        Envelope {
            nonce: bytes(&v["nonce"]).try_into().unwrap(),
            ciphertext: bytes(&v["ciphertext"]),
            keys: v["keys"].as_array().unwrap().iter().map(|k| WrappedKey { device: k["device"].as_str().unwrap().into(), enc: bytes(&k["enc"]), wrapped: bytes(&k["wrapped"]) }).collect(),
        }
    }

    fn cross() -> Value {
        serde_json::from_str(include_str!("../tests/vectors/sealed-v1.json")).unwrap()
    }

    #[test]
    fn this_side_makes_the_websites_envelopes_byte_for_byte_from_the_same_draws() {
        let v = cross();
        assert_eq!(v["protocol"].as_str(), Some(PROTOCOL));
        for name in VECTORS {
            let made = &v[name];
            let sender = DeviceKey::from_secret(arr(made["sender_secret"].as_str().unwrap()));
            let devices: Vec<[u8; 32]> = made["recipient_secrets"].as_array().unwrap().iter().map(|s| DeviceKey::from_secret(arr(s.as_str().unwrap())).public()).collect();
            let draws: Vec<Vec<u8>> = made["random_draws"].as_array().unwrap().iter().map(bytes).collect();
            let env = seal_with(made["text"].as_str().unwrap(), &vector_header(made), &sender, &devices, &mut Replay(draws)).unwrap();
            assert_eq!(env, vector_envelope(made), "{name}");
        }
    }

    #[test]
    fn this_side_opens_the_websites_envelopes_on_every_device_they_name() {
        let v = cross();
        for name in VECTORS {
            let made = &v[name];
            let sender = DeviceKey::from_secret(arr(made["sender_secret"].as_str().unwrap()));
            for secret in made["recipient_secrets"].as_array().unwrap() {
                let me = DeviceKey::from_secret(arr(secret.as_str().unwrap()));
                let text = open(&vector_envelope(made), &vector_header(made), &me, &sender.public()).unwrap();
                assert_eq!(text, made["text"].as_str().unwrap(), "{name}");
            }
        }
    }
}
