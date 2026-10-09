//! A model client, built and its failures worded: shared by agent runs and
//! chat (the native chat's spec §2.9 "shared now", §3.3.5 step 5; row C6 of
//! the chat core's spec). Extracted from the run manager with no change
//! in behaviour; the run manager's tests pin it unchanged.
//!
//! - [`build`]: every client is made through [`choices::client_config`],
//!   which reads the endpoint's key and sets `local`, so a client for an
//!   endpoint on this machine bypasses proxies (the chat spec's rule N6).
//!   A [`ModelFactory`] may hand back a different model for the
//!   configuration (tests keep every call off the network with it); without
//!   one, or when it declines, the client is the Chat Completions model.
//! - [`error_sentence`] and [`refusal_sentence`]: one sentence for a failed
//!   model call, never the transport's text: no response body, no key, and
//!   no address beyond the base URL with its credentials, query and fragment
//!   removed; a refusal is quoted, cut to [`REFUSAL_CHARS`].
//!
//! Nothing here touches the network: building a client opens no connection.

use std::sync::Arc;

use lattice_agents::{
    ChatCompletionsConfig, ChatCompletionsModel, Model, ModelError, sanitize_url_for_trace,
};
use lattice_protocol::{Refusal, RefusalKind};

use crate::bound::cap_text;
use crate::choices;
use crate::env::Env;
use crate::keys::KeyStore;
use crate::llama::ManagedEndpoint;
use crate::registry::ModelEndpoint;

/// A hook that may give a client a different model than the one its
/// configuration describes. It is offered each real endpoint's client
/// configuration and returns `None` to use the real client.
pub type ModelFactory = Arc<dyn Fn(&ChatCompletionsConfig) -> Option<Arc<dyn Model>> + Send + Sync>;

/// The longest a model's refusal may be when an error quotes it.
pub const REFUSAL_CHARS: usize = 500;

/// What an endpoint's address that the client cannot use says.
pub const ADDRESS_REFUSAL: &str = "That model's address could not be used.";

/// A client, and the base URL its errors may name.
pub struct Built {
    pub model: Arc<dyn Model>,
    pub base_url: String,
}

/// The client for `endpoint`, with its key read now.
pub fn build(
    endpoint: &ModelEndpoint,
    env: &dyn Env,
    keys: &KeyStore,
    factory: Option<&ModelFactory>,
) -> Result<Built, Refusal> {
    let config = choices::client_config(endpoint, env, keys);
    let base_url = config.base_url.clone();
    let model = match factory.and_then(|factory| factory(&config)) {
        Some(model) => model,
        None => Arc::new(
            ChatCompletionsModel::new(config)
                .map_err(|_| Refusal::new(RefusalKind::Invalid, ADDRESS_REFUSAL))?,
        ),
    };
    Ok(Built { model, base_url })
}

/// The client for the core's managed llama.cpp server (spec §22 LR2),
/// through [`choices::managed_client_config`] and the same factory hook.
pub fn build_managed(
    endpoint: &ManagedEndpoint,
    factory: Option<&ModelFactory>,
) -> Result<Built, Refusal> {
    let config = choices::managed_client_config(endpoint);
    let base_url = config.base_url.clone();
    let model = match factory.and_then(|factory| factory(&config)) {
        Some(model) => model,
        None => Arc::new(
            ChatCompletionsModel::new(config)
                .map_err(|_| Refusal::new(RefusalKind::Invalid, ADDRESS_REFUSAL))?,
        ),
    };
    Ok(Built { model, base_url })
}

/// One sentence for a failed model call. `base_url` is the client's address,
/// named only as a trace shows it.
pub fn error_sentence(error: &ModelError, base_url: Option<&str>) -> String {
    match error {
        ModelError::Connection(_) => match base_url
            .map(sanitize_url_for_trace)
            .filter(|shown| !shown.is_empty())
        {
            Some(url) => format!("Could not reach the model at {url}."),
            None => "Could not reach the model.".to_owned(),
        },
        ModelError::Status(code) => format!("The model server answered {code}."),
        ModelError::Timeout => "The model did not answer in time.".to_owned(),
        ModelError::Protocol(_) => {
            "The model server sent something this runtime could not read.".to_owned()
        }
        ModelError::Behavior(_) => {
            "The model asked for something this runtime could not follow.".to_owned()
        }
        ModelError::Truncated => "The model ran out of tokens before answering.".to_owned(),
        ModelError::Failed(_) => "The model could not be used.".to_owned(),
    }
}

/// "The model refused: <its words, at most 500 characters>".
pub fn refusal_sentence(refusal: &str) -> String {
    format!("The model refused: {}", cap_text(refusal, REFUSAL_CHARS))
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;
    use crate::env::MapEnv;
    use crate::state::StateRoot;
    use crate::testkit::TempDir;

    fn endpoint(base_url: &str) -> ModelEndpoint {
        ModelEndpoint {
            id: "lab".into(),
            label: "Lab".into(),
            kind: crate::registry::EndpointKind::OpenaiCompatible,
            base_url: base_url.into(),
            model: "m".into(),
            api_key_name: String::new(),
            enabled: true,
            builtin: false,
            note: String::new(),
        }
    }

    #[test]
    fn every_failure_is_one_sentence_without_transport_text() {
        let secret_url = Some("https://user:hunter2@example.test:8443/v1?token=abc#frag");
        let connection = error_sentence(
            &ModelError::Connection("tcp reset by 10.0.0.9".into()),
            secret_url,
        );
        assert_eq!(
            connection,
            "Could not reach the model at https://example.test:8443/v1."
        );
        assert!(!connection.contains("hunter2") && !connection.contains("token"));
        assert_eq!(
            error_sentence(&ModelError::Connection("x".into()), None),
            "Could not reach the model."
        );
        assert_eq!(
            error_sentence(&ModelError::Status(503), secret_url),
            "The model server answered 503."
        );
        for (error, sentence) in [
            (ModelError::Timeout, "The model did not answer in time."),
            (
                ModelError::Protocol("body: secret-sentinel".into()),
                "The model server sent something this runtime could not read.",
            ),
            (
                ModelError::Behavior("secret-sentinel".into()),
                "The model asked for something this runtime could not follow.",
            ),
            (
                ModelError::Truncated,
                "The model ran out of tokens before answering.",
            ),
            (
                ModelError::Failed("secret-sentinel".into()),
                "The model could not be used.",
            ),
        ] {
            assert_eq!(error_sentence(&error, secret_url), sentence);
        }
        let long = "no ".repeat(400);
        let quoted = refusal_sentence(&long);
        assert!(quoted.starts_with("The model refused: no no"));
        assert_eq!(
            quoted.chars().count(),
            "The model refused: ".len() + REFUSAL_CHARS
        );
    }

    #[test]
    fn a_client_is_built_through_client_config_and_the_factory_sees_it() {
        let dir = TempDir::new("models-build");
        let env = Arc::new(MapEnv::new());
        let keys = KeyStore::new(env.clone(), &StateRoot::at(dir.path()));
        let seen: Arc<Mutex<Vec<(String, bool)>>> = Arc::default();
        let record = seen.clone();
        let factory: ModelFactory = Arc::new(move |config: &ChatCompletionsConfig| {
            record
                .lock()
                .unwrap()
                .push((config.base_url.clone(), config.local));
            None
        });
        let built = build(
            &endpoint("http://127.0.0.1:8000/v1"),
            env.as_ref(),
            &keys,
            Some(&factory),
        )
        .unwrap();
        assert_eq!(built.base_url, "http://127.0.0.1:8000/v1");
        let remote = build(
            &endpoint("https://example.test/v1"),
            env.as_ref(),
            &keys,
            Some(&factory),
        )
        .unwrap();
        assert_eq!(remote.base_url, "https://example.test/v1");
        assert_eq!(
            *seen.lock().unwrap(),
            [
                ("http://127.0.0.1:8000/v1".to_owned(), true),
                ("https://example.test/v1".to_owned(), false)
            ],
            "a loopback endpoint's client is local, a remote one's is not"
        );
        // An address the client refuses is one sentence.
        let refused = build(&endpoint("not a url"), env.as_ref(), &keys, None).err();
        assert_eq!(
            refused,
            Some(Refusal::new(RefusalKind::Invalid, ADDRESS_REFUSAL))
        );
    }
}
