//! The models that can write completions, with what each costs and grouped by
//! price, so the reader can compare and filter them.
//!
//! - **This PC:** every GGUF file in the models folder, free to run (it takes
//!   the graphics card's memory instead); one whose name says a
//!   fill-in-the-middle family is marked as such.
//! - **Providers:** the providers whose API has a plain `/completions`
//!   ([`PROVIDERS`]), and of their models only the fill-in-the-middle families
//!   ([`fim::family`]), whether or not their weights are open (Codestral's are
//!   not). A price is the provider's own, when its model list says one.
//!
//! The cost of a completion is estimated from the most text a request sends
//! ([`super::PREFIX_CHARS`] and [`super::SUFFIX_CHARS`], at about four
//! characters a token) and the most it asks back ([`super::MAX_TOKENS`]), and
//! shown per thousand completions: a request near the start of a short file
//! costs less.

use serde_json::Value;

use super::fim;
use super::settings::Choice;
use crate::hosted::{self, Provider};
use crate::llama::files::LocalModel;

/// The providers asked for completions: those with a plain `/completions`.
pub const PROVIDERS: &[&str] = &["openrouter", "together", "fireworks", "deepinfra", "cerebras"];

/// Tokens a completion's request sends at most, estimated.
pub const INPUT_TOKENS: f64 = ((super::PREFIX_CHARS + super::SUFFIX_CHARS) / 4) as f64;

/// A group of models by what a thousand completions cost.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Tier {
    Free,
    UnderQuarter,
    UnderDollar,
    OverDollar,
    Unlisted,
}

impl Tier {
    pub const ALL: [Tier; 5] = [Tier::Free, Tier::UnderQuarter, Tier::UnderDollar, Tier::OverDollar, Tier::Unlisted];

    /// The group's name, for the reader.
    pub fn label(self) -> &'static str {
        match self {
            Tier::Free => "Free",
            Tier::UnderQuarter => "Under $0.25 per 1,000",
            Tier::UnderDollar => "$0.25 to $1 per 1,000",
            Tier::OverDollar => "Over $1 per 1,000",
            Tier::Unlisted => "Price not listed",
        }
    }
}

/// The group an estimated cost per thousand completions falls in.
pub fn tier(per_thousand: Option<f64>) -> Tier {
    match per_thousand {
        None => Tier::Unlisted,
        Some(c) if c <= 0.0 => Tier::Free,
        Some(c) if c < 0.25 => Tier::UnderQuarter,
        Some(c) if c <= 1.0 => Tier::UnderDollar,
        Some(_) => Tier::OverDollar,
    }
}

/// US dollars for a thousand completions at the largest request, from prices
/// per million input and output tokens.
pub fn per_thousand((input, output): (f64, f64)) -> f64 {
    1000.0 * (INPUT_TOKENS * input + f64::from(super::MAX_TOKENS) * output) / 1_000_000.0
}

/// A cost for the reader: "Free", or dollars to two significant places.
pub fn cost_text(per_thousand: Option<f64>) -> String {
    match per_thousand {
        None => "price not listed".to_owned(),
        Some(c) if c <= 0.0 => "free".to_owned(),
        Some(c) if c < 0.01 => format!("about ${c:.4} per 1,000"),
        Some(c) if c < 1.0 => format!("about ${c:.3} per 1,000"),
        Some(c) => format!("about ${c:.2} per 1,000"),
    }
}

/// A model that can be chosen for completions.
#[derive(Clone, Debug, PartialEq)]
pub struct Offer {
    pub choice: Choice,
    /// Its name, for the reader.
    pub name: String,
    /// Where it runs: "This PC" or the provider's name.
    pub host: String,
    /// US dollars per million input and output tokens, when said.
    pub price: Option<(f64, f64)>,
    /// US dollars per thousand completions, estimated.
    pub per_thousand: Option<f64>,
    pub tier: Tier,
    /// Whether its name says a fill-in-the-middle family (always, for a
    /// provider's model; a local file may be one without saying so).
    pub fim: bool,
    /// Whether it can be used now: a provider's model needs the provider's key.
    pub ready: bool,
}

/// The models folder's files, each free to run.
pub fn local_offers(models: &[LocalModel]) -> Vec<Offer> {
    models
        .iter()
        .map(|m| Offer {
            choice: Choice::Local { model: m.name.clone() },
            name: m.name.clone(),
            host: "This PC".to_owned(),
            price: Some((0.0, 0.0)),
            per_thousand: Some(0.0),
            tier: Tier::Free,
            fim: fim::family(&m.name).is_some(),
            ready: true,
        })
        .collect()
}

/// Whether a provider's row is a fill-in-the-middle model.
pub fn fim_row(_: &Provider, row: &Value) -> bool {
    let id = row.get("id").and_then(Value::as_str).unwrap_or("").to_lowercase();
    !hosted::NOT_CHAT.iter().any(|w| id.contains(w)) && fim::family(&id).is_some()
}

/// A provider's fill-in-the-middle models from its `/models` answer.
pub fn hosted_offers(provider: &Provider, value: &Value, ready: bool) -> Option<Vec<Offer>> {
    let models = hosted::parse_models_where(provider, value, fim_row)?;
    Some(
        models
            .into_iter()
            .map(|m| {
                let per = m.price.map(per_thousand);
                Offer {
                    choice: Choice::Hosted { provider: provider.id.to_owned(), model: m.id.clone() },
                    name: m.name,
                    host: provider.label.to_owned(),
                    price: m.price,
                    per_thousand: per,
                    tier: tier(per),
                    fim: true,
                    ready,
                }
            })
            .collect(),
    )
}

/// The offers in their groups, cheapest group first; in a group, the cheapest
/// first, then those usable now, then by name.
pub fn grouped(mut offers: Vec<Offer>) -> Vec<(Tier, Vec<Offer>)> {
    offers.sort_by(|a, b| {
        a.tier
            .cmp(&b.tier)
            .then(a.per_thousand.unwrap_or(f64::MAX).total_cmp(&b.per_thousand.unwrap_or(f64::MAX)))
            .then(b.ready.cmp(&a.ready))
            .then(b.fim.cmp(&a.fim))
            .then(a.name.to_lowercase().cmp(&b.name.to_lowercase()))
            .then(a.host.cmp(&b.host))
    });
    let mut groups: Vec<(Tier, Vec<Offer>)> = Vec::new();
    for offer in offers {
        match groups.last_mut() {
            Some((tier, list)) if *tier == offer.tier => list.push(offer),
            _ => groups.push((offer.tier, vec![offer])),
        }
    }
    groups
}
