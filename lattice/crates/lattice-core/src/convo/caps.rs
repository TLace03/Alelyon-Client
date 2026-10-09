//! What a model can do (the chat core's spec §4.5, as amended by §22
//! LR8): whether it takes tools, whether it sees images, and its context.
//!
//! For the managed llama.cpp server the facts come from the GGUF header
//! (context length) and from `/props` on the running server (chat template,
//! `n_ctx`, vision). Tool calling is assumed only when all three hold:
//! - `/props` reports a chat template;
//! - `--jinja` is on, which the launch always passes ([`JINJA_ON`]);
//! - the tool-call probe has passed once for this binary and model, recorded
//!   beside Python's `grammar-probes.json` (`llama::probes::TOOL_PROBES`).
//!
//! Otherwise tools are [`Tri::No`] and the turn is a plain turn; the UI says
//! so with [`PLAIN_FALLBACK`]. Before the probe has run for a binary and
//! model, tools are [`Tri::Unknown`]: the first Agent-mode turn sends it
//! (`convo::probe`), checking the template on `/props` first, and a failure
//! ends that turn, so a model never gets tools it has not shown it can call.
//!
//! Endpoints are [`Tri::Unknown`] until the Models increment declares them;
//! `Unknown` behaves as `Yes` with the §4.5 fallback (`convo::agent`, row
//! E11: a first tool-bearing request answered 400 or 422 ends the turn, and
//! the choice takes no tools for the rest of the process).

use crate::llama::gguf::Description;
use crate::llama::server::Props;

/// The launch always passes `--jinja` (`llama::server::command`).
pub const JINJA_ON: bool = true;

/// What the composer says when a managed model has not shown it can call
/// tools. PROVISIONAL.
pub const PLAIN_FALLBACK: &str =
    "This model has not shown it can call tools here, so this turn is a plain answer.";

/// Yes, no, or not known.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Tri {
    Yes,
    No,
    Unknown,
}

/// What a model can do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelCaps {
    pub tools: Tri,
    pub vision: Tri,
    pub context_tokens: Option<u32>,
}

impl ModelCaps {
    /// An endpoint nothing has declared yet.
    pub const UNKNOWN: Self = Self {
        tools: Tri::Unknown,
        vision: Tri::Unknown,
        context_tokens: None,
    };
}

/// The managed server's capabilities, from the header, `/props` and the
/// tool-call probe's record (LR8).
pub fn managed_caps(
    header: Option<&Description>,
    props: Option<&Props>,
    tool_probe_passed: bool,
) -> ModelCaps {
    let template = props.is_some_and(|props| props.chat_template.is_some());
    let tools = if template && JINJA_ON && tool_probe_passed {
        Tri::Yes
    } else {
        Tri::No
    };
    let vision = match props.and_then(|props| props.vision) {
        Some(true) => Tri::Yes,
        Some(false) => Tri::No,
        None => Tri::Unknown,
    };
    let context_tokens = props
        .and_then(|props| props.n_ctx)
        .and_then(|n| u32::try_from(n).ok())
        .filter(|n| *n > 0)
        .or_else(|| header.and_then(Description::context_tokens));
    ModelCaps {
        tools,
        vision,
        context_tokens,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn props(template: bool, n_ctx: Option<u64>, vision: Option<bool>) -> Props {
        Props {
            chat_template: template.then(|| "{{x}}".to_owned()),
            n_ctx,
            vision,
        }
    }

    #[test]
    fn tools_need_a_template_jinja_and_a_passed_probe() {
        let full = props(true, Some(4096), Some(false));
        assert_eq!(managed_caps(None, Some(&full), true).tools, Tri::Yes);
        assert_eq!(
            managed_caps(None, Some(&full), false).tools,
            Tri::No,
            "no probe record"
        );
        let bare = props(false, Some(4096), None);
        assert_eq!(
            managed_caps(None, Some(&bare), true).tools,
            Tri::No,
            "no template"
        );
        assert_eq!(managed_caps(None, None, true).tools, Tri::No, "no /props");
    }

    #[test]
    fn the_context_comes_from_props_then_the_header_and_vision_from_props() {
        let caps = managed_caps(None, Some(&props(true, Some(8192), Some(true))), false);
        assert_eq!(caps.context_tokens, Some(8192));
        assert_eq!(caps.vision, Tri::Yes);
        let none = managed_caps(None, None, false);
        assert_eq!(
            none,
            ModelCaps {
                tools: Tri::No,
                vision: Tri::Unknown,
                context_tokens: None
            }
        );
        assert_eq!(
            managed_caps(None, Some(&props(true, Some(0), None)), false).context_tokens,
            None
        );
    }
}
