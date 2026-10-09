//! Token usage: summing it over a run and writing it into spans.
//!
//! Ports `agents.usage` (0.22.3): `Usage.add`, `usage_delta`,
//! `turn_usage_to_span_data`, `task_usage_to_span_data`,
//! `model_usage_to_span_usage` and `attach_usage_to_span`, for the four counts
//! [`lattice_protocol::Usage`] carries. The SDK also tracks cached and reasoning
//! token detail; this port does not receive it, so those detail fields are
//! written as `0`, which is what the SDK writes for a model that reports none.
//!
//! Usage is absent, not zero, when no model call reported any: a local server
//! that ignores `stream_options` must not look like a model that used no tokens.

use lattice_protocol::Usage;
use serde_json::{Value, json};

pub(crate) fn add(a: Usage, b: Usage) -> Usage {
    Usage {
        requests: a.requests.saturating_add(b.requests),
        input_tokens: a.input_tokens.saturating_add(b.input_tokens),
        output_tokens: a.output_tokens.saturating_add(b.output_tokens),
        total_tokens: a.total_tokens.saturating_add(b.total_tokens),
    }
}

/// The usage added between two snapshots (`usage_delta`).
pub(crate) fn delta(start: Usage, end: Usage) -> Usage {
    Usage {
        requests: end.requests.saturating_sub(start.requests),
        input_tokens: end.input_tokens.saturating_sub(start.input_tokens),
        output_tokens: end.output_tokens.saturating_sub(start.output_tokens),
        total_tokens: end.total_tokens.saturating_sub(start.total_tokens),
    }
}

/// `attach_usage_to_span` writes nothing for an all-zero delta.
pub(crate) fn is_zero(usage: Usage) -> bool {
    usage.requests == 0
        && usage.input_tokens == 0
        && usage.output_tokens == 0
        && usage.total_tokens == 0
}

/// `turn_usage_to_span_data`.
pub(crate) fn turn_span_usage(usage: Usage) -> Value {
    json!({
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cached_input_tokens": 0,
        "cache_write_input_tokens": 0,
    })
}

/// `task_usage_to_span_data`.
pub(crate) fn task_span_usage(usage: Usage) -> Value {
    json!({
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cached_input_tokens": 0,
        "cache_write_input_tokens": 0,
        "requests": usage.requests,
        "total_tokens": usage.total_tokens,
    })
}

/// `model_usage_to_span_usage`: one model call's usage on its generation span.
pub(crate) fn generation_span_usage(usage: Usage) -> Value {
    json!({
        "requests": usage.requests,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
        "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": 0},
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn usage(requests: u64, input: u64, output: u64) -> Usage {
        Usage {
            requests,
            input_tokens: input,
            output_tokens: output,
            total_tokens: input + output,
        }
    }

    #[test]
    fn usage_adds_and_subtracts_without_wrapping() {
        let sum = add(usage(1, 10, 2), usage(1, 5, 1));
        assert_eq!(sum, usage(2, 15, 3));
        assert_eq!(delta(usage(1, 10, 2), sum), usage(1, 5, 1));
        assert_eq!(delta(sum, usage(1, 10, 2)), Usage::default());
        assert!(is_zero(Usage::default()));
        assert!(!is_zero(usage(1, 0, 0)));
        assert_eq!(
            add(usage(u64::MAX, 0, 0), usage(1, 0, 0)).requests,
            u64::MAX
        );
    }

    #[test]
    fn span_usage_has_the_sdks_keys() {
        let turn = turn_span_usage(usage(1, 9, 2));
        assert_eq!(
            turn,
            json!({"input_tokens": 9, "output_tokens": 2, "cached_input_tokens": 0, "cache_write_input_tokens": 0})
        );
        let task = task_span_usage(usage(3, 49, 15));
        assert_eq!(task["requests"], 3);
        assert_eq!(task["total_tokens"], 64);
        let generation = generation_span_usage(usage(1, 1200, 34));
        assert_eq!(generation["input_tokens"], 1200);
        assert_eq!(
            generation["output_tokens_details"],
            json!({"reasoning_tokens": 0})
        );
    }
}
