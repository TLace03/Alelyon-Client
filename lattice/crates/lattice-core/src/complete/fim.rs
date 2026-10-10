//! Fill-in-the-middle: how a code model is asked for the text between what is
//! before the cursor and what is after it.
//!
//! A local model is asked through llama.cpp's `/infill`, which writes the
//! model's own fill-in-the-middle tokens from its GGUF file. A hosted model is
//! asked through its provider's plain `/completions`, so the prompt is written
//! here, as each family's tokenizer writes it: only the families below are
//! offered for completions from a provider.

/// A family of code models trained to fill in the middle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    /// Qwen2.5-Coder and Qwen3-Coder.
    Qwen,
    /// Mistral's Codestral.
    Codestral,
    /// DeepSeek-Coder.
    DeepSeek,
    /// StarCoder, StarCoder2 and IBM's Granite code models.
    StarCoder,
    /// Google's CodeGemma.
    CodeGemma,
    /// Meta's Code Llama.
    CodeLlama,
}

/// The family a model id or name belongs to, if it is one of those above.
pub fn family(id: &str) -> Option<Family> {
    let id = id.to_lowercase();
    let has = |w: &str| id.contains(w);
    if has("codestral") {
        Some(Family::Codestral)
    } else if has("qwen") && has("coder") {
        Some(Family::Qwen)
    } else if has("deepseek-coder") || has("deepseek_coder") {
        Some(Family::DeepSeek)
    } else if has("starcoder") || (has("granite") && has("code")) {
        Some(Family::StarCoder)
    } else if has("codegemma") {
        Some(Family::CodeGemma)
    } else if has("codellama") || has("code-llama") {
        Some(Family::CodeLlama)
    } else {
        None
    }
}

impl Family {
    /// The prompt that asks for the middle between `prefix` and `suffix`.
    pub fn prompt(self, prefix: &str, suffix: &str) -> String {
        match self {
            Family::Qwen => format!("<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>"),
            Family::Codestral => format!("[SUFFIX]{suffix}[PREFIX]{prefix}"),
            Family::DeepSeek => format!("<｜fim▁begin｜>{prefix}<｜fim▁hole｜>{suffix}<｜fim▁end｜>"),
            Family::StarCoder => format!("<fim_prefix>{prefix}<fim_suffix>{suffix}<fim_middle>"),
            Family::CodeGemma => format!("<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>"),
            Family::CodeLlama => format!("<PRE> {prefix} <SUF>{suffix} <MID>"),
        }
    }

    /// The tokens that end the middle, as text.
    pub fn stops(self) -> &'static [&'static str] {
        match self {
            Family::Qwen => &["<|endoftext|>", "<|fim_pad|>", "<|file_sep|>", "<|im_end|>", "<|fim_prefix|>"],
            Family::Codestral => &["</s>", "[PREFIX]", "[SUFFIX]"],
            Family::DeepSeek => &["<｜end▁of▁sentence｜>", "<｜fim▁begin｜>", "<｜EOT｜>"],
            Family::StarCoder => &["<|endoftext|>", "<file_sep>", "<fim_prefix>"],
            Family::CodeGemma => &["<|file_separator|>", "<eos>", "<|fim_prefix|>"],
            Family::CodeLlama => &["<EOT>", "</s>", "<PRE>"],
        }
    }
}

/// Every family's stop tokens, for an answer whose family is not known (a
/// local model's `/infill`).
pub fn all_stops() -> impl Iterator<Item = &'static str> {
    [
        Family::Qwen,
        Family::Codestral,
        Family::DeepSeek,
        Family::StarCoder,
        Family::CodeGemma,
        Family::CodeLlama,
    ]
    .into_iter()
    .flat_map(|f| f.stops().iter().copied())
}
