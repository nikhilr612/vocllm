//! A module to define, and derive CLI parser to obtain all settings and configuration data for the application.


use std::{fs, path::PathBuf};

use clap::{Args, Parser, Subcommand, ValueEnum};
use log::error;

use crate::chat::ChatTemplate;

const DEFUALT_SYSTEM_PROMPT: &str = "
You are a professional interactive AI assistant.
Your job is to answer any queries and perform any actions required of you to the best of your ability.
You may optionally be provided with additional context which must be incorporated into your answer.
You may optionally also be provided several tools that you may use to satisfy user's requests.
Your answers must be concise and correct. Never refuse to answer.
";

#[derive(Debug, Parser)]
#[command(version, about)]
pub struct CliArgs {
	#[command(subcommand)]
	pub command: Commands,
	#[arg(short, long)]
	/// Path to GGUF file to load.
	pub model_path: String,
	#[arg(short = 'T', long)]
	/// Path to HF tokeniser data file for the model. If not specified, will look for 'tokenizer.json' in same directory as model_path.
	/// Ideally, this should be inferred from data within GGUF, however, as candle doesn't provide any methods for that yet, this is used as fallback.
	pub tokenizer_json: Option<String>,
	// #[arg(short = 'C', long)]
	// /// Choose one of few preset configuration for the base model. Ideally, this should be reading its own json file.
	// pub config_option: Option<String>,
	#[arg(long, default_value_t = 42)]
	/// Seed value to use for generation. Important for reproducability.
	pub seed: u64,
	#[arg(long, default_value_t = 0.7)]
	pub temperature: f64,
	#[arg(long)]
	pub top_p: Option<f64>,
	#[arg(long, default_value_t = 1.1)]
	pub repeat_penalty: f32,
	#[arg(long, default_value_t = 64)]
	pub repeat_last_n: usize,

	#[arg(short = 'c', long, default_value_t = false)]
	/// Use CPU when true. Otherwise CUDA/CUDNN.
	pub cpu: bool,
	#[arg(long)]
	/// Specify path to a file containing potentially partially summarized chat history to be loaded.
	/// If unspecified, a text file will be opened in local directory with model path information for this chat.
	/// History file is updated and saved on exit, unless `incognito` is set to true.
	pub historyfile: Option<String>,
	#[arg(long)]
	/// Path to file containing text that will comprise the perpetural system prompt that will be provided along with user prompt, rag context and other details.
	/// If unspecified, a default system prompt will be used.
	pub sysprompt: Option<String>,
	#[arg(long)]
	/// A string specifying tts option for speech synthesis, in the form "<tts provider>/<internal data>". Ex: "sapi/ZIRA".
	/// If unspecified, no speech synthesis will be performed.
	pub ttsopt: Option<String>,
	#[arg(short = 'i', long, default_value_t = false)]
	/// When set, chat history will not be saved, although chat history will be active.
	pub incognito: bool,
	#[arg(long, default_value_t = false)]
	/// When set, disable chat history.
	pub disable_history: bool,
	#[arg(short, long, default_value_t = false)]
	/// When set, Unless RUST_LOG is also set, default log level is 'trace', log level is 'warn'
	pub verbose: bool,
	#[arg(short = 'B', long)]
	pub base_model: SupportedBaseModels,
	#[arg(short, long)]
	/// Wehn set, block printing/rendering until LLM emits EOS token.
	pub no_stream: bool,
	#[arg(long)]
	/// Set End-Of-Statement token. If GGUF defines an internal EOS, this is value is overriden.
	/// If GGUF does not specify any EOS, then it is required to set this.
	pub eos_token: Option<u32>,
	#[arg(long, default_value_t = 4096)]
	/// The rough count of how many tokens to retain in history. This value should not be bigger than context size.
	pub history_count: usize,	// TODO: Infer context size from GGUF and set this to a proportionate value.
	/// The chat template to apply to user prompt.
	#[arg(short = 't', long,  default_value = "chat-ml")]
	pub template: ChatTemplate
}

impl CliArgs {
	pub fn fix_options(&mut self) {
		let mpath = PathBuf::from(self.model_path.clone());

		// Find tokenizer.json path.
		if self.tokenizer_json.is_none() {
			self.tokenizer_json = mpath.parent()
			.expect("Failed to extract path to parent directory of module path, to search for default tokenizer.json")
			.join("tokenizer.json")
			.to_str()
			.map(|e| e.to_owned());
		}

		// Load system prompt
		self.sysprompt = if let Some(ppath) = &self.sysprompt {
			match fs::read_to_string(ppath) {
				Ok(text) => Some(text),
				Err(e) => {
					error!("Failed to read system prompt from {}, cause: \"{:?}\" ", ppath, e);
					Some(DEFUALT_SYSTEM_PROMPT.to_owned())
				}
			}
		} else {
			Some(DEFUALT_SYSTEM_PROMPT.to_owned())
		}
	}
}

#[derive(ValueEnum, Clone, Debug)]
pub enum SupportedBaseModels {
	Mistral,
	Llama,
	Rwkv
}

#[derive(Debug, Subcommand)]
pub enum Commands {
	/// Enter into a loop, where every iteration stdin is read as user prompt to LLM and inference output is printed/rendered.
	Ripl,
	/// Execute exactly one prompt for LLM with the provided system prompt. All chat history related options are overidden and disabled.
	Single(PromptArg)
}

#[derive(Debug, Args)]
pub struct PromptArg {
	/// The user prompt to be fed to the LLM verbatim.
	pub prompt: String
}