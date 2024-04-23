use std::{collections::VecDeque, fmt::{Debug, Display}};

use clap::ValueEnum;

#[derive(Debug)]
pub enum ChatRole {
	System,
	User,
	Assistant
}

impl Display for ChatRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
        	Self::System => write!(f, "system"),
        	Self::User => write!(f, "user"),
        	Self::Assistant => write!(f, "assistant")
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum ChatTemplate {
	ChatML,
	IMessenger
}

impl ChatTemplate {

	pub fn apply_one(&self, role: ChatRole, message: &str) -> String {
		match self {
		    Self::ChatML => format!("<|im_start|>{}\n{}<|im_end|>\n", role, message),
		    Self::IMessenger => format!("{}: {}\n", role.to_string().to_uppercase(), message)
		}
	}

	pub fn generation_lead(&self) -> &str {
		match self {
			Self::ChatML => "<|im_start|>assistant\n",
			Self::IMessenger => "ASSISTANT: "
		}
	}

	pub fn insert_history(&self, buf: &mut String, history: &ChatHistory) {
		for (_, line) in history.message_queue.iter() {
			buf.push_str(line);
		}
	}
}

pub struct ChatHistory {
	rough_token_count: usize,
	token_limit: usize,
	message_queue: VecDeque<(usize, String)>
}

impl ChatHistory {

	pub fn new(limit: usize) -> ChatHistory {
		ChatHistory { rough_token_count: 0, token_limit: limit, message_queue: VecDeque::new() }
	}

	pub fn record_message(&mut self, message: &str) {
		let n_new_tokens = (message.split_whitespace().count() * 4) / 3;
		self.message_queue.push_back((n_new_tokens, message.to_owned()));
		self.rough_token_count += n_new_tokens;

		// Current strategy is to just discard old chats.
		// TODO: Add chat history summarization.
		while self.rough_token_count > self.token_limit {
			if let Some((n, _)) = self.message_queue.pop_front() {
				self.rough_token_count -= n;
			} else {
				panic!("Cannot remove anything from history to reduce token count! This should not happen.");
			}
		}
	}
}

pub fn make_prompt_with_history(template: ChatTemplate, system_prompt: &str, user_prompt: &str, mut additional_context: Option<String>, history: &mut ChatHistory) -> String {
	let mut ret = String::new();
	ret.push_str(&template.apply_one(ChatRole::System, system_prompt));
	template.insert_history(&mut ret, history);
	if let Some(actx) = additional_context.take() {
		let formatted_context = template.apply_one(ChatRole::System, &actx);
		ret.push_str(&formatted_context);
	}
	let user_prompt = template.apply_one(ChatRole::User, user_prompt);
	history.record_message(&user_prompt);
	ret.push_str(template.generation_lead());
	ret
}

pub fn make_prompt(template: ChatTemplate, system_prompt: &str, user_prompt: &str, mut additional_context: Option<String>) -> String {
	let mut ret = String::new();
	ret.push_str(&template.apply_one(ChatRole::System, system_prompt));
	if let Some(text) = additional_context.take() {
		ret.push_str(&template.apply_one(ChatRole::System, &text));
	}
	ret.push_str(&template.apply_one(ChatRole::User, user_prompt));
	ret
}