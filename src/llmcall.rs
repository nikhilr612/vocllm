//! Module to load and make inference calls on LLMs.

use candle_core::quantized::gguf_file::Content;
use std::fs::File;
use std::time::Instant;

use log::{debug, info, trace};
use log::{error, warn};
use candle_transformers::generation::LogitsProcessor;
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_llama::ModelWeights;
use tokenizers::Tokenizer;

use crate::argsc::CliArgs;

const DEBUG_TOKEN_COUNT: usize = 128;

pub struct QuantizedTextGenerator {
	model: ModelWeights,
	device: Device,
	tokenizer: Tokenizer,
	logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
    eos: u32
}

fn get_device(cpu: bool) -> Device{
	if cpu {
		Device::Cpu
	} else if candle_core::utils::cuda_is_available() {
		match Device::new_cuda(0) {
			Ok(a) => a,
			Err(e) => {
				error!("Failed to acquire cuda device, cause: \"{:?}\", falling back to CPU", e);
				Device::Cpu
			}
		}
	} else {
		warn!("CUDA is not available. Falling back to CPU");
		Device::Cpu
	}
}

fn load_model_infallible(path: &str, device: &Device) -> (ModelWeights, Option<u32>) {
    trace!("Loading model {}", path);
    let load_start = Instant::now();
    let mut file = File::open(path).expect("Failed to open model file.");
    let model = Content::read(&mut file).map_err(|e| e.with_path(path)).expect("Failed to read GGUF file content");
    trace!("Checking metadata for EOS information...");
    let eos_token_id = model.metadata.get("tokenizer.ggml.eos_token_id").and_then(|v| v.to_u32().ok());
    let mut total_size_in_bytes = 0;
    trace!("Inspecting tensors...");
    for (_, tensor) in model.tensor_infos.iter() {
        let elem_count = tensor.shape.elem_count();
        total_size_in_bytes +=
            elem_count * tensor.ggml_dtype.type_size() / tensor.ggml_dtype.block_size();
    }

    let n_tensors = model.tensor_infos.len();
    trace!("Loading model weights...");
    let ret = ModelWeights::from_gguf(model, &mut file, device).expect("Failed to load model from GGUF file.");

    info!("Successfully loaded model: {} [{} tensors, {} bytes] in {}s", path, n_tensors, total_size_in_bytes, load_start.elapsed().as_secs());
    (ret, eos_token_id)
}

impl QuantizedTextGenerator {
	pub fn from_args(args: &CliArgs) -> Self {
		let device = get_device(args.cpu);
		debug!("Active Device: {:?}", device);

		trace!("Attempting to create tokenizer...");
		let raw_tokenizer = Tokenizer::from_file(args.tokenizer_json.as_ref().unwrap()).expect("Failed to create tokenizer.");
		trace!("Tokenizer loaded.");

        // Unused.
		// let _config: QMistralConfig = match args.config_option.as_deref() {
		// 	None => QMistralConfig::config_7b_v0_1(true),
		// 	Some("chatml") => QMistralConfig::config_chat_ml(true),
		// 	Some("amazon") => QMistralConfig::config_amazon_mistral_lite(true),
		// 	Some(t) => {
		// 		warn!("Unrecognized configuration option '{}' for Mistral base model. Falling back to default configuration.", t);
		// 		QMistralConfig::config_7b_v0_1(true)
		// 	}
		// };

        // NOTE: Due to a questionable implementation for Quantized Mistral model in candle, llama model will be used.
		// let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(args.model_path.clone(), &device).expect("Failed to create VarBuilder");
        // let model = QMistralModel::new(&config, vb).expect("Failed to load model.");

        let (model, eos_meta) = load_model_infallible(&args.model_path, &device);
        let eos = eos_meta.or(args.eos_token).unwrap_or_else(|| {
            error!("GGUF does not define appropriate metadata, and neither was EOS supplied via arguments.");
            panic!("Failed to identify EOS token.");
        });

        debug!("Using seed: {}", args.seed);

        let logits_processor = LogitsProcessor::new(args.seed, Some(args.temperature), args.top_p);
        
        Self {
            model,
            tokenizer: raw_tokenizer,
            logits_processor,
            repeat_penalty: args.repeat_penalty,
            repeat_last_n: args.repeat_last_n,
            device,
            eos
        }
	}

    /// Invoke the LLM and yield generated output.
    /// If any errors occur, log and panic.
    pub fn invoke_infallible(&mut self, prompt: &str) -> String {
        // Encode the prompt.
        let mut tokens = self.tokenizer.encode(prompt, true).unwrap_or_else(|e| {
            error!("Failed to encode prompt {prompt} with tokenizer");
            panic!("{e:?}");
        }).get_ids().to_vec();

        trace!("Tokenized prompt.");

        trace!("Starting generation.");
        let start_time = Instant::now();
        let mut generation_count = 0;
        let mut flag = true;

        loop {
            let (context, seqoff) = if flag {
                flag = false;
                (tokens.as_slice(), 0)
            } else {
                let off = tokens.len().saturating_sub(1);
                (&tokens[off..], off)
            };

            let input = Tensor::new(context, &self.device).and_then(|t| t.unsqueeze(0)).expect("Could not initialize context tensor.");
            let mut logits = self.model.forward(&input, seqoff)
                        .and_then(|t| t.squeeze(0))
                        .and_then(|t| t.squeeze(0))
                        .and_then(|t| t.to_dtype(candle_core::DType::F32))
                        .expect("Failed to obtain logits.");
            if self.repeat_penalty != 1.0 {
                let repeat_context = &tokens[tokens.len().saturating_sub(self.repeat_last_n)..];
                logits = candle_transformers::utils::apply_repeat_penalty(&logits, self.repeat_penalty, repeat_context).expect("Could not apply repeat penalty");
            }
            
            let next_token = self.logits_processor.sample(&logits).expect("Could not sample token from logits");
            tokens.push(next_token);
            generation_count += 1;
            
            if (generation_count % DEBUG_TOKEN_COUNT) == 0 {
                debug!("Got {} tokens so far.", generation_count);
                if cfg!(debug_assertions) {
                    trace!("Decoded tokens {:?}", self.tokenizer.decode(&tokens, false));
                }
            }

            if next_token == self.eos {
                break;
            }
        }

        trace!("Finished token generation.");
        let t = start_time.elapsed().as_secs();
        debug!("Genereated {} tokens in {}s [avg: {}t/s]", generation_count, t, generation_count as f64 / (t as f64));
        trace!("Decoding...");
        self.tokenizer.decode(&tokens, true).unwrap_or_else(|e| {
            error!("Failed to decode generated tokens: {tokens:?}");
            panic!("Tokenizer decode resulted in error. {e:?}");
        })[prompt.len()..].to_owned()
    }
    
    /*
	pub fn talk_and_map<F>(&mut self, prompt: &str, mut cb: F)
	where F: FnMut(&str) {	
	   unimplemented!("Will implement this once a mechanism to stream lines from tokens is established.");	
	}*/
}