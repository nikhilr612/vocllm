use candle_core::utils as candle_utils;
use log::{debug, info, trace};
use clap::Parser;
use argsc::CliArgs;

use crate::llmcall::QuantizedTextGenerator;

mod argsc;
mod llmcall;
mod chat;

fn main() {
    let mut args = CliArgs::parse();
    if std::env::var("RUST_LOG").is_err() {
        if args.verbose {
            std::env::set_var("RUST_LOG", "trace");
        } else {
            std::env::set_var("RUST_LOG", "warn");
        }
    }
    pretty_env_logger::init();
    info!("CUDA Available? {}", candle_utils::cuda_is_available());
    info!("avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle_utils::with_avx(),
        candle_utils::with_neon(),
        candle_utils::with_simd128(),
        candle_utils::with_f16c()
    );
    debug!("Received {:#?}", args);
    args.fix_options();

    let mut g = match args.base_model {
        argsc::SupportedBaseModels::Mistral | argsc::SupportedBaseModels::Llama => {
            //Conveniently candle supports all llama architecture ggufs under the same model.
            QuantizedTextGenerator::from_args(&args)
        },
        argsc::SupportedBaseModels::Rwkv => todo!("Will implement once support for Llama-based GGUFs is complete."),
    };

    match args.command {
        argsc::Commands::Ripl => todo!("Will implement after line streaming."),
        argsc::Commands::Single(parg) => {
            if args.no_stream {
                trace!("Building prompt...");
                let p = chat::make_prompt(args.template, args.sysprompt.as_ref().unwrap(), &parg.prompt, None);
                let r = g.invoke_infallible(&p);
                println!("{}", r);
            } else {
                todo!("Implement line-streaming.")
            }
        }
    }
}
