# VocLLM
A simple, no-nonsense local CLI-based LLM inferencing tool, with [lancedb](https://lancedb.com/) integration for RAG support and function calling.

# Yet another CLI LLM inference tool.
Although every library providing LLM invocation/inferencing capabilities exposes a CLI interface, like, [llama-cpp](https://github.com/ggerganov/llama.cpp), [rustformers/llm](https://github.com/rustformers/llm), basic RAG+TTS support is missing. VocLLM does not aim to be the most flexible, or configurable solution in this regard, but is instead a quick-and-dirty tool for testing out RAG systems, and voice-interfaced LLM.

# Why Rust?
A python implementation leveraging `llama-cpp`, and `lancedb` bindings would be trivial, and suitably satisfactory.
However, I chose Rust, because everything is better in Rust. (Trust me)

# Planned Features
- [ ] Mistral Support
- [ ] Chat Template
- [ ] Local Native TTS
- [ ] RWKV Support
- [ ] LanceDB RAG
- [ ] Function Calling