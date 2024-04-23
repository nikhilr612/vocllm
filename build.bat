@echo off

set RUSTFLAGS=-Ctarget-feature=+sse3,+avx

set RUST_LOG=trace
cargo run -- %*

set /P choice=Build Release? [y/]
if %choice%=="y" (
cargo build --release
)

set /P choice2=Build Custom Profile? [y/]
set RUSTFLAGS=-Ctarget-cpu=native
if %choice2%=="y" (
cargo build --profile native
)