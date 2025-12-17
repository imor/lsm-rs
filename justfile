LOG_LEVEL := "debug"

# Common prefix for lints
CLIPPY := "cargo clippy --no-default-features --tests"

all: tests lint

tests: sync-tests async-tests no-compression-tests \
       wisckey-tests \
       wisckey-no-compression-tests wisckey-sync-tests

sync-tests:
    cd sync && just default-tests

async-tests:
    env RUST_BACKTRACE=1 RUST_LOG={{LOG_LEVEL}} cargo test --no-default-features

no-compression-tests:
    env RUST_BACKTRACE=1 RUST_LOG={{LOG_LEVEL}} cargo test --no-default-features

wisckey-tests:
    env RUST_BACKTRACE=1 RUST_LOG={{LOG_LEVEL}} cargo test --no-default-features --features=snappy-compression,wisckey

wisckey-no-compression-tests:
    env RUST_BACKTRACE=1 RUST_LOG={{LOG_LEVEL}} cargo test --no-default-features --features=wisckey

wisckey-sync-tests:
    cd sync && just wisckey-tests

lint: sync-lint async-lint wisckey-lint \
      wisckey-no-compression-lint \
      bigtest-lint

fix-formatting:
    cargo fmt
    cd sync && just fix-formatting
    cd bigtest && cargo fmt

check-formatting:
    cargo fmt --check
    cd sync && just check-formatting

clean:
    rm -rf target/

update-dependencies:
    cargo update
    cd sync && cargo update

udeps:
    cargo udeps --all-targets --release
    cd sync && just udeps

sync-lint:
    cd sync && just lint

async-lint:
    {{CLIPPY}} -- -D warnings

wisckey-lint:
    {{CLIPPY}} --features=snappy-compression,wisckey -- -D warnings

wisckey-no-compression-lint:
    {{CLIPPY}} --features=wisckey -- -D warnings

bigtest-lint:
    {{CLIPPY}} --package=lsm-bigtest

bigtest-many:
    cargo run --release --package=lsm-bigtest -- -n100000 --entry-size=1024

bigtest-large:
    cargo run --release --package=lsm-bigtest -- -n100 --entry-size=100000

