use std::sync::Arc;

use clap::Parser;
use rand::{Rng, SeedableRng};

use lsm::{Database, Params, StartMode};

#[derive(Parser)]
struct Args {
    #[clap(long, short = 'n', default_value_t = 100_000)]
    #[clap(help = "The number of insertions per thread")]
    num_insertions: usize,

    #[clap(long, short = 't', default_value_t = 10)]
    num_threads: usize,

    #[clap(long, default_value_t = 1_000_000)]
    key_range: usize,

    #[clap(long, default_value_t = 1024)]
    entry_size: usize,

    #[clap(long, default_value = "/tmp")]
    #[clap(
        help = "Where to create the temporary working directory? Note, this is the parent directoy of the directory not the directoy itself.
It is recommended to use a tmpfs to not wear out a physical disk"
    )]
    workdir_location: String,
}

#[tokio::main]
async fn main() {
    env_logger::init();

    let args = Args::parse();

    if args.num_insertions == 0 {
        panic!("Need to insert at least one entry");
    }

    if args.key_range == 0 {
        panic!("Key range cannot be zero");
    }

    println!("Creating working directory and empty database");
    let tmp_dir = tempfile::Builder::new()
        .prefix("lsm-bigest-")
        .tempdir_in(args.workdir_location)
        .expect("Failed to create working directory");

    let mut db_path = tmp_dir.path().to_path_buf();
    db_path.push("storage.lsm");

    let params = Params {
        db_path,
        ..Default::default()
    };

    let database = Arc::new(
        Database::new_with_params(StartMode::CreateOrOverride, params)
            .await
            .expect("Failed to create database instance"),
    );

    println!(
        "Inserting a total of {} entries of size {} across {} threads",
        args.num_insertions * args.num_threads,
        args.entry_size,
        args.num_threads
    );

    let tasks: Vec<_> = (1..=args.num_threads)
        .map(|idx| {
            let database = database.clone();
            tokio::spawn(async move {
                let mut rng = rand::rngs::SmallRng::from_rng(&mut rand::rng());

                for count in 1..=args.num_insertions {
                    let key_idx = rng.random_range(0..args.key_range);
                    let key = format!("key{key_idx}").as_bytes().to_vec();

                    let mut value = vec![0; args.entry_size];
                    rng.fill(value.as_mut_slice());

                    database.put(key, value).await.expect("Insert failed");

                    if count % 10_000 == 0 {
                        println!(
                            "Thread #{idx} inserted {count} entries so far ({}%)",
                            (count as f64) * 100.0 / (args.num_insertions as f64)
                        );
                    }
                }
                println!("Thread #{idx} is done");
            })
        })
        .collect();

    for task in tasks {
        task.await.unwrap();
    }
}
