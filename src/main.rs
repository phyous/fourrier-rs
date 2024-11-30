use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;

mod audio;
mod speech;
mod visualization;
mod init;

use audio::{load_audio, compute_spectrogram};
use speech::transcribe_audio;
use visualization::Visualizer;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Path to the audio file to analyze
    #[arg(short, long)]
    input: PathBuf,

    /// Window size for FFT (must be a power of 2)
    #[arg(short, long, default_value = "1024")]
    window_size: usize,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    println!("Loading audio file...");
    let audio_data = load_audio(&cli.input)?;

    println!("Computing spectrogram...");
    let spectrogram = compute_spectrogram(&audio_data, cli.window_size)?;

    println!("Transcribing audio...");
    let transcription = transcribe_audio(&cli.input)?;

    let visualizer = Visualizer::new(audio_data, spectrogram, transcription);
    visualizer.run()?;

    Ok(())
}
