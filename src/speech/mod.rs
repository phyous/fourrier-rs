use anyhow::{Result, anyhow};
use std::path::Path;
use whisper_rs::{WhisperContext, FullParams, SamplingStrategy};
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use symphonia::core::audio::Signal;
use std::fs::File;

pub struct TranscriptionSegment {
    pub text: String,
    pub start: f64,
    pub end: f64,
}

fn load_audio_for_whisper<P: AsRef<Path>>(path: P) -> Result<Vec<f32>> {
    println!("Loading audio file for Whisper...");
    let file = File::open(&path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let hint = Hint::new();
    let format_opts = FormatOptions::default();
    let metadata_opts = MetadataOptions::default();
    let decoder_opts = DecoderOptions::default();

    let probed = symphonia::default::get_probe().format(&hint, mss, &format_opts, &metadata_opts)?;
    let mut format = probed.format;
    
    // Get sample rate before processing packets
    let track = format.default_track().unwrap();
    let sample_rate = track.codec_params.sample_rate.unwrap_or(16000);
    println!("Audio format: {:?}", track.codec_params.codec);
    println!("Sample rate: {} Hz", sample_rate);
    println!("Channels: {:?}", track.codec_params.channels);

    let mut decoder = symphonia::default::get_codecs().make(&track.codec_params, &decoder_opts)?;
    let mut samples = Vec::new();

    println!("Decoding audio...");
    while let Ok(packet) = format.next_packet() {
        let decoded = decoder.decode(&packet)?;
        match decoded {
            symphonia::core::audio::AudioBufferRef::F32(buf) => {
                samples.extend_from_slice(buf.chan(0));
            },
            symphonia::core::audio::AudioBufferRef::U8(buf) => {
                samples.extend(buf.chan(0).iter().map(|&x| (x as f32 / 128.0) - 1.0));
            },
            symphonia::core::audio::AudioBufferRef::U16(buf) => {
                samples.extend(buf.chan(0).iter().map(|&x| (x as f32 / 32768.0) - 1.0));
            },
            symphonia::core::audio::AudioBufferRef::U32(buf) => {
                samples.extend(buf.chan(0).iter().map(|&x| (x as f32 / 2147483648.0) - 1.0));
            },
            symphonia::core::audio::AudioBufferRef::S8(buf) => {
                samples.extend(buf.chan(0).iter().map(|&x| x as f32 / 128.0));
            },
            symphonia::core::audio::AudioBufferRef::S16(buf) => {
                samples.extend(buf.chan(0).iter().map(|&x| x as f32 / 32768.0));
            },
            symphonia::core::audio::AudioBufferRef::S32(buf) => {
                samples.extend(buf.chan(0).iter().map(|&x| x as f32 / 2147483648.0));
            },
            _ => {
                println!("Unsupported audio format, skipping packet");
                continue;
            }
        }
    }

    println!("Loaded {} samples", samples.len());
    // Debug: Check sample values
    if !samples.is_empty() {
        println!("First few samples: {:?}", &samples[..5.min(samples.len())]);
        println!("Sample range: [{}, {}]", 
            samples.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
            samples.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b))
        );
    }

    // Normalize samples to [-1, 1] range if needed
    let max_abs = samples.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    if max_abs > 1.0 {
        println!("Normalizing samples...");
        for sample in &mut samples {
            *sample /= max_abs;
        }
    }

    // Resample to 16kHz if needed
    if sample_rate != 16000 {
        println!("Resampling from {}Hz to 16kHz...", sample_rate);
        let ratio = 16000.0 / sample_rate as f32;
        let new_len = (samples.len() as f32 * ratio) as usize;
        let mut resampled = Vec::with_capacity(new_len);
        
        for i in 0..new_len {
            let src_idx = (i as f32 / ratio) as usize;
            if src_idx < samples.len() {
                resampled.push(samples[src_idx]);
            }
        }
        samples = resampled;
        println!("Resampled to {} samples", samples.len());
    }

    Ok(samples)
}

pub fn transcribe_audio<P: AsRef<Path>>(path: P) -> Result<Vec<TranscriptionSegment>> {
    println!("Starting transcription process...");
    
    // Load the audio
    let audio_samples = load_audio_for_whisper(&path)?;
    
    // Load the model
    println!("Loading Whisper model...");
    let ctx = WhisperContext::new("whisper-base.bin")
        .map_err(|e| anyhow!("Failed to load Whisper model: {}", e))?;
    
    // Configure parameters
    println!("Configuring Whisper parameters...");
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_language(Some("en"));
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_timestamps(true);
    params.set_token_timestamps(true);
    params.set_duration_ms(0);
    params.set_translate(false);
    params.set_no_context(true);
    params.set_single_segment(false);
    params.set_max_initial_ts(1.0);
    params.set_max_len(0);
    params.set_split_on_word(true);
    
    // Create state
    println!("Creating Whisper state...");
    let mut state = ctx.create_state()?;
    
    // Process the audio
    println!("Processing audio with Whisper ({} samples)...", audio_samples.len());
    match state.full(params, &audio_samples) {
        Ok(_) => println!("Successfully processed audio"),
        Err(e) => {
            println!("Error processing audio: {}", e);
            return Err(anyhow!("Failed to process audio: {}", e));
        }
    }
    
    // Get the number of segments
    let num_segments = match state.full_n_segments() {
        Ok(n) => {
            println!("Found {} segments", n);
            n
        },
        Err(e) => {
            println!("Error getting segments: {}", e);
            return Err(anyhow!("Failed to get segments: {}", e));
        }
    };
    
    let mut segments = Vec::new();
    
    // Process each segment
    for i in 0..num_segments {
        println!("Processing segment {}", i);
        
        let segment_text = state.full_get_segment_text(i)
            .map_err(|e| anyhow!("Failed to get segment text: {}", e))?;
        
        if segment_text.trim().is_empty() {
            println!("Segment {} is empty, skipping", i);
            continue;
        }
        
        let start = state.full_get_segment_t0(i)
            .map_err(|e| anyhow!("Failed to get segment start time: {}", e))? as f64 / 100.0;
        let end = state.full_get_segment_t1(i)
            .map_err(|e| anyhow!("Failed to get segment end time: {}", e))? as f64 / 100.0;
        
        println!("Segment {}: [{:.2}-{:.2}] {}", i, start, end, segment_text);
        
        segments.push(TranscriptionSegment {
            text: segment_text,
            start,
            end,
        });
    }
    
    if segments.is_empty() {
        println!("Warning: No transcription segments were generated!");
    } else {
        println!("Successfully generated {} transcription segments", segments.len());
    }
    
    Ok(segments)
} 