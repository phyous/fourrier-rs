use anyhow::Result;
use rustfft::{FftPlanner, num_complex::Complex};
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use symphonia::core::audio::Signal;
use std::fs::File;
use std::path::Path;

pub struct AudioData {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

pub struct SpectrogramData {
    pub time_points: Vec<f32>,
    pub frequencies: Vec<f32>,
    pub magnitudes: Vec<Vec<f32>>,
}

pub fn load_audio<P: AsRef<Path>>(path: P) -> Result<AudioData> {
    let file = File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let hint = Hint::new();
    let format_opts = FormatOptions::default();
    let metadata_opts = MetadataOptions::default();
    let decoder_opts = DecoderOptions::default();

    let probed = symphonia::default::get_probe().format(&hint, mss, &format_opts, &metadata_opts)?;
    let mut format = probed.format;
    let track = format.default_track().unwrap();
    let mut decoder = symphonia::default::get_codecs().make(&track.codec_params, &decoder_opts)?;

    let mut samples = Vec::new();
    let sample_rate = track.codec_params.sample_rate.unwrap();

    while let Ok(packet) = format.next_packet() {
        let decoded = decoder.decode(&packet)?;
        match decoded {
            symphonia::core::audio::AudioBufferRef::F32(buf) => {
                samples.extend_from_slice(buf.chan(0));
            },
            symphonia::core::audio::AudioBufferRef::F64(buf) => {
                samples.extend(buf.chan(0).iter().map(|&x| x as f32));
            },
            symphonia::core::audio::AudioBufferRef::U8(buf) => {
                samples.extend(buf.chan(0).iter().map(|&x| (x as f32 / 128.0) - 1.0));
            },
            symphonia::core::audio::AudioBufferRef::U16(buf) => {
                samples.extend(buf.chan(0).iter().map(|&x| (x as f32 / 32768.0) - 1.0));
            },
            symphonia::core::audio::AudioBufferRef::U24(buf) => {
                samples.extend(buf.chan(0).iter().map(|&x| {
                    let value = x.inner() as u32;
                    (value as f32 / 8388608.0) - 1.0
                }));
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
            symphonia::core::audio::AudioBufferRef::S24(buf) => {
                samples.extend(buf.chan(0).iter().map(|&x| {
                    let value = x.inner() as i32;
                    value as f32 / 8388608.0
                }));
            },
            symphonia::core::audio::AudioBufferRef::S32(buf) => {
                samples.extend(buf.chan(0).iter().map(|&x| x as f32 / 2147483648.0));
            },
        }
    }

    Ok(AudioData {
        samples,
        sample_rate,
    })
}

pub fn compute_spectrogram(audio_data: &AudioData, window_size: usize) -> Result<SpectrogramData> {
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(window_size);
    
    let hop_size = window_size / 2;
    let num_frames = (audio_data.samples.len() - window_size) / hop_size;
    
    let mut magnitudes = Vec::with_capacity(num_frames);
    let mut time_points = Vec::with_capacity(num_frames);
    
    let window = hann_window(window_size);
    
    for frame_idx in 0..num_frames {
        let start = frame_idx * hop_size;
        let mut frame: Vec<Complex<f32>> = audio_data.samples[start..start + window_size]
            .iter()
            .zip(window.iter())
            .map(|(&s, &w)| Complex::new(s * w, 0.0))
            .collect();
            
        fft.process(&mut frame);
        
        let magnitude: Vec<f32> = frame[..window_size/2]
            .iter()
            .map(|c| (c.norm() / window_size as f32).log10() * 20.0)
            .collect();
            
        magnitudes.push(magnitude);
        time_points.push(start as f32 / audio_data.sample_rate as f32);
    }
    
    let frequencies: Vec<f32> = (0..window_size/2)
        .map(|i| i as f32 * audio_data.sample_rate as f32 / window_size as f32)
        .collect();
        
    Ok(SpectrogramData {
        time_points,
        frequencies,
        magnitudes,
    })
}

fn hann_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (size - 1) as f32).cos()))
        .collect()
} 