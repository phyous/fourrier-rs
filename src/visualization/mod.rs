use anyhow::Result;
use crossterm::terminal::{disable_raw_mode, enable_raw_mode};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Style};
use ratatui::symbols;
use ratatui::widgets::{Block, Borders, Dataset, GraphType, Chart, Paragraph, Wrap};
use ratatui::text::Span;
use ratatui::Terminal;
use std::io::stdout;
use std::time::Duration;

use crate::audio::{AudioData, SpectrogramData};
use crate::speech::TranscriptionSegment;

pub struct Visualizer {
    audio_data: AudioData,
    spectrogram: SpectrogramData,
    transcription: Vec<TranscriptionSegment>,
}

impl Visualizer {
    pub fn new(
        audio_data: AudioData,
        spectrogram: SpectrogramData,
        transcription: Vec<TranscriptionSegment>,
    ) -> Self {
        Self {
            audio_data,
            spectrogram,
            transcription,
        }
    }

    pub fn run(&self) -> Result<()> {
        enable_raw_mode()?;
        let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))?;
        terminal.clear()?;

        terminal.draw(|frame| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Percentage(30),
                    Constraint::Percentage(35),
                    Constraint::Percentage(35),
                ])
                .margin(1)
                .split(frame.size());

            self.draw_transcription(frame, chunks[0]);
            self.draw_waveform(frame, chunks[1]);
            self.draw_spectrogram(frame, chunks[2]);
        })?;

        // Wait briefly to show the visualization
        std::thread::sleep(Duration::from_secs(5));
        
        disable_raw_mode()?;
        terminal.clear()?;
        Ok(())
    }

    fn draw_transcription(&self, frame: &mut ratatui::Frame, area: Rect) {
        let text = self
            .transcription
            .iter()
            .map(|seg| {
                format!(
                    "[{:.2}s - {:.2}s] {}",
                    seg.start, seg.end, seg.text
                )
            })
            .collect::<Vec<_>>()
            .join("\n");

        let paragraph = Paragraph::new(text)
            .block(Block::default().title("Transcription").borders(Borders::ALL))
            .wrap(Wrap { trim: true });
        frame.render_widget(paragraph, area);
    }

    fn draw_waveform(&self, frame: &mut ratatui::Frame, area: Rect) {
        // Find the maximum amplitude for proper scaling
        let max_amplitude = self.audio_data.samples
            .iter()
            .map(|&x| x.abs())
            .fold(0.0f32, f32::max);

        // Scale factor to use most of the available space
        let scale = if max_amplitude > 0.0 { 0.95 / max_amplitude } else { 1.0 };

        // Calculate step size based on available width
        let points_per_column = (self.audio_data.samples.len() / area.width as usize).max(1);
        
        // Create data points with RMS values for better visualization
        let waveform_data: Vec<(f64, f64)> = self.audio_data.samples
            .chunks(points_per_column)
            .enumerate()
            .map(|(i, chunk)| {
                let rms = (chunk.iter().map(|&x| x * x).sum::<f32>() / chunk.len() as f32).sqrt();
                (
                    i as f64 * points_per_column as f64 / self.audio_data.sample_rate as f64,
                    (rms * scale) as f64,
                )
            })
            .collect();

        let duration = self.audio_data.samples.len() as f64 / self.audio_data.sample_rate as f64;
        let time_labels: Vec<Span> = (0..=5)
            .map(|i| {
                let time = duration * i as f64 / 5.0;
                Span::raw(format!("{:.1}s", time))
            })
            .collect();

        let datasets = vec![Dataset::default()
            .name("Waveform")
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::Cyan))
            .data(&waveform_data)];

        let chart = Chart::new(datasets)
            .block(Block::default().title("Waveform").borders(Borders::ALL))
            .x_axis(
                ratatui::widgets::Axis::default()
                    .title("Time (s)")
                    .bounds([0.0, duration])
                    .labels(time_labels)
            )
            .y_axis(
                ratatui::widgets::Axis::default()
                    .title("Amplitude")
                    .bounds([-1.0, 1.0])
                    .labels(vec!["-1.0", "-0.5", "0.0", "0.5", "1.0"].into_iter().map(Span::raw).collect())
            );

        frame.render_widget(chart, area);
    }

    fn draw_spectrogram(&self, frame: &mut ratatui::Frame, area: Rect) {
        let max_freq_idx = self.spectrogram.frequencies.len().min(100);
        let time_step = (self.spectrogram.time_points.len() / area.width as usize).max(1);
        
        // Create intensity-based points
        let mut points_by_intensity = vec![Vec::new(); 4]; // 4 intensity levels
        
        for t in (0..self.spectrogram.time_points.len()).step_by(time_step) {
            let time = self.spectrogram.time_points[t];
            for f in 0..max_freq_idx {
                let magnitude = self.spectrogram.magnitudes[t][f];
                let intensity = ((magnitude + 100.0) / 100.0).max(0.0).min(1.0);
                
                if intensity > 0.1 {
                    let intensity_level = (intensity * 3.99) as usize;
                    points_by_intensity[intensity_level].push((
                        time as f64,
                        self.spectrogram.frequencies[f] as f64,
                    ));
                }
            }
        }

        let colors = [Color::Blue, Color::Green, Color::Yellow, Color::Red];
        let mut datasets = Vec::new();
        
        for (intensity_level, points) in points_by_intensity.iter().enumerate() {
            if !points.is_empty() {
                datasets.push(
                    Dataset::default()
                        .marker(symbols::Marker::Block)
                        .graph_type(GraphType::Scatter)
                        .style(Style::default().fg(colors[intensity_level]))
                        .data(points)
                );
            }
        }

        let duration = *self.spectrogram.time_points.last().unwrap_or(&0.0) as f64;
        let max_freq = self.spectrogram.frequencies[max_freq_idx - 1];
        
        let time_labels: Vec<Span> = (0..=5)
            .map(|i| Span::raw(format!("{:.1}s", duration * i as f64 / 5.0)))
            .collect();
            
        let freq_labels: Vec<Span> = (0..=4)
            .map(|i| Span::raw(format!("{:.0}Hz", max_freq * i as f32 / 4.0)))
            .collect();

        let chart = Chart::new(datasets)
            .block(Block::default().title("Spectrogram").borders(Borders::ALL))
            .x_axis(
                ratatui::widgets::Axis::default()
                    .title("Time (s)")
                    .bounds([0.0, duration])
                    .labels(time_labels)
            )
            .y_axis(
                ratatui::widgets::Axis::default()
                    .title("Frequency (Hz)")
                    .bounds([0.0, max_freq as f64])
                    .labels(freq_labels)
            );

        frame.render_widget(chart, area);
    }
} 