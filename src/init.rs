use ctor::ctor;

#[ctor]
fn init() {
    // Suppress Whisper output by redirecting stdout during initialization
    let stdout = std::io::stdout();
    let _handle = stdout.lock();
    
    // Set environment variables
    std::env::set_var("WHISPER_PRINT_DEBUG", "0");
    std::env::set_var("WHISPER_PRINT_PROGRESS", "0");
    
    // Configure logging
    env_logger::Builder::new()
        .filter_level(log::LevelFilter::Error)
        .init();
} 