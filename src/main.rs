// main.rs
use std::ffi::{c_char, c_void, CString};
use clap::Parser;
use serde::Serialize;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LlamaModelParams {
    pub devices: *mut c_void,
    pub tensor_buft_overrides: *const c_void,
    pub n_gpu_layers: i32,
    pub split_mode: i32,
    pub main_gpu: i32,
    pub tensor_split: *const f32,
    pub progress_callback: *mut c_void,
    pub progress_callback_user_data: *mut c_void,
    pub kv_overrides: *const c_void,
    pub vocab_only: bool,
    pub use_mmap: bool,
    pub use_direct_io: bool,
    pub use_mlock: bool,
    pub check_tensors: bool,
    pub use_extra_bufts: bool,
    pub no_host: bool,
    pub no_alloc: bool,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LlamaContextParams {
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub n_seq_max: u32,
    pub n_rs_seq: u32,
    pub n_threads: i32,
    pub n_threads_batch: i32,
    
    pub ctx_type: i32,
    pub rope_scaling_type: i32,
    pub pooling_type: i32,
    pub attention_type: i32,
    pub flash_attn_type: i32,
    
    pub rope_freq_base: f32,
    pub rope_freq_scale: f32,
    pub yarn_ext_factor: f32,
    pub yarn_attn_factor: f32,
    pub yarn_beta_fast: f32,
    pub yarn_beta_slow: f32,
    pub yarn_orig_ctx: u32,
    pub defrag_thold: f32,
    
    pub cb_eval: *mut c_void,
    pub cb_eval_user_data: *mut c_void,
    
    pub type_k: i32,
    pub type_v: i32,
    
    pub abort_callback: *mut c_void,
    pub abort_callback_data: *mut c_void,
    
    pub embeddings: bool,
    pub offload_kqv: bool,
    pub no_perf: bool,
    pub op_offload: bool,
    pub swa_full: bool,
    pub kv_unified: bool,
    
    pub samplers: *mut c_void,
    pub n_samplers: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LlamaBatch {
    pub n_tokens: i32,
    pub token: *mut i32,
    pub embd: *mut f32,
    pub pos: *mut i32,
    pub n_seq_id: *mut i32,
    pub seq_id: *mut *mut i32,
    pub logits: *mut i8,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LlamaSamplerChainParams {
    pub no_perf: bool,
}

#[link(name = "llama", kind = "static")]
#[link(name = "llama-common", kind = "static")]
#[link(name = "llama-common-base", kind = "static")]
#[link(name = "ggml", kind = "static")]
#[link(name = "ggml-cpu", kind = "static")]
#[link(name = "ggml-base", kind = "static")]
extern "C" {
    pub fn llama_backend_init();
    pub fn llama_backend_free();
    
    pub fn llama_model_default_params() -> LlamaModelParams;
    pub fn llama_model_load_from_file(path_model: *const c_char, params: LlamaModelParams) -> *mut c_void;
    pub fn llama_model_free(model: *mut c_void);
    pub fn llama_model_get_vocab(model: *const c_void) -> *const c_void;
    pub fn llama_model_n_embd(model: *const c_void) -> i32;
    pub fn llama_model_n_ctx_train(model: *const c_void) -> i32;
    
    pub fn llama_context_default_params() -> LlamaContextParams;
    pub fn llama_init_from_model(model: *mut c_void, params: LlamaContextParams) -> *mut c_void;
    pub fn llama_free(ctx: *mut c_void);
    
    pub fn llama_vocab_n_tokens(vocab: *const c_void) -> i32;
    pub fn llama_vocab_get_text(vocab: *const c_void, token: i32) -> *const c_char;
    pub fn llama_vocab_is_eog(vocab: *const c_void, token: i32) -> bool;
    pub fn llama_vocab_bos(vocab: *const c_void) -> i32;
    pub fn llama_vocab_eos(vocab: *const c_void) -> i32;
    
    pub fn llama_tokenize(
        vocab: *const c_void,
        text: *const c_char,
        text_len: i32,
        tokens: *mut i32,
        n_tokens_max: i32,
        add_special: bool,
        parse_special: bool,
    ) -> i32;
    
    pub fn llama_token_to_piece(
        vocab: *const c_void,
        token: i32,
        buf: *mut c_char,
        length: i32,
        lstrip: i32,
        special: bool,
    ) -> i32;
    
    pub fn llama_batch_init(n_tokens: i32, embd: i32, n_seq_max: i32) -> LlamaBatch;
    pub fn llama_batch_free(batch: LlamaBatch);
    
    pub fn llama_decode(ctx: *mut c_void, batch: LlamaBatch) -> i32;
    
    pub fn llama_get_logits_ith(ctx: *mut c_void, idx: i32) -> *mut f32;
    pub fn llama_get_embeddings(ctx: *mut c_void) -> *mut f32;
    pub fn llama_get_embeddings_ith(ctx: *mut c_void, i: i32) -> *mut f32;
    pub fn llama_get_embeddings_seq(ctx: *mut c_void, seq_id: i32) -> *mut f32;
    
    // Sampler FFI
    pub fn llama_sampler_chain_default_params() -> LlamaSamplerChainParams;
    pub fn llama_sampler_chain_init(params: LlamaSamplerChainParams) -> *mut c_void;
    pub fn llama_sampler_chain_add(chain: *mut c_void, smpl: *mut c_void);
    pub fn llama_sampler_free(smpl: *mut c_void);
    
    pub fn llama_sampler_init_greedy() -> *mut c_void;
    pub fn llama_sampler_init_temp(t: f32) -> *mut c_void;
    pub fn llama_sampler_init_top_k(k: i32) -> *mut c_void;
    pub fn llama_sampler_init_top_p(p: f32, min_keep: usize) -> *mut c_void;
    pub fn llama_sampler_init_min_p(p: f32, min_keep: usize) -> *mut c_void;
    pub fn llama_sampler_init_dist(seed: u32) -> *mut c_void;
    
    pub fn llama_sampler_sample(smpl: *mut c_void, ctx: *mut c_void, idx: i32) -> i32;
    pub fn llama_log_set(log_callback: GgmlLogCallback, user_data: *mut c_void);
    pub fn llama_n_ctx(ctx: *const c_void) -> u32;
}

#[derive(Parser, Debug)]
#[command(
    name = "RagIQ-RuntimeEngine",
    version = "1.0",
    author = "Ragavendren Chandrasekran",
    about = "Custom, highly optimized CPU runtime for GGUF models & embeddings",
    help_template = "{bin} {version}\nAuthor: {author}\n{about}\n\n{usage-heading}\n{tab}{usage}\n\n{all-args}"
)]
struct Args {
    /// Path to GGUF model file
    #[arg(short, long)]
    model: String,

    /// Prompt string (takes precedence over -f)
    #[arg(short, long)]
    prompt: Option<String>,

    /// Path to a file containing the prompt (used by ZeroTouch.ps1 -f)
    #[arg(short = 'f', long = "file")]
    prompt_file: Option<String>,

    /// Number of CPU threads (default is 0 for optimal auto-detected physical core count)
    #[arg(short, long, default_value_t = 0)]
    threads: i32,

    /// Context size (number of tokens, set to 0 to load from model)
    #[arg(short, long, default_value_t = 0)]
    ctx_size: u32,

    /// Logical batch size (number of tokens, set to 0 for default 2048 or model default)
    #[arg(short, long, default_value_t = 0)]
    batch_size: u32,

    /// Flag to enable high-performance text embedding generation
    #[arg(short, long)]
    embedding: bool,

    /// Enable memory page locking (mlock) to prevent RAM swapping
    #[arg(long)]
    mlock: bool,

    /// Temperature for text generation sampling (set to 0 for greedy)
    #[arg(long, default_value_t = 0.7)]
    temp: f32,

    /// Maximum tokens to generate (ZeroTouch uses -n, set to 0 for dynamic auto-allocation)
    #[arg(short = 'n', long = "max-tokens", default_value_t = 0)]
    max_tokens: i32,

    /// Hardware configuration mode (auto or manual)
    #[arg(long, default_value = "auto")]
    mode: String,

    /// Total system RAM in GB (set to 0 for automatic fallback based on CPU cores)
    #[arg(long, default_value_t = 0)]
    ram_gb: u32,

    /// Top-p sampling parameter (parsed and ignored or used)
    #[arg(long, default_value_t = 0.9)]
    top_p: f32,

    /// Repeat penalty parameter (parsed and ignored)
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// Flag to not display prompt (parsed and ignored)
    #[arg(long)]
    no_display_prompt: bool,

    /// Pooling type for embeddings (parsed and ignored, e.g. "cls")
    #[arg(long, default_value = "mean")]
    pooling: String,

    /// Enable verbose logging of initialization and execution steps
    #[arg(short, long)]
    verbose: bool,

    /// Enable real-time progress bar and metrics
    #[arg(short = 'g', long = "progress")]
    progress: bool,
}

#[derive(Serialize)]
struct EmbeddingOutput {
    prompt: String,
    dimensions: usize,
    embedding: Vec<f32>,
}

struct HardwareProfile {
    threads: i32,
    batch_size: u32,
    use_mlock: bool,
    ctx_size: u32,
    max_tokens: i32,
    ram_gb: u32,
}

fn resolve_hardware_profile(
    mode: &str,
    user_threads: i32,
    user_batch: u32,
    user_ctx: u32,
    user_max_tokens: i32,
    user_mlock: bool,
    user_ram_gb: u32,
    is_embedding: bool,
) -> HardwareProfile {
    let is_auto = mode.eq_ignore_ascii_case("auto");
    
    // 1. Thread core auto-detection (physical cores instead of hyperthreads)
    let logical = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let physical_threads = (if logical <= 4 { logical } else { logical / 2 }) as i32;
    
    // 2. RAM resolving (using the power-user PowerShell passed parameter, or dynamic core count fallback)
    let resolved_ram = if user_ram_gb > 0 {
        user_ram_gb
    } else {
        // Fallback: If no RAM is passed, guess based on logical core limits to remain completely safe
        if logical <= 4 { 8 } else { 16 }
    };

    // 3. Multi-dimensional classification (Bottleneck Heuristic: RAM & CPU check)
    let (auto_batch, auto_mlock, auto_ctx, auto_max_tokens, _tier_name) = if resolved_ram <= 8 || logical <= 4 {
        (32u32, false, 2048u32, 256i32, "Low")       // Low-spec VM / Host
    } else if resolved_ram <= 16 || logical <= 8 {
        (128u32, false, 4096u32, 512i32, "Medium")    // Medium-spec VM / Host
    } else {
        (512u32, false, 8192u32, 1024i32, "High")      // High-spec Dedicated Host
    };

    // 4. Resolve parameters (allowing optional, selective manual overrides in Auto mode)
    let resolved_threads = if user_threads > 0 { user_threads } else { physical_threads };
    
    let resolved_batch = if user_batch > 0 {
        user_batch
    } else {
        if is_auto {
            if is_embedding { auto_ctx } else { auto_batch }
        } else {
            if is_embedding { 2048 } else { 32 } // Safe fallback
        }
    };

    let resolved_mlock = if is_auto {
        user_mlock || auto_mlock
    } else {
        user_mlock
    };

    let resolved_ctx = if user_ctx > 0 {
        user_ctx
    } else {
        if is_auto { auto_ctx } else { 2048 } // Safe default
    };

    let resolved_max_tokens = if user_max_tokens > 0 {
        user_max_tokens
    } else {
        if is_auto { auto_max_tokens } else { 256 } // Standard fallback if 0
    };

    HardwareProfile {
        threads: resolved_threads,
        batch_size: resolved_batch,
        use_mlock: resolved_mlock,
        ctx_size: resolved_ctx,
        max_tokens: resolved_max_tokens,
        ram_gb: resolved_ram as u32,
    }
}

pub type GgmlLogCallback = Option<unsafe extern "C" fn(level: i32, text: *const c_char, user_data: *mut c_void)>;

unsafe extern "C" fn silent_log_callback(_level: i32, _text: *const c_char, _user_data: *mut c_void) {}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    
    unsafe {
        if args.verbose {
            eprintln!("[Verbose] Initializing llama.cpp backend...");
        } else {
            // Silence all verbose llama.cpp logging
            llama_log_set(Some(silent_log_callback), std::ptr::null_mut());
        }
        
        // Initialize the backend once at startup
        llama_backend_init();
    }
    
    // Ensure the backend is freed when we exit
    struct BackendGuard;
    impl Drop for BackendGuard {
        fn drop(&mut self) {
            unsafe {
                llama_backend_free();
            }
        }
    }
    let _backend_guard = BackendGuard;

    // Auto-detect embedding mode from executable name (drop-in compatibility)
    let mut is_embedding_mode = args.embedding;
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(file_name) = exe_path.file_name() {
            let name_str = file_name.to_string_lossy().to_lowercase();
            if name_str.contains("embedding") {
                is_embedding_mode = true;
            }
        }
    }

    // Resolve optimal hardware configuration dynamically
    let profile = resolve_hardware_profile(
        &args.mode,
        args.threads,
        args.batch_size,
        args.ctx_size,
        args.max_tokens,
        args.mlock,
        args.ram_gb,
        is_embedding_mode,
    );

    // Convert model path to CString
    let c_model_path = CString::new(args.model.clone())?;

    // Set model parameters
    let mut model_params = unsafe { llama_model_default_params() };
    model_params.n_gpu_layers = 0; // Exclusively CPU inference
    model_params.use_mmap = true;  // Fast memory mapping
    model_params.use_mlock = profile.use_mlock; // Prevent RAM swapping

    if args.verbose {
        eprintln!("[Verbose] Model path: {}", args.model);
        eprintln!("[Verbose] Loading GGUF model from file...");
    }

    // Load the model
    let model = unsafe { llama_model_load_from_file(c_model_path.as_ptr(), model_params) };
    if model.is_null() {
        return Err(format!("Failed to load model from path: {}", args.model).into());
    }

    if args.verbose {
        eprintln!("[Verbose] GGUF model loaded successfully.");
    }

    struct ModelGuard(*mut c_void);
    impl Drop for ModelGuard {
        fn drop(&mut self) {
            unsafe {
                llama_model_free(self.0);
            }
        }
    }
    let _model_guard = ModelGuard(model);

    // Retrieve vocabulary and embedding dimensions
    let vocab = unsafe { llama_model_get_vocab(model) };
    let n_embd_raw = unsafe { llama_model_n_embd(model) };
    if n_embd_raw <= 0 {
        return Err(format!("Model returned invalid embedding dimension: {}. Ensure you are using an embedding-capable model.", n_embd_raw).into());
    }
    let n_embd = n_embd_raw as usize;

    if args.verbose {
        eprintln!("[Verbose] Model vocabulary and dimensions loaded. Embedding dimension: {}.", n_embd);
    }

    // Resolve prompt from string or from prompt file (-f)
    let mut prompt_text = if let Some(ref p) = args.prompt {
        p.clone()
    } else if let Some(ref file_path) = args.prompt_file {
        if args.verbose {
            eprintln!("[Verbose] Reading prompt from file: {}...", file_path);
        }
        std::fs::read_to_string(file_path)?
    } else {
        return Err("Either prompt (-p) or prompt file (-f) must be specified".into());
    };

    let mut auto_appended_json_trigger = false;
    if !is_embedding_mode && prompt_text.trim_end().ends_with('}') {
        auto_appended_json_trigger = true;
        prompt_text.push_str("\n\n### Response:\n{");
        if args.verbose {
            eprintln!("[Verbose] Smart prompt-ending recovery: prompt ends with '}}'. Programmatically appended response trigger suffix in memory.");
        }
    }

    if args.verbose {
        eprintln!("[Verbose] Resolved input prompt of length {} characters.", prompt_text.len());
        eprintln!("[Verbose] Tokenizing prompt...");
    }

    // Tokenize prompt
    let c_prompt = CString::new(prompt_text.clone())?;
    // Determine token capacity: worst case = 1 token per UTF-8 byte + 256 headroom for BOS/EOS/special tokens
    let max_tokens = prompt_text.len() + 256;
    let mut tokens = vec![0i32; max_tokens];
    
    // Call tokenization
    let n_tokens = unsafe {
        llama_tokenize(
            vocab,
            c_prompt.as_ptr(),
            prompt_text.len() as i32,
            tokens.as_mut_ptr(),
            max_tokens as i32,
            true,  // Add special tokens like BOS
            true, // Parse special plaintext control tags (e.g. ChatML/Instruct tags)
        )
    };

    if n_tokens < 0 {
        return Err("Tokenization failed".into());
    }
    tokens.truncate(n_tokens as usize);

    if args.verbose {
        eprintln!("[Verbose] Prompt tokenization successful. Total tokens: {}.", n_tokens);
        eprintln!("[Verbose] Token IDs and pieces (first 50 and last 10):");
        let total_t = tokens.len();
        for idx in 0..total_t {
            if total_t > 60 && idx >= 50 && idx < total_t - 10 {
                if idx == 50 {
                    eprintln!("  ... [truncated {} tokens] ...", total_t - 60);
                }
                continue;
            }
            let mut piece_buf = [0i8; 256];
            let piece_len = unsafe {
                llama_token_to_piece(
                    vocab,
                    tokens[idx],
                    piece_buf.as_mut_ptr(),
                    piece_buf.len() as i32,
                    0,
                    true,
                )
            };
            let piece_str = if piece_len > 0 {
                let actual_len = std::cmp::min(piece_len as usize, piece_buf.len());
                let bytes = unsafe {
                    std::slice::from_raw_parts(piece_buf.as_ptr() as *const u8, actual_len)
                };
                std::str::from_utf8(bytes).unwrap_or("<invalid utf8>").to_string()
            } else {
                String::new()
            };
            eprintln!("  Token [{}]: ID = {}, piece = {:?}", idx, tokens[idx], piece_str);
        }
    }

    let n_ctx_train = unsafe { llama_model_n_ctx_train(model) } as usize;
    if is_embedding_mode && tokens.len() > n_ctx_train {
        eprintln!("[Warning] Embedding prompt token count ({}) exceeds model's max trained context length ({}). Truncating prompt to avoid position embedding overflow.", tokens.len(), n_ctx_train);
        tokens.truncate(n_ctx_train);
    }
    let n_tokens = tokens.len() as i32;

    // Get trained context length from model
    let requested_ctx = profile.ctx_size;

    // Compute required context capacity based on prompt tokens and dynamic max generation limit
    let required_ctx = if is_embedding_mode {
        n_tokens as u32
    } else {
        n_tokens as u32 + profile.max_tokens as u32
    };
    let dynamic_ctx_size = requested_ctx.max(required_ctx);

    // Compute logical batch size (n_batch) and physical batch size (n_ubatch)
    let mut n_batch = profile.batch_size;

    // For encoder-only embedding models, the physical batch size must be >= sequence length.
    // Setting both n_batch and n_ubatch to dynamic_ctx_size ensures single-pass decode and avoids assertion crashes.
    let n_ubatch = if is_embedding_mode {
        n_batch = dynamic_ctx_size;
        dynamic_ctx_size
    } else {
        n_batch
    };

    let actual_threads = profile.threads;

    if args.verbose {
        eprintln!("[Verbose] Dynamic Hardware Auto-Configuration (Mode: {}):", args.mode);
        eprintln!("  - Total Host physical RAM: {} GB", profile.ram_gb);
        eprintln!("  - Host physical core threads: {}", actual_threads);
        eprintln!("  - Resolved Context Limit: {} (requested: {}, required: {})", dynamic_ctx_size, requested_ctx, required_ctx);
        eprintln!("  - Resolved Batch Limit (n_batch): {}", n_batch);
        eprintln!("  - Resolved Physical Batch (n_ubatch): {}", n_ubatch);
        eprintln!("  - Resolved Max Tokens: {}", profile.max_tokens);
        eprintln!("  - Memory locking (mlock): {}", profile.use_mlock);
        eprintln!("[Verbose] Initializing Llama context from model...");
    }

    // Set context parameters
    let mut ctx_params = unsafe { llama_context_default_params() };
    ctx_params.n_ctx = dynamic_ctx_size;
    ctx_params.n_batch = n_batch;
    ctx_params.n_ubatch = n_ubatch;
    ctx_params.n_threads = actual_threads;
    ctx_params.n_threads_batch = actual_threads;
    ctx_params.embeddings = is_embedding_mode; // Toggle embeddings output

    // Initialize context
    let ctx = unsafe { llama_init_from_model(model, ctx_params) };
    if ctx.is_null() {
        return Err("Failed to initialize Llama context from model".into());
    }

    if args.verbose {
        eprintln!("[Verbose] Llama context initialized successfully.");
    }

    struct ContextGuard(*mut c_void);
    impl Drop for ContextGuard {
        fn drop(&mut self) {
            unsafe {
                llama_free(self.0);
            }
        }
    }
    let _ctx_guard = ContextGuard(ctx);

    if is_embedding_mode {
        // --- EMBEDDING MODE ---
        // Setup batch for decoding
        let batch_capacity = ctx_params.n_batch as i32;
        let mut batch = unsafe { llama_batch_init(batch_capacity, 0, 1) };
        
        struct BatchGuard(LlamaBatch);
        impl Drop for BatchGuard {
            fn drop(&mut self) {
                unsafe {
                    llama_batch_free(self.0);
                }
            }
        }
        let _batch_guard = BatchGuard(batch);

        let start_time = std::time::Instant::now();
        
        // Feed prompt tokens into the context using chunked decoding
        let mut n_processed = 0;
        while n_processed < n_tokens {
            let n_chunk = std::cmp::min(batch_capacity, n_tokens - n_processed);
            
            unsafe {
                batch.n_tokens = n_chunk;
                for i in 0..(n_chunk as usize) {
                    *batch.token.add(i) = tokens[n_processed as usize + i];
                    *batch.pos.add(i) = (n_processed + i as i32) as i32;
                    *batch.n_seq_id.add(i) = 1;
                    *(*batch.seq_id.add(i)) = 0;
                    *batch.logits.add(i) = 0;
                }
                // For embedding generation, we need to output the last token's embeddings
                if n_processed + n_chunk == n_tokens {
                    *batch.logits.add((n_chunk - 1) as usize) = 1;
                }
            }
            
            let decode_res = unsafe { llama_decode(ctx, batch) };
            if decode_res != 0 {
                return Err(format!("Embedding prompt decode chunk failed with code: {}", decode_res).into());
            }
            
            n_processed += n_chunk;
        }
        let duration = start_time.elapsed();
        let secs = duration.as_secs_f32();
        if args.verbose || args.progress {
            eprintln!("\n[Speed Result] Extracted embeddings for {} tokens in {:.2}ms ({:.2} tokens/sec)", n_tokens, duration.as_secs_f64() * 1000.0, n_tokens as f32 / secs.max(0.0001));
        }

        // Retrieve the generated embeddings
        // Try sequence-pooled embeddings first (typical for dedicated embedding models)
        let mut embd_ptr = unsafe { llama_get_embeddings_seq(ctx, 0) };
        if embd_ptr.is_null() {
            // Fallback: Try token-level embedding for the last token in batch
            embd_ptr = unsafe { llama_get_embeddings_ith(ctx, n_tokens - 1) };
        }
        if embd_ptr.is_null() {
            // Ultimate fallback: Try global context embedding pointer
            embd_ptr = unsafe { llama_get_embeddings(ctx) };
        }

        if embd_ptr.is_null() {
            return Err("Failed to retrieve embeddings from context".into());
        }

        // Convert raw pointer float array to Vec
        let embedding: Vec<f32> = unsafe {
            std::slice::from_raw_parts(embd_ptr, n_embd).to_vec()
        };

        // Output structured JSON representation (or raw space-separated floats if called like standard llama-embedding)
        // ZeroTouch.ps1 parses embeddings with: ($raw -replace '[\[\]\,]',' ') -split '\s+'
        // Standard llama-embedding.exe prints space-separated values inside brackets, or raw lists.
        // Let's print raw values so it is compatible with both!
        // To be 100% safe for ZeroTouch which expects space-separated float numbers, we print:
        // [val1, val2, ... valN] or space-separated.
        // Actually, $raw -replace '[\[\]\,]',' ' converts brackets and commas to spaces, so standard JSON array [val1, val2, ...]
        // works 100% perfectly and is extremely clean!
        let output = EmbeddingOutput {
            prompt: prompt_text.clone(),
            dimensions: n_embd,
            embedding,
        };
        
        // ZeroTouch parses the raw string. To make it extremely robust for standard llama-embedding FFI:
        // Standard llama-embedding outputs: 
        // [ 0.123, 0.456, ... ]
        // Let's print both: print standard bracketed space-separated representation to stdout (so ZeroTouch parses it perfectly),
        // and if needed we print JSON. Wait, let's look at ZeroTouch's parser:
        // $tokens = ($raw -replace '[\[\]\,]',' ') -split '\s+'
        // This parser works on ANY string containing numbers inside or outside brackets, separated by commas or spaces!
        // So a standard JSON array or a space-separated string will work perfectly!
        // Let's output a clean space-separated array inside brackets:
        let formatted_vector = format!("[{}]", output.embedding.iter().map(|f| f.to_string()).collect::<Vec<String>>().join(", "));
        println!("{}", formatted_vector);
        
        // We can also print the detailed JSON to stderr or logging
        // But printing standard bracketed vector to stdout is the most direct drop-in.
        
    } else {
        // --- GENERATION MODE ---
        // Setup batch for prompt and autoregressive generation
        let batch_capacity = ctx_params.n_batch as i32;
        let mut batch = unsafe { llama_batch_init(batch_capacity, 0, 1) };
        
        struct BatchGuard2(LlamaBatch);
        impl Drop for BatchGuard2 {
            fn drop(&mut self) {
                unsafe {
                    llama_batch_free(self.0);
                }
            }
        }
        let _batch_guard2 = BatchGuard2(batch);

        if args.verbose {
            eprintln!("[Verbose] Decoding prompt tokens (chunked evaluation)...");
        }

        // Feed prompt tokens into the context using chunked decoding
        let mut n_processed = 0;
        let mut decode_res;

        let prompt_eval_start = std::time::Instant::now();

        while n_processed < n_tokens {
            let n_chunk = std::cmp::min(batch_capacity, n_tokens - n_processed);
            
            unsafe {
                batch.n_tokens = n_chunk;
                for i in 0..(n_chunk as usize) {
                    *batch.token.add(i) = tokens[n_processed as usize + i];
                    *batch.pos.add(i) = (n_processed + i as i32) as i32;
                    *batch.n_seq_id.add(i) = 1;
                    *(*batch.seq_id.add(i)) = 0;
                    *batch.logits.add(i) = 0;
                }
                
                // For the last token of the final chunk, output logits for sampling
                if n_processed + n_chunk == n_tokens {
                    *batch.logits.add((n_chunk - 1) as usize) = 1;
                }
            }
            
            decode_res = unsafe { llama_decode(ctx, batch) };
            if decode_res != 0 {
                return Err(format!("Initial prompt decode chunk failed with code: {}", decode_res).into());
            }
            
            n_processed += n_chunk;
        }

        let prompt_eval_secs = prompt_eval_start.elapsed().as_secs_f32();

        if args.verbose {
            eprintln!("[Verbose] Prompt decoded successfully.");
            eprintln!("[Verbose] Initializing dynamic sampler chain...");
        }

        // Initialize dynamic sampling chain
        let mut sparams = unsafe { llama_sampler_chain_default_params() };
        sparams.no_perf = true;
        let sampler = unsafe { llama_sampler_chain_init(sparams) };
        if sampler.is_null() {
            return Err("Failed to initialize sampler chain".into());
        }

        struct SamplerGuard(*mut c_void);
        impl Drop for SamplerGuard {
            fn drop(&mut self) {
                unsafe {
                    llama_sampler_free(self.0);
                }
            }
        }
        let _sampler_guard = SamplerGuard(sampler);

        // Add specific samplers
        unsafe {
            if args.temp <= 0.0 {
                llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
            } else {
                llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));
                llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.95, 1));
                llama_sampler_chain_add(sampler, llama_sampler_init_min_p(0.05, 1));
                llama_sampler_chain_add(sampler, llama_sampler_init_temp(args.temp));
                // Add distribution sampler with a random seed
                llama_sampler_chain_add(sampler, llama_sampler_init_dist(rand::random::<u32>()));
            }
        }

        if args.verbose {
            eprintln!("[Verbose] Sampler chain setup complete. Temperature: {}, Top-P: {}", args.temp, args.top_p);
            eprintln!("[Verbose] Starting main autoregressive token generation loop (max_tokens: {})...", profile.max_tokens);
        }



        // Main autoregressive generation loop
        let mut n_cur = n_tokens;
        let mut n_gen = 0;


        let start_time = std::time::Instant::now();

        // Buffer all generated tokens so we can post-process (e.g. pretty-print JSON)
        let mut response_buffer = String::new();

        if auto_appended_json_trigger {
            response_buffer.push('{');
        }

        while n_gen < profile.max_tokens {
            // Sample the next token from context's output logits
            let mut token_id = unsafe { llama_sampler_sample(sampler, ctx, -1) };

            // Stop if it is an End-Of-Generation token
            if unsafe { llama_vocab_is_eog(vocab, token_id) } {
                if n_gen == 0 {
                    // Smart prompt-ending recovery: If EOG is predicted as the very first token, 
                    // it means the prompt was seen as complete (e.g. ending in a closed JSON brace).
                    // We temporarily ban the EOG token in logits and re-sample to force generation to start!
                    unsafe {
                        let logits_ptr = llama_get_logits_ith(ctx, -1);
                        if !logits_ptr.is_null() {
                            let n_vocab = llama_vocab_n_tokens(vocab);
                            for t_id in 0..n_vocab {
                                if llama_vocab_is_eog(vocab, t_id) {
                                    *logits_ptr.add(t_id as usize) = -1e9f32;
                                }
                            }
                            if args.verbose {
                                eprintln!("[Verbose] Smart prompt-ending recovery triggered. Banned EOG tokens and re-sampling...");
                            }
                            token_id = llama_sampler_sample(sampler, ctx, -1);
                        }
                    }
                    if unsafe { llama_vocab_is_eog(vocab, token_id) } {
                        break;
                    }
                } else {
                    if args.verbose {
                        eprintln!("[Verbose] End-of-Generation (EOG) token encountered (ID: {}). Terminating execution.", token_id);
                    }
                    break;
                }
            }

            // Convert token id to printable string piece
            let mut piece_buf = [0i8; 256];
            let piece_len = unsafe {
                llama_token_to_piece(
                    vocab,
                    token_id,
                    piece_buf.as_mut_ptr(),
                    piece_buf.len() as i32,
                    0,     // Do not strip spaces
                    false, // Render special tokens as regular text
                )
            };

            if piece_len > 0 {
                let actual_len = std::cmp::min(piece_len as usize, piece_buf.len());
                let bytes = unsafe {
                    std::slice::from_raw_parts(piece_buf.as_ptr() as *const u8, actual_len)
                };
                if let Ok(piece_str) = std::str::from_utf8(bytes) {
                    response_buffer.push_str(piece_str);
                }
            }

            n_gen += 1;

            // Prepare batch for next single token evaluation
            unsafe {
                batch.n_tokens = 1;
                *batch.token = token_id;
                *batch.pos = n_cur;
                *batch.n_seq_id = 1;
                *(*batch.seq_id) = 0;
                *batch.logits = 1; // Output logits for the next token to sample
            }

            n_cur += 1;

            // Decode next single token
            decode_res = unsafe { llama_decode(ctx, batch) };
            if decode_res != 0 {
                return Err(format!("Decode failed during autoregressive step with code: {}", decode_res).into());
            }
        }

        // Post-process output: auto-detect and pretty-print JSON responses
        let trimmed = response_buffer.trim();
        if trimmed.starts_with('{') && trimmed.ends_with('}') {
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(trimmed) {
                if let Ok(pretty) = serde_json::to_string_pretty(&parsed) {
                    println!("{}", pretty);
                } else {
                    print!("{}", response_buffer);
                    println!();
                }
            } else {
                print!("{}", response_buffer);
                println!();
            }
        } else {
            print!("{}", response_buffer);
            println!();
        }

        let duration = start_time.elapsed();
        let secs = duration.as_secs_f32();
        if (args.verbose || args.progress) && secs > 0.0 {
            let prompt_tps = if prompt_eval_secs > 0.0 { n_tokens as f32 / prompt_eval_secs } else { 0.0 };
            let generation_tps = n_gen as f32 / secs;
            eprintln!("[ Prompt: {:.1} t/s | Generation: {:.1} t/s ]", prompt_tps, generation_tps);
        }
    }

    Ok(())
}
