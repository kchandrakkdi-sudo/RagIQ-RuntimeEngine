// src/lib.rs
// C-compatible Dynamic Link Library implementation of RagIQ-RuntimeEngine

use std::ffi::{c_char, c_void, CStr, CString};

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

pub type GgmlLogCallback = Option<unsafe extern "C" fn(level: i32, text: *const c_char, user_data: *mut c_void)>;

unsafe extern "C" fn silent_log_callback(_level: i32, _text: *const c_char, _user_data: *mut c_void) {}

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

    // 4. Resolve parameters (allowing optional overrides in Auto mode)
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

// Internal helper to convert Rust CStr safely to rust String
unsafe fn to_rust_string(ptr: *const c_char) -> String {
    if ptr.is_null() {
        return String::new();
    }
    CStr::from_ptr(ptr).to_string_lossy().into_owned()
}

/// Safe CString builder for FFI returns â€” replaces interior null bytes with spaces
/// instead of panicking, since panics across FFI boundaries are undefined behavior.
fn safe_cstring(s: String) -> CString {
    let sanitized: Vec<u8> = s.into_bytes().into_iter()
        .map(|b| if b == 0 { b' ' } else { b })
        .collect();
    CString::new(sanitized).unwrap_or_else(|_| CString::new("").unwrap())
}

/// Safe CString builder for known-safe static error strings (no null bytes possible).
#[inline]
fn err_cstr(msg: &'static str) -> *mut c_char {
    CString::new(msg).unwrap().into_raw()
}

/// Unified entry point to run a single inference on-demand (loads and frees memory).
/// Returns a raw C-string pointer that MUST be freed using `free_string()`.
#[no_mangle]
pub unsafe extern "C" fn run_single_inference(
    model_path: *const c_char,
    prompt: *const c_char,
    max_tokens: i32,
    threads: i32,
    ctx_size: u32,
    use_mlock: bool,
    temp: f32,
    batch_size: u32,
    mode: *const c_char,
    ram_gb: u32,
) -> *mut c_char {
    let r_model_path = to_rust_string(model_path);
    let mut r_prompt = to_rust_string(prompt);
    let mut r_mode = to_rust_string(mode);
    if r_mode.is_empty() {
        r_mode = "auto".to_string();
    }

    if r_model_path.is_empty() || r_prompt.is_empty() {
        return err_cstr("Error: Invalid arguments");
    }

    let is_embedding_mode = false;
    let profile = resolve_hardware_profile(
        &r_mode,
        threads,
        batch_size,
        ctx_size,
        max_tokens,
        use_mlock,
        ram_gb,
        is_embedding_mode,
    );

    let mut auto_appended_json_trigger = false;
    if r_prompt.trim_end().ends_with('}') {
        auto_appended_json_trigger = true;
        r_prompt.push_str("\n\n### Response:\n{");
    }

    // Silence llama.cpp logging
    llama_log_set(Some(silent_log_callback), std::ptr::null_mut());

    // Initialize backend
    llama_backend_init();

    let c_model_path = safe_cstring(r_model_path);
    let mut model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    model_params.use_mmap = true;
    model_params.use_mlock = profile.use_mlock;

    // Load the model
    let model = llama_model_load_from_file(c_model_path.as_ptr(), model_params);
    if model.is_null() {
        llama_backend_free();
        return err_cstr("Error: Failed to load model");
    }

    let vocab = llama_model_get_vocab(model);

    // Tokenize prompt
    let c_prompt = safe_cstring(r_prompt.clone());
    let max_cap = r_prompt.len() + 256;
    let mut tokens = vec![0i32; max_cap];
    let n_tokens = llama_tokenize(vocab, c_prompt.as_ptr(), r_prompt.len() as i32, tokens.as_mut_ptr(), max_cap as i32, true, true);

    if n_tokens < 0 {
        llama_model_free(model);
        llama_backend_free();
        return CString::new("Error: Tokenization failed").unwrap().into_raw();
    }
    tokens.truncate(n_tokens as usize);

    // Get trained context length from model
    let requested_ctx = profile.ctx_size;

    // Compute required context capacity
    let required_ctx = n_tokens as u32 + profile.max_tokens as u32;
    let dynamic_ctx_size = requested_ctx.max(required_ctx);

    // Context parameters
    let mut ctx_params = llama_context_default_params();
    ctx_params.n_ctx = dynamic_ctx_size;
    let n_batch = profile.batch_size;
    ctx_params.n_batch = n_batch;
    ctx_params.n_ubatch = n_batch;
    let actual_threads = profile.threads;
    ctx_params.n_threads = actual_threads;
    ctx_params.n_threads_batch = actual_threads;
    ctx_params.embeddings = false;

    // Initialize context
    let ctx = llama_init_from_model(model, ctx_params);
    if ctx.is_null() {
        llama_model_free(model);
        llama_backend_free();
        return CString::new("Error: Failed to initialize context").unwrap().into_raw();
    }

    // Setup batch for prompt and autoregressive generation
    let batch_capacity = ctx_params.n_batch as i32;
    let mut batch = llama_batch_init(batch_capacity, 0, 1);

    // Feed prompt tokens into the context using chunked decoding
    let mut n_processed = 0;
    let mut decode_res;

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
        
        decode_res = llama_decode(ctx, batch);
        if decode_res != 0 {
            llama_batch_free(batch);
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return err_cstr("Error: Initial prompt decode chunk failed");
        }
        
        n_processed += n_chunk;
    }

    // Sampler setup
    let mut sparams = llama_sampler_chain_default_params();
    sparams.no_perf = true;
    let sampler = llama_sampler_chain_init(sparams);
    if sampler.is_null() {
        llama_batch_free(batch);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return err_cstr("Error: Failed to initialize sampler");
    }

    if temp <= 0.0 {
        llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));
        llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.95, 1));
        llama_sampler_chain_add(sampler, llama_sampler_init_min_p(0.05, 1));
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(temp));
        llama_sampler_chain_add(sampler, llama_sampler_init_dist(rand::random::<u32>()));
    }

    let mut response_output = String::new();
    let mut n_cur = n_tokens;
    let mut n_gen = 0;

    while n_gen < profile.max_tokens {
        let mut token_id = llama_sampler_sample(sampler, ctx, -1);
        if llama_vocab_is_eog(vocab, token_id) {
            if n_gen == 0 {
                // Smart prompt-ending recovery: If EOG is predicted as the very first token,
                // we temporarily ban the EOG token in logits and re-sample to force generation to start!
                let logits_ptr = llama_get_logits_ith(ctx, -1);
                if !logits_ptr.is_null() {
                    let n_vocab = llama_vocab_n_tokens(vocab);
                    for t_id in 0..n_vocab {
                        if llama_vocab_is_eog(vocab, t_id) {
                            *logits_ptr.add(t_id as usize) = -1e9f32;
                        }
                    }
                    token_id = llama_sampler_sample(sampler, ctx, -1);
                }
                if llama_vocab_is_eog(vocab, token_id) {
                    break;
                }
            } else {
                break;
            }
        }

        let mut piece_buf = [0i8; 256];
        let piece_len = llama_token_to_piece(vocab, token_id, piece_buf.as_mut_ptr(), piece_buf.len() as i32, 0, false);
        if piece_len > 0 {
            let actual_len = std::cmp::min(piece_len as usize, piece_buf.len());
            let bytes = std::slice::from_raw_parts(piece_buf.as_ptr() as *const u8, actual_len);
            if let Ok(piece_str) = std::str::from_utf8(bytes) {
                response_output.push_str(piece_str);
            }
        }

        n_gen += 1;

        batch.n_tokens = 1;
        *batch.token = token_id;
        *batch.pos = n_cur;
        *batch.n_seq_id = 1;
        *(*batch.seq_id) = 0;
        *batch.logits = 1;

        n_cur += 1;

        decode_res = llama_decode(ctx, batch);
        if decode_res != 0 {
            break;
        }
    }

    // Cleanup resources
    llama_sampler_free(sampler);
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    // Convert response to raw C-string pointer
    let mut final_response = response_output;
    if auto_appended_json_trigger {
        final_response.insert(0, '{');
    }

    // Post-process: auto-detect and pretty-print JSON responses
    let trimmed = final_response.trim();
    if trimmed.starts_with('{') && trimmed.ends_with('}') {
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(trimmed) {
            if let Ok(pretty) = serde_json::to_string_pretty(&parsed) {
                final_response = pretty;
            }
        }
    }

    // Use safe_cstring to handle any null bytes in generated text (LLM output can contain them)
    safe_cstring(final_response).into_raw()
}

/// Unified entry point to extract embeddings on-demand (loads and frees memory).
/// Returns a space/comma-separated bracketed float array representing the semantic vector.
/// Returns a raw C-string pointer that MUST be freed using `free_string()`.
#[no_mangle]
pub unsafe extern "C" fn run_single_embedding(
    model_path: *const c_char,
    prompt: *const c_char,
    threads: i32,
    ctx_size: u32,
    use_mlock: bool,
    batch_size: u32,
    mode: *const c_char,
    ram_gb: u32,
) -> *mut c_char {
    let r_model_path = to_rust_string(model_path);
    let r_prompt = to_rust_string(prompt);
    let mut r_mode = to_rust_string(mode);
    if r_mode.is_empty() {
        r_mode = "auto".to_string();
    }

    if r_model_path.is_empty() || r_prompt.is_empty() {
        return err_cstr("Error: Invalid arguments");
    }

    let is_embedding_mode = true;
    let profile = resolve_hardware_profile(
        &r_mode,
        threads,
        batch_size,
        ctx_size,
        0, // max_tokens not used for embedding extraction
        use_mlock,
        ram_gb,
        is_embedding_mode,
    );

    llama_log_set(Some(silent_log_callback), std::ptr::null_mut());
    llama_backend_init();

    let c_model_path = safe_cstring(r_model_path);
    let mut model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    model_params.use_mmap = true;
    model_params.use_mlock = profile.use_mlock;

    let model = llama_model_load_from_file(c_model_path.as_ptr(), model_params);
    if model.is_null() {
        llama_backend_free();
        return err_cstr("Error: Failed to load model");
    }

    let vocab = llama_model_get_vocab(model);
    let n_embd_raw = llama_model_n_embd(model);
    if n_embd_raw <= 0 {
        llama_model_free(model);
        llama_backend_free();
        return err_cstr("Error: Model has invalid embedding dimension â€” use an embedding-capable model");
    }
    let n_embd = n_embd_raw as usize;

    // Tokenize prompt
    let c_prompt = safe_cstring(r_prompt.clone());
    let max_cap = r_prompt.len() + 256;
    let mut tokens = vec![0i32; max_cap];
    let n_tokens = llama_tokenize(vocab, c_prompt.as_ptr(), r_prompt.len() as i32, tokens.as_mut_ptr(), max_cap as i32, true, true);

    if n_tokens < 0 {
        llama_model_free(model);
        llama_backend_free();
        return err_cstr("Error: Tokenization failed");
    }
    tokens.truncate(n_tokens as usize);

    let n_ctx_train = llama_model_n_ctx_train(model) as usize;
    if tokens.len() > n_ctx_train {
        tokens.truncate(n_ctx_train);
    }
    let n_tokens = tokens.len() as i32;

    // Get trained context length from model
    let requested_ctx = profile.ctx_size;

    // Compute required context capacity
    let required_ctx = n_tokens as u32;
    let dynamic_ctx_size = requested_ctx.max(required_ctx);

    let mut ctx_params = llama_context_default_params();
    ctx_params.n_ctx = dynamic_ctx_size;
    // For embedding models, n_batch and n_ubatch MUST cover the full sequence length
    // to avoid assertion crashes in encoder-only (e.g. BERT-style) models.
    let n_batch = dynamic_ctx_size;
    ctx_params.n_batch = n_batch;
    ctx_params.n_ubatch = n_batch;
    let actual_threads = profile.threads;
    ctx_params.n_threads = actual_threads;
    ctx_params.n_threads_batch = actual_threads;
    ctx_params.embeddings = true;

    let ctx = llama_init_from_model(model, ctx_params);
    if ctx.is_null() {
        llama_model_free(model);
        llama_backend_free();
        return err_cstr("Error: Failed to initialize context");
    }

    // Setup batch for decoding
    let batch_capacity = ctx_params.n_batch as i32;
    let mut batch = llama_batch_init(batch_capacity, 0, 1);

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
        
        let decode_res = llama_decode(ctx, batch);
        if decode_res != 0 {
            llama_batch_free(batch);
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return err_cstr("Error: Embedding prompt decode chunk failed");
        }
        
        n_processed += n_chunk;
    }

    let mut embd_ptr = llama_get_embeddings_seq(ctx, 0);
    if embd_ptr.is_null() {
        embd_ptr = llama_get_embeddings_ith(ctx, n_tokens - 1);
    }
    if embd_ptr.is_null() {
        embd_ptr = llama_get_embeddings(ctx);
    }

    if embd_ptr.is_null() {
        llama_batch_free(batch);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return err_cstr("Error: Failed to retrieve embeddings");
    }

    let raw_embedding = std::slice::from_raw_parts(embd_ptr, n_embd);
    
    // Format output as standard space-separated bracketed JSON float array
    let formatted_vector = format!("[{}]", raw_embedding.iter().map(|f| f.to_string()).collect::<Vec<String>>().join(", "));

    // Cleanup resources
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    // Convert response string to raw C-string pointer using safe_cstring (no panic on null bytes)
    safe_cstring(formatted_vector).into_raw()
}

/// Frees the C-string returned by `run_single_inference` or `run_single_embedding`.
#[no_mangle]
pub unsafe extern "C" fn free_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        let _ = CString::from_raw(ptr);
    }
}
