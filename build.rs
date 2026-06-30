fn main() {
    // Add various possible CMake output directories to search path
    println!("cargo:rustc-link-search=native=d:/AI/llamacpp/build/src/Release");
    println!("cargo:rustc-link-search=native=d:/AI/llamacpp/build/common/Release");
    println!("cargo:rustc-link-search=native=d:/AI/llamacpp/build/ggml/src/Release");
    
    // Statically link the C++ core libraries
    println!("cargo:rustc-link-lib=static=llama");
    println!("cargo:rustc-link-lib=static=llama-common");
    println!("cargo:rustc-link-lib=static=llama-common-base");
    println!("cargo:rustc-link-lib=static=ggml");
    println!("cargo:rustc-link-lib=static=ggml-cpu");
    println!("cargo:rustc-link-lib=static=ggml-base");

    // Inject Windows version metadata and resources
    if cfg!(target_os = "windows") {
        let mut res = winresource::WindowsResource::new();
        res.set("CompanyName", "RagIQ");
        res.set("FileDescription", "RagIQ RuntimeEngine - Local SLM Unified Inference Runtime");
        res.set("ProductName", "RagIQ");
        res.set("OriginalFilename", "RagIQ-RuntimeEngine");
        res.set("LegalCopyright", "Copyright (C) 2026 RagIQ. All rights reserved.");
        res.set("ProductVersion", "1.0.0.0");
        res.set("FileVersion", "1.0.0.0");
        res.compile().unwrap();
    }
}
