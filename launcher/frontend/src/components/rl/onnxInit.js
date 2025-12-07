/**
 * ONNX Runtime Initialization
 * Must be imported BEFORE any ONNX Runtime usage
 *
 * This file configures ONNX Runtime to use standard WASM files only,
 * avoiding JSEP (JavaScript Execution Provider) which causes dynamic
 * import issues with Vite.
 */
import * as ort from 'onnxruntime-web'

// Configure ONNX Runtime BEFORE first use
console.log('[ONNX-INIT] Configuring ONNX Runtime Web...')

// Set WASM paths to public directory
ort.env.wasm.wasmPaths = `${window.location.origin}/onnx/`

// CRITICAL: Configure to use standard WASM only (no JSEP)
// This must be set BEFORE any InferenceSession.create() calls
ort.env.wasm.proxy = false        // Disable JSEP proxy
ort.env.wasm.numThreads = 1       // Use single-threaded WASM
ort.env.wasm.simd = true          // Enable SIMD for performance

// Additional settings to ensure standard WASM usage
ort.env.wasm.initTimeout = 30000  // 30 second timeout

// Disable WebGPU entirely
ort.env.webgpu = {
  profilingMode: 'off',
  preferredLayout: 'NCHW'
}

// Log configuration for debugging
console.log('[ONNX-INIT] ✓ Configuration applied:')
console.log('  WASM paths:', ort.env.wasm.wasmPaths)
console.log('  Proxy (JSEP) disabled:', ort.env.wasm.proxy === false)
console.log('  Num threads:', ort.env.wasm.numThreads)
console.log('  SIMD enabled:', ort.env.wasm.simd)
console.log('  Expected WASM file: ort-wasm-simd-threaded.mjs/.wasm')

export default ort
