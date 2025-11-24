import { Link } from 'react-router-dom'
import { ArrowLeft, Sparkles, Zap, Shield, Layers, Brain, GitBranch, Target, Settings } from 'lucide-react'

export default function GANModelDetails() {
  return (
    <div className="container mx-auto px-4 py-8">
      {/* Header with back link */}
      <div className="bg-white rounded-2xl shadow-xl p-8 mb-8">
        <Link to="/gan" className="inline-flex items-center text-amber-600 hover:text-amber-700 mb-4">
          <ArrowLeft size={20} className="mr-2" />
          Back to GAN Generator
        </Link>
        <div className="flex items-center">
          <Sparkles size={48} className="text-amber-600 mr-4" />
          <div>
            <h1 className="text-4xl font-bold text-gray-800">Dual Conditional WGAN-GP</h1>
            <p className="text-gray-600 text-lg">Deep Dive into the Architecture</p>
          </div>
        </div>
      </div>

      {/* Overview */}
      <div className="bg-white rounded-2xl shadow-xl p-8 mb-8">
        <h2 className="text-2xl font-bold text-gray-800 mb-4 flex items-center">
          <Brain size={24} className="mr-3 text-purple-600" />
          Architecture Overview
        </h2>
        <p className="text-gray-700 mb-4">
          This model is a <strong>Dual Conditional Wasserstein GAN with Gradient Penalty (WGAN-GP)</strong> designed
          specifically for generating synthetic military vehicle images. Unlike standard GANs, this architecture
          accepts two conditional inputs: <strong>tank type</strong> and <strong>view angle</strong>, allowing precise
          control over the generated output.
        </p>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
          <div className="bg-purple-50 rounded-xl p-4">
            <h3 className="font-bold text-purple-800 mb-2">Output Size</h3>
            <p className="text-3xl font-bold text-purple-600">200x200</p>
            <p className="text-sm text-purple-700">RGB Images (3 channels)</p>
          </div>
          <div className="bg-blue-50 rounded-xl p-4">
            <h3 className="font-bold text-blue-800 mb-2">Latent Dimension</h3>
            <p className="text-3xl font-bold text-blue-600">100</p>
            <p className="text-sm text-blue-700">Random noise vector size</p>
          </div>
          <div className="bg-green-50 rounded-xl p-4">
            <h3 className="font-bold text-green-800 mb-2">Embedding Dimension</h3>
            <p className="text-3xl font-bold text-green-600">50</p>
            <p className="text-sm text-green-700">Per condition (tank + view)</p>
          </div>
        </div>
      </div>

      {/* Generator Architecture */}
      <div className="bg-white rounded-2xl shadow-xl p-8 mb-8">
        <h2 className="text-2xl font-bold text-gray-800 mb-4 flex items-center">
          <Zap size={24} className="mr-3 text-amber-600" />
          Generator Architecture
        </h2>
        <p className="text-gray-700 mb-6">
          The generator transforms random noise combined with conditional embeddings into realistic images through
          a series of transposed convolutions with self-attention for global coherence.
        </p>

        {/* Input Processing */}
        <div className="bg-gray-50 rounded-xl p-6 mb-6">
          <h3 className="font-bold text-gray-800 mb-3">Input Processing</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-white rounded-lg p-4 border border-gray-200">
              <p className="font-semibold text-gray-800">Noise Vector</p>
              <p className="text-sm text-gray-600">z ~ N(0, 1)</p>
              <p className="text-lg font-mono text-amber-600">100 dims</p>
            </div>
            <div className="bg-white rounded-lg p-4 border border-gray-200">
              <p className="font-semibold text-gray-800">Tank Embedding</p>
              <p className="text-sm text-gray-600">Learned lookup table</p>
              <p className="text-lg font-mono text-amber-600">50 dims</p>
            </div>
            <div className="bg-white rounded-lg p-4 border border-gray-200">
              <p className="font-semibold text-gray-800">View Embedding</p>
              <p className="text-sm text-gray-600">Learned lookup table</p>
              <p className="text-lg font-mono text-amber-600">50 dims</p>
            </div>
          </div>
          <p className="text-center text-gray-600 mt-4">
            Concatenated input: <span className="font-mono font-bold">200 dimensions</span>
          </p>
        </div>

        {/* Network Layers */}
        <h3 className="font-bold text-gray-800 mb-4">Network Stages</h3>
        <div className="space-y-3">
          <div className="flex items-center bg-gradient-to-r from-amber-50 to-white rounded-lg p-4 border-l-4 border-amber-500">
            <div className="w-24 text-center">
              <span className="font-mono text-sm bg-amber-100 px-2 py-1 rounded">1x1</span>
            </div>
            <div className="mx-4 text-amber-500">→</div>
            <div className="flex-1">
              <p className="font-semibold">Stage 1: ConvTranspose2d (200 → 1024 channels)</p>
              <p className="text-sm text-gray-600">+ BatchNorm + ReLU</p>
            </div>
            <div className="w-24 text-center">
              <span className="font-mono text-sm bg-amber-100 px-2 py-1 rounded">4x4</span>
            </div>
          </div>

          <div className="flex items-center bg-gradient-to-r from-amber-50 to-white rounded-lg p-4 border-l-4 border-amber-400">
            <div className="w-24 text-center">
              <span className="font-mono text-sm bg-amber-100 px-2 py-1 rounded">4x4</span>
            </div>
            <div className="mx-4 text-amber-500">→</div>
            <div className="flex-1">
              <p className="font-semibold">ConvTranspose2d (1024 → 512 channels)</p>
              <p className="text-sm text-gray-600">+ BatchNorm + ReLU</p>
            </div>
            <div className="w-24 text-center">
              <span className="font-mono text-sm bg-amber-100 px-2 py-1 rounded">8x8</span>
            </div>
          </div>

          <div className="flex items-center bg-gradient-to-r from-amber-50 to-white rounded-lg p-4 border-l-4 border-amber-400">
            <div className="w-24 text-center">
              <span className="font-mono text-sm bg-amber-100 px-2 py-1 rounded">8x8</span>
            </div>
            <div className="mx-4 text-amber-500">→</div>
            <div className="flex-1">
              <p className="font-semibold">ConvTranspose2d (512 → 256 channels)</p>
              <p className="text-sm text-gray-600">+ BatchNorm + ReLU</p>
            </div>
            <div className="w-24 text-center">
              <span className="font-mono text-sm bg-amber-100 px-2 py-1 rounded">16x16</span>
            </div>
          </div>

          <div className="flex items-center bg-gradient-to-r from-amber-50 to-white rounded-lg p-4 border-l-4 border-amber-400">
            <div className="w-24 text-center">
              <span className="font-mono text-sm bg-amber-100 px-2 py-1 rounded">16x16</span>
            </div>
            <div className="mx-4 text-amber-500">→</div>
            <div className="flex-1">
              <p className="font-semibold">ConvTranspose2d (256 → 128 channels)</p>
              <p className="text-sm text-gray-600">+ BatchNorm + ReLU</p>
            </div>
            <div className="w-24 text-center">
              <span className="font-mono text-sm bg-amber-100 px-2 py-1 rounded">32x32</span>
            </div>
          </div>

          {/* Self-Attention Layer */}
          <div className="flex items-center bg-gradient-to-r from-purple-100 to-purple-50 rounded-lg p-4 border-l-4 border-purple-500">
            <div className="w-24 text-center">
              <span className="font-mono text-sm bg-purple-200 px-2 py-1 rounded">32x32</span>
            </div>
            <div className="mx-4 text-purple-500">→</div>
            <div className="flex-1">
              <p className="font-semibold text-purple-800">Self-Attention Layer</p>
              <p className="text-sm text-purple-700">Query/Key/Value projections with learnable gamma</p>
            </div>
            <div className="w-24 text-center">
              <span className="font-mono text-sm bg-purple-200 px-2 py-1 rounded">32x32</span>
            </div>
          </div>

          <div className="flex items-center bg-gradient-to-r from-amber-50 to-white rounded-lg p-4 border-l-4 border-amber-400">
            <div className="w-24 text-center">
              <span className="font-mono text-sm bg-amber-100 px-2 py-1 rounded">32x32</span>
            </div>
            <div className="mx-4 text-amber-500">→</div>
            <div className="flex-1">
              <p className="font-semibold">ConvTranspose2d (128 → 64 channels)</p>
              <p className="text-sm text-gray-600">+ BatchNorm + ReLU</p>
            </div>
            <div className="w-24 text-center">
              <span className="font-mono text-sm bg-amber-100 px-2 py-1 rounded">50x50</span>
            </div>
          </div>

          <div className="flex items-center bg-gradient-to-r from-amber-50 to-white rounded-lg p-4 border-l-4 border-amber-400">
            <div className="w-24 text-center">
              <span className="font-mono text-sm bg-amber-100 px-2 py-1 rounded">50x50</span>
            </div>
            <div className="mx-4 text-amber-500">→</div>
            <div className="flex-1">
              <p className="font-semibold">ConvTranspose2d (64 → 32 channels)</p>
              <p className="text-sm text-gray-600">+ BatchNorm + ReLU</p>
            </div>
            <div className="w-24 text-center">
              <span className="font-mono text-sm bg-amber-100 px-2 py-1 rounded">100x100</span>
            </div>
          </div>

          <div className="flex items-center bg-gradient-to-r from-green-50 to-white rounded-lg p-4 border-l-4 border-green-500">
            <div className="w-24 text-center">
              <span className="font-mono text-sm bg-green-100 px-2 py-1 rounded">100x100</span>
            </div>
            <div className="mx-4 text-green-500">→</div>
            <div className="flex-1">
              <p className="font-semibold text-green-800">Output: ConvTranspose2d (32 → 3 channels)</p>
              <p className="text-sm text-green-700">+ Tanh activation (output range: [-1, 1])</p>
            </div>
            <div className="w-24 text-center">
              <span className="font-mono text-sm bg-green-200 px-2 py-1 rounded">200x200</span>
            </div>
          </div>
        </div>
      </div>

      {/* Discriminator Architecture */}
      <div className="bg-white rounded-2xl shadow-xl p-8 mb-8">
        <h2 className="text-2xl font-bold text-gray-800 mb-4 flex items-center">
          <Shield size={24} className="mr-3 text-blue-600" />
          Discriminator (Critic) Architecture
        </h2>
        <p className="text-gray-700 mb-6">
          The discriminator evaluates images and outputs a score (not probability) indicating how "real" an image appears.
          In WGAN-GP, it's called a "critic" because it outputs unbounded scores rather than probabilities.
        </p>

        {/* Key Features */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
          <div className="bg-blue-50 rounded-xl p-4">
            <h3 className="font-bold text-blue-800 mb-2">Spectral Normalization</h3>
            <p className="text-sm text-blue-700">
              All convolutional and linear layers use spectral normalization to stabilize training
              by constraining the Lipschitz constant of the discriminator.
            </p>
          </div>
          <div className="bg-blue-50 rounded-xl p-4">
            <h3 className="font-bold text-blue-800 mb-2">No Batch Normalization</h3>
            <p className="text-sm text-blue-700">
              WGAN-GP recommends avoiding BatchNorm in the critic as it can interfere with the
              gradient penalty computation.
            </p>
          </div>
        </div>

        {/* Network Flow */}
        <h3 className="font-bold text-gray-800 mb-4">Network Stages</h3>
        <div className="space-y-3">
          <div className="flex items-center bg-gradient-to-r from-blue-50 to-white rounded-lg p-4 border-l-4 border-blue-500">
            <div className="w-24 text-center">
              <span className="font-mono text-sm bg-blue-100 px-2 py-1 rounded">200x200</span>
            </div>
            <div className="mx-4 text-blue-500">→</div>
            <div className="flex-1">
              <p className="font-semibold">Stage 1: Conv2d (3 → 32 → 64 → 128 channels)</p>
              <p className="text-sm text-gray-600">+ LeakyReLU(0.2) at each layer</p>
            </div>
            <div className="w-24 text-center">
              <span className="font-mono text-sm bg-blue-100 px-2 py-1 rounded">25x25</span>
            </div>
          </div>

          {/* Self-Attention */}
          <div className="flex items-center bg-gradient-to-r from-purple-100 to-purple-50 rounded-lg p-4 border-l-4 border-purple-500">
            <div className="w-24 text-center">
              <span className="font-mono text-sm bg-purple-200 px-2 py-1 rounded">25x25</span>
            </div>
            <div className="mx-4 text-purple-500">→</div>
            <div className="flex-1">
              <p className="font-semibold text-purple-800">Self-Attention Layer</p>
              <p className="text-sm text-purple-700">Captures global dependencies for better discrimination</p>
            </div>
            <div className="w-24 text-center">
              <span className="font-mono text-sm bg-purple-200 px-2 py-1 rounded">25x25</span>
            </div>
          </div>

          <div className="flex items-center bg-gradient-to-r from-blue-50 to-white rounded-lg p-4 border-l-4 border-blue-400">
            <div className="w-24 text-center">
              <span className="font-mono text-sm bg-blue-100 px-2 py-1 rounded">25x25</span>
            </div>
            <div className="mx-4 text-blue-500">→</div>
            <div className="flex-1">
              <p className="font-semibold">Stage 2: Conv2d (128 → 256 → 512 → 1024 channels)</p>
              <p className="text-sm text-gray-600">+ LeakyReLU(0.2) at each layer</p>
            </div>
            <div className="w-24 text-center">
              <span className="font-mono text-sm bg-blue-100 px-2 py-1 rounded">4x4</span>
            </div>
          </div>

          <div className="flex items-center bg-gradient-to-r from-green-50 to-white rounded-lg p-4 border-l-4 border-green-500">
            <div className="w-24 text-center">
              <span className="font-mono text-sm bg-green-100 px-2 py-1 rounded">16384</span>
            </div>
            <div className="mx-4 text-green-500">→</div>
            <div className="flex-1">
              <p className="font-semibold text-green-800">Classifier: Flatten + Concat embeddings + Linear layers</p>
              <p className="text-sm text-green-700">Image features (16384) + Tank embed (50) + View embed (50) → 512 → 1</p>
            </div>
            <div className="w-24 text-center">
              <span className="font-mono text-sm bg-green-200 px-2 py-1 rounded">Score</span>
            </div>
          </div>
        </div>
      </div>

      {/* Self-Attention Mechanism */}
      <div className="bg-white rounded-2xl shadow-xl p-8 mb-8">
        <h2 className="text-2xl font-bold text-gray-800 mb-4 flex items-center">
          <Layers size={24} className="mr-3 text-purple-600" />
          Self-Attention Mechanism
        </h2>
        <p className="text-gray-700 mb-6">
          Based on the <strong>SAGAN (Self-Attention GAN)</strong> paper, self-attention allows the model to
          capture long-range dependencies by letting distant pixels influence each other, improving global coherence.
        </p>

        <div className="bg-purple-50 rounded-xl p-6">
          <h3 className="font-bold text-purple-800 mb-4">How It Works</h3>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
            <div className="bg-white rounded-lg p-4 text-center">
              <p className="font-semibold text-gray-800">Query (Q)</p>
              <p className="text-sm text-gray-600">1x1 Conv</p>
              <p className="font-mono text-purple-600">C → C/8</p>
            </div>
            <div className="bg-white rounded-lg p-4 text-center">
              <p className="font-semibold text-gray-800">Key (K)</p>
              <p className="text-sm text-gray-600">1x1 Conv</p>
              <p className="font-mono text-purple-600">C → C/8</p>
            </div>
            <div className="bg-white rounded-lg p-4 text-center">
              <p className="font-semibold text-gray-800">Value (V)</p>
              <p className="text-sm text-gray-600">1x1 Conv</p>
              <p className="font-mono text-purple-600">C → C</p>
            </div>
            <div className="bg-white rounded-lg p-4 text-center">
              <p className="font-semibold text-gray-800">Gamma (γ)</p>
              <p className="text-sm text-gray-600">Learnable</p>
              <p className="font-mono text-purple-600">Starts at 0</p>
            </div>
          </div>

          <div className="bg-white rounded-lg p-4">
            <p className="font-mono text-center text-gray-800">
              Attention = softmax(Q · K<sup>T</sup>) · V
            </p>
            <p className="font-mono text-center text-gray-800 mt-2">
              Output = γ · Attention + Input <span className="text-gray-500">(residual connection)</span>
            </p>
          </div>

          <p className="text-sm text-purple-700 mt-4">
            The learnable gamma parameter starts at 0, allowing the model to first learn local features
            before gradually incorporating global attention patterns.
          </p>
        </div>
      </div>

      {/* WGAN-GP Training */}
      <div className="bg-white rounded-2xl shadow-xl p-8 mb-8">
        <h2 className="text-2xl font-bold text-gray-800 mb-4 flex items-center">
          <Target size={24} className="mr-3 text-red-600" />
          WGAN-GP Training
        </h2>
        <p className="text-gray-700 mb-6">
          The Wasserstein GAN with Gradient Penalty addresses mode collapse and training instability issues
          found in standard GANs by using the Wasserstein distance and enforcing a Lipschitz constraint.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Discriminator Loss */}
          <div className="bg-red-50 rounded-xl p-6">
            <h3 className="font-bold text-red-800 mb-3">Critic Loss</h3>
            <div className="bg-white rounded-lg p-4 font-mono text-sm mb-4">
              <p>L<sub>D</sub> = E[D(fake)] - E[D(real)] + λ · GP</p>
            </div>
            <ul className="text-sm text-red-700 space-y-2">
              <li>• <strong>E[D(real)]:</strong> Mean critic score for real images</li>
              <li>• <strong>E[D(fake)]:</strong> Mean critic score for generated images</li>
              <li>• <strong>GP:</strong> Gradient penalty term (λ = 10)</li>
            </ul>
          </div>

          {/* Generator Loss */}
          <div className="bg-green-50 rounded-xl p-6">
            <h3 className="font-bold text-green-800 mb-3">Generator Loss</h3>
            <div className="bg-white rounded-lg p-4 font-mono text-sm mb-4">
              <p>L<sub>G</sub> = -E[D(fake)]</p>
            </div>
            <ul className="text-sm text-green-700 space-y-2">
              <li>• Generator tries to maximize critic scores for fake images</li>
              <li>• No log or sigmoid - direct Wasserstein distance optimization</li>
              <li>• More stable gradients than standard GAN</li>
            </ul>
          </div>
        </div>

        {/* Gradient Penalty */}
        <div className="mt-6 bg-gray-50 rounded-xl p-6">
          <h3 className="font-bold text-gray-800 mb-3">Gradient Penalty (GP)</h3>
          <div className="bg-white rounded-lg p-4 font-mono text-sm mb-4 text-center">
            <p>GP = E[(||∇<sub>x̂</sub>D(x̂)|| - 1)²]</p>
            <p className="text-xs text-gray-500 mt-2">where x̂ = α · x<sub>real</sub> + (1-α) · x<sub>fake</sub>, α ~ U(0,1)</p>
          </div>
          <p className="text-sm text-gray-700">
            The gradient penalty enforces the 1-Lipschitz constraint by penalizing the critic when
            the gradient norm deviates from 1. This is computed on interpolated samples between
            real and fake images.
          </p>
        </div>
      </div>

      {/* Training Hyperparameters */}
      <div className="bg-white rounded-2xl shadow-xl p-8 mb-8">
        <h2 className="text-2xl font-bold text-gray-800 mb-4 flex items-center">
          <Settings size={24} className="mr-3 text-gray-600" />
          Training Configuration
        </h2>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-gray-50 rounded-xl p-4 text-center">
            <p className="text-sm text-gray-600">Batch Size</p>
            <p className="text-2xl font-bold text-gray-800">16</p>
          </div>
          <div className="bg-gray-50 rounded-xl p-4 text-center">
            <p className="text-sm text-gray-600">Learning Rate</p>
            <p className="text-2xl font-bold text-gray-800">5e-5</p>
          </div>
          <div className="bg-gray-50 rounded-xl p-4 text-center">
            <p className="text-sm text-gray-600">Adam β1, β2</p>
            <p className="text-2xl font-bold text-gray-800">0.0, 0.9</p>
          </div>
          <div className="bg-gray-50 rounded-xl p-4 text-center">
            <p className="text-sm text-gray-600">Gradient Penalty λ</p>
            <p className="text-2xl font-bold text-gray-800">10</p>
          </div>
          <div className="bg-gray-50 rounded-xl p-4 text-center">
            <p className="text-sm text-gray-600">Critic Steps</p>
            <p className="text-2xl font-bold text-gray-800">1</p>
          </div>
          <div className="bg-gray-50 rounded-xl p-4 text-center">
            <p className="text-sm text-gray-600">Epochs</p>
            <p className="text-2xl font-bold text-gray-800">200</p>
          </div>
          <div className="bg-gray-50 rounded-xl p-4 text-center">
            <p className="text-sm text-gray-600">Gradient Clipping</p>
            <p className="text-2xl font-bold text-gray-800">1.0</p>
          </div>
          <div className="bg-gray-50 rounded-xl p-4 text-center">
            <p className="text-sm text-gray-600">Weight Init</p>
            <p className="text-2xl font-bold text-gray-800">N(0, 0.02)</p>
          </div>
        </div>
      </div>

      {/* Conditioning Mechanism */}
      <div className="bg-white rounded-2xl shadow-xl p-8 mb-8">
        <h2 className="text-2xl font-bold text-gray-800 mb-4 flex items-center">
          <GitBranch size={24} className="mr-3 text-teal-600" />
          Dual Conditioning
        </h2>
        <p className="text-gray-700 mb-6">
          The model uses learned embeddings to condition generation on two categorical inputs:
          tank type and view angle. This enables precise control over the output.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-teal-50 rounded-xl p-6">
            <h3 className="font-bold text-teal-800 mb-3">Generator Conditioning</h3>
            <p className="text-sm text-teal-700 mb-4">
              Both embeddings are concatenated with the noise vector at the input layer,
              allowing the entire network to be influenced by the conditions.
            </p>
            <div className="bg-white rounded-lg p-4">
              <p className="font-mono text-sm text-center">
                input = concat(noise, tank_embed, view_embed)
              </p>
              <p className="font-mono text-sm text-center mt-2 text-gray-500">
                [100] + [50] + [50] = [200]
              </p>
            </div>
          </div>

          <div className="bg-teal-50 rounded-xl p-6">
            <h3 className="font-bold text-teal-800 mb-3">Discriminator Conditioning</h3>
            <p className="text-sm text-teal-700 mb-4">
              Embeddings are concatenated with image features at the classifier layer,
              allowing the critic to judge if the image matches the conditions.
            </p>
            <div className="bg-white rounded-lg p-4">
              <p className="font-mono text-sm text-center">
                input = concat(img_features, tank_embed, view_embed)
              </p>
              <p className="font-mono text-sm text-center mt-2 text-gray-500">
                [16384] + [50] + [50] = [16484]
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* References */}
      <div className="bg-gray-50 rounded-2xl shadow-xl p-8">
        <h2 className="text-xl font-bold text-gray-800 mb-4">References & Further Reading</h2>
        <ul className="space-y-2 text-gray-700">
          <li>• <strong>WGAN:</strong> Arjovsky et al., "Wasserstein GAN" (2017)</li>
          <li>• <strong>WGAN-GP:</strong> Gulrajani et al., "Improved Training of Wasserstein GANs" (2017)</li>
          <li>• <strong>SAGAN:</strong> Zhang et al., "Self-Attention Generative Adversarial Networks" (2018)</li>
          <li>• <strong>Conditional GAN:</strong> Mirza & Osindero, "Conditional Generative Adversarial Nets" (2014)</li>
          <li>• <strong>Spectral Normalization:</strong> Miyato et al., "Spectral Normalization for GANs" (2018)</li>
        </ul>
      </div>
    </div>
  )
}
